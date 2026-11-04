#include "live/live_inspector.h"
#include "live/live_requests.h"

#include <crude_json.h>
#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace caliper_live {
namespace {
using Json = crude_json::value;
using Object = crude_json::object;
using Array = crude_json::array;

const Json& field(const Json& object, const char* key) {
    static const Json empty;
    return object.is_object() && object.contains(key) ? object[key] : empty;
}
std::string string(const Json& value, const std::string& fallback = {}) {
    return value.is_string() ? value.get<std::string>() : fallback;
}
double number(const Json& value, double fallback = 0) {
    if (!value.is_number()) return fallback;
    const double n = value.get<double>();
    return std::isfinite(n) ? n : fallback;
}
bool boolean(const Json& value) { return value.is_boolean() && value.get<bool>(); }
int integer(const Json& value, int fallback = 0) {
    const double n = number(value, fallback);
    return n >= 0 && n <= std::numeric_limits<int>::max() && std::floor(n) == n
        ? static_cast<int>(n) : fallback;
}
double seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
std::string byte_size(double bytes) {
    char buffer[64];
    if (bytes >= 1024 * 1024) std::snprintf(buffer, sizeof buffer, "%.1f MiB", bytes / (1024 * 1024));
    else if (bytes >= 1024) std::snprintf(buffer, sizeof buffer, "%.1f KiB", bytes / 1024);
    else std::snprintf(buffer, sizeof buffer, "%.0f B", bytes);
    return buffer;
}

struct Tensor {
    std::string name, group, shape, dtype;
    int rows = 0, cols = 0;
    double bytes = 0, size = 0;
    bool tied = false;
};
struct Pending {
    int id = 0;
    std::string op;
};

} // namespace

struct Inspector::Impl {
    RequestChannel channel;
    std::string session, error, model, output, generation_status;
    std::set<std::string> capabilities;
    bool visible = false, ready = false, alive = true, was_connected = false;
    bool stop_requested = false, output_changed = false;
    double retry_at = 0, hello_at = 0;
    int next_id = 1, hello_id = 0, stop_id = 0;
    Pending active;
    bool want_list = false;
    std::string want_show;

    char prompt[16384] = "Write one sentence about the Moon.";
    int max_new_tokens = 128, top_k = 40;
    float temperature = 0.7f, top_p = 0.9f;
    bool thinking = false;
    int tokens = 0;
    double tokens_per_second = 0;

    std::vector<Tensor> tensors;
    std::string selected;
    char filter[256] = {};
    int row = 0, col = 0;
    double value = 0, factor = 1;
    bool have_value = false;
    int value_row = 0, value_col = 0;
    double read_value = 0;
    int sample_rows = 0, sample_cols = 0;
    std::vector<double> heatmap, plot_heatmap;
    std::vector<int> row_indices, col_indices;
    double heat_min = 0, heat_max = 1;
    std::vector<std::string> edits;

    bool supports(const char* op) const { return capabilities.count(op) != 0; }
    bool idle() const { return ready && alive && channel.connected() && active.id == 0; }
    const Tensor* tensor() const {
        const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor& t) { return t.name == selected; });
        return it == tensors.end() ? nullptr : &*it;
    }
    void protocol_error(const char* message) {
        error = message;
        channel.close();
    }
    bool send(const char* op, Object fields = {}) {
        if (!idle() || !supports(op)) return false;
        const int id = next_id++;
        fields["id"] = static_cast<double>(id);
        fields["op"] = op;
        if (!channel.send(Json(std::move(fields)).dump())) {
            error = channel.error();
            return false;
        }
        active = {id, op};
        error.clear();
        return true;
    }
    void stop() {
        if (!channel.connected() || active.op != "generate" || stop_requested) return;
        const int id = next_id++;
        if (channel.send(Json(Object{{"id", static_cast<double>(id)}, {"op", "stop"}}).dump())) {
            stop_id = id;
            stop_requested = true;
            generation_status = "Stopping";
        } else error = channel.error();
    }
    void read_edits(const Json& result) {
        const Json& items = field(result, "edits");
        if (!items.is_array()) return;
        edits.clear();
        for (const Json& item : items.get<Array>()) {
            std::string description = string(field(item, "description"));
            if (!description.empty()) edits.push_back(std::move(description));
        }
    }
    void select(const std::string& name) {
        selected = name;
        row = col = 0;
        have_value = false;
        heatmap.clear(); plot_heatmap.clear(); row_indices.clear(); col_indices.clear();
        sample_rows = sample_cols = 0;
        edits.clear();
        want_show = name;
    }
    bool read_tensors(const Json& result) {
        const Json& items = field(result, "tensors");
        if (!items.is_array() || items.get<Array>().size() > 10000) return false;
        std::vector<Tensor> next;
        for (const Json& item : items.get<Array>()) {
            Tensor t;
            t.name = string(field(item, "name"));
            t.group = string(field(item, "group"), "Model");
            t.dtype = string(field(item, "dtype"));
            t.rows = integer(field(item, "rows"));
            t.cols = integer(field(item, "cols"));
            t.bytes = number(field(item, "bytes"));
            t.size = number(field(item, "size"));
            t.tied = boolean(field(item, "tied"));
            const Json& shape = field(item, "shape");
            if (t.name.empty() || t.name.size() > 1024 || !shape.is_array() ||
                t.rows <= 0 || t.cols <= 0 || t.bytes < 0 || t.size < 0) return false;
            t.shape = "[";
            for (const Json& dimension : shape.get<Array>()) {
                if (t.shape.size() > 1) t.shape += ", ";
                const int d = integer(dimension, -1);
                if (d < 0) return false;
                t.shape += std::to_string(d);
            }
            t.shape += "]";
            next.push_back(std::move(t));
        }
        tensors = std::move(next);
        model = string(field(result, "model"));
        if (!tensor()) {
            if (tensors.empty()) selected.clear();
            else select(tensors.front().name);
        } else want_show = selected;
        return true;
    }
    bool read_heatmap(const Json& result) {
        if (string(field(result, "name")) != selected) return true;
        const int nr = integer(field(result, "sample_rows")), nc = integer(field(result, "sample_cols"));
        const auto& values = field(result, "values");
        const auto& ri = field(result, "row_indices");
        const auto& ci = field(result, "col_indices");
        if (nr < 1 || nc < 1 || nr > 256 || nc > 256 || !values.is_array() ||
            values.get<Array>().size() != static_cast<size_t>(nr * nc) ||
            !ri.is_array() || !ci.is_array() || ri.get<Array>().size() != static_cast<size_t>(nr) ||
            ci.get<Array>().size() != static_cast<size_t>(nc)) return false;
        const Tensor* t = tensor();
        if (!t) return false;
        std::vector<double> data;
        std::vector<int> rows, cols;
        for (const Json& v : values.get<Array>()) {
            if (!v.is_number() || !std::isfinite(v.get<double>())) return false;
            data.push_back(v.get<double>());
        }
        for (const Json& v : ri.get<Array>()) {
            int i = integer(v, -1);
            if (i < 0 || i >= t->rows) return false;
            rows.push_back(i);
        }
        for (const Json& v : ci.get<Array>()) {
            int i = integer(v, -1);
            if (i < 0 || i >= t->cols) return false;
            cols.push_back(i);
        }
        heatmap = std::move(data);
        // ImPlot's heatmap puts its first row at the upper Y bound. With a
        // matrix-style downward row axis, reverse only the drawing buffer.
        plot_heatmap.resize(heatmap.size());
        for (int r = 0; r < nr; ++r)
            std::copy_n(heatmap.data() + r * nc, nc, plot_heatmap.data() + (nr - r - 1) * nc);
        row_indices = std::move(rows); col_indices = std::move(cols);
        sample_rows = nr; sample_cols = nc;
        const auto range = std::minmax_element(heatmap.begin(), heatmap.end());
        heat_min = *range.first; heat_max = *range.second;
        if (heat_min == heat_max) { heat_min -= 0.5; heat_max += 0.5; }
        read_edits(result);
        return true;
    }
    void metrics(const Json& message) {
        tokens = integer(field(message, "tokens"), tokens);
        tokens_per_second = number(field(message, "tokens_per_second"), tokens_per_second);
    }
    void receive(const std::string& line) {
        const Json reply = Json::parse(line);
        const int id = integer(field(reply, "id"), -1);
        if (!reply.is_object() || id < 0 || !field(reply, "done").is_boolean()) {
            protocol_error("Invalid reply from script"); return;
        }
        if (id == hello_id) {
            hello_id = 0;
            const Json& result = field(reply, "result");
            const Json& caps = field(result, "capabilities");
            if (!boolean(field(reply, "done")) || integer(field(result, "protocol")) != 1 || !caps.is_array()) {
                visible = true;
                protocol_error("Unsupported request protocol"); return;
            }
            capabilities.clear();
            for (const Json& cap : caps.get<Array>()) if (cap.is_string()) capabilities.insert(cap.get<std::string>());
            visible = supports("generate") || supports("list_tensors");
            ready = true;
            error.clear();
            want_list = supports("list_tensors");
            return;
        }
        if (id == stop_id) {
            stop_id = 0;
            if (field(reply, "error").is_string()) error = string(field(reply, "error"));
            return;
        }
        if (id != active.id || id == 0) return; // never apply a delayed reply to a later selection
        if (!boolean(field(reply, "done"))) {
            if (active.op == "generate") {
                const Json& event = field(reply, "event");
                const std::string part = string(field(event, "text"));
                if (output.size() + part.size() > 1024 * 1024) {
                    stop(); error = "Output limit reached"; return;
                }
                output += part;
                output_changed = true;
                metrics(event);
            }
            return;
        }
        const std::string op = active.op;
        active = {};
        if (field(reply, "error").is_string()) {
            error = string(field(reply, "error"));
            if (op == "generate") { generation_status = "Generation failed"; stop_requested = false; }
            return;
        }
        const Json& result = field(reply, "result");
        if (op == "generate") {
            metrics(result);
            generation_status = boolean(field(result, "cancelled")) ? "Stopped" : "Complete";
            stop_requested = false;
        } else if (op == "list_tensors") {
            if (!read_tensors(result)) protocol_error("Invalid tensor list from script");
        } else if (op == "show_tensor") {
            if (!read_heatmap(result)) protocol_error("Invalid tensor overview from script");
        } else if (op == "read_weight") {
            const Json& v = field(result, "value");
            if (string(field(result, "name")) != selected || !v.is_number() || !std::isfinite(v.get<double>())) {
                error = "Invalid weight value from script"; return;
            }
            value_row = integer(field(result, "row"));
            value_col = integer(field(result, "col"));
            read_value = value = v.get<double>();
            have_value = true;
        } else if (op == "edit_weight" || op == "revert_tensor") {
            have_value = false;
            read_edits(result);
            want_show = selected;
        }
    }
    void poll(const std::string& token, bool producer_alive) {
        session = token;
        alive = producer_alive;
        const double now = seconds();
        if (!alive) {
            if (channel.connected()) channel.close();
        } else if (!channel.connected() && now >= retry_at) {
            retry_at = now + 1.0;
            if (channel.connect(session)) {
                const int id = next_id++;
                hello_id = id; hello_at = now;
                channel.send(Json(Object{{"id", static_cast<double>(id)}, {"op", "hello"}}).dump());
                was_connected = true;
            }
        }
        for (const auto& line : channel.poll()) receive(line);
        if (hello_id && channel.connected() && now - hello_at > 5) protocol_error("Request handshake timed out");
        if (was_connected && !channel.connected()) {
            was_connected = false;
            ready = false; hello_id = stop_id = 0;
            if (active.op == "generate") generation_status = "Generation interrupted";
            if (active.id) error = "Connection ended; the operation was not retried";
            else if (visible && error.empty()) error = alive ? channel.error() : "Session ended";
            active = {};
            stop_requested = false;
            want_list = false; want_show.clear();
            retry_at = now + 1.0;
        }
        if (!idle()) return;
        if (want_list) {
            want_list = false;
            send("list_tensors");
        } else if (!want_show.empty()) {
            std::string name = std::move(want_show);
            want_show.clear();
            send("show_tensor", {{"name", name}});
        }
    }
    void status() const {
        if (!alive) ImGui::TextUnformatted("Session ended");
        else if (!ready) ImGui::TextUnformatted("Connecting");
        else if (!active.op.empty() && active.op != "generate") ImGui::TextUnformatted("Working");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
    }
    void generate() {
        if (send("generate", {{"prompt", prompt}, {"max_new_tokens", static_cast<double>(max_new_tokens)},
                {"temperature", static_cast<double>(temperature)}, {"top_p", static_cast<double>(top_p)},
                {"top_k", static_cast<double>(top_k)}, {"thinking", thinking}})) {
            output.clear(); tokens = 0; tokens_per_second = 0;
            generation_status = "Generating";
            stop_requested = false;
            output_changed = true;
        }
    }
    void inference(float x, float y, float w, float h) {
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Inference###caliper.inspector.inference")) {
            status();
            ImGui::TextUnformatted("Prompt");
            ImGui::InputTextMultiline("##prompt", prompt, sizeof prompt, ImVec2(-1, 110));
            if (ImGui::BeginTable("settings", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn(); ImGui::TextUnformatted("Max new tokens"); ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##max_new_tokens", &max_new_tokens);
                max_new_tokens = std::clamp(max_new_tokens, 1, 1024);
                ImGui::TableNextColumn(); ImGui::TextUnformatted("Temperature"); ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##temperature", &temperature, 0, 0, "%.2f");
                temperature = std::isfinite(temperature) ? std::clamp(temperature, 0.0f, 2.0f) : 0.7f;
                ImGui::TableNextColumn(); ImGui::TextUnformatted("Top p"); ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##top_p", &top_p, 0, 0, "%.2f");
                top_p = std::isfinite(top_p) ? std::clamp(top_p, 0.0f, 1.0f) : 0.9f;
                ImGui::TableNextColumn(); ImGui::TextUnformatted("Top k"); ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##top_k", &top_k);
                top_k = std::clamp(top_k, 0, 1000);
                ImGui::EndTable();
            }
            ImGui::Checkbox("Thinking", &thinking);
            ImGui::BeginDisabled(!idle() || prompt[0] == '\0');
            if (ImGui::Button("Generate")) generate();
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(active.op != "generate" || stop_requested || !channel.connected());
            if (ImGui::Button("Stop")) stop();
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("%d tokens  %.1f tokens/s", tokens, tokens_per_second);
            if (!generation_status.empty()) ImGui::TextUnformatted(generation_status.c_str());
            ImGui::Separator();
            if (ImGui::BeginChild("output", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
                const bool at_end = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1;
                ImGui::PushTextWrapPos(0);
                ImGui::TextUnformatted(output.c_str());
                ImGui::PopTextWrapPos();
                if (output_changed && at_end) ImGui::SetScrollHereY(1);
                output_changed = false;
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }
    void model_tree(float x, float y, float w, float h) {
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Model###caliper.inspector.model")) {
            if (!supports("generate")) status();
            if (!model.empty()) ImGui::TextUnformatted(model.c_str());
            double total_bytes = 0;
            for (const auto& t : tensors) total_bytes += t.bytes;
            ImGui::Text("%zu tensors  %s", tensors.size(), byte_size(total_bytes).c_str());
            ImGui::SetNextItemWidth(-80);
            ImGui::InputTextWithHint("##filter", "Filter tensors", filter, sizeof filter);
            ImGui::SameLine();
            ImGui::BeginDisabled(!idle());
            if (ImGui::Button("Refresh")) want_list = true;
            ImGui::EndDisabled();
            if (ImGui::BeginTable("tensors", 4, ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV, ImVec2(0, 0))) {
                ImGui::TableSetupColumn("Tensor", ImGuiTableColumnFlags_WidthStretch, 3);
                ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthStretch, 1.5f);
                ImGui::TableSetupColumn("Dtype", ImGuiTableColumnFlags_WidthStretch, 1);
                ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch, 1);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();
                std::vector<std::string> groups;
                for (const auto& t : tensors)
                    if (std::find(groups.begin(), groups.end(), t.group) == groups.end()) groups.push_back(t.group);
                for (const auto& group : groups) {
                    bool matches = false;
                    for (const auto& t : tensors)
                        if (t.group == group && (filter[0] == '\0' || t.name.find(filter) != std::string::npos)) { matches = true; break; }
                    if (!matches) continue;
                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::PushID(group.c_str());
                    if (filter[0]) ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                    const bool open = ImGui::TreeNodeEx(group.c_str(), ImGuiTreeNodeFlags_SpanAllColumns);
                    if (open) {
                        for (const auto& t : tensors) {
                            if (t.group != group || (filter[0] && t.name.find(filter) == std::string::npos)) continue;
                            ImGui::TableNextRow(); ImGui::TableNextColumn();
                            ImGui::PushID(t.name.c_str());
                            std::string label = t.name;
                            const auto prefix = label.find(group + ".");
                            if (prefix != std::string::npos) label.erase(0, prefix + group.size() + 1);
                            ImGui::BeginDisabled(!idle());
                            if (ImGui::Selectable(label.c_str(), selected == t.name, ImGuiSelectableFlags_SpanAllColumns)) select(t.name);
                            ImGui::EndDisabled();
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%.0f values", t.name.c_str(), t.size);
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(t.shape.c_str());
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(t.dtype.c_str());
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(byte_size(t.bytes).c_str());
                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
    void edit(const char* action, double operand) {
        send("edit_weight", {{"name", selected}, {"action", action},
            {"row", static_cast<double>(row)}, {"col", static_cast<double>(col)}, {"value", operand}});
    }
    void tensor_view(float x, float y, float w, float h) {
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Tensor###caliper.inspector.tensor")) {
            const Tensor* t = tensor();
            if (!t) ImGui::TextUnformatted("Select a tensor");
            else {
                ImGui::TextWrapped("%s", t->name.c_str());
                ImGui::Text("%s  %s  %s", t->shape.c_str(), t->dtype.c_str(), byte_size(t->bytes).c_str());
                if (t->tied) ImGui::TextUnformatted("Shared with the output layer");
                ImGui::Text("Rows %d  Columns %d", t->rows, t->cols);
                ImGui::TextUnformatted("Dimensions before the last are flattened into rows.");
                if (!heatmap.empty()) {
                    ImGui::Text("Overview: %d x %d sampled values", sample_rows, sample_cols);
                    ImPlot::PushColormap(ImPlotColormap_Viridis);
                    if (ImPlot::BeginPlot("##weights", ImVec2(-1, std::max(150.0f, ImGui::GetContentRegionAvail().y * 0.42f)),
                            ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
                        ImPlot::SetupAxes("Column", "Row", ImPlotAxisFlags_Lock, ImPlotAxisFlags_Lock | ImPlotAxisFlags_Invert);
                        ImPlot::SetupAxesLimits(0, t->cols, 0, t->rows, ImPlotCond_Always);
                        ImPlot::PlotHeatmap("Weight", plot_heatmap.data(), sample_rows, sample_cols, heat_min, heat_max,
                            nullptr, ImPlotPoint(0, 0), ImPlotPoint(t->cols, t->rows));
                        if (ImPlot::IsPlotHovered()) {
                            const ImPlotPoint p = ImPlot::GetPlotMousePos();
                            if (p.x >= 0 && p.x < t->cols && p.y >= 0 && p.y < t->rows) {
                                const int c = std::min(sample_cols - 1, static_cast<int>(p.x * sample_cols / t->cols));
                                const int r = std::min(sample_rows - 1, static_cast<int>(p.y * sample_rows / t->rows));
                                ImGui::SetTooltip("Sample (%d, %d)\n%.8g", row_indices[r], col_indices[c], heatmap[r * sample_cols + c]);
                                if (idle() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                    row = row_indices[r]; col = col_indices[c]; have_value = false;
                                }
                            }
                        }
                        ImPlot::EndPlot();
                    }
                    ImPlot::PopColormap();
                    ImGui::Text("Overview range %.6g to %.6g", heat_min, heat_max);
                } else ImGui::TextUnformatted(active.op == "show_tensor" ? "Loading overview" : "Overview unavailable");
                ImGui::BeginDisabled(!idle());
                const float input_width = std::max(80.0f, (ImGui::GetContentRegionAvail().x - 100) * 0.5f);
                ImGui::SetNextItemWidth(input_width);
                if (ImGui::InputInt("Row", &row)) have_value = false;
                ImGui::SameLine(); ImGui::SetNextItemWidth(input_width);
                if (ImGui::InputInt("Col", &col)) have_value = false;
                row = std::clamp(row, 0, t->rows - 1); col = std::clamp(col, 0, t->cols - 1);
                ImGui::BeginDisabled(!supports("read_weight"));
                if (ImGui::Button("Read")) send("read_weight", {{"name", selected}, {"row", static_cast<double>(row)}, {"col", static_cast<double>(col)}});
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (have_value) ImGui::Text("(%d, %d) = %.9g", value_row, value_col, read_value);
                else ImGui::TextUnformatted("Value not read");
                ImGui::SetNextItemWidth(input_width);
                ImGui::InputDouble("Value", &value, 0, 0, "%.9g");
                ImGui::SameLine();
                ImGui::BeginDisabled(!supports("edit_weight") || !std::isfinite(value));
                if (ImGui::Button("Set")) edit("set", value);
                ImGui::EndDisabled();
                ImGui::SetNextItemWidth(input_width);
                ImGui::InputDouble("Scale", &factor, 0, 0, "%.6g");
                ImGui::BeginDisabled(!supports("edit_weight"));
                if (ImGui::Button("Zero row")) edit("zero_row", 0);
                ImGui::SameLine();
                if (ImGui::Button("Zero column")) edit("zero_column", 0);
                ImGui::BeginDisabled(!std::isfinite(factor));
                if (ImGui::Button("Scale row")) edit("scale_row", factor);
                ImGui::SameLine();
                if (ImGui::Button("Scale column")) edit("scale_column", factor);
                ImGui::EndDisabled();
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(!supports("revert_tensor"));
                if (ImGui::Button("Revert tensor")) send("revert_tensor", {{"name", selected}});
                ImGui::EndDisabled();
                ImGui::EndDisabled();
                if (active.op == "generate") ImGui::TextUnformatted("Stop generation to edit weights.");
                ImGui::Separator();
                ImGui::TextUnformatted("Edits");
                if (edits.empty()) ImGui::TextUnformatted("No edits");
                else for (const auto& edit : edits) ImGui::TextWrapped("%s", edit.c_str());
            }
        }
        ImGui::End();
    }
    void draw(float x, float y, float width, float height) {
        if (!visible) return;
        const float gap = 8;
        const bool have_inference = supports("generate") || capabilities.empty();
        const bool have_tensors = supports("list_tensors");
        const float left = have_tensors ? width * 0.42f : width;
        if (have_inference) inference(x, y, std::max(350.0f, left - gap), std::max(300.0f, height - gap));
        if (have_tensors) {
            const float tx = have_inference ? x + left : x;
            const float tw = have_inference ? width - left : width;
            const float tree_height = std::max(180.0f, height * 0.36f);
            model_tree(tx, y, std::max(350.0f, tw - gap), tree_height - gap);
            tensor_view(tx, y + tree_height, std::max(350.0f, tw - gap), std::max(300.0f, height - tree_height - gap));
        }
    }
};

Inspector::Inspector() : impl_(std::make_unique<Impl>()) {}
Inspector::~Inspector() = default;
void Inspector::poll(const std::string& session, bool alive) { impl_->poll(session, alive); }
void Inspector::draw(float x, float y, float width, float height) { impl_->draw(x, y, width, height); }

} // namespace caliper_live
