#include "live/live_inspector.h"
#include "live/live_requests.h"
#include "live/live_model3d.h"
#include "live/tensor_geometry.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"
#include "ui/sternbild_theme.h"

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
namespace Geometry = tensor_geometry;

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

struct Tensor : ModelTensor {
    std::string dtype;
    double bytes = 0, size = 0;
    bool vector() const { return shape.size() == 1; }
};
struct Pending {
    int id = 0;
    std::string op, purpose, name;
    Geometry::Region region;
    Geometry::Target target;
};
struct Sample {
    std::string name;
    Geometry::Region region;
    int rows = 0, cols = 0;
    std::vector<double> values;
    std::vector<int> row_indices, col_indices;
    double minimum = 0, maximum = 1;
    bool exact = false;
    uint64_t revision = 0, texture = 0;
    bool texture_dirty = true;
};
Object region_fields(Geometry::Region r) {
    return {{"row", static_cast<double>(r.row)}, {"col", static_cast<double>(r.col)},
            {"rows", static_cast<double>(r.rows)}, {"cols", static_cast<double>(r.cols)}};
}
ImVec2 point(double x, double y) { return ImVec2(static_cast<float>(x), static_cast<float>(y)); }
int layer_from_name(const std::string& name) {
    const auto start = name.find(".layers.");
    if (start == std::string::npos) return -1;
    const auto end = name.find('.', start + 8);
    const std::string value = name.substr(start + 8, end - start - 8);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) return -1;
    try { return std::stoi(value); } catch (...) { return -1; }
}

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
    std::string queued_op;
    Object queued_fields;
    Model3D model3d;
    caliper_host::HostRenderer* renderer = nullptr;
    Sample detail, overview;
    Geometry::Region region, drag_region;
    Geometry::Target target, last_target, overview_target;
    Geometry::Region last_region;
    bool region_ready = false, detail_needed = false, overview_needed = false;
    bool dragging = false, scroll_selected = false, open_selected_group = false;
    bool force_layout = false, reset_layout = true;
    double detail_due = 0;
    std::string detail_name;
    ImVec2 drag_origin;

    ~Impl() { release_sample(detail); release_sample(overview); }
    void release_sample(Sample& sample) {
        if (sample.texture && renderer) renderer->tex_release(sample.texture);
        sample.texture = 0;
    }
    void clear_samples() {
        release_sample(detail); release_sample(overview);
        detail = {}; overview = {};
    }
    bool background() const { return active.op == "show_tensor"; }
    bool can_control() const {
        return ready && alive && channel.connected() && queued_op.empty() &&
               (!active.id || background());
    }

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
    bool issue(const char* op, Object fields = {}, const std::string& purpose = {},
               const std::string& name = {}, Geometry::Region wanted = {}, Geometry::Target size = {}) {
        if (!idle() || !supports(op)) return false;
        std::string request_name = name;
        const auto named = fields.find("name");
        if (request_name.empty() && named != fields.end()) request_name = string(named->second);
        const int id = next_id++;
        fields["id"] = static_cast<double>(id);
        fields["op"] = op;
        if (!channel.send(Json(std::move(fields)).dump())) { error = channel.error(); return false; }
        active = {id, op, purpose, request_name, wanted, size};
        if (purpose.empty()) error.clear();
        return true;
    }
    bool send(const char* op, Object fields = {}) {
        if (!can_control() || !supports(op)) return false;
        if (active.id) { queued_op = op; queued_fields = std::move(fields); return true; }
        return issue(op, std::move(fields));
    }
    bool request_sample(const std::string& name, const char* purpose,
                        Geometry::Region wanted, Geometry::Target size) {
        Object fields{{"name", name}};
        if (supports("tensor_regions")) {
            fields["region"] = region_fields(wanted);
            fields["target"] = Object{{"width", static_cast<double>(size.width)},
                                      {"height", static_cast<double>(size.height)}};
        }
        return issue("show_tensor", std::move(fields), purpose, name, wanted, size);
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
    void select(const std::string& name, bool from_model3d = false) {
        if (from_model3d) {
            filter[0] = '\0'; open_selected_group = scroll_selected = true;
        }
        if (selected == name) return;
        selected = name;
        row = col = 0; have_value = false;
        clear_samples();
        region_ready = false; detail_needed = true; overview_needed = true; detail_due = 0;
        edits.clear();
        open_selected_group = scroll_selected = true;
    }
    bool read_tensors(const Json& result) {
        const Json& items = field(result, "tensors");
        if (!items.is_array() || items.get<Array>().size() > 10000) return false;
        std::vector<Tensor> next;
        for (const Json& item : items.get<Array>()) {
            Tensor t;
            t.name = string(field(item, "name")); t.group = string(field(item, "group"), "Model");
            t.dtype = string(field(item, "dtype")); t.rows = integer(field(item, "rows"));
            t.cols = integer(field(item, "cols")); t.bytes = number(field(item, "bytes"));
            t.size = number(field(item, "size")); t.tied = boolean(field(item, "tied"));
            t.edited = boolean(field(item, "edited")); t.revision = integer(field(item, "revision"));
            t.layer_index = integer(field(item, "layer_index"), layer_from_name(t.name));
            t.layer_type = string(field(item, "layer_type"));
            if (t.layer_type.empty()) {
                if (t.name.find(".linear_attn.") != std::string::npos) t.layer_type = "linear_attention";
                else if (t.name.find(".self_attn.") != std::string::npos) t.layer_type = "full_attention";
            }
            t.role = string(field(item, "role"));
            if (t.role.empty()) {
                if (t.name.find("embed_tokens.weight") != std::string::npos) t.role = "embedding";
                else if (t.layer_index < 0 && t.name.find("norm.weight") != std::string::npos) t.role = "final_norm";
                else t.role = "weight";
            }
            const Json& shape = field(item, "shape");
            if (t.name.empty() || t.name.size() > 1024 || !shape.is_array() ||
                t.rows <= 0 || t.cols <= 0 || t.bytes < 0 || t.size < 0) return false;
            t.shape_label = "[";
            for (const Json& dimension : shape.get<Array>()) {
                if (t.shape_label.size() > 1) t.shape_label += ", ";
                const int d = integer(dimension, -1); if (d < 0) return false;
                t.shape.push_back(d); t.shape_label += std::to_string(d);
            }
            t.shape_label += "]";
            static_cast<ModelTensor&>(t).shape_label = t.shape_label;
            next.push_back(std::move(t));
        }
        // Older hosts expose attention kinds only in some tensor names.
        for (auto& t : next) if (t.layer_type.empty() && t.layer_index >= 0)
            for (const auto& other : next) if (other.layer_index == t.layer_index && !other.layer_type.empty()) {
                t.layer_type = other.layer_type; break;
            }
        tensors = std::move(next); model = string(field(result, "model"));
        std::vector<ModelTensor> scene; scene.reserve(tensors.size());
        for (const auto& t : tensors) scene.push_back(t);
        model3d.set_tensors(scene);
        if (!tensor()) {
            if (tensors.empty()) selected.clear(); else select(tensors.front().name);
        } else { detail_needed = overview_needed = true; detail_due = 0; }
        return true;
    }
    bool parse_sample(const Json& result, const Pending& request, Sample& sample) {
        if (string(field(result, "name")) != request.name) return false;
        const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor& t) { return t.name == request.name; });
        if (it == tensors.end()) return false;
        const int nr = integer(field(result, "sample_rows")), nc = integer(field(result, "sample_cols"));
        const auto& values = field(result, "values"); const auto& ri = field(result, "row_indices");
        const auto& ci = field(result, "col_indices");
        if (nr < 1 || nc < 1 || nr > Geometry::max_target_side || nc > Geometry::max_target_side ||
            static_cast<int64_t>(nr) * nc > Geometry::max_samples || !values.is_array() ||
            values.get<Array>().size() != static_cast<size_t>(nr * nc) || !ri.is_array() || !ci.is_array() ||
            ri.get<Array>().size() != static_cast<size_t>(nr) || ci.get<Array>().size() != static_cast<size_t>(nc)) return false;
        Sample next; next.name = request.name; next.rows = nr; next.cols = nc;
        const auto& received = field(result, "region");
        if (received.is_object()) {
            next.region = {integer(field(received, "row")), integer(field(received, "col")),
                           integer(field(received, "rows")), integer(field(received, "cols"))};
            if (next.region != Geometry::clamp(next.region, it->rows, it->cols)) return false;
        } else next.region = {0, 0, it->rows, it->cols};
        for (const Json& v : values.get<Array>()) {
            if (!v.is_number() || !std::isfinite(v.get<double>())) return false;
            next.values.push_back(v.get<double>());
        }
        int previous = -1;
        for (const Json& v : ri.get<Array>()) {
            const int i = integer(v, -1);
            if (i < next.region.row || i >= next.region.row + next.region.rows || i <= previous) return false;
            next.row_indices.push_back(i); previous = i;
        }
        previous = -1;
        for (const Json& v : ci.get<Array>()) {
            const int i = integer(v, -1);
            if (i < next.region.col || i >= next.region.col + next.region.cols || i <= previous) return false;
            next.col_indices.push_back(i); previous = i;
        }
        const auto range = std::minmax_element(next.values.begin(), next.values.end());
        next.minimum = *range.first; next.maximum = *range.second;
        next.exact = nr == next.region.rows && nc == next.region.cols;
        next.revision = integer(field(result, "revision"));
        release_sample(sample); sample = std::move(next);
        return true;
    }
    void update_edited(const std::string& name, const Json& result, bool fallback) {
        for (auto& t : tensors) if (t.name == name) {
            t.edited = field(result, "edited").is_boolean() ? boolean(field(result, "edited")) : fallback;
            t.revision = integer(field(result, "revision"), static_cast<int>(t.revision + 1));
            model3d.invalidate(name, t.edited); break;
        }
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
        const Pending completed = active;
        const std::string op = completed.op;
        active = {};
        if (field(reply, "error").is_string()) {
            if (completed.purpose == "thumbnail") {
                model3d.thumbnail_failed(completed.name); return;
            }
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
            Sample sample;
            if (!parse_sample(result, completed, sample)) {
                if (completed.purpose == "thumbnail") model3d.thumbnail_failed(completed.name);
                else protocol_error("Invalid tensor region from script");
                return;
            }
            if (completed.purpose == "thumbnail") {
                std::vector<float> values(sample.values.begin(), sample.values.end());
                if (!model3d.set_thumbnail(completed.name, sample.rows, sample.cols, values,
                                           sample.minimum, sample.maximum, sample.revision))
                    model3d.thumbnail_failed(completed.name);
            } else if (completed.name == selected) {
                if (completed.purpose == "overview") {
                    release_sample(overview); overview = std::move(sample);
                } else if (!supports("tensor_regions") || completed.region == region) {
                    release_sample(detail); detail = std::move(sample); read_edits(result);
                }
            }
        } else if (op == "read_weight") {
            if (completed.name != selected) return;
            const Json& v = field(result, "value");
            if (string(field(result, "name")) != selected || !v.is_number() || !std::isfinite(v.get<double>())) {
                error = "Invalid weight value from script"; return;
            }
            value_row = integer(field(result, "row"));
            value_col = integer(field(result, "col"));
            read_value = value = v.get<double>();
            have_value = true;
        } else if (op == "edit_weight" || op == "revert_tensor") {
            update_edited(completed.name, result, op == "edit_weight");
            if (completed.name == selected) {
                have_value = false; read_edits(result);
                detail_needed = overview_needed = true; detail_due = 0;
            }
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
            if (active.op == "generate" || queued_op == "generate") generation_status = "Generation interrupted";
            if (active.id) error = "Connection ended; the operation was not retried";
            else if (visible && error.empty()) error = alive ? channel.error() : "Session ended";
            if (active.purpose == "thumbnail") model3d.thumbnail_failed(active.name);
            active = {};
            stop_requested = false;
            want_list = false; queued_op.clear(); queued_fields.clear();
            retry_at = now + 1.0;
        }
        if (!idle()) return;
        if (!queued_op.empty()) {
            const std::string op = std::move(queued_op); Object fields = std::move(queued_fields);
            queued_op.clear(); queued_fields.clear(); issue(op.c_str(), std::move(fields));
        } else if (want_list) { want_list = false; issue("list_tensors"); }
        else if (tensor() && region_ready && detail_needed && now >= detail_due) {
            if (request_sample(selected, "detail", region, target)) {
                detail_needed = false; last_region = region; last_target = target;
            }
        } else if (tensor() && region_ready && overview_needed && supports("tensor_regions")) {
            const Tensor& t = *tensor();
            if (request_sample(selected, "overview", {0, 0, t.rows, t.cols}, overview_target)) overview_needed = false;
        }
    }
    void thumbnails() {
        if (!idle() || !queued_op.empty() ||
            (region_ready && (detail_needed || (overview_needed && supports("tensor_regions"))))) return;
        const std::string name = model3d.next_thumbnail();
        if (name.empty()) return;
        const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor& t) { return t.name == name; });
        if (it == tensors.end()) { model3d.thumbnail_failed(name); return; }
        Geometry::Target size = it->vector() ? Geometry::Target{1, 128} : Geometry::Target{128, 128};
        if (!request_sample(name, "thumbnail", {0, 0, it->rows, it->cols}, size)) model3d.thumbnail_failed(name);
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
    void place_window(const char*, float x, float y, float w, float h) {
        ImGui::SetNextWindowPos(ImVec2(x, y), force_layout ? ImGuiCond_Always : ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), force_layout ? ImGuiCond_Always : ImGuiCond_FirstUseEver);
    }
    void inference(float x, float y, float w, float h) {
        place_window("Inference###caliper.inspector.inference", x, y, w, h);
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
            ImGui::BeginDisabled(!can_control() || prompt[0] == '\0');
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
        place_window("Model###caliper.inspector.model", x, y, w, h);
        if (ImGui::Begin("Model###caliper.inspector.model")) {
            if (!supports("generate")) status();
            if (!model.empty()) ImGui::TextUnformatted(model.c_str());
            double total_bytes = 0;
            for (const auto& t : tensors) total_bytes += t.bytes;
            ImGui::Text("%zu tensors  %s", tensors.size(), byte_size(total_bytes).c_str());
            if (ImGui::SmallButton("Reset layout")) reset_layout = true;
            ImGui::SetNextItemWidth(-80);
            ImGui::InputTextWithHint("##filter", "Filter tensors", filter, sizeof filter);
            ImGui::SameLine();
            ImGui::BeginDisabled(!can_control());
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
                    if (filter[0] || (open_selected_group && tensor() && tensor()->group == group))
                        ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                    const bool open = ImGui::TreeNodeEx(group.c_str(), ImGuiTreeNodeFlags_SpanAllColumns);
                    if (open) {
                        for (const auto& t : tensors) {
                            if (t.group != group || (filter[0] && t.name.find(filter) == std::string::npos)) continue;
                            ImGui::TableNextRow(); ImGui::TableNextColumn();
                            ImGui::PushID(t.name.c_str());
                            std::string label = t.name;
                            const auto prefix = label.find(group + ".");
                            if (prefix != std::string::npos) label.erase(0, prefix + group.size() + 1);
                            ImGui::BeginDisabled(!can_control());
                            if (ImGui::Selectable(label.c_str(), selected == t.name, ImGuiSelectableFlags_SpanAllColumns)) select(t.name);
                            ImGui::EndDisabled();
                            if (scroll_selected && selected == t.name) { ImGui::SetScrollHereY(0.5f); scroll_selected = false; }
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%.0f values", t.name.c_str(), t.size);
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(t.shape_label.c_str());
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(t.dtype.c_str());
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(byte_size(t.bytes).c_str());
                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
                open_selected_group = false;
            }
        }
        ImGui::End();
    }
    void edit(const char* action, double operand) {
        send("edit_weight", {{"name", selected}, {"action", action},
            {"row", static_cast<double>(row)}, {"col", static_cast<double>(col)}, {"value", operand}});
    }
    void make_texture(Sample& sample, bool vector) {
        if (!renderer || sample.values.empty() || !sample.texture_dirty) return;
        const int width = vector ? sample.rows : sample.cols;
        const int height = vector ? sample.cols : sample.rows;
        release_sample(sample);
        std::vector<float> values(sample.values.begin(), sample.values.end());
        std::vector<uint8_t> rgba(values.size() * 4);
        float lo = static_cast<float>(sample.minimum), hi = static_cast<float>(sample.maximum);
        if (lo == hi) { lo -= 0.5f; hi += 0.5f; }
        caliper_host::map_f32_to_rgba8(values.data(), width, height,
            caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS), lo, hi, rgba.data());
        sample.texture = renderer->tex_create_rgba8(width, height);
        if (sample.texture && !renderer->tex_upload_rgba8(sample.texture, rgba.data(), width, height))
            release_sample(sample);
        sample.texture_dirty = false;
    }
    ImU32 heat_color(double value, const Sample& sample) const {
        const double span = sample.maximum - sample.minimum;
        const double amount = span > 0 ? (value - sample.minimum) / span : 0.5;
        const int index = std::clamp(static_cast<int>(amount * 255 + 0.5), 0, 255);
        return caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS)[index];
    }
    bool picture_contains(const Sample& sample, Geometry::Region shown) const {
        return !sample.values.empty() && shown.row >= sample.region.row && shown.col >= sample.region.col &&
            shown.row + shown.rows <= sample.region.row + sample.region.rows &&
            shown.col + shown.cols <= sample.region.col + sample.region.cols;
    }
    void paint_image(Sample& sample, Geometry::Region shown, Geometry::Rect picture, bool vector) {
        if (picture.width <= 0 || picture.height <= 0 || !picture_contains(sample, shown)) return;
        ImDrawList* list = ImGui::GetWindowDrawList();
        make_texture(sample, vector);
        const double u0 = static_cast<double>(shown.col - sample.region.col) / sample.region.cols;
        const double v0 = static_cast<double>(shown.row - sample.region.row) / sample.region.rows;
        const double u1 = static_cast<double>(shown.col + shown.cols - sample.region.col) / sample.region.cols;
        const double v1 = static_cast<double>(shown.row + shown.rows - sample.region.row) / sample.region.rows;
        if (sample.texture && renderer) {
            list->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(sample.texture)),
                point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height),
                vector ? point(v0, u0) : point(u0, v0), vector ? point(v1, u1) : point(u1, v1));
        } else {
            // The CPU fixture takes the same rectangles and colormap as the
            // texture path; a normal app frame draws a single textured quad.
            list->PushClipRect(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height), true);
            for (int r = 0; r < sample.rows; ++r) for (int c = 0; c < sample.cols; ++c) {
                const double row0 = sample.region.row + static_cast<double>(r) * sample.region.rows / sample.rows;
                const double col0 = sample.region.col + static_cast<double>(c) * sample.region.cols / sample.cols;
                const double row1 = sample.region.row + static_cast<double>(r + 1) * sample.region.rows / sample.rows;
                const double col1 = sample.region.col + static_cast<double>(c + 1) * sample.region.cols / sample.cols;
                const double ax = vector ? (row0 - shown.row) / shown.rows : (col0 - shown.col) / shown.cols;
                const double bx = vector ? (row1 - shown.row) / shown.rows : (col1 - shown.col) / shown.cols;
                const double ay = vector ? (col0 - shown.col) / shown.cols : (row0 - shown.row) / shown.rows;
                const double by = vector ? (col1 - shown.col) / shown.cols : (row1 - shown.row) / shown.rows;
                list->AddRectFilled(point(picture.x + ax * picture.width, picture.y + ay * picture.height),
                    point(picture.x + bx * picture.width, picture.y + by * picture.height),
                    heat_color(sample.values[r * sample.cols + c], sample));
            }
            list->PopClipRect();
        }
        list->AddRect(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height),
                      ImGui::GetColorU32(ImGuiCol_Border));
    }
    void color_scale(const Sample& sample, Geometry::Rect area) {
        if (sample.values.empty()) return;
        auto* list = ImGui::GetWindowDrawList();
        const double height = std::max(30.0, area.height - 34);
        const auto* lut = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
        for (int i = 0; i < 128; ++i) {
            const double y0 = area.y + i * height / 128, y1 = area.y + (i + 1) * height / 128;
            list->AddRectFilled(point(area.x, y0), point(area.x + 12, y1), lut[255 - i * 2]);
        }
        char high[32], low[32];
        std::snprintf(high, sizeof high, "%.5g", sample.maximum);
        std::snprintf(low, sizeof low, "%.5g", sample.minimum);
        list->AddText(point(area.x + 16, area.y), ImGui::GetColorU32(ImGuiCol_Text), high);
        list->AddText(point(area.x + 16, area.y + height - ImGui::GetTextLineHeight()), ImGui::GetColorU32(ImGuiCol_Text), low);
        list->AddText(point(area.x, area.y + height + 4), ImGui::GetColorU32(ImGuiCol_Text), "Value");
    }
    void request_detail(Geometry::Region next, Geometry::Target size, bool immediately = false) {
        const Tensor* t = tensor(); if (!t) return;
        next = Geometry::clamp(next, t->rows, t->cols);
        const bool changed = region != next || target.width != size.width || target.height != size.height;
        region = next; target = size;
        if (supports("tensor_regions") && (last_region != region || last_target.width != size.width || last_target.height != size.height)) {
            if (!detail_needed || changed) detail_due = immediately || detail.values.empty() ? 0 : seconds() + 0.05;
            detail_needed = true;
        }
    }
    void show_hover(const Sample& sample, Geometry::Cell cell) {
        if (!cell.valid || sample.values.empty()) return;
        const auto r = std::lower_bound(sample.row_indices.begin(), sample.row_indices.end(), static_cast<int>(cell.row));
        const auto c = std::lower_bound(sample.col_indices.begin(), sample.col_indices.end(), static_cast<int>(cell.col));
        if (r != sample.row_indices.end() && c != sample.col_indices.end() && *r == cell.row && *c == cell.col) {
            const auto index = static_cast<size_t>(r - sample.row_indices.begin()) * sample.cols + (c - sample.col_indices.begin());
            ImGui::SetTooltip("(%lld, %lld)\n%.9g", static_cast<long long>(cell.row), static_cast<long long>(cell.col), sample.values[index]);
        } else {
            ImGui::SetTooltip("(%lld, %lld)\nZoom to cells or use Read for the exact value", static_cast<long long>(cell.row), static_cast<long long>(cell.col));
        }
    }
    void tensor_picture(const Tensor& t, Geometry::Rect allocated) {
        const auto layout = Geometry::view_layout(t.rows, t.cols, t.vector(), allocated);
        const Geometry::Rect detail_bounds = t.vector() ? layout.strip : layout.detail;
        if (!region_ready) {
            region = Geometry::initial_region(t.rows, t.cols, detail_bounds, t.vector());
            region_ready = true; detail_needed = true; detail_due = 0;
        }
        Geometry::Rect picture = t.vector() ? Geometry::fit(region.cols, region.rows, detail_bounds)
                                             : Geometry::fit(region.rows, region.cols, detail_bounds);
        const float scale = std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x);
        auto size = Geometry::sampling_target(region, picture, scale, t.vector());
        if (t.vector()) std::swap(size.width, size.height);
        request_detail(region, size);
        if (supports("tensor_regions") && (layout.elongated || t.vector())) {
            auto size = t.vector()
                ? Geometry::sampling_target({0, 0, t.rows, t.cols},
                    {layout.plot.x, layout.plot.y, layout.plot.width, 1}, scale, true)
                : Geometry::sampling_target({0, 0, t.rows, t.cols}, layout.overview, scale, false);
            if (t.vector()) std::swap(size.width, size.height);
            if (size.width != overview_target.width || size.height != overview_target.height) {
                overview_target = size; overview_needed = true;
            }
        } else overview_needed = false;
        if (!supports("tensor_regions")) {
            overview_needed = false;
            if (detail.name == selected && !detail.values.empty()) detail_needed = false;
        }
        if (t.vector() && layout.plot.width > 0 && layout.plot.height > 0) {
            ImGui::SetCursorScreenPos(point(layout.plot.x, layout.plot.y));
            if (ImPlot::BeginPlot("##vector.values", point(layout.plot.width, layout.plot.height), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_Lock, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, t.rows, ImPlotCond_Always);
                const Sample& values = overview.values.empty() ? detail : overview;
                if (!values.values.empty()) {
                    std::vector<double> indices(values.row_indices.begin(), values.row_indices.end());
                    ImPlot::PlotLine("Value", indices.data(), values.values.data(), static_cast<int>(indices.size()));
                }
                const ImVec2 pos = ImPlot::GetPlotPos(), extent = ImPlot::GetPlotSize();
                const auto band = Geometry::selection_rect({region.col, region.row, region.cols, region.rows},
                    t.cols, t.rows, {pos.x, pos.y, extent.x, extent.y});
                auto* draw = ImPlot::GetPlotDrawList();
                draw->PushClipRect(pos, ImVec2(pos.x + extent.x, pos.y + extent.y), true);
                draw->AddRectFilled(point(band.x, band.y), point(band.x + band.width, band.y + band.height),
                                    ImGui::GetColorU32(ImGuiCol_Header, 0.35f));
                draw->AddRect(point(band.x, band.y), point(band.x + band.width, band.y + band.height),
                              ImGui::GetColorU32(ImGuiCol_PlotLines));
                draw->PopClipRect();
                ImPlot::EndPlot();
            }
        }
        paint_image(detail, region, picture, t.vector());
        auto* list = ImGui::GetWindowDrawList();
        const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
        char axis[100];
        if (t.vector()) std::snprintf(axis, sizeof axis, "Index %lld to %lld", static_cast<long long>(region.row), static_cast<long long>(region.row + region.rows - 1));
        else std::snprintf(axis, sizeof axis, "Column %lld to %lld", static_cast<long long>(region.col), static_cast<long long>(region.col + region.cols - 1));
        list->AddText(point(detail_bounds.x, detail_bounds.y + detail_bounds.height + 3), text, axis);
        if (!t.vector()) {
            std::snprintf(axis, sizeof axis, "Row %lld", static_cast<long long>(region.row));
            list->AddText(point(detail_bounds.x, detail_bounds.y - ImGui::GetTextLineHeight()), text, axis);
            std::snprintf(axis, sizeof axis, "%lld", static_cast<long long>(region.row + region.rows - 1));
            list->AddText(point(detail_bounds.x, detail_bounds.y + detail_bounds.height - ImGui::GetTextLineHeight()), text, axis);
        }
        if (layout.elongated) {
            Sample& sample = overview.values.empty() ? detail : overview;
            const Geometry::Region whole{0, 0, t.rows, t.cols};
            paint_image(sample, whole, layout.overview, false);
            const auto selection = Geometry::selection_rect(region, t.rows, t.cols, layout.overview);
            list->AddRect(point(selection.x, selection.y), point(selection.x + selection.width, selection.y + selection.height),
                          ImGui::GetColorU32(ImGuiCol_PlotLines), 0.0f, 2.0f, ImDrawFlags_None);
            ImGui::SetCursorScreenPos(point(layout.overview.x, layout.overview.y));
            ImGui::InvisibleButton("##overview", point(std::max(1.0, layout.overview.width), std::max(1.0, layout.overview.height)));
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Whole tensor\nClick to move the visible region");
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    request_detail(Geometry::overview_region(region, t.rows, t.cols, layout.overview,
                                    ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y), size, true);
            }
        }
        ImGui::SetCursorScreenPos(point(detail_bounds.x, detail_bounds.y));
        ImGui::InvisibleButton("##tensor.picture", point(std::max(1.0, detail_bounds.width), std::max(1.0, detail_bounds.height)),
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        const auto mouse = ImGui::GetIO().MousePos;
        const auto cell = Geometry::hit_test(region, picture, mouse.x, mouse.y, t.vector());
        if (ImGui::IsItemHovered() && cell.valid) {
            show_hover(detail, cell);
            if (ImGui::GetIO().MouseWheel != 0)
                request_detail(Geometry::zoom(region, t.rows, t.cols, std::pow(0.8, ImGui::GetIO().MouseWheel), cell), size, true);
        }
        if (ImGui::IsItemActivated() && cell.valid) {
            dragging = true; drag_region = region; drag_origin = mouse;
            row = static_cast<int>(cell.row); col = static_cast<int>(cell.col); have_value = false;
        }
        if (dragging && ImGui::IsItemActive() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle)))
            request_detail(Geometry::pan(drag_region, t.rows, t.cols, mouse.x - drag_origin.x, mouse.y - drag_origin.y, picture, t.vector()), size, true);
        if (!ImGui::IsItemActive()) dragging = false;
    }
    void tensor_view(float x, float y, float w, float h) {
        place_window("Tensor###caliper.inspector.tensor", x, y, w, h);
        if (ImGui::Begin("Tensor###caliper.inspector.tensor")) {
            const Tensor* t = tensor();
            if (!t) ImGui::TextUnformatted("Select a tensor");
            else {
                ImGui::TextWrapped("%s", t->name.c_str());
                ImGui::Text("%s  %s  %s", t->shape_label.c_str(), t->dtype.c_str(), byte_size(t->bytes).c_str());
                if (t->tied) ImGui::TextUnformatted("Shared with the output layer");
                ImGui::Text("Rows %lld  Columns %lld", static_cast<long long>(t->rows), static_cast<long long>(t->cols));
                if (!supports("tensor_regions")) ImGui::TextUnformatted("Overview only: restart the host to enable region sampling.");
                ImGui::BeginDisabled(!can_control());
                if (ImGui::Button("Fit tensor")) {
                    if (t->vector()) region_ready = false;
                    else { region = {0, 0, t->rows, t->cols}; region_ready = true; }
                    detail_needed = supports("tensor_regions"); detail_due = 0;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cells")) {
                    const auto extent = ImGui::GetContentRegionAvail();
                    Geometry::Region cells;
                    cells.rows = t->vector() ? std::max(1, static_cast<int>(extent.x / Geometry::strip_height)) : 12;
                    cells.cols = t->vector() ? 1 : std::max(1, static_cast<int>(extent.x / Geometry::strip_height));
                    cells.row = row - cells.rows / 2; cells.col = col - cells.cols / 2;
                    region = Geometry::clamp(cells, t->rows, t->cols); region_ready = true;
                    detail_needed = supports("tensor_regions"); detail_due = 0;
                }
                ImGui::EndDisabled();
                const auto available = ImGui::GetContentRegionAvail();
                const double image_height = std::clamp(static_cast<double>(available.y) - 225.0, t->vector() ? 130.0 : 140.0, 330.0);
                const auto start = ImGui::GetCursorScreenPos();
                const Geometry::Rect image_area{start.x + (t->vector() ? 0 : 32), start.y + 18,
                    std::max(20.0, static_cast<double>(available.x) - (t->vector() ? 85 : 115)), std::max(20.0, image_height - 42)};
                tensor_picture(*t, image_area);
                color_scale(detail, {start.x + available.x - 78, start.y + 18, 78, image_height - 18});
                ImGui::SetCursorScreenPos(start);
                ImGui::Dummy(ImVec2(available.x, static_cast<float>(image_height)));
                if (!detail.values.empty()) ImGui::Text("Region: %d x %d sampled values%s", detail.rows, detail.cols, detail.exact ? " (exact cells)" : "");
                else ImGui::TextUnformatted("Loading region");
                ImGui::TextUnformatted("Wheel to zoom; drag to pan. Select cells or use Row and Col.");
                ImGui::BeginDisabled(!can_control());
                const float input_width = std::max(80.0f, (ImGui::GetContentRegionAvail().x - 100) * 0.5f);
                ImGui::SetNextItemWidth(input_width);
                if (ImGui::InputInt("Row", &row)) have_value = false;
                ImGui::SameLine(); ImGui::SetNextItemWidth(input_width);
                if (ImGui::InputInt("Col", &col)) have_value = false;
                row = std::clamp(row, 0, static_cast<int>(t->rows - 1)); col = std::clamp(col, 0, static_cast<int>(t->cols - 1));
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
    void draw(float x, float y, float width, float height, caliper_host::HostRenderer* current_renderer) {
        if (!visible) return;
        if (renderer != current_renderer) {
            clear_samples(); renderer = current_renderer; detail_needed = overview_needed = true;
        }
        force_layout = reset_layout; reset_layout = false;
        const auto windows = Geometry::window_layout({x, y, width, height});
        auto draw_window = [](auto function, Geometry::Rect r) {
            function(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.width), static_cast<float>(r.height));
        };
        if (supports("generate") || capabilities.empty())
            draw_window([&](float a,float b,float c,float d) { inference(a,b,c,d); }, windows.inference);
        if (supports("list_tensors")) {
            draw_window([&](float a,float b,float c,float d) { model_tree(a,b,c,d); }, windows.tree);
            const auto r = windows.model3d;
            if (force_layout) {
                ImGui::SetWindowPos("Model 3D###caliper.inspector.model3d", point(r.x, r.y));
                ImGui::SetWindowSize("Model 3D###caliper.inspector.model3d", point(r.width, r.height));
            }
            const std::string clicked = model3d.draw(renderer, r.x, r.y, r.width, r.height, selected);
            if (!clicked.empty()) {
                select(clicked, true);
                ImGui::SetWindowCollapsed("Tensor###caliper.inspector.tensor", false);
                ImGui::SetWindowFocus("Tensor###caliper.inspector.tensor");
            }
            draw_window([&](float a,float b,float c,float d) { tensor_view(a,b,c,d); }, windows.tensor);
            thumbnails();
        }
    }
};

Inspector::Inspector() : impl_(std::make_unique<Impl>()) {}
Inspector::~Inspector() = default;
void Inspector::poll(const std::string& session, bool alive) { impl_->poll(session, alive); }
void Inspector::draw(float x, float y, float width, float height, caliper_host::HostRenderer* renderer) { impl_->draw(x, y, width, height, renderer); }

} // namespace caliper_live
