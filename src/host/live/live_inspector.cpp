#include "live/live_inspector.h"
#include "live/live_requests.h"
#include "live/live_model3d.h"
#include "live/live_trace.h"
#include "live/tensor_geometry.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"
#include "ui/sternbild_theme.h"

#include <crude_json.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>

#include <algorithm>
#include <array>
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
// Every value of a tensor, fetched once in whole rows (tensor_values), so
// zoom and pan redraw from one mipmapped texture without asking the host.
struct Values {
    std::string name;
    int rows = 0, cols = 0, loaded = 0;
    uint64_t revision = 0, texture = 0;
    std::vector<float> data;
    double minimum = 0, maximum = 1;
    bool texture_dirty = true;
    bool complete() const { return !name.empty() && rows > 0 && loaded == rows; }
};
// A texture side every Metal GPU supports, and the most values kept (64 MB):
// every tensor of the 0.8B model but the embedding.
constexpr int kMaxValueSide = 16384;
constexpr int64_t kMaxValues = int64_t{1} << 24;

bool decode_base64(const std::string& text, std::vector<uint8_t>& out) {
    static const auto table = [] {
        std::array<int8_t, 256> t{}; t.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) t[static_cast<uint8_t>(alphabet[i])] = static_cast<int8_t>(i);
        return t;
    }();
    if (text.size() % 4) return false;
    out.clear(); out.reserve(text.size() / 4 * 3);
    for (size_t i = 0; i < text.size(); i += 4) {
        uint32_t n = 0; int padding = 0;
        for (size_t k = 0; k < 4; ++k) {
            const char c = text[i + k];
            if (c == '=' && i + 4 == text.size() && k >= 2) { ++padding; n <<= 6; continue; }
            const int v = table[static_cast<uint8_t>(c)];
            if (v < 0 || padding) return false;
            n = n << 6 | static_cast<uint32_t>(v);
        }
        out.push_back(static_cast<uint8_t>(n >> 16));
        if (padding < 2) out.push_back(static_cast<uint8_t>(n >> 8));
        if (padding < 1) out.push_back(static_cast<uint8_t>(n));
    }
    return true;
}

Object region_fields(Geometry::Region r) {
    return {{"row", static_cast<double>(r.row)}, {"col", static_cast<double>(r.col)},
            {"rows", static_cast<double>(r.rows)}, {"cols", static_cast<double>(r.cols)}};
}
ImVec2 point(double x, double y) { return ImVec2(static_cast<float>(x), static_cast<float>(y)); }
// 248320 as "248,320".
std::string grouped(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}
// The four panes, docked into one dockspace like the landing screen's panes.
constexpr const char* kInferencePane = "Inference###caliper.inspector.inference";
constexpr const char* kModelPane = "Model###caliper.inspector.model";
constexpr const char* kModel3DPane = "Model 3D###caliper.inspector.model3d";
constexpr const char* kTensorPane = "Tensor###caliper.inspector.tensor";
constexpr const char* kTracePane = "Trace###caliper.inspector.trace";

// A pane: a window whose pane_header strip replaces the title bar, bronze while
// the pane has focus. Its tables and output scroll inside it, never the pane.
bool begin_pane(const char* window, const char* title, const char* subtitle) {
    const bool open = ImGui::Begin(window, nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (open) sternbild::pane_header(title, subtitle, nullptr, ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows));
    return open;
}
// A form label on the line of the field after it, in the secondary ink.
void field_label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
void faint_text(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
// Text on the current line, right-aligned to the content edge when there is room.
void right_text(const char* text, ImU32 colour = sternbild::col::secondary) {
    ImGui::SameLine();
    const float width = ImGui::CalcTextSize(text).x;
    const float x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - width;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
std::string layer_kind(const std::string& type) {
    if (type.find("full") != std::string::npos) return "full attention";
    if (type.find("linear") != std::string::npos) return "linear attention";
    return {};
}

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
    std::string queued_op, queued_purpose;
    Object queued_fields;
    TracePane trace_pane;
    Model3D model3d;
    caliper_host::HostRenderer* renderer = nullptr;
    Sample detail, overview;
    Values values, incoming;
    bool values_needed = false, values_stale = false;
    std::string values_failed;
    // The view: a continuous camera, the window it sees, and the whole cells
    // that window touches (what a region request samples).
    Geometry::Camera camera, drag_camera;
    Geometry::Window window;
    Geometry::Rect view_bounds;
    Geometry::Region region;
    double scroll_grab = 0;
    Geometry::Target target, last_target, overview_target;
    Geometry::Region last_region;
    bool region_ready = false, detail_needed = false, overview_needed = false;
    bool dragging = false, scroll_selected = false, open_selected_group = false;
    // A click without a drag selects a cell; its value is read into the edit bar.
    Geometry::Cell press_cell;
    bool cell_selected = false, focus_value = false, focus_after_read = false;
    bool force_layout = false, reset_layout = true, docked = false, layout_built = false;
    double detail_due = 0;
    std::string detail_name;
    ImVec2 drag_origin;

    ~Impl() { release_sample(detail); release_sample(overview); release_values(values); }
    void release_sample(Sample& sample) {
        if (sample.texture && renderer) renderer->tex_release(sample.texture);
        sample.texture = 0;
    }
    void release_values(Values& v) {
        if (v.texture && renderer) renderer->tex_release(v.texture);
        v.texture = 0; v.texture_dirty = true;
    }
    bool uses_values(const Tensor& t) const {
        return supports("tensor_values") && t.name != values_failed && t.rows <= kMaxValueSide &&
               t.cols <= kMaxValueSide && static_cast<int64_t>(t.rows) * t.cols <= kMaxValues;
    }
    bool values_shown() const { return values.complete() && values.name == selected; }
    // Whether the Tensor pane still waits for data; thumbnails wait behind it.
    bool view_waiting() const {
        const Tensor* t = tensor();
        if (!t || !region_ready) return false;
        if (uses_values(*t)) return values_needed;
        return detail_needed || (overview_needed && supports("tensor_regions"));
    }
    void clear_samples() {
        release_sample(detail); release_sample(overview);
        detail = {}; overview = {};
    }
    bool background() const { return active.op == "show_tensor" || active.op == "tensor_values"; }
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
    bool send(const char* op, Object fields = {}, const std::string& purpose = {}) {
        if (!can_control() || !supports(op)) return false;
        if (active.id) { queued_op = op; queued_fields = std::move(fields); queued_purpose = purpose; return true; }
        return issue(op, std::move(fields), purpose);
    }
    // A trace request's run: the intact one, or the one with interventions.
    int trace_run(const Pending& request) const { return request.purpose == "intervened" ? 1 : 0; }
    void request_values(const Tensor& t) {
        if (incoming.name != t.name || incoming.rows != t.rows || incoming.cols != t.cols) {
            incoming = {}; incoming.name = t.name; incoming.rows = t.rows; incoming.cols = t.cols;
            incoming.data.assign(static_cast<size_t>(t.rows) * t.cols, 0.0f);
        }
        issue("tensor_values", {{"name", t.name}, {"row", static_cast<double>(incoming.loaded)}}, "values", t.name);
    }
    // One reply: whole rows from where the last ended, bfloat16 bits or float32.
    bool read_values(const Json& result) {
        if (string(field(result, "name")) != incoming.name || integer(field(result, "rows")) != incoming.rows ||
            integer(field(result, "cols")) != incoming.cols || integer(field(result, "row"), -1) != incoming.loaded) return false;
        const int count = integer(field(result, "count"));
        if (count < 1 || count > incoming.rows - incoming.loaded) return false;
        const std::string encoding = string(field(result, "encoding"));
        const size_t width = encoding == "bfloat16" ? 2 : encoding == "float32" ? 4 : 0;
        const size_t cells = static_cast<size_t>(count) * incoming.cols;
        std::vector<uint8_t> bytes;
        if (!width || !decode_base64(string(field(result, "data")), bytes) || bytes.size() != cells * width) return false;
        const uint64_t revision = static_cast<uint64_t>(integer(field(result, "revision")));
        if (incoming.loaded == 0) incoming.revision = revision;
        else if (revision != incoming.revision) { incoming.loaded = 0; return true; }   // edited meanwhile: start over
        float* out = incoming.data.data() + static_cast<size_t>(incoming.loaded) * incoming.cols;
        for (size_t i = 0; i < cells; ++i) {
            const uint8_t* b = bytes.data() + i * width;
            const uint32_t bits = width == 2 ? static_cast<uint32_t>(b[0] | b[1] << 8) << 16
                                             : static_cast<uint32_t>(b[0] | b[1] << 8 | b[2] << 16) | static_cast<uint32_t>(b[3]) << 24;
            float v; std::memcpy(&v, &bits, sizeof v);
            if (!std::isfinite(v)) return false;
            out[i] = v;
        }
        incoming.loaded += count;
        return true;
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
        if (!channel.connected() || (active.op != "generate" && active.op != "trace") || stop_requested) return;
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
        row = col = 0; have_value = false; cell_selected = false; focus_after_read = false;
        clear_samples();
        release_values(values); values = {}; incoming = {};
        values_needed = true; values_stale = false; values_failed.clear();
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
        } else { detail_needed = overview_needed = true; detail_due = 0; values_needed = true; incoming = {}; }
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
            if (active.op == "trace") {
                std::string why;
                if (!trace_pane.event(trace_run(active), field(reply, "event"), why)) { stop(); error = why; }
            } else if (active.op == "generate") {
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
            if (op == "tensor_values") {   // fall back to region requests for this tensor
                values_failed = completed.name; values_needed = false; incoming = {};
                detail_needed = overview_needed = true; detail_due = 0;
            }
            error = string(field(reply, "error"));
            if (op == "generate") { generation_status = "Generation failed"; stop_requested = false; }
            if (op == "trace") { trace_pane.failed(trace_run(completed), error); stop_requested = false; }
            if (op == "tokenize") { std::string ignored; trace_pane.tokenized(Json(), ignored); }
            return;
        }
        const Json& result = field(reply, "result");
        if (op == "generate") {
            metrics(result);
            generation_status = boolean(field(result, "cancelled")) ? "Stopped" : "Complete";
            stop_requested = false;
        } else if (op == "trace") {
            trace_pane.finished(trace_run(completed), result);
            stop_requested = false;
        } else if (op == "tokenize") {
            std::string why;
            if (!trace_pane.tokenized(result, why)) error = why;
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
        } else if (op == "tensor_values") {
            if (completed.name != selected || incoming.name != selected) return;
            if (!read_values(result)) { protocol_error("Invalid tensor values from script"); return; }
            if (incoming.complete()) {
                const auto range = std::minmax_element(incoming.data.begin(), incoming.data.end());
                incoming.minimum = *range.first; incoming.maximum = *range.second;
                release_values(values); values = std::move(incoming); incoming = {};
                values_needed = values_stale = false;
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
            if (focus_after_read) { focus_value = true; focus_after_read = false; }
        } else if (op == "edit_weight" || op == "revert_tensor") {
            update_edited(completed.name, result, op == "edit_weight");
            if (completed.name == selected) {
                have_value = false; read_edits(result);
                detail_needed = overview_needed = true; detail_due = 0;
                values_needed = values_stale = true; incoming = {};
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
            if (active.op == "tensor_values") { incoming = {}; values_needed = true; }
            if (active.op == "trace") trace_pane.failed(trace_run(active), "Connection ended");
            active = {};
            stop_requested = false;
            want_list = false; queued_op.clear(); queued_fields.clear(); queued_purpose.clear();
            retry_at = now + 1.0;
        }
        if (!idle()) return;
        const Tensor* shown = tensor();
        const bool whole = shown && uses_values(*shown);
        if (!whole) values_needed = false;
        if (!queued_op.empty()) {
            const std::string op = std::move(queued_op), purpose = std::move(queued_purpose);
            Object fields = std::move(queued_fields);
            queued_op.clear(); queued_fields.clear(); queued_purpose.clear();
            issue(op.c_str(), std::move(fields), purpose);
        } else if (want_list) { want_list = false; issue("list_tensors"); }
        else if (whole) { if (values_needed) request_values(*shown); }
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
        if (!idle() || !queued_op.empty() || view_waiting()) return;
        const std::string name = model3d.next_thumbnail();
        if (name.empty()) return;
        const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor& t) { return t.name == name; });
        if (it == tensors.end()) { model3d.thumbnail_failed(name); return; }
        const auto shape = model3d::thumbnail_shape(*it);
        const Geometry::Target size{shape[1], shape[0]};
        if (!request_sample(name, "thumbnail", {0, 0, it->rows, it->cols}, size)) model3d.thumbnail_failed(name);
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
        if (docked) return;   // the dockspace places the panes
        ImGui::SetNextWindowPos(ImVec2(x, y), force_layout ? ImGuiCond_Always : ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), force_layout ? ImGuiCond_Always : ImGuiCond_FirstUseEver);
    }
    const char* connection() const {
        if (!alive) return "session ended";
        if (!ready) return "connecting";
        if (active.op == "generate") return "generating";
        if (active.op == "trace") return "tracing";
        if (!active.op.empty() && !background()) return "working";   // thumbnails load quietly
        return "ready";
    }
    void inference(float x, float y, float w, float h) {
        place_window(kInferencePane, x, y, w, h);
        if (begin_pane(kInferencePane, "Inference", connection())) {
            if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
            field_label("Prompt");
            const float prompt_height = std::clamp(ImGui::GetContentRegionAvail().y * 0.22f, 72.0f, 200.0f);
            ImGui::InputTextMultiline("##prompt", prompt, sizeof prompt, ImVec2(-FLT_MIN, prompt_height));
            // Two label-field pairs per row, the labels in one column so the
            // fields line up (form layout, Bloomberg-style dense settings).
            const ImGuiStyle& style = ImGui::GetStyle();
            const float half = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;
            const float label_width = ImGui::CalcTextSize("Max new tokens").x + style.ItemSpacing.x;
            const float field_width = std::max(40.0f, half - label_width - style.ItemSpacing.x);
            const float second = ImGui::GetCursorPosX() + half + style.ItemSpacing.x;
            auto pair = [&](const char* label, float column) {
                if (column > 0) ImGui::SameLine(column);
                const float start = ImGui::GetCursorPosX();
                field_label(label);
                ImGui::SameLine(start + label_width);
                ImGui::SetNextItemWidth(field_width);
            };
            pair("Max new tokens", 0);
            ImGui::InputInt("##max_new_tokens", &max_new_tokens, 0, 0);
            max_new_tokens = std::clamp(max_new_tokens, 1, 1024);
            pair("Temperature", second);
            ImGui::InputFloat("##temperature", &temperature, 0, 0, "%.2f");
            temperature = std::isfinite(temperature) ? std::clamp(temperature, 0.0f, 2.0f) : 0.7f;
            pair("Top p", 0);
            ImGui::InputFloat("##top_p", &top_p, 0, 0, "%.2f");
            top_p = std::isfinite(top_p) ? std::clamp(top_p, 0.0f, 1.0f) : 0.9f;
            pair("Top k", second);
            ImGui::InputInt("##top_k", &top_k, 0, 0);
            top_k = std::clamp(top_k, 0, 1000);
            ImGui::Checkbox("Thinking", &thinking);
            ImGui::Spacing();
            ImGui::BeginDisabled(!can_control() || prompt[0] == '\0');
            if (sternbild::primary_button("Generate")) generate();
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(active.op != "generate" || stop_requested || !channel.connected());
            if (sternbild::word_button("Stop")) stop();
            ImGui::EndDisabled();
            char stats[64];
            std::snprintf(stats, sizeof stats, "%d tokens  %.1f tokens/s", tokens, tokens_per_second);
            right_text(stats);
            if (!generation_status.empty()) faint_text(generation_status.c_str());
            ImGui::Spacing();
            field_label("Output");
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
        place_window(kModelPane, x, y, w, h);
        double total_bytes = 0;
        std::set<std::string> dtypes;
        for (const auto& t : tensors) { total_bytes += t.bytes; dtypes.insert(t.dtype); }
        if (begin_pane(kModelPane, "Model", supports("generate") ? nullptr : connection())) {
            if (!supports("generate") && !error.empty()) ImGui::TextWrapped("%s", error.c_str());
            if (!model.empty()) {
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(model.c_str());
                char summary[96];
                std::snprintf(summary, sizeof summary, "%zu tensors  %s%s%s", tensors.size(), byte_size(total_bytes).c_str(),
                              dtypes.size() == 1 ? "  " : "", dtypes.size() == 1 ? dtypes.begin()->c_str() : "");
                right_text(summary);
            }
            const ImGuiStyle& style = ImGui::GetStyle();
            const float words = ImGui::CalcTextSize("Refresh").x + ImGui::CalcTextSize("Reset layout").x +
                                4 * style.FramePadding.x + 2 * style.ItemSpacing.x;
            ImGui::SetNextItemWidth(std::max(60.0f, ImGui::GetContentRegionAvail().x - words));
            ImGui::InputTextWithHint("##filter", "Filter tensors", filter, sizeof filter);
            ImGui::SameLine();
            ImGui::BeginDisabled(!can_control());
            if (sternbild::word_button("Refresh")) want_list = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (sternbild::word_button("Reset layout")) reset_layout = true;
            if (ImGui::BeginTable("tensors", 3, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit, ImVec2(0, 0))) {
                ImGui::TableSetupColumn("Tensor", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("[248320, 1024]").x);
                ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("485.0 MiB").x);
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
                    std::string kind;
                    for (const auto& t : tensors) if (t.group == group && !t.layer_type.empty()) { kind = layer_kind(t.layer_type); break; }
                    if (!kind.empty()) {
                        ImGui::SameLine(0, ImGui::GetStyle().ItemSpacing.x * 2);
                        ImGui::PushStyleColor(ImGuiCol_Text, kind == "full attention" ? sternbild::col::parchment : sternbild::col::faint);
                        ImGui::TextUnformatted(kind.c_str());
                        ImGui::PopStyleColor();
                    }
                    if (open) {
                        for (const auto& t : tensors) {
                            if (t.group != group || (filter[0] && t.name.find(filter) == std::string::npos)) continue;
                            ImGui::TableNextRow(); ImGui::TableNextColumn();
                            ImGui::PushID(t.name.c_str());
                            std::string label = t.name;
                            const auto prefix = label.find(group + ".");
                            if (prefix != std::string::npos) label.erase(0, prefix + group.size() + 1);
                            for (const char* common : {"language_model.model.", "model."})
                                if (label.rfind(common, 0) == 0) { label.erase(0, std::strlen(common)); break; }
                            ImGui::BeginDisabled(!can_control());
                            if (ImGui::Selectable(label.c_str(), selected == t.name, ImGuiSelectableFlags_SpanAllColumns)) select(t.name);
                            ImGui::EndDisabled();
                            if (scroll_selected && selected == t.name) { ImGui::SetScrollHereY(0.5f); scroll_selected = false; }
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%.0f values", t.name.c_str(), t.size);
                            ImGui::TableNextColumn(); ImGui::TextUnformatted(t.shape_label.c_str());
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
    void make_values_texture(Values& v, bool vector) {
        if (!renderer || !v.complete() || !v.texture_dirty) return;
        release_values(v);
        const int width = vector ? v.rows : v.cols, height = vector ? v.cols : v.rows;
        std::vector<uint8_t> rgba(v.data.size() * 4);
        float lo = static_cast<float>(v.minimum), hi = static_cast<float>(v.maximum);
        if (lo == hi) { lo -= 0.5f; hi += 0.5f; }
        caliper_host::map_f32_to_rgba8(v.data.data(), width, height,
            caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS), lo, hi, rgba.data());
        v.texture = renderer->tex_create_rgba8_mipmapped(width, height);
        if (v.texture && renderer->tex_upload_rgba8(v.texture, rgba.data(), width, height)) renderer->tex_generate_mipmaps(v.texture);
        else release_values(v);
        v.texture_dirty = false;
    }
    static ImU32 heat_color(double value, double minimum, double maximum) {
        const double span = maximum - minimum;
        const double amount = span > 0 ? (value - minimum) / span : 0.5;
        const int index = std::clamp(static_cast<int>(amount * 255 + 0.5), 0, 255);
        return caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS)[index];
    }
    // A cell's value written inside it when it fits, in the ink that reads on its colour.
    static void cell_text(ImDrawList* list, ImVec2 a, ImVec2 b, double value, ImU32 fill) {
        char text[32];
        std::snprintf(text, sizeof text, "%.3g", value);
        const ImVec2 size = ImGui::CalcTextSize(text);
        if (b.x - a.x < size.x + 6 || b.y - a.y < size.y + 4) return;
        const ImVec4 rgb = ImGui::ColorConvertU32ToFloat4(fill);
        const bool light = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z > 0.5f;
        list->AddText(point((a.x + b.x - size.x) * 0.5, (a.y + b.y - size.y) * 0.5),
                      light ? sternbild::col::deep : sternbild::col::parchment, text);
    }
    void paint_image(Sample& sample, Geometry::Window shown, Geometry::Rect picture, bool vector, bool crisp = false) {
        if (picture.width <= 0 || picture.height <= 0 || sample.values.empty() || !(shown.rows > 0) || !(shown.cols > 0) ||
            sample.region.rows <= 0 || sample.region.cols <= 0) return;
        ImDrawList* list = ImGui::GetWindowDrawList();
        make_texture(sample, vector);
        // Tensor coordinates to the picture: rows run down a matrix and along a vector's strip.
        const auto at = [&](double r, double c) {
            return point(picture.x + (vector ? (r - shown.row) / shown.rows : (c - shown.col) / shown.cols) * picture.width,
                         picture.y + (vector ? (c - shown.col) / shown.cols : (r - shown.row) / shown.rows) * picture.height);
        };
        const auto& region = sample.region;
        list->PushClipRect(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height), true);
        if (sample.texture && renderer && !crisp) {
            // The sample's own region, placed where it lies in the shown window:
            // a sample from before a zoom or pan stays in place, shrunk or
            // enlarged, until the next one arrives.
            list->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(sample.texture)),
                           at(region.row, region.col), at(region.row + region.rows, region.col + region.cols));
        } else {
            // The CPU fixture takes the same rectangles and colormap as the
            // texture path; a normal app frame draws a single textured quad.
            for (int r = 0; r < sample.rows; ++r) for (int c = 0; c < sample.cols; ++c) {
                const double row0 = region.row + static_cast<double>(r) * region.rows / sample.rows;
                const double col0 = region.col + static_cast<double>(c) * region.cols / sample.cols;
                const double row1 = region.row + static_cast<double>(r + 1) * region.rows / sample.rows;
                const double col1 = region.col + static_cast<double>(c + 1) * region.cols / sample.cols;
                const ImVec2 a = at(row0, col0), b = at(row1, col1);
                const double cell_value = sample.values[r * sample.cols + c];
                const ImU32 fill = heat_color(cell_value, sample.minimum, sample.maximum);
                list->AddRectFilled(a, b, fill);
                // Zoomed in far enough, cells get a grid and then their values,
                // like a spreadsheet, so a zoom shows as cells growing.
                if (crisp && b.x - a.x >= 12 && b.y - a.y >= 12) list->AddRect(a, b, sternbild::with_alpha(sternbild::col::deep, 160));
                if (crisp) cell_text(list, a, b, cell_value, fill);
            }
        }
        list->PopClipRect();
        list->AddRect(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height),
                      ImGui::GetColorU32(ImGuiCol_Border));
    }
    // The whole tensor as one mipmapped texture, the shown window picked out by
    // texture coordinates, so zoom and pan only move coordinates. Drawn smaller
    // than its cells, the mipmaps average the cells under each pixel; larger,
    // cells are sampled nearest so they stay sharp squares, then get a grid
    // and their values.
    void paint_values(Values& v, Geometry::Window shown, Geometry::Rect picture, bool vector) {
        if (picture.width <= 0 || picture.height <= 0 || !v.complete() || !(shown.rows > 0) || !(shown.cols > 0)) return;
        ImDrawList* list = ImGui::GetWindowDrawList();
        make_values_texture(v, vector);
        const ImVec2 a = point(picture.x, picture.y), b = point(picture.x + picture.width, picture.y + picture.height);
        const double left = vector ? shown.row : shown.col, top = vector ? shown.col : shown.row;
        const double cell = std::min(picture.width / (vector ? shown.rows : shown.cols),
                                     picture.height / (vector ? shown.cols : shown.rows));
        list->PushClipRect(a, b, true);
        const bool textured = v.texture && renderer;
        if (textured) {
            const double rows = v.rows, cols = v.cols;
            const ImVec2 uv0 = vector ? point(shown.row / rows, 0) : point(shown.col / cols, shown.row / rows);
            const ImVec2 uv1 = vector ? point((shown.row + shown.rows) / rows, 1)
                                      : point((shown.col + shown.cols) / cols, (shown.row + shown.rows) / rows);
            const auto& platform = ImGui::GetPlatformIO();
            const bool sharp = cell * std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x) >= 2 &&
                               platform.DrawCallback_SetSamplerNearest && platform.DrawCallback_SetSamplerLinear;
            if (sharp) list->AddCallback(platform.DrawCallback_SetSamplerNearest, nullptr);
            list->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(v.texture)), a, b, uv0, uv1);
            if (sharp) list->AddCallback(platform.DrawCallback_SetSamplerLinear, nullptr);
        }
        // Whole cells in view: squares without a renderer (the CPU fixture),
        // and a grid and values over the texture once they are large.
        const Geometry::Region cells = Geometry::cover(shown, v.rows, v.cols);
        const bool squares = !textured && cells.rows * cells.cols <= 65536;
        if (squares || cell >= 12) {
            for (int64_t r = cells.row; r < cells.row + cells.rows; ++r) for (int64_t c = cells.col; c < cells.col + cells.cols; ++c) {
                const ImVec2 p0 = point(picture.x + ((vector ? r : c) - left) * cell, picture.y + ((vector ? c : r) - top) * cell);
                const ImVec2 p1 = point(p0.x + cell, p0.y + cell);
                const double value = v.data[static_cast<size_t>(r) * v.cols + c];
                const ImU32 fill = heat_color(value, v.minimum, v.maximum);
                if (squares) list->AddRectFilled(p0, p1, fill);
                if (cell >= 12) {
                    list->AddRect(p0, p1, sternbild::with_alpha(sternbild::col::deep, 160));
                    cell_text(list, p0, p1, value, fill);
                }
            }
        }
        list->PopClipRect();
        list->AddRect(a, b, ImGui::GetColorU32(ImGuiCol_Border));
    }
    void color_scale(double minimum, double maximum, bool known, Geometry::Rect area) {
        if (!known || area.height < 40) return;
        auto* list = ImGui::GetWindowDrawList();
        const double height = std::max(30.0, area.height - 34);
        const auto* lut = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
        for (int i = 0; i < 128; ++i) {
            const double y0 = area.y + i * height / 128, y1 = area.y + (i + 1) * height / 128;
            list->AddRectFilled(point(area.x, y0), point(area.x + 12, y1), lut[255 - i * 2]);
        }
        char high[32], low[32];
        std::snprintf(high, sizeof high, "%.5g", maximum);
        std::snprintf(low, sizeof low, "%.5g", minimum);
        list->AddText(point(area.x + 16, area.y), ImGui::GetColorU32(ImGuiCol_Text), high);
        list->AddText(point(area.x + 16, area.y + height - ImGui::GetTextLineHeight()), ImGui::GetColorU32(ImGuiCol_Text), low);
        list->AddText(point(area.x, area.y + height + 4), ImGui::GetColorU32(ImGuiCol_Text), "Value");
    }
    // The whole cells in view, for a region request after the view settles.
    void request_detail(Geometry::Region next, Geometry::Target size) {
        const Tensor* t = tensor(); if (!t) return;
        next = Geometry::clamp(next, t->rows, t->cols);
        const bool changed = region != next || target.width != size.width || target.height != size.height;
        region = next; target = size;
        if (supports("tensor_regions") && !uses_values(*t) &&
            (last_region != region || last_target.width != size.width || last_target.height != size.height)) {
            if (!detail_needed || changed) detail_due = detail.values.empty() ? 0 : seconds() + 0.05;
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
            ImGui::SetTooltip("(%lld, %lld)\nClick to select, double-click to zoom in", static_cast<long long>(cell.row), static_cast<long long>(cell.col));
        }
    }
    void show_value_hover(const Values& v, Geometry::Cell cell) {
        if (!cell.valid || cell.row >= v.rows || cell.col >= v.cols) return;
        ImGui::SetTooltip("(%lld, %lld)\n%.9g", static_cast<long long>(cell.row), static_cast<long long>(cell.col),
                          v.data[static_cast<size_t>(cell.row) * v.cols + cell.col]);
    }
    void tensor_picture(const Tensor& t, Geometry::Rect allocated, const Geometry::Fold& map) {
        const bool vector = t.vector();
        const auto layout = Geometry::view_layout(t.rows, t.cols, vector, allocated);
        const Geometry::Rect bounds = vector ? layout.strip : layout.detail;
        view_bounds = bounds;
        if (!region_ready) {
            camera = Geometry::initial_camera(t.rows, t.cols, vector, bounds);
            region_ready = true; detail_needed = true; detail_due = 0;
        }
        camera = Geometry::clamp(camera, t.rows, t.cols, vector, bounds);
        const auto shot = Geometry::shoot(camera, t.rows, t.cols, vector, bounds);
        window = shot.window;
        const Geometry::Rect picture = shot.picture;
        const Geometry::Region cells = Geometry::cover(window, t.rows, t.cols);
        const bool whole_values = uses_values(t), shown_values = values_shown();
        const float scale = std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x);
        auto size = Geometry::sampling_target(cells, picture, scale, vector);
        if (vector) std::swap(size.width, size.height);
        request_detail(cells, size);
        // Without whole values, an averaged sample of the whole tensor serves
        // the map beside the picture and the vector plot.
        const Geometry::Region whole_cells{0, 0, t.rows, t.cols};
        if (!whole_values && supports("tensor_regions") && (vector || map.scale > 0)) {
            auto overview_size = vector
                ? Geometry::sampling_target(whole_cells, {layout.plot.x, layout.plot.y, layout.plot.width, 1}, scale, true)
                : Geometry::fold_target(map, t.rows, t.cols, scale);
            if (vector) std::swap(overview_size.width, overview_size.height);
            if (overview_size.width != overview_target.width || overview_size.height != overview_target.height) {
                overview_target = overview_size; overview_needed = true;
            }
        } else overview_needed = false;
        if (!supports("tensor_regions")) {
            overview_needed = false;
            if (detail.name == selected && !detail.values.empty()) detail_needed = false;
        }
        if (vector && layout.plot.width > 0 && layout.plot.height > 0) {
            ImGui::SetCursorScreenPos(point(layout.plot.x, layout.plot.y));
            if (ImPlot::BeginPlot("##vector.values", point(layout.plot.width, layout.plot.height), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_Lock, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, t.rows, ImPlotCond_Always);
                if (shown_values) ImPlot::PlotLine("Value", values.data.data(), values.rows);
                else {
                    const Sample& line = overview.values.empty() ? detail : overview;
                    if (!line.values.empty()) {
                        std::vector<double> indices(line.row_indices.begin(), line.row_indices.end());
                        ImPlot::PlotLine("Value", indices.data(), line.values.data(), static_cast<int>(indices.size()));
                    }
                }
                const ImVec2 pos = ImPlot::GetPlotPos(), extent = ImPlot::GetPlotSize();
                const auto band = Geometry::selection_rect(Geometry::Window{window.col, window.row, window.cols, window.rows},
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
        auto* list = ImGui::GetWindowDrawList();
        const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
        const double cell_px = camera.cell;
        if (shown_values) paint_values(values, window, picture, vector);
        else if (whole_values) {
            list->AddRectFilled(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height), sternbild::col::plate);
            char loading[48];
            std::snprintf(loading, sizeof loading, "Loading values %d%%", incoming.rows ? incoming.loaded * 100 / incoming.rows : 0);
            const ImVec2 extent = ImGui::CalcTextSize(loading);
            list->AddText(point(picture.x + (picture.width - extent.x) * 0.5, picture.y + (picture.height - extent.y) * 0.5),
                          sternbild::col::faint, loading);
        } else {
            // While a new region is on its way, the averaged whole-tensor overview
            // fills whatever the last detail sample does not cover. Cells of six
            // points or more are crisp squares rather than a smoothed texture.
            if (!overview.values.empty() && overview.name == t.name && detail.region != region)
                paint_image(overview, window, picture, vector);
            paint_image(detail, window, picture, vector, detail.exact && cell_px >= 6 && detail.region == region);
        }
        // The selected cell: its row and column shaded, the cell outlined in
        // bronze (at least five points, so it shows zoomed out too).
        const bool row_in = row + 1 > window.row && row < window.row + window.rows;
        const bool col_in = vector || (col + 1 > window.col && col < window.col + window.cols);
        if (cell_selected && (row_in || col_in)) {
            list->PushClipRect(point(picture.x, picture.y), point(picture.x + picture.width, picture.y + picture.height), true);
            const ImU32 shade = sternbild::with_alpha(sternbild::col::bronze, 40);
            if (vector) {
                const double x0 = picture.x + (row - window.row) * cell_px, cx = x0 + cell_px * 0.5, hw = std::max(2.5, cell_px * 0.5);
                if (row_in) list->AddRect(point(cx - hw, picture.y), point(cx + hw, picture.y + picture.height), sternbild::col::bronze, 0, 0, 2.0f);
            } else {
                const double y0 = picture.y + (row - window.row) * cell_px, y1 = y0 + cell_px;
                const double x0 = picture.x + (col - window.col) * cell_px, x1 = x0 + cell_px;
                if (row_in) list->AddRectFilled(point(picture.x, y0), point(picture.x + picture.width, y1), shade);
                if (col_in) list->AddRectFilled(point(x0, picture.y), point(x1, picture.y + picture.height), shade);
                if (row_in && col_in) {
                    const double cx = (x0 + x1) * 0.5, cy = (y0 + y1) * 0.5, half = std::max(2.5, cell_px * 0.5);
                    list->AddRect(point(cx - half, cy - half), point(cx + half, cy + half), sternbild::col::bronze, 0, 0, 2.0f);
                }
            }
            list->PopClipRect();
        }
        char columns[100];
        if (vector) {
            std::snprintf(columns, sizeof columns, "Index %lld to %lld", static_cast<long long>(cells.row), static_cast<long long>(cells.row + cells.rows - 1));
            list->AddText(point(picture.x, bounds.y + bounds.height + 3), text, columns);
        } else {
            char first_row[64], last_row[64];
            std::snprintf(first_row, sizeof first_row, "Row %lld", static_cast<long long>(cells.row));
            std::snprintf(last_row, sizeof last_row, "%lld", static_cast<long long>(cells.row + cells.rows - 1));
            std::snprintf(columns, sizeof columns, "Column %lld to %lld", static_cast<long long>(cells.col), static_cast<long long>(cells.col + cells.cols - 1));
            const auto labels = Geometry::axis_labels(picture, ImGui::CalcTextSize(first_row).x,
                ImGui::CalcTextSize(last_row).x, ImGui::CalcTextSize(columns).x, ImGui::GetTextLineHeight());
            list->AddText(point(labels.first_row.x, labels.first_row.y), text, first_row);
            if (cells.rows > 1 && labels.last_row.height > 0)
                list->AddText(point(labels.last_row.x, labels.last_row.y), text, last_row);
            list->AddText(point(labels.columns.x, labels.columns.y), text, columns);
        }
        const auto mouse = ImGui::GetIO().MousePos;
        if (map.scale > 0) tensor_map(t, map, bounds);
        // A tall matrix's rows run far past any pane: a scrollbar along the
        // picture spans all of them. Drag the thumb, click the track to jump,
        // or turn the wheel over it to scroll a quarter view at a time.
        if (layout.tall && layout.scroll.width > 0 && layout.scroll.height > 0) {
            const auto& track = layout.scroll;
            const auto thumb = Geometry::scroll_thumb(track, window, t.rows);
            ImGui::SetCursorScreenPos(point(track.x, track.y));
            ImGui::InvisibleButton("##rows.scroll", point(track.width, track.height));
            const bool on_thumb = mouse.y >= thumb.y && mouse.y < thumb.y + thumb.height;
            if (ImGui::IsItemActivated()) scroll_grab = on_thumb ? mouse.y - (thumb.y + thumb.height * 0.5) : 0;
            if (ImGui::IsItemActive()) camera.y = Geometry::scroll_centre(track, window, t.rows, mouse.y - scroll_grab);
            if (ImGui::IsItemHovered() && ImGui::GetIO().MouseWheel != 0) camera.y -= ImGui::GetIO().MouseWheel * window.rows * 0.25;
            if (ImGui::IsItemHovered() && !ImGui::IsItemActive()) {
                const auto at = static_cast<int64_t>(std::clamp((mouse.y - track.y) / track.height, 0.0, 1.0) * (t.rows - 1));
                ImGui::SetTooltip("Row %s of %s", grouped(at).c_str(), grouped(t.rows).c_str());
            }
            const bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
            const float round = static_cast<float>(track.width * 0.5);
            list->AddRectFilled(point(track.x, track.y), point(track.x + track.width, track.y + track.height), sternbild::col::plate, round);
            list->AddRectFilled(point(thumb.x + 2, thumb.y), point(thumb.x + thumb.width - 2, thumb.y + thumb.height),
                                hot ? sternbild::col::bronze : sternbild::col::secondary, round);
        }
        ImGui::SetCursorScreenPos(point(bounds.x, bounds.y));
        ImGui::InvisibleButton("##tensor.picture", point(std::max(1.0, bounds.width), std::max(1.0, bounds.height)),
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        auto cell = Geometry::hit_test(window, picture, mouse.x, mouse.y, vector);
        if (cell.valid) { cell.row = std::min<int64_t>(cell.row, t.rows - 1); cell.col = std::min<int64_t>(cell.col, t.cols - 1); }
        const bool picture_hovered = ImGui::IsItemHovered();
        // The wheel zooms continuously about the pointer; every view comes
        // from the same camera, so nothing waits on whole cells or the host.
        if (picture_hovered && ImGui::GetIO().MouseWheel != 0)
            camera = Geometry::zoom(camera, t.rows, t.cols, vector, bounds, std::pow(1.25, ImGui::GetIO().MouseWheel), mouse.x, mouse.y);
        if (picture_hovered && cell.valid) {
            if (shown_values) show_value_hover(values, cell); else show_hover(detail, cell);
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                // Zoom until cells are large enough to show their values.
                camera.cell = std::max(camera.cell, 60.0);
                camera = Geometry::centre_on(camera, t.rows, t.cols, vector, bounds, cell);
            }
        }
        // A press then release on the same cell without a drag selects it; a
        // drag (four points, or the middle button) pans.
        if (ImGui::IsItemActivated()) { press_cell = cell; drag_camera = camera; drag_origin = mouse; dragging = false; }
        if (ImGui::IsItemActive() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0))) {
            dragging = true;
            camera = Geometry::pan(drag_camera, t.rows, t.cols, vector, bounds, mouse.x - drag_origin.x, mouse.y - drag_origin.y);
        }
        if (ImGui::IsItemDeactivated()) {
            if (!dragging && press_cell.valid && cell.valid && cell.row == press_cell.row && cell.col == press_cell.col) select_cell(cell);
            dragging = false;
        }
        if (picture_hovered) ImGui::SetMouseCursor(dragging ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Hand);
    }
    // The whole tensor beside the picture with the view framed in bronze. A
    // tall one folds into the 3D view's equal row blocks, read left to right
    // as one column. Click or drag in it to move the view.
    void tensor_map(const Tensor& t, const Geometry::Fold& map, Geometry::Rect bounds) {
        auto* list = ImGui::GetWindowDrawList();
        const auto mouse = ImGui::GetIO().MousePos;
        const bool sampled = !overview.values.empty() && overview.name == t.name;
        if (values_shown()) make_values_texture(values, false);
        else if (sampled) make_texture(overview, false);
        const uint64_t texture = values_shown() ? values.texture : sampled ? overview.texture : 0;
        for (int b = 0; b < map.blocks; ++b) {
            const auto block = Geometry::fold_block(map, t.cols, b);
            const ImVec2 a = point(block.x, block.y), z = point(block.x + block.width, block.y + block.height);
            // Texture rows are the tensor's rows evenly binned, so a block's
            // share of them is its share of the texture.
            const double v0 = static_cast<double>(b) * map.block_rows / t.rows;
            const double v1 = std::min(1.0, static_cast<double>(b + 1) * map.block_rows / t.rows);
            if (texture && renderer)
                list->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(texture)), a, z, point(0, v0), point(1, v1));
            else list->AddRectFilled(a, z, sternbild::col::plate);
            list->AddRect(a, z, sternbild::col::rule);
        }
        for (const auto& part : Geometry::fold_selection(map, window, t.rows, t.cols)) {
            const double cx = part.x + part.width * 0.5, cy = part.y + part.height * 0.5;
            const double hw = std::max(2.0, part.width * 0.5), hh = std::max(2.0, part.height * 0.5);
            list->AddRect(point(cx - hw, cy - hh), point(cx + hw, cy + hh), sternbild::col::bronze, 0, 0, 1.5f);
        }
        if (map.blocks > 1) {
            char caption[96];
            std::snprintf(caption, sizeof caption, "%d columns of %s rows", map.blocks, grouped(map.block_rows).c_str());
            list->AddText(point(map.area.x, map.area.y + map.area.height + 3), sternbild::col::faint, caption);
        }
        ImGui::SetCursorScreenPos(point(map.area.x, map.area.y));
        ImGui::InvisibleButton("##overview", point(std::max(1.0, map.area.width), std::max(1.0, map.area.height)));
        const auto picked = Geometry::fold_hit(map, t.rows, t.cols, mouse.x, mouse.y);
        if (ImGui::IsItemHovered() && !ImGui::IsItemActive() && picked.valid) {
            if (map.blocks > 1) {
                const int64_t b = picked.row / map.block_rows;
                ImGui::SetTooltip("Rows %s to %s, column %lld of %d\nClick or drag to move the view",
                    grouped(b * map.block_rows).c_str(), grouped(std::min<int64_t>(t.rows, (b + 1) * map.block_rows) - 1).c_str(),
                    static_cast<long long>(b + 1), map.blocks);
            } else ImGui::SetTooltip("Whole tensor: click or drag to move the view");
        }
        if (ImGui::IsItemActive()) camera = Geometry::centre_on(camera, t.rows, t.cols, false, bounds, picked);
    }
    void select_cell(Geometry::Cell cell) {
        row = static_cast<int>(cell.row); col = static_cast<int>(cell.col);
        cell_selected = true; have_value = false;
        if (values_shown() && !values_stale) {   // every value is here already
            value_row = row; value_col = col;
            read_value = value = values.data[static_cast<size_t>(row) * values.cols + col];
            have_value = true; focus_value = true;
            return;
        }
        if (supports("read_weight") && send("read_weight", {{"name", selected}, {"row", static_cast<double>(row)}, {"col", static_cast<double>(col)}}))
            focus_after_read = true;
    }
    // The Trace pane shares Model 3D's place in the dock; each has a switch to the other.
    void trace_view(float x, float y, float w, float h) {
        place_window(kTracePane, x, y, w, h);
        if (begin_pane(kTracePane, "Trace", "Jacobian lens")) {
            if (sternbild::word_button("Model 3D")) ImGui::SetWindowFocus(kModel3DPane);
            const auto ask = trace_pane.draw(can_control() && queued_op.empty(), active.op == "trace");
            if (ask.op == "stop") stop();
            else if (ask.op == "trace") {
                if (send("trace", ask.fields, ask.run == 1 ? "intervened" : "intact")) trace_pane.started(ask.run);
            } else if (ask.op == "tokenize" && !send("tokenize", ask.fields)) {
                std::string ignored; trace_pane.tokenized(Json(), ignored);
            }
        }
        ImGui::End();
    }
    void tensor_view(float x, float y, float w, float h) {
        place_window(kTensorPane, x, y, w, h);
        const Tensor* t = tensor();
        std::string subtitle = t ? t->name : std::string("none selected");
        for (const char* prefix : {"language_model.model.", "model."})
            if (subtitle.rfind(prefix, 0) == 0) { subtitle.erase(0, std::strlen(prefix)); break; }
        if (begin_pane(kTensorPane, "Tensor", subtitle.c_str())) {
            if (!t) faint_text("Select a tensor in Model or Model 3D.");
            else tensor_body(*t);
        }
        ImGui::End();
    }
    void tensor_body(const Tensor& t) {
        const ImGuiStyle& style = ImGui::GetStyle();
        char facts[160];
        std::snprintf(facts, sizeof facts, "%s  %s  %s", t.shape_label.c_str(), t.dtype.c_str(), byte_size(t.bytes).c_str());
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(facts);
        if (t.tied) { ImGui::SameLine(0, style.ItemSpacing.x * 3); faint_text("Shared with the output layer"); }
        if (!supports("tensor_regions")) faint_text("Overview only: restart the host to enable region sampling.");

        ImGui::BeginDisabled(!can_control());
        if (sternbild::word_button("Fit tensor")) {
            if (t.vector()) region_ready = false;
            else camera = {};   // the smallest cell: the whole tensor
        }
        ImGui::SameLine();
        if (sternbild::word_button("Cells")) {
            camera.cell = std::max(camera.cell, 60.0);
            camera = Geometry::centre_on(camera, t.rows, t.cols, t.vector(), view_bounds, {row, col, true});
        }
        ImGui::EndDisabled();
        char sampled[200];
        const double magnification = camera.cell / Geometry::min_cell(t.rows, t.cols, t.vector(), view_bounds);
        if (uses_values(t)) {
            if (values_shown()) std::snprintf(sampled, sizeof sampled, "Zoom %.3gx%s", magnification, values_stale ? "   Updating" : "");
            else std::snprintf(sampled, sizeof sampled, "Loading values");
        } else if (!detail.values.empty())
            std::snprintf(sampled, sizeof sampled, "Zoom %.3gx   Region: %d x %d sampled values%s", magnification,
                          detail.rows, detail.cols, detail.exact ? " (exact cells)" : "");
        else std::snprintf(sampled, sizeof sampled, "Loading region");
        right_text(sampled, sternbild::col::faint);

        // The picture takes every line the edit bar below does not need; the
        // row labels sit in the margin on its left. A matrix's map and the
        // colour scale stack in a column on its right, so neither covers it.
        const float row_height = ImGui::GetFrameHeightWithSpacing();
        const float edit_height = 4 * row_height + style.ItemSpacing.y * 2 + 1;
        const ImVec2 available = ImGui::GetContentRegionAvail();
        const float picture_height = std::max(140.0f, available.y - edit_height);
        const ImVec2 start = ImGui::GetCursorScreenPos();
        const double line = ImGui::GetTextLineHeight();
        const double label_room = t.vector() ? 0 : ImGui::CalcTextSize("Row 248320").x + style.ItemSpacing.x;
        const double scale_room = ImGui::CalcTextSize("-0.0000000").x + 24;
        const bool has_map = !t.vector();
        const double side = has_map ? std::max(scale_room, 168.0) : scale_room;
        const Geometry::Rect image_area{start.x + label_room, start.y + line + 4,
            std::max(20.0, available.x - label_room - side - style.ItemSpacing.x),
            std::max(20.0, picture_height - 2 * line - 12)};
        const double side_x = start.x + available.x - side + style.ItemSpacing.x, side_width = side - style.ItemSpacing.x;
        double side_y = start.y + line + 4;
        Geometry::Fold map{};
        if (has_map) {
            const int blocks = model3d::row_blocks(t);
            map = Geometry::fold(t.rows, t.cols, blocks,
                {side_x, side_y, side_width, std::min(blocks > 1 ? 200.0 : 120.0, image_area.height * 0.45)});
            side_y += map.area.height + line * (blocks > 1 ? 2.5 : 1);
        }
        tensor_picture(t, image_area, map);
        const bool known = values_shown() || !detail.values.empty();
        color_scale(values_shown() ? values.minimum : detail.minimum, values_shown() ? values.maximum : detail.maximum, known,
                    {side_x, side_y, side_width, start.y + picture_height - 4 - side_y});
        ImGui::SetCursorScreenPos(start);
        ImGui::Dummy(ImVec2(available.x, picture_height));

        // Edit bar: fixed-width fields with their labels before them.
        ImGui::Separator();
        const float number_width = ImGui::CalcTextSize("00000000").x + 2 * style.FramePadding.x;
        const float value_width = ImGui::CalcTextSize("-0.000000000").x + 2 * style.FramePadding.x;
        const float group_gap = style.ItemSpacing.x * 4;
        ImGui::BeginDisabled(!can_control());
        field_label("Row"); ImGui::SameLine(); ImGui::SetNextItemWidth(number_width);
        // Enter in Row or Col goes to that cell: the view centres on it and selects it.
        bool go = ImGui::InputInt("##row", &row, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::IsItemEdited()) have_value = false;
        ImGui::SameLine(); field_label("Col"); ImGui::SameLine(); ImGui::SetNextItemWidth(number_width);
        go |= ImGui::InputInt("##col", &col, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::IsItemEdited()) have_value = false;
        row = std::clamp(row, 0, static_cast<int>(t.rows - 1)); col = std::clamp(col, 0, static_cast<int>(t.cols - 1));
        if (go) {
            select_cell({row, col, true});
            camera = Geometry::centre_on(camera, t.rows, t.cols, t.vector(), view_bounds, {row, col, true});
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!supports("read_weight"));
        if (sternbild::word_button("Read")) send("read_weight", {{"name", selected}, {"row", static_cast<double>(row)}, {"col", static_cast<double>(col)}});
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        if (have_value) ImGui::Text("(%d, %d) = %.9g", value_row, value_col, read_value);
        else faint_text("Value not read");

        // After a cell is selected and read, the Value field takes the keyboard:
        // type the new value and press Enter to set it.
        field_label("Value"); ImGui::SameLine(); ImGui::SetNextItemWidth(value_width);
        if (focus_value) { ImGui::SetKeyboardFocusHere(); focus_value = false; }
        const bool entered = ImGui::InputDouble("##value", &value, 0, 0, "%.9g", ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        ImGui::BeginDisabled(!supports("edit_weight") || !std::isfinite(value));
        const bool set_pressed = sternbild::word_button("Set");
        ImGui::EndDisabled();
        if ((set_pressed || entered) && supports("edit_weight") && std::isfinite(value)) edit("set", value);
        ImGui::SameLine(0, group_gap); field_label("Scale"); ImGui::SameLine(); ImGui::SetNextItemWidth(number_width);
        ImGui::InputDouble("##scale", &factor, 0, 0, "%.6g");
        ImGui::SameLine();
        ImGui::BeginDisabled(!supports("edit_weight") || !std::isfinite(factor));
        if (sternbild::word_button("Scale row")) edit("scale_row", factor);
        ImGui::SameLine();
        if (sternbild::word_button("Scale column")) edit("scale_column", factor);
        ImGui::EndDisabled();

        ImGui::BeginDisabled(!supports("edit_weight"));
        if (sternbild::word_button("Zero row")) edit("zero_row", 0);
        ImGui::SameLine();
        if (sternbild::word_button("Zero column")) edit("zero_column", 0);
        ImGui::EndDisabled();
        ImGui::SameLine(0, group_gap);
        ImGui::BeginDisabled(!supports("revert_tensor"));
        if (sternbild::word_button("Revert tensor")) send("revert_tensor", {{"name", selected}});
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (active.op == "generate") { ImGui::SameLine(0, group_gap); faint_text("Stop generation to edit weights."); }

        field_label("Edits"); ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        if (edits.empty()) faint_text("No edits");
        else {
            ImGui::Text("%zu", edits.size());
            ImGui::SameLine();
            ImGui::TextUnformatted(edits.back().c_str());
            if (edits.size() > 1 && ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                for (const auto& line_text : edits) ImGui::TextUnformatted(line_text.c_str());
                ImGui::EndTooltip();
            }
        }
    }
    // The panes share one dockspace over the live page's work area, as the
    // landing screen's do: Model left, Model 3D over Tensor in the centre,
    // Inference right, with no tab bars, and splitters that keep the user's
    // proportions between runs. The tiling is built on first use, when a pane
    // has never been docked here, and on Reset layout.
    void dock_host(float x, float y, float width, float height) {
        ImGui::SetNextWindowPos(ImVec2(x, y));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::Begin("##caliper.inspector.host", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar(3);
        const ImGuiID dock = ImGui::GetID("caliper.inspector.dockspace");
        if (!layout_built || reset_layout) {
            bool rebuild = reset_layout || !ImGui::DockBuilderGetNode(dock) || ImGui::DockBuilderGetNode(dock)->IsEmpty();
            for (const char* pane : {kModelPane, kModel3DPane, kTensorPane, kInferencePane}) {
                const ImGuiWindowSettings* settings = ImGui::FindWindowSettingsByID(ImHashStr(pane));
                if (!settings || settings->DockId == 0) rebuild = true;
            }
            if (rebuild) {
                ImGui::DockBuilderRemoveNode(dock);
                ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dock, ImVec2(width, height));
                ImGuiID left = 0, rest = 0, right = 0, centre = 0, top = 0, bottom = 0;
                ImGui::DockBuilderSplitNode(dock, ImGuiDir_Left, 0.20f, &left, &rest);
                ImGui::DockBuilderSplitNode(rest, ImGuiDir_Right, 0.30f, &right, &centre);
                ImGui::DockBuilderSplitNode(centre, ImGuiDir_Up, 0.50f, &top, &bottom);
                ImGui::DockBuilderDockWindow(kModelPane, left);
                ImGui::DockBuilderDockWindow(kTracePane, top);
                ImGui::DockBuilderDockWindow(kModel3DPane, top);
                ImGui::DockBuilderDockWindow(kTensorPane, bottom);
                ImGui::DockBuilderDockWindow(kInferencePane, right);
                ImGui::DockBuilderFinish(dock);
            }
            layout_built = true; reset_layout = false;
        }
        ImGui::DockSpace(dock, ImVec2(0, 0), ImGuiDockNodeFlags_NoUndocking | ImGuiDockNodeFlags_NoWindowMenuButton |
                                             ImGuiDockNodeFlags_NoTabBar);
        ImGui::End();
    }
    void draw(float x, float y, float width, float height, caliper_host::HostRenderer* current_renderer) {
        if (!visible) return;
        if (renderer != current_renderer) {
            clear_samples(); release_values(values); renderer = current_renderer; detail_needed = overview_needed = true;
        }
        docked = (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable) != 0;
        if (docked) dock_host(x, y, width, height);
        force_layout = !docked && reset_layout;
        if (!docked) reset_layout = false;
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
            model3d.offer_trace = supports("trace");
            const std::string clicked = model3d.draw(renderer, r.x, r.y, r.width, r.height, selected);
            if (model3d.trace_clicked) { model3d.trace_clicked = false; ImGui::SetWindowFocus(kTracePane); }
            if (supports("trace"))
                draw_window([&](float a,float b,float c,float d) { trace_view(a,b,c,d); }, windows.model3d);
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
