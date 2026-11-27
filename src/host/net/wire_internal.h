// wire_internal.h: shared helpers and internal codec entry points across the
// wire protocol v3 translation units (wire.cpp, wire_controls.cpp,
// wire_scene.cpp, wire_timelines.cpp, wire_programs.cpp).
#pragma once

#include "net/wire.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace caliper_net {
namespace wire {
namespace detail {

struct MsgError {
    std::string code;
    std::string message;
};

[[noreturn]] inline void bad(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_BAD_REQUEST, msg}; }
[[noreturn]] inline void unsupported(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_UNSUPPORTED, msg}; }
[[noreturn]] inline void bad_cursor(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_BAD_CURSOR, msg}; }

// ---- field checks (_wire.py's _is_int, _u32, _name, ...) ------------------------------

inline bool is_int(const Json& v) { return v.is_int(); }
inline bool is_num(const Json& v) { return v.is_number(); }
inline bool is_uint(const Json& v) {
    uint64_t u;
    return v.get_u64(&u);
}
inline bool is_u32(const Json& v) {
    uint64_t u;
    return v.get_u64(&u) && u <= 0xffffffffull;
}
inline bool is_str(const Json& v) { return v.is_string(); }
inline bool is_obj(const Json& v) { return v.is_object(); }
inline bool is_list(const Json& v) { return v.is_array(); }
inline bool is_bool(const Json& v) { return v.is_bool(); }
inline bool is_text(const Json& v) { return v.is_string() && v.as_string().size() <= CALIPER_NET_TEXT_MAX; }
inline bool is_finite(const Json& v) { return v.is_number() && std::isfinite(v.as_double()); }
inline bool is_nonneg(const Json& v) { return is_finite(v) && v.as_double() >= 0; }
inline bool is_positive_num(const Json& v) { return is_finite(v) && v.as_double() > 0; }
inline bool present(const Json* v) { return v && !v->is_null(); }

inline bool is_name(const Json& v) {
    return v.is_string() && !v.as_string().empty() && v.as_string().size() <= CALIPER_NET_NAME_MAX;
}
inline bool name_ok(const Json* v) { return v && is_name(*v); }

inline bool is_strs(const Json& v) {
    if (!v.is_array()) return false;
    for (const Json& x : v.items())
        if (!x.is_string()) return false;
    return true;
}

inline bool is_session_id(const Json& v) {
    if (!v.is_string()) return false;
    const std::string& s = v.as_string();
    if (s.empty() || s.size() > CALIPER_NET_SESSION_ID_MAX) return false;
    for (unsigned char c : s)
        if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-') || c >= 0x80) return false;
    return true;
}

inline bool is_identity(const Json& v) {
    if (!v.is_object()) return false;
    const Json* id = v.find("id");
    const Json* script = v.find("script");
    const Json* pid = v.find("pid");
    const Json* start = v.find("start_unix_ms");
    return id && is_session_id(*id) && script && script->is_string() && pid && is_uint(*pid) && start &&
           is_uint(*start);
}

// An integer of magnitude at most 2^53, exact in every implementation.
inline bool is_int53(const Json& v) {
    int64_t i = 0;
    uint64_t u = 0;
    if (v.get_i64(&i)) return i >= -(int64_t(1) << 53) && i <= (int64_t(1) << 53);
    return v.get_u64(&u) && u <= (uint64_t(1) << 53);
}

// A JSON integer within [lo, hi], never a float. Both bounds are below 2^63, so no arithmetic is needed.
inline bool int_in(const Json& v, uint64_t lo, uint64_t hi) {
    uint64_t u = 0;
    return v.get_u64(&u) && u >= lo && u <= hi;
}

// An unsigned integer every codec holds exactly: 0 to 2^53 - 1.
inline constexpr uint64_t kSafeInt = (uint64_t(1) << 53) - 1;
inline bool is_safe_uint(const Json& v) { return int_in(v, 0, kSafeInt); }

// 1 to `limit` bytes of text.
inline bool bounded_text(const Json* v, size_t limit) {
    return v && v->is_string() && !v->as_string().empty() && v->as_string().size() <= limit;
}

inline bool one_of(const Json& v, std::initializer_list<const char*> words) {
    if (!v.is_string()) return false;
    for (const char* w : words)
        if (v.as_string() == w) return true;
    return false;
}

// The optional key `key` of `o` (absent or null passes) when `check` holds for it.
template <typename Check>
inline bool opt_ok(const Json& o, const char* key, Check check) {
    const Json* v = o.find(key);
    return !present(v) || check(*v);
}

inline std::string show(const Json& v) {
    std::string s = v.dump();
    if (s.size() > 80) s = s.substr(0, 77) + "...";
    return s;
}

template <typename Check>
inline const Json& req(const Json& m, const char* key, Check check, const char* what) {
    const Json* v = m.find(key);
    if (!v) bad(std::string("missing '") + key + "'");
    if (!check(*v)) bad(std::string("'") + key + "' must be " + what + ", got " + show(*v));
    return *v;
}

// The value, or nullptr when absent or null (Python's meta.get(key) is None).
template <typename Check>
inline const Json* opt(const Json& m, const char* key, Check check, const char* what) {
    const Json* v = m.find(key);
    if (!v || v->is_null()) return nullptr;
    if (!check(*v)) bad(std::string("'") + key + "' must be " + what + ", got " + show(*v));
    return v;
}

inline uint64_t u64(const Json& v) {
    uint64_t u = 0;
    v.get_u64(&u);
    return u;
}
inline uint32_t u32(const Json& v) { return (uint32_t)u64(v); }
inline int64_t i64(const Json& v) {
    int64_t i = 0;
    if (!v.get_i64(&i)) {
        uint64_t u = 0;
        v.get_u64(&u);
        i = (int64_t)std::min<uint64_t>(u, (uint64_t)INT64_MAX);
    }
    return i;
}

inline std::vector<std::string> strs(const Json& v) {
    std::vector<std::string> out;
    for (const Json& x : v.items()) out.push_back(x.as_string());
    return out;
}

inline Json str_list(const std::vector<std::string>& v) {
    Json a = Json::array();
    for (const auto& s : v) a.push(s);
    return a;
}

inline Section named(const Section& s, const char* name) {
    Section o = s;
    o.name = name;
    return o;
}

// Strict UTF-8 check (wire.cpp).
bool utf8_ok(const uint8_t* p, size_t n);

// Controls, interventions, interpretation and linked selection (wire_controls.cpp).
bool is_set_value(const Json& v);
bool interpretation_object(const Json& v);
bool interpretation_request(const Json& v);
Controls decode_controls(const Frame& f);
Control decode_control(const Frame& f);
ControlReply decode_control_reply(const Frame& f);
Selection decode_selection(const Frame& f);
void controls_to_meta(const Controls& c, Json& m, std::vector<Section>& sections);
void control_to_meta(const Control& c, Json& m);
void control_reply_to_meta(const ControlReply& r, Json& m);
void selection_to_meta(const Selection& s, Json& m);

// Scenes (wire_scene.cpp).
const Section* part_section(const Frame& f, const char* name, int64_t dtype, std::vector<int64_t> shape);
SceneUpdate decode_scene_update(const Frame& f);
SceneTransforms decode_scene_transforms(const Frame& f);
SceneClear decode_scene_clear(const Frame& f);
SceneClasses decode_scene_classes(const Frame& f);
void scene_update_to_meta(const SceneUpdate& u, Json& m, std::vector<Section>& sections);
void scene_transforms_to_meta(const SceneTransforms& t, Json& m, std::vector<Section>& sections);
void scene_clear_to_meta(const SceneClear& c, Json& m, std::vector<Section>& sections);
void scene_classes_to_meta(const SceneClasses& k, Json& m, std::vector<Section>& sections);

// Timelines, text logs, bar charts and recordings (wire_timelines.cpp).
bool is_recorded(const Json& v);
bool is_recording_ref(const Json& v);
bool is_timelines(const Json& v);
void snapshot_32(const Frame& f, Snapshot& s);
void series_32(const Frame& f, Series& s);
void scene_stamp(const Frame& f, bool is_static, int64_t rows, std::optional<Section>* stamp,
                 std::optional<Section>* stamps = nullptr);
Text decode_text(const Frame& f);
Recordings decode_recordings(const Frame& f);
RecordingOpen decode_recording_open(const Frame& f);
Cursor decode_cursor(const Frame& f);
TimeState decode_time_state(const Frame& f);
void text_to_meta(const Text& x, Json& m, std::vector<Section>& sections);
void cursor_to_meta(const Cursor& c, Json& m);
void time_state_to_meta(const TimeState& t, Json& m);

// GPU programs, piece instances and resources (wire_programs.cpp).
bool program_id_ok(const Json* v);
bool piece_pin_ok(const Json& v);
bool param_name_ok(const std::string& s);
Program decode_program(const Frame& f);
ProgramState decode_program_state(const Frame& f);
Resource decode_resource(const Frame& f);
void program_to_meta(const Program& p, Json& m, std::vector<Section>& sections);
void program_state_to_meta(const ProgramState& s, Json& m);
void resource_to_meta(const Resource& r, Json& m, std::vector<Section>& sections);

}  // namespace detail
}  // namespace wire
}  // namespace caliper_net
