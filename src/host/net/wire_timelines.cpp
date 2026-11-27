// wire_timelines.cpp: protocol 3.2 timelines, stamps, text logs, bar charts,
// recordings, cursor, and time_state codecs (wire_internal.h).
#include "net/wire_internal.h"

#include <cstring>
#include <set>

namespace caliper_net {
namespace wire {

namespace {

using namespace detail;

bool is_timeline_name(const Json& v) {
    if (!v.is_string() || v.as_string().empty() || v.as_string().size() > 63) return false;
    for (unsigned char c : v.as_string())
        if (!(std::isalnum(c) || c == '_' || c == '.' || c == '-') || c >= 0x80) return false;
    return true;
}

bool is_stamp_text(const Json& v) { return v.is_string() && stamp_text_ok(v.as_string()); }
bool is_timeline_kind(const Json& v) { return one_of(v, {CALIPER_NET_TIMELINE_SEQUENCE, CALIPER_NET_TIMELINE_TIME}); }

bool opt_bool(const Json& o, const char* key) {
    const Json* v = o.find(key);
    return !v || v->is_null() || v->is_bool();
}

bool is_recording_entry(const Json& e) {
    if (!e.is_object()) return false;
    auto has = [&](const char* k, auto ok) {
        const Json* v = e.find(k);
        return v && ok(*v);
    };
    if (!(has("id", is_session_id) && has("title", is_str) && has("script", is_str) && has("bytes", is_uint) &&
          has("items", is_uint) && has("complete", [](const Json& v) { return v.is_bool(); })))
        return false;
    for (const char* k : {"created_unix_ms", "ended_unix_ms"}) {
        const Json* v = e.find(k);
        if (v && !v->is_null() && !is_uint(*v)) return false;
    }
    const Json* tl = e.find("timelines");
    if (!tl || !tl->is_array() || tl->size() > CALIPER_NET_MAX_TIMELINES) return false;
    for (const Json& t : tl->items()) {
        if (!t.is_object()) return false;
        const Json* n = t.find("name");
        const Json* k = t.find("kind");
        const Json* lo = t.find("min");
        const Json* hi = t.find("max");
        if (!(n && is_timeline_name(*n) && k && is_timeline_kind(*k) && opt_bool(t, "epoch") && lo &&
              is_stamp_text(*lo) && hi && is_stamp_text(*hi)))
            return false;
    }
    return true;
}

int64_t stamp_value(const std::string& s) { return std::stoll(s); }   // after stamp_text_ok

bool is_speed(const Json& v) {
    return is_finite(v) && v.as_double() >= CALIPER_NET_MIN_SPEED && v.as_double() <= CALIPER_NET_MAX_SPEED;
}
bool is_pace(const Json& v) { return one_of(v, {CALIPER_NET_PACE_RECORDED, CALIPER_NET_PACE_EVEN}); }
bool is_loop(const Json& v) {
    return v.is_array() && v.size() == 2 && is_stamp_text(v.at(0)) && is_stamp_text(v.at(1)) &&
           stamp_value(v.at(0).as_string()) <= stamp_value(v.at(1).as_string());
}

bool is_ranges(const Json& v) {
    if (!v.is_object()) return false;
    for (const auto& kv : v.members())
        if (!is_timeline_name(Json(kv.first)) || !is_loop(kv.second)) return false;
    return true;
}

}  // namespace

namespace detail {

// The stamp sections of a message (_stamp): `stamp` i64 (T,), and where rows >= 0 `stamps` i64 (rows, T),
// T from 1 to CALIPER_NET_MAX_TIMELINES; a static item carries neither.
void scene_stamp(const Frame& f, bool is_static, int64_t rows, std::optional<Section>* stamp,
                 std::optional<Section>* stamps) {
    const Section* s = f.section("stamp");
    const Section* ss = rows >= 0 ? f.section("stamps") : nullptr;
    if (s && ss) bad("'stamp' and 'stamps' together");
    if (is_static && (s || ss)) bad("a static item carries no stamp");
    const int64_t tmax = CALIPER_NET_MAX_TIMELINES;
    if (s && !(s->is(CALIPER_NET_DTYPE_I64, 1) && s->shape[0] >= 1 && s->shape[0] <= tmax))
        bad("'stamp' must be i64 (T,) with T from 1 to 8");
    if (ss && !(ss->is(CALIPER_NET_DTYPE_I64, 2) && ss->shape[0] == rows && ss->shape[1] >= 1 && ss->shape[1] <= tmax))
        bad("'stamps' must be i64 (" + std::to_string(rows) + ", T) with T from 1 to 8");
    if (s) *stamp = *s;
    if (ss && stamps) *stamps = *ss;
}

bool is_timelines(const Json& v) {
    if (!v.is_array() || v.size() < 2 || v.size() > CALIPER_NET_MAX_TIMELINES) return false;
    std::set<std::string> names;
    for (const Json& t : v.items()) {
        if (!t.is_object()) return false;
        const Json* n = t.find("name");
        const Json* k = t.find("kind");
        if (!(n && is_timeline_name(*n) && k && is_timeline_kind(*k) && opt_bool(t, "epoch"))) return false;
        names.insert(n->as_string());
    }
    if (names.size() != v.size()) return false;
    return v.at(0).find("name")->as_string() == "step" && v.at(0).find("kind")->as_string() == "sequence" &&
           v.at(1).find("name")->as_string() == "time" && v.at(1).find("kind")->as_string() == "time";
}

bool is_recorded(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_MAX_RECORDED) return false;
    for (const Json& x : v.items())
        if (!x.is_object()) return false;
    return true;
}

bool is_recording_ref(const Json& v) {
    if (!v.is_object()) return false;
    const Json* id = v.find("id");
    const Json* seekable = v.find("seekable");
    const Json* complete = v.find("complete");
    return id && is_session_id(*id) && seekable && seekable->is_bool() && complete && complete->is_bool();
}

void snapshot_32(const Frame& f, Snapshot& s) {
    const Json& m = f.meta;
    if (s.kind == CALIPER_NET_KIND_BARS && !s.empty) {
        const int64_t b = s.data.shape[0];
        if (const Section* c = part_section(f, "colors", CALIPER_NET_DTYPE_U8, {b, 4})) s.colors = *c;
        if (const Json* l = opt(m, "labels", [&](const Json& v) {
                if (!v.is_array() || (int64_t)v.size() != b) return false;
                for (const Json& x : v.items())
                    if (!x.is_string() || x.as_string().size() > CALIPER_NET_NAME_MAX) return false;
                return true;
            }, "a label of at most 63 bytes per bar"))
            s.labels = *l;
        if (const Json* h = opt(m, "horizontal", is_bool, "a boolean")) s.horizontal = h->as_bool();
    }
    if (const Json* v = opt(m, "static", is_bool, "a boolean")) s.is_static = v->as_bool();
    if (const Json* v = opt(m, "cursor", is_u32, "a uint32")) s.cursor = u32(*v);
    if (const Json* v = opt(m, "cadence", is_obj, "an object")) s.cadence = *v;
    scene_stamp(f, s.is_static, -1, &s.stamp);
}

void series_32(const Frame& f, Series& s) {
    const Json& m = f.meta;
    const int64_t n = s.values.shape[0];
    if (const Section* ix = part_section(f, "indices", CALIPER_NET_DTYPE_I64, {n})) s.indices = *ix;
    if (const Json* d = opt(m, "decimated", [](const Json& v) {
            int64_t i = 0;
            return v.is_int() && v.get_i64(&i) && i >= 1;
        }, "an integer >= 1"))
        s.decimated = u64(*d);
    if (s.indices.has_value() != s.decimated.has_value()) bad("'indices' and 'decimated' go together");
    if (const Json* v = opt(m, "cursor", is_u32, "a uint32")) s.cursor = u32(*v);
    scene_stamp(f, false, n, &s.stamp, &s.stamps);
}

Text decode_text(const Frame& f) {
    const Json& m = f.meta;
    const Section* t = f.section("text");
    if (!t || !t->is(CALIPER_NET_DTYPE_U8, 1)) bad("a text message needs a (L,) u8 'text' section");
    const Section* o = f.section("offsets");
    if (!o || !o->is(CALIPER_NET_DTYPE_I32, 1) || o->shape[0] < 2)
        bad("a text message needs an (M + 1,) i32 'offsets' section, M >= 1");
    const int64_t n = o->shape[0] - 1;
    if (n > (int64_t)CALIPER_NET_TEXT_MAX_LINES) bad(std::to_string(n) + " lines, more than 65536 in one message");
    const Section* lv = part_section(f, "levels", CALIPER_NET_DTYPE_U8, {n});
    if (!lv) bad("a text message needs an (M,) u8 'levels' section");
    Text x;
    x.sub = u32(req(m, "sub", is_u32, "a uint32"));
    x.view = req(m, "view", is_name, "a view name").as_string();
    x.start = u64(req(m, "start", is_uint, "an unsigned integer"));
    x.total = u64(req(m, "total", is_uint, "an unsigned integer"));
    x.stamp_ns = u64(req(m, "stamp_ns", is_uint, "an unsigned integer"));
    if (const Json* v = opt(m, "dropped", is_uint, "an unsigned integer")) x.dropped = u64(*v);
    if (const Json* v = opt(m, "cursor", is_u32, "a uint32")) x.cursor = u32(*v);
    if (x.start + (uint64_t)n < x.start || x.start + (uint64_t)n > x.total)
        bad("text: " + std::to_string(n) + " lines from " + std::to_string(x.start) + " with total " +
            std::to_string(x.total));
    x.text = *t;
    x.offsets = *o;
    x.levels = *lv;
    std::vector<int32_t> offs((size_t)n + 1);
    std::memcpy(offs.data(), o->data, offs.size() * 4);
    if (offs[0] != 0 || (int64_t)offs[(size_t)n] != t->shape[0])
        bad("'offsets' must start at 0 and end at the bytes of 'text'");
    for (int64_t i = 0; i < n; ++i)
        if (offs[(size_t)i + 1] < offs[(size_t)i]) bad("'offsets' must not decrease");
    for (int64_t i = 0; i < n; ++i)
        if (offs[(size_t)i + 1] - offs[(size_t)i] > (int32_t)CALIPER_NET_TEXT_LINE_MAX)
            bad("a line is at most 4096 bytes");
    for (int64_t i = 0; i < n; ++i)
        if (lv->data[i] > CALIPER_NET_LEVEL_ERROR) bad("a level is 0 (debug), 1 (info), 2 (warning) or 3 (error)");
    for (int64_t i = 0; i < n; ++i)
        if (!utf8_ok(t->data + offs[(size_t)i], (size_t)(offs[(size_t)i + 1] - offs[(size_t)i])))
            bad("a line of 'text' is not UTF-8");
    scene_stamp(f, false, n, &x.stamp, &x.stamps);
    return x;
}

Recordings decode_recordings(const Frame& f) {
    Recordings r;
    r.recordings = req(f.meta, "recordings", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_RECORDINGS) return false;
        for (const Json& e : v.items())
            if (!is_recording_entry(e)) return false;
        return true;
    }, "a list of at most 1024 recordings");
    return r;
}

RecordingOpen decode_recording_open(const Frame& f) {
    RecordingOpen r;
    r.request_id = u32(req(f.meta, "request_id", is_u32, "a uint32"));
    r.recording = req(f.meta, "recording", is_session_id, "a recording id").as_string();
    return r;
}

Cursor decode_cursor(const Frame& f) {
    const Json& m = f.meta;
    Cursor c;
    c.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    c.op = req(m, "op", [](const Json& v) {
        return one_of(v, {CALIPER_NET_CURSOR_SEEK, CALIPER_NET_CURSOR_PLAY, CALIPER_NET_CURSOR_PAUSE,
                          CALIPER_NET_CURSOR_STEP, CALIPER_NET_CURSOR_FOLLOW});
    }, "seek, play, pause, step or follow").as_string();
    auto get = [&](const char* key, auto ok, const char* what, bool need) -> const Json* {
        const Json* v = m.find(key);
        if (!v || v->is_null()) {
            if (need) bad_cursor("'" + c.op + "' needs '" + key + "'");
            return nullptr;
        }
        if (!ok(*v)) bad_cursor(std::string("'") + key + "' must be " + what);
        return v;
    };
    const bool seek = c.op == CALIPER_NET_CURSOR_SEEK, play = c.op == CALIPER_NET_CURSOR_PLAY,
               step = c.op == CALIPER_NET_CURSOR_STEP;
    if (seek || play || step) c.timeline = get("timeline", is_timeline_name, "a timeline name", true)->as_string();
    if (seek) c.value = get("value", is_stamp_text, "stamp text", true)->as_string();
    if (step) {
        const Json* n = get("count", [](const Json& v) {
            int64_t i = 0;
            return v.is_int() && v.get_i64(&i) && i != 0 && i >= -(int64_t)CALIPER_NET_MAX_CURSOR_STEP &&
                   i <= (int64_t)CALIPER_NET_MAX_CURSOR_STEP;
        }, "an integer from -1000 to 1000, not 0", true);
        c.count = i64(*n);
    }
    if (play) {
        if (const Json* v = get("speed", is_speed, "a number from 0.0625 to 64", false)) c.speed = v->as_double();
        if (const Json* v = get("pace", is_pace, "recorded or even", false)) c.pace = v->as_string();
        if (const Json* v = get("loop", is_loop, "[start, end] as stamp text", false)) c.loop = *v;
    }
    return c;
}

TimeState decode_time_state(const Frame& f) {
    const Json& m = f.meta;
    TimeState t;
    if (const Json* v = opt(m, "request_id", is_u32, "a uint32")) t.request_id = u32(*v);
    t.session = req(m, "session", is_session_id, "a session id").as_string();
    t.generation = u32(req(m, "generation", is_u32, "a uint32"));
    t.state = req(m, "state", [](const Json& v) {
        return one_of(v, {CALIPER_NET_TIME_FOLLOWING, CALIPER_NET_TIME_PAUSED, CALIPER_NET_TIME_PLAYING});
    }, "following, paused or playing").as_string();
    t.timeline = req(m, "timeline", is_timeline_name, "a timeline name").as_string();
    if (const Json* v = opt(m, "cursor", is_stamp_text, "stamp text or null")) t.cursor = v->as_string();
    if (const Json* v = opt(m, "speed", is_speed, "a number from 0.0625 to 64")) t.speed = v->as_double();
    if (const Json* v = opt(m, "pace", is_pace, "recorded or even")) t.pace = v->as_string();
    if (const Json* v = opt(m, "loop", is_loop, "[start, end] as stamp text")) t.loop = *v;
    t.ranges = req(m, "ranges", is_ranges, "an object of [min, max] stamp text per timeline");
    t.live = req(m, "live", is_bool, "a boolean").as_bool();
    if ((t.state == CALIPER_NET_TIME_FOLLOWING) != !t.cursor.has_value())
        bad("the cursor is null exactly while following");
    return t;
}

void text_to_meta(const Text& x, Json& m, std::vector<Section>& sections) {
    m.set("sub", x.sub);
    m.set("view", x.view);
    m.set("start", x.start);
    m.set("total", x.total);
    m.set("stamp_ns", x.stamp_ns);
    if (x.dropped) m.set("dropped", *x.dropped);
    if (x.cursor) m.set("cursor", *x.cursor);
    sections.push_back(named(x.text, "text"));
    sections.push_back(named(x.offsets, "offsets"));
    sections.push_back(named(x.levels, "levels"));
    if (x.stamps) sections.push_back(named(*x.stamps, "stamps"));
    else if (x.stamp) sections.push_back(named(*x.stamp, "stamp"));
}

void cursor_to_meta(const Cursor& c, Json& m) {
    m.set("request_id", c.request_id);
    m.set("op", c.op);
    if (c.timeline) m.set("timeline", *c.timeline);
    if (c.value) m.set("value", *c.value);
    if (c.count) m.set("count", *c.count);
    if (c.speed) m.set("speed", *c.speed);
    if (c.pace) m.set("pace", *c.pace);
    if (!c.loop.is_null()) m.set("loop", c.loop);
}

void time_state_to_meta(const TimeState& t, Json& m) {
    if (t.request_id) m.set("request_id", *t.request_id);
    m.set("session", t.session);
    m.set("generation", t.generation);
    m.set("state", t.state);
    m.set("timeline", t.timeline);
    m.set("cursor", t.cursor ? Json(*t.cursor) : Json());
    if (t.speed) m.set("speed", *t.speed);
    if (t.pace) m.set("pace", *t.pace);
    if (!t.loop.is_null()) m.set("loop", t.loop);
    m.set("ranges", t.ranges);
    m.set("live", t.live);
}

}  // namespace detail

bool stamp_text_ok(const std::string& v) {
    if (v.empty() || v.size() > CALIPER_NET_STAMP_TEXT_MAX) return false;
    const bool neg = v[0] == '-';
    size_t i = neg ? 1 : 0;
    if (i == v.size()) return false;
    for (size_t k = i; k < v.size(); ++k)
        if (v[k] < '0' || v[k] > '9') return false;
    while (i + 1 < v.size() && v[i] == '0') ++i;   // leading zeros
    const std::string digits = v.substr(i);
    const std::string lim = neg ? "9223372036854775808" : "9223372036854775807";
    return digits.size() < lim.size() || (digits.size() == lim.size() && digits <= lim);
}

std::string Text::line(size_t i) const {
    int32_t a = 0, b = 0;
    std::memcpy(&a, offsets.data + 4 * i, 4);
    std::memcpy(&b, offsets.data + 4 * (i + 1), 4);
    return std::string((const char*)text.data + a, (size_t)(b - a));
}

}  // namespace wire
}  // namespace caliper_net
