// wire.cpp: frames and messages of wire protocol v3 (wire.h). The checks and
// their order follow python/caliper/_wire.py line for line, so the two codecs
// accept and refuse the same frames.
#include "net/wire.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <set>
#include <type_traits>

namespace caliper_net {
namespace wire {
namespace {

// ---- little-endian ------------------------------------------------------------------

uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
void wr16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void wr32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
uint64_t align8(uint64_t n) { return (n + (CALIPER_NET_SECTION_ALIGN - 1)) & ~uint64_t(CALIPER_NET_SECTION_ALIGN - 1); }

// ---- field checks (_wire.py's _is_int, _u32, _name, ...) ------------------------------

struct MsgError {
    std::string code;
    std::string message;
};

[[noreturn]] void bad(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_BAD_REQUEST, msg}; }

bool is_int(const Json& v) { return v.is_int(); }
bool is_num(const Json& v) { return v.is_number(); }
bool is_uint(const Json& v) {
    uint64_t u;
    return v.get_u64(&u);
}
bool is_u32(const Json& v) {
    uint64_t u;
    return v.get_u64(&u) && u <= 0xffffffffull;
}
bool is_str(const Json& v) { return v.is_string(); }
bool is_obj(const Json& v) { return v.is_object(); }
bool is_list(const Json& v) { return v.is_array(); }
bool is_name(const Json& v) {
    return v.is_string() && !v.as_string().empty() && v.as_string().size() <= CALIPER_NET_NAME_MAX;
}
bool is_strs(const Json& v) {
    if (!v.is_array()) return false;
    for (const Json& x : v.items())
        if (!x.is_string()) return false;
    return true;
}
bool is_session_id(const Json& v) {
    if (!v.is_string()) return false;
    const std::string& s = v.as_string();
    if (s.empty() || s.size() > CALIPER_NET_SESSION_ID_MAX) return false;
    for (unsigned char c : s)
        if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-') || c >= 0x80) return false;
    return true;
}
bool is_identity(const Json& v) {
    if (!v.is_object()) return false;
    const Json* id = v.find("id");
    const Json* script = v.find("script");
    const Json* pid = v.find("pid");
    const Json* start = v.find("start_unix_ms");
    return id && is_session_id(*id) && script && script->is_string() && pid && is_uint(*pid) && start &&
           is_uint(*start);
}

std::string show(const Json& v) {
    std::string s = v.dump();
    if (s.size() > 80) s = s.substr(0, 77) + "...";
    return s;
}

template <typename Check>
const Json& req(const Json& m, const char* key, Check check, const char* what) {
    const Json* v = m.find(key);
    if (!v) bad(std::string("missing '") + key + "'");
    if (!check(*v)) bad(std::string("'") + key + "' must be " + what + ", got " + show(*v));
    return *v;
}

// The value, or nullptr when absent or null (Python's meta.get(key) is None).
template <typename Check>
const Json* opt(const Json& m, const char* key, Check check, const char* what) {
    const Json* v = m.find(key);
    if (!v || v->is_null()) return nullptr;
    if (!check(*v)) bad(std::string("'") + key + "' must be " + what + ", got " + show(*v));
    return v;
}

uint64_t u64(const Json& v) {
    uint64_t u = 0;
    v.get_u64(&u);
    return u;
}
uint32_t u32(const Json& v) { return (uint32_t)u64(v); }
int64_t i64(const Json& v) {
    int64_t i = 0;
    if (!v.get_i64(&i)) {
        uint64_t u = 0;
        v.get_u64(&u);
        i = (int64_t)std::min<uint64_t>(u, (uint64_t)INT64_MAX);
    }
    return i;
}

std::vector<std::string> strs(const Json& v) {
    std::vector<std::string> out;
    for (const Json& x : v.items()) out.push_back(x.as_string());
    return out;
}

Json str_list(const std::vector<std::string>& v) {
    Json a = Json::array();
    for (const auto& s : v) a.push(s);
    return a;
}

// ---- 3.2 draft: the checks of the 3.2 keys and sections of the 3.0 and 3.1 messages, defined
// in the 3.2 block at the end ----
bool is_recorded(const Json& v);
bool is_recording_ref(const Json& v);
bool is_timelines(const Json& v);
void snapshot_32(const Frame& f, Snapshot& s);
void series_32(const Frame& f, Series& s);
void scene_stamp(const Frame& f, bool is_static, int64_t rows, std::optional<Section>* stamp,
                 std::optional<Section>* stamps = nullptr);

// ---- view entries -------------------------------------------------------------------------

bool view_entry(const Json& v, ViewInfo* out) {
    if (!is_obj(v)) return false;
    const Json* id = v.find("id");
    const Json* name = v.find("name");
    const Json* kind = v.find("kind");
    const Json* sources = v.find("sources");
    if (!(id && is_uint(*id) && name && is_name(*name) && kind && is_int(*kind) && sources &&
          is_list(*sources)))
        return false;
    ViewInfo vi;
    vi.id = u64(*id);
    vi.name = name->as_string();
    vi.kind = i64(*kind);
    for (const Json& s : sources->items()) {
        if (!is_obj(s)) return false;
        const Json* sn = s.find("name");
        const Json* dims = s.find("dims");
        if (!(sn && is_name(*sn) && dims && is_list(*dims))) return false;
        SourceInfo si;
        si.name = sn->as_string();
        if (const Json* dt = s.find("dtype"); dt && dt->is_string()) si.dtype = dt->as_string();
        for (const Json& d : dims->items()) {
            if (!is_obj(d)) return false;
            const Json* dn = d.find("name");
            const Json* sz = d.find("size");
            const Json* it = d.find("items");   // 3.1: an item space
            if (!(dn && is_name(*dn) && sz && is_uint(*sz))) return false;
            if (it && !it->is_null() && !is_name(*it)) return false;
            si.dims.push_back({dn->as_string(), u64(*sz), it && it->is_string() ? it->as_string() : std::string()});
        }
        vi.sources.push_back(std::move(si));
    }
    const Json* def = v.find("default");
    if (def && !def->is_null() && !is_obj(*def)) return false;
    if (def && def->is_object()) vi.default_view = *def;
    const Json* items = v.find("items");   // 3.1: a series' item space
    if (items && !items->is_null() && !is_name(*items)) return false;
    if (items && items->is_string()) vi.items = items->as_string();
    const Json* recorded = v.find("recorded");   // 3.2 draft
    if (recorded && !recorded->is_null() && !is_recorded(*recorded)) return false;
    if (out) *out = std::move(vi);
    return true;
}

// ---- message decoders (from_frame) ----------------------------------------------------------

Hello hello(const Frame& f) {
    const Json& m = f.meta;
    Hello h;
    h.role = req(m, "role", [](const Json& v) {
        return v.is_string() && (v.as_string() == CALIPER_NET_ROLE_VIEWER || v.as_string() == CALIPER_NET_ROLE_PRODUCER);
    }, "viewer or producer").as_string();
    if (const Json* s = opt(m, "session", is_identity, "a session identity")) h.session = *s;
    if (h.role == CALIPER_NET_ROLE_PRODUCER && h.session.is_null()) bad("a producer's hello needs 'session'");
    if (const Json* c = opt(m, "capabilities", is_strs, "a list of strings")) h.capabilities = strs(*c);
    if (const Json* s = opt(m, "software", is_obj, "an object")) h.software = *s;
    return h;
}

Welcome welcome(const Frame& f) {
    const Json& m = f.meta;
    Welcome w;
    w.role = req(m, "role", [](const Json& v) {
        return v.is_string() && (v.as_string() == CALIPER_NET_ROLE_PRODUCER || v.as_string() == CALIPER_NET_ROLE_HUB);
    }, "producer or hub").as_string();
    if (const Json* c = opt(m, "capabilities", is_strs, "a list of strings")) w.capabilities = strs(*c);
    if (const Json* s = opt(m, "software", is_obj, "an object")) w.software = *s;
    if (const Json* s = opt(m, "session", is_identity, "a session identity")) w.session = *s;
    if (const Json* c = opt(m, "control", [](const Json& v) { return v.is_bool(); }, "a boolean"))
        w.control = c->as_bool();
    return w;
}

Bye bye(const Frame& f) {
    Bye b;
    b.reason = req(f.meta, "reason", is_str, "a string").as_string();
    if (const Json* s = opt(f.meta, "message", is_str, "a string")) b.message = s->as_string();
    return b;
}

Error error(const Frame& f) {
    const Json& m = f.meta;
    Error e;
    e.code = req(m, "code", is_str, "a string").as_string();
    if (const Json* s = opt(m, "message", is_str, "a string")) e.message = s->as_string();
    if (const Json* s = opt(m, "fatal", [](const Json& v) { return v.is_bool(); }, "a boolean")) e.fatal = s->as_bool();
    if (const Json* s = opt(m, "in_reply_to", is_u32, "a message type")) e.in_reply_to = u32(*s);
    if (const Json* s = opt(m, "request_id", is_u32, "a uint32")) e.request_id = u32(*s);
    if (const Json* s = opt(m, "sub", is_u32, "a uint32")) e.sub = u32(*s);
    if (const Json* s = opt(m, "view", is_str, "a string")) e.view = s->as_string();
    if (const Json* s = opt(m, "session", is_str, "a string")) e.session = s->as_string();
    return e;
}

uint32_t nonce(const Frame& f) { return u32(req(f.meta, "nonce", is_u32, "a uint32")); }

Sessions sessions(const Frame& f) {
    Sessions s;
    const Json& list = req(f.meta, "sessions", [](const Json& v) {
        if (!v.is_array()) return false;
        for (const Json& x : v.items()) {
            const Json* st = x.find("state");
            const Json* views = x.find("views");
            const Json* rec = x.find("recording");   // 3.2 draft
            if (!(is_identity(x) && st && st->is_string() &&
                  (st->as_string() == CALIPER_NET_SESSION_LIVE || st->as_string() == CALIPER_NET_SESSION_ENDED) &&
                  views && is_uint(*views) && (!rec || rec->is_null() || is_recording_ref(*rec))))
                return false;
        }
        return true;
    }, "a list of sessions");
    s.sessions = list;
    for (const Json& x : list.items()) {
        SessionInfo si;
        Identity::parse(x, &si.identity);
        si.state = x.find("state")->as_string();
        si.views = u64(*x.find("views"));
        s.list.push_back(std::move(si));
    }
    return s;
}

Attach attach(const Frame& f) {
    if (!f.meta.has("session")) bad("missing 'session'");
    Attach a;
    if (const Json* s = opt(f.meta, "session", is_session_id, "a session id or null")) a.session = s->as_string();
    return a;
}

Views views(const Frame& f) {
    Views v;
    const Json& list = req(f.meta, "views", [](const Json& x) {
        if (!x.is_array()) return false;
        for (const Json& e : x.items())
            if (!view_entry(e, nullptr)) return false;
        return true;
    }, "a list of views");
    if (const Json* s = opt(f.meta, "session", is_session_id, "a session id")) v.session = s->as_string();
    if (const Json* t = opt(f.meta, "timelines", is_timelines, "2 to 8 timelines, step and time first"))
        v.timelines = *t;   // 3.2 draft
    v.views = list;
    for (const Json& e : list.items()) {
        ViewInfo vi;
        view_entry(e, &vi);
        v.list.push_back(std::move(vi));
    }
    return v;
}

bool is_rate(const Json& v) { return v.is_number() && v.as_double() > 0 && std::isfinite(v.as_double()); }

Subscribe subscribe(const Frame& f) {
    const Json& m = f.meta;
    Subscribe s;
    s.sub = u32(req(m, "sub", is_u32, "a uint32"));
    s.view = req(m, "view", is_name, "a view name").as_string();
    s.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    if (const Json* r = opt(m, "max_rate", is_rate, "a positive number")) s.max_rate = r->as_double();
    if (const Json* d = opt(m, "data_view", is_obj, "an object")) s.data_view = *d;
    return s;
}

Unsubscribe unsubscribe(const Frame& f) {
    Unsubscribe u;
    u.sub = u32(req(f.meta, "sub", is_u32, "a uint32"));
    return u;
}

ViewRequest view_request(const Frame& f) {
    const Json& m = f.meta;
    ViewRequest r;
    r.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    r.sub = u32(req(m, "sub", is_u32, "a uint32"));
    r.view = req(m, "view", is_name, "a view name").as_string();
    r.data_view = req(m, "data_view", is_obj, "an object");
    return r;
}

ViewReply view_reply(const Frame& f) {
    const Json& m = f.meta;
    ViewReply r;
    const Json* dv = opt(m, "data_view", is_obj, "an object");
    const Json* err = opt(m, "error", [](const Json& v) {
        const Json* c = v.find("code");
        return v.is_object() && c && c->is_string();
    }, "{code, message}");
    if ((dv == nullptr) == (err == nullptr)) bad("a view_reply carries exactly one of 'data_view' and 'error'");
    r.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    r.sub = u32(req(m, "sub", is_u32, "a uint32"));
    r.view = req(m, "view", is_name, "a view name").as_string();
    if (dv) r.data_view = *dv;
    if (err) r.error = *err;
    return r;
}

void check_payload(Snapshot& s) {
    const Section& d = s.data;
    int64_t want_dt = -1;
    size_t want_rank = 0;
    switch (s.kind) {
        case CALIPER_NET_KIND_HEATMAP: want_dt = CALIPER_NET_DTYPE_F32; want_rank = 2; break;
        case CALIPER_NET_KIND_IMAGE: want_dt = CALIPER_NET_DTYPE_U8; want_rank = 3; break;
        case CALIPER_NET_KIND_POINTS: want_dt = CALIPER_NET_DTYPE_F32; want_rank = 2; break;
        case CALIPER_NET_KIND_HISTOGRAM: want_dt = CALIPER_NET_DTYPE_F32; want_rank = 1; break;
        case CALIPER_NET_KIND_BARS: want_dt = CALIPER_NET_DTYPE_F32; want_rank = 1; break;   // 3.2 draft
        default: return;   // an unknown kind: the caller ignores it
    }
    if (d.dtype != want_dt || d.shape.size() != want_rank)
        bad("kind " + std::to_string(s.kind) + ": 'data' is dtype " + std::to_string(d.dtype) + " rank " +
            std::to_string(d.shape.size()));
    if (s.kind == CALIPER_NET_KIND_IMAGE && d.shape[2] != 4) bad("an image's 'data' is (H, W, 4)");
    if (s.kind == CALIPER_NET_KIND_BARS && (d.shape[0] < 1 || d.shape[0] > (int64_t)CALIPER_NET_MAX_BARS))
        bad("a bar chart has 1 to 4096 bars, got " + std::to_string(d.shape[0]));
    if (s.kind == CALIPER_NET_KIND_POINTS) {
        if ((d.shape[1] != 2 && d.shape[1] != 3) || (s.points_dims && *s.points_dims != d.shape[1]))
            bad("points 'data' is (N, points_dims) with points_dims 2 or 3");
        s.points_dims = d.shape[1];
        if (s.aux && (s.aux->dtype != CALIPER_NET_DTYPE_F32 || s.aux->shape.size() != 1 ||
                      s.aux->shape[0] != d.shape[0]))
            bad("points 'aux' is (N,) f32");
    }
}

Snapshot snapshot(const Frame& f) {
    const Json& m = f.meta;
    Snapshot s;
    s.kind = i64(req(m, "kind", is_int, "a kind id"));
    if (const Json* e = opt(m, "empty", [](const Json& v) { return v.is_bool(); }, "a boolean"))
        s.empty = e->as_bool();   // 3.2 draft
    const Section* data = f.section("data");
    if (s.empty) {
        if (!f.sections.empty()) bad("an empty snapshot carries no section");
    } else if (!data) {
        bad("a snapshot needs a 'data' section");
    }
    if (data) s.data = *data;
    if (const Section* aux = f.section("aux")) s.aux = *aux;
    if (const Json* st = opt(m, "stats", [](const Json& v) {
            if (!v.is_array() || v.size() > CALIPER_NET_MAX_STATS) return false;
            for (const Json& x : v.items())
                if (!(x.is_null() || x.is_number())) return false;
            return true;
        }, "a list of numbers or nulls")) {
        for (const Json& x : st->items()) s.stats.push_back(x.is_null() ? std::nan("") : x.as_double());
    }
    s.sub = u32(req(m, "sub", is_u32, "a uint32"));
    s.view = req(m, "view", is_name, "a view name").as_string();
    s.seq = u64(req(m, "seq", is_uint, "an unsigned integer"));
    s.stamp_ns = u64(req(m, "stamp_ns", is_uint, "an unsigned integer"));
    s.data_view = req(m, "data_view", is_obj, "an object");
    if (const Json* v = opt(m, "colormap", is_int, "an integer")) s.colormap = i64(*v);
    if (const Json* v = opt(m, "flags", is_uint, "an unsigned integer")) s.flags = u64(*v);
    if (const Json* v = opt(m, "vmin", is_num, "a number")) s.vmin = v->as_double();
    if (const Json* v = opt(m, "vmax", is_num, "a number")) s.vmax = v->as_double();
    if (const Json* v = opt(m, "axes", is_list, "a list")) s.axes = *v;
    if (const Json* v = opt(m, "tile", is_obj, "an object")) s.tile = *v;
    if (const Json* v = opt(m, "points_dims", [](const Json& x) {
            return x.is_number() && (x.as_double() == 2.0 || x.as_double() == 3.0);
        }, "2 or 3"))
        s.points_dims = (int64_t)v->as_double();
    if (const Json* v = opt(m, "label_count", is_uint, "an unsigned integer")) s.label_count = u64(*v);
    if (!s.empty) check_payload(s);
    snapshot_32(f, s);   // 3.2 draft
    return s;
}

Series series(const Frame& f) {
    const Json& m = f.meta;
    const Section* v = f.section("values");
    if (!v || v->dtype != CALIPER_NET_DTYPE_F32 || v->shape.size() != 1) bad("a series needs a (M,) f32 'values' section");
    Series s;
    s.sub = u32(req(m, "sub", is_u32, "a uint32"));
    s.view = req(m, "view", is_name, "a view name").as_string();
    s.start = u64(req(m, "start", is_uint, "an unsigned integer"));
    s.total = u64(req(m, "total", is_uint, "an unsigned integer"));
    s.stamp_ns = u64(req(m, "stamp_ns", is_uint, "an unsigned integer"));
    s.values = *v;
    const int64_t n = v->shape[0];
    const bool sparse = f.section("indices") != nullptr;   // 3.2 draft: a decimated recorded series
    if (n < 1 || n > (int64_t)CALIPER_NET_SERIES_MAX_VALUES ||
        (!sparse && (s.start + (uint64_t)n < s.start || s.start + (uint64_t)n > s.total)))
        bad("series: " + std::to_string(n) + " values from " + std::to_string(s.start) + " with total " +
            std::to_string(s.total));
    series_32(f, s);   // 3.2 draft
    return s;
}

Event event(const Frame& f) {
    const Json& m = f.meta;
    Event e;
    e.detail = req(m, "detail", is_obj, "an object");
    for (const char* key : {"indices", "row_ids", "items"}) {
        const Json* lst = e.detail.find(key);
        if (!lst || lst->is_null()) continue;
        bool ok = lst->is_array() && lst->size() <= CALIPER_NET_MAX_EVENT_INDICES;
        if (ok)
            for (const Json& x : lst->items())
                if (!is_uint(x)) { ok = false; break; }
        if (!ok)
            bad(std::string("detail.") + key + " must be at most " + std::to_string(CALIPER_NET_MAX_EVENT_INDICES) +
                " unsigned integers");
    }
    {   // 3.1: an item space and its item ids go together
        const Json* sp = e.detail.find("space");
        const Json* it = e.detail.find("items");
        const bool has_sp = sp && !sp->is_null(), has_it = it && !it->is_null();
        if (has_sp != has_it || (has_sp && !is_name(*sp)))
            bad("detail.space (an item space name) and detail.items go together");
    }
    e.view = req(m, "view", is_name, "a view name").as_string();
    e.type = req(m, "type", is_str, "a string").as_string();
    if (const Json* v = opt(m, "sub", is_u32, "a uint32")) e.sub = u32(*v);
    if (const Json* v = opt(m, "seq", is_uint, "an integer")) e.seq = u64(*v);
    if (const Json* v = opt(m, "data_view", is_obj, "an object")) e.data_view = *v;
    return e;
}

// ---- 3.1: controls and linked selection (the rules of _wire.py) ----------------------------

bool is_bool(const Json& v) { return v.is_bool(); }
bool is_text(const Json& v) { return v.is_string() && v.as_string().size() <= CALIPER_NET_TEXT_MAX; }
bool is_finite(const Json& v) { return v.is_number() && std::isfinite(v.as_double()); }
// An integer of magnitude at most 2^53, exact in every implementation.
bool is_int53(const Json& v) {
    int64_t i = 0;
    uint64_t u = 0;
    if (v.get_i64(&i)) return i >= -(int64_t(1) << 53) && i <= (int64_t(1) << 53);
    return v.get_u64(&u) && u <= (uint64_t(1) << 53);
}
bool one_of(const Json& v, std::initializer_list<const char*> words) {
    if (!v.is_string()) return false;
    for (const char* w : words)
        if (v.as_string() == w) return true;
    return false;
}
bool is_op(const Json& v) {
    return one_of(v, {CALIPER_NET_OP_PAUSE, CALIPER_NET_OP_RESUME, CALIPER_NET_OP_STEP, CALIPER_NET_OP_SET,
                      CALIPER_NET_OP_RUN, CALIPER_NET_OP_INTERVENE});
}
bool is_set_value(const Json& v) { return is_finite(v) || v.is_bool() || is_name(v); }
bool present(const Json* v) { return v && !v->is_null(); }

// Whether v is a value of parameter p's type, its range aside.
bool fits_type(const Json& p, const Json& v) {
    const std::string& t = p.find("type")->as_string();
    if (t == CALIPER_NET_PARAM_NUMBER) return is_finite(v);
    if (t == CALIPER_NET_PARAM_INTEGER) return v.is_int() && is_int53(v);
    if (t == CALIPER_NET_PARAM_BOOLEAN) return v.is_bool();
    if (!v.is_string()) return false;
    for (const Json& c : p.find("choices")->items())
        if (c.is_string() && c.as_string() == v.as_string()) return true;
    return false;
}

bool unique_names(const Json& list) {
    std::set<std::string> seen;
    for (const Json& e : list.items())
        if (!seen.insert(e.find("name")->as_string()).second) return false;
    return true;
}

bool param_entry(const Json& p) {
    if (!p.is_object()) return false;
    const Json* name = p.find("name");
    const Json* type = p.find("type");
    if (!(name && is_name(*name) && type &&
          one_of(*type, {CALIPER_NET_PARAM_NUMBER, CALIPER_NET_PARAM_INTEGER, CALIPER_NET_PARAM_BOOLEAN,
                         CALIPER_NET_PARAM_CHOICE})))
        return false;
    const std::string& t = type->as_string();
    if (t == CALIPER_NET_PARAM_NUMBER || t == CALIPER_NET_PARAM_INTEGER) {
        auto bound = [&](const Json& x) {
            return t == CALIPER_NET_PARAM_NUMBER ? is_finite(x) : (x.is_int() && is_int53(x));
        };
        const Json* lo = p.find("min");
        const Json* hi = p.find("max");
        if ((present(lo) && !bound(*lo)) || (present(hi) && !bound(*hi))) return false;
        if (present(lo) && present(hi) && lo->as_double() > hi->as_double()) return false;
        const Json* log = p.find("log");
        if (present(log) && !log->is_bool()) return false;
        if (present(log) && log->as_bool() && !(present(lo) && lo->as_double() > 0 && present(hi))) return false;
    } else if (t == CALIPER_NET_PARAM_CHOICE) {
        const Json* ch = p.find("choices");
        if (!(ch && ch->is_array() && ch->size() >= 1 && ch->size() <= CALIPER_NET_MAX_CHOICES)) return false;
        std::set<std::string> seen;
        for (const Json& c : ch->items())
            if (!is_name(c) || !seen.insert(c.as_string()).second) return false;
    }
    const Json* help = p.find("help");
    if (present(help) && !is_text(*help)) return false;
    const Json* value = p.find("value");
    if (!value || !fits_type(p, *value)) return false;
    const Json* pending = p.find("pending");
    return !present(pending) || fits_type(p, *pending);
}

bool action_entry(const Json& a) {
    if (!a.is_object()) return false;
    const Json* name = a.find("name");
    const Json* running = a.find("running");
    const Json* queued = a.find("queued");
    const Json* help = a.find("help");
    return name && is_name(*name) && running && running->is_bool() && queued && queued->is_bool() &&
           (!present(help) || is_text(*help));
}

bool run_obj(const Json& r) {
    if (!r.is_object()) return false;
    const Json* state = r.find("state");
    const Json* step = r.find("step");
    const Json* left = r.find("steps_left");
    return state &&
           one_of(*state, {CALIPER_NET_RUN_NONE, CALIPER_NET_RUN_RUNNING, CALIPER_NET_RUN_PAUSING,
                           CALIPER_NET_RUN_PAUSED, CALIPER_NET_RUN_STEPPING}) &&
           (!present(step) || is_uint(*step)) && (!present(left) || is_uint(*left));
}

// ---- 3.2 draft: interventions (CONTROLS "interventions", CONTROL op "intervene") ----------------
// A JSON integer within [lo, hi], never a float. Both bounds are below 2^63, so no arithmetic is needed.
bool int_in(const Json& v, uint64_t lo, uint64_t hi) {
    uint64_t u = 0;
    return v.get_u64(&u) && u >= lo && u <= hi;
}
// An unsigned integer every codec holds exactly: 0 to 2^53 - 1.
constexpr uint64_t kSafeInt = (uint64_t(1) << 53) - 1;
bool is_safe_uint(const Json& v) { return int_in(v, 0, kSafeInt); }
bool opt_step(const Json* v) { return !present(v) || is_safe_uint(*v); }
bool is_nonneg(const Json& v) { return is_finite(v) && v.as_double() >= 0; }
// 1 to `limit` bytes of text.
bool bounded_text(const Json* v, size_t limit) {
    return v && v->is_string() && !v->as_string().empty() && v->as_string().size() <= limit;
}
bool point_ok(const Json* v) { return bounded_text(v, CALIPER_NET_POINT_MAX); }
bool name_ok(const Json* v) { return v && is_name(*v); }

bool intervene_request(const Json& v) {
    if (!v.is_object()) return false;
    const Json* set = v.find("set");
    const Json* clear = v.find("clear");
    const Json* all = v.find("clear_all");
    if (int(present(set)) + int(present(clear)) + int(present(all)) != 1) return false;
    if (present(set)) {
        return set->is_object() && point_ok(set->find("point")) && name_ok(set->find("op")) &&
               opt_step(set->find("from_step")) && opt_step(set->find("until_step"));
    }
    if (present(clear)) return point_ok(clear);
    return all->is_bool() && all->as_bool();
}

bool point_entry(const Json& v) {
    if (!v.is_object() || !point_ok(v.find("point")) || !point_ok(v.find("module"))) return false;
    const Json* kind = v.find("kind");
    if (!(kind && one_of(*kind, {"module", "head"}))) return false;
    const Json* head = v.find("head");
    if (present(head) && !is_u32(*head)) return false;
    const Json* type = v.find("type");
    if (present(type) && !is_name(*type)) return false;
    const Json* shape = v.find("shape");
    if (!present(shape)) return true;
    if (!shape->is_array() || shape->size() > CALIPER_NET_MAX_DIMS) return false;
    for (const Json& d : shape->items())
        if (!is_safe_uint(d)) return false;
    return true;
}

bool active_entry(const Json& v) {
    if (!v.is_object() || !point_ok(v.find("point")) || !name_ok(v.find("op"))) return false;
    if (!opt_step(v.find("from_step")) || !opt_step(v.find("until_step")) || !opt_step(v.find("since_step")))
        return false;
    const Json* st = v.find("state");
    return st && one_of(*st, {"pending", "active", "ended"});
}

bool interventions_ok(const Json& v) {
    if (!v.is_object()) return false;
    const Json* points = v.find("points");
    const Json* active = v.find("active");
    if (!(points && points->is_array() && points->size() <= CALIPER_NET_MAX_POINTS_LISTED)) return false;
    if (!(active && active->is_array() && active->size() <= CALIPER_NET_MAX_INTERVENTIONS)) return false;
    for (const Json& p : points->items())
        if (!point_entry(p)) return false;
    for (const Json& a : active->items())
        if (!active_entry(a)) return false;
    return true;
}

Controls controls(const Frame& f) {
    const Json& m = f.meta;
    Controls c;
    c.run = req(m, "run", run_obj, "a run state");
    c.params = req(m, "params", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_PARAMS) return false;
        for (const Json& p : v.items())
            if (!param_entry(p)) return false;
        return unique_names(v);
    }, "at most 64 parameters with unique names");
    c.actions = req(m, "actions", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_ACTIONS) return false;
        for (const Json& a : v.items())
            if (!action_entry(a)) return false;
        return unique_names(v);
    }, "at most 64 actions with unique names");
    scene_stamp(f, false, -1, &c.stamp);   // 3.2 draft
    if (const Json* v = opt(m, "interventions", interventions_ok, "{points, active} of intervention points and interventions"))
        c.interventions = *v;
    return c;
}

Control control(const Frame& f) {
    const Json& m = f.meta;
    Control c;
    c.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    c.op = req(m, "op", is_op, "pause, resume, step, set, run or intervene").as_string();
    const Json* name = opt(m, "name", is_name, "a parameter or action name");
    const bool named = c.op == CALIPER_NET_OP_SET || c.op == CALIPER_NET_OP_RUN;
    if (named && !name) bad("'" + c.op + "' needs 'name'");
    if (c.op == CALIPER_NET_OP_SET) c.value = req(m, "value", is_set_value, "a number, a boolean or a choice");
    if (c.op == CALIPER_NET_OP_STEP)
        if (const Json* n = opt(m, "count", [](const Json& v) { return is_u32(v) && u64(v) >= 1; },
                                "a positive uint32"))
            c.count = u32(*n);
    if (c.op == CALIPER_NET_OP_INTERVENE)   // 3.2 draft
        c.intervene = req(m, "intervene", intervene_request,
                          "exactly one of {set: {point, op, from_step, until_step}}, {clear: point} and {clear_all: true}");
    if (named) c.name = name->as_string();
    return c;
}

ControlReply control_reply(const Frame& f) {
    const Json& m = f.meta;
    ControlReply r;
    r.status = req(m, "status", [](const Json& v) {
        return one_of(v, {CALIPER_NET_STATUS_OK, CALIPER_NET_STATUS_QUEUED, CALIPER_NET_STATUS_ERROR});
    }, "ok, queued or error").as_string();
    r.op = req(m, "op", is_op, "a control op").as_string();
    const Json* err = opt(m, "error", [](const Json& v) {
        const Json* code = v.find("code");
        const Json* msg = v.find("message");
        return v.is_object() && code && code->is_string() && (!present(msg) || is_text(*msg));
    }, "{code, message}");
    if ((err == nullptr) == (r.status == CALIPER_NET_STATUS_ERROR))
        bad("a control_reply carries 'error' exactly when its status is 'error'");
    if (r.status == CALIPER_NET_STATUS_QUEUED && r.op != CALIPER_NET_OP_RUN) bad("only 'run' is queued");
    r.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    if (const Json* v = opt(m, "name", is_name, "a parameter or action name")) r.name = v->as_string();
    if (const Json* v = opt(m, "value", is_set_value, "a number, a boolean or a choice")) r.value = *v;
    if (const Json* v = opt(m, "message", is_text, "a string of at most 1024 bytes")) r.message = v->as_string();
    if (err) r.error = *err;
    return r;
}

bool detail_cell(const Json& v) {
    return v.is_null() || v.is_bool() || is_finite(v) ||
           (v.is_string() && v.as_string().size() <= CALIPER_NET_DETAIL_TEXT_MAX);
}

bool details_obj(const Json& d) {
    if (!d.is_object()) return false;
    const Json* fields = d.find("fields");
    const Json* items = d.find("items");
    const Json* values = d.find("values");
    if (!(fields && fields->is_array() && fields->size() <= CALIPER_NET_MAX_DETAIL_FIELDS)) return false;
    std::set<std::string> seen;
    for (const Json& x : fields->items())
        if (!is_name(x) || !seen.insert(x.as_string()).second) return false;
    if (!(items && items->is_array() && items->size() <= CALIPER_NET_MAX_DETAIL_ITEMS)) return false;
    for (const Json& x : items->items())
        if (!is_uint(x)) return false;
    if (!(values && values->is_array() && values->size() == items->size())) return false;
    for (const Json& row : values->items()) {
        if (!row.is_array() || row.size() != fields->size()) return false;
        for (const Json& c : row.items())
            if (!detail_cell(c)) return false;
    }
    return true;
}

Selection selection(const Frame& f) {
    const Json& m = f.meta;
    Selection s;
    s.space = req(m, "space", is_name, "an item space name").as_string();
    s.items = req(m, "items", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_EVENT_INDICES) return false;
        uint64_t prev = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            uint64_t x = 0;
            if (!v.at(i).get_u64(&x) || (i > 0 && x <= prev)) return false;
            prev = x;
        }
        return true;
    }, "at most 32768 ascending unsigned integers");
    if (const Json* v = opt(m, "view", is_name, "a view name")) s.view = v->as_string();
    if (const Json* v = opt(m, "truncated", is_bool, "a boolean")) s.truncated = v->as_bool();
    if (const Json* v = opt(m, "details", details_obj, "{fields, items, values} within the limits")) s.details = *v;
    if (const Json* v = opt(m, "details_error", is_text, "a string of at most 1024 bytes"))
        s.details_error = v->as_string();
    return s;
}

// ---- message encoders (to_meta) -------------------------------------------------------------

Json stats_json(const std::vector<double>& v) {
    Json a = Json::array();
    for (double x : v) a.push(std::isfinite(x) ? Json(x) : Json());
    return a;
}

Section named(const Section& s, const char* name) {
    Section o = s;
    o.name = name;
    return o;
}

// ---- protocol 3.2 draft: the scenes subset (the 3.2 block of caliper/live_net.h) ----
// The checks and their order follow the block at the end of _wire.py line for
// line, so that the codecs answer every frame with the same code. The public
// functions come first; the decoders and encoders sit in the anonymous
// namespace again, for decode(), to_meta() and encode().
}  // namespace

namespace {
constexpr int kPartCount = 7;
const char* const kParts[kPartCount] = {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES, CALIPER_NET_PART_ARROWS,
                                        CALIPER_NET_PART_BOXES,  CALIPER_NET_PART_MESH,  CALIPER_NET_PART_IMAGE,
                                        CALIPER_NET_PART_CAMERA};
enum { kPoints, kLines, kArrows, kBoxes, kMesh, kImage, kCamera };
// The array sections of each part in the order a sender writes them (PART_SECTIONS).
const char* const kPartSections[kPartCount][8] = {
    {"positions", "colors", "values", "class_ids", "radii", nullptr},
    {"positions", "strip_lengths", "indices", "colors", "values", nullptr},
    {"origins", "vectors", "colors", "values", nullptr},
    {"centers", "half_sizes", "rotations", "colors", "class_ids", nullptr},
    {"positions", "indices", "normals", "colors", "values", "uvs", "texture", nullptr},
    {"pixels", nullptr},
    {nullptr}};
// The keys of each part in the order their types are checked (PART_KEYS).
const char* const kPartKeys[kPartCount][11] = {
    {"color", "radius", "labels", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"mode", "color_per", "color", "width", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"color", "width", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"color", "labels", "fill", "width", "stats", nullptr},
    {"color", "colormap", "flags", "vmin", "vmax", "stats", nullptr},
    {"form", "opacity", "scale", "cloud", "colormap", "flags", "vmin", "vmax", "format", "stats", nullptr},
    {"fx", "fy", "cx", "cy", "width", "height", "convention", "plane_distance", nullptr}};
const char* const kCameraRequired[] = {"fx", "fy", "cx", "cy", "width", "height", "convention", nullptr};

bool is_positive_num(const Json& v) { return is_finite(v) && v.as_double() > 0; }

// A VIEWS scene's "view_camera" as a viewer reads it (_view_camera): null for none, for a 2-D scene and
// for a value it cannot read.
Json view_camera_of(const Json& sc, bool is3d) {
    const Json* vc = sc.find("view_camera");
    if (!vc || vc->is_null() || !is3d) return Json();
    if (vc->is_string()) return !vc->as_string().empty() && entity_path_ok(vc->as_string()) ? *vc : Json();
    if (!vc->is_object()) return Json();
    Json out = Json::object();
    for (const auto& kv : vc->members()) {
        const std::string& k = kv.first;
        const Json& v = kv.second;
        if (k != "target" && k != "distance" && k != "azimuth" && k != "elevation") return Json();
        if (v.is_null()) continue;
        if (k == "target") {
            if (!v.is_array() || v.size() != 3) return Json();
            for (const Json& x : v.items())
                if (!is_finite(x)) return Json();
        } else if (k == "distance") {
            if (!is_positive_num(v)) return Json();
        } else if (k == "azimuth") {
            if (!is_finite(v)) return Json();
        } else if (!is_finite(v) || v.as_double() < -89 || v.as_double() > 89) {
            return Json();
        }
        out.set(k, v);
    }
    return out;
}
}  // namespace

const Section* SceneUpdate::array(const std::string& name) const {
    for (const Section& s : arrays)
        if (s.name == name) return &s;
    return nullptr;
}

int scene_part_index(const std::string& p) {
    for (int i = 0; i < kPartCount; ++i)
        if (p == kParts[i]) return i;
    return -1;
}

const char* const* scene_part_sections(int part) {
    static const char* const none[1] = {nullptr};
    return part >= 0 && part < kPartCount ? kPartSections[part] : none;
}

bool entity_path_ok(const std::string& v) {
    if (v.empty()) return true;
    if (v.size() > CALIPER_NET_PATH_MAX) return false;
    size_t segs = 1, len = 0, start = 0;
    for (size_t i = 0; i <= v.size(); ++i) {
        if (i == v.size() || v[i] == '/') {
            const std::string seg = v.substr(start, i - start);
            if (len < 1 || len > 63 || seg == "." || seg == "..") return false;
            if (i < v.size()) ++segs;
            start = i + 1;
            len = 0;
            continue;
        }
        const unsigned char c = (unsigned char)v[i];
        if (!(std::isalnum(c) || c == '_' || c == '.' || c == '-') || c >= 0x80) return false;
        ++len;
    }
    return segs <= CALIPER_NET_PATH_SEGMENTS_MAX;
}

bool scene_view_info(const Json& entry, SceneView* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    const Json* sc = entry.find("scene");
    const Json empty = Json::object();
    if (!sc || sc->is_null()) sc = &empty;
    if (!sc->is_object()) return fail("'scene' must be an object, got " + show(*sc));
    SceneView v;
    const Json* space = sc->find("space");   // a null key means the same as an absent one
    if (!space || space->is_null()) v.space = CALIPER_NET_SPACE_3D;
    else if (space->is_string() &&
             (space->as_string() == CALIPER_NET_SPACE_3D || space->as_string() == CALIPER_NET_SPACE_2D))
        v.space = space->as_string();
    else return fail("scene space " + show(*space) + "; expected '3d' or '2d'");
    const bool is3d = v.space == CALIPER_NET_SPACE_3D;
    const Json* up = sc->find("up");
    if (!up || up->is_null()) {
        v.up = is3d ? "+y" : "-y";
    } else {
        bool ok = false;
        if (up->is_string()) {
            const std::string& u = up->as_string();
            ok = u == "+y" || u == "-y" || (is3d && (u == "+x" || u == "-x" || u == "+z" || u == "-z"));
        }
        if (!ok) return fail("scene up " + show(*up) + " is not one of the axes of a " + v.space + " scene");
        v.up = up->as_string();
    }
    const Json* units = sc->find("units");
    if (!units || units->is_null()) v.units = "m";
    else if (units->is_string() && units->as_string().size() <= CALIPER_NET_UNITS_MAX) v.units = units->as_string();
    else return fail("scene units must be a string of at most 15 bytes, got " + show(*units));
    v.view_camera = view_camera_of(*sc, is3d);
    if (out) *out = std::move(v);
    return true;
}

Json scene_data_view() {
    Json inc = Json::array();
    inc.push("**");
    Json ent = Json::object();
    ent.set("include", inc);
    ent.set("exclude", Json::array());
    Json b = Json::object();
    b.set("max_points", (uint64_t)CALIPER_NET_DEFAULT_MAX_POINTS);
    b.set("max_items", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_ITEMS);
    b.set("max_vertices", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_VERTICES);
    b.set("max_extent", (uint64_t)CALIPER_NET_SCENE_DEFAULT_MAX_EXTENT);
    Json dv = Json::object();
    dv.set("entities", ent);
    dv.set("budget", b);
    return dv;
}

bool resolve_scene_data_view(const Json& view, const Json& request, Json* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    const Json dv = scene_data_view();
    auto is_default = [&](const Json& v) { return v.is_null() || (v.is_object() && v.size() == 0) || v == dv; };
    if (!request.is_null() && !request.is_object()) return fail("a data view is a JSON object");
    const Json* def = view.find("default");
    if (def && !is_default(*def)) return fail("a scene's own default data view is not built yet");
    if (!is_default(request))
        return fail("only the default data view of a scene is built yet: send {} or none; entity filters and "
                    "budgets come in a later increment");
    if (out) *out = dv;
    return true;
}

namespace {

[[noreturn]] void unsupported(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_UNSUPPORTED, msg}; }

bool is_path(const Json& v) { return v.is_string() && entity_path_ok(v.as_string()); }
bool is_rgba(const Json& v) {
    if (!v.is_array() || v.size() != 4) return false;
    for (const Json& x : v.items()) {
        int64_t i = 0;
        if (!x.get_i64(&i) || i < 0 || i > 255) return false;
    }
    return true;
}
bool is_stats(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_MAX_STATS) return false;
    for (const Json& x : v.items())
        if (!(x.is_null() || x.is_number())) return false;
    return true;
}
bool is_labels(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > CALIPER_NET_MAX_LABELS) return false;
    for (const Json& x : v.items())
        if (!x.is_string() || x.as_string().size() > CALIPER_NET_NAME_MAX) return false;
    return true;
}
bool is_side(const Json& v) {
    int64_t i = 0;
    return v.get_i64(&i) && i >= 1 && i <= (int64_t)CALIPER_NET_MAX_IMAGE_SIDE;
}

// The check of a part key and what it expects, as _KEY_CHECKS (a camera's width is its image width).
bool key_ok(int part, const std::string& key, const Json& v, const char** what) {
    if (key == "color") { *what = "[r, g, b, a] of integers from 0 to 255"; return is_rgba(v); }
    if (part == kCamera && key == "width") { *what = "an integer from 1 to 8192"; return is_side(v); }
    if (key == "radius" || key == "width" || key == "scale" || key == "fx" || key == "fy" ||
        key == "plane_distance") {
        *what = "a positive number";
        return is_positive_num(v);
    }
    if (key == "labels") { *what = "a list of 1 to 4096 strings of at most 63 bytes"; return is_labels(v); }
    if (key == "colormap") { *what = "an integer"; return is_int(v); }
    if (key == "flags") { *what = "an unsigned integer"; return is_uint(v); }
    if (key == "vmin" || key == "vmax") { *what = "a number"; return is_num(v); }
    if (key == "stats") { *what = "at most 64 numbers or nulls"; return is_stats(v); }
    if (key == "mode") {
        *what = "segments, strips or edges";
        return one_of(v, {CALIPER_NET_LINES_SEGMENTS, CALIPER_NET_LINES_STRIPS, CALIPER_NET_LINES_EDGES});
    }
    if (key == "color_per") {
        *what = "vertex, strip or edge";
        return one_of(v, {CALIPER_NET_COLOR_PER_VERTEX, CALIPER_NET_COLOR_PER_STRIP, CALIPER_NET_COLOR_PER_EDGE});
    }
    if (key == "fill" || key == "cloud") { *what = "a boolean"; return v.is_bool(); }
    if (key == "form") {
        *what = "color, depth, segmentation or encoded";
        return one_of(v, {CALIPER_NET_FORM_COLOR, CALIPER_NET_FORM_DEPTH, CALIPER_NET_FORM_SEGMENTATION,
                          CALIPER_NET_FORM_ENCODED});
    }
    if (key == "opacity") {
        *what = "a number from 0 to 1";
        return is_finite(v) && v.as_double() >= 0 && v.as_double() <= 1;
    }
    if (key == "format") { *what = "jpeg or png"; return one_of(v, {CALIPER_NET_FORMAT_JPEG, CALIPER_NET_FORMAT_PNG}); }
    if (key == "cx" || key == "cy") { *what = "a finite number"; return is_finite(v); }
    if (key == "height") { *what = "an integer from 1 to 8192"; return is_side(v); }
    *what = "rdf or rub";   // convention
    return one_of(v, {CALIPER_NET_CONVENTION_RDF, CALIPER_NET_CONVENTION_RUB});
}

Json scene_part_keys(const Json& m, int part) {
    Json out = Json::object();
    for (const char* const* k = kPartKeys[part]; *k; ++k) {
        const Json* v = m.find(*k);
        if (!v || v->is_null()) continue;
        const char* what = "";
        if (!key_ok(part, *k, *v, &what)) bad(std::string("'") + *k + "' must be " + what + ", got " + show(*v));
        if (std::string(*k) == "stats") {
            Json a = Json::array();
            for (const Json& x : v->items()) a.push(x.is_null() ? Json(std::nan("")) : Json(x.as_double()));
            out.set(*k, a);
        } else {
            out.set(*k, *v);
        }
    }
    return out;
}

std::string str_key(const Json& params, const char* key) {
    const Json* v = params.find(key);
    return v ? v->as_string() : std::string();
}

// The section `name` of a part, or nullptr; `shape` holds the rows it must have (-1: any) and its rank.
const Section* part_section(const Frame& f, const char* name, int64_t dtype, std::vector<int64_t> shape) {
    const Section* s = f.section(name);
    if (!s) return nullptr;
    bool ok = s->encoding == CALIPER_NET_ENCODING_RAW && s->dtype == dtype && s->shape.size() == shape.size();
    for (size_t i = 0; ok && i < shape.size(); ++i)
        if (shape[i] >= 0 && shape[i] != s->shape[i]) ok = false;
    if (!ok)
        bad(std::string("'") + name + "' has the wrong dtype or shape (dtype " + std::to_string(s->dtype) + ", rank " +
            std::to_string(s->shape.size()) + ")");
    return s;
}

void required(const Frame& f, const char* part, const char* name) {
    if (!f.section(name)) bad(std::string("a ") + part + " part needs a '" + name + "' section");
}

// A required f32 (N, D) section, D 2 or 3, N at most `limit` (_rows_section).
const Section* rows_section(const Frame& f, const char* part, const char* name, uint64_t limit, const char* what,
                            int64_t* n, int64_t* d) {
    required(f, part, name);
    const Section* s = part_section(f, name, CALIPER_NET_DTYPE_F32, {-1, -1});
    *n = s->shape[0];
    *d = s->shape[1];
    if (*d != 2 && *d != 3) bad(std::string("'") + name + "' is (N, 2) or (N, 3)");
    if (*n > (int64_t)limit)
        bad(std::to_string(*n) + " " + what + ", more than " + std::to_string(limit) + " in a batch");
    return s;
}

bool has_array(const std::vector<Section>& arrays, const char* name) {
    for (const Section& s : arrays)
        if (s.name == name) return true;
    return false;
}

void scene_colours(const Frame& f, int64_t rows, std::vector<Section>* arrays) {
    const Section* c = part_section(f, "colors", CALIPER_NET_DTYPE_U8, {rows, 4});
    const Section* v = part_section(f, "values", CALIPER_NET_DTYPE_F32, {rows});
    if (c && v) bad("'colors' and 'values' together");
    if (c) arrays->push_back(*c);
    if (v) arrays->push_back(*v);
}

void label_count(const Json& params, int64_t n) {
    const Json* l = params.find("labels");
    if (l && l->size() != 1 && (int64_t)l->size() != n)
        bad(std::to_string(l->size()) + " labels for " + std::to_string(n) +
            " items: give one label for all or one per item");
}

void class_ids(const Frame& f, int64_t n, std::vector<Section>* arrays) {
    if (const Section* s = part_section(f, "class_ids", CALIPER_NET_DTYPE_I32, {n})) arrays->push_back(*s);
}

void decode_points(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_POINTS, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    scene_colours(f, n, arrays);
    class_ids(f, n, arrays);
    if (const Section* r = part_section(f, "radii", CALIPER_NET_DTYPE_F32, {n})) arrays->push_back(*r);
    label_count(params, n);
}

void decode_lines(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    if (!params.find("mode")) bad("a lines part needs 'mode'");
    const std::string md = str_key(params, "mode");
    const std::string color_per = str_key(params, "color_per");
    const bool has_cp = params.find("color_per") != nullptr;
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_LINES, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    if (md != CALIPER_NET_LINES_EDGES && f.section("indices")) bad("lines of mode '" + md + "' take no 'indices'");
    int64_t rows = n;
    if (md == CALIPER_NET_LINES_SEGMENTS) {
        if (n % 2) bad("segments need an even number of vertices, got " + std::to_string(n));
        if (f.section("strip_lengths")) bad("segments take no 'strip_lengths'");
        if (has_cp && color_per != CALIPER_NET_COLOR_PER_VERTEX) bad("segments take color_per 'vertex'");
    } else if (md == CALIPER_NET_LINES_STRIPS) {
        const Section* sl = part_section(f, "strip_lengths", CALIPER_NET_DTYPE_I32, {-1});
        if (!sl) bad("strips need a 'strip_lengths' section");
        const int64_t cnt = sl->shape[0];
        if (cnt > (int64_t)CALIPER_NET_MAX_BATCH_ITEMS)
            bad(std::to_string(cnt) + " strips, more than " + std::to_string(CALIPER_NET_MAX_BATCH_ITEMS) +
                " in a batch");
        int64_t total = 0;
        for (int64_t i = 0; i < cnt; ++i) {
            int32_t len;
            std::memcpy(&len, sl->data + 4 * i, 4);
            if (len < 1) bad("every strip length is at least 1");
            total += len;
        }
        if (total != n)
            bad("the strip lengths add up to " + std::to_string(total) + ", not to the " + std::to_string(n) +
                " vertices");
        arrays->push_back(*sl);
        if (color_per == CALIPER_NET_COLOR_PER_EDGE) bad("strips take color_per 'vertex' or 'strip'");
        if (color_per == CALIPER_NET_COLOR_PER_STRIP) rows = cnt;
    } else {
        if (f.section("strip_lengths")) bad("edges take no 'strip_lengths'");
        const Section* idx = part_section(f, "indices", CALIPER_NET_DTYPE_I32, {-1, 2});
        if (!idx) bad("edges need an 'indices' section, (E, 2) i32");
        if (idx->shape[0] > (int64_t)CALIPER_NET_MAX_BATCH_POINTS)
            bad(std::to_string(idx->shape[0]) + " edges, more than " + std::to_string(CALIPER_NET_MAX_BATCH_POINTS) +
                " in a batch");
        arrays->push_back(*idx);
        if (color_per == CALIPER_NET_COLOR_PER_STRIP) bad("edges take color_per 'vertex' or 'edge'");
        if (color_per == CALIPER_NET_COLOR_PER_EDGE) rows = idx->shape[0];
    }
    scene_colours(f, rows, arrays);
}

void decode_arrows(const Frame& f, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(
        *rows_section(f, CALIPER_NET_PART_ARROWS, "origins", CALIPER_NET_MAX_BATCH_ITEMS, "arrows", &n, &d));
    required(f, CALIPER_NET_PART_ARROWS, "vectors");
    arrays->push_back(*part_section(f, "vectors", CALIPER_NET_DTYPE_F32, {n, d}));
    scene_colours(f, n, arrays);
}

void decode_boxes(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(
        *rows_section(f, CALIPER_NET_PART_BOXES, "centers", CALIPER_NET_MAX_BATCH_ITEMS, "boxes", &n, &d));
    required(f, CALIPER_NET_PART_BOXES, "half_sizes");
    arrays->push_back(*part_section(f, "half_sizes", CALIPER_NET_DTYPE_F32, {n, d}));
    if (const Section* r = part_section(f, "rotations", CALIPER_NET_DTYPE_F32, {n, 4})) {
        if (d != 3) bad("'rotations' are for 3-D boxes only");
        arrays->push_back(*r);
    }
    if (const Section* c = part_section(f, "colors", CALIPER_NET_DTYPE_U8, {n, 4})) arrays->push_back(*c);
    class_ids(f, n, arrays);
    label_count(params, n);
}

void decode_mesh(const Frame& f, std::vector<Section>* arrays) {
    int64_t n = 0, d = 0;
    arrays->push_back(*rows_section(f, CALIPER_NET_PART_MESH, "positions", CALIPER_NET_MAX_BATCH_POINTS,
                                    "positions", &n, &d));
    if (const Section* idx = part_section(f, "indices", CALIPER_NET_DTYPE_I32, {-1, 3})) {
        if (idx->shape[0] > (int64_t)CALIPER_NET_MAX_BATCH_POINTS)
            bad(std::to_string(idx->shape[0]) + " triangles, more than " +
                std::to_string(CALIPER_NET_MAX_BATCH_POINTS) + " in a batch");
        arrays->push_back(*idx);
    } else if (n % 3) {
        bad("a mesh without 'indices' needs a multiple of 3 vertices, got " + std::to_string(n));
    }
    if (const Section* nrm = part_section(f, "normals", CALIPER_NET_DTYPE_F32, {n, 3})) arrays->push_back(*nrm);
    scene_colours(f, n, arrays);
    const Section* uv = part_section(f, "uvs", CALIPER_NET_DTYPE_F32, {n, 2});
    const Section* tex = part_section(f, "texture", CALIPER_NET_DTYPE_U8, {-1, -1, 4});
    if (uv && (has_array(*arrays, "colors") || has_array(*arrays, "values")))
        bad("'uvs' and 'colors' or 'values' together");
    if (!uv != !tex) bad("'uvs' and 'texture' go together");
    if (tex) {
        const int64_t side = CALIPER_NET_MAX_IMAGE_SIDE;
        if (tex->shape[0] < 1 || tex->shape[0] > side || tex->shape[1] < 1 || tex->shape[1] > side)
            bad("a texture is 1 to 8192 pixels on a side");
        arrays->push_back(*uv);
        arrays->push_back(*tex);
    }
}

// The pixels each image form takes (_FORM_PIXELS).
bool pixels_fit(const std::string& form, const Section& p) {
    const int64_t dt = p.dtype;
    const size_t r = p.shape.size();
    if (form == CALIPER_NET_FORM_COLOR) return dt == CALIPER_NET_DTYPE_U8 && (r == 3 || r == 2);
    if (form == CALIPER_NET_FORM_DEPTH) return (dt == CALIPER_NET_DTYPE_F32 || dt == CALIPER_NET_DTYPE_U16) && r == 2;
    if (form == CALIPER_NET_FORM_SEGMENTATION)
        return (dt == CALIPER_NET_DTYPE_U16 || dt == CALIPER_NET_DTYPE_U8) && r == 2;
    return dt == CALIPER_NET_DTYPE_U8 && r == 1;   // encoded
}

void decode_image(const Frame& f, const Json& params, std::vector<Section>* arrays) {
    if (!params.find("form")) bad("an image part needs 'form'");
    const std::string form = str_key(params, "form");
    required(f, CALIPER_NET_PART_IMAGE, "pixels");
    const Section* p = f.section("pixels");
    if (p->encoding != CALIPER_NET_ENCODING_RAW || !pixels_fit(form, *p))
        bad("the pixels of a " + form + " image cannot be dtype " + std::to_string(p->dtype) + " of rank " +
            std::to_string(p->shape.size()));
    if (form == CALIPER_NET_FORM_COLOR && p->shape.size() == 3 && p->shape[2] != 4)
        bad("colour pixels are (H, W, 4) or (H, W)");
    if (form == CALIPER_NET_FORM_ENCODED) {
        if (p->shape[0] > (int64_t)CALIPER_NET_MAX_ENCODED_BYTES)
            bad("an encoded frame of " + std::to_string(p->shape[0]) + " bytes, more than 16777216");
        if (!params.find("format")) bad("an encoded image needs 'format'");
    } else if (p->shape[0] > (int64_t)CALIPER_NET_MAX_IMAGE_SIDE || p->shape[1] > (int64_t)CALIPER_NET_MAX_IMAGE_SIDE) {
        bad("an image is at most 8192 pixels on a side");
    }
    arrays->push_back(*p);
}

void decode_camera(const Json& params) {
    for (const char* const* k = kCameraRequired; *k; ++k)
        if (!params.find(*k)) bad(std::string("a camera part needs '") + *k + "'");
}

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

bool is_rows_map(const Json& v) {
    if (!v.is_object()) return false;
    for (const auto& kv : v.members())
        if (!is_uint(kv.second)) return false;
    return true;
}

bool in_part(int part, const std::string& name) {
    for (const char* const* k = scene_part_sections(part); *k; ++k)
        if (name == *k) return true;
    return false;
}

// The checks of a piece of a split update (_piece), in this order: the keys of `split` (update, pieces,
// piece, first, then rows in piece 0 only), the part's keys in piece 0, and that `first` names exactly
// the piece's array sections, each raw with at least one dimension, within `rows` in piece 0.
void scene_piece(const Frame& f, const Json& m, int part, SceneUpdate& u) {
    const Json& sp = u.split;
    auto need = [&](const char* key, auto ok, const char* what) -> const Json& {
        const Json* v = sp.find(key);
        if (!v || v->is_null() || !ok(*v)) bad(std::string("'split.") + key + "' must be " + what);
        return *v;
    };
    need("update", is_uint, "an unsigned integer");
    int64_t pieces = 0, piece = 0;
    need("pieces", [&](const Json& v) {
        return v.is_int() && v.get_i64(&pieces) && pieces >= 2 && pieces <= (int64_t)CALIPER_NET_MAX_PIECES;
    }, "an integer from 2 to 16");
    need("piece", [&](const Json& v) { return v.is_int() && v.get_i64(&piece) && piece >= 0 && piece < pieces; },
         "a piece number below 'pieces'");
    const Json& first = need("first", is_rows_map, "an object of row numbers");
    const Json* rows = nullptr;
    if (piece == 0) {
        rows = &need("rows", is_rows_map, "an object of row counts");
    } else if (const Json* r = sp.find("rows"); r && !r->is_null()) {
        bad("only piece 0 carries 'split.rows'");
    }
    if (piece == 0) u.params = scene_part_keys(m, part);
    for (const Section& s : f.sections) {
        if (s.name == "stamp" || !in_part(part, s.name)) continue;   // the stamp, or a section of no array
        if (s.encoding != CALIPER_NET_ENCODING_RAW || s.shape.empty())
            bad("a piece of '" + s.name + "' is a raw array of at least one dimension");
        u.arrays.push_back(s);
    }
    bool same = first.size() == u.arrays.size();
    for (const Section& s : u.arrays) same = same && first.find(s.name) != nullptr;
    if (!same) bad("'split.first' does not name exactly the arrays of the piece");
    if (rows) {
        for (const auto& kv : rows->members())
            if (!in_part(part, kv.first)) bad("'split.rows' names '" + kv.first + "', not an array of the part");
        // Unsigned, with no sum that can wrap: a first row near 2^64 must not come back below the count.
        for (const Section& s : u.arrays) {
            const Json* r = rows->find(s.name);
            const uint64_t f0 = u64(*first.find(s.name)), n = (uint64_t)s.shape[0];
            if (!r || s.shape[0] < 0 || f0 > u64(*r) || n > u64(*r) - f0)
                bad("the rows of '" + s.name + "' in piece 0 pass 'split.rows'");
        }
    }
}

template <typename M>
void scene_common(const Json& m, M& out) {
    out.sub = u32(req(m, "sub", is_u32, "a uint32"));
    out.view = req(m, "view", is_name, "a view name").as_string();
    out.seq = u64(req(m, "seq", is_uint, "an unsigned integer"));
    out.stamp_ns = u64(req(m, "stamp_ns", is_uint, "an unsigned integer"));
}

template <typename M>
void scene_flags(const Json& m, M& out) {
    if (const Json* v = opt(m, "static", is_bool, "a boolean")) out.is_static = v->as_bool();
    if (const Json* v = opt(m, "state", is_bool, "a boolean")) out.state = v->as_bool();
    if (const Json* v = opt(m, "cursor", is_u32, "a uint32")) out.cursor = u32(*v);
}

SceneUpdate scene_update(const Frame& f) {
    const Json& m = f.meta;
    SceneUpdate u;
    scene_common(m, u);
    u.path = req(m, "path", is_path, "an entity path").as_string();
    u.part = req(m, "part", is_str, "a string").as_string();
    const int part = scene_part_index(u.part);
    if (part < 0) bad("unknown part '" + u.part + "'");
    scene_flags(m, u);
    if (const Json* v = opt(m, "axes", is_list, "a list")) u.axes = *v;
    if (const Json* v = opt(m, "items", is_name, "an item space name")) u.items = v->as_string();
    if (const Json* sp = opt(m, "split", is_obj, "an object")) {
        u.split = *sp;
        scene_piece(f, m, part, u);
    } else {
    u.params = scene_part_keys(m, part);
    switch (part) {
        case kPoints: decode_points(f, u.params, &u.arrays); break;
        case kLines: decode_lines(f, u.params, &u.arrays); break;
        case kArrows: decode_arrows(f, &u.arrays); break;
        case kBoxes: decode_boxes(f, u.params, &u.arrays); break;
        case kMesh: decode_mesh(f, &u.arrays); break;
        case kImage: decode_image(f, u.params, &u.arrays); break;
        default: decode_camera(u.params); break;
    }
    }
    scene_stamp(f, u.is_static, -1, &u.stamp);
    return u;
}

bool is_transform_paths(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > CALIPER_NET_MAX_TRANSFORMS) return false;
    for (const Json& p : v.items())
        if (!p.is_string() || p.as_string().empty() || !entity_path_ok(p.as_string())) return false;
    return true;
}

SceneTransforms scene_transforms(const Frame& f) {
    const Json& m = f.meta;
    SceneTransforms t;
    scene_common(m, t);
    scene_flags(m, t);
    t.paths = strs(req(m, "paths", is_transform_paths, "a list of 1 to 1024 entity paths other than the root"));
    if (std::set<std::string>(t.paths.begin(), t.paths.end()).size() != t.paths.size())
        bad("'paths' names an entity twice");
    if (!f.section("matrices")) bad("a scene_transforms needs a 'matrices' section");
    t.matrices = *part_section(f, "matrices", CALIPER_NET_DTYPE_F32, {(int64_t)t.paths.size(), 16});
    scene_stamp(f, t.is_static, (int64_t)t.paths.size(), &t.stamp, &t.stamps);
    return t;
}

bool is_clear_part(const Json& v) {
    return one_of(v, {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES, CALIPER_NET_PART_ARROWS, CALIPER_NET_PART_BOXES,
                      CALIPER_NET_PART_MESH, CALIPER_NET_PART_IMAGE, CALIPER_NET_PART_CAMERA,
                      CALIPER_NET_PART_TRANSFORM});
}

SceneClear scene_clear(const Frame& f) {
    const Json& m = f.meta;
    SceneClear c;
    scene_common(m, c);
    c.path = req(m, "path", is_path, "an entity path").as_string();
    if (const Json* v = opt(m, "recursive", is_bool, "a boolean")) c.recursive = v->as_bool();
    if (const Json* v = opt(m, "parts", [](const Json& x) {
            if (!x.is_array() || x.size() == 0) return false;
            for (const Json& e : x.items())
                if (!is_clear_part(e)) return false;
            return true;
        }, "a list of parts"))
        c.parts = strs(*v);
    if (const Json* v = opt(m, "reset", is_bool, "a boolean")) c.reset = v->as_bool();
    scene_flags(m, c);
    if (c.reset && !c.path.empty()) bad("a reset clears the root: its 'path' is \"\"");
    scene_stamp(f, c.is_static, -1, &c.stamp);
    return c;
}

bool is_class_entry(const Json& e) {
    if (!e.is_object()) return false;
    const Json* id = e.find("id");
    const Json* label = e.find("label");
    const Json* color = e.find("color");
    int64_t i = 0;
    return id && id->is_int() && id->get_i64(&i) && i >= 0 && i <= (int64_t)CALIPER_NET_MAX_CLASS_ID && label &&
           label->is_string() && label->as_string().size() <= CALIPER_NET_NAME_MAX && color && is_rgba(*color);
}

SceneClasses scene_classes(const Frame& f) {
    const Json& m = f.meta;
    SceneClasses k;
    scene_common(m, k);
    scene_flags(m, k);
    k.classes = req(m, "classes", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_CLASSES) return false;
        for (const Json& e : v.items())
            if (!is_class_entry(e)) return false;
        return true;
    }, "a list of at most 4096 {id, label, color} entries");
    int64_t prev = -1;
    for (const Json& e : k.classes.items()) {
        int64_t i = 0;
        e.find("id")->get_i64(&i);
        if (i <= prev) bad("'classes' must be in ascending id order without repeats");
        prev = i;
    }
    scene_stamp(f, k.is_static, -1, &k.stamp);
    return k;
}

template <typename M>
void scene_write_flags(Json& m, const M& x) {
    if (x.is_static) m.set("static", true);
    if (x.state) m.set("state", true);
    if (x.cursor) m.set("cursor", *x.cursor);
}

template <typename M>
void scene_write_common(Json& m, const M& x) {
    m.set("sub", x.sub);
    m.set("view", x.view);
    m.set("seq", x.seq);
    m.set("stamp_ns", x.stamp_ns);
}

// The metadata and sections that SceneUpdate.to_meta() and to_sections() write.
void scene_update_to_meta(const SceneUpdate& u, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, u);
    m.set("path", u.path);
    m.set("part", u.part);
    const bool camera = u.part == CALIPER_NET_PART_CAMERA;
    for (const auto& kv : u.params.members()) {
        const std::string& k = kv.first;
        const Json& v = kv.second;
        if (v.is_null()) continue;
        if (k == "stats") {
            Json a = Json::array();
            for (const Json& x : v.items())
                a.push(x.is_number() && std::isfinite(x.as_double()) ? Json(x.as_double()) : Json());
            m.set("stats", a);
        } else if (camera && (k == "width" || k == "height")) {
            m.set(k, v);
        } else if (v.is_number() && (k == "vmin" || k == "vmax" || k == "radius" || k == "width" || k == "opacity" ||
                                     k == "scale" || k == "fx" || k == "fy" || k == "cx" || k == "cy" ||
                                     k == "plane_distance")) {
            m.set(k, v.as_double());
        } else {
            m.set(k, v);
        }
    }
    scene_write_flags(m, u);
    if (!u.axes.is_null()) m.set("axes", u.axes);
    if (u.items) m.set("items", *u.items);
    if (!u.split.is_null()) m.set("split", u.split);
    for (const char* const* k = scene_part_sections(scene_part_index(u.part)); *k; ++k)
        if (const Section* s = u.array(*k)) sections.push_back(*s);
    if (u.stamp) sections.push_back(named(*u.stamp, "stamp"));
}

void scene_transforms_to_meta(const SceneTransforms& t, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, t);
    m.set("paths", str_list(t.paths));
    scene_write_flags(m, t);
    sections.push_back(named(t.matrices, "matrices"));
    if (t.stamps) sections.push_back(named(*t.stamps, "stamps"));
    else if (t.stamp) sections.push_back(named(*t.stamp, "stamp"));
}

void scene_clear_to_meta(const SceneClear& c, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, c);
    m.set("path", c.path);
    if (c.recursive) m.set("recursive", true);
    if (c.reset) m.set("reset", true);
    scene_write_flags(m, c);
    if (c.parts) m.set("parts", str_list(*c.parts));
    if (c.stamp) sections.push_back(named(*c.stamp, "stamp"));
}

void scene_classes_to_meta(const SceneClasses& k, Json& m, std::vector<Section>& sections) {
    scene_write_common(m, k);
    m.set("classes", k.classes);
    scene_write_flags(m, k);
    if (k.stamp) sections.push_back(named(*k.stamp, "stamp"));
}

// ---- timelines, text logs, bar charts and the recording messages ----

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

// Strict UTF-8, as Python's decode("utf-8") and a fatal TextDecoder read it.
bool utf8_ok(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n) {
        const uint8_t c = p[i];
        if (c < 0x80) { ++i; continue; }
        size_t len = 0;
        uint32_t cp = 0, min = 0;
        if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
        else return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += len;
    }
    return true;
}

Text text(const Frame& f) {
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
        if (offs[(size_t)i + 1] - offs[(size_t)i] > (int32_t)CALIPER_NET_TEXT_LINE_MAX) bad("a line is at most 4096 bytes");
    for (int64_t i = 0; i < n; ++i)
        if (lv->data[i] > CALIPER_NET_LEVEL_ERROR) bad("a level is 0 (debug), 1 (info), 2 (warning) or 3 (error)");
    for (int64_t i = 0; i < n; ++i)
        if (!utf8_ok(t->data + offs[(size_t)i], (size_t)(offs[(size_t)i + 1] - offs[(size_t)i])))
            bad("a line of 'text' is not UTF-8");
    scene_stamp(f, false, n, &x.stamp, &x.stamps);
    return x;
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

Recordings recordings(const Frame& f) {
    Recordings r;
    r.recordings = req(f.meta, "recordings", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_MAX_RECORDINGS) return false;
        for (const Json& e : v.items())
            if (!is_recording_entry(e)) return false;
        return true;
    }, "a list of at most 1024 recordings");
    return r;
}

RecordingOpen recording_open(const Frame& f) {
    RecordingOpen r;
    r.request_id = u32(req(f.meta, "request_id", is_u32, "a uint32"));
    r.recording = req(f.meta, "recording", is_session_id, "a recording id").as_string();
    return r;
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

[[noreturn]] void bad_cursor(const std::string& msg) { throw MsgError{CALIPER_NET_ERR_BAD_CURSOR, msg}; }

Cursor cursor(const Frame& f) {
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

bool is_ranges(const Json& v) {
    if (!v.is_object()) return false;
    for (const auto& kv : v.members())
        if (!is_timeline_name(Json(kv.first)) || !is_loop(kv.second)) return false;
    return true;
}

TimeState time_state(const Frame& f) {
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

// ---- 3.2 draft: GPU programs (the block "GPU programs and interventions" of _wire.py) ----------

constexpr const char* kOutputGeometry = "geometry";
constexpr const char* kOutputPixels = "pixels";
constexpr const char* kOutputTensor = "tensor";
constexpr size_t kParamsMax = 16;          // names in a params object
constexpr size_t kParamNameMax = 63;       // bytes of a params name
constexpr size_t kBuiltinMax = 32;         // bytes of a declaration's "builtin"
constexpr uint64_t kTensorElementsMax = 1000000;

bool lower(char c) { return c >= 'a' && c <= 'z'; }
bool digit(char c) { return c >= '0' && c <= '9'; }
// The operator form "operator.<spec id>.<n>": a spec id (a lower-case letter, then lower-case letters, digits, _ or -),
// a dot, and 1 to 5 decimal digits.
bool operator_id_ok(const std::string& s) {
    static const std::string kPrefix = "operator.";
    if (s.size() <= kPrefix.size() || s.compare(0, kPrefix.size(), kPrefix) != 0) return false;
    const size_t dot = s.rfind('.');
    if (dot <= kPrefix.size()) return false;   // no spec id
    const size_t n = s.size() - dot - 1;
    if (n < 1 || n > 5) return false;
    for (size_t i = dot + 1; i < s.size(); ++i)
        if (!digit(s[i])) return false;
    if (!lower(s[kPrefix.size()])) return false;
    for (size_t i = kPrefix.size(); i < dot; ++i)
        if (!(lower(s[i]) || digit(s[i]) || s[i] == '_' || s[i] == '-')) return false;
    return true;
}
bool program_id_ok(const Json* v) {
    if (!v || !v->is_string()) return false;
    const std::string& s = v->as_string();
    if (s.empty() || s.size() > CALIPER_NET_PROGRAM_ID_MAX) return false;
    if (s.compare(0, 9, "operator.") == 0) return operator_id_ok(s);
    if (!lower(s[0])) return false;
    for (char c : s)
        if (!(lower(c) || digit(c) || c == '_' || c == '-')) return false;
    return true;
}
// A params name: a WGSL field name of 1 to 63 bytes, [A-Za-z_][A-Za-z0-9_]*, not starting with "__".
bool param_name_ok(const std::string& s) {
    if (s.empty() || s.size() > kParamNameMax) return false;
    if (!(lower(s[0]) || (s[0] >= 'A' && s[0] <= 'Z') || s[0] == '_')) return false;
    if (s.size() >= 2 && s[0] == '_' && s[1] == '_') return false;
    for (char c : s)
        if (!(lower(c) || (c >= 'A' && c <= 'Z') || digit(c) || c == '_')) return false;
    return true;
}
// 1 to 16 names, each a valid name with a finite number.
bool params_ok(const Json& v) {
    if (!v.is_object() || v.size() < 1 || v.size() > kParamsMax) return false;
    for (const auto& kv : v.members())
        if (!param_name_ok(kv.first) || !is_finite(kv.second)) return false;
    return true;
}
bool tensor_shape_ok(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > 4) return false;
    uint64_t prod = 1;
    for (const Json& e : v.items()) {
        if (!int_in(e, 1, kTensorElementsMax)) return false;
        prod *= e.as_double() > 0 ? (uint64_t)e.as_double() : 1;
        if (prod > kTensorElementsMax) return false;
    }
    return true;
}
bool sha256_hex_ok(const Json* v) {
    if (!v || !v->is_string() || v->as_string().size() != 64) return false;
    for (char c : v->as_string())
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
// The optional key `key` of `o` (absent or null passes) when `check` holds for it.
template <typename Check>
bool opt_ok(const Json& o, const char* key, Check check) {
    const Json* v = o.find(key);
    return !present(v) || check(*v);
}
bool arrays_ok(const Json& v) {
    static const char* const kArrays[] = {"positions", "colors", "values", "radii"};
    if (!v.is_array() || v.size() < 1 || v.size() > 4) return false;
    std::set<std::string> seen;
    for (const Json& x : v.items()) {
        if (!x.is_string()) return false;
        bool known = false;
        for (const char* a : kArrays) known = known || x.as_string() == a;
        if (!known || !seen.insert(x.as_string()).second) return false;
    }
    return seen.count("positions") == 1;
}
bool input_entry(const Json& v) {
    return v.is_object() && name_ok(v.find("var")) && bounded_text(v.find("source"), CALIPER_NET_TENSOR_NAME_MAX);
}

// _check_declaration: output, then inputs, limits, workgroups and entry.
void check_declaration(const Json& d) {
    const Json& out = req(d, "output", is_obj, "an object");
    const std::string kind = req(out, "kind", [](const Json& v) {
        return one_of(v, {kOutputGeometry, kOutputPixels, kOutputTensor});
    }, "geometry, pixels or tensor").as_string();
    if (kind == kOutputTensor) {
        req(out, "shape", tensor_shape_ok, "1 to 4 integers of at least 1 with at most 1000000 elements");
        for (const auto& kv : out.members())
            if (kv.first != "kind" && kv.first != "shape") bad("a tensor output has only 'kind' and 'shape'");
    } else if (kind == kOutputGeometry) {
        req(out, "part", [](const Json& v) { return one_of(v, {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES}); },
            "points or lines");
        req(out, "max_count", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_MAX_COUNT); },
            "an integer from 1 to 1000000");
        req(out, "arrays", arrays_ok,
            "unique names of positions, colors, values and radii, with positions");
        opt(out, "entity", [](const Json& v) { return v.is_string() && entity_path_ok(v.as_string()); },
            "an entity path");
    } else {
        auto side = [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_SIDE_MAX); };
        req(out, "width", side, "an integer from 1 to 2048");
        req(out, "height", side, "an integer from 1 to 2048");
        opt(out, "format", [](const Json& v) { return one_of(v, {"rgba8unorm"}); }, "rgba8unorm");
    }
    opt(d, "inputs", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_PROGRAM_MAX_INPUTS) return false;
        for (const Json& e : v.items())
            if (!input_entry(e)) return false;
        return true;
    }, "at most 8 inputs {var, source}");
    if (const Json* lim = opt(d, "limits", is_obj, "an object")) {
        opt(*lim, "time_ms", [](const Json& v) {
            return is_finite(v) && v.as_double() > 0 && v.as_double() <= CALIPER_NET_PROGRAM_TIME_MS_MAX;
        }, "a number above 0 and at most 50");
        opt(*lim, "memory_mb", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_MEMORY_MB_MAX); },
            "an integer from 1 to 512");
        opt(*lim, "every", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_EVERY_MAX); },
            "an integer from 1 to 10000");
    }
    opt(d, "workgroups", [](const Json& v) {
        if (!v.is_array() || v.size() != 3) return false;
        for (const Json& e : v.items())
            if (!int_in(e, 1, CALIPER_NET_PROGRAM_WORKGROUPS_MAX)) return false;
        return true;
    }, "three integers from 1 to 65535");
    opt(d, "entry", is_name, "an entry point name");
    opt(d, "params", params_ok, "1 to 16 names of finite numbers");
    opt(d, "builtin", [](const Json& v) { return bounded_text(&v, kBuiltinMax); }, "a string of 1 to 32 bytes");
}

// ---- 3.2 draft: instances of pieces and their resources (capability "resources") ----------------------
// The pin of a piece: name [a-z][a-z0-9_]* of 1 to 32 bytes, version 1 to 1,000,000, and hash "sha256:" with
// 64 lowercase hex digits.
bool piece_pin_ok(const Json& v) {
    if (!v.is_object()) return false;
    const Json* name = v.find("name");
    if (!name || !name->is_string()) return false;
    const std::string& n = name->as_string();
    if (n.empty() || n.size() > CALIPER_NET_PIECE_NAME_MAX || !lower(n[0])) return false;
    for (char c : n)
        if (!(lower(c) || digit(c) || c == '_')) return false;
    const Json* version = v.find("version");
    if (!version || !int_in(*version, 1, CALIPER_NET_PIECE_VERSION_MAX)) return false;
    const Json* hash = v.find("hash");
    if (!hash || !hash->is_string() || hash->as_string().compare(0, 7, "sha256:") != 0) return false;
    const Json hex(hash->as_string().substr(7));
    return sha256_hex_ok(&hex);
}
// One value of "bind": live:<a tensor name of 1 to 255 bytes>, or program:<program id>/<resource name>.
bool binding_ok(const Json& v) {
    if (!v.is_string()) return false;
    const std::string& s = v.as_string();
    if (s.compare(0, 5, "live:") == 0) return s.size() > 5 && s.size() - 5 <= CALIPER_NET_TENSOR_NAME_MAX;
    if (s.compare(0, 8, "program:") != 0) return false;
    const size_t slash = s.find('/', 8);
    if (slash == std::string::npos) return false;
    const Json id(s.substr(8, slash - 8));
    return program_id_ok(&id) && param_name_ok(s.substr(slash + 1));
}
bool bind_ok(const Json& v) {
    if (!v.is_object() || v.size() > CALIPER_NET_PROGRAM_MAX_INPUTS) return false;
    for (const auto& kv : v.members())
        if (!param_name_ok(kv.first) || !binding_ok(kv.second)) return false;
    return true;
}
bool segments_ok(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_PIECE_SEGMENTS_MAX) return false;
    for (const Json& x : v.items()) {
        if (!x.is_object() || !name_ok(x.find("part"))) return false;
        const Json* first = x.find("first_line");
        const Json* lines = x.find("lines");
        if (!first || !int_in(*first, 1, 0xffffffffull) || !lines || !is_u32(*lines)) return false;
    }
    return true;
}
// An optional (B,) u8 section of 1 to `most` bytes of UTF-8.
std::optional<std::string> text_section(const Frame& f, const char* name, uint64_t most) {
    const Section* s = f.section(name);
    if (!s) return std::nullopt;
    if (!s->is(CALIPER_NET_DTYPE_U8, 1)) bad(std::string("'") + name + "' must be a (B,) u8 section");
    if (s->shape[0] < 1 || (uint64_t)s->shape[0] > most)
        bad(std::string("'") + name + "' holds 1 to " + std::to_string(most) + " bytes, got " +
            std::to_string(s->shape[0]));
    if (!utf8_ok(s->data, (size_t)s->shape[0])) bad(std::string("'") + name + "' is not UTF-8");
    return std::string(reinterpret_cast<const char*>(s->data), (size_t)s->shape[0]);
}
// The set of an instance of a piece, in _wire.py's order: no key or section of the one-pass form, piece,
// manifest, bind, params, panel, segments, then the sections common, derive and draw.
void program_instance(const Frame& f, Program* p) {
    const Json& m = f.meta;
    if (m.find("declaration") || m.find("scene") || m.find("sha256") || f.section(CALIPER_NET_SECTION_WGSL))
        bad("an instance of a piece carries no 'declaration', 'scene', 'sha256' or 'wgsl'");
    p->piece = req(m, "piece", piece_pin_ok, "a pin {name, version, hash}");
    p->manifest = req(m, "manifest", is_obj, "an object");
    if (const Json* v = opt(m, "bind", bind_ok, "at most 8 inputs bound to live:<tensor> or program:<id>/<resource>"))
        p->bind = *v;
    if (const Json* v = opt(m, "params", params_ok, "1 to 16 names (WGSL field names) of finite numbers"))
        p->params = *v;
    if (const Json* v = opt(m, "panel", is_name, "a panel id")) p->panel = v->as_string();
    if (const Json* v = opt(m, "segments", segments_ok, "at most 16 segments {part, first_line, lines}"))
        p->segments = *v;
    p->common = text_section(f, CALIPER_NET_SECTION_COMMON, CALIPER_NET_PIECE_COMMON_MAX);
    p->derive = text_section(f, CALIPER_NET_SECTION_DERIVE, CALIPER_NET_PROGRAM_SOURCE_MAX);
    p->draw = text_section(f, CALIPER_NET_SECTION_DRAW, CALIPER_NET_PROGRAM_SOURCE_MAX);
    if (!p->derive && !p->draw) bad("an instance of a piece carries the section 'derive' or 'draw'");
}

// RESOURCE, in _wire.py's order: program, resource, step, count, stride, width and height, the byte limit
// on count before any product, then the section. Each product below is of values already bounded, thus
// none wraps: width and height are at most 4096, and count is at most RESOURCE_MAX_BYTES / stride.
Resource resource(const Frame& f) {
    const Json& m = f.meta;
    Resource r;
    r.program = req(m, "program", [](const Json& v) { return program_id_ok(&v); },
                    "a program id of at most 55 bytes").as_string();
    r.resource = req(m, "resource", [](const Json& v) { return v.is_string() && param_name_ok(v.as_string()); },
                     "a WGSL name of 1 to 63 bytes").as_string();
    r.step = u64(req(m, "step", is_safe_uint, "an unsigned integer"));
    r.count = u32(req(m, "count", is_u32, "a uint32"));
    r.stride = u32(req(m, "stride", [](const Json& v) {
                           return int_in(v, 4, CALIPER_NET_RESOURCE_STRIDE_MAX) && u64(v) % 4 == 0;
                       }, "a multiple of 4 from 4 to 256"));
    auto side = [](const Json& v) { return int_in(v, 1, CALIPER_NET_RESOURCE_SIDE_MAX); };
    const Json* width = opt(m, "width", side, "an integer from 1 to 4096");
    const Json* height = opt(m, "height", side, "an integer from 1 to 4096");
    if (!width != !height) bad("a texture holds both 'width' and 'height'");
    if (width) {
        if (r.stride != 4 || (uint64_t)r.count != u64(*width) * u64(*height))
            bad("a texture has a stride of 4 and width * height elements");
        r.width = u32(*width);
        r.height = u32(*height);
    }
    if (r.count > CALIPER_NET_RESOURCE_MAX_BYTES / r.stride)
        bad("a resource holds at most " + std::to_string(CALIPER_NET_RESOURCE_MAX_BYTES) + " bytes");
    const Section* s = f.section(CALIPER_NET_SECTION_RESOURCE);
    if (!s || !s->is(CALIPER_NET_DTYPE_U8, 1)) bad("a resource needs a (B,) u8 'resource' section");
    const uint64_t want = (uint64_t)r.count * r.stride;
    if (s->shape[0] < 0 || (uint64_t)s->shape[0] != want)
        bad("'resource' holds count * stride = " + std::to_string(want) + " bytes, got " + std::to_string(s->shape[0]));
    r.data = *s;
    return r;
}

void resource_to_meta(const Resource& r, Json& m, std::vector<Section>& sections) {
    m.set("program", r.program);
    m.set("resource", r.resource);
    m.set("step", r.step);
    m.set("count", r.count);
    m.set("stride", r.stride);
    if (r.width) m.set("width", *r.width);
    if (r.height) m.set("height", *r.height);
    sections.push_back(Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {(int64_t)r.data.size},
                                     r.data.data, r.data.size));
}

Program program(const Frame& f) {
    const Json& m = f.meta;
    Program p;
    p.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    p.op = req(m, "op", [](const Json& v) {
                   return one_of(v, {CALIPER_NET_PROGRAM_SET, CALIPER_NET_PROGRAM_REMOVE, CALIPER_NET_PROGRAM_PARAMS});
               }, "set, remove or params").as_string();
    p.id = req(m, "id", [](const Json& v) { return program_id_ok(&v); },
               "a program id of at most 55 bytes").as_string();
    if (p.op == CALIPER_NET_PROGRAM_REMOVE) return p;
    if (p.op == CALIPER_NET_PROGRAM_PARAMS) {
        if (m.find("declaration")) bad("a params op carries no 'declaration'");
        if (!f.sections.empty()) bad("a params op carries no section");
        p.params = req(m, "params", params_ok, "1 to 16 names (WGSL field names) of finite numbers");
        return p;
    }
    if (m.find("piece")) {
        program_instance(f, &p);
        return p;
    }
    const Json& d = req(m, "declaration", is_obj, "an object");
    check_declaration(d);
    const bool geometry = d.find("output")->find("kind")->as_string() == kOutputGeometry;
    const Json* scene = geometry ? &req(m, "scene", is_name, "a scene name") : opt(m, "scene", is_name, "a scene name");
    const Json& sha = req(m, "sha256", [](const Json& v) { return sha256_hex_ok(&v); }, "64 lowercase hex digits");
    const Section* w = f.section(CALIPER_NET_SECTION_WGSL);
    if (!w || !w->is(CALIPER_NET_DTYPE_U8, 1)) bad("a program set needs a (B,) u8 'wgsl' section");
    if (w->shape[0] < 1 || w->shape[0] > (int64_t)CALIPER_NET_PROGRAM_SOURCE_MAX)
        bad("'wgsl' holds 1 to " + std::to_string(CALIPER_NET_PROGRAM_SOURCE_MAX) + " bytes, got " +
            std::to_string(w->shape[0]));
    if (!utf8_ok(w->data, (size_t)w->shape[0])) bad("'wgsl' is not UTF-8");
    p.declaration = d;
    p.wgsl = std::string(reinterpret_cast<const char*>(w->data), (size_t)w->shape[0]);
    p.sha256 = sha.as_string();
    if (scene) p.scene = scene->as_string();
    return p;
}

bool program_error_ok(const Json& v) {
    if (!v.is_object() || !name_ok(v.find("code"))) return false;
    const Json* msg = v.find("message");
    if (!(msg && is_text(*msg))) return false;
    return opt_ok(v, "line", is_u32) && opt_ok(v, "column", is_u32);
}

bool status_item(const Json& v) {
    if (!v.is_object() || !program_id_ok(v.find("id"))) return false;
    const Json* kind = v.find("kind");
    const Json* state = v.find("state");
    // A status item's kind: the output's kind, or "operator" for the program of a built-in operator.
    if (!(kind && one_of(*kind, {kOutputGeometry, kOutputPixels, kOutputTensor, "operator", CALIPER_NET_STATUS_PIECE})))
        return false;
    if (!(state && one_of(*state, {"waiting", "review", "compiled", "running", "refused", "stopped"}))) return false;
    auto every = [](const Json& x) { return int_in(x, 1, CALIPER_NET_PROGRAM_EVERY_MAX); };
    if (!opt_ok(v, "every", every) || !opt_ok(v, "every_min", every) || !opt_ok(v, "last_pass_ms", is_nonneg) ||
        !opt_ok(v, "last_cost_ms", is_nonneg) || !opt_ok(v, "step_ms", is_nonneg) ||
        !opt_ok(v, "overhead_share", is_nonneg) || !opt_ok(v, "passes_per_s", is_nonneg) ||
        !opt_ok(v, "last_step", is_safe_uint) || !opt_ok(v, "runs", is_safe_uint) ||
        !opt_ok(v, "memory_bytes", is_safe_uint) || !opt_ok(v, "memory_limit_bytes", is_safe_uint) ||
        !opt_ok(v, "error", program_error_ok))
        return false;
    const Json* sha = v.find("sha256");
    return !present(sha) || (sha->is_string() && sha->as_string().empty()) || sha256_hex_ok(sha);
}

bool status_items(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_MAX_PROGRAMS) return false;
    std::set<std::string> ids;
    for (const Json& x : v.items()) {
        if (!status_item(x)) return false;
        ids.insert(x.find("id")->as_string());
    }
    return ids.size() == v.size();
}

ProgramState program_state(const Frame& f) {
    const Json& m = f.meta;
    ProgramState s;
    if (const Json* v = opt(m, "request_id", is_u32, "a uint32")) s.request_id = u32(*v);
    if (const Json* v = opt(m, "step", is_safe_uint, "an unsigned integer")) s.step = u64(*v);
    if (const Json* v = opt(m, "step_ms", is_nonneg, "a number of at least 0")) s.step_ms = v->as_double();
    s.programs = req(m, "programs", status_items, "at most 16 status items with unique program ids");
    return s;
}

void program_to_meta(const Program& p, Json& m, std::vector<Section>& sections) {
    m.set("request_id", p.request_id);
    m.set("op", p.op);
    m.set("id", p.id);
    if (!p.declaration.is_null()) m.set("declaration", p.declaration);
    if (!p.params.is_null()) m.set("params", p.params);
    if (p.scene) m.set("scene", *p.scene);
    if (p.sha256) m.set("sha256", *p.sha256);
    if (!p.piece.is_null()) m.set("piece", p.piece);
    if (!p.manifest.is_null()) m.set("manifest", p.manifest);
    if (!p.bind.is_null()) m.set("bind", p.bind);
    if (p.panel) m.set("panel", *p.panel);
    if (!p.segments.is_null()) m.set("segments", p.segments);
    // The sections in _wire.py's order: wgsl, common, derive, draw.
    const std::pair<const char*, const std::optional<std::string>*> texts[] = {
        {CALIPER_NET_SECTION_WGSL, &p.wgsl},
        {CALIPER_NET_SECTION_COMMON, &p.common},
        {CALIPER_NET_SECTION_DERIVE, &p.derive},
        {CALIPER_NET_SECTION_DRAW, &p.draw}};
    for (const auto& [name, text] : texts)
        if (*text)
            sections.push_back(Section::make(name, CALIPER_NET_DTYPE_U8, {(int64_t)(*text)->size()}, (*text)->data(),
                                             (*text)->size()));
}

void program_state_to_meta(const ProgramState& s, Json& m) {
    if (s.request_id) m.set("request_id", *s.request_id);
    if (s.step) m.set("step", *s.step);
    if (s.step_ms) m.set("step_ms", *s.step_ms);
    m.set("programs", s.programs);
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
// ---- end of the protocol 3.2 draft block ----

struct Encoded {
    uint16_t type = 0;
    Json meta = Json::object();
    std::vector<Section> sections;
};

Encoded to_meta(const Message& msg) {
    Encoded e;
    e.type = message_type(msg);
    Json& m = e.meta;
    if (const auto* h = std::get_if<Hello>(&msg)) {
        m.set("role", h->role);
        m.set("capabilities", str_list(h->capabilities));
        if (!h->software.is_null()) m.set("software", h->software);
        if (!h->session.is_null()) m.set("session", h->session);
    } else if (const auto* w = std::get_if<Welcome>(&msg)) {
        m.set("role", w->role);
        m.set("capabilities", str_list(w->capabilities));
        if (!w->software.is_null()) m.set("software", w->software);
        if (!w->session.is_null()) m.set("session", w->session);
        if (w->control) m.set("control", *w->control);
    } else if (const auto* b = std::get_if<Bye>(&msg)) {
        m.set("reason", b->reason);
        m.set("message", b->message);
    } else if (const auto* r = std::get_if<Error>(&msg)) {
        m.set("code", r->code);
        m.set("message", r->message);
        m.set("fatal", r->fatal);
        if (r->in_reply_to) m.set("in_reply_to", *r->in_reply_to);
        if (r->request_id) m.set("request_id", *r->request_id);
        if (r->sub) m.set("sub", *r->sub);
        if (r->view) m.set("view", *r->view);
        if (r->session) m.set("session", *r->session);
    } else if (const auto* p = std::get_if<Ping>(&msg)) {
        m.set("nonce", p->nonce);
    } else if (const auto* q = std::get_if<Pong>(&msg)) {
        m.set("nonce", q->nonce);
    } else if (const auto* s = std::get_if<Sessions>(&msg)) {
        m.set("sessions", s->sessions);
    } else if (const auto* a = std::get_if<Attach>(&msg)) {
        m.set("session", a->session ? Json(*a->session) : Json());
    } else if (const auto* v = std::get_if<Views>(&msg)) {
        m.set("session", v->session ? Json(*v->session) : Json());
        m.set("views", v->views);
        if (!v->timelines.is_null()) m.set("timelines", v->timelines);   // 3.2 draft
    } else if (const auto* s = std::get_if<Subscribe>(&msg)) {
        m.set("sub", s->sub);
        m.set("view", s->view);
        m.set("request_id", s->request_id);
        if (s->max_rate) m.set("max_rate", *s->max_rate);
        if (!s->data_view.is_null()) m.set("data_view", s->data_view);
    } else if (const auto* u = std::get_if<Unsubscribe>(&msg)) {
        m.set("sub", u->sub);
    } else if (const auto* r = std::get_if<ViewRequest>(&msg)) {
        m.set("request_id", r->request_id);
        m.set("sub", r->sub);
        m.set("view", r->view);
        m.set("data_view", r->data_view);
    } else if (const auto* r = std::get_if<ViewReply>(&msg)) {
        m.set("request_id", r->request_id);
        m.set("sub", r->sub);
        m.set("view", r->view);
        if (!r->data_view.is_null()) m.set("data_view", r->data_view);
        if (!r->error.is_null()) m.set("error", r->error);
    } else if (const auto* s = std::get_if<Snapshot>(&msg)) {
        m.set("sub", s->sub);
        m.set("view", s->view);
        m.set("seq", s->seq);
        m.set("stamp_ns", s->stamp_ns);
        m.set("kind", s->kind);
        m.set("data_view", s->data_view);
        m.set("colormap", s->colormap);
        m.set("flags", s->flags);
        m.set("vmin", s->vmin);
        m.set("vmax", s->vmax);
        m.set("stats", stats_json(s->stats));
        m.set("axes", s->axes);
        if (!s->tile.is_null()) m.set("tile", s->tile);
        if (s->points_dims) m.set("points_dims", *s->points_dims);
        if (s->label_count) m.set("label_count", *s->label_count);
        if (!s->labels.is_null()) m.set("labels", s->labels);   // 3.2 draft
        if (s->horizontal) m.set("horizontal", true);
        if (s->is_static) m.set("static", true);
        if (s->cursor) m.set("cursor", *s->cursor);
        if (s->empty) m.set("empty", true);
        if (!s->cadence.is_null()) m.set("cadence", s->cadence);
        if (!s->empty) e.sections.push_back(named(s->data, "data"));
        if (s->aux) e.sections.push_back(named(*s->aux, "aux"));
        if (s->colors) e.sections.push_back(named(*s->colors, "colors"));
        if (s->stamp) e.sections.push_back(named(*s->stamp, "stamp"));
    } else if (const auto* s = std::get_if<Series>(&msg)) {
        m.set("sub", s->sub);
        m.set("view", s->view);
        m.set("start", s->start);
        m.set("total", s->total);
        m.set("stamp_ns", s->stamp_ns);
        if (s->decimated) m.set("decimated", *s->decimated);   // 3.2 draft
        if (s->cursor) m.set("cursor", *s->cursor);
        e.sections.push_back(named(s->values, "values"));
        if (s->stamps) e.sections.push_back(named(*s->stamps, "stamps"));
        else if (s->stamp) e.sections.push_back(named(*s->stamp, "stamp"));
        if (s->indices) e.sections.push_back(named(*s->indices, "indices"));
    } else if (const auto* v = std::get_if<Event>(&msg)) {
        m.set("view", v->view);
        m.set("type", v->type);
        m.set("detail", v->detail);
        if (v->sub) m.set("sub", *v->sub);
        if (v->seq) m.set("seq", *v->seq);
        if (!v->data_view.is_null()) m.set("data_view", v->data_view);
    } else if (const auto* c = std::get_if<Controls>(&msg)) {
        m.set("run", c->run);
        m.set("params", c->params);
        m.set("actions", c->actions);
        if (c->stamp) e.sections.push_back(named(*c->stamp, "stamp"));   // 3.2 draft
        if (!c->interventions.is_null()) m.set("interventions", c->interventions);
    } else if (const auto* c = std::get_if<Control>(&msg)) {
        m.set("request_id", c->request_id);
        m.set("op", c->op);
        if (c->name) m.set("name", *c->name);
        if (!c->value.is_null()) m.set("value", c->value);
        if (c->count) m.set("count", *c->count);
        if (!c->intervene.is_null()) m.set("intervene", c->intervene);
    } else if (const auto* r = std::get_if<ControlReply>(&msg)) {
        m.set("request_id", r->request_id);
        m.set("op", r->op);
        m.set("status", r->status);
        if (r->name) m.set("name", *r->name);
        if (!r->value.is_null()) m.set("value", r->value);
        if (r->message) m.set("message", *r->message);
        if (!r->error.is_null()) m.set("error", r->error);
    } else if (const auto* s = std::get_if<Selection>(&msg)) {
        m.set("space", s->space);
        m.set("items", s->items);
        if (s->view) m.set("view", *s->view);
        if (s->truncated) m.set("truncated", true);
        if (!s->details.is_null()) m.set("details", s->details);
        if (s->details_error) m.set("details_error", *s->details_error);
    } else if (const auto* su = std::get_if<SceneUpdate>(&msg)) {
        scene_update_to_meta(*su, m, e.sections);
    } else if (const auto* st = std::get_if<SceneTransforms>(&msg)) {
        scene_transforms_to_meta(*st, m, e.sections);
    } else if (const auto* sc = std::get_if<SceneClear>(&msg)) {
        scene_clear_to_meta(*sc, m, e.sections);
    } else if (const auto* sk = std::get_if<SceneClasses>(&msg)) {
        scene_classes_to_meta(*sk, m, e.sections);
    } else if (const auto* tx = std::get_if<Text>(&msg)) {
        text_to_meta(*tx, m, e.sections);
    } else if (const auto* rs = std::get_if<Recordings>(&msg)) {
        m.set("recordings", rs->recordings);
    } else if (const auto* ro = std::get_if<RecordingOpen>(&msg)) {
        m.set("request_id", ro->request_id);
        m.set("recording", ro->recording);
    } else if (const auto* cu = std::get_if<Cursor>(&msg)) {
        cursor_to_meta(*cu, m);
    } else if (const auto* ts = std::get_if<TimeState>(&msg)) {
        time_state_to_meta(*ts, m);
    } else if (const auto* pg = std::get_if<Program>(&msg)) {
        program_to_meta(*pg, m, e.sections);
    } else if (const auto* ps = std::get_if<ProgramState>(&msg)) {
        program_state_to_meta(*ps, m);
    } else if (const auto* rc = std::get_if<Resource>(&msg)) {
        resource_to_meta(*rc, m, e.sections);
    } else if (const auto* u = std::get_if<Unknown>(&msg)) {
        m = u->meta.is_object() ? u->meta : Json::object();
        m.erase("sections");
        e.sections = u->sections;
    }
    return e;
}

}  // namespace

// ---- Section, Frame, identities -------------------------------------------------------------

int64_t Section::count() const {
    int64_t n = 1;
    for (int64_t s : shape) {
        if (s < 0) return -1;
        if (s != 0 && n > INT64_MAX / s) return -1;
        n *= s;
    }
    return n;
}

Section Section::make(const std::string& name, int64_t dtype, std::vector<int64_t> shape, const void* data,
                      size_t size) {
    Section s;
    s.name = name;
    s.dtype = dtype;
    s.shape = std::move(shape);
    s.data = static_cast<const uint8_t*>(data);
    s.size = size;
    return s;
}

const Section* Frame::section(const std::string& name) const {
    for (const auto& s : sections)
        if (s.name == name) return &s;
    return nullptr;
}

Json Identity::to_json() const {
    Json j = Json::object();
    j.set("id", id);
    j.set("script", script);
    j.set("pid", pid);
    j.set("start_unix_ms", start_unix_ms);
    return j;
}

bool Identity::parse(const Json& j, Identity* out) {
    if (!is_identity(j)) return false;
    out->id = j.find("id")->as_string();
    out->script = j.find("script")->as_string();
    out->pid = u64(*j.find("pid"));
    out->start_unix_ms = u64(*j.find("start_unix_ms"));
    return true;
}

Json SessionInfo::to_json() const {
    Json j = identity.to_json();
    j.set("state", state);
    j.set("views", views);
    return j;
}

// ---- frames -----------------------------------------------------------------------------------

bool decode_frame(const uint8_t* buf, size_t n, Frame* out, FrameError* err) {
    auto fail = [&](int code, const char* reason, const std::string& msg) {
        if (err) *err = FrameError{code, reason, msg};
        return false;
    };
    // 1-4: the fixed header
    if (n < 8) return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "truncated", std::to_string(n) + " bytes, fewer than 8");
    const uint32_t magic = rd32(buf);
    const uint16_t major = rd16(buf + 4);
    if (magic != CALIPER_NET_MAGIC) return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "bad_magic", "bad magic");
    if (major != CALIPER_NET_PROTOCOL_MAJOR)
        return fail(CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR, "unsupported_major", "major " + std::to_string(major));
    if (n < CALIPER_NET_HEADER_BYTES)
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "truncated", std::to_string(n) + " bytes, fewer than 24");
    const uint16_t minor = rd16(buf + 6);
    const uint16_t type = rd16(buf + 8);
    const uint16_t flags = rd16(buf + 10);
    const uint32_t frame_len = rd32(buf + 12);
    const uint32_t meta_len = rd32(buf + 16);
    const uint16_t count = rd16(buf + 20);
    // 5: limits
    if (frame_len > CALIPER_NET_MAX_FRAME_BYTES || meta_len > CALIPER_NET_MAX_META_BYTES ||
        count > CALIPER_NET_MAX_SECTIONS)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big",
                    "frame_len " + std::to_string(frame_len) + ", meta_len " + std::to_string(meta_len) + ", " +
                        std::to_string(count) + " sections");
    // 6: the message is the frame
    if (n != frame_len)
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, n < frame_len ? "truncated" : "malformed",
                    "message of " + std::to_string(n) + " bytes, frame_len " + std::to_string(frame_len));
    // 7: the layout ends at frame_len
    const uint64_t table_end = CALIPER_NET_HEADER_BYTES + 4ull * count;
    if (table_end > frame_len)
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "section table past the end of the frame");
    uint32_t lens[CALIPER_NET_MAX_SECTIONS];
    uint64_t offs[CALIPER_NET_MAX_SECTIONS];
    for (uint16_t i = 0; i < count; ++i) lens[i] = rd32(buf + CALIPER_NET_HEADER_BYTES + 4u * i);
    const uint64_t meta_end = table_end + meta_len;
    uint64_t pos = meta_end;
    for (uint16_t i = 0; i < count; ++i) {
        pos = align8(pos);
        offs[i] = pos;
        pos += lens[i];
    }
    if (pos != frame_len)
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed",
                    "layout ends at " + std::to_string(pos) + ", frame_len " + std::to_string(frame_len));
    // 8: the metadata
    Json meta;
    std::string why;
    if (!Json::parse((const char*)buf + table_end, meta_len, &meta, &why, CALIPER_NET_MAX_JSON_DEPTH))
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "metadata: " + why);
    if (!meta.is_object()) return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "metadata is not a JSON object");
    // 9: the section descriptors
    std::vector<Section> sections;
    const Json* descs = meta.find("sections");
    if (count) {
        if (!descs || !descs->is_array() || descs->size() != count)
            return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed",
                        "'sections' does not describe " + std::to_string(count) + " sections");
        std::set<std::string> names;
        for (uint16_t i = 0; i < count; ++i) {
            const Json& d = descs->at(i);
            if (!d.is_object())
                return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "section descriptor is not an object");
            const Json* name = d.find("name");
            const Json* dt = d.find("dtype");
            const Json* shape = d.find("shape");
            const Json* enc = d.find("encoding");
            bool ok = name && name->is_string() && dt && dt->is_int() && (!enc || enc->is_int()) && shape &&
                      shape->is_array();
            if (ok)
                for (const Json& x : shape->items())
                    if (!is_uint(x)) { ok = false; break; }
            if (!ok) return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "section descriptor " + show(d));
            if (!names.insert(name->as_string()).second)
                return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed",
                            "section name '" + name->as_string() + "' repeated");
            Section s;
            s.name = name->as_string();
            s.dtype = i64(*dt);
            s.encoding = enc ? i64(*enc) : CALIPER_NET_ENCODING_RAW;
            bool huge = false;
            for (const Json& x : shape->items()) {
                const uint64_t v = u64(x);
                if (v > (uint64_t)INT64_MAX) huge = true;
                s.shape.push_back((int64_t)std::min<uint64_t>(v, (uint64_t)INT64_MAX));
            }
            s.data = buf + offs[i];
            s.size = lens[i];
            s.offset = (size_t)offs[i];
            const bool known = s.dtype >= 0 && s.dtype <= 0x7fffffff && caliper_net_dtype_size((int32_t)s.dtype) != 0;
            if (s.encoding == CALIPER_NET_ENCODING_RAW && known) {
                const int64_t cnt = huge ? -1 : s.count();
                const uint64_t item = caliper_net_dtype_size((int32_t)s.dtype);
                const bool match = cnt >= 0 && (uint64_t)cnt <= UINT64_MAX / item && (uint64_t)cnt * item == lens[i];
                if (!match)
                    return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed",
                                "section '" + s.name + "': " + std::to_string(lens[i]) +
                                    " bytes disagree with its shape");
            }
            sections.push_back(std::move(s));
        }
    } else if (descs && !descs->is_null() && !(descs->is_array() && descs->size() == 0)) {
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "'sections' given for a frame without sections");
    }
    out->type = type;
    out->flags = flags;
    out->major = major;
    out->minor = minor;
    out->meta = std::move(meta);
    out->sections = std::move(sections);
    return true;
}

bool encode_frame(uint16_t type, const Json& meta_in, const std::vector<Section>& sections, std::vector<uint8_t>* out,
                  FrameError* err, uint16_t minor, uint16_t flags) {
    auto fail = [&](int code, const char* reason, const std::string& msg) {
        if (err) *err = FrameError{code, reason, msg};
        return false;
    };
    if (sections.size() > CALIPER_NET_MAX_SECTIONS)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big", std::to_string(sections.size()) + " sections");
    Json meta = meta_in.is_object() ? meta_in : Json::object();
    meta.erase("sections");
    if (!sections.empty()) {
        Json descs = Json::array();
        for (const auto& s : sections) {
            Json d = Json::object();
            d.set("name", s.name);
            d.set("dtype", s.dtype);
            Json shape = Json::array();
            for (int64_t x : s.shape) shape.push(x);
            d.set("shape", std::move(shape));
            if (s.encoding != CALIPER_NET_ENCODING_RAW) d.set("encoding", s.encoding);
            descs.push(std::move(d));
        }
        meta.set("sections", std::move(descs));
    }
    const std::string mbytes = meta.dump();
    if (mbytes.size() > CALIPER_NET_MAX_META_BYTES)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big", "metadata of " + std::to_string(mbytes.size()) + " bytes");
    uint64_t pos = CALIPER_NET_HEADER_BYTES + 4ull * sections.size() + mbytes.size();
    std::vector<uint64_t> offs;
    for (const auto& s : sections) {
        pos = align8(pos);
        offs.push_back(pos);
        pos += s.size;
    }
    if (pos > CALIPER_NET_MAX_FRAME_BYTES)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big", "frame of " + std::to_string(pos) + " bytes");
    out->assign((size_t)pos, 0);
    uint8_t* b = out->data();
    wr32(b, CALIPER_NET_MAGIC);
    wr16(b + 4, CALIPER_NET_PROTOCOL_MAJOR);
    wr16(b + 6, minor);
    wr16(b + 8, type);
    wr16(b + 10, flags);
    wr32(b + 12, (uint32_t)pos);
    wr32(b + 16, (uint32_t)mbytes.size());
    wr16(b + 20, (uint16_t)sections.size());
    wr16(b + 22, 0);
    for (size_t i = 0; i < sections.size(); ++i) wr32(b + CALIPER_NET_HEADER_BYTES + 4 * i, (uint32_t)sections[i].size);
    std::memcpy(b + CALIPER_NET_HEADER_BYTES + 4 * sections.size(), mbytes.data(), mbytes.size());
    for (size_t i = 0; i < sections.size(); ++i)
        if (sections[i].size) std::memcpy(b + offs[i], sections[i].data, sections[i].size);
    return true;
}

bool encode_head(uint16_t type, const Json& meta, const std::vector<uint32_t>& section_lens, size_t tail_len,
                 std::vector<uint8_t>* head, FrameError* err, uint16_t minor, uint16_t flags) {
    auto fail = [&](int code, const char* reason, const std::string& msg) {
        if (err) *err = FrameError{code, reason, msg};
        return false;
    };
    if (section_lens.empty() || section_lens.size() > CALIPER_NET_MAX_SECTIONS)
        return fail(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "malformed", "a head needs 1 to 8 sections");
    const std::string mbytes = meta.dump();
    if (mbytes.size() > CALIPER_NET_MAX_META_BYTES)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big", "metadata of " + std::to_string(mbytes.size()) + " bytes");
    const uint64_t meta_end = CALIPER_NET_HEADER_BYTES + 4ull * section_lens.size() + mbytes.size();
    const uint64_t head_len = align8(meta_end);
    const uint64_t frame_len = head_len + tail_len;
    if (frame_len > CALIPER_NET_MAX_FRAME_BYTES)
        return fail(CALIPER_NET_CLOSE_TOO_BIG, "too_big", "frame of " + std::to_string(frame_len) + " bytes");
    head->assign((size_t)head_len, 0);
    uint8_t* b = head->data();
    wr32(b, CALIPER_NET_MAGIC);
    wr16(b + 4, CALIPER_NET_PROTOCOL_MAJOR);
    wr16(b + 6, minor);
    wr16(b + 8, type);
    wr16(b + 10, flags);
    wr32(b + 12, (uint32_t)frame_len);
    wr32(b + 16, (uint32_t)mbytes.size());
    wr16(b + 20, (uint16_t)section_lens.size());
    wr16(b + 22, 0);
    for (size_t i = 0; i < section_lens.size(); ++i) wr32(b + CALIPER_NET_HEADER_BYTES + 4 * i, section_lens[i]);
    std::memcpy(b + CALIPER_NET_HEADER_BYTES + 4 * section_lens.size(), mbytes.data(), mbytes.size());
    return true;
}

// ---- messages ---------------------------------------------------------------------------------

DecodeResult decode(const uint8_t* buf, size_t n, Message* out) {
    DecodeResult r;
    FrameError fe;
    if (!decode_frame(buf, n, &r.frame, &fe)) {
        r.status = DecodeResult::Status::BadFrame;
        r.close_code = fe.close_code;
        r.reason = fe.reason;
        r.message = fe.message;
        return r;
    }
    r.type = r.frame.type;
    const Frame& f = r.frame;
    try {
        switch (f.type) {
            case CALIPER_NET_MSG_HELLO: *out = hello(f); break;
            case CALIPER_NET_MSG_WELCOME: *out = welcome(f); break;
            case CALIPER_NET_MSG_BYE: *out = bye(f); break;
            case CALIPER_NET_MSG_ERROR: *out = error(f); break;
            case CALIPER_NET_MSG_PING: *out = Ping{nonce(f)}; break;
            case CALIPER_NET_MSG_PONG: *out = Pong{nonce(f)}; break;
            case CALIPER_NET_MSG_SESSIONS: *out = sessions(f); break;
            case CALIPER_NET_MSG_ATTACH: *out = attach(f); break;
            case CALIPER_NET_MSG_VIEWS: *out = views(f); break;
            case CALIPER_NET_MSG_SUBSCRIBE: *out = subscribe(f); break;
            case CALIPER_NET_MSG_UNSUBSCRIBE: *out = unsubscribe(f); break;
            case CALIPER_NET_MSG_VIEW_REQUEST: *out = view_request(f); break;
            case CALIPER_NET_MSG_VIEW_REPLY: *out = view_reply(f); break;
            case CALIPER_NET_MSG_SNAPSHOT: *out = snapshot(f); break;
            case CALIPER_NET_MSG_SERIES: *out = series(f); break;
            case CALIPER_NET_MSG_EVENT: *out = event(f); break;
            case CALIPER_NET_MSG_CONTROLS: *out = controls(f); break;
            case CALIPER_NET_MSG_CONTROL: *out = control(f); break;
            case CALIPER_NET_MSG_CONTROL_REPLY: *out = control_reply(f); break;
            case CALIPER_NET_MSG_SELECTION: *out = selection(f); break;
            case CALIPER_NET_MSG_SCENE_UPDATE: *out = scene_update(f); break;
            case CALIPER_NET_MSG_SCENE_TRANSFORMS: *out = scene_transforms(f); break;
            case CALIPER_NET_MSG_SCENE_CLEAR: *out = scene_clear(f); break;
            case CALIPER_NET_MSG_SCENE_CLASSES: *out = scene_classes(f); break;
            case CALIPER_NET_MSG_TEXT: *out = text(f); break;
            case CALIPER_NET_MSG_RECORDINGS: *out = recordings(f); break;
            case CALIPER_NET_MSG_RECORDING_OPEN: *out = recording_open(f); break;
            case CALIPER_NET_MSG_CURSOR: *out = cursor(f); break;
            case CALIPER_NET_MSG_TIME_STATE: *out = time_state(f); break;
            case CALIPER_NET_MSG_PROGRAM: *out = program(f); break;
            case CALIPER_NET_MSG_PROGRAM_STATE: *out = program_state(f); break;
            case CALIPER_NET_MSG_RESOURCE: *out = resource(f); break;
            default: *out = Unknown{f.type, f.meta, f.sections}; break;
        }
    } catch (const MsgError& e) {
        r.status = DecodeResult::Status::BadMessage;
        r.error_code = e.code;
        r.message = e.message;
    }
    return r;
}

bool encode(const Message& m, std::vector<uint8_t>* out, std::string* err, uint16_t minor) {
    Encoded e = to_meta(m);
    FrameError fe;
    if (!encode_frame(e.type, e.meta, e.sections, out, &fe, minor)) {
        if (err) *err = fe.reason + ": " + fe.message;
        return false;
    }
    return true;
}

uint16_t message_type(const Message& m) {
    return std::visit([](const auto& x) -> uint16_t {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, Unknown>) return x.type;
        else return T::kType;
    }, m);
}

const char* type_name(uint16_t type) {
    switch (type) {
        case CALIPER_NET_MSG_HELLO: return "hello";
        case CALIPER_NET_MSG_WELCOME: return "welcome";
        case CALIPER_NET_MSG_BYE: return "bye";
        case CALIPER_NET_MSG_ERROR: return "error";
        case CALIPER_NET_MSG_PING: return "ping";
        case CALIPER_NET_MSG_PONG: return "pong";
        case CALIPER_NET_MSG_SESSIONS: return "sessions";
        case CALIPER_NET_MSG_ATTACH: return "attach";
        case CALIPER_NET_MSG_VIEWS: return "views";
        case CALIPER_NET_MSG_SUBSCRIBE: return "subscribe";
        case CALIPER_NET_MSG_UNSUBSCRIBE: return "unsubscribe";
        case CALIPER_NET_MSG_VIEW_REQUEST: return "view_request";
        case CALIPER_NET_MSG_VIEW_REPLY: return "view_reply";
        case CALIPER_NET_MSG_SNAPSHOT: return "snapshot";
        case CALIPER_NET_MSG_SERIES: return "series";
        case CALIPER_NET_MSG_EVENT: return "event";
        case CALIPER_NET_MSG_CONTROLS: return "controls";
        case CALIPER_NET_MSG_CONTROL: return "control";
        case CALIPER_NET_MSG_CONTROL_REPLY: return "control_reply";
        case CALIPER_NET_MSG_SELECTION: return "selection";
        case CALIPER_NET_MSG_SCENE_UPDATE: return "scene_update";
        case CALIPER_NET_MSG_SCENE_TRANSFORMS: return "scene_transforms";
        case CALIPER_NET_MSG_SCENE_CLEAR: return "scene_clear";
        case CALIPER_NET_MSG_SCENE_CLASSES: return "scene_classes";
        case CALIPER_NET_MSG_TEXT: return "text";
        case CALIPER_NET_MSG_RECORDINGS: return "recordings";
        case CALIPER_NET_MSG_RECORDING_OPEN: return "recording_open";
        case CALIPER_NET_MSG_CURSOR: return "cursor";
        case CALIPER_NET_MSG_TIME_STATE: return "time_state";
        case CALIPER_NET_MSG_PROGRAM: return "program";
        case CALIPER_NET_MSG_PROGRAM_STATE: return "program_state";
        case CALIPER_NET_MSG_RESOURCE: return "resource";
        default: return "?";
    }
}

// ---- protocol 3.2 draft: stamp text, text lines and split updates ----

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

SplitUpdates::Result SplitUpdates::add(const SceneUpdate& piece, WholeUpdate* out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return Result::Refused;
    };
    const Json& sp = piece.split;
    const uint64_t uid = u64(*sp.find("update"));
    const int64_t num = i64(*sp.find("piece"));
    auto stamp_bytes = [](const SceneUpdate& u) {
        return u.stamp ? std::vector<uint8_t>(u.stamp->data, u.stamp->data + u.stamp->size) : std::vector<uint8_t>();
    };
    if (num == 0) {
        for (auto it = open_.begin(); it != open_.end();) {
            if (it->update == uid || (it->p0.path == piece.path && it->p0.part == piece.part)) {
                it = open_.erase(it);
                ++discarded_;
            } else {
                ++it;
            }
        }
        Assembly a;
        a.update = uid;
        a.p0 = piece;
        a.p0.arrays.clear();
        a.stamp_bytes = stamp_bytes(piece);
        a.p0.stamp.reset();
        a.pieces = i64(*sp.find("pieces"));
        a.rows = *sp.find("rows");
        open_.push_back(std::move(a));
        while (open_.size() > CALIPER_NET_MAX_INCOMPLETE_UPDATES) {
            open_.erase(open_.begin());
            ++discarded_;
        }
    }
    auto it = std::find_if(open_.begin(), open_.end(), [&](const Assembly& a) { return a.update == uid; });
    if (it == open_.end()) {
        ++dropped_;
        return Result::Dropped;
    }
    Assembly& a = *it;
    auto refuse = [&](const std::string& m) {
        open_.erase(std::find_if(open_.begin(), open_.end(), [&](const Assembly& x) { return x.update == uid; }));
        ++discarded_;
        return fail(m);
    };
    // The checks of _Assembly.add, in its order.
    if (piece.path != a.p0.path || piece.part != a.p0.part || piece.seq != a.p0.seq ||
        i64(*sp.find("pieces")) != a.pieces || stamp_bytes(piece) != a.stamp_bytes)
        return refuse("a piece's path, part, seq, pieces or stamp differ from its update's piece 0");
    if (std::find(a.seen.begin(), a.seen.end(), num) != a.seen.end())
        return refuse("piece " + std::to_string(num) + " arrived twice");
    a.seen.push_back(num);
    const Json& first = *sp.find("first");
    for (const Section& s : piece.arrays) {
        const Json* rows_j = a.rows.find(s.name);
        if (!rows_j) return refuse("a piece holds '" + s.name + "', which 'split.rows' does not name");
        const uint32_t item = s.dtype >= 0 && s.dtype <= 0x7fffffff ? caliper_net_dtype_size((int32_t)s.dtype) : 0u;
        if (item == 0) return refuse("a piece of '" + s.name + "' has an unknown dtype");
        if (s.shape.empty() || s.shape[0] < 0) return refuse("a piece of '" + s.name + "' has no rows");
        const std::vector<int64_t> trailing(s.shape.begin() + 1, s.shape.end());
        // The bytes of one row, held at most at one past the update limit so that no product wraps: a
        // larger row refuses any count of rows but 0 below.
        uint64_t rowbytes = item;
        for (int64_t d : trailing) {
            if (d < 0) return refuse("a piece of '" + s.name + "' has a negative dimension");
            rowbytes = d && rowbytes > (CALIPER_NET_MAX_UPDATE_BYTES + 1) / (uint64_t)d
                           ? CALIPER_NET_MAX_UPDATE_BYTES + 1
                           : rowbytes * (uint64_t)d;
        }
        const uint64_t rows = u64(*rows_j);
        if (rows > (uint64_t)INT64_MAX) return refuse("'split.rows' of '" + s.name + "' is too large");
        auto ai = std::find_if(a.arrays.begin(), a.arrays.end(), [&](const auto& x) { return x.first == s.name; });
        if (ai == a.arrays.end()) {
            const uint64_t total = rows * rowbytes;
            if ((rowbytes && rows > CALIPER_NET_MAX_UPDATE_BYTES / rowbytes) ||
                a.bytes + total > CALIPER_NET_MAX_UPDATE_BYTES)
                return refuse("a split update of more than 536870912 bytes");
            a.bytes += total;
            Array arr;
            arr.dtype = s.dtype;
            arr.trailing = trailing;
            arr.bytes.resize((size_t)total);
            a.arrays.emplace_back(s.name, std::move(arr));
            ai = a.arrays.end() - 1;
        } else if (ai->second.dtype != s.dtype || ai->second.trailing != trailing) {
            return refuse("the pieces of '" + s.name + "' differ in dtype or shape");
        }
        // Unsigned, with no sum that can wrap: f0 + n <= rows holds after this check, so the overlap test
        // and the copy below stay inside the array.
        const Json* fj = first.find(s.name);
        if (!fj || !is_uint(*fj)) return refuse("'split.first' does not name '" + s.name + "'");
        const uint64_t f0 = u64(*fj), n = (uint64_t)s.shape[0];
        if (f0 > rows || n > rows - f0) return refuse("rows of '" + s.name + "' pass its rows");
        if ((uint64_t)s.size != n * rowbytes)
            return refuse("a piece of '" + s.name + "' holds bytes that disagree with its shape");
        for (const auto& r : ai->second.ranges)
            if (f0 < r.first + r.second && r.first < f0 + n) return refuse("two pieces hold the same rows of '" + s.name + "'");
        ai->second.ranges.emplace_back(f0, n);
        if (s.size) std::memcpy(ai->second.bytes.data() + (size_t)(f0 * rowbytes), s.data, s.size);
    }
    if ((int64_t)a.seen.size() < a.pieces) return Result::Pending;
    for (const auto& kv : a.rows.members()) {
        auto ai = std::find_if(a.arrays.begin(), a.arrays.end(), [&](const auto& x) { return x.first == kv.first; });
        uint64_t got = 0;
        if (ai != a.arrays.end())
            for (const auto& r : ai->second.ranges) got += r.second;
        if (got != u64(kv.second)) return refuse("the pieces ended before every row of '" + kv.first + "' arrived");
    }
    // The whole update: piece 0's keys with the whole arrays and the stamp, checked as an unsplit update.
    struct Keep {
        std::vector<std::pair<std::string, Array>> arrays;
        std::vector<uint8_t> stamp;
    };
    auto keep = std::make_shared<Keep>();
    keep->arrays = std::move(a.arrays);
    keep->stamp = std::move(a.stamp_bytes);
    SceneUpdate p0 = std::move(a.p0);
    const Json rows = a.rows;
    open_.erase(std::find_if(open_.begin(), open_.end(), [&](const Assembly& x) { return x.update == uid; }));
    Frame f;
    f.type = CALIPER_NET_MSG_SCENE_UPDATE;
    p0.split = Json();
    f.meta = to_meta(Message(p0)).meta;
    const int part = scene_part_index(p0.part);
    for (const char* const* k = scene_part_sections(part); *k; ++k) {
        for (const auto& kv : keep->arrays) {
            if (kv.first != *k) continue;
            std::vector<int64_t> shape{(int64_t)u64(*rows.find(kv.first))};
            shape.insert(shape.end(), kv.second.trailing.begin(), kv.second.trailing.end());
            f.sections.push_back(Section::make(kv.first, kv.second.dtype, shape, kv.second.bytes.data(),
                                               kv.second.bytes.size()));
        }
    }
    if (!keep->stamp.empty())
        f.sections.push_back(Section::make("stamp", CALIPER_NET_DTYPE_I64, {(int64_t)keep->stamp.size() / 8},
                                           keep->stamp.data(), keep->stamp.size()));
    try {
        SceneUpdate whole = scene_update(f);
        if (out) {
            out->update = std::move(whole);
            out->keep = keep;
        }
        return Result::Whole;
    } catch (const MsgError& e) {
        ++discarded_;
        return fail(e.message);
    }
}

}  // namespace wire
}  // namespace caliper_net
