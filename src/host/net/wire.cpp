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
            if (!(is_identity(x) && st && st->is_string() &&
                  (st->as_string() == CALIPER_NET_SESSION_LIVE || st->as_string() == CALIPER_NET_SESSION_ENDED) &&
                  views && is_uint(*views)))
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
        default: return;   // an unknown kind: the caller ignores it
    }
    if (d.dtype != want_dt || d.shape.size() != want_rank)
        bad("kind " + std::to_string(s.kind) + ": 'data' is dtype " + std::to_string(d.dtype) + " rank " +
            std::to_string(d.shape.size()));
    if (s.kind == CALIPER_NET_KIND_IMAGE && d.shape[2] != 4) bad("an image's 'data' is (H, W, 4)");
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
    const Section* data = f.section("data");
    if (!data) bad("a snapshot needs a 'data' section");
    s.data = *data;
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
    check_payload(s);
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
    if (n < 1 || n > (int64_t)CALIPER_NET_SERIES_MAX_VALUES || s.start + (uint64_t)n < s.start ||
        s.start + (uint64_t)n > s.total)
        bad("series: " + std::to_string(n) + " values from " + std::to_string(s.start) + " with total " +
            std::to_string(s.total));
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
                      CALIPER_NET_OP_RUN});
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
    return c;
}

Control control(const Frame& f) {
    const Json& m = f.meta;
    Control c;
    c.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    c.op = req(m, "op", is_op, "pause, resume, step, set or run").as_string();
    const Json* name = opt(m, "name", is_name, "a parameter or action name");
    const bool named = c.op == CALIPER_NET_OP_SET || c.op == CALIPER_NET_OP_RUN;
    if (named && !name) bad("'" + c.op + "' needs 'name'");
    if (c.op == CALIPER_NET_OP_SET) c.value = req(m, "value", is_set_value, "a number, a boolean or a choice");
    if (c.op == CALIPER_NET_OP_STEP)
        if (const Json* n = opt(m, "count", [](const Json& v) { return is_u32(v) && u64(v) >= 1; },
                                "a positive uint32"))
            c.count = u32(*n);
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
                 std::optional<Section>* stamps = nullptr) {
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
    if (const Json* sp = m.find("split"); sp && !sp->is_null()) unsupported("split updates are not built yet");
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
        e.sections.push_back(named(s->data, "data"));
        if (s->aux) e.sections.push_back(named(*s->aux, "aux"));
    } else if (const auto* s = std::get_if<Series>(&msg)) {
        m.set("sub", s->sub);
        m.set("view", s->view);
        m.set("start", s->start);
        m.set("total", s->total);
        m.set("stamp_ns", s->stamp_ns);
        e.sections.push_back(named(s->values, "values"));
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
    } else if (const auto* c = std::get_if<Control>(&msg)) {
        m.set("request_id", c->request_id);
        m.set("op", c->op);
        if (c->name) m.set("name", *c->name);
        if (!c->value.is_null()) m.set("value", c->value);
        if (c->count) m.set("count", *c->count);
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
        default: return "?";
    }
}

}  // namespace wire
}  // namespace caliper_net
