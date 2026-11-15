// wire.cpp: frames and messages of wire protocol v3 (wire.h). The checks and
// their order follow python/caliper/_wire.py line for line, so the two codecs
// accept and refuse the same frames.
#include "net/wire.h"

#include "net/ws.h"

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
            if (!(dn && is_name(*dn) && sz && is_uint(*sz))) return false;
            si.dims.push_back({dn->as_string(), u64(*sz)});
        }
        vi.sources.push_back(std::move(si));
    }
    const Json* def = v.find("default");
    if (def && !def->is_null() && !is_obj(*def)) return false;
    if (def && def->is_object()) vi.default_view = *def;
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
    for (const char* key : {"indices", "row_ids"}) {
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
    e.view = req(m, "view", is_name, "a view name").as_string();
    e.type = req(m, "type", is_str, "a string").as_string();
    if (const Json* v = opt(m, "sub", is_u32, "a uint32")) e.sub = u32(*v);
    if (const Json* v = opt(m, "seq", is_uint, "an integer")) e.seq = u64(*v);
    if (const Json* v = opt(m, "data_view", is_obj, "an object")) e.data_view = *v;
    return e;
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
    } else if (const auto* u = std::get_if<Unknown>(&msg)) {
        m = u->meta.is_object() ? u->meta : Json::object();
        m.erase("sections");
        e.sections = u->sections;
    }
    return e;
}

}  // namespace

// ---- the upgrade request -------------------------------------------------------------

namespace {
std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}
bool all_digits(const std::string& s) {
    if (s.empty() || s.size() > 5) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}
}  // namespace

bool host_allowed(const std::string& host, uint16_t port) {
    const std::string h = lower_ascii(host);
    const std::string p = ":" + std::to_string(port);
    if (h == "127.0.0.1" + p || h == "localhost" + p) return true;
    return port == 80 && (h == "127.0.0.1" || h == "localhost");
}

bool origin_allowed(const std::string& origin) {
    if (origin == "null") return true;
    const std::string o = lower_ascii(origin);
    const std::string vs = "vscode-webview://";
    if (o.compare(0, vs.size(), vs) == 0) {
        const std::string auth = o.substr(vs.size());
        if (auth.empty()) return false;
        for (char c : auth)
            if (c == '/' || c == '?' || c == '#' || c == '@' || (unsigned char)c <= ' ') return false;
        return true;
    }
    std::string rest;
    if (o.compare(0, 7, "http://") == 0) rest = o.substr(7);
    else if (o.compare(0, 8, "https://") == 0) rest = o.substr(8);
    else return false;
    for (const char* host : {"127.0.0.1", "localhost", "[::1]"}) {
        const std::string h = host;
        if (rest == h) return true;
        if (rest.compare(0, h.size() + 1, h + ":") == 0 && all_digits(rest.substr(h.size() + 1))) return true;
    }
    return false;
}

bool token_equal(const std::string& a, const std::string& b) {
    // Every byte of the expected token is read whatever the input, so the time
    // says nothing about how much of it matched.
    unsigned diff = a.size() != b.size() ? 1u : 0u;
    for (size_t i = 0; i < b.size(); ++i) diff |= (unsigned)(uint8_t)(i < a.size() ? a[i] : 0) ^ (uint8_t)b[i];
    return diff == 0;
}

int upgrade_status(const HttpRequest& r, const std::string& session_token, uint16_t port) {
    if (r.path != CALIPER_NET_PATH) return 404;
    if (!host_allowed(r.host, port)) return 403;
    if (r.has_origin && !origin_allowed(r.origin)) return 403;
    std::string token;
    if (!r.query_param(CALIPER_NET_TOKEN_PARAM, &token) || session_token.empty() ||
        !token_equal(token, session_token))
        return 403;
    return 0;
}

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
        default: return "?";
    }
}

}  // namespace wire
}  // namespace caliper_net
