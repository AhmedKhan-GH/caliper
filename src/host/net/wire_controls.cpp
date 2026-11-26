// wire_controls.cpp: protocol 3.1 controls and linked selection, plus 3.2
// interventions and interpretation messages (wire_internal.h).
#include "net/wire_internal.h"

#include <cmath>
#include <set>

namespace caliper_net {
namespace wire {
namespace detail {

namespace {

bool is_op(const Json& v) {
    return one_of(v, {CALIPER_NET_OP_PAUSE, CALIPER_NET_OP_RESUME, CALIPER_NET_OP_STEP, CALIPER_NET_OP_SET,
                      CALIPER_NET_OP_RUN, CALIPER_NET_OP_INTERVENE, CALIPER_NET_OP_INTERPRET});
}

bool param_text(const Json& v) {
    if (!v.is_string()) return false;
    const auto& s = v.as_string();
    return s.size() <= CALIPER_NET_PARAM_TEXT_MAX && s.find('\0') == std::string::npos &&
           utf8_ok(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

// Whether v is a value of parameter p's type, its range aside.
bool fits_type(const Json& p, const Json& v) {
    const std::string& t = p.find("type")->as_string();
    if (t == CALIPER_NET_PARAM_NUMBER) return is_finite(v);
    if (t == CALIPER_NET_PARAM_INTEGER) return v.is_int() && is_int53(v);
    if (t == CALIPER_NET_PARAM_BOOLEAN) return v.is_bool();
    if (t == CALIPER_NET_PARAM_TEXT) return param_text(v);
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
                         CALIPER_NET_PARAM_CHOICE, CALIPER_NET_PARAM_TEXT})))
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

// ---- 3.2 draft: interventions and interpretation ------------------------------------------------

bool opt_step(const Json* v) { return !present(v) || is_safe_uint(*v); }
bool point_ok(const Json* v) { return bounded_text(v, CALIPER_NET_POINT_MAX); }

bool interpretation_text(const Json& v, size_t limit, bool empty = true) {
    if (!v.is_string()) return false;
    const auto& s = v.as_string();
    return (empty || !s.empty()) && s.size() <= limit && s.find('\0') == std::string::npos &&
           utf8_ok(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

bool interpretation_value(const Json& v, unsigned depth, size_t& count) {
    if (depth > 8 || ++count > 16384) return false;
    if (v.is_null() || v.is_bool()) return true;
    if (v.is_string()) return interpretation_text(v, CALIPER_NET_INTERPRET_STRING_MAX);
    if (v.is_integer_token()) return is_int53(v) && std::abs(v.as_double()) <= double(kSafeInt);
    if (v.is_number()) return is_finite(v);
    if (v.is_array()) {
        if (v.size() > 512) return false;
        for (const auto& x : v.items())
            if (!interpretation_value(x, depth + 1, count)) return false;
        return true;
    }
    if (!v.is_object() || v.size() > 64) return false;
    for (const auto& kv : v.members()) {
        if (!interpretation_text(Json(kv.first), 63, false) ||
            !interpretation_value(kv.second, depth + 1, count))
            return false;
    }
    return true;
}

bool intervention_number(const Json& value) {
    return is_finite(value) && (!value.is_integer_token() || std::abs(value.as_double()) <= double(kSafeInt));
}

bool intervention_spec(const Json& value, bool lifecycle = false) {
    if (!value.is_object()) return false;
    const auto* op = value.find("op");
    const auto* kind = value.find("kind");
    if (present(op) && present(kind) && *op != *kind) return false;
    const auto* selected = present(kind) ? kind : op;
    if (!selected || !one_of(*selected, {"ablate", "add", "scale", "clamp", "patch", "steer"})) return false;
    const auto* point = value.find("point");
    if (!point || !interpretation_text(*point, 255, false)) return false;
    const auto& name = selected->as_string();
    std::set<std::string> fields = {"point", "op", "kind", "from_step", "until_step"};
    if (name == "add") fields.insert({"vector", "coefficient", "positions"});
    if (name == "scale") fields.insert({"feature", "multiplier", "positions", "dictionary"});
    if (name == "clamp") fields.insert({"feature", "value", "positions", "dictionary"});
    if (name == "patch") fields.insert({"source_prompt", "source_position", "positions"});
    if (lifecycle) {
        fields.insert({"since_step", "state", "error", "error_code"});
        const auto* state = value.find("state");
        if (!state || !one_of(*state, {"pending", "active", "ended"})) return false;
        if (auto* e = value.find("error"); e && !interpretation_text(*e, 1024)) return false;
        if (auto* e = value.find("error_code"); e && !interpretation_text(*e, 63, false)) return false;
    }
    for (const auto& item : value.members())
        if (!fields.count(item.first)) return false;
    for (const auto* key : {"from_step", "until_step", "since_step"})
        if (!opt_step(value.find(key))) return false;
    const auto* start = value.find("from_step");
    const auto* end = value.find("until_step");
    if (present(start) && present(end) && end->as_double() <= start->as_double()) return false;
    if (auto* positions = value.find("positions")) {
        if (!positions->is_array() || positions->size() > 32) return false;
        std::set<uint64_t> seen;
        for (const auto& item : positions->items()) {
            uint64_t index = 0;
            if (!int_in(item, 0, 31) || !item.get_u64(&index) || !seen.insert(index).second) return false;
        }
    }
    if (auto* dictionary = value.find("dictionary"); dictionary && !interpretation_text(*dictionary, 255, false))
        return false;
    if (name == "add") {
        const auto* vector = value.find("vector");
        const auto* coefficient = value.find("coefficient");
        if (!vector || !vector->is_array() || !vector->size() || vector->size() > 16384 ||
            !coefficient || !intervention_number(*coefficient))
            return false;
        for (const auto& item : vector->items())
            if (!intervention_number(item)) return false;
    }
    if (name == "scale" || name == "clamp") {
        const auto* feature = value.find("feature");
        const auto* number = value.find(name == "scale" ? "multiplier" : "value");
        if (!feature || !int_in(*feature, 0, 2147483647) || !number || !intervention_number(*number)) return false;
    }
    if (name == "patch") {
        const auto* prompt = value.find("source_prompt");
        if (!prompt || !interpretation_text(*prompt, 4096, false)) return false;
        if (auto* position = value.find("source_position"); position && !int_in(*position, 0, 31)) return false;
    }
    return value.dump().size() <= 262144;
}

bool intervene_request(const Json& value) {
    if (!value.is_object()) return false;
    for (const auto& item : value.members())
        if (item.first != "set" && item.first != "clear" && item.first != "clear_all") return false;
    const auto* set = value.find("set");
    const auto* clear = value.find("clear");
    const auto* all = value.find("clear_all");
    if (int(present(set)) + int(present(clear)) + int(present(all)) != 1) return false;
    if (present(set)) return intervention_spec(*set);
    if (present(clear)) return interpretation_text(*clear, 255, false);
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

bool active_entry(const Json& value) { return intervention_spec(value, true); }

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

}  // namespace

bool is_set_value(const Json& v) { return is_finite(v) || v.is_bool() || param_text(v); }

bool interpretation_object(const Json& v) {
    size_t count = 0;
    return v.is_object() && interpretation_value(v, 0, count) && v.dump().size() <= CALIPER_NET_INTERPRET_MAX_BYTES;
}

bool interpretation_request(const Json& v) {
    if (!interpretation_object(v) || v.size() != 3) return false;
    const auto* version = v.find("version");
    const auto* method = v.find("method");
    const auto* params = v.find("params");
    if (!version || !int_in(*version, 1, 1) || !method || !method->is_string() ||
        !params || !params->is_object())
        return false;
    const auto& m = method->as_string();
    std::set<std::string> allowed;
    if (m == "feature.dictionary" || m == "trace.status" || m == "trace.validate") {}
    else if (m == "patch.sweep") allowed = {"clean_prompt", "corrupted_prompt", "metric"};
    else if (m == "feature.show") allowed = {"layer", "feature", "node"};
    else if (m == "feature.examples") allowed = {"layer", "feature", "node", "limit"};
    else if (m == "trace.run") allowed = {"prompt", "max_targets", "max_nodes", "max_edges",
        "max_n_logits", "desired_logit_prob", "node_threshold", "edge_threshold", "batch_size", "update_interval"};
    else if (m == "trace.read") allowed = {"node_offset", "node_limit", "edge_offset", "edge_limit"};
    else return false;
    for (const auto& kv : params->members())
        if (!allowed.count(kv.first)) return false;
    const struct { const char* key; uint64_t low, high; } bounds[] = {
        {"layer", 0, 4095}, {"feature", 0, 2147483647}, {"node", 0, 8191}, {"limit", 0, 16},
        {"max_targets", 0, 5000u}, {"max_nodes", 1, 8192u}, {"max_edges", 0, 262144u},
        {"max_n_logits", 1, 10}, {"update_interval", 1, 128},
        {"node_offset", 0, 8192}, {"edge_offset", 0, 262144}, {"node_limit", 0, 256}, {"edge_limit", 0, 256}};
    for (const auto& b : bounds)
        if (auto* x = params->find(b.key); x && !int_in(*x, b.low, b.high)) return false;
    if (m == "feature.show" || m == "feature.examples")
        return (params->has("node") && !params->has("layer") && !params->has("feature")) ||
               (!params->has("node") && params->has("layer") && params->has("feature"));
    if (m == "patch.sweep") {
        for (auto* key : {"clean_prompt", "corrupted_prompt"}) {
            auto* prompt = params->find(key);
            if (!prompt || !interpretation_text(*prompt, CALIPER_NET_INTERPRET_STRING_MAX, false)) return false;
        }
        auto* metric = params->find("metric");
        if (!metric || !metric->is_object() || metric->size() != 2) return false;
        auto* positive = metric->find("positive_token");
        auto* negative = metric->find("negative_token");
        return positive && negative && int_in(*positive, 0, 2147483647) && int_in(*negative, 0, 2147483647) &&
               *positive != *negative;
    }
    if (m == "trace.run") {
        if (auto* batch = params->find("batch_size"); batch && !batch->is_null() && !int_in(*batch, 1, 512))
            return false;
        for (const auto* key : {"desired_logit_prob", "node_threshold", "edge_threshold"}) {
            if (const auto* x = params->find(key); x && (!is_finite(*x) || x->as_double() < 0 || x->as_double() > 1))
                return false;
        }
        auto* prompt = params->find("prompt");
        return prompt && interpretation_text(*prompt, CALIPER_NET_INTERPRET_STRING_MAX, false);
    }
    return true;
}

Controls decode_controls(const Frame& f) {
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

Control decode_control(const Frame& f) {
    const Json& m = f.meta;
    Control c;
    c.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    c.op = req(m, "op", is_op, "pause, resume, step, set, run or intervene").as_string();
    const Json* name = opt(m, "name", is_name, "a parameter or action name");
    const bool is_named = c.op == CALIPER_NET_OP_SET || c.op == CALIPER_NET_OP_RUN;
    if (is_named && !name) bad("'" + c.op + "' needs 'name'");
    if (c.op == CALIPER_NET_OP_SET) c.value = req(m, "value", is_set_value, "a number, a boolean or a choice");
    if (c.op == CALIPER_NET_OP_STEP)
        if (const Json* n = opt(m, "count", [](const Json& v) { return is_u32(v) && u64(v) >= 1; },
                                "a positive uint32"))
            c.count = u32(*n);
    if (c.op == CALIPER_NET_OP_INTERVENE)   // 3.2 draft
        c.intervene = req(m, "intervene", intervene_request,
                          "exactly one of {set: {point, op, from_step, until_step}}, {clear: point} and {clear_all: true}");
    if (c.op == CALIPER_NET_OP_INTERPRET)
        c.interpretation = req(m, "interpretation", interpretation_request, "a version 1 interpretation request");
    else if (m.has("interpretation")) bad("interpretation requires op interpret");
    if (is_named) c.name = name->as_string();
    return c;
}

ControlReply decode_control_reply(const Frame& f) {
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
    if (r.status == CALIPER_NET_STATUS_QUEUED && r.op != CALIPER_NET_OP_RUN && r.op != CALIPER_NET_OP_INTERPRET)
        bad("only 'run' is queued");
    r.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    if (const Json* v = opt(m, "name", is_name, "a parameter or action name")) r.name = v->as_string();
    if (const Json* v = opt(m, "value", is_set_value, "a number, a boolean or a choice")) r.value = *v;
    if (const Json* v = opt(m, "message", is_text, "a string of at most 1024 bytes")) r.message = v->as_string();
    if (err) r.error = *err;
    if (const Json* result = opt(m, "result", interpretation_object, "bounded structured result")) r.result = *result;
    if (m.has("result") && (r.op != CALIPER_NET_OP_INTERPRET || r.status != CALIPER_NET_STATUS_OK))
        bad("result requires a successful interpretation reply");
    if (r.op == CALIPER_NET_OP_INTERPRET && r.status == CALIPER_NET_STATUS_OK && r.result.is_null())
        bad("a successful interpretation reply requires result");
    return r;
}

Selection decode_selection(const Frame& f) {
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

void controls_to_meta(const Controls& c, Json& m, std::vector<Section>& sections) {
    m.set("run", c.run);
    m.set("params", c.params);
    m.set("actions", c.actions);
    if (c.stamp) sections.push_back(named(*c.stamp, "stamp"));   // 3.2 draft
    if (!c.interventions.is_null()) m.set("interventions", c.interventions);
}

void control_to_meta(const Control& c, Json& m) {
    m.set("request_id", c.request_id);
    m.set("op", c.op);
    if (c.name) m.set("name", *c.name);
    if (!c.value.is_null()) m.set("value", c.value);
    if (c.count) m.set("count", *c.count);
    if (!c.intervene.is_null()) m.set("intervene", c.intervene);
    if (!c.interpretation.is_null()) m.set("interpretation", c.interpretation);
}

void control_reply_to_meta(const ControlReply& r, Json& m) {
    m.set("request_id", r.request_id);
    m.set("op", r.op);
    m.set("status", r.status);
    if (r.name) m.set("name", *r.name);
    if (!r.value.is_null()) m.set("value", r.value);
    if (r.message) m.set("message", *r.message);
    if (!r.error.is_null()) m.set("error", r.error);
    if (!r.result.is_null()) m.set("result", r.result);
}

void selection_to_meta(const Selection& s, Json& m) {
    m.set("space", s.space);
    m.set("items", s.items);
    if (s.view) m.set("view", *s.view);
    if (s.truncated) m.set("truncated", true);
    if (!s.details.is_null()) m.set("details", s.details);
    if (s.details_error) m.set("details_error", *s.details_error);
}

}  // namespace detail
}  // namespace wire
}  // namespace caliper_net
