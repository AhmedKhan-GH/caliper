// The current values of the inputs that references read
// (caliper_builder/session/values.h, design/instrument-interface.md).
//
// The store keeps a copy of the open spec's state variables, controls and
// `select` options, and the value of each state variable, of each control
// bound to `value` (or the last value a script reported for a control bound
// to a parameter), the last pick of each panel and the cursor. A control
// bound to a state variable holds no value of its own: it shows and writes
// the state's value, and a reference to it reads the state.
//
// Evaluation writes in one case only: a range with `follow` moves with its
// upper bound while its upper end equals that bound, and keeps its span. The
// store moves it after each change of an input, without an event, as a play
// move of the cursor adds none.
//
// Events are pushed after the store's lock is released.
#include <caliper_builder/session/values.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/spec/state.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <set>

namespace caliper_builder {

namespace spec {
// state.cpp: why `v` does not fit `s`, or empty when it fits.
std::string state_value_problem(const StateVar& s, const nlohmann::json& v);
}  // namespace spec

namespace {

using json = nlohmann::json;
using spec::ControlKind;
using spec::ControlTarget;
using spec::StateType;

constexpr double kEps = 1e-9;

bool integral(const json& v) {
    if (v.is_number_integer()) return true;
    if (!v.is_number_float()) return false;
    const double d = v.get<double>();
    return std::isfinite(d) && std::floor(d) == d && std::fabs(d) <= 9.007199254740992e15;
}

// Two values are the same value: equal JSON, or equal numbers.
bool same_value(const json& a, const json& b) {
    if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>();
    if (a.is_array() && b.is_array() && a.size() == b.size()) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!same_value(a[i], b[i])) return false;
        return true;
    }
    return a == b;
}

// A value in the form its state type keeps: an integer as an integer.
json normalized(StateType type, const json& v) {
    if (type == StateType::integer && integral(v)) return json(static_cast<std::int64_t>(v.get<double>()));
    if (type == StateType::number && v.is_number()) return json(v.get<double>());
    if (type == StateType::range && v.is_array() && v.size() == 2 && v[0].is_number() && v[1].is_number())
        return json::array({v[0].get<double>(), v[1].get<double>()});
    return v;
}

std::optional<ControlTarget> target_of(const spec::Control& c) { return spec::parse_control_target(c.bind); }

std::optional<std::string> bound_state_id(const spec::Control& c) {
    const auto t = target_of(c);
    if (t && t->kind == ControlTarget::Kind::state) return t->name;
    return std::nullopt;
}

// Every reference of `s` with the document path of its reader: a source
// step, a panel's options or slot, an entity, an annotation, a control's
// options, or the params of a program or a stage.
void each_reference(const spec::Spec& s,
                    const std::function<void(const std::string& reader, const spec::Reference& ref)>& f) {
    auto json_ref = [&](const std::string& reader, const json& v) {
        if (!spec::looks_like_reference(v)) return false;
        if (auto r = spec::parse_reference(v)) f(reader, *r);
        return true;
    };
    auto text = [&](const std::string& reader, const spec::BoundText& t) {
        for (const auto& [name, field] : t.fields)
            if (std::holds_alternative<spec::Reference>(field)) f(reader, std::get<spec::Reference>(field));
    };
    auto index = [](const std::string& base, std::size_t i) { return base + "[" + std::to_string(i) + "]"; };
    for (std::size_t i = 0; i < s.sources.size(); ++i)
        for (std::size_t j = 0; j < s.sources[i].transforms.size(); ++j)
            for (const auto& [key, ref] : s.sources[i].transforms[j].references)
                f(index(index("source", i) + ".transform", j), ref);
    for (std::size_t i = 0; i < s.panels.size(); ++i) {
        const spec::Panel& p = s.panels[i];
        const std::string pp = index("panel", i);
        for (const auto& [key, ref] : p.options.references) f(pp + ".options", ref);
        if (p.options.bound_text) text(pp + ".options", *p.options.bound_text);
        for (const auto& [slot, value] : p.bind)
            if (value.reference) f(pp + ".bind." + slot, *value.reference);
        for (std::size_t k = 0; k < p.entities.size(); ++k) {
            const spec::Entity& e = p.entities[k];
            const std::string ep = index(pp + ".entity", k);
            for (const auto& [slot, value] : e.slots)
                if (spec::is_reference(value)) f(ep, std::get<spec::Reference>(value));
            if (e.options.is_object())
                for (auto it = e.options.begin(); it != e.options.end(); ++it) json_ref(ep, it.value());
        }
        for (std::size_t k = 0; k < p.annotations.size(); ++k) {
            const spec::Annotation& a = p.annotations[k];
            const std::string ap = index(pp + ".annotation", k);
            json_ref(ap, a.value);
            if (!json_ref(ap, a.range) && a.range.is_array())
                for (const auto& v : a.range) json_ref(ap, v);
            if (!json_ref(ap, a.at) && a.at.is_array())
                for (const auto& v : a.at) json_ref(ap, v);
            if (a.text) text(ap, *a.text);
        }
    }
    for (std::size_t i = 0; i < s.controls.size(); ++i) {
        const spec::ControlOptions& o = s.controls[i].options;
        for (const auto& [key, ref] : o.references) f(index("control", i) + ".options", ref);
        if (o.bound_text) text(index("control", i) + ".options", *o.bound_text);
    }
    for (std::size_t i = 0; i < s.programs.size(); ++i)
        for (const auto& [name, value] : s.programs[i].params)
            if (spec::is_reference(value)) f(index("program", i) + ".params", std::get<spec::Reference>(value));
    for (std::size_t i = 0; i < s.stages.size(); ++i)
        for (const auto& [name, value] : s.stages[i].params)
            if (spec::is_reference(value)) f(index("stage", i) + ".params", std::get<spec::Reference>(value));
}

// The input that `ref` reads: a reference to a control bound to a state
// reads the state.
InputKey input_of(const spec::Reference& ref, const std::map<std::string, std::string>& control_states) {
    switch (ref.kind) {
        case spec::Reference::Kind::control: {
            const auto it = control_states.find(ref.target);
            if (it != control_states.end()) return InputKey{InputKey::Kind::state, it->second};
            return InputKey{InputKey::Kind::control, ref.target};
        }
        case spec::Reference::Kind::selection: return InputKey{InputKey::Kind::selection, ref.target};
        case spec::Reference::Kind::state:
            if (ref.target == spec::kCursorState) return InputKey{InputKey::Kind::cursor, ""};
            return InputKey{InputKey::Kind::state, ref.target};
    }
    return InputKey{};
}

std::map<std::string, std::string> control_states_of(const spec::Spec& s) {
    std::map<std::string, std::string> out;
    for (const auto& c : s.controls)
        if (auto st = bound_state_id(c)) out.emplace(c.id, *st);
    return out;
}

json cursor_field(const CursorState& c, const std::string& field) {
    if (field == "step") return c.step;
    if (field == "first") return c.first;
    if (field == "latest") return c.latest;
    if (field == "live") return c.live;
    if (field == "playing") return c.playing;
    if (field == "speed") return c.speed;
    return nullptr;
}

bool same_cursor(const CursorState& a, const CursorState& b) {
    return a.step == b.step && a.first == b.first && a.latest == b.latest && a.live == b.live &&
           a.playing == b.playing && a.speed == b.speed;
}

}  // namespace

struct ValueStore::Impl {
    // What a range with `follow` keeps between evaluations: its span, whether
    // its upper end stands at its upper bound, and that bound when it last
    // evaluated.
    struct Follow {
        double span = 0;
        bool following = true;
        std::optional<double> max;
    };

    mutable std::mutex mutex;
    bool opened = false;
    std::string applet;
    std::map<std::string, spec::StateVar> states;
    std::map<std::string, spec::Control> controls;
    std::map<std::string, std::string> control_states;   // control id to the state it binds
    std::map<std::string, spec::StateWrite> selects;      // panel id to its `select`
    std::set<std::string> panels;
    std::map<std::string, json> state_values;
    std::map<std::string, json> control_values;
    std::map<std::string, PickFields> picks;
    std::map<std::string, Follow> follows;
    CursorState cursor;
    std::map<InputKey, std::uint64_t> versions;
    std::uint64_t counter = 0;

    void bump(const InputKey& k) { versions[k] = ++counter; }

    // ---- reading ---------------------------------------------------------------

    std::optional<double> evaluate_bound(const spec::Control& c, const char* key) const {
        const auto ref = c.options.references.find(key);
        if (ref != c.options.references.end()) {
            json v = read(ref->second);
            if (v.is_null()) v = ref->second.fallback;
            if (v.is_number() && std::isfinite(v.get<double>())) return v.get<double>();
            return std::nullopt;
        }
        const std::optional<double>& literal = std::string_view(key) == "min" ? c.options.min : c.options.max;
        if (literal) return literal;
        const auto st = control_states.find(c.id);
        if (st != control_states.end()) {
            const auto s = states.find(st->second);
            if (s != states.end()) return std::string_view(key) == "min" ? s->second.min : s->second.max;
        }
        return std::nullopt;
    }

    // A range value clamped to the bounds of `c`, as a person sees it.
    json clamp_pair(const spec::Control& c, json v) const {
        if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number()) return v;
        double lo = v[0].get<double>(), hi = v[1].get<double>();
        const auto mn = evaluate_bound(c, "min"), mx = evaluate_bound(c, "max");
        if (mn) lo = std::max(lo, *mn), hi = std::max(hi, *mn);
        if (mx) lo = std::min(lo, *mx), hi = std::min(hi, *mx);
        return json::array({lo, hi});
    }

    // The value of the control `id`: its state's, or its own.
    json control_now(const std::string& id) const {
        const auto c = controls.find(id);
        if (c == controls.end()) return nullptr;
        const auto st = control_states.find(id);
        if (st != control_states.end()) {
            const auto v = state_values.find(st->second);
            return v == state_values.end() ? json() : v->second;
        }
        const auto v = control_values.find(id);
        if (v == control_values.end()) return nullptr;
        return c->second.kind == ControlKind::range ? clamp_pair(c->second, v->second) : v->second;
    }

    static json pair_field(const json& v, const std::optional<std::string>& field) {
        if (!field) return v;
        if (!v.is_array() || v.size() != 2) return nullptr;
        if (*field == "lo") return v[0];
        if (*field == "hi") return v[1];
        return nullptr;
    }

    // The value `ref` reads, without its default.
    json read(const spec::Reference& ref) const {
        switch (ref.kind) {
            case spec::Reference::Kind::state: {
                if (ref.target == spec::kCursorState) return cursor_field(cursor, ref.field.value_or("step"));
                const auto v = state_values.find(ref.target);
                return v == state_values.end() ? json() : pair_field(v->second, ref.field);
            }
            case spec::Reference::Kind::control: return pair_field(control_now(ref.target), ref.field);
            case spec::Reference::Kind::selection: {
                const auto p = picks.find(ref.target);
                if (p == picks.end()) return nullptr;
                if (ref.entity && p->second.entity != *ref.entity) return nullptr;
                const json fields = pick_fields_json(p->second);
                if (!ref.field) return fields;
                const auto f = fields.find(*ref.field);
                return f == fields.end() ? json() : *f;
            }
        }
        return nullptr;
    }

    // ---- writing ---------------------------------------------------------------

    // The value that `c` takes for `in`, as a person's edit gives it: a
    // number clamped and snapped, a range ordered, clamped and at least its
    // min_span wide, or a choice. An error when the value does not fit.
    Error fit(const spec::Control& c, const json& in, json* out) const {
        const spec::ControlOptions& o = c.options;
        auto bad = [&](const std::string& what) {
            return Error("state.bad_value", "The control " + c.id + " takes " + what + ".", "args.value");
        };
        auto is_choice = [&](const json& v) {
            for (const auto& ch : o.choices)
                if (same_value(ch.value, v)) return true;
            return o.choices.empty() && o.choices_from.has_value() && (v.is_string() || integral(v));
        };
        const auto mn = evaluate_bound(c, "min"), mx = evaluate_bound(c, "max");
        switch (c.kind) {
            case ControlKind::slider: {
                if (!in.is_number() || !std::isfinite(in.get<double>())) return bad("a number");
                double v = in.get<double>();
                if (o.step && *o.step > 0 && mn) v = *mn + std::round((v - *mn) / *o.step) * *o.step;
                if (mn) v = std::max(v, *mn);
                if (mx) v = std::min(v, *mx);
                *out = v;
                return {};
            }
            case ControlKind::range: {
                if (!in.is_array() || in.size() != 2 || !in[0].is_number() || !in[1].is_number() ||
                    !std::isfinite(in[0].get<double>()) || !std::isfinite(in[1].get<double>()))
                    return bad("a pair [lo, hi] of numbers");
                double lo = std::min(in[0].get<double>(), in[1].get<double>());
                double hi = std::max(in[0].get<double>(), in[1].get<double>());
                if (mn) lo = std::max(lo, *mn), hi = std::max(hi, *mn);
                if (mx) lo = std::min(lo, *mx), hi = std::min(hi, *mx);
                if (o.min_span && hi - lo < *o.min_span) {
                    hi = mx ? std::min(*mx, lo + *o.min_span) : lo + *o.min_span;
                    if (hi - lo < *o.min_span) lo = mn ? std::max(*mn, hi - *o.min_span) : hi - *o.min_span;
                }
                *out = json::array({lo, hi});
                return {};
            }
            case ControlKind::toggle:
                if (!in.is_boolean()) return bad("true or false");
                *out = in;
                return {};
            case ControlKind::picker:
                if (!is_choice(in)) return bad("one of its choices");
                *out = in;
                return {};
            case ControlKind::text: {
                if (!in.is_string()) return bad("a text");
                const std::size_t limit =
                    o.max_length ? static_cast<std::size_t>(std::max<std::int64_t>(*o.max_length, 0))
                                 : spec::limits::kStateTextBytes;
                if (in.get_ref<const std::string&>().size() > limit)
                    return bad("a text of at most " + std::to_string(limit) + " bytes");
                *out = in;
                return {};
            }
            case ControlKind::segmented: {
                if (o.multiple && *o.multiple) {
                    if (!in.is_array()) return bad("a list of its choices");
                    for (const auto& v : in)
                        if (!is_choice(v)) return bad("a list of its choices");
                    // In the order of the choices, each once.
                    json list = json::array();
                    for (const auto& ch : o.choices)
                        for (const auto& v : in)
                            if (same_value(ch.value, v)) {
                                list.push_back(ch.value);
                                break;
                            }
                    *out = list;
                    return {};
                }
                if (!is_choice(in)) return bad("one of its choices");
                *out = in;
                return {};
            }
            default: break;
        }
        return Error("control.not_settable", "The control " + c.id + " holds no value that control.set can set.",
                     "args.id");
    }

    // Stores a state's value; returns the event when it changed.
    std::optional<Event> store_state(const std::string& id, const json& value, const std::string& writer,
                                     CommandOrigin origin) {
        const spec::StateVar& s = states.at(id);
        const json now = normalized(s.type, value);
        json& slot = state_values[id];
        if (same_value(slot, now) && slot.type() == now.type()) return std::nullopt;
        const json previous = slot;
        slot = now;
        bump(InputKey{InputKey::Kind::state, id});
        for (const auto& [cid, sid] : control_states)
            if (sid == id) {
                bump(InputKey{InputKey::Kind::control, cid});
                note_set(cid, now);
            }
        Event e;
        e.kind = EventKind::state;
        e.origin = command_origin_name(origin);
        if (controls.count(writer) || panels.count(writer)) e.panel = writer;
        e.fields = json{{"state", id}, {"value", now}, {"previous", previous}, {"writer", writer}};
        return e;
    }

    // After a person or a command set the value of a range with `follow`:
    // its span, and whether its upper end stands at its upper bound.
    void note_set(const std::string& control, const json& v) {
        const auto f = follows.find(control);
        if (f == follows.end() || !v.is_array() || v.size() != 2) return;
        const double lo = v[0].get<double>(), hi = v[1].get<double>();
        f->second.span = hi - lo;
        const auto mx = evaluate_bound(controls.at(control), "max");
        f->second.following = !mx || hi >= *mx - kEps;
        f->second.max = mx;
    }

    // Moves each range with `follow` whose upper bound moved while its upper
    // end stood on it. No event, as for a play move of the cursor.
    void refresh_follows() {
        for (auto& [id, f] : follows) {
            const spec::Control& c = controls.at(id);
            const auto mx = evaluate_bound(c, "max");
            if (!mx) continue;
            if (!f.max) {
                // The first evaluation: a start value whose upper end reaches
                // the bound follows it.
                const json v = control_now_raw(id);
                if (v.is_array() && v.size() == 2) f.following = v[1].get<double>() >= *mx - kEps;
            }
            const bool moved = !f.max || std::fabs(*f.max - *mx) > kEps;
            f.max = mx;
            if (!moved || !f.following) continue;
            const auto mn = evaluate_bound(c, "min");
            double lo = *mx - f.span;
            if (mn) lo = std::max(lo, std::min(*mn, *mx));
            const json now = json::array({lo, *mx});
            const auto st = control_states.find(id);
            if (st != control_states.end()) {
                json& slot = state_values[st->second];
                if (!same_value(slot, now)) {
                    slot = now;
                    bump(InputKey{InputKey::Kind::state, st->second});
                    bump(InputKey{InputKey::Kind::control, id});
                }
            } else {
                json& slot = control_values[id];
                if (!same_value(slot, now)) {
                    slot = now;
                    bump(InputKey{InputKey::Kind::control, id});
                }
            }
        }
    }

    json control_now_raw(const std::string& id) const {
        const auto st = control_states.find(id);
        if (st != control_states.end()) {
            const auto v = state_values.find(st->second);
            return v == state_values.end() ? json() : v->second;
        }
        const auto v = control_values.find(id);
        return v == control_values.end() ? json() : v->second;
    }

    // The value a control bound to `value` starts with.
    json start_value(const spec::Control& c) const {
        const spec::ControlOptions& o = c.options;
        if (o.value) return *o.value;
        if (!o.value_other.is_null()) return o.value_other;
        switch (c.kind) {
            case ControlKind::slider: return evaluate_bound(c, "min").value_or(0.0);
            case ControlKind::range: {
                const auto mn = evaluate_bound(c, "min"), mx = evaluate_bound(c, "max");
                return json::array({mn.value_or(0.0), mx.value_or(mn.value_or(0.0))});
            }
            case ControlKind::toggle: return false;
            case ControlKind::picker: return o.choices.empty() ? json() : o.choices.front().value;
            case ControlKind::text: return "";
            case ControlKind::segmented:
                if (o.multiple && *o.multiple) return json::array();
                return o.choices.empty() ? json() : o.choices.front().value;
            default: return nullptr;
        }
    }

    void push(EventSink* events, std::vector<Event>& out) {
        if (!events) return;
        for (auto& e : out) events->push(std::move(e));
    }
};

ValueStore::ValueStore() : impl_(std::make_unique<Impl>()) {}
ValueStore::~ValueStore() = default;

void ValueStore::reset(const spec::Spec& spec) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl& m = *impl_;
    // Another applet starts over; an edit of the same one keeps each value
    // that the new spec still has with the same type.
    const bool same = m.opened && m.applet == spec.applet.id;
    m.opened = true;
    m.applet = spec.applet.id;
    const auto old_states = std::move(m.states);
    const auto old_controls = std::move(m.controls);
    const auto old_state_values = std::move(m.state_values);
    const auto old_control_values = std::move(m.control_values);
    const auto old_follows = std::move(m.follows);
    m.states.clear();
    m.controls.clear();
    m.control_states.clear();
    m.selects.clear();
    m.panels.clear();
    m.state_values.clear();
    m.control_values.clear();
    m.follows.clear();

    for (const auto& s : spec.states) {
        if (s.id == spec::kCursorState) continue;
        m.states.emplace(s.id, s);
        json v = spec::state_initial_value(s);
        if (same) {
            const auto os = old_states.find(s.id);
            const auto ov = old_state_values.find(s.id);
            if (os != old_states.end() && os->second.type == s.type && ov != old_state_values.end() &&
                spec::state_value_problem(s, ov->second).empty())
                v = ov->second;
        }
        m.state_values[s.id] = normalized(s.type, v);
    }
    for (const auto& c : spec.controls) {
        m.controls.emplace(c.id, c);
        if (auto st = bound_state_id(c); st && m.states.count(*st)) m.control_states.emplace(c.id, *st);
    }
    for (const auto& p : spec.panels) {
        m.panels.insert(p.id);
        if (p.options.select) m.selects.emplace(p.id, *p.options.select);
    }
    for (const auto& c : spec.controls) {
        const auto t = target_of(c);
        if (!t || m.control_states.count(c.id)) continue;
        if (t->kind == ControlTarget::Kind::param) {
            // The value the script reported stays while the control binds the same parameter.
            const auto oc = old_controls.find(c.id);
            const auto ov = old_control_values.find(c.id);
            if (same && oc != old_controls.end() && oc->second.bind == c.bind && ov != old_control_values.end())
                m.control_values[c.id] = ov->second;
            continue;
        }
        if (t->kind != ControlTarget::Kind::value) continue;
        json v = m.start_value(c);
        const auto oc = old_controls.find(c.id);
        const auto ov = old_control_values.find(c.id);
        if (same && oc != old_controls.end() && oc->second.kind == c.kind && oc->second.bind == c.bind &&
            ov != old_control_values.end()) {
            json kept;
            if (!m.fit(c, ov->second, &kept)) v = kept;
        }
        if (!v.is_null()) m.control_values[c.id] = v;
    }
    for (const auto& c : spec.controls) {
        if (c.kind != ControlKind::range || !c.options.follow || !*c.options.follow) continue;
        const json v = m.control_now_raw(c.id);
        Impl::Follow f;
        if (v.is_array() && v.size() == 2) f.span = v[1].get<double>() - v[0].get<double>();
        const auto of = old_follows.find(c.id);
        if (same && of != old_follows.end()) f = of->second;
        m.follows.emplace(c.id, f);
    }
    if (!same) m.picks.clear();
    for (auto it = m.picks.begin(); it != m.picks.end();)
        it = m.panels.count(it->first) ? std::next(it) : m.picks.erase(it);
    m.refresh_follows();
    // Every reader evaluates again once after a reset.
    for (const auto& [id, v] : m.state_values) m.bump(InputKey{InputKey::Kind::state, id});
    for (const auto& [id, c] : m.controls) m.bump(InputKey{InputKey::Kind::control, id});
    for (const auto& id : m.panels) m.bump(InputKey{InputKey::Kind::selection, id});
}

nlohmann::json ValueStore::value(const spec::Reference& ref) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    json v = impl_->read(ref);
    return v.is_null() ? ref.fallback : v;
}

std::uint64_t ValueStore::version(const spec::Reference& ref) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->versions.find(input_of(ref, impl_->control_states));
    return it == impl_->versions.end() ? 0 : it->second;
}

Error ValueStore::set_state(const std::string& id, const nlohmann::json& value, const std::string& writer,
                            CommandOrigin origin, EventSink* events) {
    std::vector<Event> out;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        Impl& m = *impl_;
        if (id == spec::kCursorState)
            return Error("state.read_only",
                         "The state cursor is the time cursor; move it with cursor.set or cursor.step.", "args.id");
        const auto s = m.states.find(id);
        if (s == m.states.end()) return Error("state.not_found", "The open spec has no state variable " + id + ".", "args.id");
        const std::string why = spec::state_value_problem(s->second, value);
        if (!why.empty())
            return Error("state.bad_value", "The value for the state " + id + " " + why + ".", "args.value");
        if (auto e = m.store_state(id, value, writer, origin)) out.push_back(std::move(*e));
        m.refresh_follows();
    }
    impl_->push(events, out);
    return {};
}

Error ValueStore::set_control(const std::string& id, const nlohmann::json& value, CommandOrigin origin,
                              EventSink* events) {
    std::vector<Event> out;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        Impl& m = *impl_;
        const auto c = m.controls.find(id);
        if (c == m.controls.end()) return Error("control.not_found", "The open spec has no control " + id + ".", "args.id");
        const auto t = target_of(c->second);
        if (t && t->kind == ControlTarget::Kind::param) {
            // The value the script reported, which a reference to the control
            // reads; the control's own widget sends and records the change.
            if (!same_value(m.control_values[id], value)) {
                m.control_values[id] = value;
                m.bump(InputKey{InputKey::Kind::control, id});
                m.refresh_follows();
            }
            return {};
        }
        if (!t || (t->kind != ControlTarget::Kind::value && t->kind != ControlTarget::Kind::state))
            return Error("control.not_settable",
                         "The control " + id + " binds " + c->second.bind +
                             "; control.set sets a control bound to value or to a state.",
                         "args.id");
        json fitted;
        if (Error err = m.fit(c->second, value, &fitted)) return err;
        const auto st = m.control_states.find(id);
        if (st != m.control_states.end()) {
            const spec::StateVar& s = m.states.at(st->second);
            fitted = normalized(s.type, s.type == StateType::integer && fitted.is_number()
                                            ? json(std::round(fitted.get<double>()))
                                            : fitted);
            const std::string why = spec::state_value_problem(s, fitted);
            if (!why.empty())
                return Error("state.bad_value", "The value for the state " + s.id + " " + why + ".", "args.value");
        } else if (t->kind != ControlTarget::Kind::value) {
            return Error("control.not_settable", "The control " + id + " binds a state that the spec does not have.",
                         "args.id");
        }
        Event e;
        e.kind = EventKind::control;
        e.panel = id;
        e.origin = command_origin_name(origin);
        e.fields = json{{"control", id}, {"bind", c->second.bind}, {"value", fitted}};
        out.push_back(std::move(e));
        if (st != m.control_states.end()) {
            if (auto se = m.store_state(st->second, fitted, id, origin)) out.push_back(std::move(*se));
        } else if (!same_value(m.control_values[id], fitted)) {
            m.control_values[id] = fitted;
            m.bump(InputKey{InputKey::Kind::control, id});
        }
        m.note_set(id, fitted);
        m.refresh_follows();
    }
    impl_->push(events, out);
    return {};
}

nlohmann::json ValueStore::control_value(const std::string& id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->control_now(id);
}

void ValueStore::record_pick(const std::string& panel, const PickFields& fields, CommandOrigin origin,
                             EventSink* events) {
    std::vector<Event> out;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        Impl& m = *impl_;
        m.picks[panel] = fields;
        m.bump(InputKey{InputKey::Kind::selection, panel});
        const auto sel = m.selects.find(panel);
        if (sel != m.selects.end() && (!sel->second.entity || *sel->second.entity == fields.entity) &&
            m.states.count(sel->second.state)) {
            const json all = pick_fields_json(fields);
            const auto v = all.find(sel->second.field);
            const spec::StateVar& s = m.states.at(sel->second.state);
            if (v != all.end() && spec::state_value_problem(s, *v).empty())
                if (auto e = m.store_state(s.id, *v, panel, origin)) out.push_back(std::move(*e));
        }
        m.refresh_follows();
    }
    impl_->push(events, out);
}

std::optional<PickFields> ValueStore::last_pick(const std::string& panel) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->picks.find(panel);
    if (it == impl_->picks.end()) return std::nullopt;
    return it->second;
}

void ValueStore::clear_pick(const std::string& panel) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->picks.erase(panel)) {
        impl_->bump(InputKey{InputKey::Kind::selection, panel});
        impl_->refresh_follows();
    }
}

void ValueStore::set_cursor(const CursorState& cursor) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (same_cursor(impl_->cursor, cursor)) return;
    impl_->cursor = cursor;
    impl_->bump(InputKey{InputKey::Kind::cursor, ""});
    impl_->refresh_follows();
}

nlohmann::json ValueStore::describe_states(const spec::Spec& spec) const {
    const auto control_states = control_states_of(spec);
    std::map<std::string, std::vector<std::string>> readers, writers;
    each_reference(spec, [&](const std::string& reader, const spec::Reference& ref) {
        const InputKey k = input_of(ref, control_states);
        if (k.kind != InputKey::Kind::state) return;
        auto& list = readers[k.id];
        if (std::find(list.begin(), list.end(), reader) == list.end()) list.push_back(reader);
    });
    for (std::size_t i = 0; i < spec.controls.size(); ++i)
        if (auto st = bound_state_id(spec.controls[i])) writers[*st].push_back("control[" + std::to_string(i) + "]");
    for (std::size_t i = 0; i < spec.panels.size(); ++i)
        if (spec.panels[i].options.select)
            writers[spec.panels[i].options.select->state].push_back("panel[" + std::to_string(i) + "].options.select");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    json out = json::array();
    for (const auto& s : spec.states) {
        const auto v = impl_->state_values.find(s.id);
        out.push_back(json{{"id", s.id},
                           {"type", spec::state_type_name(s.type)},
                           {"value", v == impl_->state_values.end() ? spec::state_initial_value(s) : v->second},
                           {"default", spec::state_initial_value(s)},
                           {"writers", writers[s.id]},
                           {"readers", readers[s.id]}});
    }
    return out;
}

nlohmann::json pick_fields_json(const PickFields& fields) {
    nlohmann::json out = {{"entity", fields.entity}};
    if (fields.index) out["index"] = *fields.index;
    if (fields.row_id) out["row_id"] = *fields.row_id;
    if (fields.row) out["row"] = *fields.row;
    if (fields.column) out["column"] = *fields.column;
    if (fields.position) {
        out["x"] = (*fields.position)[0];
        out["y"] = (*fields.position)[1];
        out["z"] = (*fields.position)[2];
    }
    if (fields.value) out["value"] = *fields.value;
    if (fields.step) out["step"] = *fields.step;
    if (fields.annotation) out["annotation"] = *fields.annotation;
    return out;
}

// ---- DependencyIndex ------------------------------------------------------------
// The readers of each input. A control is a reader of its own value, and of
// the state it binds, under the path "control[i]"; its options are the reader
// "control[i].options". The cycle graph follows from these: the options of a
// control that read an input lead to every input whose readers hold the
// control itself, which is what the control writes.

void DependencyIndex::rebuild(const spec::Spec& spec) {
    readers_.clear();
    const auto control_states = control_states_of(spec);
    auto add = [&](const InputKey& k, const std::string& reader) {
        auto& list = readers_[k];
        if (std::find(list.begin(), list.end(), ReaderId{reader}) == list.end()) list.push_back(ReaderId{reader});
    };
    each_reference(spec, [&](const std::string& reader, const spec::Reference& ref) {
        add(input_of(ref, control_states), reader);
    });
    for (std::size_t i = 0; i < spec.controls.size(); ++i) {
        const spec::Control& c = spec.controls[i];
        const std::string self = "control[" + std::to_string(i) + "]";
        add(InputKey{InputKey::Kind::control, c.id}, self);
        if (auto st = bound_state_id(c)) add(InputKey{InputKey::Kind::state, *st}, self);
    }
    // The generation of a source: the panels and entities that bind it, and
    // the sources that read it with `from`.
    auto source_of = [](const std::string& binding) { return binding.substr(0, binding.find('.')); };
    for (std::size_t i = 0; i < spec.sources.size(); ++i)
        if (spec.sources[i].from)
            add(InputKey{InputKey::Kind::source, *spec.sources[i].from}, "source[" + std::to_string(i) + "]");
    for (std::size_t i = 0; i < spec.panels.size(); ++i) {
        const spec::Panel& p = spec.panels[i];
        const std::string pp = "panel[" + std::to_string(i) + "]";
        for (const auto& [slot, value] : p.bind)
            for (const auto& b : value.bindings) add(InputKey{InputKey::Kind::source, source_of(b)}, pp + ".bind." + slot);
        for (std::size_t k = 0; k < p.entities.size(); ++k) {
            const spec::Entity& e = p.entities[k];
            const std::string ep = pp + ".entity[" + std::to_string(k) + "]";
            if (e.source) add(InputKey{InputKey::Kind::source, *e.source}, ep);
            for (const auto& [slot, value] : e.slots)
                if (!spec::is_reference(value))
                    add(InputKey{InputKey::Kind::source, source_of(std::get<std::string>(value))}, ep);
        }
    }
}

std::vector<ReaderId> DependencyIndex::readers_of(const InputKey& input) const {
    auto it = readers_.find(input);
    return it == readers_.end() ? std::vector<ReaderId>{} : it->second;
}

std::vector<spec::Diagnostic> DependencyIndex::cycles() const {
    // What each control writes: the inputs whose readers hold "control[i]".
    std::map<std::string, std::vector<InputKey>> writes;
    for (const auto& [input, readers] : readers_) {
        if (input.kind != InputKey::Kind::control && input.kind != InputKey::Kind::state) continue;
        for (const auto& r : readers)
            if (r.path.rfind("control[", 0) == 0 && r.path.back() == ']') writes[r.path].push_back(input);
    }
    std::map<InputKey, std::vector<std::pair<InputKey, std::string>>> edges;   // to the input, through the options
    for (const auto& [input, readers] : readers_)
        for (const auto& r : readers) {
            const std::string suffix = ".options";
            if (r.path.size() <= suffix.size() || r.path.compare(r.path.size() - suffix.size(), suffix.size(), suffix))
                continue;
            const std::string control = r.path.substr(0, r.path.size() - suffix.size());
            const auto w = writes.find(control);
            if (w == writes.end()) continue;
            for (const InputKey& to : w->second) edges[input].push_back({to, r.path});
        }
    std::vector<spec::Diagnostic> out;
    std::map<InputKey, int> mark;
    std::vector<std::pair<InputKey, std::string>> stack;   // the input, and the options that led to it
    std::set<std::string> reported;
    std::function<void(const InputKey&)> visit = [&](const InputKey& n) {
        mark[n] = 1;
        for (const auto& [next, through] : edges[n]) {
            if (mark[next] == 1) {
                // The cycle runs from `next` on the stack to here, then back.
                std::vector<std::string> items{through};
                for (auto it = stack.rbegin(); it != stack.rend() && !(it->first == next); ++it) items.push_back(it->second);
                std::sort(items.begin(), items.end());
                items.erase(std::unique(items.begin(), items.end()), items.end());
                std::string key, names;
                for (const auto& s : items) {
                    key += s + "\n";
                    names += (names.empty() ? "" : ", ") + s;
                }
                if (!reported.insert(key).second) continue;
                spec::Diagnostic d;
                d.code = "spec.reference_cycle";
                d.message = "The options of a control read the value that it writes, through " + names + ".";
                d.path = items.front();
                out.push_back(std::move(d));
            } else if (mark[next] == 0) {
                stack.push_back({next, through});
                visit(next);
                stack.pop_back();
            }
        }
        mark[n] = 2;
    };
    std::vector<InputKey> nodes;
    for (const auto& [n, e] : edges) nodes.push_back(n);
    for (const auto& n : nodes)
        if (mark[n] == 0) visit(n);
    return out;
}

}  // namespace caliper_builder
