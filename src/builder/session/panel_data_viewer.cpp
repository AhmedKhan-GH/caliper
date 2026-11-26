// panel_data_viewer.cpp: what panel.data gives the browser's spec page for
// the instances of pieces on a scene panel and for a control of the layout
// (panel_data.h). Both are reads: the page draws the programs with its
// WebAssembly views and shows each control with its state, and nothing here
// changes the session.
#include <caliper_builder/session/panel_data.h>

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/state.h>
#include <caliper_builder/text/bound_text.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace caliper_builder {

namespace {

using nlohmann::json;

const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const std::uint8_t* p, std::size_t n) {
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 2 < n; i += 3) {
        const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8) | p[i + 2];
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += kB64[v & 63];
    }
    if (i + 1 == n) {
        const std::uint32_t v = std::uint32_t(p[i]) << 16;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == n) {
        const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8);
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

// FNV-1a over text, for the digest that goes into the panel's version.
void mix(std::uint64_t* h, const std::string& s) {
    for (unsigned char c : s) {
        *h ^= c;
        *h *= 1099511628211ull;
    }
    *h ^= 0xff;
    *h *= 1099511628211ull;
}

std::uint32_t line_count(const std::string& t) {
    const std::uint32_t n = std::uint32_t(std::count(t.begin(), t.end(), '\n'));
    return t.empty() || t.back() == '\n' ? n : n + 1;
}

// The value of a reference now: the cursor's fields from the history, else
// the session's values, else its default.
json read_reference(const spec::Reference& r, const ViewerInputs& in,
                    const spec::Spec* shown = nullptr, const RunState* run = nullptr) {
    if (r.kind == spec::Reference::Kind::control && shown) {
        const spec::Control* control = shown->find_control(r.target);
        const auto target = control ? spec::parse_control_target(control->bind) : std::nullopt;
        if (target && target->kind == spec::ControlTarget::Kind::param) {
            if (!run || !run->attached || r.field) return r.fallback;
            for (const auto& param : run->params)
                if (param.name == target->name)
                    return param.pending.is_null() ? param.value : param.pending;
            return r.fallback;
        }
    }
    if (r.kind == spec::Reference::Kind::state && r.target == spec::kCursorState && in.history) {
        const CursorState cur = in.history->cursor();
        const std::string f = r.field.value_or("step");
        if (f == "first") return cur.first;
        if (f == "latest") return cur.latest;
        if (f == "step") return cur.step;
    }
    return in.values ? in.values->value(r) : r.fallback;
}

json error_json(const std::string& code, const std::string& message) {
    return {{"code", code}, {"message", message}};
}

}  // namespace

json viewer_programs(const spec::Spec& spec, const std::string& panel_id, const ViewerInputs& in, std::string* digest) {
    namespace pg = programs;
    json out = json::array();
    std::uint64_t h = 1469598103934665603ull;
    std::uint64_t bytes = 0;
    // The step whose resources the panel shows: the newest while the cursor
    // is live, else the cursor's, as the app's scene panels draw them.
    std::int64_t step = std::numeric_limits<std::int64_t>::max();
    if (in.history) {
        const CursorState cur = in.history->cursor();
        if (!cur.live) step = cur.step;
    }
    for (const spec::Program& p : spec.programs) {
        if (!p.piece || p.panel.value_or("") != panel_id) continue;
        const spec::PieceRef& ref = *p.piece;
        json item = {{"id", p.id},
                     {"title", p.title ? json(*p.title) : json(nullptr)},
                     {"piece", {{"name", ref.name}, {"version", ref.version}, {"from", spec::piece_from_name(ref.from)}}},
                     {"state", "ready"},
                     {"error", nullptr},
                     {"instance", nullptr},
                     {"resources", json::array()}};
        auto refuse = [&](const std::string& state, const std::string& code, const std::string& message) {
            item["state"] = state;
            item["error"] = error_json(code, message);
            mix(&h, p.id + " " + state + " " + code);
            out.push_back(std::move(item));
        };
        if (out.size() >= kViewerProgramsMax) break;
        const std::string pin = ref.name + "@" + std::to_string(ref.version);
        if (ref.hash.empty()) {
            refuse("refused", "spec.missing_key", pin_without_hash_message(ref));
            continue;
        }
        // The pinned piece from the library, this workspace or the spec, by
        // its name and version only: the page never names a file.
        const FoundPiece f = in.library ? resolve_spec_pin(*in.library, in.workspace, spec, ref) : FoundPiece{};
        if (!in.library || f.error) {
            refuse("refused", f.error ? f.error.code : std::string(pg::kPieceMissing),
                   f.error ? f.error.message : "The piece " + pin + " is not available here.");
            continue;
        }
        // The hash of the text that goes out, as the app checks it before it
        // sends an instance to the script.
        const std::string hash = pg::piece_hash(f.piece);
        if (hash != ref.hash) {
            refuse("refused", pg::kPieceHashMismatch,
                   "The piece " + pin + " has the hash " + hash + ", and the program " + p.id + " pins " + ref.hash + ".");
            continue;
        }
        // A program of a foreign spec stays in review in the browser too: its
        // files never leave the app until a person reviews its hash.
        if (spec.provenance.foreign) {
            const std::string hex = hash.compare(0, 7, "sha256:") == 0 ? hash.substr(7) : hash;
            const pg::Verdict v = pg::check_provenance(spec.provenance, spec::program_target(p), hex);
            if (!v.ok()) {
                refuse("review", "program.needs_review", v.message);
                continue;
            }
        }
        if (!f.piece.draw) {
            refuse("refused", pg::kPieceBadManifest, "The piece " + pin + " has no draw half.");
            continue;
        }
        const pg::Resolved resolved = pg::resolve_imports(f.piece);
        if (!resolved.verdict.ok()) {
            refuse("refused", resolved.verdict.code, resolved.verdict.message);
            continue;
        }
        const std::string manifest = pg::canonical_manifest(f.piece);
        json manifest_json = json::parse(manifest, nullptr, false);
        if (manifest.empty() || manifest_json.is_discarded()) {
            refuse("refused", pg::kProgramNotBuilt, "The manifest of the piece " + pin + " has no canonical form yet.");
            continue;
        }
        auto file = [&](const char* name) {
            auto it = resolved.files.find(name);
            return it == resolved.files.end() ? std::string() : it->second;
        };
        // The instance form of PROGRAM set that the app sends a script, with
        // the params' values of now.
        json params = json::object();
        for (const auto& [name, place] : p.params) {
            double v = 0;
            if (!spec::is_reference(place)) {
                v = std::get<double>(place);
            } else {
                const spec::Reference& r = std::get<spec::Reference>(place);
                const json now = read_reference(r, in);
                if (now.is_number() && std::isfinite(now.get<double>())) v = now.get<double>();
                else if (r.fallback.is_number()) v = r.fallback.get<double>();
            }
            params[name] = v;
        }
        const std::string common_file = file("common.wgsl");
        json segments = json::array();
        for (const pg::Segment& seg : resolved.lib_segments)
            segments.push_back({{"part", seg.part}, {"first_line", seg.first_line}, {"lines", seg.lines}});
        if (!common_file.empty())
            segments.push_back({{"part", "common.wgsl"},
                                {"first_line", line_count(resolved.lib_text) + 1},
                                {"lines", line_count(common_file)}});
        json bind = json::object();
        for (const auto& [input, binding] : p.bind) bind[input] = binding;
        json inst = {{"op", "set"},
                     {"id", p.id},
                     {"piece", {{"name", ref.name}, {"version", ref.version}, {"hash", ref.hash}}},
                     {"manifest", std::move(manifest_json)},
                     {"bind", std::move(bind)},
                     {"params", params},
                     {"panel", panel_id},
                     {"segments", std::move(segments)},
                     {"common", resolved.lib_text + common_file},
                     {"draw", file("draw.wgsl")}};
        // The derive half goes too, because the pin's hash covers it; the
        // page's module checks that hash before it draws.
        if (f.piece.derive) inst["derive"] = file("derive.wgsl");
        item["instance"] = std::move(inst);
        item["params"] = params;
        mix(&h, p.id + " " + hash + " " + params.dump());
        // The newest kept version of each published resource at the step.
        if (in.history)
            for (const spec::Resource& r : f.piece.resources) {
                if (!r.publish) continue;
                const std::optional<ResourceVersion> v = in.history->resource_at(p.id + "/" + r.id, step);
                if (!v || !v->bytes) continue;
                if (bytes + v->nbytes() > kViewerResourceBytesMax) {
                    item["bounds"] = "resources cut at " + std::to_string(kViewerResourceBytesMax) + " bytes";
                    break;
                }
                bytes += v->nbytes();
                item["resources"].push_back({{"resource", r.id},
                                             {"step", v->step},
                                             {"version", std::to_string(v->serial)},
                                             {"count", v->count},
                                             {"stride", v->stride},
                                             {"width", v->width},
                                             {"height", v->height},
                                             {"format", v->format},
                                             {"elements", v->elements ? json(*v->elements) : json(nullptr)},
                                             {"base64", base64(v->bytes->data(), v->bytes->size())}});
                mix(&h, r.id + " " + std::to_string(v->step) + " " + std::to_string(v->serial));
            }
        out.push_back(std::move(item));
    }
    if (digest) {
        static const char kHex[] = "0123456789abcdef";
        std::string d(16, '0');
        for (int i = 15; i >= 0; --i, h >>= 4) d[std::size_t(i)] = kHex[h & 15];
        *digest = d;
    }
    return out;
}

json control_data_result(const spec::Spec& spec, const spec::Control& c, const ViewerInputs& in, RunControl* run,
                         const std::string& version) {
    using T = spec::ControlTarget::Kind;
    const std::optional<spec::ControlTarget> t = spec::parse_control_target(c.bind);
    json d = {{"id", c.id},
              {"kind", spec::control_kind_name(c.kind)},
              {"title", c.title.value_or(c.id)},
              {"bind", c.bind},
              {"value", nullptr},
              {"text", ""},
              {"enabled", false},
              {"why", ""}};
    const RunState st = run ? run->state() : RunState{};
    // What the viewer sees and may not change: every control acts on the
    // run or on the app's values, and the viewer origin only reads.
    d["why"] = "The browser viewer only reads: use the app to change this control.";
    if (t && (t->kind == T::intervene || t->kind == T::param || t->kind == T::action || t->kind == T::run_pause)) {
        if (!run || !st.attached) d["why"] = "No script is attached.";
    }
    if (t && t->kind == T::intervene) {
        bool on = false;
        std::string text = "ablate " + t->name;
        if (run && st.attached)
            for (const Intervention& i : run->interventions())
                if (i.point == t->name && i.op == "ablate" && i.state != "ended") {
                    on = true;
                    if (i.state == "active" && i.since_step) text += "  \xC2\xB7 since step " + std::to_string(*i.since_step);
                    else text += "  \xC2\xB7 at the next step";
                    break;
                }
        d["value"] = on;
        d["text"] = text;
    } else if (t && t->kind == T::param) {
        for (const RunParam& p : st.params)
            if (p.name == t->name) d["value"] = p.pending.is_null() ? p.value : p.pending;
        d["text"] = t->name;
    } else if (t && (t->kind == T::action || t->kind == T::run_pause)) {
        d["text"] = t->kind == T::run_pause ? (st.state == "paused" ? "Resume" : "Pause") : c.title.value_or(t->name);
    } else if (t && t->kind == T::cursor) {
        if (in.history) {
            const CursorState cur = in.history->cursor();
            d["value"] = {{"step", cur.step}, {"first", cur.first}, {"latest", cur.latest}, {"live", cur.live}};
        }
    } else if (c.kind == spec::ControlKind::readout) {
        if (c.options.bound_text) {
            const spec::BoundText& bt = *c.options.bound_text;
            std::vector<std::string> names;
            std::map<std::string, json> values;
            for (const auto& [name, field] : bt.fields) {
                names.push_back(name);
                values[name] = std::holds_alternative<spec::Reference>(field)
                                   ? read_reference(std::get<spec::Reference>(field), in, &spec, &st)
                                   : json(std::get<std::string>(field));
            }
            text::ParsedText parsed;
            const Error e = text::parse_bound_text(bt.format, names, &parsed);
            d["text"] = e ? bt.format : text::format_bound_text(parsed, values);
        }
        if (run && (c.bind == "param:output.status" || c.id == "trace_status")) {
            const json prog = run->progress_state();
            if (prog.is_object()) {
                d["progress"] = prog;
                const std::string live = format_progress_status(prog);
                if (!live.empty()) d["text"] = live;
            }
        }
        d["why"] = "";
    } else {
        // A control bound to `value` or to a state: the session's value, else its start.
        json v = in.values ? in.values->control_value(c.id) : json();
        if (v.is_null() && t && t->kind == T::state)
            for (const spec::StateVar& s : spec.states)
                if (s.id == t->name) v = spec::state_initial_value(s);
        if (v.is_null()) v = c.options.value ? json(*c.options.value) : c.options.value_other;
        if (v.is_null() && c.kind == spec::ControlKind::toggle) v = false;
        if (v.is_null() && (c.kind == spec::ControlKind::picker || c.kind == spec::ControlKind::segmented) &&
            !c.options.choices.empty())
            v = c.options.choices.front().value;
        d["value"] = v;
    }
    if (c.options.min) d["min"] = *c.options.min;
    if (c.options.max) d["max"] = *c.options.max;
    if (!c.options.choices.empty()) {
        json choices = json::array();
        for (const spec::Choice& ch : c.options.choices)
            choices.push_back({{"value", ch.value}, {"label", ch.label ? json(*ch.label) : json(nullptr)}});
        d["choices"] = std::move(choices);
    }
    return {{"panel", c.id},
            {"primitive", "control"},
            {"version", version},
            {"unchanged", false},
            {"error", nullptr},
            {"state", nullptr},
            {"bounds", json::array()},
            {"data", std::move(d)}};
}

}  // namespace caliper_builder
