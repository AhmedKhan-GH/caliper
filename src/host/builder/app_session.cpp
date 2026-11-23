// The builder session in the app (app_session.h).
#include "builder/app_session.h"

#if CALIPER_WITH_BUILDER
#include "api/session_gateway.h"
#include "builder/panel_renderer.h"

#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/session/session.h>

#include "caliper/live_net.h"   // the ops and run states of wire protocol 3.1
#include "live/live_attach.h"     // RunLink, Controls

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <limits>
#include <fstream>
#include <optional>
#include <thread>
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <utility>

namespace caliper_host {

namespace {

// `path` with the home folder written as "~", as the Activity pane shows paths.
std::string shown_path(const std::string& path) {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    if (!home || !*home) return path;
    std::string h = home;
    while (h.size() > 1 && (h.back() == '/' || h.back() == '\\')) h.pop_back();
    if (path.compare(0, h.size(), h) != 0) return path;
    if (path.size() == h.size()) return "~";
    const char next = path[h.size()];
    if (next != '/' && next != '\\') return path;
    return "~" + path.substr(h.size());
}

}  // namespace

#if CALIPER_WITH_BUILDER

namespace cb = caliper_builder;
using nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

std::string base64(const std::string& in) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = (uint32_t)(uint8_t)in[i] << 16 | (uint32_t)(uint8_t)in[i + 1] << 8 | (uint8_t)in[i + 2];
        out += k[v >> 18 & 63];
        out += k[v >> 12 & 63];
        out += k[v >> 6 & 63];
        out += k[v & 63];
    }
    if (i + 1 == in.size()) {
        const uint32_t v = (uint32_t)(uint8_t)in[i] << 16;
        out += k[v >> 18 & 63];
        out += k[v >> 12 & 63];
        out += "==";
    } else if (i + 2 == in.size()) {
        const uint32_t v = (uint32_t)(uint8_t)in[i] << 16 | (uint32_t)(uint8_t)in[i + 1] << 8;
        out += k[v >> 18 & 63];
        out += k[v >> 12 & 63];
        out += k[v >> 6 & 63];
        out += '=';
    }
    return out;
}

cb::CommandOutput failed(const std::string& code, const std::string& message, const std::string& where = {}) {
    return cb::CommandOutput::fail(cb::Error(code, message, where));
}

// The commands that may take long enough to hold a frame: they read files or
// run queries. history.redo of an import runs its loaders again, but the
// history commands must run on the frame thread (design/command-layer.md).
bool runs_on_worker(const cb::CommandInvocation& inv) {
    const std::string& n = inv.name;
    if (n == "data.import" || n == "data.query" || n == "loader.run") return true;
    if (n == "loader.create" || n == "loader.update") {
        const auto it = inv.args.find("run");
        return it != inv.args.end() && it->is_boolean() && it->get<bool>();
    }
    return false;
}

// The jobs tray's label of a command on the worker: its name and, when it
// names one, the loader.
std::string job_label(const cb::CommandInvocation& inv) {
    const auto it = inv.args.find("id");
    if (it != inv.args.end() && it->is_string()) return inv.name + " " + it->get<std::string>();
    return inv.name;
}

json parse_object(const std::string& text) {
    json j = json::parse(text, nullptr, false);
    return j.is_object() ? j : json();
}

}  // namespace

// What the API's gateway and its held commands reach the session through:
// null once the session closes, so a command held by the server after that
// answers command.shutdown instead of touching a closed session.
struct GatewayCore {
    std::mutex m;
    AppSession::Impl* im = nullptr;
};

// ---- the app's run control ------------------------------------------------------------

namespace {

// A scalar of wire protocol 3.1 as the command table's JSON, and back.
json from_net(const caliper_net::Json& v) {
    using T = caliper_net::Json::Type;
    switch (v.type()) {
        case T::Bool: return v.as_bool();
        case T::Int: {
            int64_t i = 0;
            return v.get_i64(&i) ? json(i) : json(v.as_double());
        }
        case T::UInt: {
            uint64_t u = 0;
            return v.get_u64(&u) ? json(u) : json(v.as_double());
        }
        case T::Double: return v.as_double();
        case T::String: return v.as_string();
        default: return nullptr;
    }
}

caliper_net::Json to_net(const json& v) {
    if (v.is_boolean()) return caliper_net::Json(v.get<bool>());
    if (v.is_number_unsigned()) return caliper_net::Json(v.get<uint64_t>());
    if (v.is_number_integer()) return caliper_net::Json(v.get<int64_t>());
    if (v.is_number_float()) return caliper_net::Json(v.get<double>());
    if (v.is_string()) return caliper_net::Json(v.get<std::string>());
    return caliper_net::Json();
}

// RunControls' reasons are lower-case clauses; a command's message is a sentence.
std::string sentence(std::string why) {
    if (why.empty()) return why;
    why[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(why[0])));
    if (why.back() != '.') why += '.';
    return why;
}

// The CONTROL op of an intervention (caliper_live::kOpIntervene, which this
// file does not include because live_controls.h needs ImGui).
constexpr const char* kOpIntervene = "intervene";

// Any JSON value between the wire's Json and the command table's JSON.
json from_net_json(const caliper_net::Json& v) {
    if (v.is_null()) return json();
    json out = json::parse(v.dump(), nullptr, false);
    return out.is_discarded() ? json() : out;
}

caliper_net::Json to_net_json(const json& v) {
    const std::string text = v.dump();
    caliper_net::Json out;
    std::string err;
    if (!caliper_net::Json::parse(text.data(), text.size(), &out, &err)) return caliper_net::Json();
    return out;
}

// The CONTROLS key "interventions" of the attached script ({points, active})
// as the command table's JSON, or null when the script sent none.
json interventions_json(const caliper_live::Controls* c) {
    return c ? from_net_json(c->interventions) : json();
}

std::optional<std::int64_t> opt_int(const json& o, const char* key) {
    const auto it = o.find(key);
    if (it == o.end() || !it->is_number_integer()) return std::nullopt;
    return it->get<std::int64_t>();
}

void push_run(cb::EventSink& sink, json fields) {
    cb::Event e;
    e.kind = cb::EventKind::run;
    e.origin = "script";
    e.fields = std::move(fields);
    sink.push(std::move(e));
}

}  // namespace

cb::RunState AppRunControl::state() {
    cb::RunState st;
    if (!link_ || !link_->attached()) return st;
    st.attached = true;
    st.script = link_->script();
    const caliper_live::ControlAccess access = link_->access();
    const caliper_live::Controls* c = link_->controls();
    st.control = access.allowed && c != nullptr;
    if (!access.allowed)
        st.why_not = sentence(access.why.empty() ? "the connection takes no control" : access.why);
    else if (!c)
        st.why_not = "The script has sent no controls yet.";
    if (!c) return st;
    st.state = c->run_state;
    if (c->has_step) st.step = static_cast<std::int64_t>(c->step);
    if (c->run_state == CALIPER_NET_RUN_STEPPING) st.steps_left = static_cast<std::int64_t>(c->steps_left);
    for (const caliper_live::ControlParam& p : c->params) {
        cb::RunParam r;
        r.name = p.name;
        r.type = p.type;
        r.value = from_net(p.value);
        r.pending = from_net(p.pending);
        if (p.has_min) r.min = p.min;
        if (p.has_max) r.max = p.max;
        r.log = p.log;
        r.choices = p.choices;
        r.help = p.help;
        st.params.push_back(std::move(r));
    }
    for (const caliper_live::ControlAction& a : c->actions) {
        cb::RunAction r;
        r.name = a.name;
        r.help = a.help;
        r.running = a.running;
        r.queued = a.queued;
        st.actions.push_back(std::move(r));
    }
    return st;
}

cb::Error AppRunControl::send(const std::string& op, const std::string& name, const json& value, std::int64_t count) {
    return send_net(op, name, to_net(value), count);
}

cb::Error AppRunControl::send_net(const std::string& op, const std::string& name, const caliper_net::Json& value,
                                  std::int64_t count) {
    if (!link_ || !link_->attached())
        return cb::Error(cb::kRunNotAttached, "No script is attached to Caliper.");
    const caliper_live::ControlAccess access = link_->access();
    if (!access.allowed)
        return cb::Error(cb::kRunControlNotAllowed,
                         sentence(access.why.empty() ? "the connection takes no control" : access.why));
    const auto n = static_cast<uint32_t>(
        std::min<std::int64_t>(std::max<std::int64_t>(count, 1), std::numeric_limits<uint32_t>::max()));
    const std::string why = link_->send(op, name, value, n);
    if (why.empty()) return {};
    const caliper_live::Controls* c = link_->controls();
    if (op != CALIPER_NET_OP_SET && c && c->run_state == CALIPER_NET_RUN_NONE)
        return cb::Error(cb::kRunNoStep, "The script " + link_->script() +
                                             " has not made its first step, and it takes this request only at a step.");
    return cb::Error(cb::kRunRefused, sentence(why));
}

cb::Error AppRunControl::set_param(const std::string& name, const json& value) {
    return send(CALIPER_NET_OP_SET, name, value, 0);
}

cb::Error AppRunControl::run_action(const std::string& name) { return send(CALIPER_NET_OP_RUN, name, json(), 0); }

cb::Error AppRunControl::pause(cb::RunPauseOp op, std::int64_t count) {
    switch (op) {
        case cb::RunPauseOp::pause: return send(CALIPER_NET_OP_PAUSE, "", json(), 0);
        case cb::RunPauseOp::resume: return send(CALIPER_NET_OP_RESUME, "", json(), 0);
        case cb::RunPauseOp::step: return send(CALIPER_NET_OP_STEP, "", json(), count);
    }
    return send(CALIPER_NET_OP_PAUSE, "", json(), 0);
}

std::vector<cb::InterventionPoint> intervention_points_of(const json& interventions) {
    std::vector<cb::InterventionPoint> out;
    if (!interventions.is_object() || !interventions.contains("points") || !interventions["points"].is_array())
        return out;
    for (const json& p : interventions["points"]) {
        if (!p.is_object()) continue;
        cb::InterventionPoint q;
        q.point = p.value("point", std::string());
        q.kind = p.value("kind", std::string("module"));
        q.module = p.value("module", q.point);
        q.head = opt_int(p, "head");
        q.type = p.is_object() && p.contains("type") && p["type"].is_string() ? p["type"].get<std::string>() : "";
        if (p.contains("shape") && p["shape"].is_array()) {
            std::vector<std::int64_t> shape;
            for (const json& d : p["shape"])
                if (d.is_number_integer()) shape.push_back(d.get<std::int64_t>());
            q.shape = std::move(shape);
        }
        if (!q.point.empty()) out.push_back(std::move(q));
    }
    return out;
}

std::vector<cb::Intervention> interventions_of(const json& interventions) {
    std::vector<cb::Intervention> pending, rest;
    if (!interventions.is_object() || !interventions.contains("active") || !interventions["active"].is_array())
        return pending;
    for (const json& a : interventions["active"]) {
        if (!a.is_object()) continue;
        cb::Intervention i;
        i.point = a.value("point", std::string());
        i.op = a.value("op", std::string("ablate"));
        i.from_step = opt_int(a, "from_step");
        i.until_step = opt_int(a, "until_step");
        i.since_step = opt_int(a, "since_step");
        i.state = a.value("state", std::string("pending"));
        if (i.point.empty()) continue;
        (i.state == "pending" ? pending : rest).push_back(std::move(i));
    }
    pending.insert(pending.end(), std::make_move_iterator(rest.begin()), std::make_move_iterator(rest.end()));
    return pending;
}

bool AppRunControl::interventions_supported() {
    return link_ && link_->attached() && interventions_json(link_->controls()).is_object();
}

std::vector<cb::InterventionPoint> AppRunControl::intervention_points() {
    if (!link_ || !link_->attached()) return {};
    return intervention_points_of(interventions_json(link_->controls()));
}

std::vector<cb::Intervention> AppRunControl::interventions() {
    if (!link_ || !link_->attached()) return {};
    return interventions_of(interventions_json(link_->controls()));
}

cb::Error AppRunControl::intervene_set(const cb::Intervention& i) {
    if (link_ && link_->attached() && !interventions_supported())
        return cb::Error(cb::kInterveneNotSupported, "The script " + link_->script() + " takes no interventions.");
    json set{{"point", i.point}, {"op", i.op}};
    set["from_step"] = i.from_step ? json(*i.from_step) : json();
    set["until_step"] = i.until_step ? json(*i.until_step) : json();
    return send_net(kOpIntervene, "", to_net_json(json{{"set", std::move(set)}}), 0);
}

cb::Error AppRunControl::intervene_clear(const std::string& point) {
    if (link_ && link_->attached() && !interventions_supported())
        return cb::Error(cb::kInterveneNotSupported, "The script " + link_->script() + " takes no interventions.");
    const json body = point.empty() ? json{{"clear_all", true}} : json{{"clear", point}};
    return send_net(kOpIntervene, "", to_net_json(body), 0);
}

void AppRunControl::feed_interventions(const json& interventions, const json& step, bool baseline,
                                       cb::EventSink& sink) {
    // The interventions the script applies now, by point and op, with the
    // step it applied each at.
    std::map<std::pair<std::string, std::string>, json> now;
    for (const cb::Intervention& i : interventions_of(interventions))
        if (i.state == "active") now[{i.point, i.op}] = i.since_step ? json(*i.since_step) : step;
    if (!baseline) {
        for (const auto& [key, since] : now) {
            const auto it = applied_.find(key);
            if (it == applied_.end() || it->second != since)
                push_run(sink, json{{"change", "intervene"}, {"point", key.first}, {"op", key.second}, {"step", since}});
        }
    }
    // A stop also in a baseline: applied_ then holds what the same script applied before its
    // link dropped (feed()), else nothing.
    for (const auto& [key, since] : applied_)
        if (!now.count(key))
            push_run(sink, json{{"change", "intervene"}, {"point", key.first}, {"op", nullptr}, {"step", step}});
    applied_ = std::move(now);
}

void AppRunControl::feed(cb::EventSink& sink) {
    const bool attached = link_ && link_->attached();
    const std::string script = attached ? link_->script() : std::string();
    if (attached != attached_ || (attached && script != script_)) {
        push_run(sink, json{{"change", "attach"}, {"attached", attached}, {"script", attached ? script : script_}});
        if (attached_ && !attached) {   // what the script applied stays known for its return
            dropped_script_ = script_;
            dropped_ = std::move(applied_);
        }
        attached_ = attached;
        script_ = attached ? script : std::string();
        have_ = false;
        version_ = 0;
        run_state_ = CALIPER_NET_RUN_NONE;
        params_.clear();
        results_.clear();
        applied_.clear();
        if (attached) {
            if (script == dropped_script_) applied_ = std::move(dropped_);
            dropped_.clear();
            dropped_script_.clear();
        }
    }
    if (!attached) return;
    const caliper_live::Controls* c = link_->controls();
    if (!c) return;
    const bool baseline = !have_;
    if (baseline || c->version != version_) {
        const json step = c->has_step ? json(c->step) : json();
        if (c->run_state != run_state_)
            push_run(sink, json{{"change", "state"}, {"state", c->run_state}, {"step", step}});
        std::map<std::string, std::pair<json, json>> now;
        for (const caliper_live::ControlParam& p : c->params) {
            json value = from_net(p.value);
            json pending = from_net(p.pending);
            const auto it = params_.find(p.name);
            // A value the script applied: it changed, or the pending value it
            // held has gone. The baseline's values are registrations.
            if (!baseline && it != params_.end() &&
                (it->second.first != value || (!it->second.second.is_null() && pending.is_null())))
                push_run(sink, json{{"change", "param"}, {"name", p.name}, {"value", value}, {"step", step}});
            now[p.name] = {std::move(value), std::move(pending)};
        }
        params_ = std::move(now);
        feed_interventions(interventions_json(c), step, baseline, sink);
        run_state_ = c->run_state;
        version_ = c->version;
        have_ = true;
    }
    for (const caliper_live::ControlAction& a : c->actions) {
        const caliper_live::RunLink::ActionResult r = link_->action_result(a.name);
        std::uint64_t& seen = results_[a.name];
        if (r.at_ns == 0 || r.at_ns == seen) continue;
        if (!baseline)
            push_run(sink, json{{"change", "action"}, {"name", a.name}, {"status", r.error ? "error" : "ok"},
                                {"text", r.text}});
        seen = r.at_ns;
    }
}

struct AppSession::Impl {
    enum class State { idle, opening, opened, ready, failed, closed };
    enum class Capture { none, armed, ready, failed };
    struct Line {
        std::string name, origin, error, allowed_by;
        bool ok = true;
    };

    // SessionHost for the session, alive while the session holds it.
    struct HostBinding final : cb::SessionHost {
        Impl* im;
        explicit HostBinding(Impl* i) : im(i) {}
        cb::CommandOutput app_info() override { return im->app_info(); }
        cb::CommandOutput app_applets() override { return im->app_applets(); }
        // The session always says who asked (app_launch(id, origin) below);
        // a launch with no origin is refused rather than taken as the person's.
        cb::CommandOutput app_launch(const std::string&) override {
            return failed("command.not_available", "app.launch needs the origin of whoever asked.");
        }
        cb::CommandOutput app_launch(const std::string& id, cb::CommandOrigin origin) override {
            return im->app_launch(id, cb::command_origin_name(origin));
        }
        cb::CommandOutput render_panel(const cb::spec::Spec& spec, const cb::PanelData& data, int width,
                                       int height) override {
            return im->render_panel(spec, data, width, height);
        }
        cb::CommandOutput capture_frame() override { return im->capture_frame(); }
        cb::CommandOutput camera_command(const std::string& name, const json& args) override {
            return im->camera_command(name, args);
        }
    };

    Options opt;
    Hooks hooks;
    std::thread::id frame_thread;
    uint64_t frames = 0;

    mutable std::mutex m;   // everything up to `in_flight`
    std::condition_variable cv;
    bool started = false;
    State state = State::idle;
    std::unique_ptr<cb::Session> opened;    // made on the worker, not adopted yet
    std::unique_ptr<cb::Session> session;   // adopted by the frame thread
    std::string error;
    double open_ms = 0.0;
    bool failure_reported = false;
    int in_flight = 0;   // worker jobs that make or use the session
    std::atomic<bool> busy{false};
    std::shared_ptr<HostBinding> binding;
    AppRunControl run_ctl;   // the session's run control from its adoption until close()
    std::atomic<cb::ModelRunner*> runner{nullptr};   // the session's model runner from its adoption until close()
    std::atomic<cb::ProgramHost*> program_host{nullptr};   // the hosts of GPU programs and stages, likewise
    std::atomic<cb::StageHost*> stage_host{nullptr};

    std::mutex lines_m;
    std::vector<Line> lines;   // from the journal, any thread
    std::vector<AppSession::Applied> applied;   // under lines_m: what a mode or a rule let run, the newest 64
    std::atomic<uint64_t> journaled{0};        // every journaled command
    std::atomic<uint64_t> specs_opened{0};     // every applet.open, and an agent's spec.new or spec.open, that succeeded
    std::atomic<uint64_t> agent_specs_opened{0};   // of those, the ones an agent sent
    std::atomic<uint64_t> applets_saved{0};    // every applet.save that succeeded

    // Caliper API's gateway (command_gateway()), and the pending approvals as
    // the last frame without a worker read them, for its approval lookups.
    std::shared_ptr<GatewayCore> gateway;
    std::mutex pending_m;
    std::vector<cb::PendingApproval> pending;

    // capture.frame: touched on the frame thread only.
    Capture cap = Capture::none;
    Clock::time_point armed_at;
    uint32_t cap_w = 0, cap_h = 0;
    std::vector<uint8_t> cap_px;
    std::string cap_why;

    std::mutex renderer_m;
    std::shared_ptr<PanelRenderSeam> renderer;
    AppSession::CameraSeam camera_seam;   // under renderer_m

    std::thread script;
    std::atomic<bool> script_on{false};

    std::mutex err_m;

    void say(const std::string& line) {
        std::lock_guard<std::mutex> lock(err_m);
        std::ostream& o = hooks.err ? *hooks.err : std::cerr;
        o << "[session] " << line << std::endl;
    }

    void log(const std::string& event, const std::string& object, const std::string& detail) {
        if (hooks.log) hooks.log(event, object, detail);
    }

    void run(const std::string& label, std::function<void()> fn) {
        if (hooks.run_job) hooks.run_job(label, std::move(fn));
        else std::thread(std::move(fn)).detach();
    }

    // ---- opening ---------------------------------------------------------

    void ensure_open() {
        {
            std::lock_guard<std::mutex> lock(m);
            if (!started || state != State::idle) return;
            state = State::opening;
            ++in_flight;
        }
        run("workspace", [this] {
            const auto t0 = Clock::now();
            cb::SessionOptions so;
            so.journal.on_append = [this](const cb::JournalEvent& e) { note(e); };
            so.permissions_file = opt.permissions_file;
            so.protected_files = opt.protected_files;
            cb::Session::Open o = cb::Session::open(opt.workspace, std::move(so));
            std::lock_guard<std::mutex> lock(m);
            open_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (o.session) {
                opened = std::move(o.session);
                if (state == State::opening) state = State::opened;
            } else {
                error = o.error.message.empty() ? o.error.code : o.error.message;
                if (state == State::opening) state = State::failed;
            }
            --in_flight;
            cv.notify_all();
        });
    }

    // The frame thread takes the session the worker opened, or reports why
    // it could not.
    void adopt() {
        std::unique_ptr<cb::Session> s;
        std::string why;
        double ms = 0.0;
        {
            std::lock_guard<std::mutex> lock(m);
            if (state == State::opened) {
                s = std::move(opened);
                ms = open_ms;
            } else if (state == State::failed && !failure_reported) {
                failure_reported = true;
                why = error;
            }
        }
        if (!why.empty()) {
            say("workspace not opened: " + opt.workspace + ": " + why);
            log("workspace not opened", shown_path(opt.workspace), why);
            if (!opt.script.empty()) {
                write_results_failure("the workspace could not be opened: " + why);
                script_on.store(false);
            }
            return;
        }
        if (!s) return;
        s->registry().set_frame_thread(frame_thread);
        binding = std::make_shared<HostBinding>(this);
        s->set_host(binding);
        s->set_run_control(&run_ctl);
        s->set_model_runner(runner.load());
        s->set_program_host(program_host.load());
        s->set_stage_host(stage_host.load());
        cb::Session* raw = s.get();
        {
            std::lock_guard<std::mutex> lock(m);
            session = std::move(s);
            state = State::ready;
            cv.notify_all();
        }
        char took[32];
        std::snprintf(took, sizeof took, "%.0f ms", ms);
        say("workspace " + opt.workspace + " (" + opt.from + ") opened in " + took);
        log("workspace", shown_path(opt.workspace), opt.from);
        if (!opt.script.empty()) script = std::thread([this, raw] { run_script(*raw); });
    }

    // ---- the frame loop ----------------------------------------------------

    void note(const cb::JournalEvent& e) {
        const bool agent = e.origin == cb::CommandOrigin::agent || e.origin == cb::CommandOrigin::mcp;
        if (e.ok && (e.name == "applet.open" || (agent && (e.name == "spec.new" || e.name == "spec.open")))) {
            if (agent) agent_specs_opened.fetch_add(1, std::memory_order_acq_rel);
            specs_opened.fetch_add(1, std::memory_order_acq_rel);
        }
        if (e.ok && e.name == "applet.save") applets_saved.fetch_add(1, std::memory_order_acq_rel);
        journaled.fetch_add(1, std::memory_order_acq_rel);
        Line l;
        l.name = e.name;
        l.origin = cb::command_origin_name(e.origin);
        l.ok = e.ok;
        l.error = e.error;
        l.allowed_by = e.allowed_by;
        std::lock_guard<std::mutex> lock(lines_m);
        lines.push_back(std::move(l));
        if (e.ok && !e.allowed_by.empty()) {
            applied.push_back({e.seq, e.name, cb::command_origin_name(e.origin), e.allowed_by});
            if (applied.size() > 64) applied.erase(applied.begin());
        }
    }

    void flush_lines() {
        std::vector<Line> ls;
        {
            std::lock_guard<std::mutex> lock(lines_m);
            ls.swap(lines);
        }
        for (const Line& l : ls) {
            // The origin leads, so it stays in view when a long path widens
            // the OBJECT column; DETAIL holds the error code of a failure.
            // A command a mode or a rule let run without a card says what did.
            const std::string detail = !l.ok ? l.error : l.allowed_by.empty() ? std::string() : "allowed by " + l.allowed_by;
            log(l.origin + " command", l.name, detail);
            say("command " + l.name + " (" + l.origin + ")" + (detail.empty() ? std::string() : ": " + detail));
        }
    }

    bool on_worker(cb::Session& s, const cb::QueuedItem& item) {
        if (item.kind == cb::QueuedItem::Kind::invoke) return runs_on_worker(item.invocation);
        if (item.kind == cb::QueuedItem::Kind::deny) return false;
        for (const cb::PendingApproval& p : s.registry().pending_approvals())
            if (p.approval == item.approval) return runs_on_worker(p.invocation);
        return false;
    }

    void start_worker(cb::CommandQueue& q, const std::string& label) {
        {
            std::lock_guard<std::mutex> lock(m);
            ++in_flight;
        }
        busy.store(true, std::memory_order_release);
        run(label, [this, &q] {
            q.drain(1);
            busy.store(false, std::memory_order_release);
            std::lock_guard<std::mutex> lock(m);
            --in_flight;
            cv.notify_all();
        });
    }

    void pump(cb::Session& s) {
        cb::CommandQueue& q = s.queue();
        q.drain(0);   // withdraws the approvals whose wait ran out
        if (cap == Capture::armed) {
            const double waited = std::chrono::duration<double>(Clock::now() - armed_at).count();
            if (waited < opt.capture_timeout_s) return;
            cap = Capture::failed;
            cap_why = "No frame was read back within the capture's time limit.";
        }
        const auto deadline =
            Clock::now() + std::chrono::microseconds((int64_t)(opt.budget_ms * 1000.0));
        std::size_t ran = 0;
        while (ran < opt.max_per_frame) {
            const std::optional<cb::QueuedItem> next = q.peek();
            if (!next) break;
            if (on_worker(s, *next)) {
                start_worker(q, next->kind == cb::QueuedItem::Kind::invoke ? job_label(next->invocation)
                                                                           : std::string("approved command"));
                break;
            }
            if (next->kind == cb::QueuedItem::Kind::invoke && next->invocation.name == "capture.frame" &&
                cap == Capture::none && hooks.capture_unavailable.empty() && hooks.request_capture) {
                cap = Capture::armed;
                armed_at = Clock::now();
                hooks.request_capture();
                break;
            }
            q.drain(1);
            ++ran;
            if (Clock::now() >= deadline) break;
        }
    }

    void frame() {
        {
            std::lock_guard<std::mutex> lock(m);
            if (!started || state == State::closed) return;
        }
        ++frames;
        // A chosen workspace opens once the first frame has been drawn.
        if (frames == 2 && (opt.chosen || !opt.script.empty())) ensure_open();
        adopt();
        flush_lines();
        cb::Session* s = nullptr;
        {
            std::lock_guard<std::mutex> lock(m);
            if (state == State::ready) s = session.get();
        }
        // The run events of what the attached script changed; the ring is
        // safe while a worker holds the registry.
        if (s) run_ctl.feed(s->events());
        if (!s || busy.load(std::memory_order_acquire)) return;
        pump(*s);
        flush_lines();
        // The approvals Caliper API's lookups read, while the registry is free.
        if (gateway && !busy.load(std::memory_order_acquire)) {
            std::vector<cb::PendingApproval> p = s->registry().pending_approvals();
            std::lock_guard<std::mutex> lock(pending_m);
            pending = std::move(p);
        }
    }

    // The adopted session, or null; any thread.
    cb::Session* ready_session() {
        std::lock_guard<std::mutex> lock(m);
        return state == State::ready ? session.get() : nullptr;
    }

    // Why the workspace could not be opened, once that is known; any thread.
    std::string open_failure() {
        std::lock_guard<std::mutex> lock(m);
        return state == State::failed ? (error.empty() ? std::string("unknown") : error) : std::string();
    }

    // ---- SessionHost --------------------------------------------------------

    cb::CommandOutput app_info() {
        if (!hooks.info_json) return failed("command.not_available", "app.info is not available here.");
        json j = parse_object(hooks.info_json());
        if (j.is_null()) return failed("command.failed", "The app's information could not be read.");
        return cb::CommandOutput::ok(std::move(j));
    }

    cb::CommandOutput app_applets() {
        if (!hooks.applets_json) return failed("command.not_available", "app.applets is not available here.");
        json j = parse_object(hooks.applets_json());
        if (j.is_null()) return failed("command.failed", "The app's applets could not be listed.");
        return cb::CommandOutput::ok(std::move(j));
    }

    cb::CommandOutput app_launch(const std::string& id, const std::string& origin) {
        if (!hooks.launch) return failed("command.not_available", "app.launch is not available here.");
        const LaunchOutcome o = hooks.launch(id, origin);
        if (o.status == 200) {
            json j = parse_object(o.body);
            if (j.is_null()) return failed("app.launch_failed", "The launch gave no result.", "args.id");
            return cb::CommandOutput::ok(std::move(j));
        }
        return failed(o.code.empty() ? "app.launch_failed" : o.code, o.body, "args.id");
    }

    cb::CommandOutput render_panel(const cb::spec::Spec& spec, const cb::PanelData& data, int width, int height) {
        std::shared_ptr<PanelRenderSeam> r;
        {
            std::lock_guard<std::mutex> lock(renderer_m);
            r = renderer;
        }
        if (!r)
            return failed("command.not_available",
                          "panel.render needs the spec runtime, which this build of the app does not have yet.");
        return r->render_panel(spec, data, width, height);
    }

    cb::CommandOutput camera_command(const std::string& name, const json& args) {
        AppSession::CameraSeam seam;
        {
            std::lock_guard<std::mutex> lock(renderer_m);
            seam = camera_seam;
        }
        if (!seam) return failed("command.not_available", name + " needs the spec page of the app, which is not open.");
        return seam(name, args);
    }

    cb::CommandOutput capture_frame() {
        const Capture c = cap;
        cap = Capture::none;
        if (c == Capture::ready) {
            std::string png;
            const std::vector<uint8_t> px = std::move(cap_px);
            cap_px.clear();
            if (!hooks.encode_png || !hooks.encode_png(px.data(), cap_w, cap_h, &png))
                return failed("capture.unavailable", "The frame could not be encoded as a PNG.");
            return cb::CommandOutput::ok({{"png_base64", base64(png)}, {"width", cap_w}, {"height", cap_h}});
        }
        if (c == Capture::failed) return failed("capture.unavailable", cap_why);
        if (!hooks.capture_unavailable.empty()) return failed("capture.unavailable", hooks.capture_unavailable);
        return failed("capture.unavailable",
                      "capture.frame waits for the next frame, so it runs only through the command queue.");
    }

    // ---- the command script ---------------------------------------------------

    void write_results_failure(const std::string& why) {
        if (opt.results.empty()) return;
        std::ofstream out(opt.results, std::ios::binary | std::ios::trunc);
        out << json{{"done", true}, {"commands", 0}, {"failed", 0}, {"error", why}}.dump() << "\n";
    }

    void run_script(cb::Session& s) {
        std::ifstream in(opt.script, std::ios::binary);
        std::ofstream out;
        if (!opt.results.empty()) out.open(opt.results, std::ios::binary | std::ios::trunc);
        auto write = [&](const json& j) {
            if (!out) return;
            out << j.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
            out.flush();
        };
        int count = 0, failures = 0, line_no = 0;
        if (!in) {
            say("cannot read the command script " + opt.script);
            write({{"done", true}, {"commands", 0}, {"failed", 0}, {"error", "the command script cannot be read"}});
            script_on.store(false);
            return;
        }
        std::string text;
        while (std::getline(in, text)) {
            ++line_no;
            while (!text.empty() && (text.back() == '\r' || text.back() == ' ' || text.back() == '\t'))
                text.pop_back();
            size_t first = text.find_first_not_of(" \t");
            if (first == std::string::npos || text[first] == '#') continue;
            json j = json::parse(text, nullptr, false);
            std::string name;
            std::string approved;   // the approval the script gave, for "approve": true
            cb::CommandResult r;
            if (j.is_discarded() || !j.is_object()) {
                r = cb::CommandResult::failure(
                    "", cb::Error("command.invalid_invocation", "The line is not a JSON object.", ""));
            } else {
                if (!j.contains("origin")) j["origin"] = "ui";
                if (j.contains("name") && j["name"].is_string()) name = j["name"].get<std::string>();
                // "approve": true approves the line's command, as the person
                // would, when it asks; without it a command that asks answers
                // command.needs_approval and the script goes on, so a script
                // never waits for a decision that nobody can make.
                bool approve = false;
                if (const auto a = j.find("approve"); a != j.end()) {
                    approve = a->is_boolean() && a->get<bool>();
                    j.erase(a);
                }
                cb::CommandInvocationParse p = cb::command_invocation_from_json(j);
                if (p.error) {
                    r = cb::CommandResult::failure("", p.error);
                } else {
                    cb::SubmitOptions so;
                    so.wait_for_approval = false;
                    r = s.queue().submit(std::move(p.invocation), so).get();
                    if (approve && !r.ok && r.error.code == "command.needs_approval" && !r.approval.empty()) {
                        approved = r.approval;
                        r = s.queue().approve(approved).get();
                    }
                }
            }
            ++count;
            if (!r.ok) {
                ++failures;
                say("script line " + std::to_string(line_no) + ": " + name + ": " + r.error.code + ": " +
                    r.error.message);
            }
            json line = cb::command_result_to_json(r);
            line["line"] = line_no;
            line["name"] = name;
            if (!approved.empty()) line["approved"] = approved;
            write(line);
            if (r.error.code == "command.shutdown") break;
        }
        write({{"done", true}, {"commands", count}, {"failed", failures}});
        say("script done: " + std::to_string(count) + " commands, " + std::to_string(failures) + " failed");
        script_on.store(false);
    }

    // ---- shutdown ----------------------------------------------------------

    void close() {
        // Commands the API still holds answer command.shutdown from now on.
        if (gateway) {
            std::lock_guard<std::mutex> lock(gateway->m);
            gateway->im = nullptr;
        }
        cb::Session* s = nullptr;
        {
            std::lock_guard<std::mutex> lock(m);
            if (!started || state == State::closed) return;
            state = State::closed;
            s = session.get();
            cv.notify_all();
        }
        // Waiting calls and approvals get command.shutdown; a command running
        // on the worker finishes first, since it holds the registry.
        if (s) s->queue().shutdown();
        if (script.joinable()) script.join();
        script_on.store(false);
        {
            std::unique_lock<std::mutex> lock(m);
            if (in_flight > 0 && !cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return in_flight == 0; })) {
                lock.unlock();
                say("waiting for the workspace's worker to finish");
                lock.lock();
            }
            cv.wait(lock, [&] { return in_flight == 0; });
            if (!session && opened) session = std::move(opened);
        }
        if (session) {
            session->set_host(nullptr);
            session->set_run_control(nullptr);
            session->set_model_runner(nullptr);
            session->set_program_host(nullptr);
            session->set_stage_host(nullptr);
            session->close();   // the queue and the journal
            session.reset();    // the loader store and workspace.duckdb
            say("workspace " + opt.workspace + " closed");
        }
        binding.reset();
    }
};

namespace {

std::string shutdown_envelope(const std::string& id) {
    return caliper_api::command_error_json(id, "command.shutdown", "The app's builder session has closed.", "");
}

// A command from Caliper API: handed to the session's queue as soon as the
// session is there, by whichever thread asks first whether it is ready.
class AppCommand final : public caliper_api::PendingCommand {
public:
    AppCommand(std::shared_ptr<GatewayCore> core, cb::CommandInvocation inv, cb::SubmitOptions options)
        : core_(std::move(core)), inv_(std::move(inv)), options_(options) {}

    bool ready() override {
        std::lock_guard<std::mutex> lock(m_);
        if (!inner_) {
            std::lock_guard<std::mutex> c(core_->m);
            if (!core_->im) {
                inner_ = caliper_api::ready_command(shutdown_envelope(inv_.id));
            } else if (cb::Session* s = core_->im->ready_session()) {
                inner_ = caliper_api::session_command(*s, std::move(inv_), options_);   // events.wait: a long poll
            } else {
                const std::string why = core_->im->open_failure();
                if (why.empty()) return false;   // still opening
                inner_ = caliper_api::ready_command(caliper_api::command_error_json(
                    inv_.id, "command.not_available", "The workspace could not be opened: " + why, ""));
            }
        }
        return inner_->ready();
    }

    std::string envelope() override {
        std::lock_guard<std::mutex> lock(m_);
        return inner_ ? inner_->envelope() : shutdown_envelope(inv_.id);
    }

private:
    std::mutex m_;
    std::shared_ptr<GatewayCore> core_;
    cb::CommandInvocation inv_;
    cb::SubmitOptions options_;
    std::unique_ptr<caliper_api::PendingCommand> inner_;
};

class AppGateway final : public caliper_api::CommandGateway {
public:
    explicit AppGateway(std::shared_ptr<GatewayCore> core)
        : core_(std::move(core)),
          table_(caliper_builder::command_table_to_json().dump(-1, ' ', false, json::error_handler_t::replace)) {}

    std::string table_json() const override { return table_; }

    // On the frame thread.
    std::unique_ptr<caliper_api::PendingCommand> submit(caliper_api::CommandRequest request) override {
        caliper_api::RequestInvocation r = caliper_api::request_invocation(std::move(request));
        if (!r.refused.empty()) return caliper_api::ready_command(std::move(r.refused));
        {
            std::lock_guard<std::mutex> lock(core_->m);
            if (!core_->im) return caliper_api::ready_command(shutdown_envelope(r.invocation.id));
            core_->im->ensure_open();
        }
        auto cmd = std::make_unique<AppCommand>(core_, std::move(r.invocation), r.options);
        cmd->ready();   // to the queue now when the session is there
        return cmd;
    }

    // On the frame thread: the queue's outcomes and the pending approvals of
    // the last frame without a worker, never the registry itself.
    std::string approval_json(const std::string& approval_id) override {
        std::optional<cb::CommandResult> outcome;
        std::vector<cb::PendingApproval> pending;
        {
            std::lock_guard<std::mutex> lock(core_->m);
            if (core_->im) {
                if (cb::Session* s = core_->im->ready_session()) outcome = s->queue().approval_outcome(approval_id);
                std::lock_guard<std::mutex> p(core_->im->pending_m);
                pending = core_->im->pending;
            }
        }
        return caliper_api::approval_state_json(approval_id, outcome, pending);
    }

private:
    std::shared_ptr<GatewayCore> core_;
    const std::string table_;
};

}  // namespace

AppSession::AppSession() : impl_(std::make_unique<Impl>()) {}

AppSession::~AppSession() { close(); }

void AppSession::start(Options options, Hooks hooks) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.m);
    if (im.started) return;
    im.opt = std::move(options);
    im.hooks = std::move(hooks);
    if (im.opt.from.empty()) im.opt.from = im.opt.chosen ? "--workspace" : "default";
    if (im.opt.budget_ms <= 0.0) im.opt.budget_ms = 4.0;
    if (im.opt.max_per_frame == 0) im.opt.max_per_frame = 1;
    im.frame_thread = std::this_thread::get_id();
    // A command script counts as running from now until it is done, or until
    // the workspace fails to open, so `caliper shot` keeps drawing for it.
    im.script_on.store(!im.opt.script.empty());
    im.started = true;
}

void AppSession::frame() { impl_->frame(); }

void AppSession::capture_ready(uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
    Impl& im = *impl_;
    if (im.cap != Impl::Capture::armed) return;
    im.cap = Impl::Capture::ready;
    im.cap_w = w;
    im.cap_h = h;
    im.cap_px = std::move(rgba);
}

void AppSession::capture_failed(const std::string& why) {
    Impl& im = *impl_;
    if (im.cap != Impl::Capture::armed) return;
    im.cap = Impl::Capture::failed;
    im.cap_why = why;
}

void AppSession::close() { impl_->close(); }

void AppSession::ensure_open() { impl_->ensure_open(); }

caliper_builder::Session* AppSession::wait_open(std::chrono::milliseconds limit) {
    Impl& im = *impl_;
    im.ensure_open();
    std::unique_lock<std::mutex> lock(im.m);
    if (!im.started) return nullptr;
    if (std::this_thread::get_id() != im.frame_thread)
        im.cv.wait_for(lock, limit, [&] {
            return im.state == Impl::State::ready || im.state == Impl::State::failed ||
                   im.state == Impl::State::closed;
        });
    return im.state == Impl::State::ready ? im.session.get() : nullptr;
}

caliper_builder::Session* AppSession::session() {
    std::lock_guard<std::mutex> lock(impl_->m);
    return impl_->state == Impl::State::ready ? impl_->session.get() : nullptr;
}

std::string AppSession::open_error() const {
    std::lock_guard<std::mutex> lock(impl_->m);
    return impl_->error;
}

const std::string& AppSession::workspace() const { return impl_->opt.workspace; }

bool AppSession::worker_busy() const { return impl_->busy.load(std::memory_order_acquire); }

bool AppSession::script_running() const { return impl_->script_on.load(); }

void AppSession::set_panel_renderer(std::shared_ptr<PanelRenderSeam> renderer) {
    std::lock_guard<std::mutex> lock(impl_->renderer_m);
    impl_->renderer = std::move(renderer);
}

void AppSession::set_camera_seam(CameraSeam seam) {
    std::lock_guard<std::mutex> lock(impl_->renderer_m);
    impl_->camera_seam = std::move(seam);
}

void AppSession::set_run_link(caliper_live::RunLink* link) { impl_->run_ctl.set_link(link); }

caliper_builder::RunControl* AppSession::run_control() { return &impl_->run_ctl; }

void AppSession::set_model_runner(caliper_builder::ModelRunner* runner) {
    impl_->runner.store(runner);
    std::lock_guard<std::mutex> lock(impl_->m);
    if (impl_->session) impl_->session->set_model_runner(runner);
}

void AppSession::set_program_host(caliper_builder::ProgramHost* host) {
    impl_->program_host.store(host);
    std::lock_guard<std::mutex> lock(impl_->m);
    if (impl_->session) impl_->session->set_program_host(host);
}

void AppSession::set_stage_host(caliper_builder::StageHost* host) {
    impl_->stage_host.store(host);
    std::lock_guard<std::mutex> lock(impl_->m);
    if (impl_->session) impl_->session->set_stage_host(host);
}

uint64_t AppSession::journaled() const { return impl_->journaled.load(std::memory_order_acquire); }

uint64_t AppSession::specs_opened() const { return impl_->specs_opened.load(std::memory_order_acquire); }
uint64_t AppSession::agent_specs_opened() const {
    return impl_->agent_specs_opened.load(std::memory_order_acquire);
}

uint64_t AppSession::applets_saved() const { return impl_->applets_saved.load(std::memory_order_acquire); }

std::vector<AppSession::Applied> AppSession::take_applied() {
    std::lock_guard<std::mutex> lock(impl_->lines_m);
    std::vector<Applied> out;
    out.swap(impl_->applied);
    return out;
}

std::shared_ptr<caliper_api::CommandGateway> AppSession::command_gateway() {
    Impl& im = *impl_;
    {
        std::lock_guard<std::mutex> lock(im.m);
        if (!im.started || im.state == Impl::State::closed) return nullptr;
    }
    if (!im.gateway) {
        im.gateway = std::make_shared<GatewayCore>();
        im.gateway->im = &im;
    }
    return std::make_shared<AppGateway>(im.gateway);
}

#else  // !CALIPER_WITH_BUILDER

// Without the builder there is no session: a chosen workspace and a command
// script are reported once, and the rest does nothing.
struct AppSession::Impl {
    Options opt;
};

AppSession::AppSession() : impl_(std::make_unique<Impl>()) {}
AppSession::~AppSession() = default;

void AppSession::start(Options options, Hooks hooks) {
    impl_->opt = std::move(options);
    if (!impl_->opt.chosen && impl_->opt.script.empty()) return;
    const std::string why = "the builder is not in this build";
    std::ostream& o = hooks.err ? *hooks.err : std::cerr;
    o << "[session] workspace not opened: " << impl_->opt.workspace << ": " << why << std::endl;
    if (hooks.log) hooks.log("workspace not opened", shown_path(impl_->opt.workspace), why);
}

void AppSession::frame() {}
void AppSession::capture_ready(uint32_t, uint32_t, std::vector<uint8_t>) {}
void AppSession::capture_failed(const std::string&) {}
void AppSession::close() {}
void AppSession::ensure_open() {}
caliper_builder::Session* AppSession::wait_open(std::chrono::milliseconds) { return nullptr; }
caliper_builder::Session* AppSession::session() { return nullptr; }
std::string AppSession::open_error() const { return "the builder is not in this build"; }
const std::string& AppSession::workspace() const { return impl_->opt.workspace; }
bool AppSession::worker_busy() const { return false; }
bool AppSession::script_running() const { return false; }
void AppSession::set_panel_renderer(std::shared_ptr<PanelRenderSeam>) {}
void AppSession::set_run_link(caliper_live::RunLink*) {}
caliper_builder::RunControl* AppSession::run_control() { return nullptr; }
void AppSession::set_model_runner(caliper_builder::ModelRunner*) {}
void AppSession::set_program_host(caliper_builder::ProgramHost*) {}
void AppSession::set_stage_host(caliper_builder::StageHost*) {}
uint64_t AppSession::journaled() const { return 0; }
uint64_t AppSession::specs_opened() const { return 0; }
uint64_t AppSession::agent_specs_opened() const { return 0; }
uint64_t AppSession::applets_saved() const { return 0; }
std::vector<AppSession::Applied> AppSession::take_applied() { return {}; }
std::shared_ptr<caliper_api::CommandGateway> AppSession::command_gateway() { return nullptr; }

#endif

void gate_command_script(AppSession::Options* o, bool scripts_allowed, std::string* refused) {
    if (scripts_allowed || o->script.empty()) return;
    o->script.clear();
    o->results.clear();
    if (refused) *refused = "CALIPER_COMMAND_SCRIPT runs only in the hidden app (--hidden) and in caliper shot; ignored";
}

}  // namespace caliper_host
