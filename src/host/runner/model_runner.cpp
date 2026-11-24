// The app's model runner (model_runner.h, design/hosted-runs.md).
#include "model_runner.h"

#include "run_connection.h"

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/catalog/tensor_sample.h>
#include <caliper_builder/loaders/python_loader.h>
#include <caliper_builder/loaders/tensor.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace caliper_runner {

namespace cb = caliper_builder;
namespace fs = std::filesystem;
using nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kCheckpointAction = "caliper.checkpoint";
constexpr const char* kStopAction = "caliper.stop";
constexpr std::uint64_t kLogTailBytes = 256 * 1024;
constexpr std::size_t kLogLineBytes = 2000;
// How long the runner reads a connection after its process ended, for the
// last values and replies that the process sent before its end.
constexpr int kDrainMs = 250;

cb::Error fail(std::string code, std::string message, std::string where = {}) {
    return cb::Error(std::move(code), std::move(message), std::move(where));
}

bool finished(const std::string& state) { return state == "exited" || state == "failed"; }

// The base of a default id: the script's file name in a-z, 0-9 and '-'.
std::string id_base(const std::string& script) {
    std::string stem = fs::u8path(script).stem().u8string();
    std::string out;
    for (char c : stem) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        if (ok)
            out += c;
        else if (!out.empty() && out.back() != '-')
            out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    if (out.size() > 32) out.resize(32);
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "run" : out;
}

// Valid UTF-8 with each bad byte as '?', without a trailing CR.
std::string clean_line(std::string s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
    if (s.size() > kLogLineBytes) s.resize(kLogLineBytes);
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        for (std::size_t k = 1; ok && k < n; ++k) ok = (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
        if (ok && (c >= 0x20 || c == '\t' || n > 1)) {
            out.append(s, i, n);
            i += n;
        } else {
            out += '?';
            ++i;
        }
    }
    return out;
}

// The last lines of the file at `path`, from at most its last kLogTailBytes,
// oldest first; none when it does not exist.
std::vector<std::string> tail_lines(const std::string& path) {
    std::vector<std::string> all;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) return all;
    f.seekg(0, std::ios::end);
    const auto size = static_cast<std::uint64_t>(std::max<std::streamoff>(0, f.tellg()));
    const std::uint64_t from = size > kLogTailBytes ? size - kLogTailBytes : 0;
    f.seekg(static_cast<std::streamoff>(from));
    std::string text(static_cast<std::size_t>(size - from), '\0');
    f.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(f.gcount()));
    std::size_t pos = 0;
    if (from > 0) {   // the first line is cut
        const std::size_t nl = text.find('\n');
        pos = nl == std::string::npos ? text.size() : nl + 1;
    }
    while (pos < text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        all.push_back(clean_line(text.substr(pos, nl - pos)));
        pos = nl + 1;
    }
    return all;
}

// Keeps a run's output.log near `limit` bytes: once it passed the limit, its
// content moves to output.log.1, which replaces the one before, and
// output.log starts again with a line that says so. The child appends to the
// file, so its next write lands at the new end. What it writes between the
// copy and the cut is lost. Both files have the mode 0600 on POSIX.
void rotate_output(const std::string& path, std::uint64_t limit) {
    std::error_code ec;
    const fs::path p = fs::u8path(path);
    const std::uint64_t size = fs::file_size(p, ec);
    if (ec || size <= limit) return;
    const fs::path older = fs::u8path(path + ".1");
    fs::remove(older, ec);
    fs::copy_file(p, older, ec);
    fs::permissions(older, fs::perms::owner_read | fs::perms::owner_write, ec);
    fs::resize_file(p, 0, ec);
    std::ofstream f(p, std::ios::binary | std::ios::app);
    f << "caliper: output.log reached " << limit << " bytes, so the runner moved its lines to output.log.1, which "
      << "keeps only the latest such part, and output.log starts again here\n";
}

}  // namespace

// ---- the run table -------------------------------------------------------

struct RunnerState;

// The run control of a managed run that the app does not attach to: it reads
// the runner connection's CONTROLS and sends through it.
class ConnectionControl final : public cb::RunControl {
public:
    ConnectionControl(RunnerState* state, std::string id) : state_(state), id_(std::move(id)) {}
    cb::RunState state() override;
    cb::CommandOutput interpret(const std::string& method, const json& params, std::uint32_t request) override;
    cb::Error set_param(const std::string& name, const json& value) override { return send("set", name, value, 1); }
    cb::Error run_action(const std::string& name) override { return send("run", name, json(), 1); }
    cb::Error pause(cb::RunPauseOp op, std::int64_t count) override {
        const char* name = op == cb::RunPauseOp::pause ? "pause" : op == cb::RunPauseOp::resume ? "resume" : "step";
        return send(name, "", json(), static_cast<std::uint32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(count, 0xFFFFFFFF))));
    }
    // The interventions (design/gpu-programs.md, Intervention Wire), from the
    // CONTROLS key "interventions" of the runner's connection and sent as
    // CONTROL op "intervene" through it.
    bool interventions_supported() override { return interventions_json().is_object(); }
    std::vector<cb::InterventionPoint> intervention_points() override;
    std::vector<cb::Intervention> interventions() override;
    cb::Error intervene_set(const cb::Intervention& i) override {
        if (!interventions_supported())
            return fail(cb::kInterveneNotSupported, "The script of " + id_ + " takes no interventions.");
        json set = cb::intervention_set_spec(i);
        return send("intervene", "", json{{"set", std::move(set)}}, 1);
    }
    cb::Error intervene_clear(const std::string& point) override {
        if (!interventions_supported())
            return fail(cb::kInterveneNotSupported, "The script of " + id_ + " takes no interventions.");
        return send("intervene", "", point.empty() ? json{{"clear_all", true}} : json{{"clear", point}}, 1);
    }

private:
    json interventions_json();
    cb::Error send(const std::string& op, const std::string& name, const json& value, std::uint32_t count);
    RunnerState* state_;
    std::string id_;
};

struct Run {
    cb::RunEntry entry;
    std::string script_path;   // absolute
    std::string kind = "script";
    std::string model, dictionary, prompt;
    std::string dir, link_file, output;
    std::unique_ptr<ChildProcess> proc;
    Clock::time_point started = Clock::now();
    std::string link;                        // from the link file
    std::unique_ptr<RunConnection> conn;     // kept after the run ends, for its series
    std::unique_ptr<ConnectionControl> control;
    // A fork that waits for its parent's checkpoint.
    std::string waits_for;                   // the parent's id; "" when not waiting
    std::uint32_t checkpoint_request = 0;
    // A fork's interventions as a JSON array, which its script applies before
    // its first step (CALIPER_RUN_INTERVENTIONS); "" when it has none.
    std::string interventions;
    // A fork's series start at its parent's checkpoint step: the run sources
    // put this many NaN values first, so that the fork's curve continues its
    // parent's on one step axis.
    std::uint64_t offset = 0;
    // A stop: when to ask the process to end, and when that was asked.
    std::optional<Clock::time_point> end_at;
    std::optional<Clock::time_point> terminated_at;
    // The series' totals seen, and their generations.
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> series_gen;
};

struct RunnerState {
    ModelRunner::Options options;
    std::mutex m;
    std::condition_variable wake;     // the runner's thread
    std::condition_variable changed;  // wait_for()
    std::map<std::string, std::unique_ptr<Run>> runs;
    std::vector<std::string> order;
    bool stopping = false;
    bool shut = false;
    std::thread thread;
    std::uint64_t generation = 0;     // of the run sources
    std::uint64_t revision = 1;       // of the run sources' list
    std::uint64_t changes = 0;        // counts the passes of the runner's thread that changed something
    // The app (frame thread).
    bool attached = false;
    std::string attached_link;
    std::string attach_next;          // the link to attach at the next frame
    cb::RunControl* app_control = nullptr;
    std::mutex sink_m;
    cb::EventSink* sink = nullptr;

    Run* find(const std::string& id) {
        const auto it = runs.find(id);
        return it == runs.end() ? nullptr : it->second.get();
    }

    cb::RunEntry entry_of(const Run& r) const {
        cb::RunEntry e = r.entry;
        e.attached = attached && !r.link.empty() && r.link == attached_link;
        return e;
    }

    // The runner event of a run's state (design/hosted-runs.md, Runner Events).
    static cb::Event event_of(const cb::RunEntry& e) {
        cb::Event ev;
        ev.kind = cb::EventKind::run;
        ev.origin = "script";
        ev.fields = {{"change", "runner"}, {"run", e.id}, {"state", e.state}};
        ev.fields["parent"] = e.parent ? json(*e.parent) : json();
        ev.fields["exit_code"] = e.exit_code ? json(*e.exit_code) : json();
        ev.fields["error"] = e.error ? json(*e.error) : json();
        if (e.step) ev.fields["step"] = *e.step;
        return ev;
    }

    void push_events(std::vector<cb::Event> events) {
        std::lock_guard<std::mutex> lock(sink_m);
        if (!sink) return;
        for (auto& e : events) sink->push(std::move(e));
    }
};

cb::RunState ConnectionControl::state() {
    std::lock_guard<std::mutex> lock(state_->m);
    cb::RunState st;
    Run* r = state_->find(id_);
    if (!r || !r->conn || r->conn->state() != RunConnection::State::open) return st;
    st.attached = true;
    st.script = r->entry.script;
    const json& c = r->conn->controls();
    st.control = r->conn->control_allowed() && c.is_object();
    if (!r->conn->control_allowed())
        st.why_not = "The script of " + id_ + " started its stream without control.";
    else if (!c.is_object())
        st.why_not = "The script has sent no controls yet.";
    if (!c.is_object()) return st;
    const json& run = c.value("run", json::object());
    st.state = run.value("state", std::string("none"));
    if (run.contains("step") && run["step"].is_number_integer()) st.step = run["step"].get<std::int64_t>();
    if (run.contains("steps_left") && run["steps_left"].is_number_integer())
        st.steps_left = run["steps_left"].get<std::int64_t>();
    for (const json& p : c.value("params", json::array())) {
        cb::RunParam q;
        q.name = p.value("name", std::string());
        q.type = p.value("type", std::string());
        q.value = p.value("value", json());
        q.pending = p.value("pending", json());
        if (p.contains("min") && p["min"].is_number()) q.min = p["min"].get<double>();
        if (p.contains("max") && p["max"].is_number()) q.max = p["max"].get<double>();
        q.log = p.value("log", false);
        if (p.contains("choices") && p["choices"].is_array())
            for (const json& ch : p["choices"])
                if (ch.is_string()) q.choices.push_back(ch.get<std::string>());
        q.help = p.value("help", std::string());
        st.params.push_back(std::move(q));
    }
    for (const json& a : c.value("actions", json::array())) {
        cb::RunAction q;
        q.name = a.value("name", std::string());
        q.help = a.value("help", std::string());
        q.running = a.value("running", false);
        q.queued = a.value("queued", false);
        st.actions.push_back(std::move(q));
    }
    return st;
}

json ConnectionControl::interventions_json() {
    std::lock_guard<std::mutex> lock(state_->m);
    Run* r = state_->find(id_);
    if (!r || !r->conn || r->conn->state() != RunConnection::State::open) return json();
    const json& c = r->conn->controls();
    return c.is_object() ? c.value("interventions", json()) : json();
}

namespace {

std::optional<std::int64_t> opt_int(const json& o, const char* key) {
    const auto it = o.find(key);
    if (it == o.end() || !it->is_number_integer()) return std::nullopt;
    return it->get<std::int64_t>();
}

}  // namespace

std::vector<cb::InterventionPoint> ConnectionControl::intervention_points() {
    const json iv = interventions_json();
    std::vector<cb::InterventionPoint> out;
    if (!iv.is_object() || !iv.contains("points") || !iv["points"].is_array()) return out;
    for (const json& p : iv["points"]) {
        if (!p.is_object() || !p.contains("point") || !p["point"].is_string()) continue;
        cb::InterventionPoint q;
        q.point = p["point"].get<std::string>();
        q.kind = p.value("kind", std::string("module"));
        q.module = p.value("module", q.point);
        q.head = opt_int(p, "head");
        if (p.contains("probe") && p["probe"].is_string()) q.probe = p["probe"].get<std::string>();
        q.layer = opt_int(p, "layer");
        if (p.contains("pattern")) {
            if (p["pattern"].is_null()) q.pattern.emplace(std::nullopt);
            else if (p["pattern"].is_string()) q.pattern.emplace(p["pattern"].get<std::string>());
        }
        if (p.contains("type") && p["type"].is_string()) q.type = p["type"].get<std::string>();
        if (p.contains("shape") && p["shape"].is_array()) {
            std::vector<std::int64_t> shape;
            for (const json& d : p["shape"])
                if (d.is_number_integer()) shape.push_back(d.get<std::int64_t>());
            q.shape = std::move(shape);
        }
        out.push_back(std::move(q));
    }
    return out;
}

std::vector<cb::Intervention> ConnectionControl::interventions() {
    const json iv = interventions_json();
    std::vector<cb::Intervention> pending, rest;
    if (!iv.is_object() || !iv.contains("active") || !iv["active"].is_array()) return pending;
    for (const json& a : iv["active"]) {
        if (!a.is_object() || !a.contains("point") || !a["point"].is_string()) continue;
        cb::Intervention i;
        i.specification = a;
        i.point = a["point"].get<std::string>();
        i.op = a.value("op", std::string("ablate"));
        i.from_step = opt_int(a, "from_step");
        i.until_step = opt_int(a, "until_step");
        i.since_step = opt_int(a, "since_step");
        i.state = a.value("state", std::string("pending"));
        (i.state == "pending" ? pending : rest).push_back(std::move(i));
    }
    pending.insert(pending.end(), rest.begin(), rest.end());
    return pending;
}

cb::CommandOutput ConnectionControl::interpret(const std::string& method, const json& params, std::uint32_t request) {
    std::lock_guard<std::mutex> lock(state_->m);
    Run* run = state_->find(id_);
    if (!run || !run->conn)
        return cb::CommandOutput::fail(fail(cb::kRunNotAttached, "The run has no connection."));
    const auto result = run->conn->interpret(method, params, request);
    if (result.contains("error"))
        return cb::CommandOutput::fail(fail(result["error"]["code"], result["error"]["message"]));
    state_->wake.notify_all();
    return cb::CommandOutput::ok(result);
}

cb::Error ConnectionControl::send(const std::string& op, const std::string& name, const json& value,
                                  std::uint32_t count) {
    std::lock_guard<std::mutex> lock(state_->m);
    Run* r = state_->find(id_);
    if (!r || !r->conn) return fail(cb::kRunNotAttached, "The run " + id_ + " has no connection.");
    std::string err;
    if (!r->conn->send_control(op, name, value, count, &err)) return fail(cb::kRunRefused, err);
    state_->wake.notify_all();
    return {};
}

// ---- the run sources -------------------------------------------------------

namespace {

class RunSources final : public cb::CatalogProvider, public cb::CatalogDataSource {
public:
    explicit RunSources(std::weak_ptr<RunnerState> state) : state_(std::move(state)) {}

    cb::CatalogListResult list(const cb::CatalogFilter& filter) override {
        cb::CatalogListResult out;
        if (filter.kind && *filter.kind != cb::CatalogKind::run) return out;
        for (cb::CatalogEntry& e : entries())
            if (e.name.rfind(filter.prefix, 0) == 0) {
                if (out.entries.size() >= std::max<std::size_t>(1, filter.limit)) {
                    out.truncated = true;
                    break;
                }
                out.entries.push_back(std::move(e));
            }
        return out;
    }

    cb::CatalogDescribeResult describe(const std::string& id) override {
        cb::CatalogDescribeResult out;
        for (cb::CatalogEntry& e : entries())
            if (e.id == id) out.entry = std::move(e);
        return out;
    }

    cb::CatalogSampleResult sample(const std::string& id, const cb::CatalogSampleRequest& request) override {
        cb::CatalogSampleResult out;
        std::vector<float> values;
        if (!copy(id, &values, nullptr)) {
            out.error = fail("catalog.not_found", "The run source " + id + " is not there.", id);
            return out;
        }
        const std::vector<std::int64_t> shape{static_cast<std::int64_t>(values.size())};
        cb::TensorSampleResult s = cb::sample_tensor(cb::CatalogDtype::f32, shape, values.data(),
                                                     values.size() * sizeof(float), request.start, request.count);
        if (s.error) {
            out.error = s.error;
            return out;
        }
        out.sample.id = id;
        out.sample.kind = cb::CatalogKind::run;
        out.sample.dtype = cb::CatalogDtype::f32;
        out.sample.shape = shape;
        out.sample.stats = s.stats;
        out.sample.slice = s.slice;
        return out;
    }

    std::uint64_t revision() const override {
        auto st = state_.lock();
        if (!st) return 0;
        std::lock_guard<std::mutex> lock(st->m);
        return st->revision;
    }

    std::uint64_t generation(const std::string& id) override {
        auto st = state_.lock();
        std::string run, view;
        if (!st || !split(id, &run, &view)) return 0;
        std::lock_guard<std::mutex> lock(st->m);
        Run* r = st->find(run);
        if (!r) return 0;
        const auto it = r->series_gen.find(view);
        return it == r->series_gen.end() ? 0 : it->second.second;
    }

    cb::ServedData read(const std::string& id, const cb::ServedRequest& request) override {
        cb::ServedData out;
        std::vector<float> values;
        std::uint64_t gen = 0;
        if (!copy(id, &values, &gen)) {
            out.error = fail("resolve.unavailable", "The run source " + id + " has no data: the run or its series is not there yet.", id);
            return out;
        }
        if (values.size() * sizeof(float) > request.max_bytes) {
            out.error = fail("resolve.too_large", "The run source " + id + " is past the byte limit of the request.", id);
            return out;
        }
        cb::TensorResult t = cb::TensorData::allocate(cb::DType::f32, {static_cast<std::int64_t>(values.size())});
        if (t.error) {
            out.error = t.error;
            return out;
        }
        if (!values.empty()) std::memcpy(t.tensor.mutable_data(), values.data(), values.size() * sizeof(float));
        out.tensor = std::move(t.tensor);
        out.generation = gen;
        return out;
    }

private:
    // "run:<run>/<view>" into its run and view.
    static bool split(const std::string& id, std::string* run, std::string* view) {
        if (id.rfind("run:", 0) != 0) return false;
        const std::size_t slash = id.find('/', 4);
        if (slash == std::string::npos) return false;
        *run = id.substr(4, slash - 4);
        *view = id.substr(slash + 1);
        return !run->empty() && !view->empty();
    }

    bool copy(const std::string& id, std::vector<float>* values, std::uint64_t* gen) {
        auto st = state_.lock();
        std::string run, view;
        if (!st || !split(id, &run, &view)) return false;
        std::lock_guard<std::mutex> lock(st->m);
        Run* r = st->find(run);
        if (!r || !r->conn) return false;
        const auto it = r->conn->series().find(view);
        if (it == r->conn->series().end()) return false;
        values->assign(static_cast<std::size_t>(r->offset), std::numeric_limits<float>::quiet_NaN());
        values->insert(values->end(), it->second.values.begin(), it->second.values.end());
        if (gen) {
            const auto g = r->series_gen.find(view);
            *gen = g == r->series_gen.end() ? 0 : g->second.second;
        }
        return true;
    }

    std::vector<cb::CatalogEntry> entries() {
        std::vector<cb::CatalogEntry> out;
        auto st = state_.lock();
        if (!st) return out;
        std::lock_guard<std::mutex> lock(st->m);
        for (const std::string& id : st->order) {
            Run* r = st->find(id);
            if (!r || !r->conn) continue;
            for (const auto& [view, ser] : r->conn->series()) {
                cb::CatalogEntry e;
                e.kind = cb::CatalogKind::run;
                e.name = id + "/" + view;
                e.id = cb::make_catalog_id(cb::CatalogKind::run, e.name);
                e.title = view + " of " + id;
                if (!r->conn->session().empty()) e.origin.session = r->conn->session();
                e.origin.view = view;
                e.origin.view_type = "series";
                e.origin.run_name = id;
                e.dtype = cb::CatalogDtype::f32;
                const std::size_t n = static_cast<std::size_t>(r->offset) + ser.values.size();
                e.shape = std::vector<std::int64_t>{static_cast<std::int64_t>(n)};
                e.bytes = n * sizeof(float);
                e.wire = true;
                out.push_back(std::move(e));
            }
        }
        std::sort(out.begin(), out.end(), [](const cb::CatalogEntry& a, const cb::CatalogEntry& b) { return a.id < b.id; });
        return out;
    }

    std::weak_ptr<RunnerState> state_;
};

}  // namespace

// ---- the runner ------------------------------------------------------------

std::string runner_python(const ModelRunner::Options& options, std::string* error) {
    std::string configured = options.python;
    if (configured.empty())
        if (const char* e = std::getenv("CALIPER_RUN_PYTHON"); e && *e) configured = e;
    if (configured.empty()) {
#ifdef _WIN32
        const fs::path venv = fs::u8path(options.workspace) / ".venv" / "Scripts" / "python.exe";
#else
        const fs::path venv = fs::u8path(options.workspace) / ".venv" / "bin" / "python";
#endif
        std::error_code ec;
        if (fs::exists(venv, ec)) return venv.u8string();
    }
    // An absolute path is used as it is, so that a virtual environment's
    // interpreter keeps its environment.
    if (!configured.empty() && fs::u8path(configured).is_absolute()) {
        std::error_code ec;
        if (fs::is_regular_file(fs::u8path(configured), ec)) return configured;
        if (error) *error = "The Python interpreter " + configured + " is not a file.";
        return {};
    }
    const cb::PythonInterpreter py = cb::resolve_python(configured);
    if (py.error) {
        if (error) *error = py.error.message;
        return {};
    }
    return py.path;
}

namespace {

// The child of a run (design/hosted-runs.md, Process Contract): the
// interpreter on the script with its arguments, in the workspace, with the
// app's environment without the variables that start with CALIPER_RUN_ and
// without PYTHONUNBUFFERED, then the variables of the contract, `extra` and
// PYTHONUNBUFFERED=1, and the output into the run's output.log.
ChildSpec run_spec(const ModelRunner::Options& o, const std::string& python, const Run& r,
                   const std::vector<std::pair<std::string, std::string>>& extra) {
    ChildSpec spec;
    spec.program = python;
    if (r.kind == "inference") {
        // Use installed packages, including editable installs, without lookup
        // through the workspace or Python environment overrides. Keep the
        // selected interpreter and its normal site packages.
        spec.args = {"-E", "-P", "-u", "-m", "caliper.interp._entry", "--model=" + r.model,
                     "--dictionary=" + r.dictionary};
    } else {
        spec.args.push_back(r.script_path);
        spec.args.insert(spec.args.end(), r.entry.args.begin(), r.entry.args.end());
    }
    for (auto& kv : current_environment())
        if (kv.first.rfind("CALIPER_RUN_", 0) != 0 && kv.first != "PYTHONUNBUFFERED") spec.env.push_back(kv);
    spec.env.emplace_back("CALIPER_RUN_ID", r.entry.id);
    spec.env.emplace_back("CALIPER_RUN_DIR", r.dir);
    spec.env.emplace_back("CALIPER_RUN_LINK_FILE", r.link_file);
    spec.env.emplace_back("CALIPER_RUN_APP_PID", std::to_string(current_process_id()));
    // The private run environment already carries parameters and interventions.
    // Keep the initial prompt separate from fork parameter overrides and argv.
    if (r.kind == "inference") spec.env.emplace_back("CALIPER_RUN_PROMPT", r.prompt);
    if (!r.entry.params.empty()) spec.env.emplace_back("CALIPER_RUN_PARAMS", r.entry.params.dump());
    if (!r.interventions.empty()) spec.env.emplace_back("CALIPER_RUN_INTERVENTIONS", r.interventions);
    spec.env.insert(spec.env.end(), extra.begin(), extra.end());
    spec.env.emplace_back("PYTHONUNBUFFERED", "1");
    spec.cwd = o.workspace;
    spec.output = r.output;
    return spec;
}

// The step of a kept checkpoint's file name, checkpoint-<step>.<ext> (the
// process contract): 1 to 15 digits, then a dot and 1 to 8 characters of
// a-z and 0-9. False for any other name.
bool kept_checkpoint_step(const std::string& name, std::uint64_t* step, std::string* ext) {
    const std::string prefix = "checkpoint-";
    if (name.rfind(prefix, 0) != 0) return false;
    std::size_t k = prefix.size();
    std::uint64_t v = 0;
    for (; k < name.size() && name[k] >= '0' && name[k] <= '9' && k - prefix.size() < 15; ++k)
        v = v * 10 + static_cast<std::uint64_t>(name[k] - '0');
    if (k == prefix.size() || k >= name.size() || name[k] != '.') return false;
    const std::string e = name.substr(k + 1);
    if (e.empty() || e.size() > 8) return false;
    for (const char ch : e)
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'))) return false;
    *step = v;
    *ext = e;
    return true;
}

// The newest kept checkpoint of the run folder `run_dir` at or before
// `step`, from <run dir>/kept/ (the process contract), as its path, step and
// extension; false when there is none. *oldest gets the step of the oldest
// kept checkpoint when there is one. Only a regular file counts, never a link.
bool newest_kept_checkpoint(const std::string& run_dir, std::uint64_t step, fs::path* path, std::uint64_t* found,
                            std::string* ext, std::optional<std::uint64_t>* oldest) {
    std::error_code ec;
    bool any = false;
    fs::directory_iterator it(fs::u8path(run_dir) / "kept", ec);
    for (const fs::directory_iterator end; !ec && it != end; it.increment(ec)) {
        std::error_code sec;
        if (!fs::is_regular_file(it->symlink_status(sec)) || sec) continue;
        std::uint64_t k = 0;
        std::string e;
        if (!kept_checkpoint_step(it->path().filename().u8string(), &k, &e)) continue;
        if (!*oldest || k < **oldest) *oldest = k;
        if (k <= step && (!any || k > *found)) {
            any = true;
            *found = k;
            *path = it->path();
            *ext = e;
        }
    }
    return any;
}

}  // namespace

ModelRunner::ModelRunner(Options options) : state_(std::make_shared<RunnerState>()) {
    state_->options = std::move(options);
    state_->thread = std::thread([this] { loop(); });
}

ModelRunner::~ModelRunner() { shutdown(); }

void ModelRunner::set_app_control(cb::RunControl* control) {
    std::lock_guard<std::mutex> lock(state_->m);
    state_->app_control = control;
}

void ModelRunner::set_event_sink(cb::EventSink* sink) {
    std::lock_guard<std::mutex> lock(state_->sink_m);
    state_->sink = sink;
}

std::shared_ptr<cb::CatalogProvider> ModelRunner::catalog_provider() { return std::make_shared<RunSources>(state_); }

std::shared_ptr<cb::CatalogDataSource> ModelRunner::data_source() { return std::make_shared<RunSources>(state_); }

std::string ModelRunner::frame(bool attached, const std::string& attached_link) {
    std::lock_guard<std::mutex> lock(state_->m);
    state_->attached = attached;
    state_->attached_link = attached ? attached_link : std::string();
    std::string next;
    next.swap(state_->attach_next);
    if (!next.empty() && attached && next == attached_link) next.clear();
    return next;
}

std::vector<cb::RunEntry> ModelRunner::list() {
    std::vector<cb::RunEntry> out;
    cb::RunControl* app = nullptr;
    bool user = false;
    {
        std::lock_guard<std::mutex> lock(state_->m);
        bool managed_attached = false;
        for (const std::string& id : state_->order) {
            out.push_back(state_->entry_of(*state_->runs.at(id)));
            managed_attached = managed_attached || out.back().attached;
        }
        app = state_->app_control;
        user = state_->attached && !managed_attached;
    }
    if (user && app) {
        const cb::RunState st = app->state();
        if (st.attached) {
            cb::RunEntry e;
            e.id = cb::kUserRunId;
            e.managed = false;
            e.state = st.state == "paused" ? "paused" : "running";
            e.script = st.script;
            if (st.step && *st.step >= 0) e.step = static_cast<std::uint64_t>(*st.step);
            e.attached = true;
            out.push_back(std::move(e));
        }
    }
    return out;
}

cb::Error ModelRunner::start(const cb::RunStartRequest& request, cb::RunEntry* out) {
    RunnerState& s = *state_;
    std::string py_error;
    const std::string python = runner_python(s.options, &py_error);
    if (python.empty()) return fail(cb::kRunStartFailed, py_error.empty() ? "No Python interpreter was found." : py_error);
    std::vector<cb::Event> events;
    {
        std::lock_guard<std::mutex> lock(s.m);
        if (s.shut) return fail(cb::kRunStartFailed, "The model runner has shut down.");
        const fs::path runs = fs::u8path(s.options.workspace) / "runs";
        std::string id = request.id;
        std::error_code ec;
        if (id.empty()) {
            const std::string base = request.kind == "inference" ? "inference" : id_base(request.script);
            for (int n = 1; n < 100000 && id.empty(); ++n) {
                const std::string c = base + "-" + std::to_string(n);
                if (!s.find(c) && !fs::exists(runs / c, ec)) id = c;
            }
        } else if (!cb::valid_run_id(id)) {
            return fail(cb::kRunBadId, "A run id has 1 to 40 characters, each a-z, 0-9 or -.", "args.id");
        } else if (s.find(id) || fs::exists(runs / id, ec)) {
            return fail(cb::kRunExists, "A run " + id + " exists in this workspace (runs/" + id + "); choose another id.",
                        "args.id");
        }
        const fs::path dir = runs / id;
        fs::create_directories(dir, ec);
        if (ec) return fail(cb::kRunStartFailed, "The run's folder runs/" + id + " could not be made: " + ec.message() + ".");

        auto run = std::make_unique<Run>();
        run->entry.id = id;
        run->entry.script = request.script;
        run->entry.args = request.args;
        run->entry.params = request.params.is_object() ? request.params : json::object();
        run->script_path = request.script_path;
        run->kind = request.kind;
        run->model = request.model;
        run->dictionary = request.dictionary;
        run->prompt = request.prompt;
        run->dir = dir.u8string();
        run->link_file = (dir / "link.txt").u8string();
        run->output = (dir / "output.log").u8string();

        std::string err;
        run->proc = start_child(run_spec(s.options, python, *run, {}), &err);
        if (!run->proc) {
            fs::remove_all(dir, ec);
            return fail(cb::kRunStartFailed, err);
        }
        run->started = Clock::now();
        run->control = std::make_unique<ConnectionControl>(state_.get(), id);
        *out = s.entry_of(*run);
        events.push_back(RunnerState::event_of(*out));
        s.order.push_back(id);
        s.runs[id] = std::move(run);
        ++s.revision;
        ++s.changes;
        s.changed.notify_all();
    }
    s.wake.notify_all();
    s.push_events(std::move(events));
    return {};
}

cb::Error ModelRunner::fork(const cb::RunForkRequest& request, cb::RunEntry* out) {
    RunnerState& s = *state_;
    // A fork from a kept checkpoint starts at once, thus it needs the interpreter now.
    std::string python;
    if (request.step) {
        std::string py_error;
        python = runner_python(s.options, &py_error);
        if (python.empty())
            return fail(cb::kRunStartFailed, py_error.empty() ? "No Python interpreter was found." : py_error);
    }
    std::vector<cb::Event> events;
    {
        std::lock_guard<std::mutex> lock(s.m);
        if (s.shut) return fail(cb::kRunStartFailed, "The model runner has shut down.");
        Run* parent = s.find(request.parent);
        if (!parent) return fail(cb::kRunNotFound, "There is no managed run " + request.parent + "; run.list lists the runs.", "args.run");
        const bool open = parent->conn && parent->conn->state() == RunConnection::State::open;
        if (finished(parent->entry.state) || (!request.step && !open))
            return fail(cb::kRunNotReady,
                        "The run " + request.parent + " is " + parent->entry.state +
                            ", and a fork needs a run that runs or is paused, with its stream connected.",
                        "args.run");
        const fs::path runs = fs::u8path(s.options.workspace) / "runs";
        std::error_code ec;
        std::string id = request.id;
        if (id.empty()) {
            const std::string base = parent->kind == "inference" ? "inference" : id_base(parent->entry.script);
            for (int n = 1; n < 100000 && id.empty(); ++n) {
                const std::string c = base + "-" + std::to_string(n);
                if (!s.find(c) && !fs::exists(runs / c, ec)) id = c;
            }
        } else if (!cb::valid_run_id(id)) {
            return fail(cb::kRunBadId, "A run id has 1 to 40 characters, each a-z, 0-9 or -.", "args.id");
        } else if (s.find(id) || fs::exists(runs / id, ec)) {
            return fail(cb::kRunExists, "A run " + id + " exists in this workspace (runs/" + id + "); choose another id.",
                        "args.id");
        }
        // A fork from a past step: the newest kept checkpoint at or before it,
        // which no until_step of an intervention may reach.
        fs::path kept;
        std::uint64_t kept_step = 0;
        std::string kept_ext;
        if (request.step) {
            std::optional<std::uint64_t> oldest;
            if (!newest_kept_checkpoint(parent->dir, *request.step, &kept, &kept_step, &kept_ext, &oldest))
                return fail(cb::kRunNotReady,
                            "The run " + request.parent + " kept no checkpoint at or before step " +
                                std::to_string(*request.step) +
                                (oldest ? "; its oldest kept checkpoint is at step " + std::to_string(*oldest) + "."
                                        : "; it keeps checkpoints from its first training step on."),
                            "args.step");
            for (std::size_t i = 0; i < request.interventions.size(); ++i) {
                const json& iv = request.interventions[i];
                const auto u = iv.find("until_step");
                if (u == iv.end() || !u->is_number_integer()) continue;
                const std::uint64_t until = u->is_number_unsigned() ? u->get<std::uint64_t>()
                                            : u->get<std::int64_t>() < 0 ? 0
                                                                         : static_cast<std::uint64_t>(u->get<std::int64_t>());
                if (until <= kept_step)
                    return fail(cb::kInterveneBadRange,
                                "until_step " + std::to_string(until) +
                                    " is not above the step " + std::to_string(kept_step) +
                                    " of the kept checkpoint, where the fork starts.",
                                "args.interventions[" + std::to_string(i) + "].until_step");
            }
        }
        std::uint32_t rid = 0;
        if (!request.step) {
            std::string err;
            rid = parent->conn->send_control("run", kCheckpointAction, json(), 1, &err);
            if (!rid) return fail(cb::kRunNotReady, "The checkpoint request could not be sent to " + request.parent + ": " + err);
        }
        const fs::path dir = runs / id;
        fs::create_directories(dir, ec);
        if (ec) return fail(cb::kRunStartFailed, "The run's folder runs/" + id + " could not be made: " + ec.message() + ".");
        // The fork reads its own link of the kept checkpoint, or a copy, so that
        // the parent's thinning of its kept checkpoints cannot remove it. A
        // checkpoint that the thinning removed in between gives the next newest.
        fs::path resume;
        for (int attempt = 0; request.step && resume.empty(); ++attempt) {
            const fs::path target = dir / ("resume-" + std::to_string(kept_step) + "." + kept_ext);
            std::error_code lec;
            fs::create_hard_link(kept, target, lec);
            if (lec) {
                lec.clear();
                fs::copy_file(kept, target, fs::copy_options::overwrite_existing, lec);
            }
            if (!lec) {
                resume = target;
                break;
            }
            std::optional<std::uint64_t> oldest;
            if (attempt >= 2 || !newest_kept_checkpoint(parent->dir, *request.step, &kept, &kept_step, &kept_ext, &oldest)) {
                fs::remove_all(dir, ec);
                return fail(cb::kRunStartFailed, "The kept checkpoint " + kept.filename().u8string() + " of " +
                                                     request.parent + " could not be read: " + lec.message() + ".");
            }
        }

        auto run = std::make_unique<Run>();
        run->entry.id = id;
        run->entry.script = parent->entry.script;
        run->entry.args = parent->entry.args;
        run->entry.params = request.params.is_object() ? request.params : json::object();
        run->entry.parent = request.parent;
        run->script_path = parent->script_path;
        run->kind = parent->kind;
        run->model = parent->model;
        run->dictionary = parent->dictionary;
        run->prompt = parent->prompt;
        run->dir = dir.u8string();
        run->link_file = (dir / "link.txt").u8string();
        run->output = (dir / "output.log").u8string();
        if (!request.interventions.empty()) run->interventions = request.interventions.dump();
        if (request.step) {
            run->offset = std::min<std::uint64_t>(kept_step, s.options.max_series_values);
            std::string err;
            run->proc = start_child(
                run_spec(s.options, python, *run, {{"CALIPER_RUN_RESUME", resume.u8string()}, {"CALIPER_RUN_PARENT", request.parent}}),
                &err);
            if (!run->proc) {
                fs::remove_all(dir, ec);
                return fail(cb::kRunStartFailed, err);
            }
        } else {
            run->waits_for = request.parent;
            run->checkpoint_request = rid;
        }
        run->started = Clock::now();
        run->control = std::make_unique<ConnectionControl>(state_.get(), id);
        *out = s.entry_of(*run);
        events.push_back(RunnerState::event_of(*out));
        s.order.push_back(id);
        s.runs[id] = std::move(run);
        ++s.revision;
        ++s.changes;
        s.changed.notify_all();
    }
    s.wake.notify_all();
    s.push_events(std::move(events));
    return {};
}

cb::Error ModelRunner::stop(const std::string& id, double kill_after_s, cb::RunEntry* out) {
    RunnerState& s = *state_;
    std::vector<cb::Event> events;
    {
        std::lock_guard<std::mutex> lock(s.m);
        Run* r = s.find(id);
        if (!r) return fail(cb::kRunNotFound, "There is no managed run " + id + "; run.list lists the runs.", "args.run");
        if (!finished(r->entry.state) && !r->proc) {
            // A fork that waits for its parent's checkpoint has no process:
            // it ends here, and the checkpoint's reply starts nothing.
            r->entry.stopping = true;
            r->waits_for.clear();
            r->entry.state = "exited";
            events.push_back(RunnerState::event_of(s.entry_of(*r)));
            ++s.changes;
            s.changed.notify_all();
        } else if (!finished(r->entry.state)) {
            r->entry.stopping = true;
            const auto now = Clock::now();
            bool asked = false;
            if (r->conn && r->conn->state() == RunConnection::State::open && kill_after_s > 0) {
                std::string err;
                asked = r->conn->send_control("run", kStopAction, json(), 1, &err) != 0;
            }
            const auto limit = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(asked ? kill_after_s : 0.0));
            if (!r->end_at || now + limit < *r->end_at) r->end_at = now + limit;
        }
        *out = s.entry_of(*r);
    }
    s.wake.notify_all();
    s.push_events(std::move(events));
    return {};
}

cb::Error ModelRunner::logs(const std::string& id, int lines, std::vector<std::string>* out) {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(state_->m);
        Run* r = state_->find(id);
        if (!r) return fail(cb::kRunNotFound, "There is no managed run " + id + "; run.list lists the runs.", "args.run");
        path = r->output;
    }
    const std::size_t want = static_cast<std::size_t>(std::max(1, lines));
    std::vector<std::string> all = tail_lines(path);
    if (all.size() < want) {   // the older lines of a rotated output.log
        std::vector<std::string> older = tail_lines(path + ".1");
        all.insert(all.begin(), older.begin(), older.end());
    }
    const std::size_t keep = std::min(all.size(), want);
    out->assign(all.end() - static_cast<std::ptrdiff_t>(keep), all.end());
    return {};
}

cb::RunControl* ModelRunner::control_of(const std::string& id, cb::Error* error) {
    std::lock_guard<std::mutex> lock(state_->m);
    RunnerState& s = *state_;
    if (id == cb::kUserRunId) {
        bool managed = false;
        for (const auto& [rid, r] : s.runs) managed = managed || (s.attached && !r->link.empty() && r->link == s.attached_link);
        if (s.attached && !managed && s.app_control) return s.app_control;
        *error = fail(cb::kRunNotFound, "No script of the person is attached to the app now; run.list lists the runs.", "args.run");
        return nullptr;
    }
    Run* r = s.find(id);
    if (!r) {
        *error = fail(cb::kRunNotFound, "There is no run " + id + "; run.list lists the runs.", "args.run");
        return nullptr;
    }
    if (finished(r->entry.state) || !r->conn || r->conn->state() != RunConnection::State::open) {
        *error = fail(cb::kRunNotReady, "The run " + id + " is " + r->entry.state + " and has no open stream.", "args.run");
        return nullptr;
    }
    if (s.attached && r->link == s.attached_link && s.app_control) return s.app_control;
    s.attach_next = r->link;   // the app switches to it at its next frame
    return r->control.get();
}

std::vector<cb::InterventionPoint> ModelRunner::intervention_points(const std::string& id) {
    ConnectionControl* control = nullptr;
    {
        std::lock_guard<std::mutex> lock(state_->m);
        if (Run* r = state_->find(id)) control = r->control.get();   // a run stays in the table
    }
    return control ? control->intervention_points() : std::vector<cb::InterventionPoint>();
}

void ModelRunner::shutdown() {
    RunnerState& s = *state_;
    {
        std::lock_guard<std::mutex> lock(s.m);
        if (s.shut) return;
        s.shut = true;
        s.stopping = true;
    }
    s.wake.notify_all();
    if (s.thread.joinable()) s.thread.join();
    std::lock_guard<std::mutex> lock(s.m);
    std::vector<Run*> alive;
    for (auto& [id, r] : s.runs) {
        std::int64_t code = 0;
        if (r->proc && !r->proc->poll(&code)) {
            r->proc->terminate();
            alive.push_back(r.get());
        }
        if (r->conn) r->conn->close();
    }
    const int grace_ms = static_cast<int>(s.options.kill_grace_s * 1000);
    const auto until = Clock::now() + std::chrono::milliseconds(grace_ms);
    for (Run* r : alive) {
        std::int64_t code = 0;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(until - Clock::now()).count();
        if (!r->proc->wait(static_cast<int>(std::max<std::int64_t>(0, left)), &code)) {
            r->proc->kill();
            r->proc->wait(2000, &code);
        }
        r->entry.exit_code = code;
        r->entry.state = "exited";
    }
}

bool ModelRunner::wait_for(const std::function<bool(const std::vector<cb::RunEntry>&)>& done, int timeout_ms) {
    RunnerState& s = *state_;
    const auto until = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        std::vector<cb::RunEntry> entries;
        std::uint64_t seen = 0;
        {
            std::lock_guard<std::mutex> lock(s.m);
            for (const std::string& id : s.order) entries.push_back(s.entry_of(*s.runs.at(id)));
            seen = s.changes;
        }
        // Without the lock: `done` may read the runner, such as its sources.
        if (done(entries)) return true;
        std::unique_lock<std::mutex> lock(s.m);
        if (!s.changed.wait_until(lock, until, [&] { return s.changes != seen; })) {
            lock.unlock();
            entries.clear();
            return done(list());
        }
    }
}

std::vector<std::int64_t> ModelRunner::running_pids() {
    std::vector<std::int64_t> out;
    std::lock_guard<std::mutex> lock(state_->m);
    for (auto& [id, r] : state_->runs) {
        std::int64_t code = 0;
        if (r->proc && !r->proc->poll(&code)) out.push_back(r->proc->pid());
    }
    return out;
}

// The runner's thread: polls each child and each connection, waits for link
// files and checkpoints, ends stopped runs, and wakes wait_for().
void ModelRunner::loop() {
    RunnerState& s = *state_;
    std::unique_lock<std::mutex> lock(s.m);
    while (!s.stopping) {
        if (s.options.before_pass) {
            lock.unlock();
            s.options.before_pass();
            lock.lock();
            if (s.stopping) break;
        }
        bool changed = false;
        std::vector<cb::Event> events;
        const auto now = Clock::now();
        const auto limit = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s.options.start_limit_s));
        auto set_state = [&](Run& r, const std::string& state) {
            if (r.entry.state == state) return;
            r.entry.state = state;
            events.push_back(RunnerState::event_of(s.entry_of(r)));
            changed = true;
        };
        auto fail_run = [&](Run& r, const std::string& why) {
            if (!r.entry.error) r.entry.error = why;
            if (r.proc) {
                if (!r.end_at || *r.end_at > now) r.end_at = now;
            } else {
                set_state(r, "failed");   // a fork that never started
            }
            changed = true;
        };
        // The checkpoint replies of run `id`: each starts the fork that waits
        // for it, or fails it.
        auto take_checkpoints = [&](Run& r, const std::string& id) {
            for (const RunConnection::Reply& rep : r.conn->take_replies()) {
                for (const std::string& other : s.order) {
                    Run& f = *s.runs.at(other);
                    if (f.waits_for != id || f.checkpoint_request != rep.request_id || rep.status == "queued") continue;
                    f.waits_for.clear();
                    if (rep.status != "ok" || rep.message.empty()) {
                        fail_run(f, "The parent's caliper.checkpoint failed: " +
                                        (rep.error_message.empty() ? std::string("it returned no checkpoint path.")
                                                                   : rep.error_message));
                        continue;
                    }
                    // The checkpoint is there: the fork starts from it.
                    std::string py_error;
                    const std::string python = runner_python(s.options, &py_error);
                    const ChildSpec spec =
                        run_spec(s.options, python, f, {{"CALIPER_RUN_RESUME", rep.message}, {"CALIPER_RUN_PARENT", id}});
                    // checkpoint-<step>.<ext> (the process contract) gives the step of the checkpoint.
                    const std::string file = fs::u8path(rep.message).filename().u8string();
                    if (file.rfind("checkpoint-", 0) == 0) {
                        std::uint64_t step = 0;
                        std::size_t k = 11;
                        for (; k < file.size() && file[k] >= '0' && file[k] <= '9' && step < (1ull << 40); ++k)
                            step = step * 10 + static_cast<std::uint64_t>(file[k] - '0');
                        if (k > 11) f.offset = std::min<std::uint64_t>(step, s.options.max_series_values);
                    }
                    std::string err = py_error;
                    if (!python.empty()) f.proc = start_child(spec, &err);
                    if (!f.proc) {
                        fail_run(f, err);
                    } else {
                        f.started = Clock::now();
                    }
                    changed = true;
                }
            }
        };
        for (const std::string& id : s.order) {
            Run& r = *s.runs.at(id);
            if (finished(r.entry.state)) continue;
            // 1. The process's end ends the run.
            std::int64_t code = 0;
            if (r.proc && r.proc->poll(&code)) {
                r.entry.exit_code = code;
                if (r.conn) {
                    // The last values and replies that the process sent
                    // before its end, read until its connection closes: a
                    // checkpoint reply that came in this pass still starts
                    // its fork.
                    const auto until = Clock::now() + std::chrono::milliseconds(kDrainMs);
                    while (r.conn->state() != RunConnection::State::closed && Clock::now() < until) r.conn->poll(10);
                    take_checkpoints(r, id);
                    r.conn->close();
                }
                // A stop is clean when the script ended with 0 at its checkpoint, or
                // when the runner ended it after the limit. A script that ends on its
                // own with another code during a stop failed, as without a stop.
                const bool clean = (code == 0 || (r.entry.stopping && r.terminated_at.has_value())) && !r.entry.error;
                if (!clean && !r.entry.error)
                    r.entry.error = "The script ended with exit code " + std::to_string(code) + "; run.logs shows its output.";
                set_state(r, clean ? "exited" : "failed");
                for (const std::string& other : s.order) {   // forks that wait for its checkpoint
                    Run& f = *s.runs.at(other);
                    if (f.waits_for == id) {
                        f.waits_for.clear();
                        fail_run(f, "The parent " + id + " ended before it saved a checkpoint.");
                    }
                }
                continue;
            }
            // 2. A fork waits for its parent's checkpoint.
            if (!r.proc && !r.waits_for.empty()) {
                if (now - r.started > limit) {
                    r.waits_for.clear();
                    fail_run(r, "The parent saved no checkpoint within the start limit.");
                }
                continue;
            }
            // 3. A starting run waits for its link file.
            if (r.proc && !r.conn) {
                std::error_code ec;
                if (fs::exists(fs::u8path(r.link_file), ec)) {
                    std::ifstream f(fs::u8path(r.link_file));
                    std::string link;
                    std::getline(f, link);
                    while (!link.empty() && (link.back() == '\r' || link.back() == ' ')) link.pop_back();
                    if (!link.empty()) {
                        r.link = link;
                        r.conn = std::make_unique<RunConnection>(s.options.max_series_values);
                        std::string err;
                        if (!r.conn->open(link, &err)) fail_run(r, err);
                        changed = true;
                    }
                } else if (now - r.started > limit) {
                    fail_run(r, "The script wrote no stream link within " +
                                    std::to_string(static_cast<int>(s.options.start_limit_s)) +
                                    " seconds; its caliper.watch() line writes it.");
                }
            }
            // 4. The connection: the run state, the replies and the series.
            if (r.conn && r.conn->state() != RunConnection::State::closed) {
                const bool was_open = r.conn->state() == RunConnection::State::open;
                if (r.conn->poll(0)) changed = true;
                if (!was_open && r.conn->state() == RunConnection::State::open) {
                    set_state(r, "running");
                    s.attach_next = r.link;   // a new run is the one the app shows
                }
                const json& c = r.conn->controls();
                if (c.is_object()) {
                    const json run = c.value("run", json::object());
                    if (run.contains("step") && run["step"].is_number_unsigned()) r.entry.step = run["step"].get<std::uint64_t>();
                    if (r.entry.state == "running" || r.entry.state == "paused")
                        set_state(r, run.value("state", std::string()) == "paused" ? "paused" : "running");
                }
                take_checkpoints(r, id);
                for (const auto& [view, ser] : r.conn->series()) {
                    auto& g = r.series_gen[view];
                    if (g.first != ser.total || g.second == 0) {
                        if (g.second == 0) ++s.revision;
                        g.first = ser.total;
                        g.second = ++s.generation;
                        changed = true;
                    }
                }
            }
            // 5. The output's size limit.
            if (r.proc) rotate_output(r.output, s.options.output_limit_bytes);
            // 6. A stop: ask the process to end at its limit, then end it.
            if (r.proc && r.end_at && now >= *r.end_at) {
                if (!r.terminated_at) {
                    r.proc->terminate();
                    r.terminated_at = now;
                } else if (now - *r.terminated_at >
                           std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s.options.kill_grace_s))) {
                    r.proc->kill();
                }
            }
        }
        if (changed) {
            ++s.changes;
            s.changed.notify_all();
        }
        if (!events.empty()) {
            lock.unlock();
            s.push_events(std::move(events));
            lock.lock();
            continue;
        }
        s.wake.wait_for(lock, std::chrono::milliseconds(20));
    }
}

}  // namespace caliper_runner
