#pragma once
// The app's model runner (design/hosted-runs.md, Model Runner): the
// caliper_builder::ModelRunner that run.list, run.start, run.fork, run.stop and
// run.logs act through. It owns the child processes of the managed runs, its
// own viewer connection to each (run_connection.h) and the table of runs, and
// it serves the run sources run:<id>/<view> to the session's catalog and
// spec runtime.
//
// A start makes the folder <workspace>/runs/<id>/, starts the workspace's
// Python on the script with the variables of the process contract and the
// output in output.log, and answers in the state starting. The runner's
// thread then waits for the link file, connects, marks the run running and
// asks the app to attach to it at its next frame (frame()). A fork sends
// caliper.checkpoint to its parent and starts when the action returns the
// checkpoint's path. A stop sends caliper.stop and ends the process by its id
// after the limit. A run's process exit ends the run. shutdown() ends every
// child by its id, so that none outlives the app.
//
// Threads: the interface functions and frame() run on the frame thread, the
// runner's own thread polls the children and the connections, and the run
// sources may be read from any thread. One mutex guards the run table.
#include "child_process.h"

#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/runner_commands.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace caliper_runner {

struct RunnerState;   // model_runner.cpp

class ModelRunner final : public caliper_builder::ModelRunner {
public:
    struct Options {
        std::string workspace;                 // the canonical workspace folder
        // The interpreter: an absolute path, or a name on the PATH. "" uses
        // $CALIPER_RUN_PYTHON, else <workspace>/.venv's, else python3.
        std::string python;
        double start_limit_s = 120.0;          // the wait for a link file, and for a fork's checkpoint
        double kill_grace_s = 2.0;             // from the request to end to the forced end
        std::size_t max_series_values = 262144;   // the newest values kept for each series
    };

    explicit ModelRunner(Options options);
    // shutdown().
    ~ModelRunner() override;
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // caliper_builder::ModelRunner (runner_commands.h).
    std::vector<caliper_builder::RunEntry> list() override;
    caliper_builder::Error start(const caliper_builder::RunStartRequest& request, caliper_builder::RunEntry* out) override;
    caliper_builder::Error fork(const caliper_builder::RunForkRequest& request, caliper_builder::RunEntry* out) override;
    caliper_builder::Error stop(const std::string& run, double kill_after_s, caliper_builder::RunEntry* out) override;
    caliper_builder::Error logs(const std::string& run, int lines, std::vector<std::string>* out) override;
    caliper_builder::RunControl* control_of(const std::string& run, caliper_builder::Error* error) override;

    // Frame thread, once per frame: whether the app is attached and the
    // stream link it shows ("" for none, or for a session that is not a
    // stream). Returns the link of a managed run that the app should attach
    // to now, or "".
    std::string frame(bool attached, const std::string& attached_link);
    // The app's own run control, which acts on the attached script (null to
    // take it back). Frame thread.
    void set_app_control(caliper_builder::RunControl* control);
    // Where the runner pushes its run events of change runner (null to take
    // it back). Any thread.
    void set_event_sink(caliper_builder::EventSink* sink);
    // The run sources run:<id>/<view>: one object that is the catalog
    // provider and the data source of the kind run. It stays valid after
    // the runner, and then lists nothing.
    std::shared_ptr<caliper_builder::CatalogProvider> catalog_provider();
    std::shared_ptr<caliper_builder::CatalogDataSource> data_source();
    // Ends every child by its process id: a request to end, then a forced end
    // after kill_grace_s, and waits for each exit; stops the runner's thread.
    // Safe to call more than once.
    void shutdown();

    // Tests: waits at most `timeout_ms` until `done` holds for the managed
    // runs, rechecking whenever the runner changed something. `done` runs
    // without the runner's lock, so it may read the runner.
    bool wait_for(const std::function<bool(const std::vector<caliper_builder::RunEntry>&)>& done, int timeout_ms);
    // Tests: the process ids of the children that have not ended.
    std::vector<std::int64_t> running_pids();

private:
    void loop();
    std::shared_ptr<RunnerState> state_;
};

// The name of the child's interpreter for `options`, or "" with *error.
std::string runner_python(const ModelRunner::Options& options, std::string* error);

}  // namespace caliper_runner
