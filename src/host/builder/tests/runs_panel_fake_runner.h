#pragma once
// A model runner for the runs panel's tests and its shot tool
// (runs_panel.h): it records each request and answers from a table of runs
// that the caller sets. It starts no process.
#include <caliper_builder/session/runner_commands.h>

#include <string>
#include <utility>
#include <vector>

namespace runs_panel_test {

namespace cb = caliper_builder;

struct FakeRunner final : cb::ModelRunner {
    std::vector<cb::RunEntry> runs;
    std::vector<cb::RunStartRequest> starts;
    std::vector<cb::RunForkRequest> forks;
    std::vector<std::pair<std::string, double>> stops;

    cb::RunEntry* find(const std::string& id) {
        for (auto& e : runs)
            if (e.id == id) return &e;
        return nullptr;
    }

    std::vector<cb::RunEntry> list() override { return runs; }
    cb::Error start(const cb::RunStartRequest& r, cb::RunEntry* out) override {
        starts.push_back(r);
        const std::string id = r.id.empty() ? "run-" + std::to_string(runs.size() + 1) : r.id;
        if (find(id)) return cb::Error(cb::kRunExists, "A run " + id + " exists in this workspace.", "args.id");
        cb::RunEntry e;
        e.id = id;
        e.script = r.script;
        e.args = r.args;
        e.params = r.params;
        runs.push_back(e);
        *out = e;
        return {};
    }
    cb::Error fork(const cb::RunForkRequest& r, cb::RunEntry* out) override {
        forks.push_back(r);
        const cb::RunEntry* parent = find(r.parent);
        if (!parent) return cb::Error(cb::kRunNotFound, "There is no run " + r.parent + ".");
        cb::RunEntry e;
        e.id = r.id.empty() ? r.parent + "-fork" : r.id;
        e.script = parent->script;
        e.args = parent->args;
        e.parent = r.parent;
        e.params = r.params;
        runs.push_back(e);
        *out = e;
        return {};
    }
    cb::Error stop(const std::string& run, double after, cb::RunEntry* out) override {
        stops.emplace_back(run, after);
        cb::RunEntry* e = find(run);
        if (!e) return cb::Error(cb::kRunNotFound, "There is no run " + run + ".");
        e->stopping = true;
        *out = *e;
        return {};
    }
    cb::Error logs(const std::string& run, int, std::vector<std::string>* out) override {
        if (!find(run)) return cb::Error(cb::kRunNotFound, "There is no run " + run + ".");
        out->clear();
        return {};
    }
    cb::RunControl* control_of(const std::string& run, cb::Error* error) override {
        *error = cb::Error(cb::kRunNotReady, "The run " + run + " has no connection.");
        return nullptr;
    }
};

// The runs of the shot and of the layout tests: a run, its fork with a lower
// learning rate, an evaluation that ended and a start that failed.
inline std::vector<cb::RunEntry> sample_runs() {
    std::vector<cb::RunEntry> out(4);
    out[0].id = "train-1";
    out[0].state = "running";
    out[0].script = "scripts/train.py";
    out[0].params = {{"lr", 0.05}};
    out[0].step = 1200;
    out[1].id = "train-2";
    out[1].state = "running";
    out[1].script = "scripts/train.py";
    out[1].parent = "train-1";
    out[1].params = {{"lr", 0.01}};
    out[1].step = 1300;
    out[1].attached = true;
    out[2].id = "eval-1";
    out[2].state = "exited";
    out[2].script = "scripts/eval.py";
    out[2].step = 40;
    out[2].exit_code = 0;
    out[3].id = "train-3";
    out[3].state = "failed";
    out[3].script = "scripts/train.py";
    out[3].error = "No link file after 120 seconds.";
    out[3].exit_code = 1;
    return out;
}

}  // namespace runs_panel_test
