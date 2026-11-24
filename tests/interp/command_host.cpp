// Offline command-sidecar host over the production run connection. No renderer.
#include "runner/run_connection.h"
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>
#include <csignal>
#include <iostream>
#include <thread>

using nlohmann::json;
namespace cb = caliper_builder;
namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }

class ConnectedRun final : public cb::RunControl {
public:
    caliper_runner::RunConnection connection;
    cb::RunState state() override {
        cb::RunState state;
        state.attached = connection.state() == caliper_runner::RunConnection::State::open;
        state.control = connection.control_allowed();
        state.script = "fixture";
        const auto& metadata = connection.controls();
        if (metadata.contains("run")) {
            state.state = metadata["run"].value("state", "none");
            if (metadata["run"].contains("step") && !metadata["run"]["step"].is_null()) state.step = metadata["run"]["step"];
        }
        return state;
    }
    cb::Error set_param(const std::string& name, const json& value) override {
        std::string error;
        if (connection.send_control("set",name,value,0,&error)) return {};
        return cb::Error("run.refused",error);
    }
    cb::Error run_action(const std::string&) override { return cb::Error("run.refused","No fixture action is registered."); }
    cb::Error pause(cb::RunPauseOp, std::int64_t) override { return cb::Error("run.refused","The fixture has no pause action."); }
    cb::CommandOutput interpret(const std::string& method, const json& params, std::uint32_t request) override {
        const auto result = connection.interpret(method,params,request);
        if (result.contains("error")) return cb::CommandOutput::fail(cb::Error(result["error"]["code"],result["error"]["message"]));
        return cb::CommandOutput::ok(result);
    }
};
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::signal(SIGINT,stop);
    std::signal(SIGTERM,stop);
    ConnectedRun run;
    std::string error;
    if (!run.connection.open(argv[1],&error)) { std::cerr << error; return 1; }
    cb::SessionOptions options;
    options.registry.check_results = true;
    auto opened = cb::Session::open(argv[2],options);
    if (opened.error) { std::cerr << opened.error.message; return 1; }
    auto& session = *opened.session;
    session.registry().set_frame_thread(std::this_thread::get_id());
    session.set_run_control(&run);
    cb::CommandServerOptions server_options;
    server_options.token = cb::generate_session_token();
    cb::CommandServer server(session.queue(),server_options);
    if (auto error = server.start()) { std::cerr << error.message; return 1; }
    std::string api;
    if (auto error = cb::write_api_file(argv[3],server.port(),server_options.token,&api)) {
        std::cerr << error.message; return 1;
    }
    bool announced = false;
    while (!stopped) {
        run.connection.poll(5);
        session.queue().drain();
        // The fixture harness grants each concrete command under test.
        for (const auto& pending : session.registry().pending_approvals()) session.queue().approve(pending.approval);
        if (!announced && run.connection.controls().is_object()) {
            std::cout << "ready" << std::endl;
            announced = true;
        }
    }
    session.queue().shutdown();
    server.stop();
    cb::remove_api_file(api);
    return 0;
}
