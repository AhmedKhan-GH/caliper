#pragma once
// ============================================================================
// app_hooks.h: what the app and `caliper shot` give their AppSession
// (app_session.h) that needs nothing of their own state: the options from the
// command line and the environment, the host's job system for the session's
// worker, the body of GET /v0/info for app.info, and the PNG encoder for
// capture.frame. The app adds the applets, the launch, the capture and the
// Activity pane itself. Header-only; built with the builder on or off.
// ============================================================================
#include "app_paths.h"
#include "builder/app_session.h"
#include "export_service.h"
#include "host_cli.h"
#include "host_services.h"
#include "job_system.h"
#include "negotiation.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace caliper_host {

// The workspace of workspace_choice() (--workspace, CALIPER_WORKSPACE, else
// workspace/ under the app-data folder), and the command script of
// CALIPER_COMMAND_SCRIPT with its results in CALIPER_COMMAND_RESULTS.
//
// The command script is a testing and demo aid that may claim the ui origin
// and approve what waits ("approve": true), so it runs only where
// CALIPER_TEST_APPROVALS does: in the hidden app (--hidden) and in `caliper
// shot`, which shows no window (`scripts_allowed`). A visible app refuses it
// whole, reads neither variable, and says why in `refused`.
inline AppSession::Options app_session_options(const CommandLine& cl, bool scripts_allowed = false,
                                               std::string* refused = nullptr) {
    const WorkspaceChoice w =
        workspace_choice(cl, std::getenv("CALIPER_WORKSPACE"), caliper::default_workspace_dir());
    AppSession::Options o;
    o.workspace = w.dir;
    o.chosen = w.chosen;
    o.from = w.from;
    if (const char* s = std::getenv("CALIPER_COMMAND_SCRIPT"); s && *s) o.script = s;
    if (const char* r = std::getenv("CALIPER_COMMAND_RESULTS"); r && *r) o.results = r;
    // The person's agent permissions live beside the app's other settings.
    o.permissions_file = caliper::app_data_path("agent-permissions.json");
    gate_command_script(&o, scripts_allowed, refused);
    return o;
}

inline AppSession::Hooks app_session_hooks() {
    AppSession::Hooks h;
    // The session's worker is a job of the host's job system, so a long
    // command shows in the jobs tray.
    h.run_job = [](const std::string& label, std::function<void()> fn) {
        auto* user = new std::function<void()>(std::move(fn));
        const CaliperJobFn run = [](void* u, const CaliperJobControl*) {
            std::unique_ptr<std::function<void()>> f(static_cast<std::function<void()>*>(u));
            (*f)();
        };
        if (host_job_system().submit(label, run, user) == 0)
            std::thread([user, run] { run(user, nullptr); }).detach();
    };
    h.info_json = [] {
        CliContext ctx;
        ctx.caps = host_caps();
        ctx.sdk_version = host_sdk_version();
        return info_json(ctx);
    };
    h.encode_png = [](const uint8_t* rgba, uint32_t w, uint32_t hh, std::string* png) {
        return export_encode_png(rgba, w, hh, png);
    };
    return h;
}

}  // namespace caliper_host
