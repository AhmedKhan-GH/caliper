#pragma once
// The app's host of stages (design/gpu-programs.md, Viewer Stage Interface):
// the StageHost (caliper_builder/session/program_host.h) that compiles the
// stages of the open spec's scene panels for the scene compiler
// (scene/stages.h), runs their time watchdog and pushes their `program`
// events. The app gives it to its session with AppSession::set_stage_host()
// before the first frame, and takes it back with null before exit.
#include <caliper_builder/session/program_host.h>

namespace caliper_host {

// The one stage host of the app, or null while stages are not built. It lives
// until the process ends. Frame thread.
caliper_builder::StageHost* app_stage_host();

}  // namespace caliper_host
