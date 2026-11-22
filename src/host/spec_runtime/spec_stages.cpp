// The app's host of stages (spec_stages.h, design/gpu-programs.md). The
// interface is fixed; the host comes next, and until then the app has none.
#include "spec_runtime/spec_stages.h"

namespace caliper_host {

caliper_builder::StageHost* app_stage_host() { return nullptr; }

}  // namespace caliper_host
