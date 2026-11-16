// Marks a test binary as a test run for the app-data guard (src/app_paths.h,
// THE GUARD). ctest gives every entry a data folder of its own; run without
// CALIPER_DATA_DIR, as from a shell or an IDE, a binary linked with this file
// stops at its first app-data lookup instead of opening the stores in your
// own app-data folder. Linked as the object library caliper_test_run_marker
// (tests/CMakeLists.txt), so the initializer below is never dropped.
#include "app_paths.h"

namespace {
const bool kTestRun = (caliper::app_data_guard("a test binary"), true);
}  // namespace
