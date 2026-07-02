// Compile-only proof that the installed caliper-sdk package is self-contained.
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
static_assert(CALIPER_ABI_EPOCH == 2, "epoch-2 SDK reachable from installed prefix");
