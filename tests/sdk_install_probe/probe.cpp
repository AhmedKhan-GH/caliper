// Compile-only proof that the installed caliper-sdk package is self-contained.
#include <caliper/abi_v1.h>
static_assert(CALIPER_APPLET_ABI == 1, "v1 header reachable from installed prefix");
