// The out-of-line part of host_cli.h: the signal handler of `caliper serve`.
// It has C linkage, so it is defined once here rather than inline in the
// header; the executable and caliper_tests compile this file beside the
// header's users.
#include "host_cli.h"

namespace caliper_host {
namespace cli_detail {

extern "C" void caliper_serve_on_signal(int) {
    if (caliper_net::Hub* h = serving_hub().load()) h->stop();
}

} // namespace cli_detail
} // namespace caliper_host
