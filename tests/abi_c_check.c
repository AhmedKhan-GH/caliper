/* Every public header must compile as C *standalone* — newest-first include
 * order so a header missing its own includes can't hide behind a sibling
 * that pulled them earlier (caught the tensor_bridge stdbool gap, C3 review). */
#include <caliper/services/feed_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/metrics_v1_1.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/tensor.h>
#include <caliper/services/device_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1.h>
#include <caliper/abi.h>
int caliper_abi_c_check_anchor(void) { return CALIPER_ABI_EPOCH; }
