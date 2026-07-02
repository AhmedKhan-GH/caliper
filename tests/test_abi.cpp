#include <doctest/doctest.h>
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/tensor.h>
#include <caliper/services/metrics_v1.h>
#include <cstddef>
#include <string>
#include <type_traits>

// ABI hygiene (PLATFORM.md §6c): POD, struct_size-prefixed, C-safe.
static_assert(std::is_standard_layout_v<CaliperHost>);
static_assert(std::is_standard_layout_v<CaliperFrameInfo>);
static_assert(std::is_standard_layout_v<CaliperAppletAPI>);
static_assert(std::is_standard_layout_v<CaliperAppletDescriptor>);
static_assert(std::is_standard_layout_v<CaliperUiV1>);
static_assert(std::is_standard_layout_v<CaliperLogV1>);
static_assert(offsetof(CaliperHost, struct_size) == 0);
static_assert(offsetof(CaliperFrameInfo, struct_size) == 0);
static_assert(offsetof(CaliperAppletAPI, struct_size) == 0);
static_assert(offsetof(CaliperAppletDescriptor, struct_size) == 0);
static_assert(CALIPER_ABI_EPOCH == 2);

TEST_CASE("abi: descriptor symbol name is fixed") {
    CHECK(std::string(CALIPER_DESCRIPTOR_SYMBOL) == "caliper_applet_descriptor");
}

static_assert(std::is_standard_layout_v<CaliperJobControl>);
static_assert(std::is_standard_layout_v<CaliperJobsV1>);
static_assert(std::is_standard_layout_v<CaliperDeviceV1>);
static_assert(offsetof(CaliperJobControl, struct_size) == 0);
static_assert(offsetof(CaliperJobsV1, struct_size) == 0);
static_assert(offsetof(CaliperDeviceV1, struct_size) == 0);
static_assert(CALIPER_DEV_CPU == 0 && CALIPER_DEV_CUDA == 1 && CALIPER_DEV_METAL == 2);

TEST_CASE("abi: phase-2a service ids are fixed") {
    CHECK(std::string(CALIPER_JOBS_V1) == "caliper.jobs.v1");
    CHECK(std::string(CALIPER_DEVICE_V1) == "caliper.device.v1");
}

static_assert(std::is_standard_layout_v<CaliperTensor>);
static_assert(std::is_standard_layout_v<CaliperMetricsV1>);
static_assert(offsetof(CaliperTensor, struct_size) == 0);
static_assert(offsetof(CaliperMetricsV1, struct_size) == 0);
static_assert(CALIPER_DT_F32 == 0);

TEST_CASE("abi: phase-2b service ids are fixed") {
    CHECK(std::string(CALIPER_METRICS_V1) == "caliper.metrics.v1");
}
