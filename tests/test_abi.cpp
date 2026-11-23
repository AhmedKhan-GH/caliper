#include <doctest/doctest.h>
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/ui_v1_1.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/tensor.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/metrics_v1_1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/services/feed_v1.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <cstddef>
#include <string>
#include <type_traits>

// ABI hygiene (docs/development/compatibility-and-epochs.md): POD, struct_size-prefixed, C-safe.
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

static_assert(std::is_standard_layout_v<CaliperTensorBridgeV1>);
static_assert(offsetof(CaliperTensorBridgeV1, struct_size) == 0);
static_assert(CALIPER_CMAP_VIRIDIS == 0 && CALIPER_CMAP_MAGMA == 1 &&
              CALIPER_CMAP_RDBU == 2);

TEST_CASE("abi: phase-2c service ids are fixed") {
    CHECK(std::string(CALIPER_TENSOR_BRIDGE_V1) == "caliper.tensor_bridge.v1");
}

TEST_CASE("tensor_bridge v1_1 is an additive, prefix-compatible superset of v1 (D24)") {
    CHECK(std::string(CALIPER_TENSOR_BRIDGE_V1_1) == "caliper.tensor_bridge.v1_1");
    CHECK(CALIPER_BRIDGE_CAP_STREAM_ORDERED == (1u << 0));
    // Same table: every v1 member sits at the same offset in v1_1.
    CHECK(offsetof(CaliperTensorBridgeV1_1, struct_size) ==
          offsetof(CaliperTensorBridgeV1, struct_size));
    CHECK(offsetof(CaliperTensorBridgeV1_1, texture_from_tensor) ==
          offsetof(CaliperTensorBridgeV1, texture_from_tensor));
    CHECK(offsetof(CaliperTensorBridgeV1_1, update_texture) ==
          offsetof(CaliperTensorBridgeV1, update_texture));
    CHECK(offsetof(CaliperTensorBridgeV1_1, release_texture) ==
          offsetof(CaliperTensorBridgeV1, release_texture));
    CHECK(offsetof(CaliperTensorBridgeV1_1, texture_from_tensor_mapped) ==
          offsetof(CaliperTensorBridgeV1, texture_from_tensor_mapped));
    CHECK(offsetof(CaliperTensorBridgeV1_1, alloc_shared) ==
          offsetof(CaliperTensorBridgeV1, alloc_shared));
    CHECK(offsetof(CaliperTensorBridgeV1_1, free_shared) ==
          offsetof(CaliperTensorBridgeV1, free_shared));
    // Plus exactly one query at the end.
    CHECK(sizeof(CaliperTensorBridgeV1_1) ==
          offsetof(CaliperTensorBridgeV1_1, caps) + sizeof(void*));
}

TEST_CASE("tensor_bridge v1_2 is prefix-identical to v1_1 (additive, D24 pattern)") {
    static_assert(offsetof(CaliperTensorBridgeV1_2, struct_size) ==
                  offsetof(CaliperTensorBridgeV1_1, struct_size));
    static_assert(offsetof(CaliperTensorBridgeV1_2, texture_from_tensor) ==
                  offsetof(CaliperTensorBridgeV1_1, texture_from_tensor));
    static_assert(offsetof(CaliperTensorBridgeV1_2, update_texture) ==
                  offsetof(CaliperTensorBridgeV1_1, update_texture));
    static_assert(offsetof(CaliperTensorBridgeV1_2, release_texture) ==
                  offsetof(CaliperTensorBridgeV1_1, release_texture));
    static_assert(offsetof(CaliperTensorBridgeV1_2, texture_from_tensor_mapped) ==
                  offsetof(CaliperTensorBridgeV1_1, texture_from_tensor_mapped));
    static_assert(offsetof(CaliperTensorBridgeV1_2, alloc_shared) ==
                  offsetof(CaliperTensorBridgeV1_1, alloc_shared));
    static_assert(offsetof(CaliperTensorBridgeV1_2, free_shared) ==
                  offsetof(CaliperTensorBridgeV1_1, free_shared));
    static_assert(offsetof(CaliperTensorBridgeV1_2, caps) ==
                  offsetof(CaliperTensorBridgeV1_1, caps));
    // The three v1.2 members follow the shared prefix contiguously — a padding
    // or reordering surprise in the NEW members would slip past the prefix
    // checks alone.
    static_assert(offsetof(CaliperTensorBridgeV1_2, import_allocation) ==
                  offsetof(CaliperTensorBridgeV1_1, caps) + sizeof(void*));
    static_assert(offsetof(CaliperTensorBridgeV1_2, release_allocation) ==
                  offsetof(CaliperTensorBridgeV1_2, import_allocation) + sizeof(void*));
    static_assert(offsetof(CaliperTensorBridgeV1_2, update_texture_from_alloc) ==
                  offsetof(CaliperTensorBridgeV1_2, release_allocation) + sizeof(void*));
    CHECK(sizeof(CaliperTensorBridgeV1_2) ==
          offsetof(CaliperTensorBridgeV1_2, update_texture_from_alloc) + sizeof(void*));
    CHECK(CALIPER_BRIDGE_CAP_IMPORT_ALLOC == (1u << 1));
    CHECK(std::string(CALIPER_TENSOR_BRIDGE_V1_2) == "caliper.tensor_bridge.v1_2");
    // Additive v1.2 handle kind: an in-process id<MTLBuffer> (Apple). Value is
    // frozen — 1=win32, 2=fd, 3=mtlbuffer; renumbering breaks shipped applets.
    static_assert(CALIPER_ALLOC_HANDLE_MTLBUFFER == 3u, "frozen handle kind");
}

TEST_CASE("metrics v1_1 is an additive, prefix-compatible superset of v1 (C0b)") {
    static_assert(std::is_standard_layout_v<CaliperMetricsV1_1>);
    static_assert(std::is_standard_layout_v<CaliperMetricsV1>);

    // Frozen v1 writer prefix: struct_size + the six writers land at the exact
    // same offsets in v1_1 (a v1 consumer casting a v1_1 table reads them
    // unchanged). offsetof each, slot for slot.
    CHECK(offsetof(CaliperMetricsV1_1, struct_size) ==
          offsetof(CaliperMetricsV1, struct_size));
    CHECK(offsetof(CaliperMetricsV1_1, begin_run) ==
          offsetof(CaliperMetricsV1, begin_run));
    CHECK(offsetof(CaliperMetricsV1_1, end_run) ==
          offsetof(CaliperMetricsV1, end_run));
    CHECK(offsetof(CaliperMetricsV1_1, scalar) ==
          offsetof(CaliperMetricsV1, scalar));
    CHECK(offsetof(CaliperMetricsV1_1, histogram) ==
          offsetof(CaliperMetricsV1, histogram));
    CHECK(offsetof(CaliperMetricsV1_1, image) ==
          offsetof(CaliperMetricsV1, image));
    CHECK(offsetof(CaliperMetricsV1_1, hparams_json) ==
          offsetof(CaliperMetricsV1, hparams_json));

    // The read tail follows the frozen prefix contiguously — a padding or
    // reordering surprise in query()/last_error() would slip past a prefix-only
    // check, so pin both new slots and the total size.
    CHECK(offsetof(CaliperMetricsV1_1, query) ==
          offsetof(CaliperMetricsV1, hparams_json) + sizeof(void*));
    CHECK(offsetof(CaliperMetricsV1_1, last_error) ==
          offsetof(CaliperMetricsV1_1, query) + sizeof(void*));
    CHECK(sizeof(CaliperMetricsV1)   == 56);
    CHECK(sizeof(CaliperMetricsV1_1) == 72);
    CHECK(sizeof(CaliperMetricsV1_1) ==
          offsetof(CaliperMetricsV1_1, last_error) + sizeof(void*));

    CHECK(std::string(CALIPER_METRICS_V1_1) == "caliper.metrics.v1_1");
    // Prefix parity in the id vocabulary too: v1_1 keeps the v1 dotted stem.
    CHECK(std::string(CALIPER_METRICS_V1_1).rfind("caliper.metrics.v1", 0) == 0);
}

static_assert(std::is_standard_layout_v<CaliperArtifactsV1>);
static_assert(offsetof(CaliperArtifactsV1, struct_size) == 0);

static_assert(std::is_standard_layout_v<CaliperDataV1>);
static_assert(offsetof(CaliperDataV1, struct_size) == 0);
// The Arrow C Data Interface structs cross the data.v1 boundary; they are
// spec-frozen upstream but our vendored copy must stay POD.
static_assert(std::is_standard_layout_v<ArrowSchema>);
static_assert(std::is_standard_layout_v<ArrowArray>);
static_assert(std::is_standard_layout_v<ArrowArrayStream>);

TEST_CASE("abi: phase-2f service ids are fixed") {
    CHECK(std::string(CALIPER_ARTIFACTS_V1) == "caliper.artifacts.v1");
    CHECK(std::string(CALIPER_DATA_V1) == "caliper.data.v1");
}

TEST_CASE("feed v1 layout is frozen (new service, D24 pattern)") {
    static_assert(std::is_standard_layout_v<CaliperFeedSample>);
    static_assert(std::is_standard_layout_v<CaliperFeedChannelInfo>);
    static_assert(std::is_standard_layout_v<CaliperFeedV1>);
    static_assert(offsetof(CaliperFeedV1, struct_size) == 0);

    // Sample: 24 B, 8-aligned, seq/t_ns/value/reserved0.
    CHECK(sizeof(CaliperFeedSample) == 24);
    CHECK(offsetof(CaliperFeedSample, seq) == 0);
    CHECK(offsetof(CaliperFeedSample, t_ns) == 8);
    CHECK(offsetof(CaliperFeedSample, value) == 16);
    CHECK(offsetof(CaliperFeedSample, reserved0) == 20);

    // ChannelInfo: struct_size-prefixed, fixed char fields, 152 B total.
    CHECK(sizeof(CaliperFeedChannelInfo) == 152);
    CHECK(offsetof(CaliperFeedChannelInfo, struct_size) == 0);
    CHECK(offsetof(CaliperFeedChannelInfo, id) == 4);
    CHECK(offsetof(CaliperFeedChannelInfo, name) == 68);
    CHECK(offsetof(CaliperFeedChannelInfo, units) == 132);
    CHECK(offsetof(CaliperFeedChannelInfo, nominal_hz) == 148);

    // Vtable slot order pinned: caps, channel_count, channel_info, read,
    // reserved0 — contiguous fn pointers after struct_size (padded to pointer
    // alignment).
    CHECK(offsetof(CaliperFeedV1, caps) == sizeof(void*));
    CHECK(offsetof(CaliperFeedV1, channel_count) ==
          offsetof(CaliperFeedV1, caps) + sizeof(void*));
    CHECK(offsetof(CaliperFeedV1, channel_info) ==
          offsetof(CaliperFeedV1, channel_count) + sizeof(void*));
    CHECK(offsetof(CaliperFeedV1, read) ==
          offsetof(CaliperFeedV1, channel_info) + sizeof(void*));
    CHECK(offsetof(CaliperFeedV1, reserved0) ==
          offsetof(CaliperFeedV1, read) + sizeof(void*));
    CHECK(sizeof(CaliperFeedV1) ==
          offsetof(CaliperFeedV1, reserved0) + sizeof(void*));

    CHECK(CALIPER_FEED_CAP_LIVE == (1u << 0));
    CHECK(std::string(CALIPER_FEED_V1) == "caliper.feed.v1");
}

TEST_CASE("ui v1_1 is an additive, prefix-compatible superset of v1") {
    static_assert(std::is_standard_layout_v<CaliperUiV1_1>);
    // The frozen v1 table, slot for slot.
    CHECK(offsetof(CaliperUiV1_1, struct_size) == offsetof(CaliperUiV1, struct_size));
    CHECK(offsetof(CaliperUiV1_1, imgui_context) == offsetof(CaliperUiV1, imgui_context));
    CHECK(offsetof(CaliperUiV1_1, implot_context) == offsetof(CaliperUiV1, implot_context));
    CHECK(offsetof(CaliperUiV1_1, implot3d_context) ==
          offsetof(CaliperUiV1, implot3d_context));
    CHECK(offsetof(CaliperUiV1_1, imgui_allocators) ==
          offsetof(CaliperUiV1, imgui_allocators));
    // The one appended entry, right after the v1 table.
    CHECK(offsetof(CaliperUiV1_1, dock_hint) == sizeof(CaliperUiV1));
    CHECK(sizeof(CaliperUiV1) == 40);
    CHECK(sizeof(CaliperUiV1_1) == 48);
    CHECK(std::string(CALIPER_UI_V1) == "caliper.ui.v1");
    CHECK(std::string(CALIPER_UI_V1_1) == "caliper.ui.v1_1");
    // Role values are fixed integers.
    CHECK(CALIPER_DOCK_CENTRAL == 0);
    CHECK(CALIPER_DOCK_SIDE_TOP == 1);
    CHECK(CALIPER_DOCK_SIDE_BOTTOM == 2);
}

TEST_CASE("sugar: ui::dock_hint reaches ui.v1_1 and is silent without it") {
    caliper::testing::FixtureHost fx;
    caliper::ui::dock_hint(fx.host(), "Main view", CALIPER_DOCK_CENTRAL);
    caliper::Host host(fx.host());
    caliper::ui::dock_hint(host, "Controls", CALIPER_DOCK_SIDE_TOP);
    caliper::ui::dock_hint(&host, "Details", CALIPER_DOCK_SIDE_BOTTOM);
    REQUIRE(fx.dock_hints().size() == 3);
    CHECK(fx.dock_role("Main view") == CALIPER_DOCK_CENTRAL);
    CHECK(fx.dock_role("Controls") == CALIPER_DOCK_SIDE_TOP);
    CHECK(fx.dock_role("Details") == CALIPER_DOCK_SIDE_BOTTOM);
    CHECK(fx.dock_role("Never hinted") == -1);
    // The first hint for a title counts, as in the host.
    caliper::ui::dock_hint(fx.host(), "Main view", CALIPER_DOCK_SIDE_BOTTOM);
    CHECK(fx.dock_role("Main view") == CALIPER_DOCK_CENTRAL);
    CHECK(fx.dock_hints().size() == 4);

    // A host that serves only ui.v1: nothing happens and nothing crashes.
    CaliperHost bare{};
    bare.struct_size = sizeof bare;
    bare.get_service = [](const CaliperHost*, const char* id) -> const void* {
        static const CaliperUiV1 v1{sizeof(CaliperUiV1), nullptr, nullptr, nullptr,
                                    nullptr};
        return std::string(id) == CALIPER_UI_V1 ? &v1 : nullptr;
    };
    caliper::ui::dock_hint(&bare, "Main view", CALIPER_DOCK_SIDE_TOP);
    caliper::ui::dock_hint((const caliper::Host*)nullptr, "x", 0);
    caliper::ui::dock_hint((const CaliperHost*)nullptr, "x", 0);
    CHECK(fx.dock_hints().size() == 4);
}
