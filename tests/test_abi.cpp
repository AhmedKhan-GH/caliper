#include <doctest/doctest.h>
#include <caliper/abi.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/tensor.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/geometry_v1.h>
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1_3.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <cstddef>
#include <cstring>
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

TEST_CASE("geometry v1 layout is frozen (new service, D24 pattern)") {
    static_assert(std::is_standard_layout_v<CaliperGeometryV1>);
    static_assert(std::is_standard_layout_v<CaliperGeomCamera>);
    static_assert(offsetof(CaliperGeometryV1, struct_size) == 0);
    // The camera crosses the ABI by pointer: two packed 4x4 float matrices.
    static_assert(offsetof(CaliperGeomCamera, view) == 0);
    static_assert(offsetof(CaliperGeomCamera, proj) == 16 * sizeof(float));
    CHECK(sizeof(CaliperGeomCamera) == 32 * sizeof(float));
    // Member order pinned: caps, create_view, release_view, draw_points —
    // contiguous fn pointers after struct_size (padded to pointer alignment).
    static_assert(offsetof(CaliperGeometryV1, caps) == sizeof(void*));
    static_assert(offsetof(CaliperGeometryV1, create_view) ==
                  offsetof(CaliperGeometryV1, caps) + sizeof(void*));
    static_assert(offsetof(CaliperGeometryV1, release_view) ==
                  offsetof(CaliperGeometryV1, create_view) + sizeof(void*));
    static_assert(offsetof(CaliperGeometryV1, draw_points) ==
                  offsetof(CaliperGeometryV1, release_view) + sizeof(void*));
    CHECK(sizeof(CaliperGeometryV1) ==
          offsetof(CaliperGeometryV1, draw_points) + sizeof(void*));
    CHECK(CALIPER_GEOM_CAP_IMPORTED_POINTS == (1u << 0));
    CHECK(std::string(CALIPER_GEOMETRY_V1) == "caliper.geometry.v1");
}

TEST_CASE("geometry v1_1 is prefix-identical to v1 and pins draw ABI") {
    static_assert(std::is_standard_layout_v<CaliperGeometryV1_1>);
    static_assert(std::is_standard_layout_v<CaliperGeomDraw>);
    static_assert(offsetof(CaliperGeometryV1_1, struct_size) ==
                  offsetof(CaliperGeometryV1, struct_size));
    static_assert(offsetof(CaliperGeometryV1_1, caps) ==
                  offsetof(CaliperGeometryV1, caps));
    static_assert(offsetof(CaliperGeometryV1_1, create_view) ==
                  offsetof(CaliperGeometryV1, create_view));
    static_assert(offsetof(CaliperGeometryV1_1, release_view) ==
                  offsetof(CaliperGeometryV1, release_view));
    static_assert(offsetof(CaliperGeometryV1_1, draw_points) ==
                  offsetof(CaliperGeometryV1, draw_points));
    static_assert(offsetof(CaliperGeometryV1_1, create_view_ex) ==
                  offsetof(CaliperGeometryV1, draw_points) + sizeof(void*));
    static_assert(offsetof(CaliperGeometryV1_1, draw_primitives) ==
                  offsetof(CaliperGeometryV1_1, create_view_ex) + sizeof(void*));
    static_assert(offsetof(CaliperGeometryV1_1, reserved0) ==
                  offsetof(CaliperGeometryV1_1, draw_primitives) + sizeof(void*));
    CHECK(sizeof(CaliperGeometryV1_1) ==
          offsetof(CaliperGeometryV1_1, reserved0) + sizeof(void*));

    CHECK(sizeof(CaliperGeomDraw) == 192);
    CHECK(offsetof(CaliperGeomDraw, pos_alloc) == 0);
    CHECK(offsetof(CaliperGeomDraw, topology) == 80);
    CHECK(offsetof(CaliperGeomDraw, model) == 120);
    CHECK(offsetof(CaliperGeomDraw, reserved) == 184);
    CHECK(CALIPER_GEOM_CAP_PRIMITIVES == (1u << 1));
    CHECK(CALIPER_GEOM_VIEW_DEPTH == (1u << 0));
    CHECK(CALIPER_GEOM_TOPO_TRIANGLE_STRIP == 4u);
    CHECK(CALIPER_GEOM_COLOR_VERTEX_RGBA == 2u);
    CHECK(CALIPER_GEOM_SHADE_LAMBERT == 1u);
    CHECK(CALIPER_GEOM_BLEND_ADDITIVE == 2u);
    CHECK(std::string(CALIPER_GEOMETRY_V1_1) == "caliper.geometry.v1_1");
}

TEST_CASE("geometry v1_2 preserves v1_1 and pins the textured draw tail") {
    static_assert(std::is_standard_layout_v<CaliperGeometryV1_2>);
    static_assert(std::is_standard_layout_v<CaliperGeomDrawV1_2>);
    CHECK(sizeof(CaliperGeomDraw) == 192);
    CHECK(sizeof(CaliperGeomDrawV1_2) == 216);
    CHECK(offsetof(CaliperGeomDrawV1_2, base) == 0);
    CHECK(offsetof(CaliperGeomDrawV1_2, uv_alloc) == 192);
    CHECK(offsetof(CaliperGeomDrawV1_2, uv_offset) == 200);
    CHECK(offsetof(CaliperGeomDrawV1_2, texture) == 208);

    CHECK(offsetof(CaliperGeometryV1_2, struct_size) ==
          offsetof(CaliperGeometryV1_1, struct_size));
    CHECK(offsetof(CaliperGeometryV1_2, draw_primitives) ==
          offsetof(CaliperGeometryV1_1, draw_primitives));
    CHECK(offsetof(CaliperGeometryV1_2, reserved0) ==
          offsetof(CaliperGeometryV1_1, reserved0));
    CHECK(sizeof(CaliperGeometryV1_2) == sizeof(CaliperGeometryV1_1));
    CHECK(CALIPER_GEOM_CAP_TEXTURED == (1u << 2));
    CHECK(CALIPER_GEOM_COLOR_TEXTURE == 3u);
    CHECK(std::string(CALIPER_GEOMETRY_V1_2) == "caliper.geometry.v1_2");
}

TEST_CASE("geometry v1_3 preserves v1_2 and pins the instance tail") {
    static_assert(std::is_standard_layout_v<CaliperGeometryV1_3>);
    static_assert(std::is_standard_layout_v<CaliperGeomDrawV1_3>);
    // The whole revision chain pinned in one place.
    CHECK(sizeof(CaliperGeomDraw) == 192);
    CHECK(sizeof(CaliperGeomDrawV1_2) == 216);
    CHECK(sizeof(CaliperGeomDrawV1_3) == 256);
    CHECK(offsetof(CaliperGeomDrawV1_3, base) == 0);
    CHECK(offsetof(CaliperGeomDrawV1_3, instance_alloc) == 216);
    CHECK(offsetof(CaliperGeomDrawV1_3, instance_offset) == 224);
    CHECK(offsetof(CaliperGeomDrawV1_3, instance_count) == 232);
    CHECK(offsetof(CaliperGeomDrawV1_3, instance_attr_alloc) == 240);
    CHECK(offsetof(CaliperGeomDrawV1_3, instance_attr_offset) == 248);

    // Service-table parity: same slots as v1.2 (mirror the v1_2 pins above).
    CHECK(offsetof(CaliperGeometryV1_3, struct_size) ==
          offsetof(CaliperGeometryV1_2, struct_size));
    CHECK(offsetof(CaliperGeometryV1_3, draw_primitives) ==
          offsetof(CaliperGeometryV1_2, draw_primitives));
    CHECK(offsetof(CaliperGeometryV1_3, reserved0) ==
          offsetof(CaliperGeometryV1_2, reserved0));
    CHECK(sizeof(CaliperGeometryV1_3) == sizeof(CaliperGeometryV1_2));

    CHECK(std::string(CALIPER_GEOMETRY_V1_3) == "caliper.geometry.v1_3");
    CHECK(CALIPER_GEOM_CAP_INSTANCED == (1u << 3));
}

namespace {
// Records what a v1_2-only host actually receives from the Geometry wrapper's
// v1.1-shaped draw path — proving Task 1's record widening (the v1.2
// draw_primitives entry enforces min stride 216; a raw 192-byte alias is
// refused). Statics because the C table carries no user data.
struct GeomWidenCapture {
    uint32_t draw_stride = 0;
    CaliperAllocId uv_alloc = 1;   // poison: proves the wrapper wrote zero
    uint64_t uv_offset = 1;
    CaliperTextureId texture = 1;
    CaliperGeomDraw base{};
    bool called = false;
};
GeomWidenCapture g_widen;

uint32_t stub12_caps(void) {
    return CALIPER_GEOM_CAP_PRIMITIVES | CALIPER_GEOM_CAP_TEXTURED;
}
bool stub12_draw_primitives(CaliperTextureId, const CaliperGeomCamera*,
                            const CaliperGeomDrawV1_2* draws, uint32_t,
                            uint32_t draw_stride, uint32_t) {
    g_widen.called = true;
    g_widen.draw_stride = draw_stride;
    if (draws) {
        g_widen.uv_alloc = draws[0].uv_alloc;
        g_widen.uv_offset = draws[0].uv_offset;
        g_widen.texture = draws[0].texture;
        g_widen.base = draws[0].base;
    }
    return true;
}
const CaliperGeometryV1_2 kStubGeom12 = {
    sizeof(CaliperGeometryV1_2),
    &stub12_caps,
    nullptr,   // create_view
    nullptr,   // release_view
    nullptr,   // draw_points
    nullptr,   // create_view_ex
    &stub12_draw_primitives,
    nullptr,   // reserved0
};
} // namespace

TEST_CASE("geometry wrapper widens v1.1 draws on a v1_2-only host") {
    g_widen = GeomWidenCapture{};
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_GEOMETRY_V1_2, &kStubGeom12);  // v1 and v1_1 stay NULL
    caliper::Host host(fx.host());
    caliper::Geometry geo(host);

    // The v1_2->v1_1 fallback must WORK, not be disabled.
    CHECK(geo.has_primitives());

    // Two fully-poisoned frozen 192-byte records through the v1.1-shaped overload.
    CaliperGeomDraw draws[2];
    std::memset(draws, 0xA5, sizeof(draws));
    CaliperGeomCamera cam{};
    CHECK(geo.draw_primitives(0, cam, (const CaliperGeomDraw*)draws, 2, 0));

    // Widened into a v1.2 record: min stride 216, zero tail, intact prefix.
    CHECK(g_widen.called);
    CHECK(g_widen.draw_stride ==
          static_cast<uint32_t>(sizeof(CaliperGeomDrawV1_2)));
    CHECK(g_widen.uv_alloc == 0);
    CHECK(g_widen.uv_offset == 0);
    CHECK(g_widen.texture == 0);
    CHECK(std::memcmp(&g_widen.base, &draws[0], sizeof(CaliperGeomDraw)) == 0);
}

namespace {
// Records what a v1_3-only host receives from the Geometry wrapper — the true
// analog of the v1_2 widening capture above, one tier up: both the v1.1-shaped
// and the v1.2-shaped overloads must arrive as zero-tailed 256-byte records.
struct GeomWiden13Capture {
    uint32_t draw_stride = 0;
    CaliperGeomDrawV1_3 rec{};
    bool called = false;
};
GeomWiden13Capture g_widen13;

uint32_t stub13_caps(void) {
    return CALIPER_GEOM_CAP_PRIMITIVES | CALIPER_GEOM_CAP_TEXTURED |
           CALIPER_GEOM_CAP_INSTANCED;
}
bool stub13_draw_primitives(CaliperTextureId, const CaliperGeomCamera*,
                            const CaliperGeomDrawV1_3* draws, uint32_t,
                            uint32_t draw_stride, uint32_t) {
    g_widen13.called = true;
    g_widen13.draw_stride = draw_stride;
    if (draws) g_widen13.rec = draws[0];
    return true;
}
const CaliperGeometryV1_3 kStubGeom13 = {
    sizeof(CaliperGeometryV1_3),
    &stub13_caps,
    nullptr,   // create_view
    nullptr,   // release_view
    nullptr,   // draw_points
    nullptr,   // create_view_ex
    &stub13_draw_primitives,
    nullptr,   // reserved0
};

// The five appended tail fields must all be zero after widening.
bool tail13_is_zero(const CaliperGeomDrawV1_3& r) {
    return r.instance_alloc == 0 && r.instance_offset == 0 &&
           r.instance_count == 0 && r.instance_attr_alloc == 0 &&
           r.instance_attr_offset == 0;
}
} // namespace

TEST_CASE("geometry wrapper widens v1.1 and v1.2 draws on a v1_3-only host") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_GEOMETRY_V1_3, &kStubGeom13);  // v1/v1_1/v1_2 stay NULL
    caliper::Host host(fx.host());
    caliper::Geometry geo(host);

    // The v1_3->v1_2->v1_1 fallback chain must WORK, not be disabled.
    CHECK(geo.has_primitives());
    CHECK(geo.has_instanced());
    CaliperGeomCamera cam{};

    SUBCASE("192-byte v1.1-shaped records widen to 256") {
        g_widen13 = GeomWiden13Capture{};
        CaliperGeomDraw draws[2];
        std::memset(draws, 0xA5, sizeof(draws));
        CHECK(geo.draw_primitives(0, cam, (const CaliperGeomDraw*)draws, 2, 0));

        CHECK(g_widen13.called);
        CHECK(g_widen13.draw_stride ==
              static_cast<uint32_t>(sizeof(CaliperGeomDrawV1_3)));
        // 192-byte prefix intact byte-for-byte; everything after it zero.
        CHECK(std::memcmp(&g_widen13.rec, &draws[0],
                          sizeof(CaliperGeomDraw)) == 0);
        CHECK(g_widen13.rec.base.uv_alloc == 0);
        CHECK(g_widen13.rec.base.uv_offset == 0);
        CHECK(g_widen13.rec.base.texture == 0);
        CHECK(tail13_is_zero(g_widen13.rec));
    }

    SUBCASE("216-byte v1.2-shaped records widen to 256") {
        g_widen13 = GeomWiden13Capture{};
        CaliperGeomDrawV1_2 draws[2];
        std::memset(draws, 0x5A, sizeof(draws));
        CHECK(geo.draw_primitives(0, cam,
                                  (const CaliperGeomDrawV1_2*)draws, 2, 0));

        CHECK(g_widen13.called);
        CHECK(g_widen13.draw_stride ==
              static_cast<uint32_t>(sizeof(CaliperGeomDrawV1_3)));
        // 216-byte prefix intact byte-for-byte; instance tail all-zero.
        CHECK(std::memcmp(&g_widen13.rec, &draws[0],
                          sizeof(CaliperGeomDrawV1_2)) == 0);
        CHECK(tail13_is_zero(g_widen13.rec));
    }
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
