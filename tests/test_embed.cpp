// caliper_embed_tests — the embed C ABI (caliper/embed.h) battery (R4 L2a).
//
// Drives the ABI the way an embedder would: create an OFFSCREEN core, load the
// hello fixture applet, pump frames, read the composited pixels back, and shut
// down clean — plus the refusal gates (second live core, frame/event before a
// canvas, double attach) that must fail HONESTLY, never crash.
//
// One CaliperCore per process (v0), so every case pairs create with shutdown.
// The pixel case self-skips when no Metal device is present (offscreen attach
// fails) so a headless CI run stays green, mirroring the gfx suite.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <caliper/embed.h>

// White-box internals for the §7 host-axis byte-compare (last case): the embed
// core wires its renderer into the process bridge at attach, so we drive the
// SAME public tensor_bridge.v1 service and read the texture back through the
// renderer, comparing to the shared CPU reference the gfx rows use.
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include "host_services.h"
#include "tensor_bridge.h"          // caliper_host::expand_u8_to_rgba8 (shared ref)
#include "renderer/host_renderer.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

CaliperCoreDesc base_desc() {
    CaliperCoreDesc d{};
    d.struct_size = sizeof(CaliperCoreDesc);
    d.renderer = CALIPER_RENDERER_DEFAULT;
    d.applets_dir = CALIPER_TEST_APPLETS_DIR;   // build/applets: hello lives here
    return d;
}

CaliperCanvasDesc offscreen_desc(int w, int h) {
    CaliperCanvasDesc c{};
    c.struct_size = sizeof(CaliperCanvasDesc);
    c.mode = CALIPER_CANVAS_OFFSCREEN;
    c.width = w;
    c.height = h;
    c.content_scale = 1.0f;
    return c;
}

// A frame is "drawn" if some pixel is markedly brighter than the ~(13,13,20)
// clear — i.e. the applet's dark-theme window / white text / plot actually
// rasterized. The clear never exceeds ~20 in any channel.
int bright_pixels(const std::vector<uint8_t>& px) {
    int n = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        uint8_t m = px[i];
        if (px[i + 1] > m) m = px[i + 1];
        if (px[i + 2] > m) m = px[i + 2];
        if (m > 80) ++n;
    }
    return n;
}

}  // namespace

TEST_CASE("embed/gate: one core per process; create-destroy cycles twice") {
    CaliperCoreDesc d = base_desc();

    CaliperCore* a = caliper_core_create(&d);
    REQUIRE(a != nullptr);

    // A second live core is refused with an honest NULL (not a crash).
    CaliperCore* b = caliper_core_create(&d);
    CHECK(b == nullptr);

    caliper_core_shutdown(a);

    // The lock cleared on shutdown, so a fresh create succeeds — twice in one
    // process (the ImGui context + service registry cycle cleanly).
    CaliperCore* c = caliper_core_create(&d);
    REQUIRE(c != nullptr);
    caliper_core_shutdown(c);
}

TEST_CASE("embed/gate: frame + event before a canvas refuse without crashing") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // No canvas yet: frame is a no-op that records an honest error.
    caliper_core_frame(core);
    CHECK(std::string(caliper_core_last_error(core)).find("canvas") !=
          std::string::npos);

    // Event before a canvas: silently ignored, no crash.
    CaliperInputEvent ev{};
    ev.struct_size = sizeof ev;
    ev.type = CALIPER_EVENT_MOUSE_MOVE;
    ev.x = 10.0f; ev.y = 10.0f;
    caliper_core_event(core, &ev);

    caliper_core_shutdown(core);
}

TEST_CASE("embed/offscreen: load hello, pump, read non-blank pixels, clean shutdown") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    const int W = 256, H = 256;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU (offscreen attach failed) — skipping: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // Double attach is refused (one canvas per core in v0).
    CHECK(caliper_core_attach_canvas(core, nullptr, &c) == 0);

    // With a canvas attached, an unknown manifest id is an honest refusal
    // (this is where the unknown-id path is genuinely reached — before a canvas
    // the W1 gate below refuses first).
    CHECK(caliper_core_load_applet(core, "dev.caliper.does-not-exist") == 0);

    REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);

    // Pump a few frames: window position settles (FirstUseEver) and the font
    // atlas uploads on the first composite.
    for (int i = 0; i < 4; ++i) caliper_core_frame(core);

    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), W * 4) == 1);

    // The applet actually rasterized onto the canvas (not just the clear).
    CHECK(bright_pixels(px) > 50);

    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

TEST_CASE("embed/gate: load before a canvas refuses (W1); read_pixels needs a canvas") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // W1: loading an applet before a canvas is attached is an honest refusal
    // (the applet's launch + first frame touch the renderer's ImGui backend,
    // which canvas_init wires). No canvas here — even the fixture id refuses,
    // and the reason names the canvas so the embedder knows the fix.
    CHECK(caliper_core_load_applet(core, "dev.caliper.hello") == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("canvas") !=
          std::string::npos);

    std::vector<uint8_t> px(16, 0);
    CHECK(caliper_core_read_pixels(core, px.data(), 4) == 0);   // no canvas

    caliper_core_shutdown(core);
}

// ---------------------------------------------------------------------------
// §7 host-axis byte-compare. The design's ideal is "the gfx matrix produces the
// SAME bytes under `caliper` and `embed_host`". The exe's on-screen swapchain is
// not in-process readable, and the embed offscreen canvas composites ImGui draw
// data (not a raw geometry-texture readback), so a literal end-to-end compare of
// the two composites is out of this task's honest reach.
//
// What IS the rendering seam §7 protects — and what both hosts genuinely share —
// is the tensor->texture BRIDGE upload. Here we drive it through the renderer the
// EMBED CORE created and wired (services_set_renderer at attach), via the SAME
// public caliper.tensor_bridge.v1 service an applet uses, and byte-compare the
// readback against caliper_host::expand_u8_to_rgba8 — the IDENTICAL CPU reference
// the gfx rows (mat_u8_direct) assert against with their standalone renderer. If
// both harnesses reduce to the same bytes vs the same reference, the seam is
// byte-stable across hosts. DELTA from §7's literal ideal: bridge-upload layer,
// not the final windowed composite (stated in the L2b report).
TEST_CASE("embed/§7 host-axis: bridge upload under the embed core is byte-exact "
          "vs the shared CPU reference") {
    using namespace caliper_host;

    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    const int W = 4, H = 3;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU (offscreen attach failed) — skipping §7 case: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // The embed core's renderer is now bound to the process bridge. Reach both
    // through the same surfaces an applet (service) / the gfx harness (renderer)
    // would use.
    auto* bridge =
        (const CaliperTensorBridgeV1*)services_get(CALIPER_TENSOR_BRIDGE_V1);
    REQUIRE(bridge != nullptr);
    HostRenderer* r = services_renderer();
    REQUIRE(r != nullptr);

    // A deterministic (H,W,4) u8 tensor — a fixed, contiguous pattern.
    std::vector<uint8_t> src((size_t)H * W * 4);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 3);

    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = src.data();
    t.dtype = CALIPER_DT_U8;
    t.ndim = 3;
    t.shape[0] = H; t.shape[1] = W; t.shape[2] = 4;
    t.strides[0] = W * 4; t.strides[1] = 4; t.strides[2] = 1;
    t.device = CALIPER_DEV_CPU;

    CaliperTextureId id = bridge->texture_from_tensor(&t, 0);
    REQUIRE(id != 0);

    std::vector<uint8_t> got = r->debug_readback_rgba8(id, W, H);
    std::vector<uint8_t> ref((size_t)W * H * 4);
    expand_u8_to_rgba8(src.data(), W, H, 4, ref.data());

    REQUIRE(got.size() == ref.size());
    CHECK(got == ref);   // byte-exact: the embed-core-wired seam == the CPU ref

    bridge->release_texture(id);
    caliper_core_shutdown(core);
}
