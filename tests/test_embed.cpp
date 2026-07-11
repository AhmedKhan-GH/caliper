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

TEST_CASE("embed/gate: unknown applet id refuses; read_pixels needs a canvas") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CHECK(caliper_core_load_applet(core, "dev.caliper.does-not-exist") == 0);

    std::vector<uint8_t> px(16, 0);
    CHECK(caliper_core_read_pixels(core, px.data(), 4) == 0);   // no canvas

    caliper_core_shutdown(core);
}
