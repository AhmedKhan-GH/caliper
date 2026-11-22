// A pixels output of a GPU program (design/gpu-programs.md, "Pixels Output"):
// the live view `program.p1` of kind image, RGBA8 pixels packed as
// r | g<<8 | b<<16 | a<<24, reaches an image panel whose source has the
// catalog id `live:program.p1`. The first case is headless and feeds the live
// catalog the frame a core would give it; the second, in the suite "gpu",
// feeds a real core through caliper_core_live_apply and draws the panel.
#include "spec_runtime/spec_prepare.h"
#include "spec_runtime/spec_runtime.h"

#include "catalog_providers/catalog_providers.h"

#include <caliper/embed.h>
#include <caliper/live.h>
#include <caliper/live_views.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>

#include "spec_test_canvas.h"
#include "spec_test_data.h"
#include "spec_test_workspace.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace spec_test;

namespace {

constexpr int kW = 8, kH = 4;

uint32_t pack4x8unorm(uint32_t r, uint32_t g, uint32_t b, uint32_t a) { return r | g << 8 | b << 16 | a << 24; }

// The shader's output: H x W u32 words, as the program's storage buffer holds them.
std::vector<uint32_t> words(const std::function<uint32_t(int, int)>& at) {
    std::vector<uint32_t> w((size_t)kW * kH);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) w[(size_t)y * kW + x] = at(x, y);
    return w;
}

std::vector<uint8_t> bytes_of(const std::vector<uint32_t>& w) {
    std::vector<uint8_t> b(w.size() * 4);
    std::memcpy(b.data(), w.data(), b.size());
    return b;
}

// A session whose spec has one image panel `p` over the source `p1`, bound to
// `live:program.p1`, and the catalog providers (live only).
struct Setup {
    std::unique_ptr<caliper_builder::Session> session;
    caliper_host::CatalogProviders providers;
    caliper_builder::spec::Spec spec;

    Setup(const std::string& dir_name) {
        auto opened = caliper_builder::Session::open(fresh_dir(dir_name), {});
        REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
        session = std::move(opened.session);
        providers = register_catalog_providers(*session, nullptr, nullptr);
        REQUIRE_MESSAGE(providers.error.ok(), providers.error.message);
        REQUIRE(providers.live);
    }
    json run(const std::string& name, json args) {
        caliper_builder::CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = caliper_builder::CommandOrigin::ui;
        caliper_builder::CommandResult r = session->invoke(std::move(inv));
        INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
        REQUIRE(r.ok);
        return r.result;
    }
    void build_spec() {
        run("spec.new", {{"id", "pixels"}});
        run("source.add", {{"id", "p1"}, {"catalog", "live:program.p1"}});
        run("panel.create", {{"id", "p"}, {"primitive", "image"}, {"bind", {{"values", "p1"}}}});
        spec = session->open_spec()->spec;
    }
    ~Setup() {
        if (session) unregister_catalog_providers(*session);
        providers = {};
    }
};

}  // namespace

TEST_CASE("pixels: a live image view program.p1 reaches an image panel as a [4, 8, 4] u8 tensor, byte for byte") {
    const auto w = words([](int x, int y) { return pack4x8unorm((uint32_t)(x * 30), (uint32_t)(y * 60), (uint32_t)((x + y) * 10), 255u); });
    const std::vector<uint8_t> px = bytes_of(w);

    Setup s("pixels-headless");
    LiveCatalog::ViewFrame f;
    f.name = "program.p1";
    f.kind = CALIPER_LIVE_KIND_IMAGE;
    f.updates = 1;
    f.has_payload = true;
    f.dtype = caliper_builder::DType::u8;
    f.shape = {kH, kW, 4};
    f.data = px.data();
    f.payload_bytes = px.size();
    s.providers.live->apply(true, "pixels-test", {f});

    // The catalog lists it and describes it as an image view.
    const auto described = s.providers.live->describe("live:program.p1");
    REQUIRE(described.entry);
    CHECK(described.entry->origin.view_type == "image");
    CHECK(described.entry->shape == std::vector<int64_t>{kH, kW, 4});

    s.build_spec();
    const PanelData d = s.session->resolve("p");
    REQUIRE_MESSAGE(d.error.ok(), d.error.code << ": " << d.error.message);
    REQUIRE(d.slots.size() == 1);
    CHECK(d.slots[0].slot == "values");
    const auto& t = d.slots[0].tensor;
    REQUIRE(t.valid());
    CHECK(t.dtype() == caliper_builder::DType::u8);
    CHECK(t.shape() == std::vector<int64_t>{kH, kW, 4});
    REQUIRE(t.numel() == px.size());
    size_t wrong = 0;
    for (size_t i = 0; i < px.size(); ++i) wrong += (uint8_t)t.element(i) != px[i] ? 1 : 0;
    CHECK(wrong == 0);

    // Prepared for the image primitive: RGBA, no state in place of the data.
    const PreparedPanel p = prepare_panel(s.spec, s.spec.panels[0], d);
    CHECK(p.state.ready());
    CHECK(p.grid.channels == 4);
    CHECK(p.grid.rows == kH);
    CHECK(p.grid.cols == kW);
    REQUIRE(p.grid.rgba.size() == px.size());
    // u8 range 0..255 maps back to the same bytes.
    size_t off = 0;
    for (size_t i = 0; i < px.size(); ++i) off += p.grid.rgba[i] != px[i] ? 1 : 0;
    CHECK(off == 0);
}

TEST_SUITE("gpu") {

TEST_CASE("pixels: a core's image view program.p1, drawn by an image panel, shows its colours") {
    if (!canvas()) return;
    CaliperCoreDesc cd{};
    cd.struct_size = sizeof cd;
    cd.renderer = CALIPER_RENDERER_DEFAULT;
    CaliperCore* core = caliper_core_create(&cd);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc cv{};
    cv.struct_size = sizeof cv;
    cv.mode = CALIPER_CANVAS_OFFSCREEN;
    cv.width = 64;
    cv.height = 64;
    cv.content_scale = 1.0f;
    if (caliper_core_attach_canvas(core, nullptr, &cv) != 1) {
        MESSAGE("skipped: no offscreen canvas for the embed core");
        caliper_core_shutdown(core);
        return;
    }
    // Left half red, right half blue.
    const auto w = words([](int x, int) { return x < kW / 2 ? pack4x8unorm(255, 0, 0, 255) : pack4x8unorm(0, 0, 255, 255); });
    const std::vector<uint8_t> px = bytes_of(w);
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = const_cast<uint8_t*>(px.data());
    t.dtype = CALIPER_DT_U8;
    t.ndim = 3;
    t.shape[0] = kH;
    t.shape[1] = kW;
    t.shape[2] = 4;
    t.strides[0] = (int64_t)kW * 4;
    t.strides[1] = 4;
    t.strides[2] = 1;
    t.device = CALIPER_DEV_CPU;
    CaliperLiveSnapshot snap{};
    snap.struct_size = sizeof snap;
    snap.name = "program.p1";
    snap.kind = CALIPER_LIVE_KIND_IMAGE;
    snap.data = &t;
    snap.seq = 1;
    const int applied = caliper_core_live_apply(core, &snap);
    REQUIRE_MESSAGE(applied == 1, caliper_core_last_error(core));

    {
        Setup s("pixels-gpu");
        s.providers.live->update(core, "pixels-test");
        s.build_spec();
        const PanelData d = s.session->resolve("p");
        REQUIRE_MESSAGE(d.error.ok(), d.error.code << ": " << d.error.message);
        REQUIRE(d.slots.size() == 1);
        CHECK(d.slots[0].tensor.shape() == std::vector<int64_t>{kH, kW, 4});

        SpecOffscreen* c = canvas();
        SpecColorKey key;
        key.reset(s.spec);
        SpecPanelView view;
        view.data = prepare_panel(s.spec, s.spec.panels[0], d);
        view.key = &key;
        SpecPanelFrame frame;
        const Image img = render(640, 400, [&](const ImRect& area) { frame = draw_spec_panel(view, area, c->look(), c->textures()); });
        c->with_context([&] { c->textures().release(view.texture); });
        REQUIRE(!img.px.empty());

        const auto red = img.count(near(0x0000ff, 1));
        const auto blue = img.count(near(0xff0000, 1));
        REQUIRE(red.n > 1000);
        REQUIRE(blue.n > 1000);
        // The centres of the two halves of the image: red left, blue right,
        // and each centre holds its colour.
        const int mid = (int)((frame.content.Min.x + frame.content.Max.x) / 2);
        CHECK(red.cx < mid);
        CHECK(blue.cx > mid);
        CHECK(near(0x0000ff, 1)(img.rgb((int)red.cx, (int)red.cy)));
        CHECK(near(0xff0000, 1)(img.rgb((int)blue.cx, (int)blue.cy)));
        CHECK(std::abs(red.cy - blue.cy) < 2.0);
        // The two halves are equal in size: 4 of 8 columns each.
        CHECK(std::abs((double)red.n - (double)blue.n) < 0.05 * (double)red.n);
    }
    caliper_core_shutdown(core);
}

}  // TEST_SUITE
