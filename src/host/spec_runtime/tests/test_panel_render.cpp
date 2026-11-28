// panel.render: a fake SessionHost that renders with OffscreenPanelRenderer,
// as the app's host will, bound to a session over a real workspace; the
// command returns a PNG of the panel at the size asked. The encoding helpers
// run headless.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include "spec_runtime/spec_offscreen.h"

#include "spec_test_canvas.h"
#include "spec_test_workspace.h"

#include <doctest/doctest.h>

using namespace caliper_host;
using namespace spec_test;
using nlohmann::json;

namespace {

std::string base64_decode(const std::string& in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    uint32_t buf = 0;
    int bits = 0;
    for (char c : in) {
        const int v = val(c);
        if (v < 0) continue;
        buf = buf << 6 | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((buf >> bits) & 0xff);
        }
    }
    return out;
}

// The app's host, reduced to panel.render: it draws with the runtime's
// renderer and records what the session handed it.
class FakeHost : public caliper_builder::SessionHost {
public:
    std::vector<std::string> panels;
    std::vector<size_t> slots;
    caliper_builder::CommandOutput app_info() override { return fail(); }
    caliper_builder::CommandOutput app_instruments() override { return fail(); }
    caliper_builder::CommandOutput app_launch(const std::string&) override { return fail(); }
    caliper_builder::CommandOutput capture_frame() override { return fail(); }
    explicit FakeHost(SpecOffscreen* canvas) : renderer(canvas) {}
    caliper_builder::CommandOutput render_panel(const caliper_builder::spec::Spec& spec,
                                                const caliper_builder::PanelData& data, int width, int height) override {
        panels.push_back(data.panel);
        slots.push_back(data.slots.size());
        return panel_render_output(renderer.render_panel(spec, data, width, height));
    }

private:
    static caliper_builder::CommandOutput fail() {
        return caliper_builder::CommandOutput::fail(caliper_builder::Error("command.not_bound", "not in this test"));
    }
    OffscreenPanelRenderer renderer;
};

}  // namespace

TEST_CASE("panel.render: base64 and the command's result form") {
    CHECK(spec_base64("") == "");
    CHECK(spec_base64("f") == "Zg==");
    CHECK(spec_base64("fo") == "Zm8=");
    CHECK(spec_base64("foo") == "Zm9v");
    CHECK(spec_base64("foobar") == "Zm9vYmFy");
    CHECK(base64_decode(spec_base64(std::string("\x00\xff\x10", 3))) == std::string("\x00\xff\x10", 3));
    PanelPng png;
    png.png = "abc";
    png.width = 64;
    png.height = 65;
    const auto out = panel_render_output(png);
    CHECK(out.error.ok());
    CHECK(out.result == json{{"png_base64", "YWJj"}, {"width", 64}, {"height", 65}});
    png.error = "The panel could not be drawn.";
    const auto bad = panel_render_output(png);
    CHECK(bad.error.code == "panel.render_failed");
    CHECK(bad.error.message == "The panel could not be drawn.");
}

TEST_SUITE("gpu") {

TEST_CASE("panel.render: the session's panel drawn by the host's renderer, as a PNG of the size asked") {
    if (!canvas()) return;
    const std::string root = fresh_dir("panel-render");
    auto s = dashboard_workspace(root + "/workspace", root);
    auto host = std::make_shared<FakeHost>(canvas());
    s->set_host(host);

    const json r = ok(*s, "panel.render", {{"id", "attention"}, {"width", 320}, {"height", 200}});
    REQUIRE(host->panels == std::vector<std::string>{"attention"});
    CHECK(host->slots[0] == 1);
    CHECK(r["width"] == 320);
    CHECK(r["height"] == 200);
    const std::string png = base64_decode(r["png_base64"].get<std::string>());
    int w = 0, h = 0, ch = 0;
    unsigned char* px = stbi_load_from_memory((const unsigned char*)png.data(), (int)png.size(), &w, &h, &ch, 4);
    REQUIRE(px);
    CHECK(w == 320);
    CHECK(h == 200);
    Image img;
    img.w = w;
    img.h = h;
    img.px.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    // The mean over heads of the first layer, minmax-normalised: a diagonal
    // at the top of magma on a ground at its bottom.
    const uint32_t* lut = spec_colormap_lut(kCmapMagma);
    CHECK(img.count(near(lut[255], 1)).n > 100);
    CHECK(img.count(near(lut[0], 1)).n > 1000);

    // Another panel, another size: the canvas follows.
    const json c = ok(*s, "panel.render", {{"id", "curves"}, {"width", 640}, {"height", 360}});
    CHECK(c["width"] == 640);
    const std::string png2 = base64_decode(c["png_base64"].get<std::string>());
    px = stbi_load_from_memory((const unsigned char*)png2.data(), (int)png2.size(), &w, &h, &ch, 4);
    REQUIRE(px);
    CHECK(w == 640);
    CHECK(h == 360);
    stbi_image_free(px);

    // A panel the spec does not have never reaches the host.
    caliper_builder::CommandInvocation inv;
    inv.name = "panel.render";
    inv.args = {{"id", "nowhere"}};
    const auto miss = s->invoke(std::move(inv));
    CHECK_FALSE(miss.ok);
    CHECK(miss.error.code == "panel.not_found");
    CHECK(host->panels.size() == 2);
    s->set_host(nullptr);
    s->close();
}

}  // TEST_SUITE("gpu")
