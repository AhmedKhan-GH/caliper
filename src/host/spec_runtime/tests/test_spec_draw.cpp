// Each primitive drawn into an offscreen canvas with known data and read
// back, checked by image properties that do not depend on exact pixels; the
// states; the whole runtime over a spec with tabs, previews and overlays; and
// the textures' lifetimes. Test suite "gpu": the ctest entry
// caliper_spec_runtime_gpu. Without a GPU each case says it is skipped.
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"
#include "ui/sternbild_theme.h"

#include "spec_test_canvas.h"
#include "spec_test_data.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::DType;

namespace {

const char* kTensor = "[[source]]\nid = \"w\"\ncatalog = \"tensor:w\"";
const char* kMetrics = "[[source]]\nid = \"m\"\ncatalog = \"table:run\"";

struct Drawn {
    Image img;
    SpecPanelFrame frame;
    SpecLook look;
};

// One panel over the whole canvas, as panel.render draws it.
Drawn draw_one(const spec::Spec& s, const PanelData& d, int w = 640, int h = 400) {
    Drawn out;
    SpecOffscreen* c = canvas();
    if (!c) return out;
    SpecColorKey key;
    key.reset(s);
    SpecPanelView view;
    view.data = prepare_panel(s, s.panels[0], d);
    view.key = &key;
    out.look = c->look();
    out.img = render(w, h, [&](const ImRect& area) { out.frame = draw_spec_panel(view, area, c->look(), c->textures()); });
    c->with_context([&] { c->textures().release(view.texture); });
    // CALIPER_SPEC_TEST_DUMP=<folder> keeps each image, to look at.
    if (const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP"); dir && *dir && !out.img.px.empty()) {
        static int n = 0;
        export_write_png_atomic(std::string(dir) + "/draw-" + std::to_string(n++) + "-" + s.panels[0].id + ".png",
                                out.img.px.data(), (uint32_t)w, (uint32_t)h);
    }
    return out;
}

// The content's rectangle in whole pixels.
struct Box {
    int x0, y0, x1, y1;
};
Box content(const Drawn& d) {
    return {(int)d.frame.content.Min.x, (int)d.frame.content.Min.y, (int)d.frame.content.Max.x, (int)d.frame.content.Max.y};
}

}  // namespace

TEST_SUITE("gpu") {

TEST_CASE("draw: a series draws its line in its key colour, rising to the right") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kTensor, "primitive = \"series\"\ntitle = \"Loss curve\"\nbind = { y = \"w\" }");
    std::vector<float> v(100);
    for (int i = 0; i < 100; ++i) v[i] = (float)i;
    const Drawn d = draw_one(s, data({f32("y", "w", {100}, v)}, spec::Primitive::series));
    const Box b = content(d);
    const ImU32 line = spec_palette_color(0);
    const auto all = d.img.count(near(line, 24), b.x0, b.y0, b.x1, b.y1);
    CHECK(all.n > 150);
    // Rising: the line's pixels on the left lie lower than those on the right.
    const int mid = (b.x0 + b.x1) / 2;
    const auto left = d.img.count(near(line, 24), b.x0, b.y0, mid, b.y1);
    const auto right = d.img.count(near(line, 24), mid, b.y0, b.x1, b.y1);
    REQUIRE(left.n > 20);
    REQUIRE(right.n > 20);
    CHECK(left.cy > right.cy + 50);
    // Not in the title strip, which carries the title and the info.
    CHECK(d.img.count(near(line, 24), 0, 0, d.img.w, (int)d.frame.header.Max.y).n == 0);
    CHECK(d.img.count(bright, 0, 0, d.img.w, (int)d.frame.header.Max.y).n > 20);
}

TEST_CASE("draw: a heatmap's corner cells have the colormap's end colours") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kTensor, "primitive = \"heatmap\"\nbind = { values = \"w\" }\noptions = { colormap = \"viridis\" }");
    const Drawn d = draw_one(s, data({f32("values", "w", {2, 2}, {0, 1, 2, 3})}, spec::Primitive::heatmap));
    const uint32_t* lut = spec_colormap_lut(kCmapViridis);
    // Left of the colorbar, which also holds both ends.
    const int bar = (int)(d.img.w * 0.85);
    const auto lo = d.img.count(near(lut[0], 1), 0, 0, bar, d.img.h);
    const auto hi = d.img.count(near(lut[255], 1), 0, 0, bar, d.img.h);
    REQUIRE(lo.n > 2000);
    REQUIRE(hi.n > 2000);
    CHECK(lo.cx < hi.cx);   // the first cell at the top left,
    CHECK(lo.cy < hi.cy);   // the last at the bottom right
    // The colorbar beside it: the top of the bar is the high end.
    const auto bar_hi = d.img.count(near(lut[255], 1), bar, 0, d.img.w, d.img.h);
    const auto bar_lo = d.img.count(near(lut[0], 1), bar, 0, d.img.w, d.img.h);
    REQUIRE(bar_hi.n > 0);
    REQUIRE(bar_lo.n > 0);
    CHECK(bar_hi.cy < bar_lo.cy);
}

TEST_CASE("draw: an image keeps its rows in order and its aspect") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kTensor, "primitive = \"image\"\nbind = { values = \"w\" }");
    const std::vector<uint8_t> px = {255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255};   // 2 x 2: red over blue
    const Drawn d = draw_one(s, data({tensor<uint8_t>("values", "w", DType::u8, {2, 2, 3}, px)}, spec::Primitive::image));
    const auto red = d.img.count(near(0x0000ff, 1));
    const auto blue = d.img.count(near(0xff0000, 1));
    REQUIRE(red.n > 1000);
    REQUIRE(blue.n > 1000);
    CHECK(red.cy < blue.cy);
    // Square cells in a wide panel: the image is about as tall as it is wide.
    const Image::Count both = d.img.count([](uint32_t p) { return p == 0x0000ff || p == 0xff0000; });
    CHECK(both.n == red.n + blue.n);
    CHECK(std::abs(red.cx - blue.cx) < 2.0);
}

TEST_CASE("draw: points in 2-D and 3-D, flat and coloured") {
    if (!canvas()) return;
    const ImU32 flat = SpecLook{}.flat_point;
    std::vector<float> pos2, pos3, col;
    for (int i = 0; i < 400; ++i) {
        const float a = (float)i * 0.05f;
        pos2.insert(pos2.end(), {std::cos(a) * (1 + a), std::sin(a) * (1 + a)});
        pos3.insert(pos3.end(), {std::cos(a), std::sin(a), a * 0.1f});
        col.push_back((float)i);
    }
    {
        const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\" }");
        const Drawn d = draw_one(s, data({f32("positions", "w", {400, 2}, pos2)}, spec::Primitive::points));
        CHECK(d.img.count(near(flat, 2)).n > 1000);
    }
    {
        const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\" }");
        const Drawn d = draw_one(s, data({f32("positions", "w", {400, 3}, pos3)}, spec::Primitive::points));
        CHECK(d.img.count(near(flat, 2)).n > 500);
    }
    {
        const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\", color = \"w\" }\n"
                                                "options = { colormap = \"magma\" }");
        const Drawn d = draw_one(s, data({f32("positions", "w", {400, 2}, pos2), f32("color", "w", {400}, col)},
                                         spec::Primitive::points));
        const uint32_t* lut = spec_colormap_lut(kCmapMagma);
        // The points and the colour key both hold the two ends.
        CHECK(d.img.count(near(lut[255], 3)).n > 100);
        CHECK(d.img.count(near(lut[128], 3)).n > 20);
        CHECK(d.img.count(near(flat, 2)).n == 0);
    }
    {
        // Class labels: each class in its categorical colour, the legend's
        // swatches at the right, and no colormap.
        std::vector<int64_t> labels;
        for (int i = 0; i < 400; ++i) labels.push_back(i % 4);
        const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\", color = \"w\" }\n"
                                                "options = { colormap = \"magma\" }");
        const Drawn d = draw_one(s, data({f32("positions", "w", {400, 3}, pos3),
                                          tensor<int64_t>("color", "w", DType::i64, {400}, labels)},
                                         spec::Primitive::points));
        const Box b = content(d);
        for (int k = 0; k < 4; ++k) {
            CAPTURE(k);
            CHECK(d.img.count(near(spec_class_color(k), 2), b.x0, b.y0, b.x1, b.y1).n > 40);
            // The legend's swatch, at the top right.
            CHECK(d.img.count(near(spec_class_color(k), 2), b.x1 - 80, b.y0, b.x1, b.y0 + 120).n >= 25);
        }
        CHECK(d.img.count(near(spec_class_color(4), 2), b.x0, b.y0, b.x1, b.y1).n == 0);
        CHECK(d.img.count(near(spec_colormap_lut(kCmapMagma)[200], 3)).n == 0);
    }
}

TEST_CASE("draw: a 3-D cloud fills its panel") {
    if (!canvas()) return;
    // A cube of points from corner to corner: its spread on screen is the box's.
    std::vector<float> pos;
    for (int i = 0; i < 1000; ++i)
        pos.insert(pos.end(), {(float)(i % 10), (float)((i / 10) % 10), (float)(i / 100)});
    const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\" }");
    for (const auto& size : {std::pair<int, int>{640, 400}, std::pair<int, int>{360, 640}}) {
        const Drawn d = draw_one(s, data({f32("positions", "w", {1000, 3}, pos)}, spec::Primitive::points), size.first,
                                 size.second);
        const Box b = content(d);
        int x0 = 1 << 30, x1 = -1, y0 = 1 << 30, y1 = -1;
        for (int y = b.y0; y < b.y1; ++y)
            for (int x = b.x0; x < b.x1; ++x)
                if (near(d.look.flat_point, 2)(d.img.rgb(x, y))) {
                    x0 = std::min(x0, x);
                    x1 = std::max(x1, x);
                    y0 = std::min(y0, y);
                    y1 = std::max(y1, y);
                }
        CAPTURE(size.first);
        REQUIRE(x1 > x0);
        // The cloud takes most of the panel along its tighter side.
        const double fill = std::max((double)(x1 - x0) / (b.x1 - b.x0), (double)(y1 - y0) / (b.y1 - b.y0));
        CHECK(fill > 0.6);
        CHECK(x0 >= b.x0);
        CHECK(x1 < b.x1);
    }
}

TEST_CASE("draw: a histogram's bars in the plot ink") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kTensor, "primitive = \"histogram\"\nbind = { values = \"w\" }\noptions = { bins = 8 }");
    std::vector<float> v;
    for (int i = 0; i < 64; ++i) v.push_back((float)(i % 8) * (float)(i % 3));
    const Drawn d = draw_one(s, data({f32("values", "w", {64}, v)}, spec::Primitive::histogram));
    const Box b = content(d);
    CHECK(d.img.count(near(d.look.line, 3), b.x0, b.y0, b.x1, b.y1).n > 2000);

    // On a log axis of counts, a bin of one value still shows as a bar.
    const spec::Spec lg = one_panel(kTensor, "primitive = \"histogram\"\nbind = { values = \"w\" }\n"
                                             "options = { bins = 4, range = [0.0, 4.0], y_log = true }");
    std::vector<float> w(10, 0.5f);
    w.push_back(3.5f);
    const Drawn l = draw_one(lg, data({f32("values", "w", {11}, w)}, spec::Primitive::histogram));
    const Box lb = content(l);
    const int quarter = (lb.x1 - lb.x0) / 4;
    CHECK(l.img.count(near(l.look.line, 3), lb.x1 - quarter, lb.y0, lb.x1, lb.y1).n > 200);
    CHECK(l.img.count(near(l.look.line, 3), lb.x0, lb.y0, lb.x0 + quarter + 40, lb.y1).n > 2000);
}

TEST_CASE("draw: a table shows the expected number of rows") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kMetrics, "primitive = \"table\"\nbind = { rows = \"m\" }");
    for (int n : {3, 7}) {
        json rows = json::array();
        for (int i = 0; i < n; ++i) rows.push_back(json::array({i, i * 0.5}));
        const Drawn d = draw_one(s, data({table("rows", "m", {{"step", "BIGINT"}, {"loss", "DOUBLE"}}, rows)},
                                         spec::Primitive::table));
        const Box b = content(d);
        // The lines of text under the title strip: the head, then one per row.
        CHECK(d.img.runs(bright, b.x0, b.x1, b.y0, b.y1) == n + 1);
        // The columns sit together at the left; the rest of the width is empty.
        const auto text = d.img.count(bright, b.x0, b.y0, b.x1, b.y1);
        CHECK(d.img.count(bright, b.x0 + 260, b.y0, b.x1, b.y1).n == 0);
        CHECK(text.n > 100);
    }
    // Many rows: the head and the rows that fit whole; the last row shown is
    // not cut by the panel's edge.
    json rows = json::array();
    for (int i = 0; i < 100; ++i) rows.push_back(json::array({i, i * 0.5}));
    for (int h : {400, 391}) {
        const Drawn d = draw_one(s, data({table("rows", "m", {{"step", "BIGINT"}, {"loss", "DOUBLE"}}, rows)},
                                         spec::Primitive::table),
                                 640, h);
        const Box b = content(d);
        const float row = sternbild::metric::row;
        const int fit = (int)std::floor((b.y1 - b.y0 - row) / row);
        CAPTURE(h);
        CHECK(d.img.runs(bright, b.x0, b.x1, b.y0, b.y1) == fit + 1);
    }
}

TEST_CASE("draw: a text panel shows its text and its value") {
    if (!canvas()) return;
    const spec::Spec s = one_panel(kMetrics, "primitive = \"text\"\nbind = { value = \"m.loss\" }\noptions = { text = \"Loss\" }");
    const Drawn d = draw_one(s, data({column("value", "m.loss", {0.5, 0.25})}, spec::Primitive::text));
    const Box b = content(d);
    // Two lines: the text, then the value in the large mono.
    CHECK(d.img.runs([](uint32_t p) { return (p & 0xff) > 120; }, b.x0, b.x1, b.y0, b.y1) == 2);
    CHECK(d.img.count(bright, b.x0, b.y0, b.x1, b.y1).n > 100);
}

TEST_CASE("draw: an unavailable and an empty panel show their state and no plot") {
    if (!canvas()) return;
    const spec::Spec s = parse("caliper_spec = 0\n[applet]\nid = \"t\"\n[[source]]\nid = \"f\"\ncatalog = \"feed:gpu\"\n"
                               "[[panel]]\nid = \"p\"\nprimitive = \"series\"\nbind = { y = \"f.value\" }\n");
    for (const PanelData& pd : {failed("resolve.unavailable", "f"), data({column("y", "f.value", json::array())}, spec::Primitive::series)}) {
        const Drawn d = draw_one(s, pd);
        const Box b = content(d);
        // The plot's field, in the deep ink, is not drawn.
        CHECK(d.img.count(near(d.look.deep, 0), b.x0, b.y0, b.x1, b.y1).n == 0);
        // Two short lines at the middle: the state and the source.
        const int mx = (b.x0 + b.x1) / 2;
        CHECK(d.img.runs([](uint32_t p) { return (p & 0xff) > 100; }, mx - 150, mx + 150, b.y0, b.y1) == 2);
        CHECK(d.img.count(bright, b.x0, b.y0, b.x1, b.y0 + (b.y1 - b.y0) / 3).n == 0);
    }
}

TEST_CASE("draw: the runtime lays a spec out, draws every panel, tabs, previews and overlays") {
    if (!canvas()) return;
    std::ifstream in(std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const spec::Spec s = parse(ss.str());

    // A source of known data for the three panels, counting its calls.
    struct Source : SpecPanelSource {
        int calls = 0;
        uint64_t rev = 1;
        PanelData resolve(const spec::Spec&, std::string_view id) override {
            ++calls;
            const std::string p(id);
            if (p == "curves") {
                std::vector<double> step, loss, acc;
                for (int i = 0; i < 200; ++i) {
                    step.push_back(i);
                    loss.push_back(2.0 * std::exp(-i / 50.0) + 0.1);
                    acc.push_back(0.1 + 0.8 * (1 - std::exp(-i / 40.0)));
                }
                return data({column("x", "metrics.step", step), column("y", "metrics.loss", loss, 0),
                             column("y", "metrics.accuracy", acc, 1)},
                            spec::Primitive::series, p);
            }
            if (p == "attention") {
                std::vector<float> v(16 * 16);
                for (int i = 0; i < 256; ++i) v[i] = (float)(i % 17 == 0);
                return data({f32("values", "attn", {16, 16}, v)}, spec::Primitive::heatmap, p);
            }
            json rows = json::array();
            for (int i = 0; i < 10; ++i) rows.push_back(json::array({i, 1.0 / (i + 1)}));
            return data({table("rows", "metrics", {{"step", "BIGINT"}, {"loss", "DOUBLE"}}, rows)}, spec::Primitive::table, p);
        }
        uint64_t revision() const override { return rev; }
    } source;

    SpecOffscreen* c = canvas();
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_source(&source);
    rt->set_spec(s);
    std::vector<SpecPanelFrame> frames;
    rt->set_overlay([&](const SpecPanelFrame& f) { frames.push_back(f); });
    const Image img = render(1440, 900, [&](const ImRect& area) {
        frames.clear();
        rt->draw(area, c->look());
    });
    CHECK(source.calls == 3);   // once each, not once a frame
    REQUIRE(rt->layout().panels.size() == 3);
    REQUIRE(frames.size() == 3);
    // Every panel draws something of its own in its rectangle.
    for (const auto& lp : rt->layout().panels) {
        CAPTURE(lp.id);
        const auto n = img.count(bright, (int)lp.rect.x, (int)lp.rect.y, (int)lp.rect.right(), (int)lp.rect.bottom());
        CHECK(n.n > 50);
        CHECK(rt->prepared(lp.id)->state.ready());
    }
    // The rules between the panels: the gap holds the rule colour.
    const SpecLayoutPanel* curves = rt->layout().find("curves");
    REQUIRE(curves);
    CHECK(img.rgb((int)curves->rect.right(), 200) == (c->look().rule & 0xffffff));
    // The two lines of the series in their key colours, and the table's head
    // carries the swatch of loss.
    CHECK(img.count(near(spec_palette_color(0), 30), (int)curves->rect.x, (int)curves->rect.y, (int)curves->rect.right(),
                    (int)curves->rect.bottom()).n > 100);
    CHECK(img.count(near(spec_palette_color(1), 30), (int)curves->rect.x, (int)curves->rect.y, (int)curves->rect.right(),
                    (int)curves->rect.bottom()).n > 100);

    // A preview swaps one panel's fragment and data, and is marked so.
    spec::Panel pv = *s.find_panel("rows");
    pv.title = "Preview rows";
    rt->set_preview(pv, data({}, spec::Primitive::table, "rows"));
    render(1440, 900, [&](const ImRect& area) {
        frames.clear();
        rt->draw(area, c->look());
    }, 1);
    bool marked = false;
    for (const auto& f : frames)
        if (f.id == "rows") marked = f.preview;
    CHECK(marked);
    rt->clear_preview("rows");

    // A new revision of the data resolves every panel again; a new spec only
    // the panels that changed.
    source.rev = 2;
    render(1440, 900, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    CHECK(source.calls == 6);
    spec::Spec changed = s;
    for (auto& p : changed.panels)
        if (p.id == "rows") p.title = "Rows";
    rt->set_spec(changed);
    render(1440, 900, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    CHECK(source.calls == 7);
    c->with_context([&] { rt.reset(); });
}

TEST_CASE("draw: a panel in a hidden tab is resolved when its tab is chosen") {
    if (!canvas()) return;
    const spec::Spec s = parse(R"(caliper_spec = 0
[applet]
id = "t"
[[source]]
id = "m"
catalog = "table:run"
[[panel]]
id = "a"
primitive = "text"
bind = { value = "m.loss" }
[[panel]]
id = "b"
primitive = "text"
bind = { value = "m.acc" }
[layout]
tabs = ["a", "b"]
)");
    struct Source : SpecPanelSource {
        std::vector<std::string> asked;
        PanelData resolve(const spec::Spec&, std::string_view id) override {
            asked.emplace_back(id);
            return data({column("value", std::string("m.") + (id == "a" ? "loss" : "acc"), {1.5})}, spec::Primitive::text,
                        std::string(id));
        }
    } source;
    SpecOffscreen* c = canvas();
    std::unique_ptr<SpecRuntime> rt;
    c->with_context([&] { rt = std::make_unique<SpecRuntime>(c->textures()); });
    rt->set_source(&source);
    rt->set_spec(s);
    render(640, 400, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    CHECK(source.asked == std::vector<std::string>{"a"});
    CHECK(rt->prepared("a") != nullptr);
    CHECK(rt->prepared("b") == nullptr);
    rt->tabs()[""] = 1;
    render(640, 400, [&](const ImRect& area) { rt->draw(area, c->look()); }, 1);
    CHECK(source.asked == std::vector<std::string>{"a", "b"});
    REQUIRE(rt->prepared("b") != nullptr);
    CHECK(rt->prepared("b")->text.value == "1.5");
    c->with_context([&] { rt.reset(); });
}

TEST_CASE("draw: textures are released and freed once the backend has destroyed them") {
    if (!canvas()) return;
    SpecOffscreen* c = canvas();
    const spec::Spec s = one_panel(kTensor, "primitive = \"heatmap\"\nbind = { values = \"w\" }");
    SpecPanelView view;
    view.data = prepare_panel(s, s.panels[0], data({f32("values", "w", {4, 4}, std::vector<float>(16, 1.0f))}, spec::Primitive::heatmap));
    const size_t before = c->textures().live();
    render(320, 200, [&](const ImRect& area) { draw_spec_panel(view, area, c->look(), c->textures()); }, 1);
    CHECK(c->textures().live() == before + 1);
    REQUIRE(view.texture);
    CHECK(view.texture->Status == ImTextureStatus_OK);
    c->with_context([&] { c->textures().release(view.texture); });
    CHECK(c->textures().live() == before);
    // Two frames: the backend destroys it, and collect() frees it.
    render(320, 200, [&](const ImRect&) {}, 2);
}

}  // TEST_SUITE("gpu")
