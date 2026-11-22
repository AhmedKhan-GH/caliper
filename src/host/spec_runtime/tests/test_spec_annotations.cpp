// The spec runtime tests of annotations, legends and bound texts
// (design/instrument-interface.md, Annotations and Markers).
//
// The bound text helpers are checked without a GPU. The cases of the suite
// "gpu" draw into the offscreen canvas: a series panel with each kind of
// annotation, the item labels of a scene whose anchors fall within two
// pixels, and the layout shares at two window sizes. With
// CALIPER_SPEC_TEST_DUMP=<folder> each image is kept as a PNG, to look at.
#include "spec_runtime/spec_annotations.h"

#include "spec_runtime/label_layout.h"
#include "spec_runtime/spec_layout.h"
#include "live/scene_overlays.h"
#include "scene/compile.h"

#include "export_service.h"

#include "spec_test_canvas.h"
#include "spec_test_data.h"

#include <doctest/doctest.h>

#include <caliper_builder/session/spec_edit.h>

#include <cstdlib>
#include <map>
#include <string>

using namespace caliper_host;
using namespace spec_test;
using nlohmann::json;

namespace {

void dump(const Image& img, const std::string& name) {
    const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP");
    if (!dir || !*dir || img.px.empty()) return;
    export_write_png_atomic(std::string(dir) + "/" + name + ".png", img.px.data(), (uint32_t)img.w, (uint32_t)img.h);
}

caliper_builder::spec::Reference ref(caliper_builder::spec::Reference::Kind kind, const std::string& target,
                                     json fallback, std::optional<std::string> field = std::nullopt) {
    caliper_builder::spec::Reference r;
    r.kind = kind;
    r.target = target;
    r.field = std::move(field);
    r.fallback = std::move(fallback);
    return r;
}

// Counts the pixels near the accent of the house look in a rectangle.
int accent_pixels(const Image& img, int x0, int y0, int x1, int y1) {
    return img.count(near(house_spec_look().accent, 40), x0, y0, x1, y1).n;
}

}  // namespace

TEST_CASE("a bound text reads its references' defaults without a session, and a dash for a missing value") {
    caliper_builder::spec::BoundText t;
    t.format = "Step {step:,d}  Loss {loss:.3f}  Token {token:d}";
    t.fields["step"] = ref(caliper_builder::spec::Reference::Kind::state, "cursor", 12000, "step");
    t.fields["token"] = ref(caliper_builder::spec::Reference::Kind::state, "token", 7);
    t.fields["loss"] = std::string("loss");
    const SpecInstrumentHost host;   // no session: references give their defaults
    const auto values = spec_bound_text_values(t, host);
    CHECK(values.at("step") == json(12000));
    CHECK(values.at("token") == json(7));
    CHECK(values.at("loss").is_null());
    CHECK(spec_bound_text_string(t, host, {}) == "Step 12,000  Loss \xE2\x80\x93  Token 7");
    // An extra field, such as a text panel's bound value.
    caliper_builder::spec::BoundText v;
    v.format = "Value {value:.2f}";
    CHECK(spec_bound_text_string(v, host, {{"value", 0.125}}) == "Value 0.12");
    // A format that does not parse gives no text.
    caliper_builder::spec::BoundText bad;
    bad.format = "{nope";
    CHECK(spec_bound_text_string(bad, host, {}).empty());
}

TEST_SUITE("gpu") {

TEST_CASE("annotations: a series panel draws a band, a threshold, a marker and labels that do not overlap") {
    if (!canvas()) return;
    // Three labels at almost one point of the curve, the case that drew
    // labels on top of each other, beside a band, a threshold and a marker.
    const caliper_builder::spec::Spec s = parse(
        "caliper_spec = 1\n[applet]\nid = \"t\"\n[[source]]\nid = \"w\"\ncatalog = \"tensor:w\"\n"
        "[[panel]]\nid = \"p\"\nprimitive = \"series\"\ntitle = \"Loss\"\nbind = { y = \"w\" }\n"
        "[[panel.annotation]]\nid = \"window\"\nkind = \"band\"\naxis = \"x\"\nrange = [20.0, 45.0]\ntext = \"Window\"\n"
        "[[panel.annotation]]\nid = \"floor\"\nkind = \"threshold\"\naxis = \"y\"\nvalue = 30.0\ntext = \"Floor {f:.1f}\"\n"
        "fields = { f = { state = \"floor\", default = 30.0 } }\n"
        "[[panel.annotation]]\nid = \"now\"\nkind = \"marker\"\nat = [70.0, 70.0]\ntext = \"Step {s:d}\"\n"
        "fields = { s = { state = \"cursor\", field = \"step\", default = 70 } }\n"
        "[[panel.annotation]]\nid = \"a\"\nkind = \"label\"\nat = [55.0, 55.0]\ntext = \"layer 4\"\n"
        "[[panel.annotation]]\nid = \"b\"\nkind = \"label\"\nat = [55.5, 55.5]\ntext = \"layer 5\"\n"
        "[[panel.annotation]]\nid = \"c\"\nkind = \"label\"\nat = [56.0, 55.0]\ntext = \"layer 6\"\n");
    std::vector<float> y(100);
    for (int i = 0; i < 100; ++i) y[i] = (float)i;
    SpecOffscreen* c = canvas();
    SpecColorKey key;
    key.reset(s);
    SpecPanelView view;
    view.data = prepare_panel(s, s.panels[0], data({f32("y", "w", {100}, y)}, caliper_builder::spec::Primitive::series));
    view.key = &key;
    SpecInstrumentHost host;
    view.spec_panel = &s.panels[0];
    view.instruments = &host;
    SpecPanelFrame frame;
    const Image img = render(640, 400, [&](const ImRect& area) {
        frame = draw_spec_panel(view, area, c->look(), c->textures());
        draw_panel_overlays(s.panels[0], frame.content, c->look(), host);
    });
    c->with_context([&] { c->textures().release(view.texture); });
    dump(img, "annot-series");
    const int x0 = (int)frame.content.Min.x, y0 = (int)frame.content.Min.y;
    const int x1 = (int)frame.content.Max.x, y1 = (int)frame.content.Max.y;
    // The marks are in the accent: the band's edges, the threshold and the ring.
    const int accent = accent_pixels(img, x0, y0, x1, y1);
    CHECK_MESSAGE(accent > 400, accent);
    // Text shows near the three close labels.
    const int text = img.count(bright, x0, y0, x1, y1).n;
    CHECK_MESSAGE(text > 60, text);
}

TEST_CASE("annotations: the item labels of a scene whose anchors fall within two pixels do not overlap") {
    if (!canvas()) return;
    // The anchors that a scene gives for three items drawn almost on one
    // pixel, and two items apart from them.
    std::vector<caliper_scene::LabelAnchor> anchors;
    const auto add = [&](float x, float y, const char* text) {
        caliper_scene::LabelAnchor a;
        a.x = x;
        a.y = y;
        a.item = anchors.size();
        a.depth = 1.0f + 0.01f * (float)anchors.size();
        a.text = text;
        anchors.push_back(a);
    };
    add(300, 200, "blocks.4.attn");
    add(301, 201, "blocks.5.attn");
    add(302, 200, "blocks.6.attn");
    add(140, 120, "embed");
    add(520, 320, "head");
    const SpecLook look = house_spec_look();
    const Image img = render(640, 400, [&](const ImRect& area) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(area.Min, area.Max, look.deep);
        // The items, as the scene draws them under the overlay.
        for (const auto& a : anchors) dl->AddCircleFilled(ImVec2(area.Min.x + a.x, area.Min.y + a.y), 4.0f, look.line);
        caliper_live_views::SceneOverlayInput in;
        in.draw = dl;
        in.origin = area.Min;
        in.labels = &anchors;
        in.ink = look.text;
        in.shade = IM_COL32(0, 0, 0, 160);
        caliper_live_views::draw_scene_overlays(in);
    });
    dump(img, "annot-scene-labels");
    // Some text near the three close items.
    CHECK(img.count(bright, 220, 150, 420, 260).n > 50);
}

TEST_CASE("annotations: the item labels of a scene keep off the view's own texts") {
    if (!canvas()) return;
    // Labelled items crowd the places of the selection count (top right) and
    // the scale bar (bottom right), as in the app's scene window.
    std::vector<caliper_scene::LabelAnchor> anchors;
    const auto add = [&](float x, float y, const char* text) {
        caliper_scene::LabelAnchor a;
        a.x = x;
        a.y = y;
        a.item = anchors.size();
        a.depth = 1.0f + 0.01f * (float)anchors.size();
        a.text = text;
        anchors.push_back(a);
    };
    for (int i = 0; i < 6; ++i) {
        add(564.0f + 12.0f * (float)i, 32.0f + 3.0f * (float)(i % 2), ("p1" + std::to_string(i)).c_str());
        add(540.0f + 14.0f * (float)i, 396.0f, ("p2" + std::to_string(i)).c_str());
    }
    const SpecLook look = house_spec_look();
    const LabelRect count{560, 6, 632, 26}, bar{500, 352, 632, 392};
    const std::vector<LabelRect> keep = {count, bar};
    for (const bool kept : {false, true}) {
        const Image img = render(640, 400, [&](const ImRect& area) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(area.Min, area.Max, look.deep);
            // The two texts' places, in the accent colour, under the labels.
            for (const LabelRect& r : keep)
                dl->AddRect(ImVec2(area.Min.x + r.x0, area.Min.y + r.y0), ImVec2(area.Min.x + r.x1, area.Min.y + r.y1),
                            look.accent);
            std::vector<LabelRect> screen;
            for (const LabelRect& r : keep)
                screen.push_back({area.Min.x + r.x0, area.Min.y + r.y0, area.Min.x + r.x1, area.Min.y + r.y1});
            for (const auto& a : anchors) dl->AddCircleFilled(ImVec2(area.Min.x + a.x, area.Min.y + a.y), 3.0f, look.line);
            caliper_live_views::SceneOverlayInput in;
            in.draw = dl;
            in.origin = area.Min;
            in.labels = &anchors;
            in.ink = look.text;
            in.shade = IM_COL32(0, 0, 0, 160);
            if (kept) in.keep_clear = &screen;
            caliper_live_views::draw_scene_overlays(in);
        });
        dump(img, kept ? "annot-scene-keep-clear" : "annot-scene-keep-clear-without");
        // Without the rectangles, labels cover them; with them, no label text lies inside.
        const auto bright_inside = [&](const LabelRect& r) {
            return img.count(bright, (int)r.x0 + 1, (int)r.y0 + 1, (int)r.x1 - 1, (int)r.y1 - 1).n;
        };
        const int inside = bright_inside(count) + bright_inside(bar);
        if (kept)
            CHECK(inside == 0);
        else
            CHECK(inside > 0);
    }
}

TEST_CASE("annotations: the layout shares at 2560 by 1440 and at 1440 by 900") {
    if (!canvas()) return;
    const caliper_builder::spec::Spec s = parse(R"toml(caliper_spec = 1
[applet]
id = "shares"
[[source]]
id = "w"
catalog = "tensor:w"
[[panel]]
id = "world"
primitive = "heatmap"
bind = { values = "w" }
[[panel]]
id = "detail"
primitive = "heatmap"
bind = { values = "w" }
[[panel]]
id = "loss"
primitive = "series"
bind = { y = "w" }
[[control]]
id = "lr"
kind = "slider"
bind = "value"
options = { min = 0.0, max = 1.0 }
[layout]
split = "vertical"
shares = [6, 3, 1]
children = [
  { split = "horizontal", shares = [3, 2], children = [{ panel = "world", min = [48, 20] }, "detail"] },
  "loss",
  "lr",
]
)toml");
    const SpecLook look = house_spec_look();
    for (const auto& [w, h] : {std::pair<float, float>{2560, 1440}, {1440, 900}}) {
        const SpecLayout l = compute_spec_layout(*s.layout, s, {0, 0, w, h}, SpecLayoutMetrics::for_text_scale(1.0f));
        const float k = 600.0f / w;   // drawn at 600 pixels wide
        const Image img = render(640, 400, [&](const ImRect& area) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(area.Min, area.Max, look.ground);
            for (const SpecLayoutPanel& p : l.panels) {
                const ImVec2 a(area.Min.x + 20 + p.rect.x * k, area.Min.y + 20 + p.rect.y * k);
                const ImVec2 b(a.x + p.rect.w * k, a.y + p.rect.h * k);
                dl->AddRectFilled(a, b, p.control ? look.plate_hover : look.plate);
                dl->AddRect(a, b, look.rule);
                const std::string t = p.id + " " + std::to_string((int)p.rect.w) + "x" + std::to_string((int)p.rect.h);
                dl->AddText(ImVec2(a.x + 4, a.y + 2), look.text, t.c_str());
            }
        });
        dump(img, "annot-layout-" + std::to_string((int)w) + "x" + std::to_string((int)h));
        CHECK(l.panels.size() == 4);
    }
}

}  // TEST_SUITE("gpu")
