// The layout of a spec's panels as plain rectangles (spec_layout.h), checked
// without a window: the fixtures at two window sizes, the shares of a split,
// the minimums at two text scales, tabs and the default layout.
#include "spec_runtime/spec_layout.h"

#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <fstream>
#include <numeric>
#include <sstream>

using namespace caliper_host;
namespace spec = caliper_builder::spec;

namespace {

spec::Spec fixture(const std::string& name) {
    std::ifstream in(std::string(CALIPER_SPEC_FIXTURES) + "/" + name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = spec::parse_toml(ss.str());
    REQUIRE_MESSAGE(parsed.ok(), name);
    return parsed.spec;
}

SpecLayout lay(const spec::Spec& s, float w, float h, float scale = 1.0f, const SpecTabChoice& tabs = {}) {
    const auto root = caliper_builder::effective_layout(s);
    REQUIRE(root);
    return compute_spec_layout(*root, s, {0, 0, w, h}, SpecLayoutMetrics::for_text_scale(scale), tabs);
}

bool same(const SpecRect& r, float x, float y, float w, float h) {
    return r.x == x && r.y == y && r.w == w && r.h == h;
}

// Every panel inside the area, on whole pixels, and no two overlapping.
void check_tiling(const SpecLayout& l, float w, float h) {
    for (const auto& p : l.panels) {
        CAPTURE(p.id);
        CHECK(p.rect.x >= 0);
        CHECK(p.rect.y >= 0);
        CHECK(p.rect.right() <= w);
        CHECK(p.rect.bottom() <= h);
        CHECK(p.rect.x == std::round(p.rect.x));
        CHECK(p.rect.w == std::round(p.rect.w));
        CHECK(p.rect.h == std::round(p.rect.h));
    }
    for (size_t i = 0; i < l.panels.size(); ++i)
        for (size_t j = i + 1; j < l.panels.size(); ++j) {
            const SpecRect& a = l.panels[i].rect;
            const SpecRect& b = l.panels[j].rect;
            const bool apart = a.right() <= b.x || b.right() <= a.x || a.bottom() <= b.y || b.bottom() <= a.y;
            CHECK_MESSAGE(apart, l.panels[i].id << " overlaps " << l.panels[j].id);
        }
}

}  // namespace

TEST_CASE("layout: the dashboard at 2560x1440 and 1440x900") {
    const spec::Spec s = fixture("dashboard.spec.toml");
    {
        const SpecLayout l = lay(s, 2560, 1440);
        REQUIRE(l.panels.size() == 3);
        CHECK(l.tabs.empty());
        // A vertical split of 0.65 over 1439 px (one is the rule), then a
        // horizontal split in halves over 2559 px.
        CHECK(same(l.find("curves")->rect, 0, 0, 1280, 935));
        CHECK(same(l.find("attention")->rect, 1281, 0, 1279, 935));
        CHECK(same(l.find("rows")->rect, 0, 936, 2560, 504));
        CHECK(l.find("curves")->path == "0.0");
        CHECK(l.find("rows")->path == "1");
        check_tiling(l, 2560, 1440);
    }
    {
        const SpecLayout l = lay(s, 1440, 900);
        CHECK(same(l.find("curves")->rect, 0, 0, 720, 584));
        CHECK(same(l.find("attention")->rect, 721, 0, 719, 584));
        CHECK(same(l.find("rows")->rect, 0, 585, 1440, 315));
        check_tiling(l, 1440, 900);
    }
    // An area that does not start at the origin moves every rectangle with it.
    {
        const auto root = caliper_builder::effective_layout(s);
        const SpecLayout l = compute_spec_layout(*root, s, {10, 50, 1440, 900}, SpecLayoutMetrics::for_text_scale(1));
        CHECK(same(l.find("curves")->rect, 10, 50, 720, 584));
        CHECK(same(l.find("rows")->rect, 10, 635, 1440, 315));
    }
}

TEST_CASE("layout: minimums hold while the space allows and grow with the text scale") {
    const spec::Spec s = fixture("dashboard.spec.toml");
    const SpecLayoutMetrics m1 = SpecLayoutMetrics::for_text_scale(1.0f);
    const SpecLayoutMetrics m2 = SpecLayoutMetrics::for_text_scale(2.0f);
    CHECK(m1.min_panel_w == 180);
    CHECK(m1.min_panel_h == 120);
    CHECK(m2.min_panel_w == 360);
    CHECK(m2.min_panel_h == 240);
    CHECK(m2.tab_bar_h > m1.tab_bar_h);
    CHECK(m1.gap == 1);
    CHECK(m2.gap == 1);

    // At 400x300 the ratio 0.65 would leave the table 104 px; it keeps its 120.
    {
        const SpecLayout l = lay(s, 400, 300);
        CHECK(l.find("rows")->rect.h == 120);
        CHECK(l.find("curves")->rect.h == 179);
        check_tiling(l, 400, 300);
    }
    // At twice the text scale the minimum is 240, in an 800x600 area.
    {
        const SpecLayout l = lay(s, 800, 600, 2.0f);
        CHECK(l.find("rows")->rect.h == 240);
        CHECK(l.find("curves")->rect.h == 359);
        CHECK(l.find("curves")->rect.w >= 360);
        CHECK(l.find("attention")->rect.w >= 360);
        check_tiling(l, 800, 600);
    }
    // At scale 1 the same area keeps the ratio.
    {
        const SpecLayout l = lay(s, 800, 600, 1.0f);
        CHECK(l.find("curves")->rect.h == std::round(0.65f * 599));
    }
    // Below every minimum the shares follow the minimums, and nothing leaves
    // the area.
    {
        const SpecLayout l = lay(s, 200, 100);
        check_tiling(l, 200, 100);
        CHECK(l.find("curves")->rect.h > 0);
        CHECK(l.find("rows")->rect.h > 0);
        CHECK(l.find("curves")->rect.h + 1 + l.find("rows")->rect.h == 100);
    }
}

TEST_CASE("layout: the shares of a split") {
    auto sum = [](const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.0f); };
    CHECK(spec_layout_shares(100, {10, 10}, 0.25) == std::vector<float>{25, 75});
    CHECK(spec_layout_shares(100, {10, 80}, 0.5) == std::vector<float>{20, 80});
    CHECK(spec_layout_shares(100, {60, 10}, 0.1) == std::vector<float>{60, 40});
    // Three children share equally, a child below its minimum is raised to it.
    CHECK(spec_layout_shares(300, {10, 10, 10}, 0.5) == std::vector<float>{100, 100, 100});
    CHECK(spec_layout_shares(300, {150, 10, 10}, 0.5) == std::vector<float>{150, 75, 75});
    // Whole pixels that add up to the length.
    const auto odd = spec_layout_shares(1438, {1, 1, 1}, 0.5);
    CHECK(sum(odd) == 1438);
    for (float v : odd) CHECK(v == std::round(v));
    // No room: in proportion to the minimums.
    CHECK(spec_layout_shares(100, {100, 300}, 0.5) == std::vector<float>{25, 75});
    CHECK(spec_layout_shares(0, {10, 10}, 0.5) == std::vector<float>{0, 0});
}

TEST_CASE("layout: tabs show one child under a strip of labels") {
    const spec::Spec s = fixture("every-primitive.spec.toml");
    const float tab = SpecLayoutMetrics::for_text_scale(1.0f).tab_bar_h;
    {
        const SpecLayout l = lay(s, 1440, 900);
        REQUIRE(l.tabs.size() == 1);
        CHECK(l.tabs[0].path == "");
        CHECK(l.tabs[0].active == 0);
        CHECK(l.tabs[0].labels == std::vector<std::string>{"Tensors", "Tables", "Text"});
        CHECK(same(l.tabs[0].bar, 0, 0, 1440, tab));
        // The first tab: a horizontal split of three equal shares.
        REQUIRE(l.panels.size() == 3);
        CHECK(same(l.find("s")->rect, 0, tab, 479, 900 - tab));
        CHECK(same(l.find("h")->rect, 480, tab, 480, 900 - tab));
        CHECK(same(l.find("p3")->rect, 961, tab, 479, 900 - tab));
        CHECK(l.find("img") == nullptr);
        CHECK_FALSE(l.find("s")->tab);
        check_tiling(l, 1440, 900);
    }
    {
        // The second tab holds a nested tabs node; its first tab shows.
        const SpecLayout l = lay(s, 2560, 1440, 1.0f, {{"", 1}});
        REQUIRE(l.tabs.size() == 2);
        CHECK(l.tabs[1].path == "1.1");
        CHECK(l.tabs[1].labels == std::vector<std::string>{"sample_png", "run42_metrics", "gpu"});
        REQUIRE(l.panels.size() == 2);
        const SpecLayoutPanel* p2 = l.find("p2");
        const SpecLayoutPanel* img = l.find("img");
        REQUIRE(p2);
        REQUIRE(img);
        CHECK(p2->rect.y == tab);
        CHECK(p2->rect.h == std::round(0.3f * (1440 - tab - 1)));
        CHECK(l.tabs[1].bar.y == p2->rect.bottom() + 1);
        CHECK(img->rect.y == l.tabs[1].bar.bottom());
        CHECK(img->rect.bottom() == 1440);
        CHECK(img->tab);
        CHECK_FALSE(p2->tab);
        check_tiling(l, 2560, 1440);
    }
    {
        const SpecLayout l = lay(s, 1440, 900, 1.0f, {{"", 1}, {"1.1", 2}});
        CHECK(l.find("gpu") != nullptr);
        CHECK(l.find("img") == nullptr);
        CHECK(l.tabs[1].active == 2);
    }
    {
        // A choice out of range shows the first tab.
        const SpecLayout l = lay(s, 1440, 900, 1.0f, {{"", 9}});
        CHECK(l.tabs[0].active == 0);
        CHECK(l.find("s") != nullptr);
    }
    {
        const SpecLayout l = lay(s, 1440, 900, 1.0f, {{"", 2}});
        CHECK(l.find("note") != nullptr);
        CHECK(l.tabs.size() == 2);
        CHECK(l.tabs[1].labels == std::vector<std::string>{"Text", "Last value"});
    }
}

TEST_CASE("layout: a spec without a layout stacks its panels") {
    spec::Spec s = fixture("dashboard.spec.toml");
    s.layout.reset();
    const SpecLayout l = lay(s, 1000, 902);
    REQUIRE(l.panels.size() == 3);
    CHECK(same(l.panels[0].rect, 0, 0, 1000, 300));
    CHECK(same(l.panels[1].rect, 0, 301, 1000, 300));
    CHECK(same(l.panels[2].rect, 0, 602, 1000, 300));
    CHECK(l.panels[0].id == "curves");
    // One panel fills the area.
    s.panels.resize(1);
    const SpecLayout one = lay(s, 640, 480);
    REQUIRE(one.panels.size() == 1);
    CHECK(same(one.panels[0].rect, 0, 0, 640, 480));
}

TEST_CASE("layout: a panel without a title goes by what it draws, never its id") {
    const spec::Spec d = fixture("dashboard.spec.toml");
    CHECK(spec_panel_title(d, *d.find_panel("curves")) == "Loss and accuracy");
    CHECK(spec_panel_title(d, *d.find_panel("rows")) == "run42_metrics");
    const spec::Spec e = fixture("every-primitive.spec.toml");
    CHECK(spec_panel_title(e, *e.find_panel("h")) == "acts");
    CHECK(spec_panel_title(e, *e.find_panel("p3")) == "embedding");
    CHECK(spec_panel_title(e, *e.find_panel("p2")) == "value by step");
    CHECK(spec_panel_title(e, *e.find_panel("img")) == "sample_png");
    CHECK(spec_panel_title(e, *e.find_panel("t")) == "run42_metrics");   // the table its query reads
    CHECK(spec_panel_title(e, *e.find_panel("gpu")) == "gpu");
    CHECK(spec_panel_title(e, *e.find_panel("note")) == "Text");
    CHECK(spec_binding_name(e, "probs") == "probs");                    // a derived source by its id
    CHECK(spec_binding_name(e, "metrics.value") == "value");
}

TEST_CASE("layout: the splits and their dividers, for the builder's drags") {
    const spec::Spec s = fixture("dashboard.spec.toml");
    const SpecLayout l = lay(s, 2560, 1440);
    REQUIRE(l.splits.size() == 2);
    // The root: a vertical split of 0.65, whose divider is the rule between
    // the upper row and the table.
    const SpecLayoutSplit& root = l.splits[0];
    CHECK(root.path == "");
    CHECK_FALSE(root.horizontal);
    CHECK(root.children == 2);
    CHECK(root.ratio == doctest::Approx(0.65));
    CHECK(root.length == 1439.0f);
    CHECK(same(root.rect, 0, 0, 2560, 1440));
    CHECK(same(root.divider, 0, 935, 2560, 1));
    // The upper row: side by side in halves, its divider between the curves
    // and the attention.
    const SpecLayoutSplit& row = l.splits[1];
    CHECK(row.path == "0");
    CHECK(row.horizontal);
    CHECK(row.ratio == doctest::Approx(0.5));
    CHECK(same(row.divider, 1280, 0, 1, 935));
    CHECK(row.divider.x == l.find("curves")->rect.right());
    CHECK(row.divider.right() == l.find("attention")->rect.x);
}

// ---- spec version 1: control leaves -------------------------------------------

namespace {

spec::Spec instruments() {
    spec::Spec s = fixture("instruments.spec.toml");
    REQUIRE(s.controls.size() == 2);
    return s;
}

}  // namespace

TEST_CASE("layout: a row of controls takes one control row under the panels") {
    const spec::Spec s = instruments();
    SpecLayoutMetrics m = SpecLayoutMetrics::for_text_scale(1.0f);
    const SpecLayout l = lay(s, 1200, 800);
    REQUIRE(l.panels.size() == 3);
    const SpecLayoutPanel* world = l.find("world");
    const SpecLayoutPanel* lr = l.find("lr");
    const SpecLayoutPanel* pause = l.find("pause");
    REQUIRE(world);
    REQUIRE(lr);
    REQUIRE(pause);
    CHECK_FALSE(world->control);
    CHECK(lr->control);
    CHECK(pause->control);
    CHECK(lr->rect.h == m.control_row_h);
    CHECK(pause->rect.h == m.control_row_h);
    CHECK(lr->rect.y == pause->rect.y);
    CHECK(lr->rect.w + pause->rect.w + m.gap == 1200);
    // The scene panel takes the rest of the height.
    CHECK(world->rect.h + m.gap + lr->rect.h == 800);
    CHECK(lr->rect.bottom() == 800);
    check_tiling(l, 1200, 800);

    // A control height of the runtime's own replaces the default row.
    m.control_height = [](const spec::Control& c) { return c.kind == spec::ControlKind::slider ? 50.0f : 40.0f; };
    const auto root = caliper_builder::effective_layout(s);
    const SpecLayout l2 = compute_spec_layout(*root, s, {0, 0, 1200, 800}, m);
    CHECK(l2.find("lr")->rect.h == 50);
    CHECK(l2.find("world")->rect.h == 800 - 50 - m.gap);
    CHECK(spec_layout_fixed_height(*root, s, m) == std::nullopt);
    CHECK(spec_layout_fixed_height(root->children[1], s, m) == std::optional<float>(50.0f));
}

TEST_CASE("layout: the default layout of a spec with controls and its minimums") {
    spec::Spec s = instruments();
    s.layout.reset();
    const SpecLayoutMetrics m = SpecLayoutMetrics::for_text_scale(1.0f);
    const SpecLayout l = lay(s, 900, 600);
    REQUIRE(l.panels.size() == 3);
    CHECK(l.find("lr")->rect.h == m.control_row_h);
    CHECK(l.find("world")->rect.y == 0);
    const auto root = caliper_builder::effective_layout(s);
    const SpecRect min = spec_layout_min_size(*root, m, &s);
    CHECK(min.h == m.min_panel_h + m.gap + m.control_row_h);
    CHECK(min.w == std::max(m.min_panel_w, 2 * m.min_control_w + m.gap));
    // Without the spec, every leaf counts as a panel, as before.
    CHECK(spec_layout_min_size(*root, m).h == 2 * m.min_panel_h + m.gap);
    // Only controls: the rows stack from the top and the rest stays empty.
    spec::Spec only = s;
    only.panels.clear();
    only.layout.reset();
    const SpecLayout lc = lay(only, 900, 600);
    REQUIRE(lc.panels.size() == 2);
    CHECK(lc.panels[0].rect.h == m.control_row_h);
    CHECK(lc.panels[0].rect.y == 0);
}

TEST_CASE("layout: a scene panel's default title is its scene") {
    const spec::Spec s = instruments();
    spec::Panel p = s.panels[0];
    CHECK(spec_panel_title(s, p) == "Model");
    p.title.reset();
    CHECK(spec_panel_title(s, p) == "model");
}
