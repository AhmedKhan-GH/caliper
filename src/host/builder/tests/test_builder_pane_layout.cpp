// The builder pane's regions and drop targets as plain rectangles
// (builder_pane_layout.h), and its pure helpers (builder_pane_edits.h),
// checked without a window: the regions at 2560x1440 and 1440x900 and at two
// text scales, what gives way on a small window, the drop zones, the
// dividers and the strips of the dashboard's layout, the pane's bindings for
// dropped data, and the layout tree's edits.
#include "builder/builder_pane_edits.h"
#include "builder/builder_pane_layout.h"

#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <sstream>

using namespace caliper_host;
namespace spec = caliper_builder::spec;

namespace {

spec::Spec fixture() {
    std::ifstream in(std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = spec::parse_toml(ss.str());
    REQUIRE(parsed.ok());
    return parsed.spec;
}

bool inside(const SpecRect& outer, const SpecRect& r) {
    return r.x >= outer.x && r.y >= outer.y && r.right() <= outer.right() && r.bottom() <= outer.bottom();
}

bool apart(const SpecRect& a, const SpecRect& b) {
    return a.right() <= b.x || b.right() <= a.x || a.bottom() <= b.y || b.bottom() <= a.y;
}

bool whole(const SpecRect& r) {
    return r.x == std::round(r.x) && r.y == std::round(r.y) && r.w == std::round(r.w) && r.h == std::round(r.h);
}

// The page's work area: the window less the page bar (28) and the status bar (22).
SpecRect work(float w, float h) { return {0.0f, 28.0f, w, h - 50.0f}; }

void check_regions(const BuilderPaneLayout& l, const SpecRect& a) {
    std::vector<SpecRect> regions = {l.canvas};
    if (l.left) regions.insert(regions.end(), {l.sources, l.primitives});
    if (l.right) regions.push_back(l.properties);
    if (l.strip) regions.push_back(l.strip_rect);
    if (l.chat) regions.push_back(l.chat_rect);
    for (const auto& r : regions) {
        CHECK(inside(a, r));
        CHECK(whole(r));
    }
    for (size_t i = 0; i < regions.size(); ++i)
        for (size_t j = i + 1; j < regions.size(); ++j) CHECK(apart(regions[i], regions[j]));
    for (const auto& r : l.rules) {
        CHECK(inside(a, r));
        for (const auto& g : regions) CHECK(apart(r, g));
    }
}

}  // namespace

TEST_CASE("pane layout: the regions at 2560x1440 and 1440x900") {
    const BuilderPaneMetrics m = BuilderPaneMetrics::for_text_scale(1.0f);
    for (const auto& size : {std::pair<float, float>{2560, 1440}, {1440, 900}}) {
        CAPTURE(size.first);
        const SpecRect a = work(size.first, size.second);
        BuilderPaneLayoutInput in;
        in.area = a;
        const BuilderPaneLayout l = compute_builder_pane_layout(in, m);
        REQUIRE(l.left);
        REQUIRE(l.right);
        CHECK_FALSE(l.strip);
        CHECK(l.collapsed.empty());
        check_regions(l, a);
        // The columns are shares of the width, at least their minimums; the
        // canvas takes the rest and is the largest region.
        CHECK(l.left_column.w == std::max(m.left_min, std::round(a.w * m.left_share)));
        CHECK(l.right_column.w == std::max(m.right_min, std::round(a.w * m.right_share)));
        CHECK(l.left_column.w + l.canvas.w + l.right_column.w + 2.0f == a.w);
        CHECK(l.canvas.w > l.left_column.w + l.right_column.w);
        CHECK(l.canvas.h == a.h);
        // Sources over Primitives, which keep their seven rows.
        CHECK(l.primitives.h == m.primitives_h);
        CHECK(l.sources.h + 1.0f + l.primitives.h == a.h);
        CHECK(l.properties.h == a.h);
        // The text strip under the canvas, a share of its height.
        in.text_open = true;
        const BuilderPaneLayout t = compute_builder_pane_layout(in, m);
        REQUIRE(t.strip);
        check_regions(t, a);
        CHECK(t.strip_rect.h == std::max(m.strip_min_h, std::round(a.h * m.strip_share)));
        CHECK(t.canvas.h + 1.0f + t.strip_rect.h == a.h);
        CHECK(t.strip_rect.x == t.canvas.x);
        CHECK(t.strip_rect.w == t.canvas.w);
        // The side chat in the right column's lower part, when the page gives it.
        in.chat = true;
        const BuilderPaneLayout c = compute_builder_pane_layout(in, m);
        REQUIRE(c.chat);
        check_regions(c, a);
        CHECK(c.properties.h + 1.0f + c.chat_rect.h == a.h);
        CHECK(c.properties.h >= m.properties_min_h);
    }
    // At 2560x1440 the numbers themselves.
    BuilderPaneLayoutInput in;
    in.area = work(2560, 1440);
    const BuilderPaneLayout l = compute_builder_pane_layout(in, m);
    CHECK(l.left_column.w == 435.0f);
    CHECK(l.right_column.w == 538.0f);
    CHECK(l.canvas.x == 436.0f);
    CHECK(l.canvas.w == 1585.0f);
}

TEST_CASE("pane layout: outside build mode the canvas takes the whole area") {
    BuilderPaneLayoutInput in;
    in.area = work(1440, 900);
    in.build = false;
    in.text_open = true;
    const BuilderPaneLayout l = compute_builder_pane_layout(in, BuilderPaneMetrics::for_text_scale(1.0f));
    CHECK_FALSE(l.left);
    CHECK_FALSE(l.right);
    CHECK_FALSE(l.strip);
    CHECK(l.canvas.w == 1440.0f);
    CHECK(l.canvas.h == 850.0f);
}

TEST_CASE("pane layout: minimums grow with the text scale") {
    const BuilderPaneMetrics a = BuilderPaneMetrics::for_text_scale(1.0f);
    const BuilderPaneMetrics b = BuilderPaneMetrics::for_text_scale(1.5f);
    CHECK(b.left_min == std::round(a.left_min * 1.5f));
    CHECK(b.right_min == std::round(a.right_min * 1.5f));
    CHECK(b.canvas_min_w == std::round(a.canvas_min_w * 1.5f));
    CHECK(b.strip_min_h > a.strip_min_h);
    CHECK(b.row > a.row);
    // At 1440x900 and scale 1.5 the columns are at least their larger minimums.
    BuilderPaneLayoutInput in;
    in.area = work(1440, 900);
    const BuilderPaneLayout l = compute_builder_pane_layout(in, b);
    REQUIRE(l.left);
    REQUIRE(l.right);
    CHECK(l.left_column.w >= b.left_min);
    CHECK(l.right_column.w >= b.right_min);
    CHECK(l.canvas.w >= b.canvas_min_w);
    check_regions(l, in.area);
}

TEST_CASE("pane layout: what gives way on a small window, in order") {
    const BuilderPaneMetrics m = BuilderPaneMetrics::for_text_scale(1.0f);
    BuilderPaneLayoutInput in;
    in.text_open = true;
    // Tall enough for the strip, narrow enough to shrink both columns to
    // their minimums: nothing hidden yet.
    in.area = {0, 0, m.left_min + m.right_min + m.canvas_min_w + 2.0f, 800.0f};
    BuilderPaneLayout l = compute_builder_pane_layout(in, m);
    CHECK(l.collapsed.empty());
    CHECK(l.left_column.w == m.left_min);
    CHECK(l.right_column.w == m.right_min);
    CHECK(l.canvas.w == m.canvas_min_w);
    // One pixel narrower: the right column goes first.
    in.area.w -= 1.0f;
    l = compute_builder_pane_layout(in, m);
    CHECK(l.collapsed == std::vector<std::string>{"right"});
    CHECK(l.left);
    CHECK_FALSE(l.right);
    check_regions(l, in.area);
    // Then the left.
    in.area.w = m.left_min + m.canvas_min_w;
    l = compute_builder_pane_layout(in, m);
    CHECK(l.collapsed == std::vector<std::string>{"right", "left"});
    CHECK(l.canvas.w == in.area.w);
    // Short: the strip shrinks to its minimum, then goes before anything else.
    in.area = {0, 0, 1440, m.canvas_min_h + 1.0f + m.strip_min_h};
    l = compute_builder_pane_layout(in, m);
    CHECK(l.strip);
    CHECK(l.strip_rect.h == m.strip_min_h);
    CHECK(l.canvas.h == m.canvas_min_h);
    in.area.h -= 1.0f;
    l = compute_builder_pane_layout(in, m);
    CHECK_FALSE(l.strip);
    CHECK(l.collapsed == std::vector<std::string>{"strip"});
    CHECK(l.canvas.h == in.area.h);
}

TEST_CASE("pane layout: the drop zones, dividers and strips of the dashboard") {
    const spec::Spec s = fixture();
    const BuilderPaneMetrics m = BuilderPaneMetrics::for_text_scale(1.0f);
    const SpecRect canvas{436, 28, 1585, 1390};
    const auto root = caliper_builder::effective_layout(s);
    REQUIRE(root);
    const SpecLayout lay = compute_spec_layout(*root, s, canvas, SpecLayoutMetrics::for_text_scale(1.0f));
    const SpecRect curves = lay.find("curves")->rect, attention = lay.find("attention")->rect, rows = lay.find("rows")->rect;
    const float strip = 24.0f;

    // Outside the canvas: nothing.
    CHECK(hit_drop(lay, canvas, canvas.x - 5, canvas.y + 50, 0, strip, m).zone == DropZone::none);
    // The band along the border: the whole layout's edges.
    DropHit h = hit_drop(lay, canvas, canvas.x + canvas.w * 0.5f, canvas.bottom() - 3, 0, strip, m);
    CHECK(h.zone == DropZone::root);
    CHECK(h.where == "below");
    h = hit_drop(lay, canvas, canvas.x + 2, canvas.y + 300, 0, strip, m);
    CHECK(h.zone == DropZone::root);
    CHECK(h.where == "left");
    // A panel's right edge, then its centre.
    h = hit_drop(lay, canvas, curves.right() - 20, curves.y + curves.h * 0.5f, 0, strip, m);
    CHECK(h.zone == DropZone::edge);
    CHECK(h.panel == "curves");
    CHECK(h.where == "beside");
    h = hit_drop(lay, canvas, attention.x + 20, attention.y + attention.h * 0.5f, 0, strip, m);
    CHECK(h.where == "left");
    h = hit_drop(lay, canvas, rows.x + rows.w * 0.5f, rows.y + 30, 0, strip, m);
    CHECK(h.zone == DropZone::edge);
    CHECK(h.where == "above");
    h = hit_drop(lay, canvas, attention.x + attention.w * 0.5f, attention.y + attention.h * 0.5f, 0, strip, m);
    CHECK(h.zone == DropZone::centre);
    CHECK(h.where == "tabbed");
    // A source's slots in the centre: two rows, the hit row by index.
    const std::vector<SpecRect> slots = drop_slot_rows(curves, strip, 2, m);
    REQUIRE(slots.size() == 2);
    for (const auto& r : slots) CHECK(inside(curves, r));
    CHECK(slots[0].bottom() < slots[1].y);
    h = hit_drop(lay, canvas, slots[1].x + 5, slots[1].y + 5, 2, strip, m);
    CHECK(h.zone == DropZone::slot);
    CHECK(h.slot == 1);
    CHECK(h.panel == "curves");
    // The rule between two panels is no target.
    CHECK(hit_drop(lay, canvas, curves.right() + 0.5f, curves.y + 200, 0, strip, m).zone == DropZone::none);

    // The dividers: the root's between the upper row and the table, the
    // row's between the curves and the attention, each with its slop.
    const SpecLayoutSplit* d = hit_divider(lay, canvas.x + 100, curves.bottom() + 0.5f, m);
    REQUIRE(d);
    CHECK(d->path == "");
    CHECK_FALSE(d->horizontal);
    d = hit_divider(lay, curves.right() + 3, curves.y + 100, m);
    REQUIRE(d);
    CHECK(d->path == "0");
    CHECK(hit_divider(lay, curves.right() - 10, curves.y + 100, m) == nullptr);
    // A drag of the root divider to 65 % of its length gives 0.65.
    const SpecLayoutSplit& rs = lay.splits[0];
    CHECK(divider_ratio(rs, 0, rs.rect.y + 0.65f * rs.length) == doctest::Approx(0.65));
    CHECK(divider_ratio(rs, 0, rs.rect.y + 2.0f) == doctest::Approx(0.05));
    CHECK(divider_ratio(rs, 0, rs.rect.bottom()) == doctest::Approx(0.95));

    // The strips, for a panel's drag; a panel's body is not one.
    CHECK(hit_header(lay, curves.x + 50, curves.y + 5, strip) == "curves");
    CHECK(hit_header(lay, curves.x + 50, curves.y + 60, strip).empty());
    CHECK(hit_panel(lay, curves.x + 50, curves.y + 60) == "curves");
}

TEST_CASE("pane edits: shapes in the panel headers' style") {
    PaneShape t;
    t.known = t.tabular = true;
    t.columns = {{"step", "BIGINT", true}, {"loss", "DOUBLE", true}, {"run", "VARCHAR", false}};
    t.rows = 4000;
    CHECK(pane_shape_text(t) == "4,000 \xc3\x97 3");
    t.rows.reset();
    CHECK(pane_shape_text(t) == "3 columns");
    PaneShape x;
    x.known = true;
    x.dtype = "f32";
    x.dims = {4, 8, 48, 48};
    CHECK(pane_shape_text(x) == "f32 4\xc3\x97" "8\xc3\x97" "48\xc3\x97" "48");
    CHECK(pane_count_text(1234567) == "1,234,567");
    CHECK(pane_type_numeric("DOUBLE"));
    CHECK(pane_type_numeric("decimal(10, 2)"));
    CHECK_FALSE(pane_type_numeric("VARCHAR"));
    CHECK_FALSE(pane_type_numeric("FLOAT[]"));
}

TEST_CASE("pane edits: the bindings a dropped source or column takes") {
    PaneShape t;
    t.known = t.tabular = true;
    t.columns = {{"step", "BIGINT", true}, {"loss", "DOUBLE", true}, {"accuracy", "DOUBLE", true}};
    // A table as a series: the first column as x, the others as lines.
    auto b = pane_auto_bind(spec::Primitive::series, "metrics", t);
    REQUIRE(b);
    CHECK(b->at("x").bindings == std::vector<std::string>{"metrics.step"});
    CHECK(b->at("y").bindings == std::vector<std::string>{"metrics.loss", "metrics.accuracy"});
    CHECK(b->at("y").list);
    CHECK(pane_bind_json(*b) == nlohmann::json{{"x", "metrics.step"}, {"y", {"metrics.loss", "metrics.accuracy"}}});
    // One column as a series, over the first column.
    b = pane_auto_bind(spec::Primitive::series, "metrics", t, "loss");
    REQUIRE(b);
    CHECK(pane_bind_json(*b) == nlohmann::json{{"x", "metrics.step"}, {"y", "metrics.loss"}});
    CHECK(pane_bind_json(*pane_auto_bind(spec::Primitive::table, "metrics", t)) == nlohmann::json{{"rows", "metrics"}});
    CHECK_FALSE(pane_auto_bind(spec::Primitive::heatmap, "metrics", t));
    CHECK(pane_default_primitive(t) == spec::Primitive::series);
    CHECK(pane_default_primitive(t, "loss") == spec::Primitive::series);

    PaneShape x;
    x.known = true;
    x.dtype = "f32";
    x.dims = {48, 48};
    CHECK(pane_default_primitive(x) == spec::Primitive::heatmap);
    CHECK(pane_bind_json(*pane_auto_bind(spec::Primitive::heatmap, "attn", x)) == nlohmann::json{{"values", "attn"}});
    x.dims = {4, 8, 48, 48};
    CHECK_FALSE(pane_auto_bind(spec::Primitive::heatmap, "attn", x));
    CHECK(pane_default_primitive(x) == spec::Primitive::histogram);
    x.dims = {3000, 3};
    CHECK(pane_default_primitive(x) == spec::Primitive::points);
    CHECK(pane_bind_json(*pane_auto_bind(spec::Primitive::points, "emb", x)) == nlohmann::json{{"positions", "emb"}});
    x.dims = {192, 256, 3};
    CHECK(pane_default_primitive(x) == spec::Primitive::image);
}

TEST_CASE("pane edits: ids, renames and the layout tree") {
    const spec::Spec s = fixture();
    CHECK(pane_source_id("run42_metrics", s) == "run42_metrics");
    CHECK(pane_source_id("attn.weights", s) == "attn_weights");
    CHECK(pane_source_id("metrics", s) == "metrics-2");
    CHECK(pane_source_id("7/train/loss", s) == "s7_train_loss");
    CHECK(pane_panel_id(spec::Primitive::series, s) == "series");
    CHECK(pane_valid_id("curves"));
    CHECK_FALSE(pane_valid_id("Curves"));

    // A rename reaches the layout, and a source's rename every binding.
    const spec::Spec p = pane_rename_panel(s, "rows", "table");
    CHECK(p.find_panel("table"));
    CHECK(caliper_builder::layout_to_flat(*p.layout, p).dump().find("\"table\"") != std::string::npos);
    CHECK(spec::validate(p).ok());
    const spec::Spec r = pane_rename_source(s, "metrics", "m");
    CHECK(r.find_panel("curves")->bind.at("y").bindings == std::vector<std::string>{"m.loss", "m.accuracy"});
    CHECK(r.find_panel("rows")->bind.at("rows").bindings == std::vector<std::string>{"m"});
    CHECK(spec::validate(r).ok());

    // The layout tree: the root's ratio, a panel at the whole layout's
    // bottom, a panel on another's left.
    CHECK(pane_layout_node(*s.layout, "")->ratio == 0.65);
    CHECK(pane_layout_node(*s.layout, "0.1")->panel == "attention");
    CHECK(pane_layout_node(*s.layout, "0.5") == nullptr);
    spec::LayoutNode half = pane_with_ratio(*s.layout, "", 0.5);
    CHECK_FALSE(half.ratio.has_value());
    CHECK(pane_with_ratio(half, "", 0.65) == *s.layout);
    CHECK(pane_snap_ratio(0.6473) == doctest::Approx(0.65));
    CHECK(pane_snap_ratio(0.01) == doctest::Approx(0.05));
    // The upper row, then the table below it: the fixture's layout but its ratio.
    std::optional<spec::LayoutNode> row = *pane_layout_node(*s.layout, "0");
    spec::LayoutNode below = pane_place_at_root(row, "rows", "below");
    CHECK(pane_with_ratio(below, "", 0.65) == *s.layout);
    // Moving the attention to the curves' left swaps them.
    const auto left = pane_place_next_to(s.layout, "attention", "curves", "left");
    REQUIRE(left);
    CHECK(pane_layout_node(*left, "0.0")->panel == "attention");
    CHECK(pane_layout_node(*left, "0.1")->panel == "curves");
    CHECK(pane_layout_node(*left, "")->ratio == 0.65);
    CHECK_FALSE(pane_place_next_to(s.layout, "curves", "curves", "left"));
}
