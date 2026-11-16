// The data of a spec panel made ready to draw (spec_prepare.h), headless:
// each primitive over known data, the options, the states in place of data,
// and the bounds with the record of each that applied.
#include "spec_runtime/spec_prepare.h"

#include "spec_test_data.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cctype>
#include <cmath>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::DType;

namespace {

PreparedPanel prep(const spec::Spec& s, const PanelData& d, const PrepareLimits& lim = {}) {
    REQUIRE(!s.panels.empty());
    return prepare_panel(s, s.panels[0], d, lim);
}

bool has_bound(const PreparedPanel& p, const std::string& text) {
    for (const auto& b : p.bounds)
        if (b.find(text) != std::string::npos) return true;
    return false;
}

// A state label is a short noun phrase: no sentence, at most three words.
void check_noun_phrase(const std::string& label) {
    CAPTURE(label);
    CHECK(!label.empty());
    CHECK(label.find('.') == std::string::npos);
    CHECK(std::count(label.begin(), label.end(), ' ') <= 2);
    CHECK(std::isupper((unsigned char)label[0]));
}

const char* kMetrics = "[[source]]\nid = \"m\"\ncatalog = \"table:run\"";
const char* kTensor = "[[source]]\nid = \"w\"\ncatalog = \"tensor:w\"";

}  // namespace

TEST_CASE("prepare: a series of two columns over an x column") {
    const spec::Spec s = one_panel(kMetrics, "primitive = \"series\"\nbind = { x = \"m.step\", y = [\"m.loss\", \"m.acc\"] }\n"
                                             "options = { y_log = true }");
    const PreparedPanel p = prep(s, data({column("x", "m.step", {0, 1, 2, 3}), column("y", "m.loss", {2.0, 1.0, 0.5, 0.0}, 0),
                                          column("y", "m.acc", {0.1, 0.2, nullptr, 0.4}, 1)},
                                         spec::Primitive::series));
    REQUIRE(p.state.ready());
    REQUIRE(p.series.lines.size() == 2);
    CHECK(p.series.lines[0].name == "loss");
    CHECK(p.series.lines[1].name == "acc");
    // A zero has no place on a log axis; a null is no value.
    CHECK(p.series.lines[0].y == std::vector<double>{2.0, 1.0, 0.5});
    CHECK(p.series.lines[1].x == std::vector<double>{0, 1, 3});
    CHECK(p.x_label == "step");
    CHECK(p.y_label.empty());   // two lines: the legend names them
    CHECK(p.title == "loss, acc");
    CHECK(p.info == "4 steps  2 lines");
}

TEST_CASE("prepare: a series over the index, from a rank-1 and a rank-2 tensor") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"series\"\nbind = { y = \"w\" }\noptions = { y_label = \"p\" }");
    {
        const PreparedPanel p = prep(s, data({f32("y", "w", {3}, {0.5f, 0.25f, 0.125f})}, spec::Primitive::series));
        REQUIRE(p.series.lines.size() == 1);
        CHECK(p.series.x_index);
        CHECK(p.series.lines[0].x == std::vector<double>{0, 1, 2});
        CHECK(p.x_label == "step");
        CHECK(p.y_label == "p");
        CHECK(p.info == "3 steps  last 0.125");
    }
    {
        const PreparedPanel p = prep(s, data({f32("y", "w", {2, 3}, {1, 2, 3, 4, 5, 6})}, spec::Primitive::series));
        REQUIRE(p.series.lines.size() == 2);
        CHECK(p.series.lines[1].name == "w[1]");
        CHECK(p.series.lines[1].y == std::vector<double>{4, 5, 6});
    }
}

TEST_CASE("prepare: a long series is decimated to an envelope that keeps its spikes") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"series\"\nbind = { y = \"w\" }");
    std::vector<float> v(100000);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)std::sin((double)i / 1000.0);
    v[54321] = 25.0f;   // one spike
    const PreparedPanel p = prep(s, data({f32("y", "w", {100000}, v)}, spec::Primitive::series));
    REQUIRE(p.series.lines.size() == 1);
    const SeriesLine& l = p.series.lines[0];
    CHECK(l.count == 100000);
    CHECK(l.y.size() <= 2 * 2048);
    CHECK(l.y.size() > 2048);
    CHECK(*std::max_element(l.y.begin(), l.y.end()) == 25.0);
    CHECK(has_bound(p, "decimated from 100000"));
    // At the limit nothing is decimated.
    const PreparedPanel q = prep(s, data({f32("y", "w", {8192}, std::vector<float>(8192, 1.0f))}, spec::Primitive::series));
    CHECK(q.series.lines[0].y.size() == 8192);
    CHECK(q.bounds.empty());
}

TEST_CASE("prepare: a heatmap's colours are the colormap's, over the range") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"heatmap\"\nbind = { values = \"w\" }\n"
                                            "options = { colormap = \"magma\", x_label = \"Key\" }");
    const PreparedPanel p = prep(s, data({f32("values", "w", {2, 3}, {0, 1, 2, 3, 4, 5})}, spec::Primitive::heatmap));
    REQUIRE(p.state.ready());
    const PreparedGrid& g = p.grid;
    CHECK(g.rows == 2);
    CHECK(g.cols == 3);
    CHECK(g.vmin == 0.0f);
    CHECK(g.vmax == 5.0f);
    const uint32_t* lut = spec_colormap_lut(kCmapMagma);
    auto px = [&](int r, int c) {
        const uint8_t* q = &g.rgba[((size_t)r * g.cols + c) * 4];
        return (uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24;
    };
    CHECK(px(0, 0) == lut[0]);
    CHECK(px(1, 2) == lut[255]);
    CHECK(p.info == "2\xC3\x97" "3  0 \xE2\x80\xA6 5");
    CHECK(p.x_label == "Key");

    // A fixed range clamps, and a flat tensor widens by a half each way.
    const spec::Spec r = one_panel(kTensor, "primitive = \"heatmap\"\nbind = { values = \"w\" }\noptions = { range = [0.0, 1.0] }");
    const PreparedPanel q = prep(r, data({f32("values", "w", {1, 2}, {-3, 7})}, spec::Primitive::heatmap));
    CHECK(q.grid.vmin == 0.0f);
    CHECK(q.grid.vmax == 1.0f);
    const PreparedPanel flat = prep(s, data({f32("values", "w", {2, 2}, {2, 2, 2, 2})}, spec::Primitive::heatmap));
    CHECK(flat.grid.vmin == 1.5f);
    CHECK(flat.grid.vmax == 2.5f);
}

TEST_CASE("prepare: a heatmap past the texture side is sampled at a stride") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"heatmap\"\nbind = { values = \"w\" }");
    PrepareLimits lim;
    lim.max_texture_side = 4;
    const PreparedPanel p = prep(s, data({f32("values", "w", {10, 3}, std::vector<float>(30, 1.0f))}, spec::Primitive::heatmap), lim);
    CHECK(p.grid.stride_r == 3);
    CHECK(p.grid.rows == 4);
    CHECK(p.grid.cols == 3);
    CHECK(p.grid.source_rows == 10);
    CHECK(has_bound(p, "heatmap sampled from 10x3 to 4x3"));
}

TEST_CASE("prepare: an image of u8 RGB and a one-channel float image") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"image\"\nbind = { values = \"w\" }");
    const std::vector<uint8_t> rgb = {255, 0, 0, 0, 0, 255};
    const PreparedPanel p = prep(s, data({tensor<uint8_t>("values", "w", DType::u8, {1, 2, 3}, rgb)}, spec::Primitive::image));
    REQUIRE(p.state.ready());
    CHECK(p.grid.channels == 3);
    CHECK(p.grid.rgba == std::vector<uint8_t>{255, 0, 0, 255, 0, 0, 255, 255});
    CHECK(p.info == "1\xC3\x97" "2 RGB");
    const PreparedPanel g = prep(s, data({f32("values", "w", {1, 2}, {0.0f, 1.0f})}, spec::Primitive::image));
    CHECK(g.grid.colormap == kCmapGreys);
    CHECK(g.grid.rgba == std::vector<uint8_t>{0, 0, 0, 255, 255, 255, 255, 255});
    // A four-dimensional tensor does not fit.
    const PreparedPanel bad = prep(s, data({f32("values", "w", {1, 1, 1, 1}, {0.0f})}, spec::Primitive::image));
    CHECK(bad.state.kind == PanelStateKind::mismatch);
}

TEST_CASE("prepare: points from positions and from columns, with colour and size") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\", color = \"w\" }\n"
                                            "options = { colormap = \"rdbu\" }");
    const PreparedPanel p = prep(s, data({f32("positions", "w", {3, 3}, {0, 0, 0, 1, 1, 1, 2, 2, 2}),
                                          f32("color", "w", {3}, {0, 1, 2})},
                                         spec::Primitive::points));
    REQUIRE(p.state.ready());
    CHECK(p.points.three_d);
    CHECK(p.points.x.size() == 3);
    CHECK(p.points.rgba.front() == spec_colormap_lut(kCmapRdbu)[0]);
    CHECK(p.points.rgba.back() == spec_colormap_lut(kCmapRdbu)[255]);
    CHECK(p.x_label == "x");
    CHECK(p.z_label == "z");
    CHECK(p.info == "3 points  3-D");

    const spec::Spec c = one_panel(kMetrics, "primitive = \"points\"\nbind = { x = \"m.step\", y = \"m.v\", size = \"m.v\" }");
    const PreparedPanel q = prep(c, data({column("x", "m.step", {0, 1, 2}), column("y", "m.v", {1.0, 2.0, 3.0}),
                                          column("size", "m.v", {1.0, 2.0, 3.0})},
                                         spec::Primitive::points));
    REQUIRE(q.state.ready());
    CHECK_FALSE(q.points.three_d);
    CHECK(q.points.scale == std::vector<float>{0.5f, 1.25f, 2.0f});
    CHECK(q.x_label == "step");
    CHECK(q.title == "v by step");
}

TEST_CASE("prepare: a cloud past max_points is sampled at a stride") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"points\"\nbind = { positions = \"w\" }");
    std::vector<float> pos(250000 * 2, 0.5f);
    const PreparedPanel p = prep(s, data({f32("positions", "w", {250000, 2}, pos)}, spec::Primitive::points));
    CHECK(p.points.total == 250000);
    CHECK(p.points.stride == 3);
    CHECK(p.points.x.size() == 83334);
    CHECK(has_bound(p, "points sampled from 250000 at a stride of 3"));
}

TEST_CASE("prepare: a histogram counts in its bins and range") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"histogram\"\nbind = { values = \"w\" }\n"
                                            "options = { bins = 4, range = [0.0, 4.0] }");
    const PreparedPanel p = prep(s, data({f32("values", "w", {2, 4}, {0.0f, 0.5f, 1.0f, 2.5f, 3.9f, 4.0f, 9.0f, NAN})},
                                         spec::Primitive::histogram));
    REQUIRE(p.state.ready());
    CHECK(p.histogram.counts == std::vector<double>{2, 1, 1, 2});
    CHECK(p.histogram.outside == 1);
    CHECK(p.histogram.total == 6);
    CHECK(p.y_label == "count");
    CHECK(has_bound(p, "1 values outside the range"));
    CHECK(p.info == "4 bins  0 \xE2\x80\xA6 4  n 6");
}

TEST_CASE("prepare: a table shows its columns, figures in one format, and at most max_table_rows rows") {
    const spec::Spec s = one_panel(kMetrics, "primitive = \"table\"\nbind = { rows = \"m\" }");
    json rows = json::array();
    for (int i = 0; i < 5000; ++i) rows.push_back(json::array({i, 1.0 / (i + 1), "run"}));
    const PreparedPanel p = prep(s, data({table("rows", "m", {{"step", "BIGINT"}, {"loss", "DOUBLE"}, {"name", "VARCHAR"}}, rows)},
                                         spec::Primitive::table));
    REQUIRE(p.state.ready());
    CHECK(p.table.columns == std::vector<std::string>{"step", "loss", "name"});
    CHECK(p.table.numeric == std::vector<bool>{true, true, false});
    CHECK(p.table.cells.size() == 1000);
    CHECK(p.table.total_rows == 5000);
    CHECK(p.table.truncated);
    CHECK(p.table.cells[0] == std::vector<std::string>{"0", "1.0000", "run"});
    CHECK(p.table.cells[3][1] == "0.2500");
    CHECK(p.info == "1,000 of 5,000 rows");
    CHECK(has_bound(p, "table cut from 5000 to 1000 rows"));
    CHECK(p.title == "run");
}

TEST_CASE("prepare: a text panel shows its text and the last value") {
    const spec::Spec s = one_panel(kMetrics, "primitive = \"text\"\nbind = { value = \"m.loss\" }\noptions = { text = \"Loss\" }");
    const PreparedPanel p = prep(s, data({column("value", "m.loss", {0.5, 0.25, nullptr})}, spec::Primitive::text));
    REQUIRE(p.state.ready());
    CHECK(p.text.text == "Loss");
    CHECK(p.text.value == "0.25");
    const spec::Spec t = one_panel("", "primitive = \"text\"\noptions = { text = \"A note.\" }");
    const PreparedPanel q = prep(t, data({}, spec::Primitive::text));
    CHECK(q.state.ready());
    CHECK_FALSE(q.text.has_value);
}

TEST_CASE("prepare: states take the panel's place, as short noun phrases naming the source") {
    const spec::Spec s = parse(R"(caliper_spec = 0
[applet]
id = "t"
[[source]]
id = "gpu"
catalog = "feed:gpu"
[[source]]
id = "w"
catalog = "tensor:w"
[[panel]]
id = "p"
primitive = "table"
bind = { rows = "gpu" }
[[panel]]
id = "h"
primitive = "heatmap"
bind = { values = "w" }
)");
    const PreparedPanel off = prepare_panel(s, s.panels[0], failed("resolve.unavailable", "gpu"));
    CHECK(off.state.kind == PanelStateKind::offline);
    CHECK(off.state.label == "Source offline");
    CHECK(off.state.detail == "feed:gpu");
    CHECK(off.info.empty());

    // Empty data, data that does not fit its slot, and each resolver error.
    const PreparedPanel empty = prepare_panel(s, s.panels[1], data({f32("values", "w", {0, 3}, {})}, spec::Primitive::heatmap, "h"));
    CHECK(empty.state.kind == PanelStateKind::empty);
    CHECK(empty.state.label == "No data");
    CHECK(empty.state.detail == "tensor:w");
    const PreparedPanel wrong = prepare_panel(s, s.panels[1], data({f32("values", "w", {4}, {1, 2, 3, 4})}, spec::Primitive::heatmap, "h"));
    CHECK(wrong.state.kind == PanelStateKind::mismatch);
    CHECK(wrong.state.label == "Mismatched data");

    for (const char* code : {"resolve.unavailable", "resolve.too_large", "resolve.query_failed", "resolve.timeout",
                             "resolve.unknown_column", "resolve.unknown_source", "resolve.cycle", "panel.not_found",
                             "spec.no_open_spec", "resolve.not_tabular", "resolve.not_a_tensor", "transform.bad_axis",
                             "loader.io", "resolve.failed", "anything.else"}) {
        const PanelState st = panel_state_for_error(code, "x");
        CHECK(st.kind != PanelStateKind::ready);
        check_noun_phrase(st.label);
    }
    check_noun_phrase(off.state.label);
    check_noun_phrase(empty.state.label);
    check_noun_phrase(wrong.state.label);
}

TEST_CASE("prepare: hostile shapes and values never crash") {
    const spec::Spec s = one_panel(kTensor, "primitive = \"series\"\nbind = { y = \"w\" }");
    // Every value non-finite: no line has a point, which is no data.
    const PreparedPanel p = prep(s, data({f32("y", "w", {3}, {NAN, INFINITY, -INFINITY})}, spec::Primitive::series));
    CHECK(p.state.kind == PanelStateKind::empty);
    // A slot that is missing altogether.
    const PreparedPanel q = prep(s, data({}, spec::Primitive::series));
    CHECK(q.state.kind == PanelStateKind::empty);
    // A column of strings in a numeric slot draws nothing.
    const spec::Spec c = one_panel(kMetrics, "primitive = \"histogram\"\nbind = { values = \"m.name\" }");
    const PreparedPanel h = prep(c, data({column("values", "m.name", {"a", "b"})}, spec::Primitive::histogram));
    CHECK(h.state.kind == PanelStateKind::empty);
}

TEST_CASE("prepare: numbers in the headers' four significant digits") {
    CHECK(spec_fmt4(0.0) == "0");
    CHECK(spec_fmt4(1234.5678) == "1235");
    CHECK(spec_fmt4(0.000123456) == "1.235e-4");
    CHECK(spec_fmt4(123456.0) == "1.235e5");
    CHECK(spec_fmt4(NAN) == "NaN");
}
