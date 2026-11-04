#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/tensor_geometry.h"

#include <array>
#include <cmath>
#include <limits>

namespace tg = caliper_live::tensor_geometry;
namespace {
void contained(tg::Rect inner, tg::Rect outer) {
    CHECK(std::isfinite(inner.x)); CHECK(std::isfinite(inner.y));
    CHECK(std::isfinite(inner.width)); CHECK(std::isfinite(inner.height));
    CHECK(inner.width >= 0); CHECK(inner.height >= 0);
    CHECK(inner.x >= outer.x - 1e-8); CHECK(inner.y >= outer.y - 1e-8);
    CHECK(inner.x + inner.width <= outer.x + outer.width + 1e-8);
    CHECK(inner.y + inner.height <= outer.y + outer.height + 1e-8);
}
void square_cells(tg::Rect image, double rows, double cols) {
    CHECK(image.width / cols == doctest::Approx(image.height / rows));
}
bool overlap(tg::Rect a, tg::Rect b) {
    return a.x < b.x + b.width && b.x < a.x + a.width &&
           a.y < b.y + b.height && b.y < a.y + a.height;
}
}

TEST_CASE("tensor fit preserves real dimensions and leaves empty space") {
    const tg::Rect bounds{17, 29, 600, 400};
    auto matrix = tg::fit(3584, 1024, bounds);
    contained(matrix, bounds); square_cells(matrix, 3584, 1024);
    CHECK(matrix.height == doctest::Approx(400));
    CHECK(matrix.width == doctest::Approx(400.0 * 1024 / 3584));
    CHECK(matrix.x > bounds.x);
    auto embedding = tg::fit(248320, 1024, bounds);
    contained(embedding, bounds); square_cells(embedding, 248320, 1024);
    CHECK(embedding.width == doctest::Approx(400.0 * 1024 / 248320));
    auto vector = tg::fit(1, 1024, {17, 29, 600, tg::strip_height});
    square_cells(vector, 1, 1024);
    CHECK(vector.height < tg::strip_height);
    CHECK(vector.width == doctest::Approx(600));
}

TEST_CASE("invalid and fractional fit inputs stay finite") {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(tg::fit(0, 5, {0, 0, 40, 50}).width == 0);
    CHECK(tg::fit(5, nan, {0, 0, 40, 50}).height == 0);
    CHECK(tg::fit(5, 5, {0, 0, -40, 50}).width == 0);
    auto fractional = tg::fit(7, 13, {2.5, 3.25, 89.5, 100.25});
    contained(fractional, {2.5, 3.25, 89.5, 100.25});
    square_cells(fractional, 7, 13);
}

TEST_CASE("matrix axis labels follow the fitted picture edges through letterboxing") {
    const tg::Rect area{117, 229, 600, 400};
    for (const auto shape : std::array<std::array<double, 2>, 2>{{{{3584, 1024}}, {{1024, 3584}}}}) {
        const auto picture = tg::fit(shape[0], shape[1], area);
        const auto labels = tg::axis_labels(picture, 35, 28, 140, 14, 4);
        CHECK(labels.first_row.x + labels.first_row.width + 4 == doctest::Approx(picture.x));
        CHECK(labels.first_row.y == doctest::Approx(picture.y));
        CHECK(labels.last_row.x + labels.last_row.width + 4 == doctest::Approx(picture.x));
        CHECK(labels.last_row.y + labels.last_row.height == doctest::Approx(picture.y + picture.height));
        CHECK(labels.columns.x == doctest::Approx(picture.x));
        CHECK(labels.columns.y == doctest::Approx(picture.y + picture.height + 4));
        CHECK_FALSE(overlap(labels.first_row, picture));
        CHECK_FALSE(overlap(labels.last_row, picture));
        CHECK_FALSE(overlap(labels.columns, picture));
        if (shape[0] > shape[1]) CHECK(labels.first_row.x > area.x + 100);
        else CHECK(labels.first_row.y > area.y + 100);
    }
}

TEST_CASE("embedding viewport axes remain beside its own picture after cropping") {
    const tg::Rect area{713, 641, 1170, 300};
    const auto layout = tg::view_layout(248320, 1024, false, area);
    REQUIRE(layout.elongated);
    const tg::Region region{10000, 100, 400, 600};
    const auto picture = tg::fit(region.rows, region.cols, layout.detail);
    const auto labels = tg::axis_labels(picture, 63, 35, 120, 16, 5);
    CHECK(labels.first_row.x + 63 + 5 == doctest::Approx(picture.x));
    CHECK(labels.first_row.y == doctest::Approx(picture.y));
    CHECK(labels.last_row.x + 35 + 5 == doctest::Approx(picture.x));
    CHECK(labels.last_row.y + 16 == doctest::Approx(picture.y + picture.height));
    CHECK(labels.columns.x == doctest::Approx(picture.x));
    CHECK(labels.columns.y == doctest::Approx(picture.y + picture.height + 5));
    CHECK(labels.first_row.x > layout.detail.x + 100);

    const auto tiny = tg::axis_labels({12, 34, 300, 8}, 35, 14, 120, 16);
    CHECK(tiny.last_row.height == 0);
    const auto invalid = tg::axis_labels({}, 35, 14, 120, 16);
    CHECK(invalid.first_row.height == 0);
    CHECK(invalid.columns.height == 0);
}

TEST_CASE("initial regions retain ordinary shapes and crop elongated shapes") {
    const tg::Rect area{0, 0, 600, 400};
    auto matrix = tg::initial_region(3584, 1024, area, false);
    CHECK(matrix == tg::Region{0, 0, 3584, 1024});
    auto embedding = tg::initial_region(248320, 1024, area, false);
    CHECK(embedding.row == 0); CHECK(embedding.col == 0);
    CHECK(embedding.cols == 1024); CHECK(embedding.rows == 683);
    CHECK(embedding.rows < 248320);
    auto wide = tg::initial_region(16, 12000, area, false);
    CHECK(wide.rows == 16); CHECK(wide.cols == 24);
    auto vector = tg::initial_region(1024, 1, area, true);
    CHECK(vector == tg::Region{0, 0, 25, 1});
}

TEST_CASE("default vector region fills the modest strip with square cells") {
    const tg::Rect bounds{17, 29, 600, 400};
    for (int64_t count : {int64_t{1024}, int64_t{16}}) {
        const auto layout = tg::view_layout(count, 1, true, bounds);
        const auto region = tg::initial_region(count, 1, layout.strip, true);
        const auto picture = tg::fit(region.cols, region.rows, layout.strip);
        CHECK(region.rows == (count == 1024 ? 25 : 16));
        CHECK(picture.height == doctest::Approx(tg::strip_height));
        square_cells(picture, region.cols, region.rows);
        contained(picture, layout.strip);
    }
    CHECK(tg::initial_region(1024, 1, {0, 0, 0, 0}, true).rows == 1);
    CHECK(tg::initial_region(1024, 1, {0, 0, 1, 24}, true).rows == 1);
}

TEST_CASE("zoom preserves a cell anchor and clamps at tensor edges") {
    const tg::Region source{100, 200, 100, 200};
    const tg::Cell anchor{125, 250, true};
    const auto closer = tg::zoom(source, 1000, 2000, 0.5, anchor);
    CHECK(closer.rows == 50); CHECK(closer.cols == 100);
    CHECK(closer.row == 113); CHECK(closer.col == 225);
    const auto one = tg::zoom(closer, 1000, 2000, 0.0001, anchor);
    CHECK(one.rows == 1); CHECK(one.cols == 1);
    CHECK(one.row == anchor.row); CHECK(one.col == anchor.col);
    const auto whole = tg::zoom(source, 1000, 2000, 1000, anchor);
    CHECK(whole == tg::Region{0, 0, 1000, 2000});
    CHECK(tg::zoom(source, 1000, 2000, 0, anchor) == source);
    CHECK(tg::clamp({-5, 9000, 5000, -2}, 1000, 2000) == tg::Region{0, 1999, 1000, 1});
}

TEST_CASE("pan and hit testing use the same matrix and vector coordinates") {
    const tg::Region source{100, 200, 10, 20};
    const tg::Rect picture{10, 20, 400, 200};
    const auto matrix = tg::pan(source, 1000, 2000, 40, -40, picture, false);
    CHECK(matrix == tg::Region{102, 198, 10, 20});
    auto cell = tg::hit_test(matrix, picture, 20, 30, false);
    REQUIRE(cell.valid); CHECK(cell.row == 102); CHECK(cell.col == 198);
    cell = tg::hit_test(matrix, picture, 409.9, 219.9, false);
    REQUIRE(cell.valid); CHECK(cell.row == 111); CHECK(cell.col == 217);
    CHECK_FALSE(tg::hit_test(matrix, picture, 410, 100, false).valid);
    CHECK_FALSE(tg::hit_test(matrix, picture, 20, 220, false).valid);
    const tg::Region vector{100, 0, 20, 1};
    const auto shifted = tg::pan(vector, 1024, 1, -40, 0, {10, 20, 400, 20}, true);
    CHECK(shifted == tg::Region{102, 0, 20, 1});
    cell = tg::hit_test(shifted, {10, 20, 400, 20}, 100, 25, true);
    REQUIRE(cell.valid); CHECK(cell.row == 106); CHECK(cell.col == 0);
    // Using total drag from its starting region lets several small pointer
    // movements eventually cross a cell boundary at high magnification.
    CHECK(tg::pan(vector, 1024, 1, -12, 0, {0, 0, 480, 24}, true).row == 101);
}

TEST_CASE("overview selection and recentering preserve true embedding bounds") {
    const tg::Rect overview = tg::fit(248320, 1024, {0, 0, 72, 400});
    const tg::Region region{10000, 0, 1024, 1024};
    const auto selected = tg::selection_rect(region, 248320, 1024, overview);
    contained(selected, overview);
    CHECK(selected.width == doctest::Approx(overview.width));
    CHECK(selected.height / overview.height == doctest::Approx(1024.0 / 248320));
    const auto centered = tg::overview_region(region, 248320, 1024, overview,
                                               overview.x + overview.width / 2, 200);
    CHECK(centered.row >= 123648); CHECK(centered.row <= 123649); CHECK(centered.col == 0);
    CHECK(tg::overview_region(region, 248320, 1024, overview, 0, 200) == region);
}

TEST_CASE("sampling follows physical resolution and respects cell and frame budgets") {
    auto target = tg::sampling_target({0, 0, 2000, 1000}, {0, 0, 500, 300}, 1, false);
    CHECK(target.width <= 500); CHECK(target.height <= 300);
    CHECK(target.width * target.height <= tg::max_samples);
    auto retina = tg::sampling_target({0, 0, 2000, 1000}, {0, 0, 500, 300}, 2, false);
    CHECK(retina.width >= target.width); CHECK(retina.height >= target.height);
    CHECK(retina.width * retina.height <= tg::max_samples);
    auto cells = tg::sampling_target({0, 0, 7, 13}, {0, 0, 500, 300}, 2, false);
    CHECK(cells.width == 13); CHECK(cells.height == 7);
    auto vector = tg::sampling_target({100, 0, 1024, 1}, {0, 0, 600, 24}, 1, true);
    CHECK(vector.width == 600); CHECK(vector.height == 1);
    auto huge = tg::sampling_target({0, 0, 1000000, 1000000}, {0, 0, 1e9, 1e9}, 1e9, false);
    CHECK(huge.width <= tg::max_target_side); CHECK(huge.height <= tg::max_target_side);
    CHECK(huge.width * huge.height <= tg::max_samples);
    auto invalid = tg::sampling_target({0, 0, 1, 1}, {0, 0, 0, 0}, 0, false);
    CHECK(invalid.width == 1); CHECK(invalid.height == 1);
}

TEST_CASE("view layout separates overview and detail and gives vectors one short strip") {
    const tg::Rect area{11, 22, 600, 400};
    auto embedding = tg::view_layout(248320, 1024, false, area);
    REQUIRE(embedding.elongated);
    contained(embedding.overview, area); contained(embedding.detail, area);
    CHECK_FALSE(overlap(embedding.overview, embedding.detail));
    square_cells(embedding.overview, 248320, 1024);
    CHECK(embedding.detail.width > area.width * 0.7);
    auto matrix = tg::view_layout(3584, 1024, false, area);
    CHECK_FALSE(matrix.elongated); CHECK(matrix.detail.width == area.width);
    auto wide = tg::view_layout(16, 12000, false, area);
    CHECK(wide.elongated); contained(wide.overview, area); contained(wide.detail, area);
    CHECK_FALSE(overlap(wide.overview, wide.detail));
    auto vector = tg::view_layout(1024, 1, true, area);
    CHECK(vector.strip.height == tg::strip_height);
    CHECK(vector.plot.height > vector.strip.height);
    contained(vector.plot, area); contained(vector.strip, area);
    CHECK_FALSE(overlap(vector.plot, vector.strip));
}

TEST_CASE("inspector window layout stays within its work area without overlap") {
    for (const tg::Rect work : std::array<tg::Rect, 4>{{{0, 0, 1280, 800}, {18, 44, 960, 640},
                                                     {0, 0, 320, 240}, {0, 0, 1, 1}}}) {
        const auto windows = tg::window_layout(work);
        const std::array<tg::Rect, 4> all{windows.inference, windows.tree, windows.model3d, windows.tensor};
        for (const auto& r : all) contained(r, work);
        for (size_t i = 0; i < all.size(); ++i)
            for (size_t j = i + 1; j < all.size(); ++j) CHECK_FALSE(overlap(all[i], all[j]));
        CHECK(windows.tensor.width > windows.inference.width);
    }
}
