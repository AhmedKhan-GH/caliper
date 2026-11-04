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

TEST_CASE("hit testing a whole-cell region uses matrix and vector coordinates") {
    const tg::Region matrix{102, 198, 10, 20};
    const tg::Rect picture{10, 20, 400, 200};
    auto cell = tg::hit_test(matrix, picture, 20, 30, false);
    REQUIRE(cell.valid); CHECK(cell.row == 102); CHECK(cell.col == 198);
    cell = tg::hit_test(matrix, picture, 409.9, 219.9, false);
    REQUIRE(cell.valid); CHECK(cell.row == 111); CHECK(cell.col == 217);
    CHECK_FALSE(tg::hit_test(matrix, picture, 410, 100, false).valid);
    CHECK_FALSE(tg::hit_test(matrix, picture, 20, 220, false).valid);
    cell = tg::hit_test(tg::Region{102, 0, 20, 1}, {10, 20, 400, 20}, 100, 25, true);
    REQUIRE(cell.valid); CHECK(cell.row == 106); CHECK(cell.col == 0);
}

TEST_CASE("overview selection and recentering preserve true embedding bounds") {
    const tg::Rect overview = tg::fit(248320, 1024, {0, 0, 72, 400});
    const tg::Window window{10000, 0, 1024, 1024};
    const auto selected = tg::selection_rect(window, 248320, 1024, overview);
    contained(selected, overview);
    CHECK(selected.width == doctest::Approx(overview.width));
    CHECK(selected.height / overview.height == doctest::Approx(1024.0 / 248320));
    // A click in the strip centres the view on the picked row, at the same zoom.
    const tg::Rect bounds{0, 0, 400, 400};
    const auto camera = tg::frame(window, 248320, 1024, false, bounds);
    const auto picked = tg::hit_test(tg::Region{0, 0, 248320, 1024}, overview, overview.x + overview.width / 2, 200, false);
    const auto centred = tg::shoot(tg::centre_on(camera, 248320, 1024, false, bounds, picked), 248320, 1024, false, bounds);
    CHECK(centred.window.rows == doctest::Approx(1024)); CHECK(centred.window.col == doctest::Approx(0));
    CHECK(centred.window.row + 512 == doctest::Approx(picked.row + 0.5));
    CHECK(picked.row >= 124159); CHECK(picked.row <= 124160);
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

TEST_CASE("a zoomed camera fills the bounds wherever the tensor reaches past them") {
    const tg::Rect bounds{10, 20, 900, 300};
    // down_proj: 1024 x 3584 is wider than the bounds' shape, so it starts
    // whole, letterboxed top and bottom, with square cells.
    auto camera = tg::initial_camera(1024, 3584, false, bounds);
    auto shot = tg::shoot(camera, 1024, 3584, false, bounds);
    CHECK(shot.window.rows == doctest::Approx(1024)); CHECK(shot.window.cols == doctest::Approx(3584));
    CHECK(shot.picture.width == doctest::Approx(900)); CHECK(shot.picture.height < 300);
    square_cells(shot.picture, shot.window.rows, shot.window.cols);
    contained(shot.picture, bounds);
    // Four times closer the rows reach past the bounds too: the picture is
    // the whole bounds, still with square cells.
    camera = tg::zoom(camera, 1024, 3584, false, bounds, 4, 460, 170);
    shot = tg::shoot(camera, 1024, 3584, false, bounds);
    CHECK(shot.picture.width == doctest::Approx(900)); CHECK(shot.picture.height == doctest::Approx(300));
    square_cells(shot.picture, shot.window.rows, shot.window.cols);
    CHECK(shot.window.cols == doctest::Approx(3584 / 4.0));
    CHECK(camera.cell == doctest::Approx(4 * 900 / 3584.0));
    // A matrix narrower than the bounds at this zoom keeps its true width.
    const auto narrow = tg::shoot(tg::frame({0, 0, 16, 16}, 64, 16, false, bounds), 64, 16, false, bounds);
    CHECK(narrow.window.cols == doctest::Approx(16));
    CHECK(narrow.picture.height == doctest::Approx(300)); CHECK(narrow.picture.width == doctest::Approx(300));
}

TEST_CASE("camera zoom keeps the pointed cell under the pointer and is continuous") {
    const tg::Rect bounds{0, 0, 800, 400};
    auto camera = tg::initial_camera(2048, 2048, false, bounds);
    const double x = 500, y = 120;
    // While the picture is narrower than the bounds it stays centred, so the
    // anchor holds along the sides the picture fills: here rows at once,
    // columns from the fourth step.
    for (double step : {1.03, 1.03, 1.5, 7.0, 1.001}) {
        const auto before = tg::shoot(camera, 2048, 2048, false, bounds);
        const auto under = tg::hit_test(before.window, before.picture, x, y, false);
        camera = tg::zoom(camera, 2048, 2048, false, bounds, step, x, y);
        const auto shot = tg::shoot(camera, 2048, 2048, false, bounds);
        const auto cell = tg::hit_test(shot.window, shot.picture, x, y, false);
        REQUIRE(cell.valid); CHECK(cell.row == under.row);
        if (before.picture.width == doctest::Approx(800)) CHECK(cell.col == under.col);
    }
    CHECK(tg::shoot(camera, 2048, 2048, false, bounds).picture.width == doctest::Approx(800));
    // A tiny trackpad step still changes the zoom: nothing rounds to whole cells.
    const double cell = camera.cell;
    CHECK(tg::zoom(camera, 2048, 2048, false, bounds, 1.001, x, y).cell > cell);
    // Limits: never past the fit, never past max_cell.
    CHECK(tg::zoom(camera, 2048, 2048, false, bounds, 1e-6, x, y).cell == doctest::Approx(tg::fit_cell(2048, 2048, false, bounds)));
    CHECK(tg::zoom(camera, 2048, 2048, false, bounds, 1e6, x, y).cell == doctest::Approx(tg::max_cell));
}

TEST_CASE("camera pan, cover and centring stay inside the tensor") {
    const tg::Rect bounds{0, 0, 400, 200};
    auto camera = tg::frame({100, 200, 10, 20}, 1000, 2000, false, bounds);
    auto shot = tg::shoot(camera, 1000, 2000, false, bounds);
    CHECK(shot.window.row == doctest::Approx(100)); CHECK(shot.window.col == doctest::Approx(200));
    CHECK(tg::cover(shot.window, 1000, 2000) == tg::Region{100, 200, 10, 20});
    camera = tg::pan(camera, 1000, 2000, false, bounds, 40, -40);   // 20 points a cell
    shot = tg::shoot(camera, 1000, 2000, false, bounds);
    CHECK(shot.window.row == doctest::Approx(102)); CHECK(shot.window.col == doctest::Approx(198));
    CHECK(tg::cover({101.5, 3.25, 2, 0.5}, 1000, 2000) == tg::Region{101, 3, 3, 1});
    camera = tg::pan(camera, 1000, 2000, false, bounds, 1e9, 1e9);
    shot = tg::shoot(camera, 1000, 2000, false, bounds);
    CHECK(shot.window.row == doctest::Approx(0)); CHECK(shot.window.col == doctest::Approx(0));
    camera = tg::centre_on(camera, 1000, 2000, false, bounds, {999, 1999, true});
    shot = tg::shoot(camera, 1000, 2000, false, bounds);
    CHECK(shot.window.row + shot.window.rows == doctest::Approx(1000));
    CHECK(shot.window.col + shot.window.cols == doctest::Approx(2000));
    // Vectors run along x in a strip of square cells.
    const tg::Rect strip{0, 0, 480, 24};
    const auto vector = tg::initial_camera(1024, 1, true, strip);
    shot = tg::shoot(vector, 1024, 1, true, strip);
    CHECK(vector.cell == doctest::Approx(24)); CHECK(shot.window.rows == doctest::Approx(20));
    CHECK(shot.window.cols == doctest::Approx(1)); CHECK(shot.window.row == doctest::Approx(0));
    const auto cell = tg::hit_test(shot.window, shot.picture, 100, 12, true);
    REQUIRE(cell.valid); CHECK(cell.row == 4); CHECK(cell.col == 0);
    // An embedding starts filling the bounds' width from its first row.
    const auto embedding = tg::shoot(tg::initial_camera(248320, 1024, false, bounds), 248320, 1024, false, bounds);
    CHECK(embedding.window.cols == doctest::Approx(1024)); CHECK(embedding.window.row == doctest::Approx(0));
    CHECK(embedding.picture.width == doctest::Approx(400)); CHECK(embedding.picture.height == doctest::Approx(200));
}
