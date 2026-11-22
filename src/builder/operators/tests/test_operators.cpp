// The tests of the built-in operators (caliper_builder/operators/operators.h):
// the CPU references against fixed grids, the WGSL sources and the program
// declaration of a placed operator.
#include <doctest/doctest.h>

#include <caliper_builder/operators/operators.h>
#include <caliper_builder/programs/operator_kernels.h>
#include <caliper_builder/programs/sandbox.h>

#include <cmath>
#include <vector>

namespace ops = caliper_builder::operators;
namespace sb = caliper_builder::programs;
namespace sp = caliper_builder::spec;

namespace {

// A (rows, cols) grid whose cell (r, c) holds 10 r + c.
std::vector<float> grid(int rows, int cols) {
    std::vector<float> v;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) v.push_back(float(10 * r + c));
    return v;
}

const float kNaN = std::nanf("");

}  // namespace

TEST_CASE("operators: both WGSL sources are compiled in") {
    CHECK_FALSE(ops::builtin_wgsl("surface").empty());
    CHECK_FALSE(ops::builtin_wgsl("path").empty());
    CHECK(ops::builtin_wgsl("other").empty());
    CHECK(ops::builtin_version("surface") == 1);
    CHECK(ops::builtin_version("other") == 0);
    CHECK(ops::builtin_sha256("surface") == sb::sha256_hex(ops::builtin_wgsl("surface")));
    CHECK(ops::builtin_sha256("surface").size() == 64);
    CHECK(ops::builtin_sha256("surface") != ops::builtin_sha256("path"));
    CHECK(ops::builtin_sha256("other").empty());
}

TEST_CASE("operators: surface pools blocks of f by f cells with the smallest f that fits") {
    const std::vector<float> g = grid(4, 6);
    // 24 cells fit 32: no pooling, the coordinates are the indices.
    ops::Grid same = ops::surface_reference(g.data(), 4, 6, nullptr, 32, ops::Pool::mean);
    CHECK(same.rows == 4);
    CHECK(same.cols == 6);
    CHECK(same.height == g);
    CHECK(same.x == std::vector<float>{0, 1, 2, 3});
    CHECK(same.y == std::vector<float>{0, 1, 2, 3, 4, 5});

    // At most 6 cells: f = 2 gives 2 by 3.
    ops::Grid mean = ops::surface_reference(g.data(), 4, 6, nullptr, 6, ops::Pool::mean);
    CHECK(mean.rows == 2);
    CHECK(mean.cols == 3);
    CHECK(mean.height == std::vector<float>{5.5f, 7.5f, 9.5f, 25.5f, 27.5f, 29.5f});
    CHECK(mean.x == std::vector<float>{0.5f, 2.5f});
    CHECK(mean.y == std::vector<float>{0.5f, 2.5f, 4.5f});
    ops::Grid mx = ops::surface_reference(g.data(), 4, 6, nullptr, 6, ops::Pool::max);
    CHECK(mx.height == std::vector<float>{11, 13, 15, 31, 33, 35});
    ops::Grid mn = ops::surface_reference(g.data(), 4, 6, nullptr, 6, ops::Pool::min);
    CHECK(mn.height == std::vector<float>{0, 2, 4, 20, 22, 24});

    // A ragged last block: 5 by 5 in at most 4 cells takes f = 3, blocks of rows [0, 3) and [3, 5).
    const std::vector<float> g5 = grid(5, 5);
    ops::Grid ragged = ops::surface_reference(g5.data(), 5, 5, nullptr, 4, ops::Pool::mean);
    CHECK(ragged.rows == 2);
    CHECK(ragged.cols == 2);
    CHECK(ragged.x == std::vector<float>{1.0f, 3.5f});
    CHECK(ragged.height[0] == doctest::Approx(11.0));          // rows 0-2, columns 0-2
    CHECK(ragged.height[3] == doctest::Approx(38.5));          // rows 3-4, columns 3-4: (33+34+43+44)/4

    // After a window, x is the mean step of each block of rows.
    const std::vector<float> steps = {100, 101, 102, 104};
    ops::Grid stepped = ops::surface_reference(g.data(), 4, 6, &steps, 6, ops::Pool::mean);
    CHECK(stepped.x == std::vector<float>{100.5f, 103.0f});
}

TEST_CASE("operators: a pooled cell skips NaN values, and a block of NaN values gives NaN") {
    std::vector<float> g = grid(4, 6);
    g[0] = kNaN;                       // (0, 0) in the block of (0, 1), (1, 0), (1, 1)
    g[4] = g[5] = g[10] = g[11] = kNaN;   // the whole block (0, 2)
    ops::Grid mean = ops::surface_reference(g.data(), 4, 6, nullptr, 6, ops::Pool::mean);
    CHECK(mean.height[0] == doctest::Approx((1.0 + 10.0 + 11.0) / 3.0));
    CHECK(std::isnan(mean.height[2]));
    CHECK(mean.height[1] == doctest::Approx(7.5));
    ops::Grid mx = ops::surface_reference(g.data(), 4, 6, nullptr, 6, ops::Pool::max);
    CHECK(mx.height[0] == 11.0f);
    CHECK(std::isnan(mx.height[2]));
}

TEST_CASE("operators: path keeps every s-th row with the smallest s that fits") {
    // Row r holds (r, 2 r, 3 r).
    std::vector<float> rows;
    for (int r = 0; r < 10; ++r)
        for (float k : {1.0f, 2.0f, 3.0f}) rows.push_back(k * float(r));
    ops::Polyline all = ops::path_reference(rows.data(), 10, 3, {0, 1, 2}, 3, 65536, nullptr);
    CHECK(all.values.size() == 10);
    CHECK(all.positions.size() == 30);

    // At most 4 points: s = 3 keeps the rows 0, 3, 6 and 9.
    ops::Polyline kept = ops::path_reference(rows.data(), 10, 3, {0, 1, 2}, 3, 4, nullptr);
    CHECK(kept.positions == std::vector<float>{0, 0, 0, 3, 6, 9, 6, 12, 18, 9, 18, 27});
    CHECK(kept.values == std::vector<float>{0, 3, 6, 9});

    // Two columns, in the order given, give a z of 0.
    ops::Polyline two = ops::path_reference(rows.data(), 10, 3, {2, 0, 0}, 2, 4, nullptr);
    CHECK(two.positions == std::vector<float>{0, 0, 0, 9, 3, 0, 18, 6, 0, 27, 9, 0});

    // After a window, the value of a row is its step.
    std::vector<float> steps;
    for (int r = 0; r < 10; ++r) steps.push_back(float(500 + 2 * r));
    ops::Polyline stepped = ops::path_reference(rows.data(), 10, 3, {0, 1, 2}, 3, 4, &steps);
    CHECK(stepped.values == std::vector<float>{500, 506, 512, 518});

    // A column past the input gives nothing.
    CHECK(ops::path_reference(rows.data(), 10, 3, {0, 3, 0}, 2, 4, nullptr).values.empty());
}

TEST_CASE("operators: the program of a placed operator declares the built-in source, a tensor output and its params") {
    sp::Source s;
    s.id = "terrain";
    s.catalog = "live:blocks.0.attn.proj.weight";
    sp::Transform t;
    t.op = sp::TransformOp::surface;
    t.max_cells = 6;
    t.pool = "max";
    s.transforms = {t};
    const sp::Program p = ops::operator_program(s, 0, {4, 6});
    CHECK(p.id == "operator.terrain.0");
    CHECK(p.builtin == std::optional<std::string>("surface@1"));
    CHECK(p.wgsl == ops::builtin_wgsl("surface"));
    REQUIRE(p.inputs.size() == 1);
    CHECK(p.inputs[0].var == "input");
    CHECK(p.inputs[0].source == "live:blocks.0.attn.proj.weight");
    CHECK(p.output.kind == sp::OutputKind::tensor);
    CHECK(p.output.shape == std::vector<std::int64_t>{2, 3});
    CHECK(std::get<double>(p.params.at("factor")) == 2.0);
    CHECK(std::get<double>(p.params.at("pool")) == 1.0);
    REQUIRE(p.workgroups);
    CHECK((*p.workgroups)[0] == 1);
    CHECK(sb::check_declaration(p).ok());
    CHECK(sb::check_params(p).ok());

    // The source compiles, and its bindings fit the declaration.
    if (sb::compiler_available()) {
        for (const sp::Program& q : {p, [&] {
                 sp::Source path = s;
                 path.transforms[0].op = sp::TransformOp::path;
                 path.transforms[0].max_cells.reset();
                 path.transforms[0].pool.reset();
                 path.transforms[0].columns = {3, 1};
                 path.transforms[0].max_points = 4;
                 return ops::operator_program(path, 0, {10, 5});
             }()}) {
            INFO(q.id);
            const sb::Compiled c = sb::compile_and_reflect(q.wgsl);
            INFO(c.verdict.message);
            REQUIRE(c.verdict.ok());
            const sb::Verdict v = sb::check_program(q, c.reflection);
            INFO(v.message);
            CHECK(v.ok());
        }
    } else {
        MESSAGE("no WGSL compiler in this build");
    }

    // A path keeps the columns up to the largest it reads.
    sp::Source trail = s;
    trail.transforms[0].op = sp::TransformOp::path;
    trail.transforms[0].max_cells.reset();
    trail.transforms[0].pool.reset();
    trail.transforms[0].columns = {3, 1};
    trail.transforms[0].max_points = 4;
    const sp::Program q = ops::operator_program(trail, 0, {10, 5});
    CHECK(q.builtin == std::optional<std::string>("path@1"));
    CHECK(q.output.shape == std::vector<std::int64_t>{4, 4});
    CHECK(std::get<double>(q.params.at("stride")) == 3.0);
}
