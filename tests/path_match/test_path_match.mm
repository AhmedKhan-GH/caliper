// test_path_match.mm: the match of the library piece path@1 with the built-in draw of the path
// operator's output (design/programmable-graphics.md, Library Pieces; design/instrument-interface.md,
// Path Operator).
//
// The path operator turns the rows of a tensor into a polyline, which a lines entity in strip mode
// draws: one strip through the kept rows, colored by each row's value through the colormap, with
// the entity's range (the finite range of the values, widened by 0.5 when empty, unless vmin or
// vmax is given), a width of 2 pixels and a depth offset of 1. Each case draws that twice on
// Metal, offscreen: the scene core draws a lines batch of the operator's CPU reference
// (path_reference), and the piece reads the same tensor end to end, its derive half on the
// producer runtime and its draw half in a draw runtime with the camera matrix of the built-in
// draw. The entity's data box, which scales the positions of a spec panel, is not part of either.
// The cases cover the default columns of three and of two columns, chosen columns and z 0, the
// thinning of max_points, one color and a translucent one, fixed and partly fixed ranges, a NaN
// coordinate, a pixel ratio of 2, one row, two rows, and the most vertices that the piece
// publishes (1,048,576, the operator's most); above them the piece
// thins as the operator does with max_points 1,048,576. ../lines_match/line_match.h holds the
// tolerances and the picks.
#include "../lines_match/line_match.h"

#include "scene/scene.h"

#include <caliper_builder/operators/operators.h>
#include <caliper_builder/programs/operator_kernels.h>

namespace {

using namespace line_match;
namespace ops = caliper_builder::operators;

struct Lcg {
    std::uint32_t state;
    explicit Lcg(std::uint32_t seed) : state(seed) {}
    float next() {
        state = state * 1664525u + 1013904223u;
        return float(double(state >> 8) / double(1u << 24)) * 2.0f - 1.0f;
    }
};

std::uint32_t rgba(std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a) {
    return r | g << 8 | b << 16 | a << 24;
}

// One case: a tensor of rows and the options of the operator and the entity.
struct Path {
    std::int64_t rows = 0, cols = 3;
    std::vector<float> data;
    std::array<int, 3> columns{0, 1, 2};
    int ncolumns = 3;                      // 2: z is 0
    std::int64_t max_points = 65536;
    float width = 2.0f;
    bool colormapped = true;
    std::uint32_t flat = sc::kDefaultRgba;
    std::uint32_t colormap = 0;
    std::uint32_t range = 3;               // bit 0: vmin from the values, bit 1: vmax
    float vmin = 0.0f, vmax = 1.0f;
    int depth_offset = 1;
};

// The operator's output and the entity's range of it.
struct Reference {
    ops::Polyline line;
    std::int64_t stride = 1;
    float vmin = 0, vmax = 1;
};

Reference reference_of(const Path& p) {
    Reference r;
    r.stride = caliper_builder::programs::operator_kernels::path_stride(p.rows, p.max_points);
    r.line = ops::path_reference(p.data.data(), p.rows, p.cols, p.columns, p.ncolumns, p.max_points, nullptr);
    double lo = 0, hi = 1;
    bool any = false;
    for (float v : r.line.values)
        if (std::isfinite(v)) {
            lo = any ? std::min(lo, double(v)) : v;
            hi = any ? std::max(hi, double(v)) : v;
            any = true;
        }
    if (!any) lo = 0, hi = 1;
    if (lo == hi) lo -= 0.5, hi += 0.5;
    r.vmin = (p.range & 1u) ? float(lo) : p.vmin;
    r.vmax = (p.range & 2u) ? float(hi) : p.vmax;
    return r;
}

// The built-in lines batch of a lines entity in strip mode over the operator's output.
void add_builtin(Builtin& b, const Path& p, const Reference& r) {
    const std::uint32_t n = std::uint32_t(r.line.values.size());
    sc::Batch batch;
    batch.primitive = sc::Primitive::Lines;
    batch.line_mode = sc::LineMode::Strips;
    batch.width_px = p.width;
    batch.flat_rgba = p.flat;
    batch.colormap = p.colormap;
    batch.vmin = r.vmin;
    batch.vmax = r.vmax;
    batch.depth_offset = std::int8_t(p.depth_offset);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float* q = r.line.positions.data() + std::size_t(i) * 3;
        if (std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])) batch.bounds.add({q[0], q[1], q[2]});
    }
    std::vector<std::uint32_t> mask(n, 1u);
    if (n) mask[n - 1] = 0u;
    std::vector<Array> arrays;
    arrays.push_back(array(sc::Role::Positions, r.line.positions, n, 12, gfx::VertexFormat::F32x3));
    arrays.push_back(array(sc::Role::BreakMask, mask, n, 4, gfx::VertexFormat::U8));
    if (p.colormapped) arrays.push_back(array(sc::Role::Values, r.line.values, n, 4, gfx::VertexFormat::F32));
    b.add("path", batch, arrays);
}

std::vector<Tensor> tensors_of(const Path& p) {
    return {tensor("rows", p.data, {p.rows, p.cols})};
}

Json params_of(const Path& p) {
    Json v = Json::object();
    v.set("x", std::int64_t(p.columns[0]));
    v.set("y", std::int64_t(p.columns[1]));
    v.set("z", std::int64_t(p.ncolumns == 3 ? p.columns[2] : -1));
    v.set("max_points", std::uint64_t(p.max_points));
    v.set("width", double(p.width));
    v.set("color_mode", std::uint64_t(p.colormapped ? 1 : 0));
    v.set("color", std::uint64_t(p.flat));
    v.set("colormap", std::uint64_t(p.colormap));
    v.set("range", std::uint64_t(p.range));
    v.set("vmin", double(p.vmin));
    v.set("vmax", double(p.vmax));
    v.set("depth_offset", std::int64_t(p.depth_offset));
    return v;
}

// The item map of vertex k: its row, its value and its position.
Fields vertex_of(const Reference& r, std::uint32_t k) {
    const float* q = r.line.positions.data() + std::size_t(k) * 3;
    return {{"row", double(std::int64_t(k) * r.stride)},
            {"value", double(r.line.values[k])},
            {"x", double(q[0])},
            {"y", double(q[1])},
            {"z", double(q[2])}};
}

Comparison run_case(Metal& m, const std::string& name, const Path& p, const std::vector<std::uint32_t>& items,
                    std::size_t min_lit, float pixel_ratio = 1.0f, const std::string& capture_name = "") {
    CAPTURE(name);
    const Reference r = reference_of(p);
    Builtin b(*m.backend, pixel_ratio);
    add_builtin(b, p, r);
    const std::array<float, 16> mvp = b.fit();
    const std::vector<std::uint8_t> builtin = b.draw();
    Program prog;
    make_program(m, "path", tensors_of(p), params_of(p), kPick, &prog);
    const Published* v = find_published(prog.published, "vertices");
    REQUIRE(v != nullptr);
    CHECK_MESSAGE(v->count == r.line.values.size(), name, ": ", v->count, " vertices, the operator keeps ",
                  r.line.values.size());
    const Drawn drawn = draw_programs(*m.backend, {&prog}, mvp, pixel_ratio);
    return check_case(name, b, builtin, drawn, prog, items, [&](std::uint32_t k) { return vertex_of(r, k); }, min_lit,
                      capture_name);
}

// R rows of D columns: most of a knot through the box in columns 0 to 2, whose ends lie apart,
// and other columns a ramp.
Path knot(std::int64_t rows, std::int64_t cols, std::uint32_t seed) {
    Path p;
    p.rows = rows;
    p.cols = cols;
    Lcg g(seed);
    for (std::int64_t i = 0; i < rows; ++i) {
        const float t = 5.5f * float(i) / float(std::max<std::int64_t>(rows, 2) - 1);
        for (std::int64_t c = 0; c < cols; ++c) {
            float v = 0.002f * g.next();
            if (c == 0) v += 0.8f * std::sin(2.0f * t) + 0.2f * std::cos(5.0f * t);
            else if (c == 1) v += 0.8f * std::cos(3.0f * t);
            else if (c == 2) v += 0.6f * std::sin(5.0f * t + 0.3f);
            else v += float(c) * 0.1f + 0.5f * std::sin(t * float(c));
            p.data.push_back(v);
        }
    }
    return p;
}

}  // namespace

TEST_CASE("path_match: path@1 passes piece.check, and the pixel comparison carries it as the library has it") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("path", &piece, &why), why);
    check_piece(piece);
    check_pinned(piece, dp::kPathManifest, dp::kPathHash);
}

TEST_CASE("path_match: the path through the rows draws as the built-in strip of the operator's output") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    {   // 40 rows of 3 columns, long joins to pick, through viridis with the range of the rows.
        Path few = knot(40, 3, 9);
        few.width = 6.0f;
        run_case(m, "40 rows, the default columns", few, {0, 20, 38}, 300, 1.0f, "path-match");
        // 300 rows, every row kept.
        Path p = knot(300, 3, 1);
        p.width = 4.0f;
        run_case(m, "300 rows, the default columns", p, {}, 100);
        // The entity's default width of 2 pixels, which has no interior pixel.
        Path d = p;
        d.width = 2.0f;
        run_case(m, "300 rows, the default width", d, {}, 0);
        // A translucent color, wide.
        Path t = p;
        t.colormapped = false;
        t.flat = rgba(250, 160, 60, 120);
        t.width = 7.0f;
        run_case(m, "300 rows, one translucent color", t, {}, 100);
        // Fixed ends through greys (a span of 512, which divides exactly), and a fixed start with the
        // end from the rows through rdbu.
        Path f = p;
        f.range = 0;
        f.vmin = -128.0f;
        f.vmax = 384.0f;
        f.colormap = 3;
        run_case(m, "300 rows, fixed ends", f, {}, 100);
        Path h = p;
        h.range = 2;
        h.vmin = -50.0f;
        h.colormap = 2;
        run_case(m, "300 rows, a fixed start", h, {}, 100);
        // Width 1.5 at a pixel ratio of 2.
        Path w = p;
        w.width = 1.5f;
        run_case(m, "300 rows, width 1.5 at a pixel ratio of 2", w, {}, 1, 2.0f);
    }
    {   // 1,100 rows thinned to at most 50: every 22nd row, through magma.
        Path p = knot(1100, 3, 2);
        p.max_points = 50;
        p.colormap = 1;
        p.width = 4.0f;
        run_case(m, "1,100 rows, max_points 50", p, {0, 25, 48}, 300);
    }
    {   // Two columns: z is 0. Five columns with the columns 4, 0 and 2, and with z dropped.
        Path two = knot(200, 2, 3);
        two.ncolumns = 2;
        two.columns = {0, 1, 0};
        two.width = 4.0f;
        run_case(m, "200 rows of 2 columns", two, {}, 100);
        Path five = knot(200, 5, 4);
        five.columns = {4, 0, 2};
        five.width = 4.0f;
        run_case(m, "200 rows of 5 columns, columns 4, 0 and 2", five, {}, 100);
        Path flat = five;
        flat.columns = {1, 3, 0};
        flat.ncolumns = 2;
        run_case(m, "200 rows of 5 columns, columns 1 and 3, z 0", flat, {}, 50);
    }
}

TEST_CASE("path_match: NaN, one and two rows, and the most vertices") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    {   // A NaN coordinate hides the two joins of its row.
        Path p = knot(40, 3, 5);
        p.data[std::size_t(20) * 3 + 1] = NAN;
        p.width = 6.0f;
        run_case(m, "a NaN coordinate", p, {0, 38}, 300);
    }
    {   // Two rows draw one join; one row draws nothing.
        Path two;
        two.rows = 2;
        two.data = {-0.7f, -0.4f, 0.0f, 0.6f, 0.5f, 0.2f};
        two.width = 8.0f;
        run_case(m, "two rows", two, {0}, 300);
        Path one;
        one.rows = 1;
        one.data = {0.1f, 0.2f, 0.3f};
        const Reference r = reference_of(one);
        Builtin b(*m.backend);
        add_builtin(b, one, r);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program prog;
        make_program(m, "path", tensors_of(one), params_of(one), kPick, &prog);
        CHECK(find_published(prog.published, "vertices")->count == 1);
        const Drawn drawn = draw_programs(*m.backend, {&prog}, mvp);
        const Comparison c = compare(builtin, drawn.color, drawn.ids);
        CHECK(c.differ == 0);
        std::size_t lit = 0;
        for (int y = 0; y < int(kSize); ++y)
            for (int x = 0; x < int(kSize); ++x) lit += pixel_of(drawn.color, x, y) != kClear;
        CHECK(lit == 0);
        MESSAGE("one row: 1 vertex, ", lit, " pixels lit, ", c.differ, " pixels differ");
    }
    {   // 1,048,576 rows, all kept with max_points 1,048,576: the most that the piece publishes.
        constexpr std::int64_t kMost = 1048576;
        Path p = knot(kMost, 3, 6);
        p.max_points = 1048576;
        p.colormapped = false;   // a value of each of 1,048,576 rows can sit on a boundary of two levels
        p.flat = rgba(120, 220, 160, 255);
        p.width = 3.0f;
        // Each join is below a pixel here, thus no element shows an interior pixel to pick.
        run_case(m, "1,048,576 rows", p, {}, 0);
        // One row more: the piece keeps every second row, as the operator does with max_points
        // 1,048,576.
        Path more = knot(kMost + 1, 3, 7);
        more.max_points = 1048576;
        more.colormapped = false;
        more.flat = p.flat;
        more.width = 3.0f;
        Path asked = more;
        asked.max_points = 1048576;
        const Reference r = reference_of(more);
        CHECK(r.line.values.size() == 524289);
        Builtin b(*m.backend);
        add_builtin(b, more, r);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program prog;
        make_program(m, "path", tensors_of(asked), params_of(asked), kPick, &prog);
        CHECK(find_published(prog.published, "vertices")->count == 524289);
        const Drawn drawn = draw_programs(*m.backend, {&prog}, mvp);
        check_case("1,048,577 rows with max_points 1,048,576", b, builtin, drawn, prog, {},
                   [&](std::uint32_t k) { return vertex_of(r, k); }, 0);
    }
}

CALIPER_GPU_TEST_MAIN()
