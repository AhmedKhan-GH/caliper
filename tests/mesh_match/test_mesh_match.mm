// test_mesh_match.mm: the match of the library piece mesh@1 with the built-in mesh primitive
// (design/programmable-graphics.md, Library Pieces).
//
// Each case draws one mesh twice on Metal, offscreen and without a window: once by the scene core as
// a mesh batch (SceneCompiler over BackendResources, with its index clamp), and once by the piece:
// its derive half runs on the same mesh as tensors in the script's producer runtime, and its draw
// half draws the published resources in a DrawRuntime with the camera that the scene view gives a
// program. The two images are held to the tolerances of ../boxes_match/match_rig.h, and the picks
// of the built-in draw at interior pixels of the first, a middle and the last triangle must name
// the vertex that the piece's ID image shows there, with the field `index` of its item map. The
// piece draws its faces through the index buffer `faces`, as the built-in's indexed draw does, and
// leaves the inputs that a mesh lacks (indices, normals, colors) unbound. Flat faces are held to the
// tolerances of the lit cases: the piece's opaque color pass writes no ID, as the built-in's color
// pass writes none, and its last pass writes the IDs at the depth that the color pass wrote. A
// color pass that also wrote the ID lit flat faces otherwise at their shared edges on Metal, in 140
// and 571 pixels by at most 8 levels, with an opaque or an alpha blend alike, while the matrices of
// both draws were the same. Each case prints how many pixels differ. The
// cases cover the color modes (one color, one RGBA for all, values through a colormap, RGBA of each
// vertex), lighting from normals and flat faces, translucency (an alpha below 255 of the one color
// or of a vertex), indices above the last vertex and negative ones, a NaN vertex, the depth offset,
// a mesh in 2-D, one triangle, empty input, and the capacity of the piece: 98,304 vertices of 48
// bytes and 196,608 triangles of 12 bytes of indices, 6.75 MiB of its 8 MiB of published resources
// (the built-in takes 4,194,304 of each). The built-in's texture is not part of the piece yet: a
// texture resource of the piece has one fixed size and format. Without a Metal device the test
// skips (exit 77).
//
// With CALIPER_MESH_MATCH_CAPTURE_DIR set, each case writes mesh-<case>-builtin.png,
// mesh-<case>-piece.png and mesh-<case>-diff.png (the difference scaled up 32 times) there.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#include "../boxes_match/match_rig.h"

namespace {

using namespace match_rig;

constexpr std::uint32_t kSize = 256;
constexpr std::uint32_t kDefaultRgba = 0xffe6dcdcu;
constexpr std::uint32_t kVertexCapacity = 98304;
constexpr std::uint32_t kTriangleCapacity = 196608;

struct Mesh {
    int dims = 3;
    std::vector<float> positions;       // v * dims
    std::vector<std::int32_t> indices;  // t * 3, or none: every three vertices
    std::vector<float> normals;         // v * 3, or none
    std::vector<std::uint8_t> rgba;     // v * 4 or 4, or none
    std::vector<float> values;          // v or 1, or none
    std::uint32_t flat = kDefaultRgba;
    std::uint32_t colormap = 0;
    float vmin = 0.0f, vmax = 1.0f;
    int depth_offset = 0;
    int v() const { return int(positions.size()) / dims; }
    int t() const { return indices.empty() ? v() / 3 : int(indices.size()) / 3; }
};

// A height field of nx by ny vertices over [-1, 1]^2 at height z0, tilted by `tilt` along x, with its
// normals and two triangles a cell.
Mesh grid(int nx, int ny, float z0 = 0.0f, float tilt = 0.0f, int dims = 3) {
    Mesh m;
    m.dims = dims;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const float x = -1.0f + 2.0f * float(i) / float(nx - 1), y = -1.0f + 2.0f * float(j) / float(ny - 1);
            const float z = z0 + tilt * x + 0.3f * std::sin(2.0f * x) * std::cos(2.0f * y);
            m.positions.push_back(x);
            m.positions.push_back(y);
            if (dims == 3) m.positions.push_back(z);
            const float dx = tilt + 0.6f * std::cos(2.0f * x) * std::cos(2.0f * y);
            const float dy = -0.6f * std::sin(2.0f * x) * std::sin(2.0f * y);
            const float len = std::sqrt(dx * dx + dy * dy + 1.0f);
            m.normals.insert(m.normals.end(), {-dx / len, -dy / len, 1.0f / len});
        }
    for (int j = 0; j + 1 < ny; ++j)
        for (int i = 0; i + 1 < nx; ++i) {
            const int a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
            m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
        }
    return m;
}

// RGBA of each vertex at whole levels, with an alpha of `alpha` on every fifth vertex.
void color_vertices(Mesh& m, std::uint8_t alpha) {
    for (int i = 0; i < m.v(); ++i)
        m.rgba.insert(m.rgba.end(), {std::uint8_t((i * 53 + 40) % 256), std::uint8_t((i * 97 + 90) % 256),
                                     std::uint8_t((i * 31 + 200) % 256), i % 5 == 4 ? alpha : std::uint8_t(255)});
}

// The built-in's batch of the mesh, as live_scene.cpp makes it from a SCENE_UPDATE part.
void builtin_mesh(Builtin& bt, const Mesh& m) {
    const int v = m.v();
    sc::Batch batch;
    batch.primitive = sc::Primitive::Mesh;
    batch.flat_rgba = m.flat;
    batch.colormap = m.colormap;
    batch.vmin = m.vmin;
    batch.vmax = m.vmax;
    batch.depth_offset = std::int8_t(m.depth_offset);
    batch.translucent = (m.flat >> 24) < 255;
    Upload up;
    up.add(sc::Role::Positions, m.positions, std::uint32_t(v), std::uint32_t(m.dims * 4),
           m.dims == 3 ? gfx::VertexFormat::F32x3 : gfx::VertexFormat::F32x2);
    if (!m.indices.empty()) up.add(sc::Role::Indices, m.indices, std::uint32_t(m.t()), 12, gfx::VertexFormat::S32);
    if (!m.normals.empty()) up.add(sc::Role::Normals, m.normals, std::uint32_t(v), 12, gfx::VertexFormat::F32x3);
    if (!m.rgba.empty()) {
        up.add(sc::Role::Colors, m.rgba, std::uint32_t(m.rgba.size() / 4), 4, gfx::VertexFormat::Unorm8x4);
        for (std::size_t i = 3; i < m.rgba.size(); i += 4) batch.translucent = batch.translucent || m.rgba[i] < 255;
    }
    if (!m.values.empty()) up.add(sc::Role::Values, m.values, std::uint32_t(m.values.size()), 4, gfx::VertexFormat::F32);
    for (int i = 0; i < v; ++i) {
        double p[3] = {0, 0, 0};
        for (int k = 0; k < m.dims; ++k) p[k] = m.positions[std::size_t(i * m.dims + k)];
        if (std::isfinite(p[0] + p[1] + p[2])) batch.bounds.add({p[0], p[1], p[2]});
    }
    bt.set("mesh", batch, up);
}

std::vector<float> to_unit(const std::vector<std::uint8_t>& b) {
    std::vector<float> f;
    for (std::uint8_t x : b) f.push_back(float(x) / 255.0f);
    return f;
}

// The tensors of the piece's inputs and its params. An input that the mesh lacks stays unbound.
std::vector<HostTensor> piece_tensors(const Mesh& m, caliper_net::Json* params) {
    const std::int64_t v = m.v(), d = m.dims;
    std::vector<HostTensor> t;
    t.push_back(HostTensor::of("positions", "f32", {v, d}, m.positions));
    if (!m.indices.empty()) t.push_back(HostTensor::of("indices", "i32", {std::int64_t(m.t()), 3}, m.indices));
    if (!m.normals.empty()) t.push_back(HostTensor::of("normals", "f32", {v, 3}, m.normals));
    std::uint32_t mode = 0;
    if (!m.rgba.empty()) {
        t.push_back(HostTensor::of("colors", "f32", {std::int64_t(m.rgba.size() / 4), 4}, to_unit(m.rgba)));
        mode = 2;
    } else if (!m.values.empty()) {
        t.push_back(HostTensor::of("colors", "f32", {std::int64_t(m.values.size())}, m.values));
        mode = 1;
    }
    *params = caliper_net::Json::object();
    params->set("color_mode", double(mode));
    params->set("color", double(m.flat));
    params->set("colormap", double(m.colormap));
    params->set("vmin", double(m.vmin));
    params->set("vmax", double(m.vmax));
    params->set("shading", m.normals.empty() ? 0.0 : 1.0);
    params->set("indexed", m.indices.empty() ? 0.0 : 1.0);
    params->set("depth_offset", double(m.depth_offset));
    return t;
}

std::map<std::string, double> as_map(const caliper_net::Json& j) {
    std::map<std::string, double> out;
    for (const char* k : {"color_mode", "color", "colormap", "vmin", "vmax", "shading", "indexed", "depth_offset"})
        if (const caliper_net::Json* v = j.find(k)) out[k] = v->as_double();
    return out;
}

struct Outcome {
    Comparison c;
    std::uint32_t vertices = 0, triangles = 0;   // the valid elements of the published resources
    std::uint32_t flags = 0;
    int picks = 0, asked = 0;
};

const Published& find(const std::vector<Published>& res, const char* id) {
    for (const Published& p : res)
        if (p.id == id) return p;
    FAIL("no published resource ", id);
    return res.front();
}

// Draws the mesh both ways, compares, picks the first vertex of each triangle of `triangles`, and
// writes the captures of the case.
Outcome run(Metal& mt, const std::string& name, const Mesh& m, std::vector<int> triangles) {
    gfx::Backend& g = *mt.backend;
    Outcome o;
    {
        Builtin bt(g, kSize, m.dims == 3 ? sc::Space::k3D : sc::Space::k2D);
        builtin_mesh(bt, m);
        bt.draw();
        REQUIRE(bt.draws.size() == 1);

        cs::Piece piece;
        std::string why;
        REQUIRE_MESSAGE(load_piece("mesh", &piece, &why), why);
        caliper_net::Json params;
        const std::vector<Published> res = run_derive(mt.device, piece, piece_tensors(m, &params), as_map(params));
        REQUIRE(res.size() == 2);
        const Published& vertices = find(res, "vertices");
        const Published& faces = find(res, "faces");
        o.vertices = vertices.count;
        CHECK(faces.count % 3 == 0);
        o.triangles = faces.count / 3;
        if (vertices.count > 0) {
            float flags = 0;   // vertex 0's normal.w
            std::memcpy(&flags, vertices.bytes.data() + 28, 4);
            o.flags = std::uint32_t(flags);
        }

        PieceDraw pd(g, kSize);
        pd.install(piece, "mesh");
        pd.set("vertices", vertices.bytes, vertices.count);
        pd.set("faces", faces.bytes, faces.count);
        cd::FrameInputs in = bt.frame_inputs();
        in.params = caliper_web::pack_params(piece, params);
        CHECK(pd.draw(in));

        const bool flat = m.normals.empty();
        o.c = compare(pd.color, pd.ids, bt.image, kSize, flat);
        MESSAGE("mesh ", name, ": ", m.v(), " vertices and ", m.t(), " triangles, ", o.vertices, " and ", o.triangles,
                " published, flags ", o.flags, "; ", o.c.lit, " pixels of the mesh, ", o.c.interior, " interior pixels (",
                o.c.lit_interior, " lit), ",
                o.c.edge, " edge pixels; ", o.c.differ, " differ, largest gap ", o.c.max_interior, " inside and ",
                o.c.max_edge, " at edges, ", o.c.edge_far, " edge pixels over 1 level; ", o.c.failed,
                " outside the tolerances");
        CHECK_MESSAGE(o.c.failed == 0, name, ": ", o.c.failed, " pixels outside the tolerances; ", o.c.first);
        std::vector<std::uint32_t> rows(faces.count);
        if (!rows.empty()) std::memcpy(rows.data(), faces.bytes.data(), rows.size() * 4);
        for (int t : triangles) {
            if (t < 0 || std::uint32_t(t) >= o.triangles) continue;
            // A triangle whose first vertex shows no interior pixel, as one on the far side, gives
            // its place to the next one that shows one.
            std::uint32_t at = std::uint32_t(t);
            while (at + 1 < o.triangles && at < std::uint32_t(t) + 64 && !pixel_of_item(pd.ids, kSize, rows[std::size_t(at) * 3]))
                ++at;
            if (at != std::uint32_t(t)) MESSAGE("triangle ", t, " shows no interior pixel; triangle ", at, " is picked");
            ++o.asked;
            o.picks += check_picks(bt, pd, {rows[std::size_t(at) * 3]}, sc::Primitive::Mesh);
        }
        if (o.asked) MESSAGE("mesh ", name, ": ", o.picks, " of ", o.asked, " picks name the piece's item and field");

        capture("CALIPER_MESH_MATCH_CAPTURE_DIR", "mesh-" + name + "-builtin.png", kSize, bt.image);
        capture("CALIPER_MESH_MATCH_CAPTURE_DIR", "mesh-" + name + "-piece.png", kSize, pd.color);
        capture("CALIPER_MESH_MATCH_CAPTURE_DIR", "mesh-" + name + "-diff.png", kSize, difference(pd.color, bt.image));
    }
    return o;
}

}  // namespace

TEST_CASE("mesh_match: mesh@1 passes the checks of piece.check") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("mesh", &piece, &why), why);
    check_piece(piece);
}

TEST_CASE("mesh_match: a lit mesh with RGBA of each vertex and a depth offset") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m = grid(24, 24);
    color_vertices(m, 255);
    m.depth_offset = 3;
    const Outcome o = run(mt, "lit_rgba", m, {0, m.t() / 2, m.t() - 1});
    CHECK(o.vertices == 576);
    CHECK(o.triangles == 1058);
    CHECK(o.flags == 0);
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
    CHECK(o.asked == 3);
}

TEST_CASE("mesh_match: flat faces with values through magma") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m = grid(24, 24);
    m.normals.clear();
    for (int i = 0; i < m.v(); ++i) m.values.push_back(float((i * 37) % 256) / 255.0f);
    m.colormap = 1;
    const Outcome o = run(mt, "flat_values", m, {0, m.t() / 2, m.t() - 1});
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("mesh_match: one RGBA for all vertices, lit, and one color with flat faces") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m = grid(16, 16);
    m.rgba = {200, 170, 120, 255};
    const Outcome o = run(mt, "one_rgba", m, {0, m.t() - 1});
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
    Mesh f = grid(16, 16);
    f.normals.clear();
    f.flat = 0xff3080c0u;
    const Outcome p = run(mt, "flat_color", f, {0, f.t() - 1});
    CHECK(p.c.lit >= 2000);
    CHECK(p.picks == p.asked);
}

TEST_CASE("mesh_match: translucent meshes by the alpha of a vertex and of the one color") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    // Two sheets that cross: the upper one tilted through the lower one.
    Mesh m = grid(20, 20);
    Mesh upper = grid(20, 20, 0.1f, 0.5f);
    const int base = m.v();
    m.positions.insert(m.positions.end(), upper.positions.begin(), upper.positions.end());
    m.normals.insert(m.normals.end(), upper.normals.begin(), upper.normals.end());
    for (int i : upper.indices) m.indices.push_back(i + base);
    color_vertices(m, 140);
    m.normals.clear();
    const Outcome o = run(mt, "translucent", m, {0, m.t() / 2, m.t() - 1});
    CHECK(o.flags == 1);
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
    Mesh f = m;
    f.rgba.clear();
    f.flat = 0x9040a0e0u;
    const Outcome p = run(mt, "flat_alpha", f, {0, f.t() - 1});
    CHECK(p.c.lit >= 2000);
    CHECK(p.picks == p.asked);
}

TEST_CASE("mesh_match: indices above the last vertex and negative ones, and a NaN vertex") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m = grid(12, 12);
    color_vertices(m, 255);
    m.indices[5] = 100000;   // the last vertex
    m.indices[40] = -7;      // the last vertex too
    m.positions[std::size_t(3 * 30 + 2)] = std::nanf("");
    const Outcome o = run(mt, "clamp_nan", m, {0, m.t() / 2, m.t() - 1});
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("mesh_match: a mesh in 2-D with flat faces") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m = grid(10, 10, 0.0f, 0.0f, 2);
    m.normals.clear();
    for (int i = 0; i < m.v(); ++i) m.values.push_back(float((i * 11) % 256) / 255.0f);
    const Outcome o = run(mt, "2d", m, {0, m.t() / 2, m.t() - 1});
    CHECK(o.c.lit >= 2000);
    CHECK(o.picks == o.asked);
}

TEST_CASE("mesh_match: one triangle") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    Mesh m;
    m.positions = {-0.8f, -0.6f, 0.0f, 0.9f, -0.4f, 0.3f, 0.0f, 0.8f, -0.2f};
    m.flat = 0xff40c0f0u;
    const Outcome o = run(mt, "one", m, {0});
    CHECK(o.vertices == 3);
    CHECK(o.triangles == 1);
    CHECK(o.picks == 1);
}

TEST_CASE("mesh_match: no mesh draws nothing") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    gfx::Backend& g = *mt.backend;
    // The script's runtime imports no tensor without bytes (producer.mm refuses it), thus a viewer
    // gets no version of the triangles, or a version without a valid element; the piece then draws
    // nothing, as the built-in draws nothing for a mesh without vertices, which removes the batch.
    Builtin bt(g, kSize, sc::Space::k3D);
    bt.draw();
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("mesh", &piece, &why), why);
    PieceDraw pd(g, kSize);
    pd.install(piece, "mesh");
    cd::FrameInputs in;
    in.clip_from_world = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    in.viewport = {0, 0, float(kSize), float(kSize)};
    in.params = caliper_web::pack_params(piece, caliper_net::Json::object());
    CHECK_FALSE(pd.draw(in));
    CHECK(compare(pd.color, pd.ids, bt.image, kSize).differ == 0);
    pd.set("faces", {}, 0);
    CHECK_FALSE(pd.draw(in));
    const Comparison c = compare(pd.color, pd.ids, bt.image, kSize);
    CHECK(c.differ == 0);
    MESSAGE("mesh empty: ", c.differ, " pixels differ from the built-in's empty scene");
}

TEST_CASE("mesh_match: the capacity of 98,304 vertices and 196,608 triangles, and the first of a larger mesh") {
    Metal mt;
    if (!mt.backend) CALIPER_SKIP_NO_GPU(mt.why);
    // A grid of 313 by 313 vertices, 97,969 of them and 194,688 triangles, padded with vertices and
    // copies of a triangle up to the capacities. The first and the last triangles are large and in
    // front, thus a pick finds the first and the last vertex.
    Mesh m = grid(313, 313);
    while (m.v() < int(kVertexCapacity)) {
        m.positions.insert(m.positions.end(), {0.0f, 0.0f, 0.0f});
        m.normals.insert(m.normals.end(), {0.0f, 0.0f, 1.0f});
    }
    const int v = m.v();
    auto place = [&](int i, float x, float y) {
        m.positions[std::size_t(3 * i)] = x;
        m.positions[std::size_t(3 * i + 1)] = y;
        m.positions[std::size_t(3 * i + 2)] = 0.9f;
    };
    place(0, -0.9f, -0.9f);
    place(v - 3, -0.1f, -0.9f);
    place(v - 2, -0.9f, -0.1f);
    place(v - 1, 0.9f, 0.9f);
    place(v - 4, 0.1f, 0.9f);
    place(v - 5, 0.9f, 0.1f);
    color_vertices(m, 255);
    m.indices[0] = 0;
    m.indices[1] = v - 3;
    m.indices[2] = v - 2;
    while (m.t() < int(kTriangleCapacity) - 1) m.indices.insert(m.indices.end(), {0, 1, 314});
    m.indices.insert(m.indices.end(), {v - 1, v - 4, v - 5});
    const Outcome o = run(mt, "capacity", m, {0, int(kTriangleCapacity) - 1});
    CHECK(o.vertices == kVertexCapacity);
    CHECK(o.triangles == kTriangleCapacity);
    CHECK(o.picks == o.asked);
    CHECK(o.asked == 2);

    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("mesh", &piece, &why), why);
    Mesh over = m;
    over.positions.insert(over.positions.end(), {0.0f, 0.0f, 0.0f});
    over.normals.insert(over.normals.end(), {0.0f, 0.0f, 1.0f});
    over.rgba.insert(over.rgba.end(), {1, 2, 3, 255});
    over.indices.insert(over.indices.end(), {0, 1, int(kVertexCapacity)});
    caliper_net::Json params;
    const std::vector<Published> res = run_derive(mt.device, piece, piece_tensors(over, &params), as_map(params));
    CHECK(find(res, "vertices").count == kVertexCapacity);
    CHECK(find(res, "faces").count == 3 * kTriangleCapacity);
}

CALIPER_GPU_TEST_MAIN()

