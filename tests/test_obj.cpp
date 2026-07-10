#include <doctest/doctest.h>
#include <caliper/adapters/obj.hpp>

#include <algorithm>
#include <sstream>
#include <string>

TEST_CASE("obj adapter triangulates fans and resolves negative indices") {
    std::istringstream input(R"obj(
v -1 -2 -3
v  1 -2 -3
v  1  2  3
v -1  2  3
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1
)obj");
    caliper::obj::Mesh mesh;
    std::string error;
    REQUIRE(caliper::obj::load(input, mesh, &error));
    CHECK(error.empty());
    CHECK(mesh.vertex_count() == 4);
    CHECK(mesh.triangle_count() == 2);
    CHECK(mesh.indices == std::vector<int32_t>{0, 1, 2, 0, 2, 3});
    CHECK(mesh.has_uvs);
    CHECK(mesh.has_normals);
    CHECK(*std::min_element(mesh.positions.begin(), mesh.positions.end()) == -3.f);
    CHECK(*std::max_element(mesh.positions.begin(), mesh.positions.end()) == 3.f);
}

TEST_CASE("obj adapter deduplicates the full position/uv/normal key") {
    std::istringstream input(R"obj(
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vt 0.5 0.5
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
f 1/4/1 3/3/1 2/2/1
)obj");
    caliper::obj::Mesh mesh;
    REQUIRE(caliper::obj::load(input, mesh));
    CHECK(mesh.vertex_count() == 4);
    CHECK(mesh.triangle_count() == 2);
    CHECK(mesh.indices[0] != mesh.indices[3]);
    CHECK(mesh.positions[0] == mesh.positions[9]);
    CHECK(mesh.uvs[0] != mesh.uvs[6]);
}

TEST_CASE("obj adapter zero-fills missing streams and reports their absence") {
    std::istringstream input("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    caliper::obj::Mesh mesh;
    REQUIRE(caliper::obj::load(input, mesh));
    CHECK_FALSE(mesh.has_uvs);
    CHECK_FALSE(mesh.has_normals);
    CHECK(mesh.uvs == std::vector<float>(6, 0.f));
    CHECK(mesh.normals == std::vector<float>(9, 0.f));
}

TEST_CASE("obj adapter rejects malformed input without partial output") {
    for (const char* source : {
             "v 0 0 0\nv 1 0 0\nf 1 2\n",
             "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
             "v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
             "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3\n",
         }) {
        std::istringstream input(source);
        caliper::obj::Mesh mesh;
        mesh.positions.push_back(123.f);
        std::string error;
        CHECK_FALSE(caliper::obj::load(input, mesh, &error));
        CHECK(mesh.positions.empty());
        CHECK_FALSE(error.empty());
    }
}

TEST_CASE("obj adapter loads the committed TwinScope housing") {
    // Charted, watertight finned heatsink (procedurally generated, generator not
    // shipped). Stats from the generator's self-verification: 3184 triangles,
    // 2430 loader vertices, 30 charts, 6-texel min gutter @256, closed 2-manifold.
    caliper::obj::Mesh mesh;
    std::string error;
    const std::string path = std::string(CALIPER_TEST_SOURCE_ROOT) +
        "/applets/twin_scope/assets/housing.obj";
    REQUIRE_MESSAGE(caliper::obj::load_file(path, mesh, &error), error);
    CHECK(mesh.has_uvs);
    CHECK(mesh.has_normals);
    // Exact pins for the committed asset (generator self-verification, confirmed
    // from the loader): 1592 quads triangulate to 3184 triangles, and the
    // (v,vt,vn)-dedup yields 2430 vertices (every face pairs v==vt, one of 6
    // normals). Spec envelope is 2,500-4,000 triangles.
    CHECK(mesh.triangle_count() == 3184);
    CHECK(mesh.vertex_count() == 2430);
    // Vertices are deduplicated on the (v,vt,vn) triple: far fewer than 3 per tri.
    CHECK(mesh.vertex_count() < mesh.triangle_count() * 3);
    CHECK(mesh.positions.size() == mesh.vertex_count() * 3);
    CHECK(mesh.normals.size() == mesh.vertex_count() * 3);
    CHECK(mesh.uvs.size() == mesh.vertex_count() * 2);
    // Charted atlas UVs: strictly inside [0,1] (gutter-inset, no chart touches an edge).
    const float uv_min = *std::min_element(mesh.uvs.begin(), mesh.uvs.end());
    const float uv_max = *std::max_element(mesh.uvs.begin(), mesh.uvs.end());
    CHECK(uv_min >= 0.f);
    CHECK(uv_max <= 1.f);
    CHECK(uv_min > 0.f);
    CHECK(uv_max < 1.f);
}
