// The scene store: the path grammar, the limits, the entity arrays, parts, clears and static
// items, and batch descriptions (src/host/scene/scene.h).
#include "scene/scene.h"
#include "scene_fixtures.h"

#include <ostream>  // doctest prints a std::string_view operand, which needs the complete stream type on MSVC

#include <doctest/doctest.h>

#include <string>

using namespace caliper_scene;
using namespace scene_fixtures;

TEST_CASE("scene paths follow the grammar of scenes.md") {
    CHECK(valid_path(""));
    CHECK(valid_path("world"));
    CHECK(valid_path("rig/left/rgb"));
    CHECK(valid_path("a_b.c-d/E9"));
    CHECK(valid_path(std::string(63, 'a')));
    CHECK_FALSE(valid_path(std::string(64, 'a')));
    std::string why;
    CHECK_FALSE(valid_path("a/./b", &why));
    CHECK(why.find("relative") != std::string::npos);
    CHECK_FALSE(valid_path(".."));
    CHECK_FALSE(valid_path("/a"));
    CHECK_FALSE(valid_path("a/"));
    CHECK_FALSE(valid_path("a//b"));
    CHECK_FALSE(valid_path("a b"));
    CHECK_FALSE(valid_path("a/b:c"));
    CHECK_FALSE(valid_path("caf\xc3\xa9"));
    // 16 segments pass, 17 do not.
    std::string p16 = "s";
    for (int i = 1; i < 16; ++i) p16 += "/s";
    CHECK(valid_path(p16));
    CHECK_FALSE(valid_path(p16 + "/s"));
    // 255 bytes pass, 256 do not.
    std::string long_path;
    while (long_path.size() < 255) long_path += long_path.empty() ? std::string(63, 'x') : "/" + std::string(63, 'x');
    long_path.resize(255);
    CHECK(valid_path(long_path));
    CHECK_FALSE(valid_path(long_path + "y"));
    CHECK(parent_path("a/b/c") == "a/b");
    CHECK(parent_path("a") == "");
    CHECK(path_depth("a/b/c") == 3);
}

TEST_CASE("ensure creates missing parents, keeps each parent before its children, and orders by path bytes") {
    SceneTree t;
    CHECK(t.entity_count() == 1);
    const EntityId c = t.ensure("arm/link0/link1");
    REQUIRE(c);
    CHECK(t.entity_count() == 4);
    CHECK(t.path(c) == "arm/link0/link1");
    const EntityId b = t.parent(c);
    CHECK(t.path(b) == "arm/link0");
    CHECK(b.index < c.index);
    CHECK(t.parent(t.parent(b)) == t.root());
    CHECK(t.ensure("arm/link0") == b);
    t.ensure("arm-x");
    t.ensure("a");
    // Byte order: "" < "a" < "arm" < "arm-x" < "arm/link0" ('-' is 0x2d, '/' 0x2f).
    std::vector<std::string> order;
    for (uint32_t i : t.path_order()) order.push_back(t.path(t.entity_at(i)));
    CHECK(order == std::vector<std::string>{"", "a", "arm", "arm-x", "arm/link0", "arm/link0/link1"});
    std::string err;
    CHECK_FALSE(t.ensure("bad//path", &err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("a scene holds at most 1,024 entities, the root included") {
    SceneTree t;
    std::string err;
    for (int i = 1; i < 1024; ++i) REQUIRE(t.ensure("e" + std::to_string(i), &err));
    CHECK(t.entity_count() == 1024);
    CHECK_FALSE(t.ensure("one_more", &err));
    CHECK(err.find("1,024") != std::string::npos);
    // A path whose parents would pass the limit creates nothing.
    CHECK(t.entity_count() == 1024);
    CHECK(t.find("e5"));
    CHECK(t.ensure("e5"));   // an existing entity is still found
}

TEST_CASE("a reset empties the scene and old ids die") {
    SceneTree t;
    const EntityId e = t.ensure("a/b");
    CHECK(t.valid(e));
    t.reset();
    CHECK(t.entity_count() == 1);
    CHECK_FALSE(t.valid(e));
    CHECK(t.valid(t.root()));
    std::string err;
    CHECK_FALSE(t.set_transform(e, Mat4{}, false, &err));
}

TEST_CASE("parts: the root holds no transform or camera; cameras and primitives exclude each other") {
    SceneTree t;
    std::string err;
    CHECK_FALSE(t.set_transform(t.root(), Mat4{}, false, &err));
    CHECK_FALSE(t.set_camera(t.root(), pinhole(), false, &err));
    const EntityId cam = t.ensure("rig/left");
    REQUIRE(t.set_camera(cam, pinhole(), false, &err));
    CHECK_FALSE(t.set_batch(cam, points3(4), false, &err));
    CHECK(err.find("camera") != std::string::npos);
    const EntityId pts = t.ensure("cloud");
    REQUIRE(t.set_batch(pts, points3(4), false, &err));
    CHECK_FALSE(t.set_camera(pts, pinhole(), false, &err));
    // A camera's children are 2-D: 3-D data there is refused, 2-D data accepted.
    const EntityId child = t.ensure("rig/left/marks");
    CHECK_FALSE(t.set_batch(child, points3(4), false, &err));
    CHECK(t.set_batch(child, points2(4), false, &err));
    // No camera inside a pixel space, and none above 3-D data.
    CHECK_FALSE(t.set_camera(t.ensure("rig/left/marks/cam"), pinhole(), false, &err));
    t.ensure("group/cloud");
    REQUIRE(t.set_batch(t.find("group/cloud"), points3(3), false, &err));
    CHECK_FALSE(t.set_camera(t.find("group"), pinhole(), false, &err));
    // A 2-D scene holds no camera, and takes 2-D data at its root.
    SceneTree flat(Space::k2D);
    CHECK_FALSE(flat.set_camera(flat.ensure("c"), pinhole(), false, &err));
    CHECK_FALSE(flat.set_batch(flat.root(), points3(2), false, &err));
    CHECK(flat.set_batch(flat.root(), points2(2), false, &err));
}

TEST_CASE("up axes and units") {
    SceneTree t(Space::k3D);
    CHECK(t.up() == Axis::PosY);
    CHECK(t.set_up(Axis::NegZ));
    SceneTree flat(Space::k2D);
    CHECK(flat.up() == Axis::NegY);
    std::string err;
    CHECK_FALSE(flat.set_up(Axis::PosZ, &err));
    CHECK(flat.set_up(Axis::PosY));
    CHECK(t.set_units("mm"));
    CHECK_FALSE(t.set_units("sixteen-bytes-xx", &err));
    Axis a;
    CHECK(parse_axis("-x", &a));
    CHECK(a == Axis::NegX);
    CHECK_FALSE(parse_axis("y", &a));
}

TEST_CASE("batches describe arrays and are validated, never copied") {
    SceneTree t;
    std::string err;
    const EntityId e = t.ensure("p");
    Batch b = points3(10);
    CHECK(t.set_batch(e, b, false, &err));
    const Batch* got = t.batch(e, Primitive::Points);
    REQUIRE(got);
    CHECK(got->array(Role::Positions)->buffer == b.array(Role::Positions)->buffer);
    CHECK(got->count() == 10);

    Batch bad = b;
    bad.arrays[0].offset = 6;
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    CHECK(err.find("multiple of 4") != std::string::npos);
    bad = b;
    bad.arrays[0].stride = 2052;
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    bad = b;
    bad.arrays.push_back(array(Role::Colors, 10, 4, caliper_gfx::VertexFormat::Unorm8x4));
    bad.arrays.push_back(array(Role::Values, 10, 4, caliper_gfx::VertexFormat::F32));
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    CHECK(err.find("together") != std::string::npos);
    bad = b;
    bad.arrays.push_back(array(Role::Colors, 7, 4, caliper_gfx::VertexFormat::Unorm8x4));
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    bad = b;
    bad.arrays.push_back(array(Role::Colors, 1, 0, caliper_gfx::VertexFormat::Unorm8x4));
    CHECK(t.set_batch(e, bad, false, &err));   // one colour for all
    bad = b;
    bad.arrays.push_back(array(Role::Normals, 10, 12, caliper_gfx::VertexFormat::F32x3));
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    CHECK(err.find("takes no normals") != std::string::npos);
    bad = b;
    bad.arrays[0].count = kMaxItems + 1;
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    bad = b;
    bad.labels.assign(3, "x");
    CHECK_FALSE(t.set_batch(e, bad, false, &err));
    bad.labels.assign(1, std::string(64, 'x'));
    CHECK_FALSE(t.set_batch(e, bad, false, &err));

    // Lines: strips need a mask of 4 bytes a vertex; edges need packed pairs.
    Batch strips = lines3(6, LineMode::Strips);
    CHECK(t.set_batch(e, strips, false, &err));
    strips.arrays.pop_back();
    CHECK_FALSE(t.set_batch(e, strips, false, &err));
    Batch edges = lines3(6, LineMode::Edges, 4);
    CHECK(t.set_batch(e, edges, false, &err));
    edges.arrays.push_back(array(Role::Colors, 4, 4, caliper_gfx::VertexFormat::Unorm8x4));
    CHECK_FALSE(t.set_batch(e, edges, false, &err));   // 4 rows read as per vertex (6) by default
    edges.per_edge = true;
    CHECK(t.set_batch(e, edges, false, &err));
    // The clamp and the gather read through storage bindings: offsets aligned to 256, rows packed.
    Batch misaligned = lines3(6, LineMode::Edges, 4);
    misaligned.arrays[1].offset = 128;
    CHECK_FALSE(t.set_batch(e, misaligned, false, &err));
    CHECK(err.find("256") != std::string::npos);
    misaligned = lines3(6, LineMode::Edges, 4);
    misaligned.arrays[0].stride = 16;
    CHECK_FALSE(t.set_batch(e, misaligned, false, &err));
    Batch mesh_misaligned = mesh3(4, 2);
    mesh_misaligned.arrays[1].offset = 12;
    CHECK_FALSE(t.set_batch(e, mesh_misaligned, false, &err));

    // Mesh: one of colours, values and uvs; uvs need a texture.
    Batch mesh = mesh3(4, 2);
    CHECK(t.set_batch(e, mesh, false, &err));
    mesh.arrays.push_back(array(Role::Uvs, 4, 8, caliper_gfx::VertexFormat::F32x2));
    CHECK_FALSE(t.set_batch(e, mesh, false, &err));
    mesh.texture = caliper_gfx::Texture{77};
    CHECK(t.set_batch(e, mesh, false, &err));
    mesh.arrays.push_back(array(Role::Values, 4, 4, caliper_gfx::VertexFormat::F32));
    CHECK_FALSE(t.set_batch(e, mesh, false, &err));
    CHECK(t.batch_count() == 3);   // points, lines and mesh
}

TEST_CASE("a scene holds at most 2,048 batches") {
    SceneTree t;
    std::string err;
    int made = 0;
    for (int i = 1; i < 1024 && made < 2048; ++i) {
        const EntityId e = t.ensure("e" + std::to_string(i));
        for (Primitive p : {Primitive::Points, Primitive::Lines, Primitive::Mesh}) {
            Batch b = p == Primitive::Points ? points3(1) : p == Primitive::Lines ? lines3(2, LineMode::Segments) : mesh3(3, 0);
            if (made < 2048) {
                REQUIRE(t.set_batch(e, b, false, &err));
                ++made;
            }
        }
    }
    CHECK(t.batch_count() == 2048);
    CHECK_FALSE(t.set_batch(t.ensure("e1000"), points3(1), false, &err));
    CHECK(err.find("2,048") != std::string::npos);
    // Replacing an existing batch is not a new one.
    CHECK(t.set_batch(t.find("e1"), points3(2), false, &err));
}

TEST_CASE("clears remove stamped parts, static items stay unless the clear is static") {
    SceneTree t;
    std::string err;
    const EntityId floor = t.ensure("room/floor");
    const EntityId arm = t.ensure("room/arm");
    REQUIRE(t.set_batch(floor, mesh3(4, 2), true, &err));           // static setup
    REQUIRE(t.set_batch(arm, points3(5), false, &err));             // stamped
    REQUIRE(t.set_transform(arm, translation({1, 2, 3}), false, &err));
    REQUIRE(t.set_transform(floor, translation({0, 0, -1}), true, &err));

    t.clear(t.find("room"), false);                                   // not recursive: only "room"
    CHECK(t.batch(arm, Primitive::Points));

    t.clear(t.root(), true);                                          // a recursive, stamped clear of the root
    CHECK(t.batch(floor, Primitive::Mesh));
    CHECK(t.transform(floor));
    CHECK_FALSE(t.batch(arm, Primitive::Points));
    CHECK_FALSE(t.transform(arm));
    CHECK(t.batch_static(floor, Primitive::Mesh));

    // Latest-at: a stamped batch wins over the static one, and a clear brings the static one back.
    REQUIRE(t.set_batch(floor, mesh3(6, 0), false, &err));
    CHECK(t.batch(floor, Primitive::Mesh)->count() == 6);
    CHECK_FALSE(t.batch_static(floor, Primitive::Mesh));
    t.clear(floor, false);
    CHECK(t.batch(floor, Primitive::Mesh)->count() == 4);

    // A later log brings the part back; a static clear removes static items too.
    REQUIRE(t.set_batch(arm, points3(2), false, &err));
    CHECK(t.batch(arm, Primitive::Points));
    t.clear(t.root(), true, true);
    CHECK_FALSE(t.batch(floor, Primitive::Mesh));
    CHECK_FALSE(t.transform(floor));
    CHECK(t.batch_count() == 0);
    CHECK(t.entity_count() == 4);   // the root, room and its two children stay, empty
}

TEST_CASE("a clear of named parts removes only those parts, by layer, over the subtree") {
    SceneTree t;
    std::string err;
    const EntityId cam = t.ensure("rig/left");
    const EntityId img = t.ensure("rig/left/rgb");
    const EntityId cloud = t.ensure("room/cloud");
    REQUIRE(t.set_transform(cam, translation({0, 0, 1}), false, &err));
    REQUIRE(t.set_transform(cam, translation({0, 0, 2}), true, &err));
    REQUIRE(t.set_camera(cam, pinhole(), false, &err));
    REQUIRE(t.set_batch(img, points2(3), false, &err));
    REQUIRE(t.set_batch(cloud, points3(4), false, &err));
    REQUIRE(t.set_batch(cloud, mesh3(3, 0), false, &err));

    // The transform alone: the stamped one goes, the static one stays; the camera stays.
    t.clear(cam, false, false, kClearTransform);
    REQUIRE(t.transform(cam));
    CHECK(t.transform(cam)->at(2, 3) == 2.0);
    CHECK(t.camera(cam));
    // The camera alone.
    t.clear(cam, false, false, kClearCamera);
    CHECK_FALSE(t.camera(cam));
    CHECK(t.transform(cam));
    // Points over the whole scene: the mesh stays, and so do the transforms.
    t.clear(t.root(), true, false, clear_bit(Primitive::Points));
    CHECK_FALSE(t.batch(img, Primitive::Points));
    CHECK_FALSE(t.batch(cloud, Primitive::Points));
    CHECK(t.batch(cloud, Primitive::Mesh));
    // A static clear of the transform removes both layers.
    t.clear(cam, false, true, kClearTransform);
    CHECK_FALSE(t.transform(cam));
    CHECK(t.batch(cloud, Primitive::Mesh));
}

TEST_CASE("the class table: entries, the palette rule and its limits") {
    SceneTree t;
    std::string err;
    REQUIRE(t.set_classes({{0, "floor", 0xff112233u}, {7, "obstacle", 0x80445566u}}, true, &err));
    static uint32_t texels[65536];
    t.class_texels(texels);
    CHECK(texels[0] == 0xff112233u);
    CHECK(texels[7] == 0x80445566u);
    CHECK(texels[1] == kPalette[1]);
    CHECK(texels[65535] == kPalette[65535 % 16]);
    CHECK(t.class_label(7) == "obstacle");
    CHECK(t.class_label(9) == "9");
    CHECK_FALSE(t.set_classes({{65536, "x", 0}}, true, &err));
    std::vector<ClassEntry> big(4097);
    for (uint32_t i = 0; i < big.size(); ++i) big[i].id = i;
    CHECK_FALSE(t.set_classes(big, true, &err));
    // A clear leaves the class table.
    t.clear(t.root(), true, true);
    CHECK(t.classes().size() == 2);
}

TEST_CASE("updates mark what changed") {
    SceneTree t;
    const uint64_t r0 = t.revision(), p0 = t.pose_revision();
    const EntityId e = t.ensure("a");
    CHECK(t.revision() > r0);
    CHECK(t.pose_revision() == p0);
    REQUIRE(t.set_transform(e, translation({1, 0, 0})));
    CHECK(t.pose_revision() > p0);
    const uint64_t r1 = t.revision(), p1 = t.pose_revision();
    REQUIRE(t.set_batch(e, points3(3)));
    CHECK(t.revision() > r1);
    CHECK(t.pose_revision() == p1);
}
