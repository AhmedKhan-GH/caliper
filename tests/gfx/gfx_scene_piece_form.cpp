#include <doctest/doctest.h>
#include "../gpu_skip.h"

#ifdef CALIPER_HAVE_METAL
#import <Metal/Metal.h>
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "live/scene_piece_form.h"
#include "renderer/host_renderer.h"
#include <filesystem>
#include <fstream>
#include <cstring>

caliper_host::HostRenderer* gfx_metal_renderer();

namespace {
namespace wire = caliper_net::wire;
struct Frames {
    std::vector<std::vector<uint8_t>> bytes;
    wire::Message load(const std::string& name) {
        const auto file = std::filesystem::path(CALIPER_TEST_SOURCE_ROOT) / "tests/fixtures/wire_v3/v3_2" / (name + ".bin");
        std::ifstream in(file, std::ios::binary);
        REQUIRE_MESSAGE(in.good(), name);
        bytes.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        wire::Message message;
        const auto decoded = wire::decode(bytes.back().data(), bytes.back().size(), &message);
        REQUIRE_MESSAGE(decoded.ok(), decoded.message);
        return message;
    }
};
struct View {
    std::unique_ptr<caliper_gfx::Backend> backend;
    std::unique_ptr<caliper_live_views::SceneView> scene;
    void init(bool flat) {
        auto* renderer = gfx_metal_renderer();
        REQUIRE(renderer);
        backend = caliper_gfx::make_metal_backend(renderer->native_device(), renderer->native_queue());
        scene = std::make_unique<caliper_live_views::SceneView>(*backend, caliper_gfx::shaders::kMetalPrograms);
        std::string why;
        REQUIRE_MESSAGE(scene->init(&why), why);
        scene->describe(wire::SceneView{flat ? "2d" : "3d", flat ? "-y" : "+z", ""});
    }
    std::vector<uint8_t> draw() {
        std::string why;
        scene->fit();
        REQUIRE_MESSAGE(scene->render(256, 256, 1, 0xff202020, &why), why);
        std::vector<uint8_t> pixels;
        REQUIRE_MESSAGE(scene->read_pixels(&pixels, &why), why);
        return pixels;
    }
    caliper_scene::PickResult pick(int x, int y, caliper_scene::Hit* hit) {
        std::string why;
        REQUIRE_MESSAGE(scene->request_pick(float(x) + 0.5f, float(y) + 0.5f, &why), why);
        auto result = caliper_scene::PickResult::Pending;
        for (int attempt = 0; attempt < 10000 && result == caliper_scene::PickResult::Pending; ++attempt) {
            if (scene->id_target()) backend->wait(backend->last_use(scene->id_target()));
            if (scene->depth_target()) backend->wait(backend->last_use(scene->depth_target()));
            backend->poll();
            result = scene->poll_pick(hit);
        }
        REQUIRE(result != caliper_scene::PickResult::Pending);
        return result;
    }
};
}

TEST_CASE("gfx/Metal scene piece resources: every part matches the recording converter") {
    if (!gfx_metal_renderer()) CALIPER_SKIP_NO_GPU("no Metal device");
    for (const char* suffix : {"lines_segments", "lines_strip", "lines_strips", "lines_edges", "arrows", "boxes",
                              "boxes_2d", "mesh", "mesh_normals", "mesh_textured", "mesh_colors", "image_depth",
                              "image_depth_f32", "image_segmentation", "image_segmentation_u16"}) {
        @autoreleasepool {
            CAPTURE(std::string(suffix));
            const std::string base = std::string("scene_update_") + suffix;
            Frames frames;
            auto old_message = frames.load(base);
            auto new_message = frames.load(base + "_bundle");
            const auto& old = std::get<wire::SceneUpdate>(old_message);
            auto& piece = std::get<wire::SceneUpdate>(new_message);
            caliper_live::ScenePieceInbox inbox;
            for (const auto& pair : piece.resources.members()) {
                auto message = frames.load(std::string("resource_scene_") + suffix + "_bundle_" + pair.first);
                const auto& r = std::get<wire::Resource>(message);
                REQUIRE(inbox.take(r, std::make_shared<const std::vector<uint8_t>>(r.data.data, r.data.data + r.data.size)));
            }
            std::string why;
            const auto owner = inbox.pair(piece, nullptr, &why);
            REQUIRE_MESSAGE(owner, why);
            const bool flat = old.part == "image" || old.view == "overlay" || old.view == "flat";
            View a, b;
            a.init(flat); b.init(flat);
            REQUIRE_MESSAGE(a.scene->apply(old, &why), why);
            REQUIRE_MESSAGE(b.scene->apply(piece, &why), why);
            const auto before = a.draw(), after = b.draw();
            REQUIRE(before.size() == after.size());
            size_t different = 0;
            for (size_t i = 0; i < before.size(); i += 4) different += std::memcmp(before.data() + i, after.data() + i, 4) != 0;
            MESSAGE("[scene-piece-resources] ", std::string(suffix), ": ", different, " of ", before.size() / 4, " pixels differ");
            CHECK(different == 0);
            CHECK(b.scene->shape_counts().received == 1);
            size_t hits = 0;
            // Query visible pixels spread through the frame. A pick window can include an edge.
            size_t candidates = 0;
            for (size_t i = 0; i < before.size(); i += 4)
                candidates += before[i] != 32 || before[i + 1] != 32 || before[i + 2] != 32;
            const size_t stride = std::max<size_t>(1, candidates / 24);
            size_t seen = 0;
            for (size_t i = 0; i < before.size() && hits < 24; i += 4) {
                if (before[i] == 32 && before[i + 1] == 32 && before[i + 2] == 32) continue;
                if (seen++ % stride) continue;
                const int x = int((i / 4) % 256), y = int((i / 4) / 256);
                caliper_scene::Hit ha, hb;
                const auto ra = a.pick(x, y, &ha), rb = b.pick(x, y, &hb);
                CHECK(ra == rb);
                if (ra != caliper_scene::PickResult::Hit) continue;
                ++hits;
                CHECK(a.scene->tree().path(ha.entity) == b.scene->tree().path(hb.entity));
                CHECK(ha.primitive == hb.primitive);
                CHECK(ha.item == hb.item);
                CHECK(ha.pixel[0] == hb.pixel[0]);
                CHECK(ha.pixel[1] == hb.pixel[1]);
            }
            MESSAGE("[scene-piece-resources] ", std::string(suffix), ": ", hits, " matching picks");
            CHECK(hits > 0);
        }
    }
}
#endif
