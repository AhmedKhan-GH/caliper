#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../../tests/boxes_match/match_rig.h"
#include "../../tests/draw_parity/parity_fixture.h"
#include "../../tests/interp_pieces/pieces.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

namespace mr = match_rig;
namespace dp = draw_parity;
namespace cp = caliper_builder::programs;
namespace wire = caliper_net::wire;
namespace fs = std::filesystem;
using Json = nlohmann::json;
static fs::path source, output;

static void write(const fs::path& file, const std::vector<uint8_t>& bytes) {
    std::ofstream f(file, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    REQUIRE(f.good());
}

TEST_CASE("circuit graph resources preserve native parity inputs") {
    @autoreleasepool {
        std::ifstream input(source);
        const Json actual = Json::parse(input);
        const auto piece = interp_pieces::load("attribution_graph");
        mr::Metal metal;
        REQUIRE_MESSAGE(metal.backend, metal.why);
        mr::PieceDraw draw(*metal.backend, 512);
        draw.install(piece, "graph");
        std::vector<uint8_t> frames;
        const auto add = [&](const wire::Message& message) {
            std::vector<uint8_t> bytes;
            std::string why;
            REQUIRE_MESSAGE(wire::encode(message, &bytes, &why), why);
            dp::append_frame(&frames, bytes);
        };
        caliper_net::Json params = caliper_net::Json::object();
        params.set("selected_node", actual.at("selected_node").get<int>());
        params.set("hovered_node", -1);
        wire::Program program;
        std::string why;
        REQUIRE_MESSAGE(dp::instance_message(piece, "graph", "graph", params, piece.hash, &program, &why), why);
        program.bind = caliper_net::Json::object();
        for (const auto& binding : piece.inputs) program.bind.set(binding.name, "live:" + binding.name);
        add(program);
        for (const auto& desc : piece.resources) {
            if (!desc.publish) continue;
            const auto& read = actual.at("resources").at(desc.id);
            const unsigned count = read.at("count").get<unsigned>();
            REQUIRE(count <= desc.count);
            REQUIRE(read.at("elements").size() == count);
            cp::ElementLayout layout;
            REQUIRE(cp::element_layout(desc.layout, &layout).ok());
            std::vector<uint8_t> bytes(size_t(count) * layout.stride, 0);
            for (unsigned row = 0; row < count; ++row) {
                unsigned scalar = 0;
                const auto& values = read.at("elements").at(row);
                for (const auto& field : layout.fields) {
                    for (unsigned part = 0; part < field.size / 4; ++part) {
                        REQUIRE(scalar < values.size());
                        uint8_t* address = bytes.data() + row * layout.stride + field.offset + part * 4;
                        if (field.type.find("f32") != std::string::npos) {
                            const float value = values.at(scalar).get<float>();
                            REQUIRE(std::isfinite(value));
                            std::memcpy(address, &value, 4);
                        } else if (field.type.find("i32") != std::string::npos) {
                            const int32_t value = values.at(scalar).get<int32_t>();
                            std::memcpy(address, &value, 4);
                        } else {
                            const uint32_t value = values.at(scalar).get<uint32_t>();
                            std::memcpy(address, &value, 4);
                        }
                        ++scalar;
                    }
                }
                REQUIRE(scalar == values.size());
            }
            draw.set(desc.id, bytes, count);
            auto message = dp::resource_message("graph", desc.id, bytes, 1);
            message.count = count;
            message.stride = layout.stride;
            add(message);
        }
        caliper_draw::FrameInputs in;
        in.clip_from_world = in.world_from_clip = in.view_from_world = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        in.viewport = {0,0,512,512};
        in.pixel_ratio = 1;
        in.cursor_step = in.latest_step = 1;
        in.live = true;
        in.params = caliper_web::pack_params(piece, params);
        REQUIRE(draw.draw(in));
        fs::create_directories(output);
        write(output / "native.rgba", draw.color);
        write(output / "native.id", draw.ids);
        write(output / "frames.bin", frames);
        dp::Plan plan;
        plan.size = 512;
        plan.frames = 1;
        plan.step = 1;
        dp::ProgramPlan item;
        item.id = "graph";
        item.pick_id = mr::kPiecePick;
        item.inputs = in;
        plan.programs.push_back(item);
        std::ofstream(output / "plan.json") << dp::plan_json(plan);
    }
}
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    source = argv[1];
    output = argv[2];
    doctest::Context context;
    return context.run();
}
