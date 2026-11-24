#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "pieces.h"
#include "draw/draw_runtime.h"
#include "gfx/recording_backend.h"
#include <cstring>
#include <limits>

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace gfx = caliper_gfx;

namespace {
struct Graph {
    gfx::RecordingBackend backend;
    cd::DrawRuntime runtime{backend, true};
    cd::HostTarget target;
    std::vector<std::uint8_t> strings = std::vector<std::uint8_t>(544 * 64);

    Graph(unsigned anchor = 33, unsigned offset = 32) {
        auto piece = interp_pieces::load("attribution_graph");
        piece.name = "hover_fixture"; // The runtime uses the declaration, not a library name.
        piece.draw->text->hover = caliper_builder::spec::TextHover{anchor, offset};
        piece.hash = cp::piece_hash(piece);
        const auto installed = runtime.install(piece, {"graph", {}, "graph"}, 7);
        REQUIRE_MESSAGE(installed.ok(), installed.code, ": ", installed.message);
        gfx::TextureDesc texture;
        texture.width = texture.height = 64;
        texture.format = gfx::Format::RGBA8Unorm;
        texture.usage = gfx::kRenderTarget;
        target.color = backend.create_texture(texture);
        target.width = target.height = 64;
        for (const auto& resource : piece.resources) {
            if (!resource.publish) continue;
            cp::ElementLayout layout;
            REQUIRE(cp::element_layout(resource.layout, &layout).ok());
            std::vector<std::uint8_t> bytes(resource.count * layout.stride);
            unsigned count = resource.count;
            if (resource.id == "nodes") {
                count = 2;
                // The node map has 48-byte elements, with its index at byte 16.
                const std::uint32_t one = 1;
                std::memcpy(bytes.data() + layout.stride + 16, &one, 4);
            } else if (resource.id == "curves") count = 0;
            else if (resource.id == "labels") {
                const std::uint32_t absent = 0xffffffffu;
                for (unsigned i = 0; i < count; ++i) std::memcpy(bytes.data() + i * layout.stride + 16, &absent, 4);
                // A token, a selected node, and a stale synthetic hover anchor.
                for (const auto [row, text] : {std::pair{0u, 0u}, {32u, 32u}, {33u, 32u}})
                    std::memcpy(bytes.data() + row * layout.stride + 16, &text, 4);
                const float position[4] = {0.4f, 0.9f, 0, 1};
                const float priority = 4;
                std::memcpy(bytes.data() + anchor * layout.stride, position, sizeof position);
                std::memcpy(bytes.data() + anchor * layout.stride + 20, &priority, 4);
            }
            bytes.resize(count * layout.stride);
            set(resource.id, bytes, count, 1);
        }
        put_text(0, "Token");
        put_text(32, "Selected Feature");
        put_text(33, "Hovered Feature");
        set("strings", strings, 544, 1);
        draw();
    }

    void put_text(unsigned row, const std::string& text) {
        std::memset(strings.data() + row * 64, 0, 64);
        std::memcpy(strings.data() + row * 64, text.data(), std::min<std::size_t>(64, text.size()));
    }
    void set(const std::string& name, const std::vector<std::uint8_t>& bytes, unsigned count, int step) {
        cd::ResourceVersion version;
        version.resource = name; version.step = step; version.count = count;
        version.bytes = bytes.data(); version.size = bytes.size();
        REQUIRE(runtime.set_resource(version).ok());
    }
    void draw() {
        std::vector<gfx::Frame> frames;
        REQUIRE(runtime.encode(&frames, {}, target));
        // This backend records commands on the CPU. It executes no shader and creates no device.
        for (auto& frame : frames) REQUIRE(backend.submit(std::move(frame)) != 0);
        runtime.frame_done(0);
    }
};
}

TEST_CASE("graph hover labels use a local hit without a derive update or a selection change") {
    Graph graph;
    const auto before = graph.runtime.labels();
    REQUIRE(before.size() == 2);
    CHECK(before[0].text == "Token");
    CHECK(before[1].text == "Selected Feature");
    const auto picked = graph.runtime.pick(1);
    REQUIRE(picked);
    const auto shown = graph.runtime.shown_items();
    CHECK(shown.version == graph.runtime.shown_items().version);
    const auto hovered = graph.runtime.labels(&*picked);
    REQUIRE(hovered.size() == 3);
    CHECK(hovered[0].text == before[0].text);
    CHECK(hovered[1].text == before[1].text);
    CHECK(hovered[2].text == "Hovered Feature");
    CHECK(hovered[2].position[0] == doctest::Approx(0.4f));
    CHECK(hovered[2].position[1] == doctest::Approx(0.9f));
    CHECK(hovered[2].priority == 4);
    CHECK(graph.runtime.labels().size() == 2);
    CHECK(graph.runtime.shown_items().version == shown.version);
}

TEST_CASE("a hover declaration chooses its own anchor and string offset") {
    Graph graph(31, 0);
    graph.put_text(1, "Local Label");
    graph.set("strings", graph.strings, 544, 1);
    graph.draw();
    const auto picked = graph.runtime.pick(1).value();
    const auto labels = graph.runtime.labels(&picked);
    REQUIRE(labels.size() == 4);
    CHECK(labels[1].text == "Local Label");
    CHECK(labels[1].priority == 4);
    CHECK(graph.runtime.labels().size() == 3);
}

TEST_CASE("the viewer manifest decoder preserves the hover declaration and pin") {
    namespace cs = caliper_builder::spec;
    const auto piece = interp_pieces::load("attribution_graph");
    const auto manifest = nlohmann::json::parse(cp::canonical_manifest(piece));
    nlohmann::json declaration = {{"id", "graph"}, {"panel", "graph"}, {"manifest", manifest},
        {"piece", {{"name", piece.name}, {"version", piece.version}, {"hash", cp::piece_hash(piece)}}}};
    cs::Piece decoded;
    cd::Instance instance;
    const auto check = cd::DrawRuntime::decode(declaration.dump(), piece.files.at("common.wgsl"),
        piece.files.at("draw.wgsl"), &decoded, &instance, cd::DrawChecks::pieces(), piece.files.at("derive.wgsl"));
    REQUIRE_MESSAGE(check.ok(), check.message);
    REQUIRE(decoded.draw->text->hover);
    CHECK(decoded.draw->text->hover == piece.draw->text->hover);
    CHECK(cp::piece_hash(decoded) == cp::piece_hash(piece));
}

TEST_CASE("graph hover labels reject a different program, invalid item, or stale hit") {
    Graph graph;
    auto picked = graph.runtime.pick(1).value();
    SUBCASE("different program") { picked.program = "other"; }
    SUBCASE("invalid item") { picked.item = std::numeric_limits<unsigned>::max(); }
    SUBCASE("different step") { picked.step = 2; }
    SUBCASE("changed fields") { picked.fields[0].second = 9; }
    CHECK(graph.runtime.labels(&picked).size() == 2);
}

TEST_CASE("graph hover labels retain UTF-8 checks and require matching text steps") {
    Graph graph;
    const auto picked = graph.runtime.pick(1).value();
    SUBCASE("invalid UTF-8") { graph.put_text(33, "\xc3\x28"); }
    SUBCASE("empty label") { graph.put_text(33, ""); }
    SUBCASE("newer text") {
        graph.set("strings", graph.strings, 544, 2);
        graph.draw();
        CHECK(graph.runtime.labels(&picked).size() == 2);
        return;
    }
    graph.set("strings", graph.strings, 544, 1);
    graph.draw();
    CHECK(graph.runtime.labels(&picked).size() == 2);
}

TEST_CASE("a node replacement at the same step changes the hover version identity") {
    Graph graph;
    const auto first = graph.runtime.shown_items();
    std::vector<std::uint8_t> bytes(48);
    graph.set("nodes", bytes, 1, 1);
    graph.draw();
    const auto second = graph.runtime.shown_items();
    CHECK(first.version != second.version);
    REQUIRE(cd::DrawRuntime::pick_shown(1, first));
    CHECK_FALSE(cd::DrawRuntime::pick_shown(1, second));
}
