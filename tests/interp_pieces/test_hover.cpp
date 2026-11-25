#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "pieces.h"
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"
#include "gfx/recording_backend.h"
#include <cstring>
#include <limits>

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace gfx = caliper_gfx;

namespace {
struct Graph {
    gfx::RecordingBackend backend;
    cd::DrawRuntime runtime;
    cd::HostTarget target;
    std::vector<std::uint8_t> strings = std::vector<std::uint8_t>(544 * 64);

    Graph(unsigned anchor = 33, unsigned offset = 32,
          std::optional<caliper_builder::spec::TextSelected> selected = {}, bool webgpu = true)
        : runtime(backend, webgpu) {
        auto piece = interp_pieces::load("attribution_graph");
        piece.name = "hover_fixture"; // The runtime uses the declaration, not a library name.
        piece.draw->text->hover = caliper_builder::spec::TextHover{anchor, offset};
        if (selected) piece.draw->text->selected = selected;
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
    void draw(std::int32_t selected = 0, bool use_default = false) {
        std::vector<gfx::Frame> frames;
        cd::FrameInputs inputs;
        if (!use_default) {
            inputs.params.resize(4);
            std::memcpy(inputs.params.data(), &selected, 4);
        }
        REQUIRE(runtime.encode(&frames, inputs, target));
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

TEST_CASE("draw status tracks staggered attribution resources and a new selected frame") {
    Graph graph;
    const auto first = graph.runtime.status();
    CHECK(first.frames == 1);
    CHECK(first.resource_steps == std::map<std::string, std::int64_t>{
        {"nodes", 1}, {"curves", 1}, {"labels", 1}, {"strings", 1}});
    std::map<std::string, std::vector<std::uint8_t>> bytes;
    for (const auto& slot : graph.runtime.internals().slots)
        if (slot.shown) bytes[slot.id] = slot.shown->bytes;
    graph.put_text(33, "New Selected Feature");
    graph.set("nodes", bytes.at("nodes"), 2, 2);
    CHECK(graph.runtime.status().resource_steps == first.resource_steps);
    graph.draw(1);
    CHECK(graph.runtime.status().resource_step == 2);
    CHECK(graph.runtime.status().resource_steps.at("nodes") == 2);
    CHECK(graph.runtime.status().resource_steps.at("strings") == 1);
    REQUIRE(graph.runtime.labels().size() == 1);
    graph.set("curves", {}, 0, 2);
    graph.draw(1);
    CHECK(graph.runtime.status().resource_steps.at("curves") == 2);
    graph.set("labels", bytes.at("labels"), 34, 2);
    graph.draw(1);
    CHECK(graph.runtime.status().resource_steps.at("labels") == 2);
    CHECK(graph.runtime.status().resource_steps.at("strings") == 1);
    REQUIRE(graph.runtime.labels().size() == 1);
    graph.set("strings", graph.strings, 544, 2);
    CHECK(graph.runtime.status().resource_steps.at("strings") == 1);
    graph.draw(1);
    CHECK(graph.runtime.status().frames == first.frames + 4);
    CHECK(graph.runtime.status().resource_steps == std::map<std::string, std::int64_t>{
        {"nodes", 2}, {"curves", 2}, {"labels", 2}, {"strings", 2}});
    REQUIRE(graph.runtime.labels().size() == 2);
    CHECK(graph.runtime.labels()[1].text == "New Selected Feature");
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
    CHECK(decoded.draw->text->selected == piece.draw->text->selected);
    CHECK(decoded.draw->text->hover == piece.draw->text->hover);
    CHECK(cp::piece_hash(decoded) == cp::piece_hash(piece));
}

TEST_CASE("selected text follows frame parameters without new resource versions") {
    bool webgpu = true;
    SUBCASE("browser composition") {}
    SUBCASE("native composition") { webgpu = false; }
    Graph graph(33, 32, {}, webgpu);
    const auto shown = graph.runtime.shown_items();
    REQUIRE(graph.runtime.labels().size() == 2);
    CHECK(graph.runtime.labels()[1].text == "Selected Feature");
    graph.draw(1);
    const auto changed = graph.runtime.labels();
    REQUIRE(changed.size() == 2);
    CHECK(changed[1].text == "Hovered Feature");
    CHECK(graph.runtime.shown_items().version == shown.version);
    CHECK(graph.runtime.status().resource_step == 1);
    const auto hover = graph.runtime.pick(0).value();
    const auto both = graph.runtime.labels(&hover);
    REQUIRE(both.size() == 3);
    CHECK(both[1].text == "Hovered Feature");
    CHECK(both[2].text == "Selected Feature");
    graph.draw(0);
    CHECK(graph.runtime.labels()[1].text == "Selected Feature");
    CHECK(graph.runtime.shown_items().version == shown.version);
}

TEST_CASE("selected text rejects absent or invalid items and mismatched shown resources") {
    Graph graph(33, 32, caliper_builder::spec::TextSelected{32, "selected_node", 32});
    SUBCASE("cleared selection") { graph.draw(-1); }
    SUBCASE("item beyond valid count") { graph.draw(2); }
    SUBCASE("maximum signed item") { graph.draw(std::numeric_limits<std::int32_t>::max()); }
    SUBCASE("default selection") { graph.draw(0, true); }
    SUBCASE("empty item map") {
        graph.set("nodes", {}, 0, 1);
        graph.set("curves", std::vector<std::uint8_t>(48), 1, 1);
        graph.draw();
    }
    SUBCASE("newer item map") {
        graph.set("nodes", std::vector<std::uint8_t>(48), 1, 2);
        graph.draw();
    }
    SUBCASE("newer text") {
        graph.set("strings", graph.strings, 544, 2);
        graph.draw();
    }
    SUBCASE("invalid UTF-8") {
        graph.put_text(32, "\xc3\x28");
        graph.set("strings", graph.strings, 544, 1);
        graph.draw();
    }
    SUBCASE("empty text") {
        graph.put_text(32, "");
        graph.set("strings", graph.strings, 544, 1);
        graph.draw();
    }
    const auto labels = graph.runtime.labels();
    REQUIRE(labels.size() == 1);
    CHECK(labels[0].text == "Token");
}

TEST_CASE("selected text uses its declared anchor and string offset") {
    Graph graph(33, 32, caliper_builder::spec::TextSelected{31, "selected_node", 0});
    graph.put_text(1, "Current Item");
    graph.set("strings", graph.strings, 544, 1);
    graph.draw(1);
    const auto labels = graph.runtime.labels();
    REQUIRE(labels.size() == 3);
    CHECK(labels[1].text == "Current Item");
    CHECK(labels[2].text == "Selected Feature");
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
        // Neither item label can use text from a different step than its node map.
        CHECK(graph.runtime.labels(&picked).size() == 1);
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


TEST_CASE("reference graph labels retain the last node in native and browser composition") {
    bool webgpu = true;
    SUBCASE("browser composition") {}
    SUBCASE("native composition") { webgpu = false; }
    Graph graph(33, 32, {}, webgpu);
    graph.strings.resize(8224 * 64);
    graph.put_text(8223, "Last Reference Feature");
    graph.set("strings", graph.strings, 8224, 1);
    std::vector<std::uint8_t> nodes(8192 * 48);
    const std::uint32_t last = 8191;
    std::memcpy(nodes.data() + last * 48 + 16, &last, 4);
    graph.set("nodes", nodes, 8192, 1);
    graph.draw(last);
    const auto selected = graph.runtime.labels();
    REQUIRE(selected.size() == 2);
    CHECK(selected[1].text == "Last Reference Feature");
    const auto picked = graph.runtime.pick(last);
    REQUIRE(picked);
    const auto hovered = graph.runtime.labels(&*picked);
    REQUIRE(hovered.size() == 3);
    CHECK(hovered[2].text == "Last Reference Feature");
}
