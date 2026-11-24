#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "pieces.h"
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/piece_commands.h>

TEST_CASE("graph passes piece.check with the frozen shapes and budgets") {
    namespace cp = caliper_builder::programs;
    const auto piece = interp_pieces::load("attribution_graph");
    const auto checked = caliper_builder::check_piece_json(caliper_builder::spec::to_json(piece), false, nullptr);
    REQUIRE_MESSAGE(checked.at("ok").get<bool>(), checked.dump(2));
    REQUIRE(piece.derive);
    REQUIRE(piece.draw);
    CHECK(piece.derive->passes.size() == 7);
    CHECK(piece.derive->time_ms.value_or(4) == 50);
    CHECK(piece.derive->memory_mb.value_or(64) == 64);
    CHECK(piece.draw->frame_ms.value_or(4) == 4);
    CHECK(piece.draw->memory_mb.value_or(64) == 64);
    if (cp::compiler_available()) {
        REQUIRE(checked.at("passes").size() == 9);
        for (const auto& pass : checked.at("passes")) CHECK_MESSAGE(pass.at("ok").get<bool>(), pass.dump());
        CHECK(checked.at("messages").empty());
    } else {
        MESSAGE("WGSL compiler absent: only manifest, source, and composition checks ran");
    }
    std::uint64_t bytes = 0;
    for (const auto& resource : piece.resources) {
        cp::ElementLayout layout;
        REQUIRE(cp::element_layout(resource.layout, &layout).ok());
        bytes += cp::layout_bytes(layout, resource.count);
    }
    CHECK(bytes < (1u << 20));
}

TEST_CASE("example, output, and recovery pieces compile within their budgets") {
    namespace cp = caliper_builder::programs;
    for (const std::string name : {"token_strip", "logit_bars", "heat_grid"}) {
        CAPTURE(name);
        const auto piece = interp_pieces::load(name);
        const auto checked = caliper_builder::check_piece_json(caliper_builder::spec::to_json(piece), false, nullptr);
        REQUIRE_MESSAGE(checked.at("ok").get<bool>(), checked.dump(2));
        REQUIRE(piece.derive); REQUIRE(piece.draw);
        CHECK(piece.derive->passes.size() <= 8);
        CHECK(piece.derive->time_ms.value_or(4) <= 50);
        CHECK(piece.derive->memory_mb.value_or(64) == 64);
        CHECK(piece.draw->frame_ms.value_or(4) == 4);
        CHECK(piece.draw->memory_mb.value_or(64) == 64);
        if (cp::compiler_available()) {
            CHECK(checked.at("passes").size() == piece.derive->passes.size() + piece.draw->passes.size());
            CHECK(checked.at("messages").empty());
            for (const auto& pass : checked.at("passes")) CHECK_MESSAGE(pass.at("ok").get<bool>(), pass.dump());
        }
    }
}
