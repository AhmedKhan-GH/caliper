#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <algorithm>
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
    REQUIRE(piece.draw->text->selected);
    CHECK(piece.draw->text->selected->anchor == 32);
    CHECK(piece.draw->text->selected->param == "selected_node");
    CHECK(piece.draw->text->selected->string_offset == 32);
    REQUIRE(piece.params.size() == 1);
    CHECK(piece.params[0].name == "selected_node");
    CHECK(piece.derive->passes.size() == 7);
    CHECK(piece.derive->time_ms.value_or(4) == 50);
    CHECK(piece.derive->memory_mb.value_or(64) == 64);
    CHECK(piece.draw->frame_ms.value_or(4) == 16);
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
    CHECK(bytes < (16u << 20));
    const auto capacity = [&](const char* id) {
        const auto found = std::find_if(piece.resources.begin(), piece.resources.end(),
            [&](const auto& resource) { return resource.id == id; });
        REQUIRE(found != piece.resources.end());
        return found->count;
    };
    CHECK(capacity("nodes") == 8192);
    CHECK(capacity("curves") == 262144);
    CHECK(capacity("strings") == 8224);
    CHECK(capacity("cells") == 8192);
    CHECK(capacity("offset_a") == 8192);
    CHECK(capacity("offset_b") == 8192);
    for (const auto& resource : piece.resources) if (resource.id == "cells") {
        cp::ElementLayout layout;
        REQUIRE(cp::element_layout(resource.layout,&layout).ok());
        CHECK(layout.stride == 40); // Exposure occupies the former tail padding.
    }
    CHECK(piece.params[0].max == 8191);
    for (const auto& pass : piece.derive->passes)
        CHECK(pass.workgroups[0] * 64 == (pass.entry == "edge_curves" ? 262144 : 8192));
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

TEST_CASE("hover text metadata round trips and requires bounded anchor and string indices") {
    namespace cs = caliper_builder::spec;
    namespace cp = caliper_builder::programs;
    auto piece = interp_pieces::load("attribution_graph");
    REQUIRE(piece.draw->text->hover);
    const auto parsed = cs::piece_from_json_value(cs::to_json(piece));
    REQUIRE(parsed.ok());
    CHECK(parsed.piece.draw->text->hover == piece.draw->text->hover);
    CHECK(cp::piece_hash(parsed.piece) == cp::piece_hash(piece));
    SUBCASE("anchor outside resource") { piece.draw->text->hover->anchor = 34; }
    SUBCASE("anchor outside panel limit") { piece.draw->text->hover->anchor = 512; }
    SUBCASE("string offset outside table") { piece.draw->text->hover->string_offset = 33; }
    SUBCASE("string arithmetic cannot wrap") { piece.draw->text->hover->string_offset = 0xffffffffu; }
    SUBCASE("no item map") {
        piece.draw->pick.reset();
        for (auto& pass : piece.draw->passes) pass.pick = false;
    }
    SUBCASE("no declared string count") {
        cs::PieceInput source;
        source.name = "names";
        source.kind = cs::InputKind::buffer;
        source.layout.element = "u32";
        piece.inputs.push_back(source);
        piece.draw->text->strings_from = "names";
    }
    CHECK(cp::check_piece_manifest(piece).code == cp::kProgramBadText);
}

TEST_CASE("hover text parsing rejects malformed declarations and defaults the offset to zero") {
    namespace cs = caliper_builder::spec;
    const auto piece = interp_pieces::load("attribution_graph");
    auto value = cs::to_json(piece);
    auto& hover = value["draw"]["text"]["hover"];
    SUBCASE("required anchor") { hover.erase("anchor"); }
    SUBCASE("negative anchor") { hover["anchor"] = -1; }
    SUBCASE("fractional anchor") { hover["anchor"] = 1.5; }
    SUBCASE("unknown key") { hover["other"] = 1; }
    SUBCASE("non-table hover") { hover = false; }
    SUBCASE("default offset") {
        hover.erase("string_offset");
        const auto parsed = cs::piece_from_json_value(value);
        REQUIRE(parsed.ok());
        CHECK(parsed.piece.draw->text->hover->string_offset == 0);
        return;
    }
    CHECK_FALSE(cs::piece_from_json_value(value).ok());
}

TEST_CASE("selected text metadata round trips and requires bounded integer item references") {
    namespace cs = caliper_builder::spec;
    namespace cp = caliper_builder::programs;
    auto piece = interp_pieces::load("attribution_graph");
    piece.draw->text->selected = cs::TextSelected{32, "selected_node", 32};
    REQUIRE(cp::check_piece_manifest(piece).ok());
    const auto parsed = cs::piece_from_json_value(cs::to_json(piece));
    REQUIRE(parsed.ok());
    CHECK(parsed.piece.draw->text->selected == piece.draw->text->selected);
    CHECK(cp::piece_hash(parsed.piece) == cp::piece_hash(piece));
    SUBCASE("anchor outside resource") { piece.draw->text->selected->anchor = 34; }
    SUBCASE("anchor outside panel limit") { piece.draw->text->selected->anchor = 512; }
    SUBCASE("shared hover anchor") { piece.draw->text->selected->anchor = 33; }
    SUBCASE("string offset outside table") { piece.draw->text->selected->string_offset = 33; }
    SUBCASE("string arithmetic cannot wrap") { piece.draw->text->selected->string_offset = 0xffffffffu; }
    SUBCASE("missing parameter") { piece.draw->text->selected->param = "missing"; }
    SUBCASE("floating point parameter") { piece.params[0].type = cs::ParamType::f32; }
    SUBCASE("no item map") {
        piece.draw->text->hover.reset();
        piece.draw->pick.reset();
        for (auto& pass : piece.draw->passes) pass.pick = false;
    }
    SUBCASE("no declared string count") {
        piece.draw->text->hover.reset();
        cs::PieceInput source;
        source.name = "names";
        source.kind = cs::InputKind::buffer;
        source.layout.element = "u32";
        piece.inputs.push_back(source);
        piece.draw->text->strings_from = "names";
    }
    SUBCASE("unsigned parameter") {
        piece.params[0].type = cs::ParamType::u32;
        piece.params[0].default_value = 0;
        piece.params[0].min = 0;
        CHECK(cp::check_piece_manifest(piece).ok());
        return;
    }
    CHECK(cp::check_piece_manifest(piece).code == cp::kProgramBadText);
}

TEST_CASE("selected text parsing rejects malformed declarations and defaults the offset to zero") {
    namespace cs = caliper_builder::spec;
    auto piece = interp_pieces::load("attribution_graph");
    piece.draw->text->selected = cs::TextSelected{32, "selected_node", 32};
    auto value = cs::to_json(piece);
    auto& selected = value["draw"]["text"]["selected"];
    SUBCASE("required anchor") { selected.erase("anchor"); }
    SUBCASE("required parameter") { selected.erase("param"); }
    SUBCASE("negative anchor") { selected["anchor"] = -1; }
    SUBCASE("fractional anchor") { selected["anchor"] = 1.5; }
    SUBCASE("non-string parameter") { selected["param"] = 0; }
    SUBCASE("unknown key") { selected["other"] = 1; }
    SUBCASE("non-table selected") { selected = false; }
    SUBCASE("default offset") {
        selected.erase("string_offset");
        const auto parsed = cs::piece_from_json_value(value);
        REQUIRE(parsed.ok());
        CHECK(parsed.piece.draw->text->selected->string_offset == 0);
        return;
    }
    CHECK_FALSE(cs::piece_from_json_value(value).ok());
}
