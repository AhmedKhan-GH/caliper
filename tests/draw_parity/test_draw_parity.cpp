// The parts of the pixel comparison and of the browser module's draw halves that need no GPU
// (tests/draw_parity/parity_draw.h, web/wasm/src/program_frames.h): the fixture's frames and plan,
// the decode of PROGRAM with the pin's hash, the packed params, the refusals that the module gives,
// and an install of each decoded program on the recording backend of the graphics layer as the
// browser installs it (WGSL as it is, no translator).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "parity_fixture.h"

#include "gfx/recording_backend.h"

#include <algorithm>
#include <map>

namespace {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace wire = caliper_net::wire;

struct Decoded {
    std::map<std::string, wire::Program> programs;
    std::map<std::string, std::vector<wire::Resource>> resources;
};

Decoded decode(const dp::Fixture& fx) {
    Decoded d;
    std::vector<std::pair<const std::uint8_t*, std::size_t>> frames;
    std::string why;
    REQUIRE_MESSAGE(dp::split_frames(fx.frames.data(), fx.frames.size(), &frames, &why), why);
    for (const auto& [data, n] : frames) {
        wire::Message m;
        const wire::DecodeResult r = wire::decode(data, n, &m);
        REQUIRE_MESSAGE(r.ok(), r.reason, " ", r.error_code, " ", r.message);
        if (auto* p = std::get_if<wire::Program>(&m)) d.programs[p->id] = *p;
        else if (auto* x = std::get_if<wire::Resource>(&m)) d.resources[x->program].push_back(*x);
        else FAIL("a frame that is neither PROGRAM nor RESOURCE");
    }
    return d;
}

}  // namespace

TEST_CASE("the fixture's frames are PROGRAM sets of the instance form and RESOURCE versions") {
    const dp::Fixture fx = dp::make_fixture();
    REQUIRE_MESSAGE(fx.error.empty(), fx.error);
    const Decoded d = decode(fx);
    CHECK(d.programs.size() == fx.plan.programs.size());
    for (const auto& [id, p] : d.programs) {
        CHECK(p.op == CALIPER_NET_PROGRAM_SET);
        CHECK(p.piece.is_object());
        CHECK(p.manifest.is_object());
        CHECK(p.panel.has_value());
        CHECK(p.draw.has_value());
        CHECK(p.derive.has_value());   // the pin's hash covers derive.wgsl
    }
    CHECK(d.resources.at("head_pattern").size() == 1);
    CHECK(d.resources.at("tower_arcs").size() == 3);
    for (const wire::Resource& r : d.resources.at("tower_arcs")) {
        const cd::ResourceVersion v = caliper_web::resource_version(r);
        CHECK(v.step == 100);
        CHECK(v.size == std::size_t(r.count) * r.stride);
        CHECK(v.bytes == r.data.data);
    }
}

TEST_CASE("each program of the fixture decodes as the plan expects, and the drawn ones install as in a browser") {
    const dp::Fixture fx = dp::make_fixture(64);
    REQUIRE_MESSAGE(fx.error.empty(), fx.error);
    const Decoded d = decode(fx);
    for (const dp::ProgramPlan& pp : fx.plan.programs) {
        CAPTURE(pp.id);
        cs::Piece piece;
        cd::Instance instance;
        const cd::Verdict v = caliper_web::decode_program(d.programs.at(pp.id), &piece, &instance);
        if (pp.expect == cp::kPieceHashMismatch) {
            CHECK(v.code == cp::kPieceHashMismatch);
            continue;
        }
        REQUIRE_MESSAGE(v.ok(), v.code, " ", v.message);
        CHECK(instance.program == pp.id);
        CHECK(instance.panel == d.programs.at(pp.id).panel.value());
        if (!pp.expect.empty()) continue;   // refused by WebGPU's compiler only
        caliper_gfx::RecordingBackend g;
        dp::Target target(g, fx.plan.size);
        REQUIRE(target.ok);
        cd::DrawRuntime rt(g, true);
        const cd::Verdict iv = rt.install(piece, instance, pp.pick_id);
        REQUIRE_MESSAGE(iv.ok(), iv.code, " ", iv.message);
        for (const wire::Resource& r : d.resources.at(pp.id)) CHECK(rt.set_resource(caliper_web::resource_version(r)).ok());
        std::vector<caliper_gfx::Serial> serials;
        std::string why;
        const cd::FrameInputs in =
            dp::frame_inputs(pp, fx.plan, 0, caliper_web::pack_params(piece, d.programs.at(pp.id).params));
        CHECK_MESSAGE(target.frame(rt, in, &serials, &why), why);
        CHECK(serials.size() >= 2);   // the clear and the program's frame
        CHECK(rt.status().state == cd::DrawState::running);
    }
}

TEST_CASE("a PROGRAM without its draw half, or with another file, is refused before anything is made") {
    const dp::Fixture fx = dp::make_fixture(64);
    REQUIRE_MESSAGE(fx.error.empty(), fx.error);
    const Decoded d = decode(fx);
    cs::Piece piece;
    cd::Instance instance;
    wire::Program no_draw = d.programs.at("tower_arcs");
    no_draw.draw.reset();
    CHECK(caliper_web::decode_program(no_draw, &piece, &instance).code == cp::kPieceBadManifest);
    wire::Program changed = d.programs.at("tower_arcs");
    *changed.draw += "// changed\n";
    CHECK(caliper_web::decode_program(changed, &piece, &instance).code == cp::kPieceHashMismatch);
    wire::Program no_derive = d.programs.at("tower_arcs");
    no_derive.derive.reset();   // the pin's hash covers derive.wgsl: without it the hash cannot hold
    CHECK(caliper_web::decode_program(no_derive, &piece, &instance).code == cp::kPieceHashMismatch);
}

TEST_CASE("a viewer keeps only versions of published resources at their declared sizes, and at most 16 programs") {
    const dp::Fixture fx = dp::make_fixture(64);
    REQUIRE_MESSAGE(fx.error.empty(), fx.error);
    const Decoded d = decode(fx);
    cs::Piece piece;
    cd::Instance instance;
    REQUIRE(caliper_web::decode_program(d.programs.at("tower_arcs"), &piece, &instance).ok());
    const auto published = caliper_web::published_sizes(piece);
    CHECK(published.size() == 3);   // heads, arcs and labels
    for (const wire::Resource& r : d.resources.at("tower_arcs")) CHECK(caliper_web::keeps_version(published, r));

    const std::vector<wire::Resource>& versions = d.resources.at("tower_arcs");
    const auto found =
        std::find_if(versions.begin(), versions.end(), [](const wire::Resource& r) { return r.resource == "arcs"; });
    REQUIRE(found != versions.end());
    const wire::Resource arcs = *found;
    wire::Resource other = arcs;
    other.resource = "arcs2";   // a name that the piece does not publish
    CHECK_FALSE(caliper_web::keeps_version(published, other));
    std::vector<std::uint8_t> big((published.at("arcs").count + 1) * std::size_t(arcs.stride));
    wire::Resource over = arcs;
    over.count = published.at("arcs").count + 1;   // one element above the declared count
    over.data.data = big.data();
    over.data.size = big.size();
    CHECK_FALSE(caliper_web::keeps_version(published, over));
    over.count = arcs.count;   // the bytes alone above the declared bytes
    CHECK_FALSE(caliper_web::keeps_version(published, over));

    // A private resource, or a manifest that fails its checks, gives nothing to keep.
    cs::Piece private_arcs = piece;
    for (cs::Resource& r : private_arcs.resources) r.publish = r.id != "arcs";
    CHECK(caliper_web::published_sizes(private_arcs).count("arcs") == 0);
    cs::Piece too_big = piece;
    for (cs::Resource& r : too_big.resources)
        if (r.id == "arcs") r.count = cs::piece_limits::kCountMax;
    if (cp::check_piece_manifest(too_big).code != cp::kProgramNotBuilt) CHECK(caliper_web::published_sizes(too_big).empty());

    std::map<std::string, int> programs;
    for (unsigned k = 0; k < CALIPER_NET_MAX_PROGRAMS; ++k) {
        CHECK(caliper_web::takes_program(programs, "p" + std::to_string(k)));
        programs["p" + std::to_string(k)] = 1;
    }
    CHECK_FALSE(caliper_web::takes_program(programs, "one_more"));
    CHECK(caliper_web::takes_program(programs, "p3"));   // a program set again
}

TEST_CASE("params pack in the manifest's order with their defaults and ranges") {
    const cs::Piece field = caliper_test_pieces::attention_field();   // layer i32 0..1, head i32 0..3, gain f32 0.1..10
    caliper_net::Json values = caliper_net::Json::object();
    values.set("head", 7);       // clamped to 3
    values.set("gain", 0.5);
    const std::vector<std::uint8_t> b = caliper_web::pack_params(field, values);
    REQUIRE(b.size() == 12);
    std::int32_t layer = -1, head = -1;
    float gain = 0;
    std::memcpy(&layer, b.data(), 4);
    std::memcpy(&head, b.data() + 4, 4);
    std::memcpy(&gain, b.data() + 8, 4);
    CHECK(layer == 0);           // the default
    CHECK(head == 3);
    CHECK(gain == 0.5f);
}

TEST_CASE("an entry of the layer's error list refuses only the program that it names") {
    const cd::Verdict v =
        caliper_web::refusal_of_error("tower_arcs", "render pipeline tower_arcs draw.passes[2]: the stage outputs differ");
    CHECK(v.code == cp::kProgramBrowserRefused);
    CHECK(v.where == "draw.passes[2]");
    CHECK(v.message.find("the stage outputs differ") != std::string::npos);
    CHECK(caliper_web::refusal_of_error("tower", "render pipeline tower_arcs draw.passes[2]: x").ok());
    CHECK(caliper_web::refusal_of_error("tower_arcs", "submit 12: a frame was refused").ok());
    CHECK(caliper_web::refusal_of_error("tower_arcs", "program tower_arcs draw.passes[0]: bad").where == "draw.passes[0]");
}

TEST_CASE("the plan and the frames file read back what was written") {
    const dp::Fixture fx = dp::make_fixture();
    REQUIRE_MESSAGE(fx.error.empty(), fx.error);
    dp::Plan back;
    std::string why;
    REQUIRE_MESSAGE(dp::read_plan(dp::plan_json(fx.plan), &back, &why), why);
    REQUIRE(back.programs.size() == fx.plan.programs.size());
    for (std::size_t i = 0; i < back.programs.size(); ++i) {
        CHECK(back.programs[i].id == fx.plan.programs[i].id);
        CHECK(back.programs[i].pick_id == fx.plan.programs[i].pick_id);
        CHECK(back.programs[i].expect == fx.plan.programs[i].expect);
        CHECK(back.programs[i].expect_line == fx.plan.programs[i].expect_line);
        CHECK(back.programs[i].inputs.clip_from_world == fx.plan.programs[i].inputs.clip_from_world);   // exact
        CHECK(back.programs[i].inputs.pixel_ratio == fx.plan.programs[i].inputs.pixel_ratio);
    }
    std::vector<std::pair<const std::uint8_t*, std::size_t>> frames;
    CHECK_FALSE(dp::split_frames(fx.frames.data(), fx.frames.size() - 1, &frames, &why));
    CHECK_FALSE(dp::read_plan("{\"size\": 0}", &back, &why));
}
