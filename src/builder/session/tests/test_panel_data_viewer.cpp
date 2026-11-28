// Tests of what panel.data gives the browser's spec page for the programs of
// a scene panel and for a control of the layout (panel_data.h,
// panel_data_viewer.cpp): the instance form of a library, workspace-less and
// inline piece with its newest resource, the review state of a foreign spec,
// a pin with another hash, the state of each control from the run and the
// session's values, and panel.data of a control from the viewer origin.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

namespace sp = caliper_builder::spec;

// The library's points@1, pinned with its hash, drawing in the scene panel "world".
sp::Spec points_spec(const std::string& hash) {
    sp::Spec s;
    s.version = 2;
    s.instrument.id = "viewer";
    sp::Panel world;
    world.id = "world";
    world.primitive = sp::Primitive::scene;
    s.panels.push_back(world);
    sp::Program p;
    p.id = "dots";
    p.piece = sp::PieceRef{"points", 1, hash, sp::PieceFrom::library};
    p.panel = "world";
    p.bind["positions"] = "live:emb.act";
    p.params["size"] = 3.0;
    s.programs.push_back(p);
    return s;
}

std::string library_hash() {
    const FoundPiece f = find_piece(PieceLibrary::builtin(), "", sp::Spec{}, sp::PieceFrom::library, "points", 1);
    REQUIRE_FALSE(f.error);
    return programs::piece_hash(f.piece);
}

ResourceVersion version(std::int64_t step, std::uint8_t fill) {
    ResourceVersion v;
    v.step = step;
    v.count = 2;
    v.stride = 24;
    v.bytes = std::make_shared<const std::vector<std::uint8_t>>(48, fill);
    return v;
}

struct FakeRun final : RunControl {
    RunState st;
    std::vector<Intervention> ivs;
    RunState state() override { return st; }
    Error set_param(const std::string&, const json&) override { return {}; }
    Error run_action(const std::string&) override { return {}; }
    Error pause(RunPauseOp, std::int64_t) override { return {}; }
    bool interventions_supported() override { return true; }
    std::vector<Intervention> interventions() override { return ivs; }
};

}  // namespace

TEST_CASE("a scene panel's library piece goes to the viewer in its instance form, with its newest resource") {
    const std::string hash = library_hash();
    const sp::Spec s = points_spec(hash);
    StepHistory history;
    history.configure(s);   // the open spec's instances, whose resources it keeps
    ViewerInputs in;
    in.library = &PieceLibrary::builtin();
    in.history = &history;
    std::string d0;
    json out = viewer_programs(s, "world", in, &d0);
    REQUIRE(out.size() == 1);
    CHECK(out[0]["id"] == "dots");
    CHECK(out[0]["state"] == "ready");
    CHECK(out[0]["error"].is_null());
    const json& inst = out[0]["instance"];
    CHECK(inst["op"] == "set");
    CHECK(inst["piece"]["hash"] == hash);
    CHECK(inst["panel"] == "world");
    CHECK(inst["params"]["size"] == 3.0);
    CHECK_FALSE(inst["draw"].get<std::string>().empty());
    CHECK(inst["manifest"].is_object());
    CHECK(out[0]["resources"].empty());
    CHECK(d0.size() == 16);
    // Another panel holds none of it.
    CHECK(viewer_programs(s, "other", in, nullptr).empty());

    // The newest version at a live cursor, in base64, and a new digest.
    history.append_resource("dots", "points", 0, {}, version(5, 1));
    history.append_resource("dots", "points", 0, {}, version(9, 2));
    std::string d1;
    out = viewer_programs(s, "world", in, &d1);
    REQUIRE(out[0]["resources"].size() == 1);
    const json& r = out[0]["resources"][0];
    CHECK(r["resource"] == "points");
    CHECK(r["step"] == 9);
    CHECK(r["count"] == 2);
    CHECK(r["stride"] == 24);
    CHECK(r["base64"].get<std::string>().rfind("AgIC", 0) == 0);
    CHECK(d1 != d0);
}

TEST_CASE("viewer resource versions preserve typed metadata and replacements at the same step") {
    const FoundPiece found = find_piece(PieceLibrary::builtin(), "", sp::Spec{}, sp::PieceFrom::library, "points", 1);
    REQUIRE_FALSE(found.error);
    sp::Piece piece = found.piece;
    piece.name = "typed_points";
    sp::Resource narrow;
    narrow.id = "narrow";
    narrow.kind = sp::ResourceKind::buffer;
    narrow.layout.element = "u16";
    narrow.count = 8;
    piece.resources.push_back(narrow);
    sp::Resource texture;
    texture.id = "texture";
    texture.kind = sp::ResourceKind::texture;
    texture.format = "r32uint";
    texture.width = 2;
    texture.height = 2;
    piece.resources.push_back(texture);
    piece.hash.clear();
    piece.hash = programs::piece_hash(piece);
    sp::Spec spec = points_spec(piece.hash);
    spec.pieces.push_back(piece);
    spec.programs[0].piece = sp::PieceRef{piece.name, 1, piece.hash, sp::PieceFrom::inline_piece};
    StepHistory history;
    history.configure(spec);
    ResourceVersion n;
    n.step = 9;
    n.count = 2;
    n.stride = 4;
    n.elements = 3;
    n.bytes = std::make_shared<const std::vector<std::uint8_t>>(8, 1);
    history.append_resource("dots", "narrow", 0, {}, n);
    ResourceVersion image;
    image.step = 9;
    image.count = 4;
    image.stride = 4;
    image.width = image.height = 2;
    image.format = "r32uint";
    image.bytes = std::make_shared<const std::vector<std::uint8_t>>(16, 0);
    history.append_resource("dots", "texture", 0, {}, image);
    ViewerInputs in;
    in.library = &PieceLibrary::builtin();
    in.history = &history;
    std::string first_digest;
    json out = viewer_programs(spec, "world", in, &first_digest);
    REQUIRE(out[0]["resources"].size() == 2);
    CHECK(out[0]["resources"][0]["elements"] == 3);
    CHECK(out[0]["resources"][1]["format"] == "r32uint");
    CHECK(out[0]["resources"][1]["width"] == 2);
    CHECK(out[0]["resources"][1]["height"] == 2);
    const auto first_version = out[0]["resources"][1]["version"];
    auto changed = std::make_shared<std::vector<std::uint8_t>>(16, 0);
    (*changed)[0] = 1; // The unchanged tail cannot identify this replacement.
    image.bytes = changed;
    history.append_resource("dots", "texture", 0, {}, image);
    std::string next_digest;
    out = viewer_programs(spec, "world", in, &next_digest);
    CHECK(out[0]["resources"][1]["version"] != first_version);
    CHECK(next_digest != first_digest);
}

TEST_CASE("a foreign spec's program stays in review for the viewer, and a wrong pin is refused, without files") {
    const std::string hash = library_hash();
    ViewerInputs in;
    in.library = &PieceLibrary::builtin();

    sp::Spec foreign = points_spec(hash);
    foreign.provenance.foreign = true;
    json out = viewer_programs(foreign, "world", in, nullptr);
    REQUIRE(out.size() == 1);
    CHECK(out[0]["state"] == "review");
    CHECK(out[0]["error"]["code"] == "program.needs_review");
    CHECK(out[0]["instance"].is_null());
    CHECK(out[0]["resources"].empty());

    // Reviewed: it goes out.
    sp::ReviewedSource rp;
    rp.target = sp::program_target(foreign.programs[0]);
    rp.sha256 = hash.substr(7);
    foreign.provenance.reviewed.push_back(rp);
    CHECK(viewer_programs(foreign, "world", in, nullptr)[0]["state"] == "ready");

    const std::string other = "sha256:" + std::string(64, '0');
    out = viewer_programs(points_spec(other), "world", in, nullptr);
    CHECK(out[0]["state"] == "refused");
    CHECK(out[0]["instance"].is_null());
    CHECK(out[0]["error"]["code"] == "piece.hash_mismatch");

    sp::Spec missing = points_spec(hash);
    missing.programs[0].piece->name = "no_such_piece";
    out = viewer_programs(missing, "world", in, nullptr);
    CHECK(out[0]["state"] == "refused");
    CHECK(out[0]["instance"].is_null());
}

TEST_CASE("an inline piece goes to the viewer from the spec itself") {
    const FoundPiece f = find_piece(PieceLibrary::builtin(), "", sp::Spec{}, sp::PieceFrom::library, "points", 1);
    REQUIRE_FALSE(f.error);
    sp::Piece inline_piece = f.piece;
    inline_piece.name = "my_points";
    inline_piece.hash.clear();
    const std::string hash = programs::piece_hash(inline_piece);
    inline_piece.hash = hash;
    sp::Spec s = points_spec(hash);
    s.pieces.push_back(inline_piece);
    s.programs[0].piece = sp::PieceRef{"my_points", 1, hash, sp::PieceFrom::inline_piece};
    ViewerInputs in;
    in.library = &PieceLibrary::builtin();
    const json out = viewer_programs(s, "world", in, nullptr);
    REQUIRE(out.size() == 1);
    CHECK(out[0]["state"] == "ready");
    CHECK(out[0]["piece"]["from"] == "inline");
    CHECK(out[0]["instance"]["piece"]["name"] == "my_points");
}

TEST_CASE("a control shows the run's and the session's state to the viewer, which cannot change it") {
    sp::Spec s;
    s.version = 2;
    s.instrument.id = "viewer";
    sp::Control ablate;
    ablate.id = "ablate";
    ablate.kind = sp::ControlKind::toggle;
    ablate.title = "Ablate Head 0 of Layer 1";
    ablate.bind = "intervene:blocks.1.attn#0";
    ViewerInputs in;
    // No run: off, and why.
    json r = control_data_result(s, ablate, in, nullptr, "v1");
    CHECK(r["primitive"] == "control");
    CHECK(r["version"] == "v1");
    CHECK(r["data"]["value"] == false);
    CHECK(r["data"]["enabled"] == false);
    CHECK(r["data"]["why"] == "No script is attached.");
    CHECK(r["data"]["text"] == "ablate blocks.1.attn#0");
    // An active ablation of the point: on, with the app's text.
    FakeRun run;
    run.st.attached = true;
    run.st.control = true;
    Intervention i;
    i.point = "blocks.1.attn#0";
    i.state = "active";
    i.since_step = 412;
    run.ivs.push_back(i);
    r = control_data_result(s, ablate, in, &run, "v2");
    CHECK(r["data"]["value"] == true);
    CHECK(r["data"]["enabled"] == false);
    CHECK(r["data"]["text"] == "ablate blocks.1.attn#0  \xC2\xB7 since step 412");
    CHECK(r["data"]["why"].get<std::string>().find("only reads") != std::string::npos);
    run.ivs[0].state = "ended";
    CHECK(control_data_result(s, ablate, in, &run, "")["data"]["value"] == false);

    // A slider bound to a value: the session's value, else its start.
    sp::Control lr;
    lr.id = "lr";
    lr.kind = sp::ControlKind::slider;
    lr.bind = "value";
    lr.options.min = 0.0;
    lr.options.max = 1.0;
    lr.options.value = 0.25;
    s.controls.push_back(lr);
    r = control_data_result(s, lr, in, nullptr, "");
    CHECK(r["data"]["value"] == 0.25);
    CHECK(r["data"]["min"] == 0.0);
    CHECK(r["data"]["max"] == 1.0);
    CHECK(r["data"]["enabled"] == false);
}

TEST_CASE("panel.data answers a control of the layout to the viewer origin, versioned") {
    const std::string ws = fresh_dir("viewer-controls") + "/workspace";
    std::unique_ptr<Session> s = open_session(ws);
    ok(*s, "spec.new", {{"id", "ctl"}});
    ok(*s, "spec.set_text", {{"text", R"toml(caliper_spec = 2
[instrument]
id = "ctl"
[[control]]
id = "lr"
kind = "slider"
bind = "value"
options = { min = 0.0, max = 2.0, value = 0.5 }
)toml"}});
    ok(*s, "instrument.save", json::object());
    const json a = ok(*s, "panel.data", {{"instrument", "ctl"}, {"panel", "lr"}}, CommandOrigin::viewer);
    CHECK(a["primitive"] == "control");
    CHECK(a["data"]["kind"] == "slider");
    CHECK(a["data"]["value"] == 0.5);
    CHECK(a["data"]["enabled"] == false);
    const json b = ok(*s, "panel.data", {{"instrument", "ctl"}, {"panel", "lr"}, {"known", a["version"]}}, CommandOrigin::viewer);
    CHECK(b["unchanged"] == true);
    // The viewer still may not set it.
    const auto set = invoke(*s, "control.set", {{"id", "lr"}, {"value", 1.0}}, CommandOrigin::viewer);
    CHECK_FALSE(set.ok);
    CHECK(set.error.code == "command.not_allowed");
    const auto none = invoke(*s, "panel.data", {{"instrument", "ctl"}, {"panel", "nothing"}}, CommandOrigin::viewer);
    CHECK_FALSE(none.ok);
    CHECK(none.error.code == "panel.not_found");
    // spec.get says whether the app has the instrument open, which a viewer cannot change.
    CHECK(ok(*s, "spec.get", {{"instrument", "ctl"}}, CommandOrigin::viewer)["open_in_app"] == true);
    ok(*s, "spec.new", {{"id", "other"}});
    CHECK(ok(*s, "spec.get", {{"instrument", "ctl"}}, CommandOrigin::viewer)["open_in_app"] == false);
    const auto reopen = invoke(*s, "instrument.open", {{"id", "ctl"}}, CommandOrigin::viewer);
    CHECK_FALSE(reopen.ok);
    CHECK(reopen.error.code == "command.not_allowed");
}

TEST_CASE("viewer readouts use current unplaced script parameters") {
    sp::Spec s;
    sp::Control source;
    source.id = "status_value";
    source.kind = sp::ControlKind::text;
    source.bind = "param:clamp.status";
    s.controls.push_back(source);
    sp::Control readout;
    readout.id = "status";
    readout.kind = sp::ControlKind::readout;
    sp::Reference ref;
    ref.kind = sp::Reference::Kind::control;
    ref.target = source.id;
    ref.fallback = "Unavailable";
    sp::BoundText text;
    text.format = "{status:s}";
    text.fields["status"] = ref;
    readout.options.bound_text = text;
    ViewerInputs in;
    FakeRun run;
    run.st.attached = true;
    RunParam param;
    param.name = "clamp.status";
    param.type = "text";
    param.value = "No slider clamp.";
    run.st.params.push_back(param);
    const auto shown = [&] { return control_data_result(s, readout, in, &run, "") ["data"]["text"]; };
    CHECK(shown() == "No slider clamp.");
    run.st.params.back().value = "L2 F4. Clamp 0.";
    CHECK(shown() == "L2 F4. Clamp 0.");
    run.st.params.back().pending = "Pending clamp.";
    CHECK(shown() == "Pending clamp.");
    run.st.attached = false;
    CHECK(shown() == "Unavailable");
    run.st.attached = true;
    run.st.params.clear();
    CHECK(shown() == "Unavailable");
    CHECK(control_data_result(s, readout, in, nullptr, "")["data"]["text"] == "Unavailable");
}
