// Tests of the sources of pieces and the piece commands (piece_commands.h,
// design/programmable-graphics.md: Piece Files, Piece Sources, Versions and
// Pins, Piece Commands): the seven commands on the library, the workspace and
// the inline pieces, the hash and the pins of program.create and
// program.patch, and a test for each refusal of a path that would leave the
// workspace's pieces folder.
#include "helpers.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <fstream>
#include <set>

using namespace caliper_builder;
using namespace session_test;

namespace {

const char* const kFieldDraw =
    "@fragment\nfn fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {\n"
    "    return colormap::magma(in.uv.x * params.gain);\n}\n";

// A draw-only piece with one parameter, as an agent writes it for piece.update.
json uv_field(const std::string& name = "uv_field") {
    return {{"name", name},
            {"description", "A field of the uv coordinates"},
            {"imports", {"colormap@1"}},
            {"params", {{{"name", "gain"}, {"default", 1.0}, {"min", 0.0}, {"max", 4.0}}}},
            {"draw", {{"camera", "none"}, {"passes", {{{"kind", "render"}, {"fragment", "fs"}}}}}},
            {"files", {{"draw", kFieldDraw}}}};
}

// A piece with a tensor input and a published buffer that a second piece reads.
json rows_piece() {
    return {{"name", "rows"},
            {"inputs", {{{"name", "w"}, {"kind", "tensor"}, {"rank_min", 2}, {"rank_max", 2}}}},
            {"resources", {{{"id", "row"}, {"kind", "buffer"}, {"element", "f32"}, {"count", 64}}}},
            {"derive", {{"passes", {{{"entry", "copy"}, {"writes", {"row"}}}}}}},
            {"files",
             {{"derive", "@compute @workgroup_size(64)\nfn copy(@builtin(global_invocation_id) id: vec3<u32>) {\n"
                         "    row[id.x] = w[id.x];\n}\n"}}}};
}

json bars_piece() {
    return {{"name", "bars"},
            {"inputs", {{{"name", "values"}, {"kind", "buffer"}, {"element", "f32"}}}},
            {"draw", {{"camera", "none"}, {"passes", {{{"kind", "render"}, {"fragment", "fs"}}}}}},
            {"files",
             {{"draw", "@fragment\nfn fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {\n"
                       "    return vec4<f32>(values[0], in.uv, 1.0);\n}\n"}}}};
}

std::unique_ptr<Session> piece_session(const std::string& name, std::string* workspace = nullptr) {
    const std::string ws = fresh_dir(name) + "/workspace";
    if (workspace) *workspace = ws;
    auto s = open_session(ws);
    ok(*s, "spec.new", {{"id", "pieces"}});
    ok(*s, "panel.create", {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    return s;
}

// A small library for the commands that read it: one piece, two versions.
const PieceLibrary& test_library() {
    static const std::string v1 =
        "name = \"glow\"\nversion = 1\ndescription = \"A glow\"\n\n[draw]\ncamera = \"none\"\n"
        "passes = [{ kind = \"render\", fragment = \"fs\" }]\n";
    static const std::string v2 =
        "name = \"glow\"\nversion = 2\ndescription = \"A brighter glow\"\n\n[draw]\ncamera = \"none\"\n"
        "passes = [{ kind = \"render\", fragment = \"fs\" }]\n";
    static const std::string draw =
        "@fragment\nfn fs(in: CaliperFullscreen) -> @location(0) vec4<f32> { return vec4<f32>(in.uv, 0.0, 1.0); }\n";
    static const PieceLibrary library({{"lib/colormap/1.wgsl", "// x\n"},
                                       {"pieces/glow/1/draw.wgsl", draw},
                                       {"pieces/glow/1/piece.toml", v1},
                                       {"pieces/glow/2/draw.wgsl", draw},
                                       {"pieces/glow/2/piece.toml", v2},
                                       {"pieces/stray/1/piece.toml", "name = \"other\"\n"}});
    return library;
}

CommandOutput with_library(const std::string& name, const json& args, const spec::Spec& spec,
                           const std::string& workspace) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = args;
    const CommandDescriptor* d = find_command_descriptor(name);
    REQUIRE(d);
    CommandContext ctx{inv, *d};
    return piece_command(ctx, spec, workspace, test_library());
}

bool has_code(const json& messages, const std::string& code) {
    for (const json& m : messages)
        if (m.value("code", "") == code) return true;
    return false;
}

}  // namespace

TEST_CASE("pieces: piece.check checks a piece alone and gives its hash and messages by file") {
    auto s = piece_session("piece_check");
    json r = ok(*s, "piece.check", {{"piece", uv_field()}});
    CHECK(r["ok"] == true);
    CHECK(r["hash"].get<std::string>().rfind("sha256:", 0) == 0);
    json bad = uv_field();
    bad["files"]["draw"] = std::string(kFieldDraw) + "fn caliper_x() {}\n";
    r = ok(*s, "piece.check", {{"piece", bad}});
    CHECK(r["ok"] == false);
    REQUIRE(has_code(r["messages"], "program.reserved_name"));
    for (const json& m : r["messages"])
        if (m["code"] == "program.reserved_name") {
            CHECK(m["file"] == "draw.wgsl");
            CHECK(m["line"] == 5);
        }
    bad = uv_field();
    bad["files"]["draw"] = "fn f() -> vec4<f32> { return colormap::plasma(0.5); }\n";
    r = ok(*s, "piece.check", {{"piece", bad}});
    CHECK(has_code(r["messages"], "piece.unknown_import"));
    bad = uv_field();
    bad["draw"]["passes"][0]["target"] = "glow";
    r = ok(*s, "piece.check", {{"piece", bad}});
    CHECK(has_code(r["messages"], "program.bad_pass"));
    bad = uv_field();
    bad["hash"] = "sha256:" + std::string(64, '0');
    r = ok(*s, "piece.check", {{"piece", bad}});
    CHECK(has_code(r["messages"], "piece.hash_mismatch"));
}

TEST_CASE("pieces: piece.update writes the next version, and saved versions never change") {
    std::string ws;
    auto s = piece_session("piece_update", &ws);
    json r = ok(*s, "piece.update", {{"piece", uv_field()}});
    CHECK(r["version"] == 1);
    CHECK(r["path"] == "pieces/uv_field/1");
    const std::string hash1 = r["hash"];
    CHECK(fs::exists(ws + "/pieces/uv_field/1/piece.toml"));
    CHECK(read_text(ws + "/pieces/uv_field/1/draw.wgsl") == kFieldDraw);
    CHECK(read_text(ws + "/pieces/uv_field/1/piece.toml").find(hash1) != std::string::npos);
    json next = uv_field();
    next["params"][0]["default"] = 2.0;
    r = ok(*s, "piece.update", {{"piece", next}});
    CHECK(r["version"] == 2);
    CHECK(r["hash"] != hash1);
    next["version"] = 2;
    CHECK(err(*s, "piece.update", {{"piece", next}}) == "piece.exists");
    next["version"] = 1;
    CHECK(err(*s, "piece.update", {{"piece", next}}) == "piece.exists");
    json bad = uv_field();
    bad["files"]["draw"] = "@group(0) @binding(9) var<uniform> u: f32;\n";
    CHECK(err(*s, "piece.update", {{"piece", bad}}) == "program.binding_not_allowed");

    r = ok(*s, "piece.show", {{"name", "uv_field"}});
    CHECK(r["from"] == "workspace");
    CHECK(r["piece"]["version"] == 2);
    CHECK(r["path"] == "pieces/uv_field/2");
    CHECK(r["piece"]["files"]["draw"] == kFieldDraw);
    r = ok(*s, "piece.show", {{"name", "uv_field"}, {"version", 1}, {"from", "workspace"}});
    CHECK(r["piece"]["hash"] == hash1);
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 3}}) == "piece.not_found");

    r = ok(*s, "piece.list", {{"from", "workspace"}});
    REQUIRE(r["pieces"].size() == 2);
    CHECK(r["pieces"][1]["version"] == 2);
    CHECK(r["pieces"][0]["params"] == json::array({"gain"}));
    CHECK(r["pieces"][0]["draws"] == true);
    std::set<std::string> modules;
    for (const json& m : r["modules"]) modules.insert(m.get<std::string>());
    CHECK(modules == std::set<std::string>{"camera@1", "colormap@1", "easing@1", "pick@1"});
}

TEST_CASE("pieces: piece.update writes a sample whose bytes match its shape") {
    std::string ws;
    auto s = piece_session("piece_sample", &ws);
    json p = rows_piece();
    p["sample"] = {{{"input", "w"}, {"shape", {2, 2}}, {"data", "AACAPwAAAEAAAEBAAACAQA=="}}};  // 1, 2, 3, 4
    ok(*s, "piece.update", {{"piece", p}});
    CHECK(fs::file_size(ws + "/pieces/rows/1/sample/w.bin") == 16);
    p["sample"][0]["shape"] = {2, 3};
    CHECK(err(*s, "piece.update", {{"piece", p}}) == "piece.bad_manifest");
    p["sample"][0]["data"] = "not base64!";
    CHECK(err(*s, "piece.update", {{"piece", p}}) == "piece.bad_manifest");
}

TEST_CASE("pieces: a write leaves the half-written versions of other processes on the workspace alone") {
    std::string ws;
    auto s = piece_session("piece_tmp", &ws);
    ok(*s, "piece.update", {{"piece", uv_field()}});
    // Another app process on the same workspace writes version 2 under every name that a counter of its own
    // could give; a write of this process never removes one of them.
    const fs::path dir = fs::path(ws) / "pieces" / "uv_field";
    for (int k = 0; k <= 512; ++k) {
        const fs::path other = dir / (".tmp-2-" + std::to_string(k));
        REQUIRE(fs::create_directory(other));
        std::ofstream(other / "piece.toml") << "half";
    }
    json next = uv_field();
    next["params"][0]["default"] = 2.0;
    CHECK(ok(*s, "piece.update", {{"piece", next}})["version"] == 2);
    int left = 0;
    for (const auto& entry : fs::directory_iterator(dir))
        left += entry.path().filename().string().rfind(".tmp-", 0) == 0;
    CHECK(left == 513);   // its own temporary folder became version 2, and no other one went
    int intact = 0;
    for (int k = 0; k <= 512; ++k)
        intact += read_text((dir / (".tmp-2-" + std::to_string(k)) / "piece.toml").string()) == "half";
    CHECK(intact == 513);
    CHECK(fs::exists(dir / "2" / "piece.toml"));
}

TEST_CASE("pieces: names and versions that would leave the pieces folder are refused") {
    std::string ws;
    auto s = piece_session("piece_names", &ws);
    for (const char* name : {"../x", "..", "/etc", "a/b", "UV", ".hidden"}) {
        CAPTURE(name);
        CHECK(err(*s, "piece.show", {{"name", name}}) == "command.invalid_args");
        CHECK(err(*s, "piece.remove", {{"name", name}, {"version", 1}}) == "command.invalid_args");
        json p = uv_field();
        p["name"] = name;
        CHECK(err(*s, "piece.update", {{"piece", p}}) == "command.invalid_args");
    }
    const spec::Spec empty;
    for (const char* name : {"../x", "..", "a/../../b", "x\\y"}) {
        CAPTURE(name);
        CHECK(find_piece(test_library(), ws, empty, spec::PieceFrom::workspace, name, std::nullopt).error.code ==
              "piece.bad_name");
    }
    CHECK(find_piece(test_library(), ws, empty, spec::PieceFrom::workspace, "uv", 0).error.code == "piece.bad_name");
    CHECK(!fs::exists(ws + "/x"));
}

#ifndef _WIN32
TEST_CASE("pieces: a link anywhere in the pieces folder is refused, and nothing outside is read or removed") {
    std::string ws;
    auto s = piece_session("piece_links", &ws);
    ok(*s, "piece.update", {{"piece", uv_field()}});
    const std::string outside = fresh_dir("piece_links_outside");
    write_text(outside + "/secret.wgsl", "fn secret() {}\n");
    // A piece folder that is a link to a copy outside the workspace.
    fs::create_directories(outside + "/copy");
    fs::copy(ws + "/pieces/uv_field/1", outside + "/copy/1", fs::copy_options::recursive);
    fs::create_directory_symlink(outside + "/copy", ws + "/pieces/linked");
    const spec::Spec empty;
    CHECK(find_piece(test_library(), ws, empty, spec::PieceFrom::workspace, "linked", 1).error.code ==
          "piece.bad_path");
    // A version folder that is a link.
    fs::create_directory_symlink(outside + "/copy/1", ws + "/pieces/uv_field/7");
    CHECK(find_piece(test_library(), ws, empty, spec::PieceFrom::workspace, "uv_field", 7).error.code ==
          "piece.bad_path");
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 7}}) == "piece.bad_path");
    // A file of a version that is a link to a file outside.
    fs::create_directories(ws + "/pieces/uv_field/8");
    fs::copy(ws + "/pieces/uv_field/1/piece.toml", ws + "/pieces/uv_field/8/piece.toml");
    fs::create_symlink(outside + "/secret.wgsl", ws + "/pieces/uv_field/8/draw.wgsl");
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 8}}) == "piece.bad_path");
    // A link in the sample folder.
    fs::create_directories(ws + "/pieces/uv_field/9/sample");
    fs::copy(ws + "/pieces/uv_field/1/piece.toml", ws + "/pieces/uv_field/9/piece.toml");
    fs::copy(ws + "/pieces/uv_field/1/draw.wgsl", ws + "/pieces/uv_field/9/draw.wgsl");
    fs::create_symlink(outside + "/secret.wgsl", ws + "/pieces/uv_field/9/sample/w.bin");
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 9}}) == "piece.bad_path");
    // Removing a version removes the link, never the file it names.
    fs::remove(ws + "/pieces/uv_field/7");
    ok(*s, "piece.remove", {{"name", "uv_field"}, {"version", 9}});
    CHECK(!fs::exists(ws + "/pieces/uv_field/9"));
    CHECK(fs::exists(outside + "/secret.wgsl"));
    CHECK(err(*s, "piece.remove", {{"name", "linked"}, {"version", 1}}) == "piece.bad_path");
    CHECK(fs::exists(outside + "/copy/1/draw.wgsl"));
    // The pieces folder itself as a link.
    const std::string ws2 = fresh_dir("piece_links_root") + "/workspace";
    fs::create_directories(ws2);
    fs::create_directory_symlink(outside + "/copy", ws2 + "/pieces");
    auto s2 = open_session(ws2);
    CHECK(err(*s2, "piece.show", {{"name", "uv_field"}, {"from", "workspace"}}) == "piece.bad_path");
    CHECK(err(*s2, "piece.update", {{"piece", uv_field()}}) == "piece.bad_path");
    CHECK(!fs::exists(outside + "/copy/uv_field"));
}
#endif

TEST_CASE("pieces: a saved version whose manifest, size or hash changed is refused") {
    std::string ws;
    auto s = piece_session("piece_tamper", &ws);
    ok(*s, "piece.update", {{"piece", uv_field()}});
    ok(*s, "piece.update", {{"piece", uv_field()}});
    write_text(ws + "/pieces/uv_field/2/draw.wgsl", std::string(kFieldDraw) + "// changed\n");
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 2}}) == "piece.hash_mismatch");
    write_text(ws + "/pieces/uv_field/2/draw.wgsl", std::string(70000, ' '));
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 2}}) == "piece.bad_path");
    std::string manifest = read_text(ws + "/pieces/uv_field/1/piece.toml");
    manifest.replace(manifest.find("version = 1"), 11, "version = 5");
    write_text(ws + "/pieces/uv_field/1/piece.toml", manifest);
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 1}}) == "piece.bad_manifest");
    write_text(ws + "/pieces/uv_field/3/draw.wgsl", kFieldDraw);
    CHECK(err(*s, "piece.show", {{"name", "uv_field"}, {"version", 3}}) == "piece.bad_manifest");
}

TEST_CASE("pieces: the library is read-only; piece.copy forks it into the workspace as version 1") {
    std::string ws;
    auto s = piece_session("piece_copy", &ws);
    const spec::Spec open = s->open_spec()->spec;
    REQUIRE(test_library().pieces().size() == 2);  // the stray manifest names another folder
    CommandOutput r = with_library("piece.show", {{"name", "glow"}}, open, ws);
    REQUIRE(r.error.ok());
    CHECK(r.result["from"] == "library");
    CHECK(r.result["piece"]["version"] == 2);
    CHECK(r.result["path"] == "caliper://library/pieces/glow/2/");
    r = with_library("piece.copy", {{"name", "glow"}, {"version", 1}, {"as", "my_glow"}}, open, ws);
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.result["version"] == 1);
    CHECK(r.result["path"] == "pieces/my_glow/1");
    CHECK(fs::exists(ws + "/pieces/my_glow/1/draw.wgsl"));
    CHECK(with_library("piece.copy", {{"name", "glow"}, {"as", "my_glow"}}, open, ws).error.code == "piece.exists");
    CHECK(with_library("piece.copy", {{"name", "dim"}}, open, ws).error.code == "piece.not_found");
    CHECK(with_library("piece.remove", {{"name", "glow"}, {"version", 1}}, open, ws).error.code == "piece.read_only");
    r = with_library("piece.list", json::object(), open, ws);
    REQUIRE(r.error.ok());
    CHECK(r.result["pieces"].size() == 3);
    r = with_library("piece.list", {{"from", "library"}, {"match", "glo"}}, open, ws);
    CHECK(r.result["pieces"].size() == 2);
    CHECK(err(*s, "piece.remove", {{"name", "nothing"}, {"version", 1}}) == "piece.not_found");
}

TEST_CASE("pieces: program.create pins a workspace piece with its hash and checks its bindings") {
    std::string ws;
    auto s = piece_session("piece_pins", &ws);
    const std::string hash = ok(*s, "piece.update", {{"piece", uv_field()}})["hash"];
    ok(*s, "program.create",
       {{"id", "field"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"},
        {"params", {{"gain", 2.0}}}});
    const spec::Spec created = s->open_spec()->spec;
    const spec::Program* p = created.find_program("field");
    REQUIRE(p);
    REQUIRE(p->piece);
    CHECK(p->piece->hash == hash);
    CHECK(p->piece->version == 1);
    json status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["kind"] == "piece");
    CHECK(status["programs"][0]["piece"]["hash"] == hash);

    CHECK(err(*s, "program.create", {{"id", "a"}, {"piece", {{"name", "gone"}, {"from", "workspace"}}}, {"panel", "world"}}) ==
          "piece.missing");
    CHECK(err(*s, "program.create",
              {{"id", "a"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}, {"hash", "sha256:" + std::string(64, 'a')}}},
               {"panel", "world"}}) == "piece.hash_mismatch");
    CHECK(err(*s, "program.create", {{"id", "a"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}}) ==
          "spec.missing_key");
    CHECK(err(*s, "program.create",
              {{"id", "a"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"},
               {"params", {{"brightness", 1.0}}}}) == "program.params_mismatch");
    CHECK(err(*s, "program.create",
              {{"id", "a"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"},
               {"bind", {{"extra", "live:x"}}}}) == "piece.bad_binding");
    CHECK(err(*s, "program.create",
              {{"id", "a"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"},
               {"wgsl", "fn f() {}"}}) == "program.form_mixed");

    // A tensor input bound once, and a buffer input bound to a published buffer with its layout.
    ok(*s, "piece.update", {{"piece", rows_piece()}});
    ok(*s, "piece.update", {{"piece", bars_piece()}});
    CHECK(err(*s, "program.create", {{"id", "rows"}, {"piece", {{"name", "rows"}, {"from", "workspace"}}}}) ==
          "piece.bad_binding");
    ok(*s, "program.create",
       {{"id", "rows"}, {"piece", {{"name", "rows"}, {"from", "workspace"}}}, {"bind", {{"w", "live:w"}}}});
    CHECK(err(*s, "program.create",
              {{"id", "bars"}, {"piece", {{"name", "bars"}, {"from", "workspace"}}}, {"panel", "world"},
               {"bind", {{"values", "program:rows/nothing"}}}}) == "piece.bad_binding");
    ok(*s, "program.create",
       {{"id", "bars"}, {"piece", {{"name", "bars"}, {"from", "workspace"}}}, {"panel", "world"},
        {"bind", {{"values", "program:rows/row"}}}});

    // program.patch pins the next version, with its hash.
    json next = uv_field();
    next["params"][0]["default"] = 3.0;
    const std::string hash2 = ok(*s, "piece.update", {{"piece", next}})["hash"];
    CHECK(s->open_spec()->spec.programs[0].piece->hash == hash);
    ok(*s, "program.patch", {{"id", "field"}, {"patch", {{"piece", {{"name", "uv_field"}, {"version", 2}, {"from", "workspace"}}}}}});
    CHECK(s->open_spec()->spec.programs[0].piece->hash == hash2);
    CHECK(err(*s, "piece.remove", {{"name", "uv_field"}, {"version", 2}}) == "piece.in_use");
    ok(*s, "piece.remove", {{"name", "uv_field"}, {"version", 1}});
    CHECK(err(*s, "program.patch", {{"id", "field"}, {"patch", {{"piece", {{"name", "uv_field"}, {"version", 1}, {"from", "workspace"}}}}}}) ==
          "piece.missing");
}

TEST_CASE("pieces: two cameras in one panel are refused by the editor and named by spec.validate") {
    std::string ws;
    auto s = piece_session("piece_cameras", &ws);
    ok(*s, "piece.update", {{"piece", uv_field()}});   // camera none
    json orbit = uv_field("orbit_field");
    orbit["draw"]["camera"] = "orbit";
    ok(*s, "piece.update", {{"piece", orbit}});
    ok(*s, "panel.create", {{"id", "side"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    const json flat_pin = {{"name", "uv_field"}, {"from", "workspace"}};
    const json orbit_pin = {{"name", "orbit_field"}, {"from", "workspace"}};

    // One camera in a panel, and another camera in another panel, pass.
    ok(*s, "program.create", {{"id", "flat"}, {"piece", flat_pin}, {"panel", "world"}});
    ok(*s, "program.create", {{"id", "flat_too"}, {"piece", flat_pin}, {"panel", "world"}});
    ok(*s, "program.create", {{"id", "round"}, {"piece", orbit_pin}, {"panel", "side"}});
    CHECK(ok(*s, "spec.validate")["valid"] == true);

    // Another camera in the same panel is refused before it applies, by create and by patch.
    CHECK(err(*s, "program.create", {{"id", "round_too"}, {"piece", orbit_pin}, {"panel", "world"}}) ==
          "program.camera_conflict");
    CHECK(err(*s, "program.patch", {{"id", "round"}, {"patch", {{"panel", "world"}}}}) == "program.camera_conflict");
    CHECK(s->open_spec()->spec.programs.size() == 3);

    // spec.validate names the later program of a conflict in a text and in the open spec.
    std::string text = ok(*s, "spec.get")["text"];
    const std::string side = "panel = \"side\"";
    REQUIRE(text.find(side) != std::string::npos);
    REQUIRE(text.find(side) == text.rfind(side));
    text.replace(text.find(side), side.size(), "panel = \"world\"");
    auto conflict = [](const json& result) {
        CHECK(result["valid"] == false);
        REQUIRE(result["errors"].size() == 1);
        CHECK(result["errors"][0]["code"] == "program.camera_conflict");
        CHECK(result["errors"][0]["where"] == "program[2].piece");
    };
    conflict(ok(*s, "spec.validate", {{"text", text}}));
    ok(*s, "spec.set_text", {{"text", text}});
    conflict(ok(*s, "spec.validate"));
}

TEST_CASE("pieces: program.create with inline adds the piece, and piece.save extracts it in one undo step") {
    std::string ws;
    auto s = piece_session("piece_inline", &ws);
    ok(*s, "program.create", {{"id", "field"}, {"inline", uv_field("inline_field")}, {"panel", "world"}});
    spec::Spec open = s->open_spec()->spec;  // a copy
    REQUIRE(open.pieces.size() == 1);
    const spec::Program* p = open.find_program("field");
    REQUIRE(p);
    CHECK(p->piece->from == spec::PieceFrom::inline_piece);
    CHECK(!p->piece->hash.empty());
    json r = ok(*s, "piece.list", {{"from", "inline"}});
    REQUIRE(r["pieces"].size() == 1);
    CHECK(r["pieces"][0]["pinned_by"] == json::array({"field"}));
    r = ok(*s, "piece.show", {{"name", "inline_field"}, {"from", "inline"}});
    CHECK(r["path"].is_null());
    json other = uv_field("inline_field");
    other["description"] = "Another";
    CHECK(err(*s, "program.create", {{"id", "g"}, {"inline", other}, {"panel", "world"}}) == "piece.exists");

    r = ok(*s, "piece.save", {{"name", "inline_field"}, {"dry_run", true}});
    CHECK(r["applied"] == false);
    CHECK(!fs::exists(ws + "/pieces/inline_field/1"));
    r = ok(*s, "piece.save", {{"name", "inline_field"}});
    CHECK(r["programs"] == json::array({"field"}));
    CHECK(r["path"] == "pieces/inline_field/1");
    CHECK(fs::exists(ws + "/pieces/inline_field/1/draw.wgsl"));
    open = s->open_spec()->spec;
    CHECK(open.pieces.empty());
    CHECK(open.find_program("field")->piece->from == spec::PieceFrom::workspace);
    CHECK(open.find_program("field")->piece->hash == r["hash"]);
    ok(*s, "history.undo");
    open = s->open_spec()->spec;
    CHECK(open.pieces.size() == 1);
    CHECK(open.find_program("field")->piece->from == spec::PieceFrom::inline_piece);
    CHECK(err(*s, "piece.save", {{"name", "inline_field"}}) == "piece.exists");
    CHECK(err(*s, "piece.save", {{"name", "nothing"}}) == "piece.not_found");
}

TEST_CASE("pieces: a spec that pins a missing or changed piece opens, and its program is refused") {
    std::string ws;
    auto s = piece_session("piece_refused", &ws);
    const std::string hash = ok(*s, "piece.update", {{"piece", uv_field()}})["hash"];
    ok(*s, "program.create",
       {{"id", "field"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"}});
    write_text(ws + "/pinned.spec.toml", spec::to_toml(s->open_spec()->spec));
    write_text(ws + "/pieces/uv_field/1/draw.wgsl", std::string(kFieldDraw) + "// changed\n");
    const std::uint64_t cursor = s->events().latest();
    ok(*s, "spec.open", {{"path", "pinned.spec.toml"}});
    REQUIRE(s->open_spec());
    const EventRead read = s->events().read(cursor, EventFilter{}, 64);
    bool refused = false;
    for (const Event& e : read.events)
        if (e.kind == EventKind::program && e.fields.value("change", "") == "refused" &&
            e.fields.value("program", "") == "field")
            refused = e.fields.value("code", "") == "piece.hash_mismatch";
    CHECK(refused);
    json status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["state"] == "refused");
    CHECK(status["programs"][0]["error"]["code"] == "piece.hash_mismatch");
    fs::remove_all(ws + "/pieces/uv_field");
    status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["error"]["code"] == "piece.missing");
    CHECK(hash.size() == 71);
}

// A pin needs its hash. The editor writes it only into the pins of
// program.create and program.patch, thus a hand-written pin without one opens
// and refuses its program with spec.missing_key, which is no hash mismatch;
// program.patch with the same pin writes the hash.
TEST_CASE("pieces: a spec file whose pin has no hash opens, and its program is refused as a missing key") {
    std::string ws;
    auto s = piece_session("piece_no_hash", &ws);
    ok(*s, "piece.update", {{"piece", uv_field()}});
    ok(*s, "program.create",
       {{"id", "field"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"}});
    spec::Spec saved = s->open_spec()->spec;
    REQUIRE(saved.programs.size() == 1);
    REQUIRE(saved.programs[0].piece);
    CHECK(!saved.programs[0].piece->hash.empty());
    saved.programs[0].piece->hash.clear();
    const std::string text = spec::to_toml(saved);
    CHECK(text.find("sha256:") == std::string::npos);
    write_text(ws + "/unhashed.spec.toml", text);
    const std::uint64_t cursor = s->events().latest();
    ok(*s, "spec.open", {{"path", "unhashed.spec.toml"}});
    REQUIRE(s->open_spec());
    std::string code;
    for (const Event& e : s->events().read(cursor, EventFilter{}, 64).events)
        if (e.kind == EventKind::program && e.fields.value("change", "") == "refused" &&
            e.fields.value("program", "") == "field")
            code = e.fields.value("code", "");
    CHECK(code == "spec.missing_key");
    json status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["state"] == "refused");
    CHECK(status["programs"][0]["error"]["code"] == "spec.missing_key");
    ok(*s, "program.patch",
       {{"id", "field"}, {"patch", {{"piece", {{"name", "uv_field"}, {"version", 1}, {"from", "workspace"}}}}}});
    REQUIRE(s->open_spec()->spec.programs[0].piece);
    CHECK(s->open_spec()->spec.programs[0].piece->hash.size() == 71);
    status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["state"] != "refused");
}

namespace {

struct ViewerHost : StageHost {
    std::vector<ProgramStatus> items;
    void reconcile(const spec::Spec&) override {}
    std::vector<ProgramStatus> status() const override { return items; }
};

struct Events : EventSink {
    std::vector<Event> got;
    std::uint64_t push(Event event) override {
        got.push_back(std::move(event));
        return got.size();
    }
};

}  // namespace

TEST_CASE("pieces: program.status takes an instance's draw object from the viewer's host, and events name the half") {
    auto s = piece_session("piece_draw_status");
    ok(*s, "piece.update", {{"piece", uv_field()}});
    ok(*s, "program.create", {{"id", "field"}, {"piece", {{"name", "uv_field"}, {"from", "workspace"}}}, {"panel", "world"}});
    // Without a host that draws it, the manifest gives the limits.
    json status = ok(*s, "program.status", {{"id", "field"}});
    CHECK(status["programs"][0]["draw"]["state"] == "waiting");
    CHECK(status["programs"][0]["draw"]["frame_ms"].is_null());

    ViewerHost viewer;
    ProgramStatus item;
    item.id = "field";
    item.kind = "piece";
    item.host = "viewer";
    item.half = "draw";
    DrawStatus d;
    d.state = "stopped";
    d.frame_ms = 6.5;
    d.frame_limit_ms = 4;
    d.memory_bytes = 1024;
    d.memory_limit_bytes = 64u << 20;
    d.resource_step = 120;
    d.error = ProgramError{"program.draw_over_time", "The pass draw.passes[0] took 6.5 ms of its 4 ms.", 0, 0};
    d.error_where = "draw.passes[0]";
    item.draw = d;
    viewer.items = {item};
    s->set_stage_host(&viewer);
    status = ok(*s, "program.status", {{"id", "field"}});
    const json& draw = status["programs"][0]["draw"];
    CHECK(draw["state"] == "stopped");
    CHECK(draw["frame_ms"] == 6.5);
    CHECK(draw["memory_bytes"] == 1024);
    CHECK(draw["resource_step"] == 120);
    CHECK(draw["error"]["code"] == "program.draw_over_time");
    CHECK(draw["error"]["where"] == "draw.passes[0]");
    CHECK(status["stages"].empty());
    s->set_stage_host(nullptr);

    // The event of an instance's half names the half and its pass; another program's has neither.
    Events events;
    item.state = "stopped";
    item.pass = 0;
    item.error = d.error;
    push_program_event(events, item, "stopped");
    ProgramStatus plain;
    plain.id = "rows";
    push_program_event(events, plain, "compiled");
    REQUIRE(events.got.size() == 2);
    CHECK(events.got[0].fields["half"] == "draw");
    CHECK(events.got[0].fields["pass"] == 0);
    CHECK(!events.got[1].fields.contains("half"));
    CHECK(!events.got[1].fields.contains("pass"));
    // Both keys leave through the events' JSON, which keeps only the keys of the kind.
    const json out = event_json(events.got[0]);
    CHECK(out["half"] == "draw");
    CHECK(out["pass"] == 0);
    CHECK(out["code"] == "program.draw_over_time");
}

TEST_CASE("pieces: selection.set picks an item of a program that pins a workspace piece") {
    auto s = piece_session("piece_pick_workspace");
    const json piece = {
        {"name", "picked_rows"},
        {"imports", {"pick@1"}},
        {"inputs", {{{"name", "w"}, {"kind", "tensor"}, {"rank_min", 2}, {"rank_max", 2}}}},
        {"resources", {{{"id", "rows"}, {"kind", "buffer"}, {"struct", "Row"}, {"fields", {"row: u32", "norm: f32"}}, {"count", 4}}}},
        {"derive", {{"passes", {{{"entry", "run"}, {"writes", {"rows"}}}}}}},
        {"draw",
         {{"camera", "none"},
          {"passes", {{{"kind", "render"}, {"fragment", "fs"}, {"pick", true}}}},
          {"pick", {{"resource", "rows"}, {"fields", {"row", "norm"}}}}}},
        {"files",
         {{"derive", "@compute @workgroup_size(4)\nfn run(@builtin(global_invocation_id) id: vec3<u32>) {\n"
                     "    rows[id.x] = Row(id.x, w[id.x]);\n}\n"},
          {"draw", "@fragment\nfn fs(in: CaliperFullscreen) -> CaliperPicked {\n    var out: CaliperPicked;\n"
                   "    out.color = vec4<f32>(rows[0].norm, in.uv, 1.0);\n    out.id = pick::id(0u);\n    return out;\n}\n"}}}};
    ok(*s, "piece.update", {{"piece", piece}});
    ok(*s, "program.create",
       {{"id", "rows"}, {"piece", {{"name", "picked_rows"}, {"from", "workspace"}}}, {"panel", "world"},
        {"bind", {{"w", "live:w"}}}});
    struct Row {
        std::uint32_t row;
        float norm;
    };
    const std::vector<Row> rows = {{0, 0.5f}, {1, 0.75f}, {2, 1.0f}, {3, 1.25f}};
    ResourceVersion v;
    v.step = 10;
    v.count = 4;
    v.stride = sizeof(Row);
    const auto* b = reinterpret_cast<const std::uint8_t*>(rows.data());
    v.bytes = std::make_shared<const std::vector<std::uint8_t>>(b, b + rows.size() * sizeof(Row));
    s->history().append_resource("rows", "rows", 4, {}, v);
    const json r = ok(*s, "selection.set", {{"panel", "world"}, {"program", "rows"}, {"item", 1}});
    CHECK(r["fields"]["program"] == "rows");
    CHECK(r["fields"]["item"] == 1);
    CHECK(r["fields"]["row"] == 1);
    CHECK(r["fields"]["norm"] == 0.75);
    // The pick event keeps its program and item in the events' JSON.
    Event pick;
    pick.kind = EventKind::pick;
    pick.fields = r["fields"];
    const json out = event_json(pick);
    CHECK(out["program"] == "rows");
    CHECK(out["item"] == 1);
}
