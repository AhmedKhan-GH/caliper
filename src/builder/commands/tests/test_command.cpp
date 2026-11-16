// Tests of the command model (names, descriptors, invocations, results and
// their JSON forms) and of SHA-256.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/sha256.h>

#include <doctest/doctest.h>

#include <string>

using caliper_builder::CommandDescriptor;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;
using nlohmann::json;
namespace cb = caliper_builder;

TEST_CASE("command names are group.verb, and tool names map both ways") {
    for (const char* good : {"panel.create", "spec.set_text", "history.undo", "app2.x9", "a.b"})
        CHECK(cb::check_command_name(good).ok());
    for (const char* bad : {"", "panel", "panel.", ".create", "Panel.create", "panel.Create", "pa_nel.create",
                            "panel.create.now", "panel.9x", "9p.x", "panel.cre-ate", "panel create"})
        CHECK(cb::check_command_name(bad).code == "command.invalid_invocation");
    CHECK(cb::check_command_name("a." + std::string(62, 'b')).ok());
    CHECK(cb::check_command_name("a." + std::string(63, 'b')).code == "command.invalid_invocation");

    CHECK(cb::command_tool_name("panel.create") == "panel_create");
    CHECK(cb::command_tool_name("spec.set_text") == "spec_set_text");
    CHECK(cb::command_name_from_tool("spec_set_text") == "spec.set_text");
    CHECK(cb::command_name_from_tool("panel_create") == "panel.create");
    CHECK(cb::command_tool_name("bad name").empty());
    CHECK(cb::command_name_from_tool("nounderscore").empty());
    CHECK(cb::command_name_from_tool("_x").empty());
}

TEST_CASE("an invocation round-trips through JSON") {
    CommandInvocation inv;
    inv.id = "c-41";
    inv.name = "panel.patch";
    inv.args = {{"id", "heat"}, {"patch", {{"title", "Entropy"}}}};
    inv.origin = CommandOrigin::agent;
    inv.group = "turn-7";
    const json j = cb::command_invocation_to_json(inv);
    CHECK(j["origin"] == "agent");
    CHECK(j["merge_key"].is_null());
    const auto back = cb::command_invocation_from_json(j);
    REQUIRE(back.error.ok());
    CHECK(back.invocation.id == "c-41");
    CHECK(back.invocation.args == inv.args);
    CHECK(back.invocation.origin == CommandOrigin::agent);
    CHECK(back.invocation.group == "turn-7");
    CHECK(back.invocation.merge_key.empty());

    CHECK(j["preview"].is_null());
    inv.preview = "widget-1";
    const auto with_preview = cb::command_invocation_from_json(cb::command_invocation_to_json(inv));
    REQUIRE(with_preview.error.ok());
    CHECK(with_preview.invocation.preview == "widget-1");

    // id and args may be absent.
    const auto minimal = cb::command_invocation_from_json({{"name", "app.info"}, {"origin", "api"}});
    REQUIRE(minimal.error.ok());
    CHECK(minimal.invocation.id.empty());
    CHECK(minimal.invocation.args == json::object());
}

TEST_CASE("a malformed invocation names its fault") {
    auto where = [](const json& j) {
        const auto parsed = cb::command_invocation_from_json(j);
        CHECK(parsed.error.code == "command.invalid_invocation");
        return parsed.error.where;
    };
    const json ok = {{"name", "panel.remove"}, {"origin", "ui"}, {"args", {{"id", "heat"}}}};
    CHECK(where(json::array()) == "");
    json j = ok;
    j["extra"] = 1;
    CHECK(where(j) == "extra");
    j = ok;
    j.erase("name");
    CHECK(where(j) == "name");
    j = ok;
    j.erase("origin");
    CHECK(where(j) == "origin");
    j = ok;
    j["origin"] = "root";
    CHECK(where(j) == "origin");
    j = ok;
    j["args"] = json::array();
    CHECK(where(j) == "args");
    j = ok;
    j["id"] = "has space";
    CHECK(where(j) == "id");
    j = ok;
    j["id"] = std::string(129, 'i');
    CHECK(where(j) == "id");
    j = ok;
    j["group"] = 3;
    CHECK(where(j) == "group");
    j = ok;
    j["merge_key"] = "slider ratio";
    CHECK(where(j) == "merge_key");
    j = ok;
    j["merge_key"] = "slider:heat/options.ratio";
    CHECK(cb::command_invocation_from_json(j).error.ok());
    j = ok;
    j["preview"] = "has space";
    CHECK(where(j) == "preview");
    j = ok;
    j["name"] = "Panel.remove";
    CHECK(where(j) == "name");
}

TEST_CASE("hostile arguments are refused before they are copied") {
    std::string deep = "{\"name\":\"panel.remove\",\"origin\":\"mcp\",\"args\":{\"x\":";
    deep += std::string(100000, '[') + std::string(100000, ']') + "}}";
    const auto parsed = cb::command_invocation_from_json(json::parse(deep));
    CHECK(parsed.error.code == "command.invalid_args");
}

TEST_CASE("a result is ok with a result, or not ok with an error") {
    const json ok = cb::command_result_to_json(CommandResult::success("c-1", {{"id", "heat"}}));
    CHECK(ok == json::parse(R"({"id": "c-1", "ok": true, "result": {"id": "heat"}})"));

    CommandResult ask = CommandResult::failure(
        "c-2", cb::Error("command.needs_approval", "loader.delete needs the person's approval."));
    ask.approval = "ap-1";
    const json j = cb::command_result_to_json(ask);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "command.needs_approval");
    CHECK(j["error"]["where"] == "");
    CHECK(j["error"]["approval"] == "ap-1");
    CHECK(!j.contains("result"));

    const auto back = cb::command_result_from_json(j);
    REQUIRE(back.error.ok());
    CHECK(back.result.approval == "ap-1");
    CHECK(back.result.error.code == "command.needs_approval");
    CHECK(cb::command_result_from_json(ok).result.result["id"] == "heat");

    CHECK(cb::command_result_from_json(json::parse(R"({"id": "c", "ok": true})")).error);
    CHECK(cb::command_result_from_json(json::parse(R"({"id": "c", "ok": false, "error": {"message": "x"}})")).error);
    CHECK(cb::command_result_from_json(json::parse(R"({"id": "c", "ok": true, "result": {}, "error": {}})")).error);

    // An error with no code still gets one, and invalid UTF-8 never makes a dump throw.
    CommandResult broken = CommandResult::failure("c-3", cb::Error("", std::string("bad \xFF byte")));
    CHECK(broken.error.code == "command.failed");
    CHECK_NOTHROW(cb::command_result_to_json(broken).dump());
}

TEST_CASE("a descriptor is checked: name, summary, schemas and examples") {
    CommandDescriptor d;
    d.name = "toy.set";
    d.summary = "Sets a value.";
    d.args = json::parse(R"({"type": "object", "properties": {"v": {"type": "number"}}, "required": ["v"]})");
    d.result = json::parse(R"({"type": "object"})");
    d.examples = {json{{"v", 1}}};
    CHECK(cb::check_command_descriptor(d).ok());

    const json j = cb::command_descriptor_to_json(d);
    CHECK(j["name"] == "toy.set");
    CHECK(j["examples"].size() == 1);
    CHECK(j["mutates"] == false);
    CHECK(j.size() == 10);

    CommandDescriptor bad = d;
    bad.name = "toy";
    CHECK(cb::check_command_descriptor(bad).code == "command.bad_descriptor");
    bad = d;
    bad.summary.clear();
    CHECK(cb::check_command_descriptor(bad).code == "command.bad_descriptor");
    bad = d;
    bad.args = {{"type", "array"}};
    CHECK(cb::check_command_descriptor(bad).code == "command.bad_descriptor");
    bad = d;
    bad.result = {{"oneOf", json::array()}};
    CHECK(cb::check_command_descriptor(bad).code == "command.bad_descriptor");
    bad = d;
    bad.examples = {json{{"v", "one"}}};
    const auto err = cb::check_command_descriptor(bad);
    CHECK(err.code == "command.bad_descriptor");
    CHECK(err.where == "toy.set.examples[0]");
}

TEST_CASE("origins have their names") {
    CHECK(std::string(cb::command_origin_name(CommandOrigin::mcp)) == "mcp");
    CHECK(cb::command_origin_from_name("builder") == CommandOrigin::builder);
    CHECK(!cb::command_origin_from_name("UI"));
}

TEST_CASE("SHA-256 matches the FIPS 180-4 test vectors") {
    CHECK(cb::command_sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(cb::command_sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(cb::command_sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(cb::command_sha256_hex(std::string(1000000, 'a')) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Lengths around the padding boundaries.
    const std::pair<std::size_t, const char*> boundary[] = {
        {55, "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072"},
        {56, "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e"},
        {63, "75220b47218278e656f2013bb8f0c455a25eaf01e86c64924e9d48d89776d6f2"},
        {64, "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c"},
        {65, "9537c5fdf120482f7d58d25e9ed583f52c02b4e304ea814db1633ad565aed7e9"},
        {119, "000b48d4edf0fa7bee3c6236ecd2785baa5db4eeb8bb54341b029e0d9fa5fb0c"},
        {120, "13f05a0b594787f5ecd315edc96141bd3243203d1b7d4f0836f37308b276ba98"},
    };
    for (const auto& [n, digest] : boundary) CHECK(cb::command_sha256_hex(std::string(n, 'x')) == digest);
}
