// Tests of the command journal: seq across reopening and rotation, flushing,
// redaction, cut and hostile lines, and replay. Every file lives under the
// test's build folder.
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/sha256.h>

#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using caliper_builder::CommandJournal;
using caliper_builder::CommandOrigin;
using caliper_builder::JournalEvent;
using caliper_builder::JournalOptions;
using nlohmann::json;
namespace cb = caliper_builder;
namespace fs = std::filesystem;

namespace {

// A fresh folder for one test, under the build folder.
std::string fresh_dir(const std::string& name) {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "journal" / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir.string();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void append_raw(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out << text;
}

JournalEvent do_event(const std::string& name, json args = json::object(), const std::string& group = {}) {
    JournalEvent e;
    e.origin = CommandOrigin::agent;
    e.name = name;
    e.args = std::move(args);
    e.group = group;
    return e;
}

JournalEvent history_event(const std::string& kind, std::vector<std::uint64_t> seqs, bool ok = true,
                           const std::string& error = {}) {
    JournalEvent e;
    e.name = kind == "undo" ? "history.undo" : "history.redo";
    e.event = kind;
    e.seqs = std::move(seqs);
    e.ok = ok;
    e.error = error;
    return e;
}

}  // namespace

TEST_CASE("events get increasing seqs that continue after reopening") {
    const std::string path = fresh_dir("reopen") + "/journal/commands.jsonl";  // the folder is created
    {
        auto opened = CommandJournal::open(path);
        REQUIRE(opened.error.ok());
        CHECK(opened.journal->last_seq() == 0);
        CHECK(opened.journal->append(do_event("panel.create", {{"id", "heat"}}, "turn-1")).seq == 1);
        auto failed = do_event("panel.remove", {{"id", "x"}});
        failed.ok = false;
        failed.error = "panel.not_found";
        CHECK(opened.journal->append(failed).seq == 2);
        CHECK(opened.journal->append(history_event("undo", {1})).seq == 3);
        // Each line is flushed when append returns, before the journal closes.
        CHECK(read_file(path).find("\"seq\":3") != std::string::npos);
    }
    auto reopened = CommandJournal::open(path);
    REQUIRE(reopened.error.ok());
    CHECK(reopened.journal->last_seq() == 3);
    CHECK(reopened.journal->append(do_event("panel.create", {{"id", "loss"}})).seq == 4);
    reopened.journal->close();
    CHECK(reopened.journal->append(do_event("panel.create")).error.code == "journal.closed");

    const auto read = cb::read_journal(path);
    REQUIRE(read.error.ok());
    CHECK(read.problems.empty());
    REQUIRE(read.events.size() == 4);
    const json first = cb::journal_event_to_json(read.events[0]);
    CHECK(first["group"] == "turn-1");
    CHECK(first["event"] == "do");
    CHECK(first["origin"] == "agent");
    CHECK(!first.contains("error"));
    CHECK(!first.contains("seqs"));
    CHECK(cb::journal_event_to_json(read.events[1])["error"] == "panel.not_found");
    CHECK(cb::journal_event_to_json(read.events[1])["ok"] == false);
    CHECK(read.events[2].event == "undo");
    CHECK(read.events[2].seqs == std::vector<std::uint64_t>{1});
    CHECK(cb::journal_event_to_json(read.events[3])["group"].is_null());
    // The timestamp is UTC with milliseconds.
    CHECK(read.events[0].ts.size() == 24);
    CHECK(read.events[0].ts.back() == 'Z');
}

TEST_CASE("an observer sees each event once its line is written") {
    const std::string path = fresh_dir("observer") + "/commands.jsonl";
    std::vector<JournalEvent> seen;
    JournalOptions options;
    options.on_append = [&](const JournalEvent& e) { seen.push_back(e); };
    auto opened = CommandJournal::open(path, options);
    REQUIRE(opened.error.ok());
    auto failed = do_event("loader.run", {{"id", "temps"}});
    failed.ok = false;
    failed.error = "loader.not_found";
    CHECK(opened.journal->append(do_event("panel.create", {{"token", "abc"}})).seq == 1);
    CHECK(opened.journal->append(failed).seq == 2);
    REQUIRE(seen.size() == 2);
    CHECK(seen[0].seq == 1);
    CHECK(seen[0].name == "panel.create");
    CHECK(seen[0].origin == CommandOrigin::agent);
    CHECK(seen[0].args["token"] == json{{"$redacted", "secret"}});  // what was written
    CHECK(seen[1].name == "loader.run");
    CHECK_FALSE(seen[1].ok);
    CHECK(seen[1].error == "loader.not_found");
    // An observer that throws loses its event, and the line is still written.
    std::vector<JournalEvent> before = seen;
    opened.journal->close();
    JournalOptions throwing;
    throwing.on_append = [](const JournalEvent&) { throw std::runtime_error("observer"); };
    auto again = CommandJournal::open(path, throwing);
    REQUIRE(again.error.ok());
    CHECK(again.journal->append(do_event("panel.remove")).seq == 3);
    // A closed journal writes nothing and tells nobody.
    again.journal->close();
    CHECK(again.journal->append(do_event("panel.remove")).error.code == "journal.closed");
    CHECK(seen.size() == before.size());
    CHECK(cb::read_journal(path).events.size() == 3);
}

TEST_CASE("timestamps are UTC ISO 8601 with milliseconds") {
    using namespace std::chrono;
    CHECK(cb::command_utc_timestamp(system_clock::time_point(milliseconds(1500))) == "1970-01-01T00:00:01.500Z");
    CHECK(cb::command_utc_timestamp(system_clock::time_point(seconds(1794744000))) == "2026-11-15T12:00:00.000Z");
}

TEST_CASE("large values are replaced by their SHA-256 and length") {
    const std::string big(70000, 'b');
    json args = {{"text", big}, {"id", "heat"}};
    json out = cb::redact_journal_args(args);
    CHECK(out["id"] == "heat");
    CHECK(out["text"]["$redacted"]["sha256"] == cb::command_sha256_hex(big));
    CHECK(out["text"]["$redacted"]["bytes"] == 70000);
    CHECK(cb::journal_args_redacted(out));
    CHECK_FALSE(cb::journal_args_redacted(args));

    // An array of small values whose text passes 64 KiB is replaced as a whole.
    json many = json::array();
    for (int i = 0; i < 20000; ++i) many.push_back(i);
    const std::string many_text = many.dump();
    out = cb::redact_journal_args({{"values", many}, {"n", 1}});
    CHECK(out["values"]["$redacted"]["sha256"] == cb::command_sha256_hex(many_text));
    CHECK(out["values"]["$redacted"]["bytes"] == many_text.size());
    CHECK(out["n"] == 1);

    // A big string deep inside is replaced where it is, and the rest stays.
    out = cb::redact_journal_args({{"patch", {{"options", {{"text", big}, {"bins", 32}}}}}});
    CHECK(out["patch"]["options"]["bins"] == 32);
    CHECK(out["patch"]["options"]["text"].contains("$redacted"));

    // Many members each under the limit: the args object itself is replaced.
    json wide = json::object();
    for (int i = 0; i < 5; ++i) wide["m" + std::to_string(i)] = std::string(20000, 'w');
    out = cb::redact_journal_args(wide);
    REQUIRE(out.contains("$redacted"));
    CHECK(out.size() == 1);
    // The marker of the whole object: its hash and its size as JSON text.
    const std::string wide_text = wide.dump();
    CHECK(out == json{{"$redacted", {{"sha256", cb::command_sha256_hex(wide_text)}, {"bytes", wide_text.size()}}}});

    // A value under the limit, in arguments under the limit, stays.
    out = cb::redact_journal_args({{"text", std::string(65000, 'e')}});
    CHECK(out["text"].is_string());
    out = cb::redact_journal_args({{"text", std::string(65537, 'e')}});
    CHECK(out["text"].contains("$redacted"));
}

TEST_CASE("values under keys named like secrets are never written") {
    const json args = {{"token", "abc"},       {"API_KEY", "k"},   {"github_token", "g"}, {"client_secret", "s"},
                       {"Authorization", "B"}, {"password", "p"},  {"max_tokens", 4096},  {"token_count", 3},
                       {"nested", {{"auth", {{"db_password", "pw"}}}}}};
    const json out = cb::redact_journal_args(args);
    for (const char* k : {"token", "API_KEY", "github_token", "client_secret", "Authorization", "password"})
        CHECK(out[k] == json{{"$redacted", "secret"}});
    CHECK(out["max_tokens"] == 4096);
    CHECK(out["token_count"] == 3);
    CHECK(out["nested"]["auth"]["db_password"] == json{{"$redacted", "secret"}});

    const std::string path = fresh_dir("secrets") + "/commands.jsonl";
    auto opened = CommandJournal::open(path);
    REQUIRE(opened.error.ok());
    REQUIRE(opened.journal->append(do_event("loader.create", {{"token", "never-in-the-file"}})).error.ok());
    CHECK(read_file(path).find("never-in-the-file") == std::string::npos);
}

TEST_CASE("arguments nested too deeply are written as malformed") {
    const json deep = json::parse(std::string(1000, '[') + std::string(1000, ']'));
    const json out = cb::redact_journal_args({{"x", deep}});
    CHECK(out == json{{"$redacted", "malformed"}});
}

TEST_CASE("the journal rotates by size, keeps its files and continues the seq") {
    const std::string dir = fresh_dir("rotate");
    const std::string path = dir + "/commands.jsonl";
    JournalOptions options;
    options.max_bytes = 2048;
    options.keep = 2;
    {
        auto opened = CommandJournal::open(path, options);
        REQUIRE(opened.error.ok());
        for (int i = 0; i < 60; ++i)
            REQUIRE(opened.journal->append(do_event("toy.set", {{"value", std::string(100, 'v')}})).error.ok());
        CHECK(opened.journal->last_seq() == 60);
    }
    CHECK(fs::exists(cb::journal_rotated_path(path, 1)));
    CHECK(fs::exists(cb::journal_rotated_path(path, 2)));
    CHECK_FALSE(fs::exists(cb::journal_rotated_path(path, 3)));
    CHECK(fs::file_size(path) <= 2048);
    CHECK(fs::file_size(cb::journal_rotated_path(path, 1)) <= 2048);

    // Reopening an empty current file continues from the rotated file.
    {
        std::ofstream truncate(path, std::ios::binary | std::ios::trunc);
    }
    auto reopened = CommandJournal::open(path, options);
    REQUIRE(reopened.error.ok());
    // The seq continues from the last event of the newest rotated file.
    std::uint64_t rotated_last = 0;
    {
        std::istringstream lines(read_file(cb::journal_rotated_path(path, 1)));
        for (std::string line; std::getline(lines, line);)
            if (!line.empty()) rotated_last = json::parse(line).at("seq").get<std::uint64_t>();
    }
    CHECK(rotated_last > 0);
    CHECK(reopened.journal->last_seq() == rotated_last);

    const auto read = cb::read_journal(path);
    REQUIRE(!read.events.empty());
    CHECK(read.events.front().seq > 1);  // the oldest file was rotated away
    for (std::size_t i = 1; i < read.events.size(); ++i) CHECK(read.events[i].seq > read.events[i - 1].seq);
    const auto replay = cb::replay_journal(read.events, [](const JournalEvent&) { return cb::Error(); });
    CHECK(replay.error.code == "journal.incomplete");

    CHECK(cb::journal_rotated_path("ws/journal/commands.jsonl", 2) ==
          (fs::path("ws/journal") / "commands.2.jsonl").string());
}

TEST_CASE("a line cut short is ended on reopening, and the reader reports it") {
    const std::string path = fresh_dir("cut") + "/commands.jsonl";
    {
        auto opened = CommandJournal::open(path);
        REQUIRE(opened.error.ok());
        REQUIRE(opened.journal->append(do_event("panel.create", {{"id", "a"}})).error.ok());
        REQUIRE(opened.journal->append(do_event("panel.create", {{"id", "b"}})).error.ok());
    }
    append_raw(path, "{\"seq\":3,\"ts\":\"2026-11-15T12:00:00.000Z\",\"origin\":\"ui\",\"na");
    auto reopened = CommandJournal::open(path);
    REQUIRE(reopened.error.ok());
    CHECK(reopened.journal->last_seq() == 2);
    CHECK(reopened.journal->append(do_event("panel.create", {{"id", "c"}})).seq == 3);
    reopened.journal->close();

    const auto read = cb::read_journal(path);
    REQUIRE(read.events.size() == 3);
    CHECK(read.events[2].args["id"] == "c");
    REQUIRE(read.problems.size() == 1);
    CHECK(read.problems[0] == "commands.jsonl:3: the line is not JSON.");
}

TEST_CASE("the reader skips hostile and malformed lines and reports each") {
    const std::string path = fresh_dir("hostile") + "/commands.jsonl";
    const std::string good1 =
        R"({"seq":1,"ts":"t","origin":"ui","name":"a.b","args":{},"ok":true,"group":null,"event":"do"})";
    const std::string good3 =
        R"({"seq":3,"ts":"t","origin":"ui","name":"a.b","args":{},"ok":true,"group":null,"event":"do"})";
    const std::string again2 =
        R"({"seq":2,"ts":"t","origin":"ui","name":"a.b","args":{},"ok":true,"group":null,"event":"do"})";
    std::string text;
    text += good1 + "\n";
    text += std::string(1500, 'x') + "\n";                   // longer than the limit below
    text += "not json\n";                                    // not JSON
    text += "[1,2]\n";                                       // not an event
    text += R"({"seq":2,"ts":"t","origin":"root","name":"a.b","args":{},"ok":true,"group":null,"event":"do"})" "\n";
    text += R"({"seq":2,"ts":"t","origin":"ui","name":"a.b","args":{},"ok":true,"group":null,"event":"do","seqs":[1]})" "\n";
    text += "{\"seq\":2,\"ts\":\"t\",\"origin\":\"ui\",\"name\":\"a.b\",\"args\":{\"x\":" + std::string(150, '[') +
            std::string(150, ']') + "},\"ok\":true,\"group\":null,\"event\":\"do\"}\n";  // too deep, short enough
    text += good3 + "\n";
    text += again2 + "\n";  // does not increase
    text += "\n";           // an empty line is ignored
    append_raw(path, text);

    cb::JournalReadOptions options;
    options.max_line = 1000;
    const auto read = cb::read_journal(path, options);
    REQUIRE(read.error.ok());
    REQUIRE(read.events.size() == 2);
    CHECK(read.events[0].seq == 1);
    CHECK(read.events[1].seq == 3);
    REQUIRE(read.problems.size() == 8);
    CHECK(read.problems[0] == "commands.jsonl:2: the line is longer than 1000 bytes.");
    CHECK(read.problems[1] == "commands.jsonl:3: the line is not JSON.");
    CHECK(read.problems[2] == "commands.jsonl:4: An event is a JSON object.");
    CHECK(read.problems[3] == "commands.jsonl:5: origin is ui, builder, agent, mcp or api. (origin)");
    CHECK(read.problems[4] == "commands.jsonl:6: Only an undo or redo event lists seqs. (seqs)");
    CHECK(read.problems[5] == "commands.jsonl:7: args is nested too deeply or too large. (args)");
    CHECK(read.problems[6] == "commands.jsonl:8: seq jumps from 1 to 3.");
    CHECK(read.problems[7] == "commands.jsonl:9: seq 2 does not increase; the event is skipped.");

    CHECK(cb::read_journal(fresh_dir("missing") + "/commands.jsonl").error.code == "journal.not_found");
}

TEST_CASE("a line over 1 MiB is refused by default without holding it") {
    const std::string path = fresh_dir("long") + "/commands.jsonl";
    append_raw(path, std::string((1 << 20) + 10, 'L') + "\n");
    const auto read = cb::read_journal(path);
    CHECK(read.events.empty());
    REQUIRE(read.problems.size() == 1);
    CHECK(read.problems[0] == "commands.jsonl:1: the line is longer than 1048576 bytes.");
}

TEST_CASE("replay applies the do events in effect after undo and redo, in order") {
    std::vector<JournalEvent> events;
    auto add = [&](JournalEvent e) {
        e.seq = events.size() + 1;
        events.push_back(std::move(e));
    };
    add(do_event("toy.add", {{"v", 1}}));   // 1
    add(do_event("toy.add", {{"v", 2}}));   // 2
    add(history_event("undo", {2}));        // 3
    add(do_event("toy.add", {{"v", 3}}));   // 4
    add(do_event("toy.set", {{"v", 4}}));   // 5  merged with 6
    add(do_event("toy.set", {{"v", 5}}));   // 6
    add(history_event("undo", {5, 6}));     // 7
    add(history_event("redo", {5, 6}));     // 8
    add(history_event("undo", {5, 6}));     // 9
    JournalEvent failed = do_event("toy.add", {{"v", 6}});
    failed.ok = false;
    failed.error = "toy.failed";
    add(failed);                                                       // 10
    add(history_event("undo", {4}, false, "history.undo_failed"));     // 11: changes nothing
    add(do_event("toy.add", {{"v", 7}}));                              // 12

    std::vector<int> applied;
    const auto replay = cb::replay_journal(events, [&](const JournalEvent& e) {
        applied.push_back(e.args["v"].get<int>());
        return cb::Error();
    });
    REQUIRE(replay.error.ok());
    CHECK(applied == std::vector<int>{1, 3, 7});
    CHECK(replay.applied == std::vector<std::uint64_t>{1, 4, 12});

    // A redacted event in effect stops the replay; one that was undone does not.
    std::vector<JournalEvent> redacted = {do_event("toy.add", {{"v", {{"$redacted", "secret"}}}})};
    redacted[0].seq = 1;
    CHECK(cb::replay_journal(redacted, [](const JournalEvent&) { return cb::Error(); }).error.code ==
          "journal.redacted");
    redacted.push_back(history_event("undo", {1}));
    redacted[1].seq = 2;
    CHECK(cb::replay_journal(redacted, [](const JournalEvent&) { return cb::Error(); }).error.ok());

    // An inconsistent history cannot be replayed.
    std::vector<JournalEvent> broken = {do_event("toy.add", {{"v", 1}}),
                                        history_event("undo", {1}, false, "history.inconsistent")};
    broken[0].seq = 1;
    broken[1].seq = 2;
    CHECK(cb::replay_journal(broken, [](const JournalEvent&) { return cb::Error(); }).error.code ==
          "journal.unreplayable");

    // The apply function's error stops the replay and names the seq; a throw is caught.
    const auto stopped = cb::replay_journal(events, [](const JournalEvent& e) {
        return e.seq == 4 ? cb::Error("toy.bad", "No.") : cb::Error();
    });
    CHECK(stopped.error.code == "toy.bad");
    CHECK(stopped.error.where == "seq 4");
    const auto threw = cb::replay_journal(events, [](const JournalEvent&) -> cb::Error { throw 7; });
    CHECK(threw.error.code == "journal.apply_failed");
    CHECK(cb::replay_journal({}, [](const JournalEvent&) { return cb::Error(); }).error.ok());
}

TEST_CASE("a journal path that is a folder is refused") {
    const std::string dir = fresh_dir("folder");
    auto opened = CommandJournal::open(dir);
    CHECK(opened.error.code == "journal.open_failed");
    CHECK(CommandJournal::open("").error.code == "journal.open_failed");
}
