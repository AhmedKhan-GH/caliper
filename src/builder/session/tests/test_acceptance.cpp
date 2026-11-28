// The acceptance tests of the session phase, headless. A Parquet file written
// through DuckDB and a safetensors file are imported, their loaders run, and
// the three-panel dashboard of the spec fixture dashboard.spec.toml is built
// command by command, undone and redone in part, resolved, saved, reopened
// and replayed: first in process from the ui origin, then over HTTP through
// caliper-builder-host from the mcp origin, with destructive commands that wait
// for the person's approval.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <thread>

#include "helpers.h"
#include "parquet_writer.h"

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// Writes the two data files: run42_metrics.parquet (step, loss, accuracy for
// steps 0 to 9) and attn.safetensors (one tensor `weights` [2, 3, 4, 4] whose
// element i is i).
void write_data(const std::string& dir) {
    write_parquet(dir + "/run42_metrics.parquet",
                  "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, CAST(range AS DOUBLE) / 10 AS accuracy "
                  "FROM range(10)");
    std::vector<float> w(96);
    for (int i = 0; i < 96; ++i) w[i] = static_cast<float>(i);
    write_safetensors_f32(dir + "/attn.safetensors", "weights", {2, 3, 4, 4}, w);
}

// The commands that build a spec from nothing, item by item.
std::vector<std::pair<std::string, json>> build_commands(const spec::Spec& s) {
    std::vector<std::pair<std::string, json>> out;
    json instrument{{"id", s.instrument.id}};
    if (s.instrument.title) instrument["title"] = *s.instrument.title;
    if (s.instrument.description) instrument["description"] = *s.instrument.description;
    out.emplace_back("spec.new", instrument);
    for (const auto& src : s.sources) out.emplace_back("source.add", source_command_args(src));
    for (const auto& p : s.panels) out.emplace_back("panel.create", panel_command_args(p));
    if (s.layout) out.emplace_back("layout.set", json{{"layout", layout_to_flat(*s.layout, s)}});
    return out;
}

// Checks the resolved data of the dashboard's three panels against the values
// the files hold, and returns their summaries.
json check_dashboard_data(const std::function<PanelData(const std::string&)>& resolve) {
    json summaries = json::object();
    const PanelData curves = resolve("curves");
    REQUIRE_MESSAGE(curves.error.ok(), curves.error.code << ": " << curves.error.message);
    REQUIRE(curves.slots.size() == 3);
    CHECK(curves.slots[0].slot == "x");
    CHECK(curves.slots[0].values == json{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    for (int i = 0; i < 10; ++i) {
        CHECK(curves.slots[1].values[i].get<double>() == doctest::Approx(1.0 / (i + 1)));
        CHECK(curves.slots[2].values[i].get<double>() == doctest::Approx(i / 10.0));
    }
    summaries["curves"] = panel_data_summary(curves, 64);

    // Layer 0, the mean over its three heads, then minmax: (4r + c) / 15.
    const PanelData attention = resolve("attention");
    REQUIRE_MESSAGE(attention.error.ok(), attention.error.code << ": " << attention.error.message);
    REQUIRE(attention.slots.size() == 1);
    const TensorData& t = attention.slots[0].tensor;
    REQUIRE(t.shape() == std::vector<int64_t>{4, 4});
    CHECK(t.dtype() == DType::f32);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) CHECK(t.element(r * 4 + c) == doctest::Approx((4 * r + c) / 15.0));
    summaries["attention"] = panel_data_summary(attention, 64);

    const PanelData rows = resolve("rows");
    REQUIRE_MESSAGE(rows.error.ok(), rows.error.code << ": " << rows.error.message);
    REQUIRE(rows.slots.size() == 1);
    REQUIRE(rows.slots[0].columns.size() == 2);
    CHECK(rows.slots[0].columns[0].name == "step");
    CHECK(rows.slots[0].columns[1].name == "loss");
    CHECK(rows.slots[0].values.size() == 10);
    CHECK(rows.slots[0].values[3][0] == 3);
    summaries["rows"] = panel_data_summary(rows, 64);
    return summaries;
}

}  // namespace

TEST_CASE("acceptance in process: import, build the dashboard, undo, resolve, save, reopen and replay") {
    const std::string root = fresh_dir("acceptance");
    const std::string ws = root + "/workspace";
    const std::string outside = root + "/outside";
    write_data(outside);
    const spec::Spec dashboard = parse_fixture("dashboard.spec.toml");

    json summaries;
    {
        auto s = open_session(ws);
        const json imported = ok(*s, "data.import",
                                 {{"paths", {outside + "/run42_metrics.parquet", outside + "/attn.safetensors"}}, {"run", false}});
        REQUIRE(imported["refused"].empty());
        REQUIRE(imported["loaders"].size() == 2);
        CHECK(imported["loaders"][0]["reader"] == "parquet");
        CHECK(imported["loaders"][1]["reader"] == "safetensors");
        for (const auto& l : imported["loaders"]) ok(*s, "loader.run", {{"id", l["id"]}});
        const json listed = ok(*s, "catalog.list");
        REQUIRE(listed["entries"].size() == 2);
        CHECK(listed["entries"][0]["id"] == "table:run42_metrics");
        CHECK(listed["entries"][1]["id"] == "tensor:attn_weights");
        CHECK(listed["entries"][1]["shape"] == json{2, 3, 4, 4});

        for (const auto& command : build_commands(dashboard)) {
            const std::string& name = command.first;
            const json& args = command.second;
            CAPTURE(name);
            const json r = ok(*s, name, args);
            if (r.contains("diff")) CHECK(!r["diff"].get<std::string>().empty());
        }
        CHECK(s->open_spec()->spec == dashboard);
        CHECK(s->open_spec()->text == spec::to_toml(dashboard));

        // Undo the layout and the last panel, redo one, then the other.
        const json undone = ok(*s, "history.undo", {{"steps", 2}});
        CHECK(undone["steps"][0]["label"] == "layout.set");
        CHECK(undone["steps"][1]["label"] == "panel.create");
        CHECK(s->open_spec()->spec.panels.size() == 2);
        CHECK(ok(*s, "history.list")["redo"].size() == 2);
        ok(*s, "history.redo");
        CHECK(s->open_spec()->spec.panels.size() == 3);
        CHECK(s->open_spec()->spec.layout != dashboard.layout);
        ok(*s, "history.redo");
        CHECK(s->open_spec()->spec == dashboard);
        CHECK(err(*s, "history.redo") == "history.nothing_to_redo");

        summaries = check_dashboard_data([&](const std::string& id) { return s->resolve(id); });

        const json saved = ok(*s, "instrument.save");
        CHECK(saved["spec"] == "instruments/run42-dashboard/run42-dashboard.spec.toml");
        const std::string manifest = read_text(ws + "/" + saved["manifest"].get<std::string>());
        CHECK(manifest.find("kind = \"spec\"") != std::string::npos);
        CHECK(manifest.find("spec = \"run42-dashboard.spec.toml\"") != std::string::npos);
        CHECK(manifest.find("id = \"local.run42-dashboard\"") != std::string::npos);
        CHECK(s->open_spec()->dirty == false);
        s->close();
    }

    // A new session over the same workspace finds the instrument and its data.
    {
        auto s = open_session(ws);
        CHECK(!s->open_spec());
        CHECK(ok(*s, "instrument.list")["instruments"][0]["id"] == "run42-dashboard");
        const json opened = ok(*s, "instrument.open", {{"id", "run42-dashboard"}});
        CHECK(opened["title"] == "Run 42");
        CHECK(s->open_spec()->spec == dashboard);
        CHECK(s->open_spec()->dirty == false);
        const json again = check_dashboard_data([&](const std::string& id) { return s->resolve(id); });
        CHECK(again == summaries);
    }

    // The journal replays to the same spec.
    const SpecReplay replay = replay_spec_journal(ws);
    REQUIRE_MESSAGE(replay.error.ok(), replay.error.code << ": " << replay.error.message);
    REQUIRE(replay.spec);
    CHECK(*replay.spec == dashboard);
    CHECK(replay.problems.empty());
}

#ifndef _WIN32

namespace {

// Reads one line of the host's standard output.
std::string read_line(acp::Process& p, std::string& pending, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const auto eol = pending.find('\n');
        if (eol != std::string::npos) {
            std::string line = pending.substr(0, eol);
            pending.erase(0, eol + 1);
            return line;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) return {};
        char buf[4096];
        std::size_t got = 0;
        const acp::IoStatus st = p.read_stdout(buf, sizeof buf, &got, static_cast<int>(left.count()));
        if (st == acp::IoStatus::ok) pending.append(buf, got);
        else if (st != acp::IoStatus::timeout) return {};
    }
}

struct Host {
    std::unique_ptr<acp::Process> process;
    std::string api_file;
    std::string pending;
    int port = 0;
    std::string token;

    Host(const std::string& ws, const std::string& api_dir) {
        acp::SpawnOptions so;
        so.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", ws, "--api-dir", api_dir, "--approve", "ask"};
        Error e;
        process = acp::Process::spawn(so, &e);
        REQUIRE_MESSAGE(process, e.message);
        api_file = api_dir + "/api.json";
        for (int i = 0; i < 400 && !fs::exists(api_file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        REQUIRE(fs::exists(api_file));
        const json api = json::parse(read_text(api_file));
        port = api["port"].get<int>();
        token = api["token"].get<std::string>();
        CHECK(api["url"] == "http://127.0.0.1:" + std::to_string(port) + "/v0");
        CHECK(api["pid"] == process->pid());
        struct stat st;
        REQUIRE(::stat(api_file.c_str(), &st) == 0);
        CHECK((st.st_mode & 0777) == 0600);
    }

    json call(const std::string& name, const json& args) {
        const HttpReply r = http(port, "POST", "/v0/commands/" + name, token, args.dump(), {"X-Caliper-Origin: mcp"}, 150000);
        INFO(name << " answered HTTP " << r.status << ": " << r.body);
        REQUIRE(r.status == 200);
        return r.json_body();
    }

    json ok_call(const std::string& name, const json& args) {
        const json env = call(name, args);
        INFO(name << ": " << env.dump());
        REQUIRE(env["ok"] == true);
        return env["result"];
    }

    // Sends a command that waits for approval, answers the approval with
    // `decision` once the host announces it, and returns the envelope.
    json call_with_approval(const std::string& name, const json& args, const std::string& decision) {
        json env;
        std::thread caller([&] { env = call(name, args); });
        const std::string line = read_line(*process, pending, 20000);
        const json announced = json::parse(line, nullptr, false);
        bool asked = announced.is_object() && announced.value("name", "") == name && announced.value("origin", "") == "mcp";
        if (asked) {
            const std::string text = decision + " " + announced["approval"].get<std::string>() + "\n";
            CHECK(process->write_stdin_all(text.data(), text.size(), 5000) == acp::IoStatus::ok);
        }
        caller.join();
        CHECK_MESSAGE(asked, "the host announced " << line);
        return env;
    }
};

}  // namespace

TEST_CASE("acceptance over HTTP: the same dashboard through caliper-builder-host from the mcp origin") {
    const std::string root = fresh_dir("acceptance-http");
    const std::string ws = root + "/workspace";
    write_data(ws + "/data");
    write_text(root + "/outside/other.csv", "a\n1\n");
    const spec::Spec dashboard = parse_fixture("dashboard.spec.toml");

    {
        Host host(ws, root + "/api");
        // The descriptor table, and the checks of a request.
        // The token is mcp's, so the person's undo and redo are not listed.
        const json listed = http(host.port, "GET", "/v0/commands", host.token).json_body();
        CHECK(listed["commands"].size() == command_table().size() - 2);
        CHECK(http(host.port, "GET", "/v0/commands", "").status == 401);

        // An agent imports files inside the workspace only.
        const json imported = host.ok_call(
            "data.import",
            {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors", root + "/outside/other.csv"}}, {"run", false}});
        REQUIRE(imported["loaders"].size() == 2);
        REQUIRE(imported["refused"].size() == 1);
        CHECK(imported["refused"][0]["code"] == "loader.path_refused");

        // Running a loader is destructive: the call waits for the person.
        for (const auto& l : imported["loaders"]) {
            const json ran = host.call_with_approval("loader.run", {{"id", l["id"]}}, "allow");
            CHECK(ran["ok"] == true);
            CHECK(ran["result"]["outputs"].size() == 1);
        }
        const json denied = host.call_with_approval("loader.delete", {{"id", "attn"}}, "deny");
        CHECK(denied["error"]["code"] == "command.denied");
        CHECK(host.ok_call("catalog.describe", {{"id", "tensor:attn_weights"}})["entry"]["shape"] == json{2, 3, 4, 4});

        // The dashboard, command by command, with a dry run first.
        const auto commands = build_commands(dashboard);
        for (const auto& command : commands) {
            const std::string& name = command.first;
            const json& args = command.second;
            CAPTURE(name);
            if (name == "panel.create") {
                json dry = args;
                dry["dry_run"] = true;
                const json preview = host.ok_call(name, dry);
                CHECK(preview["applied"] == false);
                CHECK(preview["diff"].get<std::string>().find("+[[panel]]") != std::string::npos);
            }
            host.ok_call(name, args);
        }
        // Undo and redo are the person's: an agent's call is refused, and
        // the spec stays as the agent built it.
        CHECK(host.call("history.undo", {{"steps", 2}})["error"]["code"] == "command.not_allowed");
        CHECK(host.call("history.redo", {{"steps", 2}})["error"]["code"] == "command.not_allowed");
        const json got = host.ok_call("spec.get", json::object());
        CHECK(got["text"] == spec::to_toml(dashboard));
        CHECK(got["dirty"] == true);

        // A new instrument is saved without approval; overwriting it waits.
        CHECK(host.ok_call("instrument.save", json::object())["id"] == "run42-dashboard");
        CHECK(host.call("instrument.save", json::object())["error"]["code"] == "instrument.exists");
        // The agent takes a change back with an edit that reverts it.
        host.ok_call("panel.patch", {{"id", "rows"}, {"patch", {{"title", "Metrics"}}}});
        host.ok_call("panel.patch", {{"id", "rows"}, {"patch", {{"title", nullptr}}}});
        const json overwritten = host.call_with_approval("instrument.save", {{"overwrite", true}}, "allow");
        CHECK(overwritten["ok"] == true);

        // SIGTERM stops the host cleanly and removes its API file.
        const acp::ExitStatus status = host.process->terminate(10000);
        CHECK(status.exited);
        CHECK(status.code == 0);
        CHECK(!fs::exists(host.api_file));
    }
    // The journal names the origin of every command the host ran.
    const JournalRead read = read_journal(ws + "/journal/commands.jsonl");
    REQUIRE(read.events.size() > 10);
    bool all_mcp = true;
    for (const auto& e : read.events) all_mcp = all_mcp && e.origin == CommandOrigin::mcp;
    CHECK(all_mcp);

    // The workspace the host left behind holds the same instrument and data.
    {
        auto s = open_session(ws);
        ok(*s, "instrument.open", {{"id", "run42-dashboard"}});
        CHECK(s->open_spec()->spec == dashboard);
        check_dashboard_data([&](const std::string& id) { return s->resolve(id); });
    }
    const SpecReplay replay = replay_spec_journal(ws);
    REQUIRE_MESSAGE(replay.error.ok(), replay.error.code << ": " << replay.error.message);
    REQUIRE(replay.spec);
    CHECK(*replay.spec == dashboard);
}

#endif
