// caliper_chat_stub_agent: an ACP agent of version 1 for the side chat's
// tests and captures, credited as "Stub agent". It needs no model and no
// network. It answers initialize and session/new, starts the first stdio MCP
// server that session/new hands it (the side chat's caliper-mcp) and speaks
// MCP to it, and on session/prompt plays the script that the person's text
// names (the last text block that is not marked for the assistant):
//
//   ... learning rate ...   the scenario of the captures: a plan, a thought,
//                  spec.get and catalog.describe through caliper-mcp as tool
//                  calls named as Claude Code names them, streamed text, then
//                  review.status: when Caliper does not review the agent's
//                  spec edits, a dry run of panel.create and a permission
//                  request with its diff, and on Allow the panel.create;
//                  when Caliper reviews them, the panel.create at once, which
//                  Caliper holds until the person decides. The panel shows
//                  metrics.lr over metrics.step beside the table.
//   ... while it trains ...   the acceptance case: catalog.list of the live
//                  sources, then spec.new, a source for each live source it
//                  needs and one whose transforms take the entropy of each
//                  layer's head-mean attention, three panels, the layout,
//                  applet.save and applet.open, each through caliper-mcp, so
//                  that each spec edit waits for the person's approval when
//                  Caliper reviews them
//   Compare ...    one chunk, a line on stderr, then exits with code 1
//   report         answers with the prompt's blocks as JSON
//   anything else  echoes the text
//
// Options: --gate-dir <dir> makes the learning-rate scenario wait, after its
// streamed text, until the file <dir>/turn exists, so that a capture can be
// taken while the turn runs; --embedded-context and --image declare those
// prompt capabilities.
#include <caliper_builder/acp/subprocess.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
namespace acp = caliper_builder::acp;
namespace fs = std::filesystem;

namespace {

std::mutex g_out_mutex;
std::mutex g_state_mutex;
json g_session_params;
const std::string g_session_id = "chat-stub-session-1";
std::atomic<bool> g_cancelled{false};
std::atomic<int> g_next_id{5000};
std::map<int, std::promise<json>> g_waiting;
std::string g_gate_dir;
bool g_embedded = false;
bool g_image = false;

void send(const json& msg) {
    const std::string line = msg.dump();
    std::lock_guard<std::mutex> lock(g_out_mutex);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void respond(const json& id, const json& result) { send({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}); }

void update(const json& u) {
    send({{"jsonrpc", "2.0"}, {"method", "session/update"}, {"params", {{"sessionId", g_session_id}, {"update", u}}}});
}

void chunk(const std::string& text, const char* kind = "agent_message_chunk") {
    update({{"sessionUpdate", kind}, {"content", {{"type", "text"}, {"text", text}}}});
}

void end_turn(const json& id, const std::string& reason = "end_turn") { respond(id, {{"stopReason", reason}}); }

void pause(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

json ask_client(const std::string& method, const json& params) {
    const int id = g_next_id++;
    std::future<json> f;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        f = g_waiting[id].get_future();
    }
    send({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}});
    if (f.wait_for(std::chrono::seconds(300)) != std::future_status::ready) return nullptr;
    return f.get();
}

void wait_gate(const std::string& name) {
    if (g_gate_dir.empty()) return;
    const fs::path gate = fs::path(g_gate_dir) / name;
    for (int i = 0; i < 6000 && !g_cancelled; ++i) {
        std::error_code ec;
        if (fs::exists(gate, ec)) return;
        pause(20);
    }
}

// ---- MCP to the server session/new handed over --------------------------------------------

class Mcp {
public:
    bool start(std::string* why) {
        json server;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            const json servers = g_session_params.value("mcpServers", json::array());
            if (servers.empty()) {
                *why = "session/new handed over no MCP server";
                return false;
            }
            server = servers[0];
        }
        acp::SpawnOptions so;
        so.argv.push_back(server.value("command", ""));
        for (const auto& a : server.value("args", json::array())) so.argv.push_back(a.get<std::string>());
        for (const auto& e : server.value("env", json::array())) so.env.push_back({e.value("name", ""), e.value("value", "")});
        caliper_builder::Error e;
        proc_ = acp::Process::spawn(so, &e);
        if (!proc_) {
            *why = e.message;
            return false;
        }
        const json init = request("initialize", {{"protocolVersion", "2025-11-25"},
                                                 {"capabilities", json::object()},
                                                 {"clientInfo", {{"name", "caliper-chat-stub"}, {"version", "1"}}}});
        if (!init.contains("result")) {
            *why = "caliper-mcp refused initialize: " + init.dump();
            return false;
        }
        write({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}});
        return true;
    }

    // The command's result (structuredContent), or {"error": ...}.
    json call(const std::string& tool, const json& args) {
        const json r = request("tools/call", {{"name", tool}, {"arguments", args}});
        if (!r.contains("result")) return {{"error", {{"code", "mcp.no_answer"}, {"message", r.dump()}}}};
        const json& res = r["result"];
        if (res.value("isError", false) || !res.contains("structuredContent")) {
            json err = {{"code", "mcp.tool_error"}};
            if (res.contains("content") && res["content"].is_array() && !res["content"].empty()) {
                const json body = json::parse(res["content"][0].value("text", std::string("{}")), nullptr, false);
                if (body.is_object() && body.contains("error")) err = body["error"];
            }
            return {{"error", err}};
        }
        return res["structuredContent"];
    }

    ~Mcp() {
        if (proc_) proc_->terminate(1000);
    }

private:
    void write(const json& j) {
        const std::string line = j.dump() + "\n";
        proc_->write_stdin_all(line.data(), line.size(), 10000);
    }
    json request(const std::string& method, const json& params) {
        const int id = ++id_;
        write({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}});
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(200);
        while (std::chrono::steady_clock::now() < end) {
            const auto nl = buf_.find('\n');
            if (nl != std::string::npos) {
                const json m = json::parse(buf_.substr(0, nl), nullptr, false);
                buf_.erase(0, nl + 1);
                if (m.is_object() && m.contains("id") && m["id"] == id && !m.contains("method")) return m;
                continue;
            }
            char b[65536];
            std::size_t got = 0;
            const acp::IoStatus st = proc_->read_stdout(b, sizeof b, &got, 100);
            if (st == acp::IoStatus::ok) buf_.append(b, got);
            else if (st == acp::IoStatus::eof || st == acp::IoStatus::error) break;
        }
        return json::object();
    }

    std::unique_ptr<acp::Process> proc_;
    std::string buf_;
    int id_ = 0;
};

// Applies a unified diff (unified_line_diff()'s) to the text it was made from.
std::string apply_unified(const std::string& old_text, const std::string& diff) {
    std::vector<std::string> old_lines;
    {
        std::stringstream ss(old_text);
        std::string l;
        while (std::getline(ss, l)) old_lines.push_back(l);
    }
    std::vector<std::string> out;
    size_t cursor = 0;   // the next old line, 0-based
    std::stringstream ds(diff);
    std::string l;
    bool in_hunk = false;
    while (std::getline(ds, l)) {
        if (l.rfind("@@ -", 0) == 0) {
            const int start = std::atoi(l.c_str() + 4);
            const size_t comma = l.find(',', 4), space = l.find(' ', 4);
            const int count = comma != std::string::npos && comma < space ? std::atoi(l.c_str() + comma + 1) : 1;
            const size_t first = count == 0 ? (size_t)start : (size_t)std::max(0, start - 1);
            while (cursor < first && cursor < old_lines.size()) out.push_back(old_lines[cursor++]);
            in_hunk = true;
            continue;
        }
        if (!in_hunk || l.empty()) continue;
        if (l[0] == ' ') {
            if (cursor < old_lines.size()) out.push_back(old_lines[cursor++]);
        } else if (l[0] == '-') {
            ++cursor;
        } else if (l[0] == '+') {
            out.push_back(l.substr(1));
        }
    }
    while (cursor < old_lines.size()) out.push_back(old_lines[cursor++]);
    std::string text;
    for (const auto& x : out) text += x + "\n";
    return text;
}

json tool_call(const std::string& id, const std::string& tool, const char* kind, const char* status, const json& input) {
    return {{"sessionUpdate", "tool_call"},
            {"toolCallId", id},
            {"title", "mcp__caliper__" + tool},
            {"kind", kind},
            {"status", status},
            {"rawInput", input}};
}

void tool_status(const std::string& id, const char* status) {
    update({{"sessionUpdate", "tool_call_update"}, {"toolCallId", id}, {"status", status}});
}

void plan(const char* a, const char* b, const char* c) {
    update({{"sessionUpdate", "plan"},
            {"entries", json::array({{{"content", "Read the open spec"}, {"priority", "high"}, {"status", a}},
                                     {{"content", "Find the learning rate in the catalog"}, {"priority", "high"}, {"status", b}},
                                     {{"content", "Add a series panel of it beside the table"}, {"priority", "medium"}, {"status", c}}})}});
}

void learning_rate(const json& id) {
    Mcp mcp;
    std::string why;
    if (!mcp.start(&why)) {
        chunk("No Caliper tools: " + why);
        end_turn(id);
        return;
    }
    plan("in_progress", "pending", "pending");
    chunk("The person wants the learning rate as a panel of its own. The open spec and the catalog say which source has it.",
          "agent_thought_chunk");
    pause(60);
    update(tool_call("t1", "spec_get", "read", "in_progress", json::object()));
    const json spec = mcp.call("spec_get", json::object());
    tool_status("t1", spec.contains("error") ? "failed" : "completed");
    plan("completed", "in_progress", "pending");
    update(tool_call("t2", "catalog_describe", "read", "in_progress", {{"id", "table:run42_metrics"}}));
    for (const char* piece : {"The metrics table run42_metrics has an lr column ", "beside step, loss and accuracy, ",
                              "so a series of lr over step fits beside the table."}) {
        pause(40);
        chunk(piece);
    }
    wait_gate("turn");
    if (g_cancelled) {
        tool_status("t2", "failed");
        end_turn(id, "cancelled");
        return;
    }
    const json described = mcp.call("catalog_describe", {{"id", "table:run42_metrics"}});
    tool_status("t2", described.contains("error") ? "failed" : "completed");
    plan("completed", "completed", "in_progress");

    const json args = {{"id", "lr"},
                       {"primitive", "series"},
                       {"title", "Learning rate"},
                       {"bind", {{"x", "metrics.step"}, {"y", json::array({"metrics.lr"})}}},
                       {"options", {{"x_label", "Step"}}},
                       {"place", {{"target", "rows"}, {"where", "beside"}}}};
    const json review = mcp.call("review_status", json::object());
    const bool reviewed = review.value("spec_edits", true);
    update(tool_call("t3", "panel_create", "edit", "pending", args));
    if (!reviewed) {
        // Caliper does not review this agent's spec edits: the agent asks itself, with the diff.
        json dry = args;
        dry["dry_run"] = true;
        const json checked = mcp.call("panel_create", dry);
        const std::string before = spec.value("text", std::string());
        const std::string after = apply_unified(before, checked.value("diff", std::string()));
        const std::string cwd = [] {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            return g_session_params.value("cwd", std::string("/"));
        }();
        const json path = spec.value("path", json());
        const std::string file = cwd + "/" + (path.is_string() ? path.get<std::string>() : std::string("run42-dashboard.spec.toml"));
        json call = {{"toolCallId", "t3"},
                     {"title", "mcp__caliper__panel_create"},
                     {"kind", "edit"},
                     {"status", "pending"},
                     {"content", json::array({{{"type", "diff"}, {"path", file}, {"oldText", before}, {"newText", after}}})},
                     {"rawInput", args}};
        const json options = json::array({{{"optionId", "allow"}, {"name", "Allow"}, {"kind", "allow_once"}},
                                          {{"optionId", "always"}, {"name", "Always allow"}, {"kind", "allow_always"}},
                                          {{"optionId", "reject"}, {"name", "Reject"}, {"kind", "reject_once"}}});
        const json answer = ask_client("session/request_permission",
                                       {{"sessionId", g_session_id}, {"toolCall", call}, {"options", options}});
        const json outcome = answer.is_object() && answer.contains("result") ? answer["result"].value("outcome", json()) : json();
        const std::string kind = outcome.is_object() ? outcome.value("outcome", std::string()) : std::string();
        const std::string option = outcome.is_object() ? outcome.value("optionId", std::string()) : std::string();
        if (kind != "selected") {
            tool_status("t3", "failed");
            end_turn(id, "cancelled");
            return;
        }
        if (option == "reject") {
            tool_status("t3", "failed");
            chunk("Left the dashboard as it was.");
            end_turn(id);
            return;
        }
    }
    tool_status("t3", "in_progress");
    const json made = mcp.call("panel_create", args);
    if (made.contains("error")) {
        tool_status("t3", "failed");
        chunk("Caliper did not add the panel: " + made["error"].value("code", std::string("?")) + ".");
        end_turn(id);
        return;
    }
    tool_status("t3", "completed");
    plan("completed", "completed", "completed");
    for (const char* piece : {"Added the panel Learning rate ", "beside the table, ", "with lr over step."}) {
        pause(30);
        chunk(piece);
    }
    end_turn(id);
}

// One Caliper command as a tool call row: in progress, then completed or failed.
json caliper_call(Mcp& mcp, const std::string& row, const std::string& tool, const char* kind, const json& args) {
    update(tool_call(row, tool, kind, "in_progress", args));
    const json r = mcp.call(tool, args);
    tool_status(row, r.contains("error") ? "failed" : "completed");
    return r;
}

void live_entropy(const json& id) {
    Mcp mcp;
    std::string why;
    if (!mcp.start(&why)) {
        chunk("No Caliper tools: " + why);
        end_turn(id);
        return;
    }
    auto steps = [](const char* a, const char* b, const char* c) {
        update({{"sessionUpdate", "plan"},
                {"entries", json::array({{{"content", "Find the stream's sources in the catalog"}, {"priority", "high"}, {"status", a}},
                                         {{"content", "Build the spec over them"}, {"priority", "high"}, {"status", b}},
                                         {{"content", "Open it"}, {"priority", "medium"}, {"status", c}}})}});
    };
    steps("in_progress", "pending", "pending");
    chunk("A live stream is attached; its sources are in the catalog as live entries.", "agent_thought_chunk");
    const json listed = caliper_call(mcp, "l1", "catalog_list", "read", {{"kind", "live"}});
    std::vector<std::string> layers;
    bool attn = false, entropy = false;
    for (const json& e : listed.value("entries", json::array())) {
        const std::string eid = e.value("id", std::string());
        if (eid == "live:attn") attn = true;
        if (eid == "live:attn.entropy") entropy = true;
        if (eid.rfind("live:entropy.l", 0) == 0) layers.push_back(eid);
    }
    std::sort(layers.begin(), layers.end());
    if (!attn || !entropy) {
        chunk("The catalog has no live attention yet; is the training script streaming?");
        end_turn(id);
        return;
    }
    chunk("The stream sends its attention as live:attn, the entropy of each head as live:attn.entropy, computed by the "
          "training script, and " + std::to_string(layers.size()) + " series of each layer's mean entropy. ");
    steps("completed", "in_progress", "pending");
    int n = 0;
    auto row = [&n] { return "e" + std::to_string(++n); };
    auto ok = [&](const json& r) { return !r.contains("error"); };
    bool good = ok(caliper_call(mcp, row(), "spec_new", "edit", {{"id", "attention-entropy"}, {"title", "Attention entropy"}}));
    good = good && ok(caliper_call(mcp, row(), "source_add", "edit", {{"id", "entropy"}, {"catalog", "live:attn.entropy"}}));
    good = good && ok(caliper_call(mcp, row(), "source_add", "edit", {{"id", "attn"}, {"catalog", "live:attn"}}));
    good = good && ok(caliper_call(mcp, row(), "source_add", "edit",
                                   {{"id", "layer_entropy"},
                                    {"from", "attn"},
                                    {"transform", json::array({{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}},
                                                               {{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}})}}));
    json history = json::array();
    for (size_t l = 0; good && l < layers.size(); ++l) {
        const std::string sid = "layer" + std::to_string(l);
        good = ok(caliper_call(mcp, row(), "source_add", "edit", {{"id", sid}, {"catalog", layers[l]}}));
        history.push_back(sid);
    }
    good = good && ok(caliper_call(mcp, row(), "panel_create", "edit",
                                   {{"id", "heads"}, {"primitive", "heatmap"}, {"title", "Attention entropy by layer and head"},
                                    {"bind", {{"values", "entropy"}}}, {"options", {{"x_label", "Head"}, {"y_label", "Layer"}}}}));
    good = good && ok(caliper_call(mcp, row(), "panel_create", "edit",
                                   {{"id", "layers"}, {"primitive", "series"}, {"title", "Entropy of the head-mean attention by layer"},
                                    {"bind", {{"y", "layer_entropy"}}}, {"options", {{"x_label", "Layer"}}}}));
    if (good && !history.empty())
        good = ok(caliper_call(mcp, row(), "panel_create", "edit",
                               {{"id", "steps"}, {"primitive", "series"}, {"title", "Mean entropy per layer by step"},
                                {"bind", {{"y", history}}}, {"options", {{"x_label", "Step"}}}}));
    good = good && ok(caliper_call(mcp, row(), "layout_set", "edit",
                                   {{"layout", {{"root", "main"},
                                                {"nodes", json::array({{{"id", "main"}, {"split", "vertical"}, {"ratio", 0.6},
                                                                        {"children", json::array({"top", "steps"})}},
                                                                       {{"id", "top"}, {"split", "horizontal"}, {"ratio", 0.5},
                                                                        {"children", json::array({"heads", "layers"})}}})}}}}));
    good = good && ok(caliper_call(mcp, row(), "applet_save", "edit", json::object()));
    if (!good) {
        chunk("Caliper refused a step, so the spec is not complete.");
        end_turn(id);
        return;
    }
    steps("completed", "completed", "in_progress");
    caliper_call(mcp, row(), "applet_open", "other", {{"id", "attention-entropy"}});
    steps("completed", "completed", "completed");
    chunk("Opened Attention entropy: the entropy by layer and head as the training script computes it, the entropy of "
          "each layer's head-mean attention computed by the spec, and each layer's mean entropy by step. They follow "
          "the stream while it trains.");
    end_turn(id);
}

void run_turn(json id, json prompt) {
    std::string text;
    for (const auto& block : prompt) {
        if (!block.is_object() || block.value("type", "") != "text") continue;
        const json aud = block.contains("annotations") ? block["annotations"].value("audience", json()) : json();
        if (aud.is_array() && aud.size() == 1 && aud[0] == "assistant") continue;
        text = block.value("text", "");
    }
    if (text.find("learning rate") != std::string::npos) {
        learning_rate(id);
    } else if (text.find("while it trains") != std::string::npos) {
        live_entropy(id);
    } else if (text.rfind("Compare", 0) == 0) {
        chunk("Reading the attention tensor of every layer.");
        std::fprintf(stderr, "stub agent: the Compare script exits here on purpose\n");
        std::fflush(stderr);
        pause(50);
        std::fflush(stdout);
        std::_Exit(1);
    } else if (text == "report") {
        chunk(prompt.dump());
        end_turn(id);
    } else {
        chunk("You said: " + text);
        end_turn(id);
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gate-dir" && i + 1 < argc) g_gate_dir = argv[++i];
        else if (a == "--embedded-context") g_embedded = true;
        else if (a == "--image") g_image = true;
    }
    std::thread turn;
    std::string line;
    while (std::getline(std::cin, line)) {
        json msg = json::parse(line, nullptr, false);
        if (!msg.is_object()) continue;
        if (msg.contains("method")) {
            const std::string method = msg.value("method", "");
            const json params = msg.value("params", json::object());
            const json id = msg.contains("id") ? msg["id"] : json();
            if (method == "initialize") {
                respond(id, {{"protocolVersion", 1},
                             {"agentCapabilities",
                              {{"loadSession", false},
                               {"promptCapabilities", {{"image", g_image}, {"audio", false}, {"embeddedContext", g_embedded}}},
                               {"mcpCapabilities", {{"http", false}, {"sse", false}}}}},
                             {"authMethods", json::array()},
                             {"agentInfo", {{"name", "caliper-chat-stub"}, {"title", "Stub agent"}, {"version", "1"}}}});
            } else if (method == "session/new") {
                {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_session_params = params;
                }
                respond(id, {{"sessionId", g_session_id},
                             {"modes", {{"currentModeId", "default"},
                                        {"availableModes", json::array({{{"id", "default"}, {"name", "Default"}}})}}}});
            } else if (method == "session/prompt") {
                if (turn.joinable()) turn.join();
                g_cancelled = false;
                turn = std::thread(run_turn, id, params.value("prompt", json::array()));
            } else if (method == "session/cancel") {
                g_cancelled = true;
            } else if (!id.is_null()) {
                send({{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", -32601}, {"message", "Method not found: " + method}}}});
            }
        } else if (msg.contains("id") && msg["id"].is_number_integer()) {
            const int id = msg["id"].get<int>();
            std::lock_guard<std::mutex> lock(g_state_mutex);
            auto it = g_waiting.find(id);
            if (it != g_waiting.end()) {
                it->second.set_value(msg);
                g_waiting.erase(it);
            }
        }
    }
    if (turn.joinable()) turn.detach();
    std::fflush(stdout);
    std::_Exit(0);
}
