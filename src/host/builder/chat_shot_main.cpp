// caliper-chat-shot: a development tool that draws a dashboard with the spec
// runtime on the left and the side chat (side_chat.h) in the right column,
// on the spec runtime's offscreen canvas, and drives the chat through a
// scripted conversation with a stub agent, writing a PNG at each state. It
// opens no window and is not installed.
//
//   caliper-chat-shot --workspace <dir> --agent <stub agent> --out-dir <dir> [options]
//
// The workspace gets the dashboard run42-dashboard when it has none: the
// files of --inputs (run42_metrics.parquet with step, loss, accuracy and lr,
// and attn.safetensors), or else a CSV of 200 steps and a small safetensors
// file that the tool writes, imported and run, and the dashboard of
// dashboard.spec.toml saved as a spec applet. The tool opens it (applet.open from the ui origin), so give it a
// folder of its own. Then it runs what the app runs for the chat: the
// session's command queue drained on this thread, Caliper API's server on
// 127.0.0.1 with the session's command gateway and an API file, and the side
// chat with the agent as a custom command named "Stub agent", started by the
// chat with this build's caliper-mcp --origin agent handed over in
// session/new. The agent is caliper_chat_stub_agent; no real agent runs.
//
// The states, each written as <prefix>-<n>-<state>-<W>x<H>.png:
//   1-empty      the chat before the first message, with the agent picker open
//   2-turn       "Add a panel of the learning rate": the turn running, its
//                streamed text and a tool call row of a Caliper command
//   3-permission the agent's permission request with the diff of its edit
//   4-done       the finished turn, the new panel in the dashboard
//   5-exited     an agent that exited in the middle of a turn, with Restart
//   6-no-agent   a chat that found no agent
//
// Options:
//   --size WxH        the canvas (1440x900)
//   --prefix NAME     the files' prefix (chat)
//   --frames N        frames drawn for each state (4)
//   --review on|off   whether Caliper reviews the agent's spec edits (off, so
//                     that the agent asks itself and the card shows its diff;
//                     with on, the edit waits for an approval, which the tool
//                     gives as the person would, and there is no 3-permission)
//   --mcp PATH        caliper-mcp (this build's by default)
//   --inputs DIR      the data files to import, as above
//
// It prints one JSON object: the files with their states, the chat's items
// as {kind, text, command, status}, the open spec's panels, and whether the
// journal recorded panel.create from the origin agent.
#include "builder/side_chat.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "api/api_server.h"
#include "api/session_gateway.h"
#include "export_service.h"
#include "net/hub.h"

#include <caliper_builder/session/session.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_host;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;

namespace {

const char* const kDashboard = R"(caliper_spec = 0

[applet]
id = "run42-dashboard"
title = "Run 42"
description = "Loss and accuracy of run 42, the attention of its first layer, and the metrics as a table."

[[source]]
id = "metrics"
catalog = "table:run42_metrics"

[[source]]
id = "attn"
catalog = "tensor:attn_weights"
  [[source.transform]]
  op = "slice"
  axis = 0
  index = 0
  [[source.transform]]
  op = "reduce"
  axis = 0
  fn = "mean"
  [[source.transform]]
  op = "normalise"
  mode = "minmax"

[[panel]]
id = "curves"
primitive = "series"
title = "Loss and accuracy"
bind = { x = "metrics.step", y = ["metrics.loss", "metrics.accuracy"] }
options = { x_label = "Step", y_log = true }

[[panel]]
id = "attention"
primitive = "heatmap"
title = "Mean attention of layer 0"
bind = { values = "attn" }
options = { colormap = "magma", range = [0.0, 1.0], x_label = "Key", y_label = "Query" }

[[panel]]
id = "rows"
primitive = "table"
bind = { rows = "metrics" }
options = { columns = ["step", "loss"] }

[layout]
split = "vertical"
ratio = 0.65
children = [{ split = "horizontal", children = ["curves", "attention"] }, "rows"]
)";

int usage(const char* why) {
    std::fprintf(stderr, "caliper-chat-shot: %s\n", why);
    std::fprintf(stderr,
                 "usage: caliper-chat-shot --workspace <dir> --agent <stub agent> --out-dir <dir>\n"
                 "                         [--size WxH] [--prefix NAME] [--frames N] [--review on|off] [--mcp PATH]\n"
                 "                         [--inputs DIR]\n");
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-chat-shot: %s\n", message.c_str());
    return 1;
}

CommandResult invoke(caliper_builder::Session& s, const std::string& name, json args) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = CommandOrigin::ui;
    return s.invoke(std::move(inv));
}

void write_safetensors(const std::string& path) {
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q)
                for (int k = 0; k <= q; ++k)   // causal, decaying with the distance, faster in later heads
                    w[((l * 3 + h) * 8 + q) * 8 + k] = std::exp(-(float)(q - k) * (0.35f + 0.3f * (float)h));
    json header;
    header["weights"] = {{"dtype", "F32"}, {"shape", {2, 3, 8, 8}}, {"data_offsets", {0, w.size() * 4}}};
    std::string h = header.dump();
    while (h.size() % 8) h += ' ';
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint64_t n = h.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
    out.write(h.data(), (std::streamsize)h.size());
    out.write(reinterpret_cast<const char*>(w.data()), (std::streamsize)(w.size() * 4));
}

// Builds the dashboard in an empty workspace, through the session's commands.
CommandResult make_files_and_import(caliper_builder::Session& s, const std::string& ws);

std::string make_dashboard(caliper_builder::Session& s, const std::string& ws, const std::string& inputs) {
    CommandResult r;
    if (!inputs.empty()) {
        const std::string a = (fs::path(inputs) / "run42_metrics.parquet").string();
        const std::string b = (fs::path(inputs) / "attn.safetensors").string();
        r = invoke(s, "data.import", {{"paths", {a, b}}, {"run", true}});
    } else {
        r = make_files_and_import(s, ws);
    }
    if (!r.ok) return "data.import: " + r.error.code + ": " + r.error.message;
    if (!r.result.value("refused", json::array()).empty()) return "data.import refused " + r.result["refused"].dump();
    r = invoke(s, "spec.new", {{"id", "run42-dashboard"}, {"title", "Run 42"}});
    if (!r.ok) return "spec.new: " + r.error.code + ": " + r.error.message;
    r = invoke(s, "spec.set_text", {{"text", kDashboard}});
    if (!r.ok) return "spec.set_text: " + r.error.code + ": " + r.error.message;
    r = invoke(s, "applet.save", {{"overwrite", true}});
    if (!r.ok) return "applet.save: " + r.error.code + ": " + r.error.message;
    return {};
}

// Small files of the same shape, for tests: 200 steps as CSV and a 2 x 3 x 8 x 8 tensor.
CommandResult make_files_and_import(caliper_builder::Session& s, const std::string& ws) {
    const fs::path data = fs::path(ws) / "data";
    fs::create_directories(data);
    {
        std::ofstream csv(data / "run42_metrics.csv", std::ios::trunc);
        csv << "step,loss,accuracy,lr\n";
        for (int i = 0; i < 200; ++i) {
            const double noise = std::fmod(std::sin(i * 12.9898) * 43758.5453, 1.0);   // a fixed ripple in [-1, 1]
            const double loss = 0.1 + 2.0 * std::exp(-i / 50.0) * (1.0 + 0.04 * noise);
            const double acc = 0.1 + 0.8 * (1.0 - std::exp(-i / 40.0));
            const double lr = 3e-4 * std::min(1.0, (i + 1) / 20.0) * 0.5 * (1.0 + std::cos(3.14159265358979 * i / 200.0));
            char line[128];
            std::snprintf(line, sizeof line, "%d,%.6f,%.6f,%.8f\n", i, loss, acc, lr);
            csv << line;
        }
    }
    write_safetensors((data / "attn.safetensors").string());
    return invoke(s, "data.import", {{"paths", {"data/run42_metrics.csv", "data/attn.safetensors"}}, {"run", true}});
}

}  // namespace

int main(int argc, char** argv) {
    std::string workspace, agent_exe, out_dir, inputs, prefix = "chat", review = "off", mcp_exe = CALIPER_MCP_EXE;
    int width = 1440, height = 900, frames = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!v) return usage(("incomplete option " + a).c_str());
        ++i;
        if (a == "--workspace") workspace = v;
        else if (a == "--agent") agent_exe = v;
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--prefix") prefix = v;
        else if (a == "--review") review = v;
        else if (a == "--mcp") mcp_exe = v;
        else if (a == "--inputs") inputs = fs::absolute(v).string();
        else if (a == "--frames") frames = std::atoi(v);
        else if (a == "--size") {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else {
            return usage(("unknown option " + a).c_str());
        }
    }
    if (workspace.empty() || agent_exe.empty() || out_dir.empty()) return usage("missing an option");
    if (review != "on" && review != "off") return usage("--review takes on or off");
    if (width < 64 || height < 64 || width > 8192 || height > 8192) return usage("--size must be from 64x64 to 8192x8192");
    if (frames < 1 || frames > 120) return usage("--frames must be from 1 to 120");
    std::error_code ec;
    if (!fs::is_directory(out_dir, ec)) return fail("the folder " + out_dir + " does not exist");
    fs::create_directories(workspace, ec);
    workspace = fs::canonical(workspace, ec).string();
    agent_exe = fs::absolute(agent_exe, ec).string();
    const std::string size = std::to_string(width) + "x" + std::to_string(height);

    // The session, with the review of the agent's spec edits as asked.
    caliper_builder::SessionOptions so;
    so.review.agent = review == "on";
    auto opened = caliper_builder::Session::open(workspace, so);
    if (opened.error) return fail("cannot open the workspace " + workspace + ": " + opened.error.message);
    std::unique_ptr<caliper_builder::Session> session = std::move(opened.session);
    session->registry().set_frame_thread(std::this_thread::get_id());
    if (!fs::exists(fs::path(workspace) / "applets" / "run42-dashboard", ec)) {
        if (std::string why = make_dashboard(*session, workspace, inputs); !why.empty()) return fail(why);
    }
    if (CommandResult r = invoke(*session, "applet.open", {{"id", "run42-dashboard"}}); !r.ok)
        return fail("applet.open: " + r.error.code + ": " + r.error.message);

    // Caliper API with the session's command layer, and its API file, as the
    // app writes it (owner only), beside the workspace.
    const fs::path data_dir = fs::path(workspace).parent_path() / (fs::path(workspace).filename().string() + "-chat-data");
    fs::create_directories(data_dir / "empty-path", ec);
    const fs::path gate_dir = data_dir / "gates";
    fs::remove_all(gate_dir, ec);
    fs::create_directories(gate_dir, ec);
    caliper_api::Options ao;
    ao.port = 0;
    ao.token = caliper_net::random_token();
    ao.info_json = "{\"host\":\"caliper-chat-shot\"}";
    auto server = std::make_unique<caliper_api::Server>(ao);
    server->set_command_gateway(caliper_api::make_session_gateway(*session));
    std::string err;
    if (!server->start(&err)) return fail("Caliper API did not start: " + err);
    const fs::path api_file = data_dir / "api.json";
    {
        json f{{"version", 1},       {"pid", (long long)::getpid()}, {"port", server->port()},
               {"token", ao.token},  {"url", server->base_url()},    {"stream_url", server->stream_url()}};
        std::ofstream(api_file, std::ios::trunc) << f.dump() << "\n";
        fs::permissions(api_file, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    }

    // The chat: the stub agent as the person's custom command; detection on
    // an empty PATH, so the four presets show as not found.
    SideChat::Options co;
    co.env.workspace = workspace;
    co.env.mcp_command = mcp_exe;
    co.env.api_file = api_file.string();
    co.env.path = (data_dir / "empty-path").string();
    co.settings.custom_name = "Stub agent";
    co.settings.custom_command = "'" + agent_exe + "' --gate-dir '" + gate_dir.string() + "'";
    co.settings.agent = "custom";
    co.client.stall_timeout_ms = 120000;
    SideChat::Hooks hooks;
    hooks.pending_approvals = [&session] { return session->registry().pending_approvals(); };
    auto chat = std::make_unique<SideChat>(co, hooks);
    SideChat::Options none = co;
    none.settings.custom_command.clear();
    none.settings.agent.clear();
    SideChat empty_chat(none);
    SideChat* shown = chat.get();

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, 1.0f, &why);
    if (!canvas) return fail("cannot draw offscreen: " + why);
    SessionPanelSource source(*session);
    std::unique_ptr<SpecRuntime> runtime;
    canvas->with_context([&] {
        runtime = std::make_unique<SpecRuntime>(canvas->textures());
        runtime->set_source(&source);
        runtime->set_spec(session->open_spec()->spec);
    });

    // The frame thread's work between two frames, as the app does it.
    auto pump = [&] {
        for (const caliper_api::Call& c : server->take())
            server->reply(c.ticket, 404, caliper_api::error_json("not in caliper-chat-shot"));
        session->queue().drain();
        chat->update();
    };
    auto wait_for = [&](const std::function<bool()>& cond, int ms) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            pump();
            if (cond()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    auto item = [&](ChatItem::Kind kind, const std::function<bool(const ChatItem&)>& pred) -> const ChatItem* {
        for (const ChatItem& it : chat->items())
            if (it.kind == kind && pred(it)) return &it;
        return nullptr;
    };

    json out{{"files", json::array()}, {"size", {width, height}}, {"review", review}};
    int failures = 0;
    auto shoot = [&](const std::string& name, const std::function<void()>& before = {}) {
        std::vector<uint8_t> rgba;
        bool first = true;
        const bool ok = canvas->render(frames, [&](const ImRect& area) {
            if (first && before) before();
            first = false;
            pump();
            if (auto open = session->open_spec()) runtime->set_spec(open->spec);
            ChatColumnInput ci;
            ci.window = {area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight()};
            const ChatColumnLayout col = layout_chat_column(ci, ChatLayoutMetrics::for_text_scale(canvas->look().text_scale));
            runtime->draw(ImRect(col.canvas.x, col.canvas.y, col.canvas.right(), col.canvas.bottom()), canvas->look());
            shown->draw(ImRect(col.chat.x, col.chat.y, col.chat.right(), col.chat.bottom()), canvas->look());
        }, &rgba, &err);
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "caliper-chat-shot: %s: %s\n", name.c_str(), err.c_str());
            return;
        }
        const std::string file = (fs::path(out_dir) / (prefix + "-" + name + "-" + size + ".png")).string();
        if (!export_write_png_atomic(file, rgba.data(), (uint32_t)width, (uint32_t)height, &err)) {
            ++failures;
            std::fprintf(stderr, "caliper-chat-shot: cannot write %s: %s\n", file.c_str(), err.c_str());
            return;
        }
        out["files"].push_back({{"name", name}, {"file", file}, {"state", chat_state_name(shown->state())}});
    };
    auto finish = [&](int code) {
        canvas->with_context([&] { runtime.reset(); });
        canvas.reset();
        chat.reset();
        server->set_command_gateway(nullptr);
        server->stop();
        session->close();
        fs::remove(api_file, ec);
        return code;
    };

    // 1: the chat before the first message, the picker open.
    shoot("1-empty", [&] { chat->open_picker(); });
    canvas->with_context([&] {
        if (GImGui && GImGui->OpenPopupStack.Size > 0) ImGui::ClosePopupToLevel(0, false);
    });

    // 2: the turn running: the request, the streamed text, a Caliper call.
    chat->set_text("Add a panel of the learning rate");
    chat->send();
    const bool running = wait_for([&] {
        const ChatItem* describe = item(ChatItem::Kind::tool, [](const ChatItem& it) {
            return it.command == "catalog.describe" && it.tool.status == "in_progress";
        });
        const ChatItem* said = item(ChatItem::Kind::agent, [](const ChatItem& it) {
            return it.text.find("beside the table.") != std::string::npos;
        });
        return describe && said;
    }, 60000);
    if (!running) return finish(fail(std::string("the turn did not reach its tool call: ") + chat_state_name(chat->state()) +
                                     " " + chat->error().code + " " + chat->error().message));
    shoot("2-turn");
    std::ofstream(gate_dir / "turn") << "go\n";

    // 3: the agent's permission request with its diff (review off), or the
    // approval Caliper holds (review on), which the tool gives as the person.
    if (review == "off") {
        if (!wait_for([&] { return item(ChatItem::Kind::permission, [](const ChatItem& it) { return it.answer.empty(); }); }, 60000))
            return finish(fail("no permission request came"));
        shoot("3-permission");
        const ChatItem* p = item(ChatItem::Kind::permission, [](const ChatItem& it) { return it.answer.empty(); });
        out["permission"] = {{"command", p->command}, {"diff_path", p->diff_path}, {"diff", p->diff},
                             {"options", json::array()}};
        for (const auto& o : p->options) out["permission"]["options"].push_back(o.kind);
        out["permission"]["compact"] = json::array();
        for (const auto& l : p->compact) out["permission"]["compact"].push_back({l.key, l.after});
        chat->answer_permission(p->permission, "allow");
    } else {
        if (!wait_for([&] { return item(ChatItem::Kind::approval, [](const ChatItem& it) { return it.answer.empty(); }); }, 60000))
            return finish(fail("no approval was mirrored"));
        const ChatItem* a = item(ChatItem::Kind::approval, [](const ChatItem& it) { return it.answer.empty(); });
        out["approval"] = {{"command", a->command}};
        shoot("3-approval");
        session->queue().approve(a->approval);
    }

    // 4: the finished turn and the new panel.
    if (!wait_for([&] { return chat->state() == ChatState::ready; }, 60000))
        return finish(fail(std::string("the turn did not finish: ") + chat_state_name(chat->state())));
    shoot("4-done");
    out["panels"] = json::array();
    if (const auto open = session->open_spec())
        for (const auto& p : open->spec.panels) out["panels"].push_back(p.id);

    // 5: an agent that exits in the middle of a turn.
    chat->set_text("Plot attention entropy per layer");
    chat->send();
    if (!wait_for([&] { return chat->state() == ChatState::exited; }, 60000))
        return finish(fail(std::string("the agent did not exit: ") + chat_state_name(chat->state())));
    shoot("5-exited");
    out["items"] = json::array();
    for (const ChatItem& it : chat->items()) {
        static const char* kinds[] = {"person", "agent", "thought", "tool", "plan", "permission", "approval", "stop", "notice"};
        json j = {{"kind", kinds[(int)it.kind]}, {"text", it.text.substr(0, 200)}};
        if (!it.command.empty()) j["command"] = it.command;
        if (it.kind == ChatItem::Kind::tool) j["status"] = it.tool.status;
        if (!it.answer.empty()) j["answer"] = it.answer;
        if (!it.detail.empty()) j["detail"] = it.detail;
        out["items"].push_back(j);
    }
    out["header_actions"] = json::array();
    for (ChatAction a : chat->header_actions()) out["header_actions"].push_back(chat_action_label(a));

    // 6: a chat that found no agent.
    shown = &empty_chat;
    shoot("6-no-agent");
    out["no_agent_state"] = chat_state_name(empty_chat.state());

    // The journal: panel.create from the origin agent.
    {
        std::ifstream j(fs::path(workspace) / "journal" / "commands.jsonl");
        std::string line;
        bool found = false;
        while (std::getline(j, line)) {
            const json e = json::parse(line, nullptr, false);
            if (!e.is_object()) continue;
            const json inv = e.contains("invocation") ? e["invocation"] : e;
            if (inv.value("name", "") == "panel.create" && inv.value("origin", "") == "agent") found = true;
        }
        out["journal_agent_panel_create"] = found;
    }
    shown = chat.get();
    std::printf("%s\n", out.dump().c_str());
    return finish(failures ? 1 : 0);
}
