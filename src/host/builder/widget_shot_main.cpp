// caliper-widget-shot: a development tool that drives the widget agent's
// inline surface (widget_surface.h) on one panel of a spec through a
// scripted scenario, with the scripted agent and no model, and writes a PNG
// of the spec runtime's offscreen canvas at each named state, as
// caliper-spec-shot does for a spec alone. It opens no window and is not
// installed.
//
//   caliper-widget-shot --workspace <dir> --applet <id> --panel <id>
//                       --prompt <text> --out-dir <dir> [options]
//
// The applet is opened as the workspace's open spec (applet.open from the ui
// origin), so the workspace is changed: give it a copy. Options:
//   --size WxH        the canvas (1440x900)
//   --prefix NAME     the files' prefix (widget)
//   --scenario NAME   accept (default): idle, prompt, running, preview,
//                     accepted, and a fresh render of the changed spec;
//                     declined: prompt, declined; handoff: prompt, handoff;
//                     unavailable: the prompt with no provider; refine:
//                     preview, a second prompt, its preview
//   --edit JSON       the panel.patch `patch` the agent sends (accept and
//                     refine; {"options": {"y_log": false}} by default)
//   --reply TEXT      the agent's one-sentence reply
//   --frames N        frames drawn for each state (4)
//
// It prints one JSON object: the files written with their states, the target
// panel's rectangle, the open spec's undo steps before and after Accept, and
// whether the open spec changed before Accept.
#include "builder/widget_agent.h"
#include "builder/widget_surface.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"

#include <caliper_builder/session/session.h>

#include <imgui.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_host;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandResult;
using std::chrono::milliseconds;

namespace {

int usage(const char* why) {
    std::fprintf(stderr, "caliper-widget-shot: %s\n", why);
    std::fprintf(stderr,
                 "usage: caliper-widget-shot --workspace <dir> --applet <id> --panel <id> --prompt <text>\n"
                 "                           --out-dir <dir> [--size WxH] [--prefix NAME] [--frames N]\n"
                 "                           [--scenario accept|declined|handoff|unavailable|refine]\n"
                 "                           [--edit JSON] [--reply TEXT]\n");
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-widget-shot: %s\n", message.c_str());
    return 1;
}

CommandResult invoke(caliper_builder::Session& s, const std::string& name, json args, CommandOrigin origin) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    return s.invoke(std::move(inv));
}

WidgetScriptStep step(int ms, WidgetScriptStep::Kind kind, std::string text) {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = kind;
    s.text = std::move(text);
    return s;
}

WidgetScriptStep edit_step(int ms, const std::string& panel, const json& patch) {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = WidgetScriptStep::Kind::edit;
    s.edit.name = "panel.patch";
    s.edit.args = {{"id", panel}, {"patch", patch}};
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string workspace, applet, panel, prompt, out_dir, prefix = "widget", scenario = "accept";
    std::string reply = "Turns off the log scale of the y axis.";
    json patch = {{"options", {{"y_log", false}}}};
    int width = 1440, height = 900, frames = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!v) return usage(("incomplete option " + a).c_str());
        ++i;
        if (a == "--workspace") workspace = v;
        else if (a == "--applet") applet = v;
        else if (a == "--panel") panel = v;
        else if (a == "--prompt") prompt = v;
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--prefix") prefix = v;
        else if (a == "--scenario") scenario = v;
        else if (a == "--reply") reply = v;
        else if (a == "--frames") frames = std::atoi(v);
        else if (a == "--edit") {
            patch = json::parse(v, nullptr, false);
            if (patch.is_discarded() || !patch.is_object()) return usage("--edit takes a JSON object");
        } else if (a == "--size") {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else {
            return usage(("unknown option " + a).c_str());
        }
    }
    if (workspace.empty() || applet.empty() || panel.empty() || out_dir.empty()) return usage("missing an option");
    if (prompt.empty()) prompt = "Use a linear y axis";
    if (width < 64 || height < 64 || width > 8192 || height > 8192) return usage("--size must be from 64x64 to 8192x8192");
    if (frames < 1 || frames > 120) return usage("--frames must be from 1 to 120");
    std::error_code ec;
    if (!fs::is_directory(out_dir, ec)) return fail("the folder " + out_dir + " does not exist");

    auto opened = caliper_builder::Session::open(workspace);
    if (opened.error) return fail("cannot open the workspace " + workspace + ": " + opened.error.message);
    std::unique_ptr<caliper_builder::Session> session = std::move(opened.session);
    {
        const CommandResult r = invoke(*session, "applet.open", {{"id", applet}}, CommandOrigin::ui);
        if (!r.ok) return fail("applet.open " + applet + ": " + r.error.code + ": " + r.error.message);
    }
    if (!session->open_spec() || !session->open_spec()->spec.find_panel(panel))
        return fail("the applet " + applet + " has no panel " + panel);

    // The scripted agent, on a clock the scenario moves by hand.
    auto clock_now = std::chrono::steady_clock::now();
    auto now = [&clock_now] { return clock_now; };
    auto advance = [&clock_now](int ms) { clock_now += milliseconds(ms); };
    WidgetScript script;   // shown as "Scripted": no model runs
    script.start_after = milliseconds(250);
    WidgetScriptTurn turn;
    if (scenario == "declined") {
        turn.steps = {step(400, WidgetScriptStep::Kind::thought, "")};
        turn.end_at = milliseconds(900);
        turn.stop_reason = "refusal";
    } else if (scenario == "handoff") {
        turn.steps = {step(300, WidgetScriptStep::Kind::progress, "spec.get")};
        turn.end_at = milliseconds(1100);
        turn.handoff = WidgetHandoff{"loader", "A loader for the validation logs is work for the side chat."};
    } else {
        turn.steps = {step(300, WidgetScriptStep::Kind::progress, "spec.get"), edit_step(900, panel, patch),
                      step(1200, WidgetScriptStep::Kind::message, reply)};
        turn.end_at = milliseconds(1400);
    }
    script.turns.push_back(turn);
    if (scenario == "refine") {
        WidgetScriptTurn second;
        second.steps = {edit_step(600, panel, {{"title", "Training curves"}}),
                        step(800, WidgetScriptStep::Kind::message, "Renames the panel to Training curves.")};
        second.end_at = milliseconds(1000);
        script.turns.push_back(second);
    }
    script.available = scenario != "unavailable";
    ScriptedWidgetAgent agent(
        script, [&](CommandInvocation inv) { return session->invoke(std::move(inv)); }, now,
        [&](const std::string& id) { return (int)session->preview(id).edits; });
    SessionWidgetSession host(*session);
    WidgetSurface::Hooks hooks;
    hooks.now = now;
    int preview_n = 0;
    hooks.make_preview_id = [&preview_n] { return "widget-shot-" + std::to_string(++preview_n); };
    std::vector<json> handed;
    hooks.ask_in_chat = [&handed](const WidgetChatRequest& r) { handed.push_back({{"request", r.request}}); };
    WidgetSurface surface(host, agent, hooks);

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, 1.0f, &why);
    if (!canvas) return fail("cannot draw offscreen: " + why);
    SessionPanelSource source(*session);
    std::unique_ptr<SpecRuntime> runtime;
    canvas->with_context([&] {
        runtime = std::make_unique<SpecRuntime>(canvas->textures());
        runtime->set_source(&source);
        runtime->set_spec(session->open_spec()->spec);
        surface.attach(*runtime);
    });

    json out{{"files", json::array()}, {"size", {width, height}}, {"scenario", scenario}};
    const std::string before_text = session->open_spec()->text;
    int failures = 0;
    auto shoot = [&](const std::string& name) {
        std::vector<uint8_t> rgba;
        std::string err;
        const bool ok = canvas->render(frames, [&](const ImRect& area) {
            if (auto open = session->open_spec()) runtime->set_spec(open->spec);
            surface.frame(canvas->look());
            runtime->draw(area, canvas->look());
        }, &rgba, &err);
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "caliper-widget-shot: %s: %s\n", name.c_str(), err.c_str());
            return;
        }
        const std::string file = (fs::path(out_dir) / (prefix + "-" + name + ".png")).string();
        if (!export_write_png_atomic(file, rgba.data(), (uint32_t)width, (uint32_t)height, &err)) {
            ++failures;
            std::fprintf(stderr, "caliper-widget-shot: cannot write %s: %s\n", file.c_str(), err.c_str());
            return;
        }
        out["files"].push_back({{"name", name}, {"file", file}, {"state", widget_state_name(surface.state())}});
    };
    auto run_for = [&](int ms) {
        for (int t = 0; t < ms; t += 50) {
            advance(50);
            surface.update();
        }
    };

    shoot("idle");
    for (const auto& lp : runtime->layout().panels)
        if (lp.id == panel) out["panel_rect"] = {lp.rect.x, lp.rect.y, lp.rect.w, lp.rect.h};
    const size_t steps_before = session->registry().history_state().undo.size();
    if (!surface.open(WidgetTarget::for_panel(panel))) return fail("the surface did not open on " + panel);
    // The prompt is typed: the input takes the keyboard on its first frames,
    // then the text arrives as the keyboard's characters.
    {
        std::vector<uint8_t> scratch;
        std::string err;
        canvas->render(2, [&](const ImRect& area) {
            surface.frame(canvas->look());
            runtime->draw(area, canvas->look());
        }, nullptr, &err);
        canvas->with_context([&] { ImGui::GetIO().AddInputCharactersUTF8(prompt.c_str()); });
    }
    shoot("prompt");
    if (surface.text() != prompt) surface.set_text(prompt);   // a canvas without keyboard focus
    if (scenario == "unavailable") {
        // nothing more: the prompt offers only Ask in chat
    } else {
        surface.submit();
        if (surface.state() != WidgetState::running) return fail(std::string("submit left the surface ") +
                                                                 widget_state_name(surface.state()) + ": " +
                                                                 surface.error().code + " " + surface.error().message);
        run_for(1000);
        if (scenario == "accept" || scenario == "refine") shoot("running");
        run_for(800);
        out["open_spec_unchanged_before_accept"] = session->open_spec()->text == before_text;
        if (scenario == "declined") {
            shoot("declined");
        } else if (scenario == "handoff") {
            shoot("handoff");
        } else {
            if (surface.state() != WidgetState::preview)
                return fail(std::string("the turn left the surface ") + widget_state_name(surface.state()) + ": " +
                            surface.error().code + " " + surface.error().message);
            shoot("preview");
            out["diff"] = json::array();
            for (const auto& d : surface.diff()) out["diff"].push_back({d.key, d.before, d.after});
            if (scenario == "refine") {
                surface.refine();
                surface.set_text("Call it Training curves");
                shoot("refine-prompt");
                surface.submit();
                run_for(1200);
                shoot("refined");
                surface.toggle_diff();
                shoot("refined-diff");
                surface.toggle_diff();
            }
            surface.accept();
            if (surface.state() != WidgetState::applied)
                return fail(std::string("Accept left the surface ") + widget_state_name(surface.state()) + ": " +
                            surface.error().code + " " + surface.error().message);
            out["undo_steps_before_accept"] = steps_before;
            out["undo_steps_after_accept"] = session->registry().history_state().undo.size();
            shoot("accepted");
            // A fresh render of the changed spec, for comparison.
            std::unique_ptr<SpecRuntime> fresh;
            canvas->with_context([&] {
                surface.detach();
                fresh = std::make_unique<SpecRuntime>(canvas->textures());
                fresh->set_source(&source);
                fresh->set_spec(session->open_spec()->spec);
                std::swap(runtime, fresh);
            });
            shoot("fresh");
            canvas->with_context([&] {
                fresh.reset();
            });
        }
    }
    out["final_state"] = widget_state_name(surface.state());
    out["handed_to_chat"] = handed;
    canvas->with_context([&] {
        surface.close();
        surface.detach();
        runtime.reset();
    });
    canvas.reset();
    std::printf("%s\n", out.dump().c_str());
    session->close();
    return failures ? 1 : 0;
}
