// caliper-builder-shot: a development tool that builds the three-panel
// dashboard of dashboard.spec.toml by hand in the builder pane
// (builder_pane.h), on an offscreen canvas, through the pane's own input
// handling: every click, drag, key and typed text goes through ImGui's event
// queue, as the platform backend feeds a person's input, and the pane turns
// it into its gestures. It writes a PNG at each named step, opens no window
// and is not installed.
//
//   caliper-builder-shot --workspace <dir> --instrument-from <spec.toml>
//                        --files <file>... --out-dir <dir> [options]
//
// The workspace is changed: give it a new folder. The spec is begun with
// spec.new from the builder origin under the instrument id, title and
// description of --instrument-from's [instrument], the files are dropped on the
// pane (data.import from the builder origin, their loaders run), and then
// the scenario builds the dashboard:
//   1 empty      build mode on the empty spec, the palettes, the two sources
//   2 series     the curves panel made from the metrics table as a series,
//                renamed, titled and labelled, Properties showing its bindings
//   3 heatmap    the attention source with its three transforms, the heatmap
//                dragged beside the curves, its options; Properties showing
//                the source's transforms and their shapes
//   4 divider    the table dropped under everything, its columns chosen, and
//                the root divider being dragged to 0.65
//   5 text       the spec text strip open beside the finished dashboard
//   6 undo       after two undos (⌘Z twice)
// then redoes both and saves the instrument (⌘S).
// Options:
//   --size WxH        the canvas (1440x900)
//   --prefix NAME     the files' prefix (builder): <prefix>-<step><suffix>.png
//   --suffix TEXT     the files' suffix, such as -1440x900 (none)
//   --text-scale S    the text scale (1)
//
// It prints one JSON object: the files written, the open spec's canonical
// text at the end, the commands the pane sent with their origins and
// groups, and the undo steps before and after the two undos.
#include "builder/builder_pane.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include "export_service.h"

#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <nlohmann/json.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_host;
namespace cb = caliper_builder;

namespace {

int usage(const char* why) {
    std::fprintf(stderr, "caliper-builder-shot: %s\n", why);
    std::fprintf(stderr,
                 "usage: caliper-builder-shot --workspace <dir> --instrument-from <spec.toml> --files <file>...\n"
                 "                            --out-dir <dir> [--size WxH] [--prefix NAME] [--suffix TEXT]\n"
                 "                            [--text-scale S]\n");
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-builder-shot: %s\n", message.c_str());
    return 1;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The person at the keyboard: input through ImGui's event queue, one frame at
// a time, aimed at the pane's named rectangles of the last frame.
class Driver {
public:
    Driver(SpecOffscreen& canvas, BuilderPane& pane, std::function<void(const ImRect&)> frame)
        : canvas_(canvas), pane_(pane), frame_(std::move(frame)) {}

    bool ok() const { return error_.empty(); }
    const std::string& error() const { return error_; }

    // One frame; with `rgba`, read back.
    void render(std::vector<uint8_t>* rgba = nullptr) {
        std::string why;
        if (!canvas_.render(1, frame_, rgba, &why)) fail_with("render: " + why);
    }
    void frames(int n) {
        for (int i = 0; i < n; ++i) render();
    }

    bool target(const std::string& name, float fx, float fy, ImVec2* at) {
        SpecRect r;
        for (int tries = 0; tries < 6 && !pane_.item_rect(name, &r); ++tries) render();
        if (!pane_.item_rect(name, &r)) {
            std::string names;
            for (const auto& n : pane_.item_names()) names += " " + n;
            fail_with("no item " + name + " on the canvas; drawn:" + names.substr(0, std::min<size_t>(names.size(), 1500)));
            return false;
        }
        *at = ImVec2(std::round(r.x + r.w * fx), std::round(r.y + r.h * fy));
        return true;
    }

    void move(ImVec2 p) {
        canvas_.with_context([&] { ImGui::GetIO().AddMousePosEvent(p.x, p.y); });
        render();
    }

    void click(const std::string& name, float fx = 0.5f, float fy = 0.5f) {
        if (!ok()) return;
        ImVec2 p;
        if (!target(name, fx, fy, &p)) return;
        canvas_.with_context([&] {
            ImGui::GetIO().AddMousePosEvent(p.x, p.y);
            ImGui::GetIO().AddMouseButtonEvent(0, true);
        });
        render();
        canvas_.with_context([&] { ImGui::GetIO().AddMouseButtonEvent(0, false); });
        render();
        render();
    }

    void type(const std::string& text) {
        if (!ok()) return;
        canvas_.with_context([&] { ImGui::GetIO().AddInputCharactersUTF8(text.c_str()); });
        render();
        render();
    }

    // `mods` as the pane reads them (ImGuiMod_Ctrl is ⌘ on macOS); the
    // events are those a platform backend sends, so ⌘ is Super there, which
    // ImGui swaps back.
    void key(ImGuiKey k, ImGuiKeyChord mods = 0) {
        if (!ok()) return;
        canvas_.with_context([&] {
            ImGuiIO& io = ImGui::GetIO();
            const ImGuiKey cmd = io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl;
            if (mods & ImGuiMod_Ctrl) io.AddKeyEvent(cmd, true);
            if (mods & ImGuiMod_Shift) io.AddKeyEvent(ImGuiMod_Shift, true);
            io.AddKeyEvent(k, true);
        });
        render();
        canvas_.with_context([&] {
            ImGuiIO& io = ImGui::GetIO();
            const ImGuiKey cmd = io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl;
            io.AddKeyEvent(k, false);
            if (mods & ImGuiMod_Shift) io.AddKeyEvent(ImGuiMod_Shift, false);
            if (mods & ImGuiMod_Ctrl) io.AddKeyEvent(cmd, false);
        });
        render();
        render();
    }

    // Presses on `from`, moves to `to` in steps and releases there, unless
    // `hold`, which leaves the button down.
    void drag(const std::string& from, float ffx, float ffy, const std::string& to, float tfx, float tfy, bool hold = false) {
        if (!ok()) return;
        ImVec2 a;
        if (!target(from, ffx, ffy, &a)) return;
        ImVec2 b;
        if (!target(to, tfx, tfy, &b)) return;
        drag_points(a, b, hold);
    }

    void drag_points(ImVec2 a, ImVec2 b, bool hold) {
        canvas_.with_context([&] {
            ImGui::GetIO().AddMousePosEvent(a.x, a.y);
            ImGui::GetIO().AddMouseButtonEvent(0, true);
        });
        render();
        const int steps = 14;
        for (int i = 1; i <= steps; ++i) {
            const float t = (float)i / (float)steps;
            move(ImVec2(std::round(a.x + (b.x - a.x) * t), std::round(a.y + (b.y - a.y) * t)));
        }
        render();
        if (hold) return;
        release();
    }

    void release() {
        canvas_.with_context([&] { ImGui::GetIO().AddMouseButtonEvent(0, false); });
        render();
        render();
    }

    // Leaves the pointer off every item, so no hover shows in a capture.
    void park(ImVec2 p) { move(p); }

    void fail_with(const std::string& why) {
        if (error_.empty()) error_ = why;
    }

private:
    SpecOffscreen& canvas_;
    BuilderPane& pane_;
    std::function<void(const ImRect&)> frame_;
    std::string error_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string workspace, instrument_from, out_dir, prefix = "builder", suffix;
    std::vector<std::string> files;
    int width = 1440, height = 900;
    float text_scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--files") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) files.push_back(argv[++i]);
            continue;
        }
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!v) return usage(("incomplete option " + a).c_str());
        ++i;
        if (a == "--workspace") workspace = v;
        else if (a == "--instrument-from") instrument_from = v;
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--prefix") prefix = v;
        else if (a == "--suffix") suffix = v;
        else if (a == "--text-scale") text_scale = (float)std::atof(v);
        else if (a == "--size") {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else {
            return usage(("unknown option " + a).c_str());
        }
    }
    if (workspace.empty() || instrument_from.empty() || out_dir.empty() || files.empty()) return usage("missing an option");
    if (width < 640 || height < 400 || width > 8192 || height > 8192) return usage("--size must be from 640x400 to 8192x8192");
    if (!(text_scale >= 0.5f && text_scale <= 3.0f)) return usage("--text-scale must be from 0.5 to 3");
    std::error_code ec;
    if (!fs::is_directory(out_dir, ec)) return fail("the folder " + out_dir + " does not exist");
    const auto parsed = cb::spec::parse_toml(read_file(instrument_from));
    if (!parsed.ok()) return fail("cannot read the spec " + instrument_from);
    const cb::spec::Instrument instrument = parsed.spec.instrument;

    auto opened = cb::Session::open(workspace);
    if (opened.error) return fail("cannot open the workspace " + workspace + ": " + opened.error.message);
    std::unique_ptr<cb::Session> session = std::move(opened.session);
    {
        // The empty spec the scenario builds in.
        cb::CommandInvocation inv;
        inv.name = "spec.new";
        inv.args = {{"id", instrument.id}};
        if (instrument.title) inv.args["title"] = *instrument.title;
        if (instrument.description) inv.args["description"] = *instrument.description;
        inv.origin = cb::CommandOrigin::builder;
        const cb::CommandResult r = session->invoke(std::move(inv));
        if (!r.ok) return fail("spec.new: " + r.error.code + ": " + r.error.message);
    }

    BuilderPane::Options po;
    po.group_prefix = "builder-shot";
    BuilderPane pane(std::make_unique<SessionBuilderPort>(*session), po);
    pane.set_build_mode(true);
    // The files dropped on the window: the import goes through the queue,
    // which the app's frame loop drains; here, at once.
    if (!pane.drop_files(files)) return fail("drop: " + pane.error().code + ": " + pane.error().message);
    session->queue().drain();
    pane.update();
    if (pane.importing() || pane.fresh().size() < 2)
        return fail("the import made " + std::to_string(pane.fresh().size()) + " sources: " + pane.error().code + " " +
                    pane.error().message);

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, text_scale, &why);
    if (!canvas) return fail("cannot draw offscreen: " + why);
    SessionPanelSource source(*session);
    std::unique_ptr<SpecRuntime> runtime;
    canvas->with_context([&] {
        runtime = std::make_unique<SpecRuntime>(canvas->textures());
        runtime->set_source(&source);
        ImGui::GetIO().ConfigInputTrickleEventQueue = true;
    });
    const std::string workspace_name = fs::path(workspace).filename().string();
    std::string page_focus;   // the spec page's focused panel, which the pane shares

    // One frame of a page in build mode: the page bar with Home, the spec's
    // title and the pane's items, the status bar, then the work area.
    auto frame = [&](const ImRect&) {
        // The canvas's own window goes behind the page's windows, which, like
        // the app's, do not come to the front on focus and so open at the back.
        ImGui::BringWindowToDisplayBack(ImGui::GetCurrentWindow());
        const SpecLook& look = canvas->look();
        const bool escape = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
        pane.keys(escape);
        {
            const float pad_y = (sternbild::metric::command_bar - ImGui::GetFontSize()) * 0.5f;
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, pad_y));
            const bool bar = ImGui::BeginMainMenuBar();
            ImGui::PopStyleVar();
            if (bar) {
                sternbild::key_button("esc", "Home");
                ImGui::Separator();
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                const auto open = pane.open_spec();
                const std::string title = open ? open->spec.instrument.title.value_or(open->spec.instrument.id) : "";
                ImGui::TextUnformatted(title.c_str());
                ImGui::PopStyleColor();
                ImGui::Separator();
                // As the app's page bar: the pane's items after the title,
                // and ⌘K Command at the right end.
                const char* key = "\xe2\x8c\x98K";
                const float command_w = ImGui::CalcTextSize(key).x + ImGui::CalcTextSize("Command").x + 32.0f;
                const float right = ImGui::GetWindowContentRegionMax().x - command_w - sternbild::metric::pad;
                pane.page_bar(right - ImGui::GetCursorPosX() - 16.0f);
                ImGui::SameLine();
                if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
                sternbild::key_button(key, "Command");
                ImGui::EndMainMenuBar();
            }
        }
        sternbild::status_bar_begin();
        {
            sternbild::status_item(canvas->renderer_name());
            char text[128];
            std::snprintf(text, sizeof text, "workspace %s", workspace_name.c_str());
            sternbild::status_item(text);
            if (const auto& open = pane.open_spec()) {
                std::snprintf(text, sizeof text, open->spec.panels.size() == 1 ? "%zu panel" : "%zu panels", open->spec.panels.size());
                sternbild::status_item(text);
                std::snprintf(text, sizeof text, open->spec.sources.size() == 1 ? "%zu source" : "%zu sources",
                              open->spec.sources.size());
                sternbild::status_item(text);
                if (open->dirty) sternbild::status_item("unsaved");
            }
        }
        sternbild::status_bar_end();
        const ImRect area = static_cast<ImGuiViewportP*>(ImGui::GetMainViewport())->GetBuildWorkRect();
        ImGui::SetNextWindowPos(area.Min);
        ImGui::SetNextWindowSize(area.GetSize());
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        const bool shown = ImGui::Begin("##caliper_spec_page", nullptr,
                                        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                            ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar(3);
        if (shown) pane.draw(area, *runtime, look, &page_focus);   // the page's focus, shared
        ImGui::End();
    };

    Driver drv(*canvas, pane, frame);
    json out{{"files", json::array()}, {"size", {width, height}}};
    int failures = 0;
    const ImVec2 rest((float)width - 3.0f, (float)height - 3.0f);   // the status bar's far corner
    auto shoot = [&](const std::string& name) {
        if (!drv.ok()) return;
        drv.frames(3);
        std::vector<uint8_t> rgba;
        drv.render(&rgba);
        if (!drv.ok()) return;
        std::string err;
        const std::string file = (fs::path(out_dir) / (prefix + "-" + name + suffix + ".png")).string();
        if (!export_write_png_atomic(file, rgba.data(), (uint32_t)width, (uint32_t)height, &err)) {
            ++failures;
            std::fprintf(stderr, "caliper-builder-shot: cannot write %s: %s\n", file.c_str(), err.c_str());
            return;
        }
        out["files"].push_back({{"name", name}, {"file", file}});
    };
    auto expect = [&](bool cond, const std::string& what) {
        if (!cond && drv.ok()) drv.fail_with(what + " (" + pane.error().code + ": " + pane.error().message + ")");
    };
    auto rename = [&](const char* field, const char* id) {
        drv.click(field);
        drv.type(id);
        drv.key(ImGuiKey_Enter);
    };
    auto typed = [&](const char* field, const char* text) {
        drv.click(field);
        drv.type(text);
        drv.key(ImGuiKey_Enter);
    };

    drv.frames(3);
    drv.park(rest);
    // 1. Build mode on the empty spec, with the palettes and the two sources.
    shoot("1-empty");

    // 2. The curves: the metrics table, named, as a series.
    drv.click("source:table:run42_metrics", 0.6f, 0.5f);
    rename("prop:source.id", "metrics");
    drv.click("primitive:series");
    expect(pane.focused_panel() == "series", "the series panel was not made");
    rename("prop:id", "curves");
    expect(pane.focused_panel() == "curves", "the series panel was not renamed");
    typed("prop:title", "Loss and accuracy");
    typed("option:x_label", "Step");
    drv.click("option:y_log");
    drv.park(rest);
    shoot("2-series");

    // 3. The heatmap: the attention tensor, its layer and its mean over heads,
    // dragged beside the curves.
    drv.click("source:tensor:attn_weights", 0.6f, 0.5f);
    rename("prop:source.id", "attn");
    drv.click("transform:add:slice");
    drv.click("transform:add:reduce");
    drv.click("transform:add:normalise");
    expect(pane.selected() == "spec:attn", "the attention source was not added");
    {
        // Onto the curves' right edge, inside its edge band at any size.
        SpecRect c;
        ImVec2 from;
        if (drv.ok() && pane.item_rect("panel:curves", &c) && drv.target("primitive:heatmap", 0.5f, 0.5f, &from))
            drv.drag_points(from, ImVec2(c.right() - 40.0f, std::round(c.y + c.h * 0.55f)), false);
    }
    expect(pane.focused_panel() == "heatmap", "the heatmap was not made");
    {
        const auto& open = pane.open_spec();
        const bool beside = open && open->spec.layout && open->spec.layout->kind == cb::spec::LayoutNode::Kind::split &&
                            open->spec.layout->split == cb::spec::SplitDirection::horizontal;
        expect(beside, "the heatmap is not beside the curves");
    }
    rename("prop:id", "attention");
    typed("prop:title", "Mean attention of layer 0");
    drv.click("colormap:magma");
    typed("option:range:min", "0");
    typed("option:x_label", "Key");
    typed("option:y_label", "Query");
    drv.click("source:spec:attn", 0.6f, 0.5f);
    drv.park(rest);
    shoot("3-heatmap");

    // 4. The table under everything, its columns, and the root divider.
    drv.click("source:spec:metrics", 0.6f, 0.5f);
    {
        SpecRect c;
        ImVec2 from;
        if (drv.ok() && pane.item_rect("canvas", &c) && drv.target("primitive:table", 0.5f, 0.5f, &from))
            drv.drag_points(from, ImVec2(std::round(c.x + c.w * 0.5f), c.bottom() - 4.0f), false);
    }
    expect(pane.focused_panel() == "table", "the table was not made");
    {
        const auto& open = pane.open_spec();
        const bool below = open && open->spec.layout && open->spec.layout->kind == cb::spec::LayoutNode::Kind::split &&
                           open->spec.layout->split == cb::spec::SplitDirection::vertical;
        expect(below, "the table is not under everything");
    }
    rename("prop:id", "rows");
    drv.click("option:columns:accuracy");
    {
        SpecRect dv, c;
        if (drv.ok() && pane.item_rect("divider:", &dv) && pane.item_rect("canvas", &c)) {
            const ImVec2 a(std::round(dv.x + dv.w * 0.5f), dv.y);
            const ImVec2 b(a.x, std::round(c.y + 0.65f * (c.h - 1.0f)));
            drv.drag_points(a, b, true);
            shoot("4-divider");
            drv.release();
        } else {
            drv.fail_with("no root divider");
        }
    }

    // 5. The spec text beside the finished dashboard.
    drv.click("bar:text");
    expect(pane.text_open(), "the text strip did not open");
    drv.click("panel:curves", 0.5f, 0.6f);
    drv.park(rest);
    const std::string finished = pane.open_spec() ? pane.open_spec()->text : std::string();
    shoot("5-text");

    // 6. Two undos.
    const size_t steps_before = session->registry().history_state(200).undo.size();
    drv.key(ImGuiKey_Z, ImGuiMod_Ctrl);
    drv.key(ImGuiKey_Z, ImGuiMod_Ctrl);
    const size_t steps_after = session->registry().history_state(200).undo.size();
    drv.park(rest);
    shoot("6-undo");
    const std::string undone = pane.open_spec() ? pane.open_spec()->text : std::string();
    // Both back, and saved.
    drv.key(ImGuiKey_Z, ImGuiMod_Ctrl | ImGuiMod_Shift);
    drv.key(ImGuiKey_Z, ImGuiMod_Ctrl | ImGuiMod_Shift);
    drv.key(ImGuiKey_S, ImGuiMod_Ctrl);
    drv.frames(2);

    if (!drv.ok()) {
        std::fprintf(stderr, "caliper-builder-shot: %s\n", drv.error().c_str());
        ++failures;
    }
    out["finished_text"] = finished;
    out["undone_text"] = undone;
    out["final_text"] = pane.open_spec() ? pane.open_spec()->text : std::string();
    out["final_path"] = pane.open_spec() && pane.open_spec()->path ? *pane.open_spec()->path : std::string();
    out["undo_steps_before_undo"] = steps_before;
    out["undo_steps_after_undo"] = steps_after;
    out["commands"] = json::array();
    for (const auto& c : pane.sent())
        out["commands"].push_back({{"name", c.name}, {"origin", cb::command_origin_name(c.origin)}, {"group", c.group},
                                   {"merge_key", c.merge_key}});
    canvas->with_context([&] { runtime.reset(); });
    canvas.reset();
    std::printf("%s\n", out.dump().c_str());
    session->close();
    return failures ? 1 : 0;
}
