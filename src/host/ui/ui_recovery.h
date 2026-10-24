#pragma once
// ============================================================================
// UI stack recovery after an applet fault (crash_guard.h, PLATFORM.md §15).
// An applet that throws or faults partway through frame() leaves open
// whatever it had begun on the shared ImGui / ImPlot / ImPlot3D contexts
// (caliper.ui.v1): windows, child windows, tables, tab bars, tree nodes,
// groups, IDs, disabled blocks, style, colour, font and item-flag pushes; a
// 2-D plot, subplot grid or aligned group, a 3-D plot, and their style and
// colormap pushes. Left alone, the host's next window would nest inside the
// applet's, and the next BeginPlot would find the old one still open.
//
//   const UiCheckpoint cp = UiCheckpoint::store();   // before the guarded call
//   if (!loader.frame(i, info)) cp.recover();        // after a call that failed
//
// store() copies a few stack sizes (ImGui's own ErrorRecoveryStoreState and
// three vector sizes per plot library): no allocation and no call into either
// plot library, so the normal path costs a handful of loads. recover() unwinds
// to the checkpoint with ImGui's ErrorRecoveryTryToRecoverState, then resets
// the plot libraries' current-plot state for the frame. Both run on the frame
// thread inside one ImGui frame, store() outside any window the call might
// end and outside any plot (the host draws none around an applet call).
//
// Recovery is best effort, as ImGui says of its own: a fault inside an ImGui
// call can leave that call's state half-written. It never aborts the host:
// ImGui's asserts on recoverable errors are switched off for its duration (a
// debug build would otherwise stop at the first "Missing End()"), its red
// error tooltip with them (a developer aid, and off the house palette), and
// what was unwound is reported in one "[ui]" line on stderr instead.
// Header-only, like frame_watchdog.h, so the host and the tests share it.
// ============================================================================

#include <imgui.h>
#include <imgui_internal.h>   // ImGuiErrorRecoveryState, ErrorCallback
#include <implot.h>
#include <implot_internal.h>  // ImPlotContext, ResetCtxForNext*
#include <implot3d.h>
#include <implot3d_internal.h>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace caliper_host {

class UiCheckpoint {
public:
    // The current stack sizes. Outside an ImGui frame (an applet launched
    // before the first frame) there is nothing to unwind and the checkpoint
    // is empty: recover() then does nothing.
    static UiCheckpoint store() {
        UiCheckpoint cp;
        ImGuiContext* g = ImGui::GetCurrentContext();
        if (!g || !g->WithinFrameScope || !g->CurrentWindow) return cp;
        cp.valid_ = true;
        ImGui::ErrorRecoveryStoreState(&cp.imgui_);
        cp.combo_depth_ = g->BeginComboDepth;
        cp.drag_source_ = g->DragDropWithinSource;
        cp.drag_target_ = g->DragDropWithinTarget;
        if (const ImPlotContext* p = ImPlot::GetCurrentContext())
            cp.implot_ = {p->ColorModifiers.Size, p->StyleModifiers.Size,
                          p->ColormapModifiers.Size};
        if (const ImPlot3DContext* p = ImPlot3D::GetCurrentContext())
            cp.implot3d_ = {p->ColorModifiers.Size, p->StyleModifiers.Size,
                            p->ColormapModifiers.Size};
        return cp;
    }

    // Unwinds everything begun or pushed since store(). Returns how many
    // items were closed or popped (0 when the call left nothing open).
    int recover() const {
        ImGuiContext* g = valid_ ? ImGui::GetCurrentContext() : nullptr;
        if (!g || !g->WithinFrameScope) return 0;
        Report report;

        // Plots first: their state lives in the plot contexts, not on ImGui's
        // stacks, and resetting it touches neither ImGui nor the windows the
        // plots were drawn in. Their pushes onto ImGui's ID and clip stacks
        // go with those windows below.
        recover_implot(report);
        recover_implot3d(report);

        // ImGui: End/EndChild/EndTable/... down to the checkpoint, every
        // recoverable error routed to the report instead of IM_ASSERT, the
        // debug log or the error tooltip. The settings are put back after.
        ImGuiIO& io = g->IO;
        const bool recovery = io.ConfigErrorRecovery;
        const bool assert_on = io.ConfigErrorRecoveryEnableAssert;
        const bool log_on = io.ConfigErrorRecoveryEnableDebugLog;
        const bool tooltip_on = io.ConfigErrorRecoveryEnableTooltip;
        const ImGuiErrorCallback callback = g->ErrorCallback;
        void* const callback_data = g->ErrorCallbackUserData;
        io.ConfigErrorRecovery = true;   // End() unwinds a window's own stacks only when set
        io.ConfigErrorRecoveryEnableAssert = false;
        io.ConfigErrorRecoveryEnableDebugLog = false;
        io.ConfigErrorRecoveryEnableTooltip = false;
        g->ErrorCallback = [](ImGuiContext*, void* user, const char* msg) {
            static_cast<Report*>(user)->add(msg ? msg : "error");
        };
        g->ErrorCallbackUserData = &report;

        ImGui::ErrorRecoveryTryToRecoverState(&imgui_);

        g->ErrorCallback = callback;
        g->ErrorCallbackUserData = callback_data;
        io.ConfigErrorRecovery = recovery;
        io.ConfigErrorRecoveryEnableAssert = assert_on;
        io.ConfigErrorRecoveryEnableDebugLog = log_on;
        io.ConfigErrorRecoveryEnableTooltip = tooltip_on;

        // State no stack records. A SetNextWindow*/SetNextItem* the applet
        // made just before it failed would otherwise apply to the host's next
        // window or item; a combo or drag-and-drop block left open would
        // misname the next combo or refuse the next drop target.
        g->NextWindowData.ClearFlags();
        g->NextItemData.ClearFlags();
        if (g->BeginComboDepth > combo_depth_) {
            g->BeginComboDepth = combo_depth_;
            report.add("Missing EndCombo()");
        }
        if (g->DragDropWithinSource != drag_source_ ||
            g->DragDropWithinTarget != drag_target_) {
            g->DragDropWithinSource = drag_source_;
            g->DragDropWithinTarget = drag_target_;
            report.add("Missing EndDragDropSource()/EndDragDropTarget()");
        }

        if (report.total > 0)
            std::fprintf(stderr, "[ui] unwound after an applet fault: %s\n",
                         report.text().c_str());
        return report.total;
    }

private:
    struct PlotStacks {
        int colors = 0;
        int styles = 0;
        int colormaps = 0;
    };

    // Distinct messages with their counts, in first-seen order: "Missing
    // End() x2, Missing PopStyleColor(), ImPlot: missing EndPlot()".
    struct Report {
        std::vector<std::pair<std::string, int>> lines;
        int total = 0;
        void add(const char* msg) {
            ++total;
            for (auto& [text, n] : lines)
                if (text == msg) {
                    ++n;
                    return;
                }
            lines.emplace_back(msg, 1);
        }
        std::string text() const {
            std::string out;
            for (const auto& [msg, n] : lines) {
                if (!out.empty()) out += ", ";
                out += msg;
                if (n > 1) out += " x" + std::to_string(n);
            }
            return out;
        }
    };

    // ImPlot keeps its open plot, subplot grid and aligned group as context
    // pointers. Ending a half-built plot properly (EndPlot) would render and
    // handle input for it from wherever the applet stopped, so the plot is
    // dropped for this frame instead, through ImPlot's own resets; the style
    // and colormap pushes are popped through ImPlot, which restores each
    // value it saved.
    void recover_implot(Report& report) const {
        ImPlotContext* p = ImPlot::GetCurrentContext();
        if (!p) return;
        if (p->CurrentPlot) {
            ImPlot::ResetCtxForNextPlot(p);   // also drops Next*Data and the item
            report.add("ImPlot: missing EndPlot()");
        }
        if (p->CurrentSubplot) {
            ImPlot::ResetCtxForNextSubplot(p);
            report.add("ImPlot: missing EndSubplots()");
        }
        if (p->CurrentAlignmentH || p->CurrentAlignmentV) {
            ImPlot::ResetCtxForNextAlignedPlots(p);
            report.add("ImPlot: missing EndAlignedPlots()");
        }
        p->CurrentItems = nullptr;
        pop_plot_stacks(report, "ImPlot", implot_, p->ColorModifiers.Size,
                        p->StyleModifiers.Size, p->ColormapModifiers.Size,
                        &ImPlot::PopStyleColor, &ImPlot::PopStyleVar,
                        &ImPlot::PopColormap);
    }

    // ImPlot3D buffers a plot's triangles in the plot's own draw list until
    // EndPlot sorts them into the window's, so an abandoned plot's buffer is
    // emptied too: otherwise the next plot with that ID (a restarted
    // applet's) would draw the stale triangles once.
    void recover_implot3d(Report& report) const {
        ImPlot3DContext* p = ImPlot3D::GetCurrentContext();
        if (!p) return;
        if (p->CurrentPlot) {
            p->CurrentPlot->DrawList.ResetBuffers();
            p->CurrentPlot = nullptr;
            report.add("ImPlot3D: missing EndPlot()");
        }
        p->CurrentItems = nullptr;
        p->CurrentItem = nullptr;
        p->NextItemData.Reset();
        pop_plot_stacks(report, "ImPlot3D", implot3d_, p->ColorModifiers.Size,
                        p->StyleModifiers.Size, p->ColormapModifiers.Size,
                        &ImPlot3D::PopStyleColor, &ImPlot3D::PopStyleVar,
                        &ImPlot3D::PopColormap);
    }

    static void pop_plot_stacks(Report& report, const char* lib, PlotStacks at,
                                int colors, int styles, int colormaps,
                                void (*pop_color)(int), void (*pop_style)(int),
                                void (*pop_colormap)(int)) {
        const std::string prefix = std::string(lib) + ": missing ";
        auto pop = [&](int now, int then, void (*fn)(int), const char* what) {
            for (int i = then; i < now; ++i) report.add((prefix + what).c_str());
            if (now > then) fn(now - then);
        };
        pop(colors, at.colors, pop_color, "PopStyleColor()");
        pop(styles, at.styles, pop_style, "PopStyleVar()");
        pop(colormaps, at.colormaps, pop_colormap, "PopColormap()");
    }

    bool valid_ = false;
    ImGuiErrorRecoveryState imgui_;
    int combo_depth_ = 0;
    bool drag_source_ = false;
    bool drag_target_ = false;
    PlotStacks implot_;
    PlotStacks implot3d_;
};

}  // namespace caliper_host
