#include "runs_dashboard.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "metrics_store.h"

namespace caliper_host {
namespace {

// Persisted UI state, keyed by run id so a selected run survives runs()
// reordering, and each tag remembers its own smoothing factor across frames.
// (Selection is tracked by run id, not list index — see note below.)
uint64_t g_selected_run = 0;                      // 0 = nothing selected
std::unordered_map<std::string, float> g_smooth;  // key: "<run>/<tag>" -> alpha

// EMA: smoothed_0 = x_0, smoothed_i = alpha*smoothed_{i-1} + (1-alpha)*x_i.
// alpha == 0 reproduces the raw series exactly.
void ema(const std::vector<double>& x, float alpha, std::vector<double>& out) {
    out.resize(x.size());
    if (x.empty()) return;
    double s = x[0];
    out[0] = s;
    for (size_t i = 1; i < x.size(); ++i) {
        s = alpha * s + (1.0 - alpha) * x[i];
        out[i] = s;
    }
}

}  // namespace

void render_runs_dashboard(MetricsStore& store, bool* p_open) {
    if (!p_open || !*p_open) return;  // gate all queries on the window being open

    ImGui::SetNextWindowSize({720.0f, 460.0f}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Runs", p_open)) {
        ImGui::End();
        return;
    }

    // Per-frame query — cheap (mutex-guarded, trivial row counts in v1).
    std::vector<RunInfo> runs = store.runs();

    // Resolve the selected run id to a live entry; if it vanished (or nothing
    // is selected yet), fall back to the first run. Index is never persisted,
    // so a shrink/reorder of runs() can never dangle.
    const RunInfo* selected = nullptr;
    for (const auto& r : runs) {
        if (r.id == g_selected_run) { selected = &r; break; }
    }
    if (!selected && !runs.empty()) {
        selected = &runs.front();
        g_selected_run = selected->id;
    }

    // --- left pane: run list + history management ---
    ImGui::BeginChild("run_list", {220.0f, 0.0f}, true);
    if (runs.empty()) {
        ImGui::TextDisabled("no runs yet");
    }
    uint64_t delete_id = 0;   // deferred: never mutate the store mid-iteration
    for (const auto& r : runs) {
        ImGui::PushID((int)r.id);
        std::string label = r.experiment + "/" + r.name;
        bool is_selected = (r.id == g_selected_run);
        if (ImGui::Selectable(label.c_str(), is_selected)) {
            g_selected_run = r.id;
        }
        // Right-click a run to delete it (a live run keeps writing into the
        // void afterward — the store makes deleted ids inert, never crashy).
        if (ImGui::BeginPopupContextItem("##runctx")) {
            if (ImGui::MenuItem("Delete run")) delete_id = r.id;
            ImGui::EndPopup();
        }
        if (!r.done) {
            ImGui::SameLine();
            ImGui::TextColored({0.40f, 0.80f, 0.45f, 1.0f}, "\xE2\x97\x8f");  // ●
        }
        ImGui::PopID();
    }
    if (!runs.empty()) {
        ImGui::Separator();
        // Two-click clear: arm, then confirm — no modal, no accidents.
        static bool clear_armed = false;
        if (!clear_armed) {
            if (ImGui::SmallButton("Clear history…")) clear_armed = true;
        } else {
            if (ImGui::SmallButton("Really clear ALL runs")) {
                store.clear_all();
                clear_armed = false;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("keep")) clear_armed = false;
        }
    }
    ImGui::EndChild();
    if (delete_id != 0) store.delete_run(delete_id);

    ImGui::SameLine();

    // --- right pane: hparams + one plot per tag ---
    ImGui::BeginChild("run_detail", {0.0f, 0.0f}, false);
    if (!selected) {
        ImGui::TextDisabled("select a run");
        ImGui::EndChild();
        ImGui::End();
        return;
    }

    ImGui::TextUnformatted((selected->experiment + " / " + selected->name).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled(selected->done ? "(done)" : "(running)");
    ImGui::Separator();
    ImGui::TextWrapped("hparams: %s",
                       selected->hparams.empty() ? "—" : selected->hparams.c_str());
    ImGui::Separator();

    std::vector<std::string> tags = store.scalar_tags(selected->id);
    if (tags.empty()) {
        ImGui::TextDisabled("no scalar tags");
    }

    std::vector<double> steps, values, smoothed;
    for (const auto& tag : tags) {
        ImGui::PushID(tag.c_str());
        ImGui::TextUnformatted(tag.c_str());

        std::string key = std::to_string(selected->id) + "/" + tag;
        float& alpha = g_smooth[key];  // default-constructs to 0.0f
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("smoothing", &alpha, 0.0f, 0.99f, "%.2f");

        auto pts = store.scalars(selected->id, tag);
        steps.clear();
        values.clear();
        steps.reserve(pts.size());
        values.reserve(pts.size());
        for (const auto& p : pts) {
            steps.push_back((double)p.first);
            values.push_back(p.second);
        }
        ema(values, alpha, smoothed);

        if (ImPlot::BeginPlot(tag.c_str(), {-1.0f, 180.0f}, ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("step", nullptr,
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (!steps.empty()) {
                if (alpha > 0.0f) {
                    // faint raw underneath the smoothed line
                    ImPlot::PlotLine("raw", steps.data(), values.data(),
                                     (int)steps.size(),
                                     ImPlotSpec(ImPlotProp_LineColor,
                                                ImVec4(0.5f, 0.5f, 0.6f, 0.35f),
                                                ImPlotProp_LineWeight, 1.0f));
                }
                ImPlot::PlotLine("smoothed", steps.data(), smoothed.data(),
                                 (int)steps.size(),
                                 ImPlotSpec(ImPlotProp_LineColor,
                                            ImVec4(0.40f, 0.65f, 0.95f, 1.0f),
                                            ImPlotProp_LineWeight, 2.0f));
            }
            ImPlot::EndPlot();
        }
        ImGui::PopID();
    }

    ImGui::EndChild();
    ImGui::End();
}

}  // namespace caliper_host
