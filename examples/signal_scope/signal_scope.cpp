// ============================================================================
// SignalScope — the exemplar Caliper applet.
//
// This file demonstrates every idiom an applet author needs. Copy this
// directory to start a new applet. The numbered EXEMPLAR comments are the
// teaching points; everything else is ordinary application code.
//
// The rules it embodies (PLATFORM.md §6):
//   - talk to the host ONLY through services obtained by name
//   - render ONLY through ImGui/ImPlot (no raw GL/Metal — that is what keeps
//     the host's renderer swappable under you, forever)
//   - never block the frame thread (see the Anti-patterns section, which
//     exists so you can watch the host's watchdog catch it)
//   - persist state ONLY under the host-provided per-applet data dir
// ============================================================================

// EXEMPLAR 1 — includes: one SDK header, then raw ImGui/ImPlot. There is no
// wrapper layer to learn; the pinned UI stack is part of the SDK contract.
#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <thread>

namespace {

constexpr float kTau = 6.28318530f;
constexpr int   kMaxLeads = 3;
constexpr float kSampleHz = 250.0f;

// Fixed-size ring buffer in the classic ImPlot scrolling-plot shape.
struct ScrollingSeries {
    static constexpr int N = 2048;
    float xs[N]{}, ys[N]{};
    int count = 0, next = 0;
    void push(float x, float y) {
        xs[next] = x; ys[next] = y;
        next = (next + 1) % N;
        count = std::min(count + 1, N);
    }
    int offset() const { return count < N ? 0 : next; }
};

// Synthetic ECG-ish lead: slow baseline wander + a periodic QRS-like spike.
float synth_lead(float t, float phase) {
    float wander = 0.06f * std::sin(kTau * 0.33f * t + phase);
    float beat   = std::pow(std::max(0.0f, std::sin(kTau * 1.2f * t + phase)), 24.0f);
    return wander + beat;
}

} // namespace

class SignalScope final : public caliper::Applet {
public:
    // EXEMPLAR 2 — on_init: the Host& handed to you here is valid for your
    // whole lifetime; keep a pointer. Log through the host (shows up in the
    // host console / dev-mode tail), never printf.
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        host.log_info("signal-scope: on_init");

        // EXEMPLAR 3 — optional services are PROBED, never assumed. This id
        // ships in Phase 2; on today's host the probe returns nullptr and the
        // applet degrades gracefully. Required services (in the manifest) are
        // guaranteed present — the host refuses to load you otherwise, so no
        // null-checking ceremony is needed for those.
        has_metrics_ = host.service("caliper.metrics.v1") != nullptr;

        // EXEMPLAR 4 — the per-applet sandbox dir is the ONLY place to write.
        // The host namespaces it by applet id; no path-building, no dotfiles.
        settings_path_ = std::string(host.data_dir()) + "/settings.txt";
        load_settings();
        return true;
    }

    // EXEMPLAR 5 — on_frame: everything visible happens here, and nothing
    // slow. Drive animation from frame.delta_sec/time_sec (never wall-clock
    // sleeps or your own timers). ImGui coordinates are LOGICAL units;
    // frame.fb_* are PHYSICAL pixels; dpi_scale converts (§6a pixel contract).
    void on_frame(const caliper::Frame& frame) override {
        step_simulation(frame);

        ImGui::SetNextWindowPos({40, 60}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({640, 480}, ImGuiCond_FirstUseEver); // logical units
        ImGui::Begin("SignalScope");

        ImGui::TextDisabled(
            "framebuffer %dx%d px  |  dpi x%.1f  |  logical %.0fx%.0f  |  metrics service: %s",
            frame.fb_width, frame.fb_height, frame.dpi_scale,
            frame.fb_width / frame.dpi_scale, frame.fb_height / frame.dpi_scale,
            has_metrics_ ? "present" : "absent (ok — optional)");

        ImGui::Checkbox("pause", &paused_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat("speed", &speed_, 0.1f, 4.0f, "%.1fx");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt("leads", &active_leads_, 1, kMaxLeads);

        if (ImPlot::BeginPlot("##scope", {-1, 300})) {
            ImPlot::SetupAxes("t (s)", "mV");
            ImPlot::SetupAxisLimits(ImAxis_X1, sim_t_ - 8.0, sim_t_, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -0.4, 1.3);
            static const char* kNames[kMaxLeads] = {"lead I", "lead II", "lead III"};
            for (int k = 0; k < active_leads_; k++)
                ImPlot::PlotLine(kNames[k], leads_[k].xs, leads_[k].ys,
                                 leads_[k].count, 0, leads_[k].offset(),
                                 sizeof(float));
            ImPlot::EndPlot();
        }

        // EXEMPLAR 6 — the anti-pattern, kept on purpose as a live demo of the
        // host watchdog (§15). Long work belongs in caliper.jobs.v1 (Phase 2);
        // blocking here freezes every applet and the host UI. Check the box,
        // watch the menu bar flag you, uncheck. The flag latches until
        // relaunch — by design, so the evidence doesn't scroll away.
        if (ImGui::CollapsingHeader("Anti-patterns (watchdog demo)")) {
            ImGui::Checkbox("block the frame thread (wrong!)", &block_frame_);
            ImGui::TextWrapped("While checked, this sleeps 300 ms inside frame(). "
                               "The host's watchdog flags this applet after 3 "
                               "consecutive slow frames. Real work goes in "
                               "background jobs, never here.");
        }
        if (block_frame_)
            std::this_thread::sleep_for(std::chrono::milliseconds(300));

        ImGui::End();
    }

    // EXEMPLAR 7 — on_cleanup: symmetric with on_init; persist, release, log.
    // After this returns, destroy() runs; never touch host services after it.
    void on_cleanup() override {
        save_settings();
        if (host_) host_->log_info("signal-scope: on_cleanup");
    }

private:
    void step_simulation(const caliper::Frame& frame) {
        if (paused_) return;
        // Advance in fixed substeps accumulated from delta_sec so sample
        // density is framerate-independent (and capped, so a debugger pause
        // doesn't spiral).
        accum_ += std::min(frame.delta_sec, 0.25) * speed_;
        const double dt = 1.0 / kSampleHz;
        while (accum_ >= dt) {
            accum_ -= dt;
            sim_t_ += dt;
            for (int k = 0; k < kMaxLeads; k++)
                leads_[k].push((float)sim_t_,
                               synth_lead((float)sim_t_, k * 0.7f));
        }
    }

    void load_settings() {
        std::ifstream in(settings_path_);
        if (!in) return;                       // first run: defaults
        std::string key;
        float value = 0.0f;
        while (in >> key >> value)
            if (key == "speed") speed_ = std::clamp(value, 0.1f, 4.0f);
        if (host_) host_->log_info("signal-scope: settings restored");
    }

    void save_settings() {
        std::ofstream out(settings_path_, std::ios::trunc);
        if (out) out << "speed " << speed_ << "\n";
    }

    caliper::Host* host_ = nullptr;
    bool has_metrics_ = false;
    std::string settings_path_;

    ScrollingSeries leads_[kMaxLeads];
    double sim_t_ = 0.0, accum_ = 0.0;
    float speed_ = 1.0f;
    int active_leads_ = 3;
    bool paused_ = false;
    bool block_frame_ = false;
};

// EXEMPLAR 8 — the entire ABI surface of this dylib. The macro generates the
// descriptor, the five C bridge functions (exception-safe at the boundary),
// and the ui::connect() call that shares the host's ImGui contexts +
// allocators. Field order is fixed: id, version, name, summary, tag, services.
// The id/version/services here MUST agree with signal_scope.caliper.toml —
// the loader verifies that agreement and refuses the applet if they drift.
CALIPER_APPLET(SignalScope,
    .id       = "dev.caliper.signal-scope",
    .version  = "0.1.0",
    .name     = "SignalScope",
    .summary  = "Exemplar applet: live multi-lead signal viewer showing every "
                "Caliper SDK idiom, including what not to do.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
