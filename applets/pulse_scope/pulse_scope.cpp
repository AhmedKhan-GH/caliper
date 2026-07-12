// ============================================================================
// PulseScope — the caliper.feed.v1 dashboard exemplar (id dev.caliper.pulse-scope
// 0.1.0). The live dashboard of THIS machine's own vitals.
//
// ONE job: make the feed service — and its honest-loss contract — visible. The
// host owns telemetry providers (this Mac's sudo-free sensors); PulseScope is a
// pure READER. It enumerates whatever channels the host vends (never a hardcoded
// list — other machines vend fewer, and the ladder covers zero), polls each on a
// jobs worker at ~10 Hz into an applet-side ring copy, and draws one INPUT-LOCKED
// ImPlot strip chart per channel with its current value + units, a climbing
// "last sample N.N s ago" staleness label, and a cumulative "lost M samples" gap
// counter (a seq jump between polls == the host ring overwrote == data lost).
//
// Deliberately CPU + ImPlot only: no tensors, no bridge, no geometry (the
// flagship thermal-twin consumes the SAME feed THROUGH tensors later). The pure
// ring-copy + gap logic lives in pulse_ring.h and is unit-tested torch-free.
//
// Threading (house frame-thread discipline): the poller is a jobs worker; it
// owns the per-channel read cursors and writes the ring copies under `mtx`.
// draw_ui SNAPSHOTS the rings under the same mutex, then renders lock-free.
//
// Honest ladder: no feed caps / zero channels -> one line ("telemetry
// unavailable on this host"), nothing faked. A channel that stops sampling ->
// its chart freezes (no new points) and its staleness label climbs; no
// interpolation, no invented values.
// ============================================================================
#include "pulse_scope.h"
#include "pulse_ring.h"

#include <caliper/caliper.hpp>
#include <caliper/services/feed_v1.h>
#include <imgui.h>
#include <implot.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pulsescope {
namespace {

// Read-only viewer flags: these strip charts are watched, not driven — no
// pan/zoom/box-select, no context menus (the recurring "input-lock read-only
// plots" house rule).
constexpr ImPlotFlags kLockedPlot = ImPlotFlags_NoInputs | ImPlotFlags_NoMenus |
                                    ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;

constexpr int   kPollPeriodMs   = 100;   // ~10 Hz, matching the provider
constexpr int   kLogEveryTicks  = 20;    // periodic channel summary: every ~2 s
constexpr std::size_t kRingPoints = 2000; // applet-side scrolling window (~200 s)

// Process-global monotonic clock in ns. Used ONLY for staleness ("how long since
// this channel last delivered a fresh sample") — clock-epoch-agnostic, so it
// never has to match the host's t_ns epoch. The x-AXIS uses t_ns directly.
int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool is_thermal(const char* id) {
    return std::strcmp(id, "sys.thermal.state") == 0;
}

// STALE threshold for a channel: a few missed sample periods (or 2 s when the
// rate is irregular / unknown).
double stale_limit_s(float nominal_hz) {
    return nominal_hz > 0.0f ? 5.0 / nominal_hz : 2.0;
}

// One channel's applet-side state. `info` and the ring cursor are set once /
// owned by the poller; `ring`, `last_fresh_steady_ns` and `ever` are shared with
// draw_ui under PulseState::mtx.
struct ChannelView {
    CaliperFeedChannelInfo info{};
    PulseRing ring{kRingPoints};
    int64_t   last_fresh_steady_ns = 0;   // when a NEW sample last arrived
    bool      ever = false;               // have we ever seen a sample?
};

}  // namespace

struct PulseState {
    caliper::Host*      host = nullptr;
    caliper::Jobs       jobs;
    const CaliperFeedV1* feed = nullptr;
    uint32_t            feed_caps = 0;

    uint64_t          job_id = 0;
    std::atomic<bool> stop{false};

    // frame-thread knob — touched only from draw_ui (draw-thread-only).
    float window_s = 60.0f;

    std::mutex mtx;                 // guards everything below
    int64_t    epoch_ns = 0;        // first t_ns ever seen; the shared axis origin
    bool       epoch_set = false;
    std::vector<ChannelView> chans; // enumerated once at init (v0 channels static)
};

namespace {

// The poller: for each enumerated channel, drain new samples past our cursor into
// the ring copy, honestly accounting seq-gap loss, and stamp freshness. Emits a
// periodic one-line summary per channel (the load-proof artifact).
void pulse_job(PulseState* st, const CaliperJobControl* ctl) {
    const uint32_t n = static_cast<uint32_t>(st->chans.size());
    std::vector<uint64_t> cursor(n, 0);        // poller-owned read cursors
    std::vector<CaliperFeedSample> buf(512);   // one poll's worth, generous
    int tick = 0;

    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        for (uint32_t i = 0; i < n; ++i) {
            const char* id = st->chans[i].info.id;
            uint32_t got = st->feed->read(id, buf.data(),
                                          static_cast<uint32_t>(buf.size()),
                                          &cursor[i]);
            if (got == 0) continue;   // no new samples (or inert feed) -> stays stale
            const int64_t fresh_at = steady_ns();
            std::lock_guard<std::mutex> lk(st->mtx);
            if (!st->epoch_set) {
                st->epoch_ns = buf[0].t_ns;   // first sample across all channels
                st->epoch_set = true;
            }
            st->chans[i].ring.ingest(buf.data(), got, st->epoch_ns);
            st->chans[i].last_fresh_steady_ns = fresh_at;
            st->chans[i].ever = true;
        }

        // Periodic channel summary — the artifact that proves channels move
        // under load ("pulse: sys.cpu.util 87.3% (fresh, 0 gaps)").
        if (st->host && ++tick >= kLogEveryTicks) {
            tick = 0;
            const int64_t now = steady_ns();
            for (uint32_t i = 0; i < n; ++i) {
                std::lock_guard<std::mutex> lk(st->mtx);
                const ChannelView& c = st->chans[i];
                if (!c.ever) continue;
                const double age = static_cast<double>(now - c.last_fresh_steady_ns) * 1e-9;
                const bool fresh = age <= stale_limit_s(c.info.nominal_hz);
                char line[192];
                if (is_thermal(c.info.id)) {
                    const int code = static_cast<int>(std::lround(c.ring.last_value()));
                    std::snprintf(line, sizeof line,
                                  "pulse: %s %d %s (%s, %llu gaps)", c.info.id, code,
                                  thermal_word(code), fresh ? "fresh" : "stale",
                                  static_cast<unsigned long long>(c.ring.gap_total()));
                } else {
                    std::snprintf(line, sizeof line,
                                  "pulse: %s %.1f%s (%s, %llu gaps)", c.info.id,
                                  static_cast<double>(c.ring.last_value()), c.info.units,
                                  fresh ? "fresh" : "stale",
                                  static_cast<unsigned long long>(c.ring.gap_total()));
                }
                st->host->log_info(line);
            }
        }

        // Sleep the poll period in small chunks so cancel is responsive.
        for (int slept = 0; slept < kPollPeriodMs &&
                            !st->stop.load() && !(ctl && ctl->cancelled(ctl));
             slept += 25)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
}

void pulse_job_tramp(void* user, const CaliperJobControl* ctl) {
    pulse_job(static_cast<PulseState*>(user), ctl);
}

// A per-channel snapshot copied out under the mutex, so drawing touches no
// shared state.
struct ChanSnap {
    CaliperFeedChannelInfo info;
    std::vector<double> xs, ys;
    uint64_t gap_total;
    int64_t  last_fresh_steady_ns;
    bool     ever;
    float    last_value;
};

}  // namespace

PulseScopeApplet::PulseScopeApplet() : s_(std::make_unique<PulseState>()) {}
PulseScopeApplet::~PulseScopeApplet() = default;

bool PulseScopeApplet::initialize(caliper::Host& host) {
    s_->host = &host;
    s_->jobs = caliper::Jobs(host);
    s_->feed = static_cast<const CaliperFeedV1*>(host.service(CALIPER_FEED_V1));

    // Enumerate the channels this host vends — DYNAMICALLY, never a hardcoded
    // list (other machines vend fewer; the ladder covers zero).
    if (s_->feed && s_->feed->caps) s_->feed_caps = s_->feed->caps();
    if (s_->feed && s_->feed->channel_count && s_->feed->channel_info) {
        const uint32_t count = s_->feed->channel_count();
        for (uint32_t i = 0; i < count; ++i) {
            ChannelView cv;
            cv.info.struct_size = sizeof(CaliperFeedChannelInfo);
            if (s_->feed->channel_info(i, &cv.info) == 1)
                s_->chans.push_back(std::move(cv));
        }
    }

    char msg[128];
    std::snprintf(msg, sizeof msg,
                  "pulse-scope: on_init — %zu channel(s) enumerated (caps 0x%x)",
                  s_->chans.size(), s_->feed_caps);
    host.log_info(msg);

    // Only spawn the poller when there is something to poll.
    if (!s_->chans.empty())
        s_->job_id = s_->jobs.submit("pulse_scope: poll", &pulse_job_tramp, s_.get());
    return true;
}

void PulseScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex, then draw lock-free ----
    std::vector<ChanSnap> snap;
    const bool has_live = (st->feed_caps & CALIPER_FEED_CAP_LIVE) != 0u &&
                          !st->chans.empty();
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        snap.reserve(st->chans.size());
        for (const auto& c : st->chans) {
            ChanSnap s;
            s.info = c.info;
            s.xs = c.ring.xs();            // copy the scrolling window (<= kRingPoints)
            s.ys = c.ring.ys();
            s.gap_total = c.ring.gap_total();
            s.last_fresh_steady_ns = c.last_fresh_steady_ns;
            s.ever = c.ever;
            s.last_value = c.ring.last_value();
            snap.push_back(std::move(s));
        }
    }

    ImGui::SetNextWindowSize(ImVec2(760, 900), ImGuiCond_FirstUseEver);
    ImGui::Begin("PulseScope");

    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextColored({0.98f, 0.86f, 0.55f, 1.0f}, "this machine, live");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Spacing();

    // ---- honest ladder: no telemetry on this host -> one line, nothing else ----
    if (!has_live) {
        ImGui::TextDisabled("telemetry unavailable on this host");
        ImGui::End();
        return;
    }

    float window = st->window_s;
    ImGui::SetNextItemWidth(240);
    if (ImGui::SliderFloat("window (s)", &window, 5.0f, 200.0f, "%.0f"))
        st->window_s = window;
    ImGui::SameLine();
    ImGui::TextDisabled("%zu channels   ·   %.0f FPS",
                        snap.size(), ImGui::GetIO().Framerate);
    ImGui::Separator();

    const int64_t now = steady_ns();

    for (const auto& s : snap) {
        const bool thermal = is_thermal(s.info.id);
        ImGui::SeparatorText(s.info.name);

        // ---- current value + units (thermal: label the unitless 0..3 code) ----
        if (!s.ever) {
            ImGui::TextDisabled("no samples yet");
        } else if (thermal) {
            const int code = static_cast<int>(std::lround(s.last_value));
            ImGui::Text("%s", thermal_word(code));
            ImGui::SameLine();
            ImGui::TextDisabled("(state %d of 3)", code);
        } else {
            ImGui::Text("%.1f %s", static_cast<double>(s.last_value), s.info.units);
        }

        // ---- staleness: climbs while a channel stops delivering (frozen chart) ----
        ImGui::SameLine();
        if (!s.ever) {
            // nothing to say yet
        } else {
            const double age = static_cast<double>(now - s.last_fresh_steady_ns) * 1e-9;
            if (age > stale_limit_s(s.info.nominal_hz))
                ImGui::TextColored({0.95f, 0.60f, 0.20f, 1.0f},
                                   "   last sample %.1f s ago (STALE)", age);
            else
                ImGui::TextDisabled("   last sample %.1f s ago", age);
        }

        // ---- cumulative gap counter: the visible-loss contract ----
        if (s.gap_total > 0) {
            ImGui::SameLine();
            ImGui::TextColored({0.95f, 0.35f, 0.35f, 1.0f}, "   lost %llu samples",
                               static_cast<unsigned long long>(s.gap_total));
        }

        // ---- one input-locked strip chart per channel ----
        const std::string plot_id = "##" + std::string(s.info.id);
        if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, 130), kLockedPlot)) {
            // Fixed y for bounded signals (%, thermal 0..3); autofit otherwise.
            const bool pct = std::strcmp(s.info.units, "%") == 0;
            const ImPlotAxisFlags yflags =
                (pct || thermal) ? ImPlotAxisFlags_Lock : ImPlotAxisFlags_AutoFit;
            ImPlot::SetupAxes("s", s.info.units[0] ? s.info.units : nullptr,
                              ImPlotAxisFlags_NoHighlight, yflags);
            if (!s.xs.empty()) {
                const double xmax = s.xs.back();
                ImPlot::SetupAxisLimits(ImAxis_X1, xmax - window, xmax,
                                        ImPlotCond_Always);
            }
            if (thermal)
                ImPlot::SetupAxisLimits(ImAxis_Y1, -0.2, 3.2, ImPlotCond_Always);
            else if (pct)
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Always);

            if (!s.xs.empty())
                ImPlot::PlotLine(s.info.id, s.xs.data(), s.ys.data(),
                                 static_cast<int>(s.xs.size()));
            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

void PulseScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        // Grace: the poller only sleeps in 25 ms chunks, so it returns promptly.
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->host) st->host->log_info("pulse-scope: on_cleanup");
}

}  // namespace pulsescope
