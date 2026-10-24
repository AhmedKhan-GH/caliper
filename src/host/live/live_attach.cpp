// live_attach.cpp: the session list and the hosted attachment of live_attach.h.
// Everything that touches the protocol goes through the live core (list_sessions,
// caliper_core_live_*), so the Caliper app and caliper-viewer read and claim
// sessions by exactly the same rules.
#include "live/live_attach.h"

#include "caliper/live.h"
#include "embed/embed_core.h"
#include "live/live_activity.h"
#include "live/live_shm.h"

#include <cstdio>
#include <ctime>
#include <utility>

namespace caliper_live {

namespace {
constexpr uint64_t kRelistNs = 1000ull * 1000 * 1000;   // the list refreshes once a second
constexpr uint64_t kStatsNs = 1000ull * 1000 * 1000;    // fps / latency over the last second
}  // namespace

// ---- row text ------------------------------------------------------------------

std::string state_text(const SessionEntry& e) {
    switch (session_state(e)) {
        case SessionState::Incompatible: return "protocol v" + std::to_string(e.protocol);
        case SessionState::Starting:     return "starting";
        case SessionState::Closing:      return "closing";
        case SessionState::Viewed:       return "viewed by " + std::to_string(e.viewer_pid);
        case SessionState::Running:      return "running";
    }
    return "running";
}

std::string clock_time(uint64_t unix_ms) {
    if (!unix_ms) return "-";
    const time_t s = (time_t)(unix_ms / 1000);
    struct tm tmv{};
    localtime_r(&s, &tmv);
    char b[32];
    std::strftime(b, sizeof b, "%H:%M:%S", &tmv);
    return b;
}

std::string age_text(uint64_t unix_ms) {
    if (!unix_ms) return "";
    const uint64_t now = caliper_live::unix_ms();
    const uint64_t s = now > unix_ms ? (now - unix_ms) / 1000 : 0;
    char b[32];
    if (s < 60) std::snprintf(b, sizeof b, "%llus ago", (unsigned long long)s);
    else if (s < 3600) std::snprintf(b, sizeof b, "%llum ago", (unsigned long long)(s / 60));
    else std::snprintf(b, sizeof b, "%lluh ago", (unsigned long long)(s / 3600));
    return b;
}

std::string view_names(const SessionEntry& e) {
    if (e.views.empty()) return "none yet";
    std::string out;
    for (const auto& v : e.views) {
        if (!out.empty()) out += ", ";
        out += v;
    }
    return out;
}

// ---- SessionList ---------------------------------------------------------------

SessionList::SessionList(std::string root) : root_(std::move(root)) {}

bool SessionList::refresh(bool force) {
    const uint64_t t = now_ns();
    if (!force && listed_ns_ && t - listed_ns_ < kRelistNs) return false;
    listed_ns_ = t;
    std::string err;
    if (!list_sessions(root(), &sessions_, &err)) error_ = err;
    else error_.clear();
    return true;
}

int SessionList::find(const std::string& session) const {
    for (size_t i = 0; i < sessions_.size(); ++i)
        if (sessions_[i].session == session) return (int)i;
    return -1;
}

int SessionList::find_pid(int32_t pid) const {
    for (size_t i = 0; i < sessions_.size(); ++i)
        if (sessions_[i].pid == pid) return (int)i;
    return -1;
}

std::string SessionList::root() const { return data_root(root_.c_str()); }

// ---- Attachment ----------------------------------------------------------------

bool Attachment::attach(caliper_host::HostRenderer* renderer, const std::string& session,
                        std::string* err) {
    if (core_) {
        *err = "already attached to '" + status_.session + "'; detach first";
        return false;
    }
    core_ = caliper_core_create_hosted(renderer);
    if (!core_) {
        *err = renderer ? "another Caliper core is live in this process (one per process)"
                        : "no renderer";
        return false;
    }
    if (!caliper_core_live_attach(core_, session.c_str())) {
        // "live_attach: <why>": the reason without the call's prefix.
        std::string why = caliper_core_last_error(core_);
        constexpr const char* kPrefix = "live_attach: ";
        if (why.rfind(kPrefix, 0) == 0) why.erase(0, std::char_traits<char>::length(kPrefix));
        *err = why;
        caliper_core_destroy_hosted(core_);
        core_ = nullptr;
        return false;
    }
    ended_ = false;
    window_ns_ = now_ns();
    activity_ = begin_viewer_activity("caliper: attached to a live session");
    refresh_status(/*reset_window=*/true);
    events_.push_back(Event::Attached);
    return true;
}

int Attachment::poll() {
    if (!core_) return 0;
    const int n = caliper_core_live_poll(core_);
    const uint64_t t = now_ns();
    const bool reset = t - window_ns_ >= kStatsNs;
    if (reset) window_ns_ = t;
    refresh_status(reset);
    if (!status_.alive && !ended_) {   // the script exited or died: once
        ended_ = true;
        events_.push_back(Event::Ended);
    }
    return n < 0 ? 0 : n;
}

void Attachment::draw(float x, float y, float w, float h, float scale) {
    if (core_) caliper_core_hosted_frame(core_, x, y, w, h, scale);
}

void Attachment::detach() {
    if (!core_) return;
    // destroy_hosted detaches first: the views' imports are released while the
    // bridge still has its renderer, and the source unmaps, clearing the
    // heartbeat and the claims, so the script stops taking snapshots.
    caliper_core_destroy_hosted(core_);
    core_ = nullptr;
    end_viewer_activity(activity_);
    activity_ = nullptr;
    status_ = Status{};
    ended_ = false;
    events_.push_back(Event::Detached);
}

std::vector<Attachment::Event> Attachment::take_events() {
    std::vector<Event> out;
    out.swap(events_);
    return out;
}

void Attachment::refresh_status(bool reset_window) {
    CaliperLiveStatus st{};
    st.struct_size = sizeof st;
    if (!caliper_core_live_status(core_, &st, reset_window ? 1 : 0) || !st.attached) {
        status_ = Status{};
        return;
    }
    status_.attached = true;
    status_.pending = st.pending != 0;
    status_.alive = st.client_alive != 0;
    status_.session = st.session ? st.session : "";
    status_.script = st.script ? st.script : "";
    status_.pid = st.client_pid;
    status_.start_unix_ms = st.start_unix_ms;
    status_.views = st.views;
    status_.updates = st.updates;
    // The window statistics are read just before each reset, so the shown
    // values always cover the last full second.
    if (reset_window) {
        status_.fps = st.fps;
        status_.latency_ms = st.latency_ms_median;
    }
}

}  // namespace caliper_live
