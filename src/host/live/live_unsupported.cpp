// live_unsupported.cpp: the live attach on platforms without a port. The
// shared-memory client and session code are macOS-only so far (live_shm.h,
// live_client.mm, live_session.mm); this file stands in for live_core.cpp and
// live_attach.cpp so the host still builds and runs elsewhere. There are never
// any sessions to attach to, and every attach fails with a reason. The typed
// views (live_views.cpp) are portable and stay.
#include "live/live_attach.h"

#include "caliper/live.h"
#include "embed/embed_core.h"

#include <cstring>

namespace {
constexpr const char* kUnsupported = "live attach is not available on this platform yet";
}  // namespace

void caliper_live_frame(CaliperCore*) {}
void caliper_live_shutdown(CaliperCore*) {}

extern "C" {

int caliper_core_live_attach(CaliperCore* core, const char*) {
    if (core) core->fail(kUnsupported);
    return 0;
}
int caliper_core_live_attach_pid(CaliperCore* core, int32_t, const char*) {
    if (core) core->fail(kUnsupported);
    return 0;
}
void caliper_core_live_detach(CaliperCore*) {}
int caliper_core_live_poll(CaliperCore*) { return 0; }
int caliper_core_live_picker(CaliperCore* core, int enable, const char*) {
    if (core && enable) core->fail(kUnsupported);
    return 0;
}
int caliper_core_live_status(CaliperCore*, CaliperLiveStatus* out, int) {
    if (out) std::memset(out, 0, sizeof(*out));
    return 0;
}
int caliper_core_live_view_status(CaliperCore*, int, CaliperLiveViewStatus* out) {
    if (out) std::memset(out, 0, sizeof(*out));
    return 0;
}
size_t caliper_core_live_window_latencies(CaliperCore*, uint64_t*, double*, size_t) { return 0; }

}  // extern "C"

namespace caliper_live {

std::string state_text(const SessionEntry&) { return {}; }
std::string clock_time(uint64_t) { return {}; }
std::string age_text(uint64_t) { return {}; }
std::string view_names(const SessionEntry&) { return {}; }

SessionList::SessionList(std::string root) : root_(std::move(root)) {}
bool SessionList::refresh(bool) { return false; }
int SessionList::find(const std::string&) const { return -1; }
int SessionList::find_pid(int32_t) const { return -1; }
std::string SessionList::root() const { return root_; }

bool Attachment::attach(caliper_host::HostRenderer*, const std::string&, std::string* err) {
    if (err) *err = kUnsupported;
    return false;
}
int Attachment::poll() { return 0; }
void Attachment::draw(float, float, float, float, float) {}
void Attachment::detach() {}
std::vector<Attachment::Event> Attachment::take_events() { return {}; }
void Attachment::refresh_status(bool) {}

}  // namespace caliper_live
