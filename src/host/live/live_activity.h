// live_activity.h: keep an attached viewer's frames coming while its window is
// hidden or occluded (host-internal; caliper-viewer and the Caliper app).
//
// App Nap throttles a hidden window's timer; the viewer heartbeat then goes
// stale and the training process stops snapshotting while the user is still
// attached. An NSProcessInfo activity with UserInitiatedAllowingIdleSystemSleep
// alone was not enough (an occluded viewer measured 5 to 11 fps for 10 s while
// the trainer ran); LatencyCritical keeps a 60 Hz tick. Hold one for as long as
// a session is attached. Plain C++ interface over live_activity.mm.
#pragma once

namespace caliper_live {

// Begins the activity; returns its token (null where there is no App Nap).
void* begin_viewer_activity(const char* reason);
// Ends an activity begun above. Safe on null.
void end_viewer_activity(void* token);

}  // namespace caliper_live
