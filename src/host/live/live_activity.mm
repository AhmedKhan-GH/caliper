// live_activity.mm: the NSProcessInfo activity of live_activity.h (ARC).
#import <Foundation/Foundation.h>

#include "live/live_activity.h"

namespace caliper_live {

void* begin_viewer_activity(const char* reason) {
    NSString* why = [NSString stringWithUTF8String:reason ? reason : "caliper live session"];
    id<NSObject> activity = [[NSProcessInfo processInfo]
        beginActivityWithOptions:(NSActivityUserInitiatedAllowingIdleSystemSleep |
                                  NSActivityLatencyCritical)
                          reason:why];
    return (__bridge_retained void*)activity;   // +1 until end_viewer_activity
}

void end_viewer_activity(void* token) {
    if (!token) return;
    id<NSObject> activity = (__bridge_transfer id<NSObject>)token;
    [[NSProcessInfo processInfo] endActivity:activity];
}

}  // namespace caliper_live
