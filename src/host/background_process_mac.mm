// Background processes on macOS (background_process.h).
#include "background_process.h"

#import <AppKit/AppKit.h>

namespace caliper_host {

void use_background_activation_policy() {
    @autoreleasepool {
        // An executable outside an app bundle starts with this policy already;
        // setting it says so explicitly and undoes a library that changed it.
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];
    }
}

bool background_activation_policy_active() {
    @autoreleasepool {
        return NSApp != nil &&
               [NSApp activationPolicy] == NSApplicationActivationPolicyProhibited;
    }
}

} // namespace caliper_host
