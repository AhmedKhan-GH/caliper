#pragma once

namespace caliper_host {

// Makes the platform threading rule observable (PLATFORM.md §15): frame()
// exceeding budget repeatedly flags the applet — "long work belongs in
// caliper.jobs". Latches until reset (applet relaunch).
class FrameWatchdog {
public:
    explicit FrameWatchdog(double budget_ms = 250.0, int threshold = 3)
        : budget_ms_(budget_ms), threshold_(threshold) {}

    void feed(double frame_ms) {
        if (flagged_) return;
        if (frame_ms > budget_ms_) {
            if (++over_ >= threshold_) flagged_ = true;
        } else {
            over_ = 0;
        }
    }
    bool flagged() const { return flagged_; }
    void reset() { over_ = 0; flagged_ = false; }

private:
    double budget_ms_;
    int threshold_;
    int over_ = 0;
    bool flagged_ = false;
};

} // namespace caliper_host
