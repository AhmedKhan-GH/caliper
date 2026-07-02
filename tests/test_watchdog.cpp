#include <doctest/doctest.h>
#include "frame_watchdog.h"
using caliper_host::FrameWatchdog;

TEST_CASE("watchdog: three consecutive overruns latch the flag") {
    FrameWatchdog w;                       // 250 ms budget, threshold 3
    w.feed(300); w.feed(300);
    CHECK_FALSE(w.flagged());
    w.feed(300);
    CHECK(w.flagged());
}

TEST_CASE("watchdog: a good frame resets the streak") {
    FrameWatchdog w;
    w.feed(300); w.feed(300); w.feed(10); w.feed(300); w.feed(300);
    CHECK_FALSE(w.flagged());
}

TEST_CASE("watchdog: flag latches until reset()") {
    FrameWatchdog w;
    w.feed(300); w.feed(300); w.feed(300);
    w.feed(1);                              // fast frame does NOT clear it
    CHECK(w.flagged());
    w.reset();
    CHECK_FALSE(w.flagged());
}
