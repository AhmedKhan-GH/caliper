// Tests of the event ring and events.wait (events.h, design/instruments.md,
// Events). The M1 track "events" owns this file; until it fills
// event_ring.cpp, events.wait is bound and answers command.not_implemented.
#include "helpers.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

TEST_CASE("the event kinds have stable names and the ring keeps 1024 events") {
    for (EventKind k : {EventKind::pick, EventKind::select, EventKind::control, EventKind::run, EventKind::spec,
                        EventKind::approval})
        CHECK(event_kind_from_name(event_kind_name(k)) == k);
    CHECK_FALSE(event_kind_from_name("hover"));
    CHECK(EventRing::kCapacity == 1024);
}

TEST_CASE("events.wait is bound and not built yet") {
    auto s = open_session(fresh_dir("events") + "/workspace");
    CHECK(err(*s, "events.wait", {{"cursor", 0}, {"timeout_s", 0}}) == "command.not_implemented");
}
