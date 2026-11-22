// Tests of the event ring, events.wait, wait_for_events() and the session's
// feeds of spec and approval events (events.h, design/instruments.md,
// Events). Waits end on futures, the ring or the published state, never on a
// fixed sleep; the frame thread of a test drains the queue between frames.
#include "helpers.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/program_host.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <regex>
#include <thread>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

using Clock = std::chrono::steady_clock;

Event make(EventKind kind, std::string panel = {}, std::string view = {}) {
    Event e;
    e.kind = kind;
    e.panel = std::move(panel);
    e.view = std::move(view);
    return e;
}

std::vector<std::uint64_t> seqs(const EventRead& r) {
    std::vector<std::uint64_t> out;
    for (const Event& e : r.events) out.push_back(e.seq);
    return out;
}

// The events of one kind in the session's ring, in the command table's form.
std::vector<json> ring_events(Session& s, EventKind kind) {
    std::vector<json> out;
    for (const Event& e : s.events().read(0, EventFilter{{kind}, {}}, EventRing::kCapacity).events)
        out.push_back(event_json(e));
    return out;
}

// A frame thread: drains the session's queue between short frames and
// counts its frames, until it stops.
struct FrameThread {
    Session& s;
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> frames{0};
    std::thread t;
    explicit FrameThread(Session& session) : s(session) {
        t = std::thread([this] {
            while (!stop) {
                s.queue().drain();
                ++frames;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));   // one frame
            }
        });
    }
    ~FrameThread() {
        stop = true;
        t.join();
    }
    // Waits until `n` more frames have run; false after 10 s.
    bool more_frames(std::uint64_t n) const {
        const std::uint64_t until = frames + n;
        const auto deadline = Clock::now() + std::chrono::seconds(10);
        while (frames < until)
            if (Clock::now() > deadline) return false;
            else std::this_thread::yield();
        return true;
    }
};

CommandInvocation wait_call(json args) {
    CommandInvocation inv;
    inv.name = "events.wait";
    inv.args = std::move(args);
    inv.origin = CommandOrigin::api;
    return inv;
}

}  // namespace

TEST_CASE("the event kinds have stable names and the ring keeps 1024 events") {
    for (EventKind k : {EventKind::pick, EventKind::select, EventKind::control, EventKind::run, EventKind::spec,
                        EventKind::approval, EventKind::program, EventKind::state, EventKind::cursor})
        CHECK(event_kind_from_name(event_kind_name(k)) == k);
    CHECK_FALSE(event_kind_from_name("hover"));
    CHECK(EventRing::kCapacity == 1024);
}

TEST_CASE("events: the ring numbers events from 1, stamps their time and gives the six common keys") {
    EventRing ring;
    CHECK(ring.latest() == 0);
    Event e = make(EventKind::pick, "", "scene");
    e.seq = 99;   // the ring sets seq and time
    e.fields = {{"entity", "modules"}, {"index", 3}, {"seq", "not this"}};
    CHECK(ring.push(e) == 1);
    CHECK(ring.push(make(EventKind::control, "lr")) == 2);
    CHECK(ring.latest() == 2);

    const EventRead r = ring.read(0, {}, 64);
    REQUIRE(r.events.size() == 2);
    const json j = event_json(r.events[0]);
    CHECK(j["seq"] == 1);
    CHECK(std::regex_match(j["time"].get<std::string>(),
                           std::regex(R"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z)")));
    CHECK(j["kind"] == "pick");
    CHECK(j["panel"].is_null());
    CHECK(j["view"] == "scene");
    CHECK(j["origin"] == "ui");
    CHECK(j["entity"] == "modules");
    CHECK(j["index"] == 3);
    const json k = event_json(r.events[1]);
    CHECK(k["panel"] == "lr");
    CHECK(k["view"].is_null());

    // Only the keys of the kind leave the ring, and a cleared selection has
    // an empty list of entities.
    Event sel = make(EventKind::select, "", "scene");
    sel.fields = {{"space", "tokens"}, {"items", {1, 2}}};
    ring.push(sel);
    const json cleared = event_json(ring.read(2, {}, 1).events.at(0));
    CHECK(cleared["entities"] == json::array());
    CHECK_FALSE(cleared.contains("space"));
    CHECK(j.size() == 8);   // the six common keys, entity and index

    ring.close();
    CHECK(ring.push(make(EventKind::pick)) == 0);   // later pushes are dropped
    CHECK(ring.latest() == 3);
}

TEST_CASE("events: the ring keeps the newest events and reports the events it dropped after a cursor") {
    EventRing ring;
    for (int i = 0; i < 1030; ++i) ring.push(make(i % 2 ? EventKind::pick : EventKind::run));
    EventRead r = ring.read(0, {}, 2000);
    CHECK(r.events.size() == 1024);
    CHECK(r.events.front().seq == 7);
    CHECK(r.lost == 6);
    CHECK(r.cursor == 1030);
    CHECK(ring.read(3, {}, 1).lost == 3);
    CHECK(ring.read(6, {}, 1).lost == 0);
    CHECK(ring.read(1000, {}, 64).lost == 0);
    // Lost counts dropped events of every kind, whatever the filter.
    CHECK(ring.read(0, EventFilter{{EventKind::control}, {}}, 64).lost == 6);

    EventRing small(4);
    for (int i = 0; i < 6; ++i) small.push(make(EventKind::pick));
    r = small.read(1, {}, 64);
    CHECK(seqs(r) == std::vector<std::uint64_t>{3, 4, 5, 6});
    CHECK(r.lost == 1);
}

TEST_CASE("events: a read after a cursor filters by kind and panel and moves the cursor past what it looked at") {
    EventRing ring;
    ring.push(make(EventKind::pick, "world", "model"));   // 1
    ring.push(make(EventKind::control, "lr"));            // 2
    ring.push(make(EventKind::pick, "", "model"));        // 3: a live view outside a spec
    ring.push(make(EventKind::spec));                     // 4
    ring.push(make(EventKind::pick, "other", "model"));   // 5

    EventRead r = ring.read(0, EventFilter{{EventKind::pick}, {}}, 64);
    CHECK(seqs(r) == std::vector<std::uint64_t>{1, 3, 5});
    CHECK(r.cursor == 5);
    CHECK_FALSE(r.more);

    // A filter of panels never matches an event without a panel.
    r = ring.read(0, EventFilter{{}, {"world", "lr"}}, 64);
    CHECK(seqs(r) == std::vector<std::uint64_t>{1, 2});
    CHECK(r.cursor == 5);   // past the events that did not match too

    r = ring.read(1, EventFilter{{EventKind::pick}, {"world"}}, 64);
    CHECK(r.events.empty());
    CHECK(r.cursor == 5);

    // The limit: more says that matching events wait, and the cursor stops
    // before the first one not returned.
    r = ring.read(0, EventFilter{{EventKind::pick}, {}}, 2);
    CHECK(seqs(r) == std::vector<std::uint64_t>{1, 3});
    CHECK(r.more);
    CHECK(r.cursor == 4);
    r = ring.read(r.cursor, EventFilter{{EventKind::pick}, {}}, 2);
    CHECK(seqs(r) == std::vector<std::uint64_t>{5});
    CHECK_FALSE(r.more);

    // Nothing after the cursor; a cursor beyond the newest event (from a ring
    // before this one) reads from the oldest event kept.
    r = ring.read(5, {}, 64);
    CHECK(r.events.empty());
    CHECK(r.cursor == 5);
    r = ring.read(40, {}, 64);
    CHECK(r.events.size() == 5);
    CHECK(r.cursor == 5);
    CHECK(EventRing().read(0, {}, 64).cursor == 0);
}

TEST_CASE("events: a waiter wakes for a matching event, not for another, and close() wakes it") {
    EventRing ring;
    ring.push(make(EventKind::run));
    const EventFilter picks{{EventKind::pick}, {}};
    CHECK(ring.wait(0, EventFilter{{EventKind::run}, {}}, Clock::now()));   // one waits already
    CHECK_FALSE(ring.wait(0, picks, Clock::now()));
    CHECK_FALSE(ring.wait(0, picks, Clock::now() + std::chrono::milliseconds(20)));

    std::promise<void> started;
    auto waiter = std::async(std::launch::async, [&] {
        started.set_value();
        return ring.wait(1, picks, Clock::now() + std::chrono::seconds(30));
    });
    started.get_future().wait();
    ring.push(make(EventKind::control, "lr"));   // no match: the waiter goes on waiting
    CHECK(waiter.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
    ring.push(make(EventKind::pick, "", "model"));
    REQUIRE(waiter.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    CHECK(waiter.get());

    auto closed = std::async(std::launch::async,
                             [&] { return ring.wait(ring.latest(), picks, Clock::now() + std::chrono::seconds(30)); });
    ring.close();
    REQUIRE(closed.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    CHECK_FALSE(closed.get());
}

TEST_CASE("events: events.wait answers with the events after the cursor in the command table's form") {
    auto s = open_session(fresh_dir("events-wait") + "/workspace");
    Event pick = make(EventKind::pick, "world", "model");
    pick.fields = {{"entity", "modules"}, {"primitive", "points"}, {"index", 2}, {"row_id", 2},
                   {"position", {0.5, 1.0, -2.0}}};
    s->events().push(pick);
    s->events().push(make(EventKind::control, "lr"));

    json r = ok(*s, "events.wait", {{"cursor", 0}, {"timeout_s", 0}});
    REQUIRE(r["events"].size() == 2);
    CHECK(r["events"][0]["entity"] == "modules");
    CHECK(r["events"][0]["panel"] == "world");
    CHECK(r["cursor"] == 2);
    CHECK(r["lost"] == 0);
    CHECK(r["more"] == false);

    r = ok(*s, "events.wait", {{"cursor", 0}, {"kinds", {"control"}}, {"panels", {"lr"}}, {"timeout_s", 0}});
    REQUIRE(r["events"].size() == 1);
    CHECK(r["events"][0]["kind"] == "control");

    r = ok(*s, "events.wait", {{"limit", 1}, {"timeout_s", 0}});
    CHECK(r["events"].size() == 1);
    CHECK(r["more"] == true);
    CHECK(r["cursor"] == 1);

    // The handler never waits, whatever timeout_s says.
    r = ok(*s, "events.wait", {{"cursor", 2}, {"timeout_s", 60}});
    CHECK(r["events"].empty());
    CHECK(r["cursor"] == 2);
    CHECK(err(*s, "events.wait", {{"timeout_s", 61}}) == "command.invalid_args");
}

TEST_CASE("events: wait_for_events holds its caller, not the frame thread, until a matching event comes") {
    auto s = open_session(fresh_dir("events-long-poll") + "/workspace");
    FrameThread frame(*s);
    s->events().push(make(EventKind::pick, "", "model"));   // 1

    // A matching event waits: the answer comes at once.
    CommandResult r = wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 0}, {"timeout_s", 30}}));
    REQUIRE(r.ok);
    CHECK(r.result["events"].size() == 1);

    // None waits: the call is held while the frame thread runs, and a match ends it.
    auto held = std::async(std::launch::async, [&] {
        return wait_for_events(s->queue(), s->events(),
                               wait_call({{"cursor", 1}, {"kinds", {"pick"}}, {"timeout_s", 30}}));
    });
    REQUIRE(frame.more_frames(20));
    CHECK(held.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);
    s->events().push(make(EventKind::control, "lr"));         // 2: no match
    REQUIRE(frame.more_frames(20));
    CHECK(held.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);
    s->events().push(make(EventKind::pick, "world", "model"));   // 3
    REQUIRE(held.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    r = held.get();
    REQUIRE(r.ok);
    REQUIRE(r.result["events"].size() == 1);
    CHECK(r.result["events"][0]["seq"] == 3);
    CHECK(r.result["cursor"] == 3);

    // timeout_s ends a wait without an event with an empty answer.
    r = wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 3}, {"timeout_s", 0.05}}));
    REQUIRE(r.ok);
    CHECK(r.result["events"].empty());
    CHECK(r.result["cursor"] == 3);
}

TEST_CASE("events: a hold taken before a thread starts keeps the session's close waiting until it goes") {
    auto s = open_session(fresh_dir("events-hold") + "/workspace");
    auto hold = std::make_unique<EventWaitHold>(s->events());
    auto closed = std::async(std::launch::async, [&] { s->close(); });
    CHECK(closed.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
    hold.reset();
    CHECK(closed.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
}

TEST_CASE("events: the ring counts the callers it holds, by the kinds their filters take") {
    EventRing ring;
    CHECK(ring.waiting() == 0);
    EventFilter picks;
    picks.kinds = {EventKind::pick};
    auto held = std::async(std::launch::async, [&] {
        return ring.wait(0, picks, std::chrono::steady_clock::now() + std::chrono::seconds(30));
    });
    while (ring.waiting() == 0) std::this_thread::yield();   // the waiter is held
    CHECK(ring.waiting(EventKind::pick) == 1);
    CHECK(ring.waiting(EventKind::run) == 0);
    Event e;
    e.kind = EventKind::pick;
    ring.push(e);
    CHECK(held.get());
    CHECK(ring.waiting() == 0);
}

TEST_CASE("events: closing the session wakes a held wait_for_events and waits for it to leave") {
    auto s = open_session(fresh_dir("events-close") + "/workspace");
    std::optional<CommandResult> got;
    {
        FrameThread frame(*s);
        auto held = std::async(std::launch::async, [&] {
            return wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 0}, {"timeout_s", 60}}));
        });
        while (s->events().waiting() == 0) std::this_thread::yield();   // the wait is held
        s->close();
        REQUIRE(held.wait_for(std::chrono::seconds(0)) == std::future_status::ready);   // close() waited for it
        got = held.get();
    }
    CHECK_FALSE(got->ok);
    CHECK(got->error.code == "command.shutdown");
}

TEST_CASE("events: wait_for_events holds at most kMaxEventWaits calls and answers one more at once") {
    auto s = open_session(fresh_dir("events-bound") + "/workspace");
    std::vector<std::future<CommandResult>> held;
    std::optional<CommandResult> over, now;
    {
        FrameThread frame(*s);
        for (std::size_t i = 0; i < kMaxEventWaits; ++i)
            held.push_back(std::async(std::launch::async, [&] {
                return wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 0}, {"timeout_s", 60}}));
            }));
        while (s->events().waiting() < kMaxEventWaits) std::this_thread::yield();
        over = wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 0}, {"timeout_s", 60}}));
        now = wait_for_events(s->queue(), s->events(), wait_call({{"cursor", 0}, {"timeout_s", 0}}));
        CHECK(s->events().waiting() == kMaxEventWaits);
        Event e;
        e.kind = EventKind::pick;
        s->events().push(e);   // the held calls answer with it
        for (auto& f : held) CHECK(f.get().ok);
    }
    REQUIRE_FALSE(over->ok);
    CHECK(over->error.code == kEventsTooManyWaits);
    CHECK(over->error.where == "timeout_s");
    CHECK(now->ok);
    CHECK(s->events().waiting() == 0);
}

TEST_CASE("events: the session feeds spec events from its journal") {
    auto s = open_session(fresh_dir("events-spec") + "/workspace");
    write_text(s->workspace() + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
    ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*s, "spec.new", {{"id", "dash"}});
    ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}},
       CommandOrigin::api);
    ok(*s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Loss"}}}, {"dry_run", true}});   // no change
    ok(*s, "history.undo");
    CHECK(err(*s, "panel.remove", {{"id", "none"}}) != "");   // a failed edit changes nothing

    const std::vector<json> spec = ring_events(*s, EventKind::spec);
    REQUIRE(spec.size() == 4);
    CHECK(spec[0]["command"] == "spec.new");
    CHECK(spec[0]["event"] == "do");
    CHECK(spec[0]["panel"].is_null());
    for (const json& e : spec) CHECK(e["applet"] == "dash");   // the open spec's applet id
    CHECK(spec[1]["command"] == "source.add");
    CHECK(spec[2]["command"] == "panel.create");
    CHECK(spec[2]["panel"] == "loss");
    CHECK(spec[2]["origin"] == "api");
    CHECK(spec[3]["command"] == "history.undo");
    CHECK(spec[3]["event"] == "undo");
    CHECK(spec[3]["origin"] == "ui");
}

TEST_CASE("events: the session feeds approval events when a card asks and when the person decides") {
    auto s = open_session(fresh_dir("events-approval") + "/workspace");
    write_text(s->workspace() + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
    ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*s, "spec.new", {{"id", "dash"}});
    ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});

    // A run command from caliper-mcp asks; approving it runs it (the session
    // has no run control here, so it then answers command.not_available).
    CommandResult r = invoke(*s, "run.set_param", {{"name", "lr"}, {"value", 0.01}}, CommandOrigin::mcp);
    REQUIRE(r.error.code == "command.needs_approval");
    const std::string first = r.approval;
    std::vector<json> a = ring_events(*s, EventKind::approval);
    REQUIRE(a.size() == 1);
    CHECK(a[0]["approval"] == first);
    CHECK(a[0]["command"] == "run.set_param");
    CHECK(a[0]["target"] == "lr");
    CHECK(a[0]["state"] == "pending");
    CHECK(a[0]["origin"] == "mcp");
    s->registry().approve(first);
    a = ring_events(*s, EventKind::approval);
    REQUIRE(a.size() == 2);
    CHECK(a[1]["approval"] == first);
    CHECK(a[1]["state"] == "approved");
    CHECK(a[1]["origin"] == "ui");

    // A spec edit under review: a denial and a withdrawal.
    s->set_review(CommandOrigin::mcp, true);
    const json panel = {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}};
    r = invoke(*s, "panel.create", panel, CommandOrigin::mcp);
    REQUIRE(r.error.code == "command.needs_approval");
    const std::string second = r.approval;
    CHECK(second != first);
    s->registry().deny(second);
    r = invoke(*s, "panel.create", panel, CommandOrigin::mcp);
    REQUIRE(r.error.code == "command.needs_approval");
    const std::string third = r.approval;
    s->registry().expire_approval(third);

    a = ring_events(*s, EventKind::approval);
    REQUIRE(a.size() == 6);
    CHECK(a[2]["approval"] == second);
    CHECK(a[2]["state"] == "pending");
    CHECK_FALSE(a[2].contains("target"));
    CHECK(a[3]["approval"] == second);
    CHECK(a[3]["state"] == "denied");
    CHECK(a[3]["command"] == "panel.create");
    CHECK(a[4]["approval"] == third);
    CHECK(a[5]["approval"] == third);
    CHECK(a[5]["state"] == "expired");
    CHECK(a[5]["origin"] == "mcp");
    // A run command that the person denies gives its approval event too,
    // although the journal does not record it.
    r = invoke(*s, "run.action", {{"name", "save"}}, CommandOrigin::mcp);
    REQUIRE(r.error.code == "command.needs_approval");
    s->registry().deny(r.approval);
    a = ring_events(*s, EventKind::approval);
    REQUIRE(a.size() == 8);
    CHECK(a[7]["approval"] == r.approval);
    CHECK(a[7]["command"] == "run.action");
    CHECK(a[7]["target"] == "save");
    CHECK(a[7]["state"] == "denied");
    CHECK(a[7]["origin"] == "ui");
    // The approval events are readable through events.wait.
    const json w = ok(*s, "events.wait", {{"kinds", {"approval"}}, {"timeout_s", 0}});
    CHECK(w["events"].size() == 8);
}

TEST_CASE("events: a program event that a host pushes reaches events.wait in the command table's form") {
    auto s = open_session(fresh_dir("events-program") + "/workspace");
    ProgramStatus stopped;
    stopped.id = "rows";
    stopped.kind = "geometry";
    stopped.host = "script";
    stopped.state = "stopped";
    stopped.panel = "world";
    stopped.error = ProgramError{"program.over_time", "The program went over 4 ms on 3 passes in a row.", 0, 0};
    push_program_event(s->events(), stopped, "stopped");
    ProgramStatus compiled;
    compiled.id = "world/programs/rows/material";
    compiled.kind = "material";
    compiled.host = "viewer";
    compiled.panel = "world";
    push_program_event(s->events(), compiled, "compiled");
    s->events().push(make(EventKind::control, "lr"));

    // The registry checks every result against the table's schema.
    json r = ok(*s, "events.wait", {{"cursor", 0}, {"kinds", {"program"}}, {"timeout_s", 0}});
    REQUIRE(r["events"].size() == 2);
    const json& e = r["events"][0];
    CHECK(e["kind"] == "program");
    CHECK(e["program"] == "rows");
    CHECK(e["change"] == "stopped");
    CHECK(e["host"] == "script");
    CHECK(e["code"] == "program.over_time");
    CHECK(e["panel"] == "world");
    CHECK(r["events"][1]["change"] == "compiled");
    CHECK(r["events"][1]["host"] == "viewer");
    CHECK_FALSE(r["events"][1].contains("code"));
    r = ok(*s, "events.wait", {{"cursor", 0}, {"kinds", {"program"}}, {"panels", {"other"}}, {"timeout_s", 0}});
    CHECK(r["events"].empty());
}
