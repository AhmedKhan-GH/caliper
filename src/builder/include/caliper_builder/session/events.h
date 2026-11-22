#pragma once
// The event ring (design/instruments.md, Events): the last 1,024 things that
// happened in the app, which the command events.wait gives to an agent.
//
// An event has a number `seq` that grows by one with each push, starting at
// 1, the time of its push, a kind and the kind's own fields. The kinds:
//   pick      a click on an item of a scene, in a scene panel or a live view
//   select    a selection made or cleared in the same places
//   control   the person finished an edit of a control of the spec
//   run       the attached script changed: its state, a parameter it applied,
//             the end of an action, its attachment
//   spec      a command changed the open spec (from the journal)
//   approval  a card asked for the person's decision, or was decided
// Hovers never enter the ring.
//
// A cursor is the seq of the last event a reader looked at; 0 reads from the
// oldest event kept. A read returns the events after the cursor that match a
// filter of kinds and panels, at most `limit`, and a new cursor past every
// event it looked at. When the ring dropped events after the cursor before
// the read, `lost` counts them, whatever their kind.
//
// Threads: push(), read(), wait() and close() are safe from any thread. The
// handler of events.wait never waits, because the command queue runs it on
// the frame thread; wait_for_events() holds a caller's own thread instead (the
// thread of a Caliper API request), then reads through the queue again.
// Nothing here throws.
//
// The ring, the handler and the feeds are in src/builder/session/event_ring.cpp.
// The spec runtime and its controls push picks, selections, control changes and
// run changes through EventSink.
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/commands/registry.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

enum class EventKind { pick, select, control, run, spec, approval };

// "pick", "select", "control", "run", "spec", "approval".
const char* event_kind_name(EventKind kind);
std::optional<EventKind> event_kind_from_name(std::string_view name);

// One event. seq and time are set by the ring on push.
struct Event {
    std::uint64_t seq = 0;
    std::string time;                 // UTC ISO 8601 with milliseconds
    EventKind kind = EventKind::pick;
    std::string panel;                // the spec panel or control; empty: none
    std::string view;                 // pick and select: the live view; empty: none
    std::string origin = "ui";        // "ui", "builder", "agent", "mcp", "api" or "script"
    nlohmann::json fields = nlohmann::json::object();   // the kind's own fields (design/instruments.md)
};

// The event in the command table's form: seq, time, kind, panel, view and
// origin (null for an empty panel or view), then the kind's fields.
nlohmann::json event_json(const Event& event);

// Where producers push events: the session's ring, or a test's recorder.
class EventSink {
public:
    virtual ~EventSink() = default;
    // Records the event and returns its seq; 0 when the sink is closed.
    virtual std::uint64_t push(Event event) = 0;
};

// Which events a reader wants. An empty list means every one; a panel filter
// never matches an event without a panel.
struct EventFilter {
    std::vector<EventKind> kinds;
    std::vector<std::string> panels;
};

struct EventRead {
    std::vector<Event> events;        // oldest first
    std::uint64_t cursor = 0;         // past every event looked at
    std::uint64_t lost = 0;           // dropped after the given cursor before this read
    bool more = false;                // more matching events wait beyond the limit
};

class EventRing : public EventSink {
public:
    static constexpr std::size_t kCapacity = 1024;

    explicit EventRing(std::size_t capacity = kCapacity);
    ~EventRing() override;
    EventRing(const EventRing&) = delete;
    EventRing& operator=(const EventRing&) = delete;

    std::uint64_t push(Event event) override;
    // The events after `cursor` that match, at most `limit`.
    EventRead read(std::uint64_t cursor, const EventFilter& filter, std::size_t limit) const;
    // Blocks until an event after `cursor` matches, the deadline passes or
    // close() is called; true when one matches.
    bool wait(std::uint64_t cursor, const EventFilter& filter, std::chrono::steady_clock::time_point deadline) const;
    // The seq of the newest event; 0 before the first.
    std::uint64_t latest() const;
    // The callers that wait() holds now, such as a long poll of events.wait;
    // with a kind, only those whose filter takes that kind.
    std::size_t waiting(std::optional<EventKind> kind = std::nullopt) const;
    // Wakes every waiter; later pushes are dropped.
    void close();

private:
    struct State;
    std::unique_ptr<State> state_;
};

// The handler of events.wait: reads the ring at once, never waits.
CommandOutput events_wait_command(const CommandContext& context, EventRing& ring);

// events.wait for a caller that may block its own thread: submits the
// invocation to the queue; when it answers with no event and timeout_s is
// above 0, waits on the ring until a matching event or the end of timeout_s,
// then submits it again with timeout_s 0. Caliper API serves events.wait
// through it.
CommandResult wait_for_events(CommandQueue& queue, EventRing& ring, CommandInvocation invocation);

// Holds EventRing::close(), and thus Session::close(), until it goes, as a
// caller inside wait_for_events() does. A caller that hands events.wait to a
// thread of its own takes one before the thread starts and lets it go after
// wait_for_events() returns, so that the session cannot close between the
// two. Take it only while the ring is open.
class EventWaitHold {
public:
    explicit EventWaitHold(const EventRing& ring);
    ~EventWaitHold();
    EventWaitHold(const EventWaitHold&) = delete;
    EventWaitHold& operator=(const EventWaitHold&) = delete;

private:
    const EventRing* ring_;
};

// The feed from the journal: a do, undo or redo of a command that edits the
// open spec becomes a spec event, and a decided approval an approval event.
// The session calls it from the journal's on_append, on the appending thread,
// with the applet id of the spec open after the command ("" for none), which a
// spec event carries as `applet`.
void note_journal_event(EventSink& sink, const JournalEvent& event, const std::string& applet = {});

}  // namespace caliper_builder
