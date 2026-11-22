// The event ring (events.h, design/instruments.md, Events).
//
// Stubs of milestone M1's interface commit: the ring records nothing and
// events.wait answers command.not_implemented until the track "events" fills
// this file. The names of the kinds are complete.
#include <caliper_builder/session/events.h>

namespace caliper_builder {

const char* event_kind_name(EventKind kind) {
    switch (kind) {
        case EventKind::pick: return "pick";
        case EventKind::select: return "select";
        case EventKind::control: return "control";
        case EventKind::run: return "run";
        case EventKind::spec: return "spec";
        case EventKind::approval: return "approval";
    }
    return "pick";
}

std::optional<EventKind> event_kind_from_name(std::string_view name) {
    for (EventKind k : {EventKind::pick, EventKind::select, EventKind::control, EventKind::run, EventKind::spec,
                        EventKind::approval})
        if (name == event_kind_name(k)) return k;
    return std::nullopt;
}

nlohmann::json event_json(const Event& event) {
    (void)event;
    return nlohmann::json::object();   // stub: track events
}

struct EventRing::State {};

EventRing::EventRing(std::size_t capacity) : state_(std::make_unique<State>()) { (void)capacity; }
EventRing::~EventRing() = default;

std::uint64_t EventRing::push(Event event) {
    (void)event;
    return 0;   // stub: track events
}

EventRead EventRing::read(std::uint64_t cursor, const EventFilter& filter, std::size_t limit) const {
    (void)filter;
    (void)limit;
    EventRead out;
    out.cursor = cursor;
    return out;   // stub: track events
}

bool EventRing::wait(std::uint64_t cursor, const EventFilter& filter,
                     std::chrono::steady_clock::time_point deadline) const {
    (void)cursor;
    (void)filter;
    (void)deadline;
    return false;   // stub: track events
}

std::uint64_t EventRing::latest() const { return 0; }

void EventRing::close() {}

CommandOutput events_wait_command(const CommandContext& context, EventRing& ring) {
    (void)ring;
    return CommandOutput::fail(Error("command.not_implemented",
                                     context.invocation.name + " is part of milestone M1 and is not built yet.",
                                     context.invocation.name));
}

CommandResult wait_for_events(CommandQueue& queue, EventRing& ring, CommandInvocation invocation) {
    (void)ring;
    return queue.submit(std::move(invocation)).get();   // stub: track events
}

void note_journal_event(EventSink& sink, const JournalEvent& event) {
    (void)sink;
    (void)event;   // stub: track events
}

}  // namespace caliper_builder
