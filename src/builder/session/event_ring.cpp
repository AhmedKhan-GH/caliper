// The event ring (events.h, design/instruments.md, Events).
//
// The ring keeps the newest events in a deque under one mutex, and a
// condition variable wakes the readers that wait in wait(). A cursor beyond
// the newest event, which a reader can hold from a ring that came before this
// one (the app started again), reads from the oldest event kept, because
// every event of this ring is new to that reader.
//
// wait_for_events() counts its callers by ring, and close() returns only
// after every caller of its ring has left, so that no caller touches the
// queue or the ring after the session that owns them goes away. The owner
// shuts its queue down before it closes the ring (Session::close()), thus a
// caller that close() wakes gets its last answer from the queue at once.
#include <caliper_builder/session/events.h>

#include <caliper_builder/session/spec_edit.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <utility>

namespace caliper_builder {

namespace {

using json = nlohmann::json;

constexpr double kDefaultTimeoutS = 10.0;
constexpr double kMaxTimeoutS = 60.0;
constexpr std::size_t kDefaultLimit = 64;
constexpr std::size_t kMaxLimit = 256;

bool matches(const Event& e, const EventFilter& f) {
    if (!f.kinds.empty() && std::find(f.kinds.begin(), f.kinds.end(), e.kind) == f.kinds.end()) return false;
    if (!f.panels.empty() &&
        (e.panel.empty() || std::find(f.panels.begin(), f.panels.end(), e.panel) == f.panels.end()))
        return false;
    return true;
}

// The arguments of events.wait, which the registry checked against the schema.
struct WaitArgs {
    std::uint64_t cursor = 0;
    EventFilter filter;
    double timeout_s = kDefaultTimeoutS;
    std::size_t limit = kDefaultLimit;
};

WaitArgs wait_args(const json& args) {
    WaitArgs w;
    if (!args.is_object()) return w;
    const auto cursor = args.find("cursor");
    if (cursor != args.end() && cursor->is_number_unsigned())
        w.cursor = cursor->get<std::uint64_t>();
    else if (cursor != args.end() && cursor->is_number_integer() && cursor->get<std::int64_t>() > 0)
        w.cursor = static_cast<std::uint64_t>(cursor->get<std::int64_t>());
    const auto kinds = args.find("kinds");
    if (kinds != args.end() && kinds->is_array())
        for (const json& k : *kinds)
            if (k.is_string())
                if (auto kind = event_kind_from_name(k.get<std::string>())) w.filter.kinds.push_back(*kind);
    const auto panels = args.find("panels");
    if (panels != args.end() && panels->is_array())
        for (const json& p : *panels)
            if (p.is_string()) w.filter.panels.push_back(p.get<std::string>());
    const auto timeout = args.find("timeout_s");
    if (timeout != args.end() && timeout->is_number())
        w.timeout_s = std::clamp(timeout->get<double>(), 0.0, kMaxTimeoutS);
    const auto limit = args.find("limit");
    if (limit != args.end() && limit->is_number_integer())
        w.limit = static_cast<std::size_t>(std::clamp<std::int64_t>(limit->get<std::int64_t>(), 1, kMaxLimit));
    return w;
}

// A target of a spec event: the panel or the control the command names.
std::string spec_event_panel(const JournalEvent& e) {
    const bool names_one = e.name == "panel.create" || e.name == "panel.patch" || e.name == "panel.remove" ||
                           e.name == "control.create" || e.name == "control.patch" || e.name == "control.remove";
    if (!names_one || !e.args.is_object()) return {};
    const auto id = e.args.find("id");
    return id != e.args.end() && id->is_string() ? id->get<std::string>() : std::string();
}

// Whether a journaled do event changed the open spec: a spec edit, or a
// command that opens another spec. A dry run is never journaled; an edit of a
// stacked preview is not journaled either.
bool changes_open_spec(const std::string& name) {
    return is_spec_edit_command(name) || name == "spec.new" || name == "spec.open" || name == "applet.open";
}

// The callers inside wait_for_events(), by ring. It is never destroyed, so
// that a ring destroyed at exit can still look at it.
struct Callers {
    std::mutex mutex;
    std::condition_variable left;
    std::map<const EventRing*, std::size_t> count;
};

Callers& callers() {
    static Callers* const c = new Callers;
    return *c;
}

}  // namespace

const char* event_kind_name(EventKind kind) {
    switch (kind) {
        case EventKind::pick: return "pick";
        case EventKind::select: return "select";
        case EventKind::control: return "control";
        case EventKind::run: return "run";
        case EventKind::spec: return "spec";
        case EventKind::approval: return "approval";
        case EventKind::program: return "program";
        case EventKind::state: return "state";
        case EventKind::cursor: return "cursor";
    }
    return "pick";
}

std::optional<EventKind> event_kind_from_name(std::string_view name) {
    for (EventKind k : {EventKind::pick, EventKind::select, EventKind::control, EventKind::run, EventKind::spec,
                        EventKind::approval, EventKind::program, EventKind::state, EventKind::cursor})
        if (name == event_kind_name(k)) return k;
    return std::nullopt;
}

// The keys a kind adds to the six common ones (command_table.json, event).
// event_json() keeps only these, so that a producer's extra detail, such as
// the item space of a scene's selection, never leaves the schema.
const std::vector<const char*>& kind_keys(EventKind kind) {
    static const std::vector<const char*> pick = {"entity", "primitive", "index", "row_id", "position", "pixel", "fields", "annotation"};
    static const std::vector<const char*> select = {"entities", "truncated"};
    static const std::vector<const char*> control = {"control", "bind", "value", "key"};
    // The runner's change (design/hosted-runs.md, Runner Events) adds run,
    // parent, exit_code and error.
    // An intervention applied at a step (design/gpu-programs.md) adds point and op.
    static const std::vector<const char*> run = {"change", "state",  "step",   "name",      "value", "status", "text",
                                                 "attached", "script", "run", "parent", "exit_code", "error",
                                                 "point", "op"};
    static const std::vector<const char*> spec = {"command", "event", "applet"};
    static const std::vector<const char*> approval = {"approval", "command", "state", "target"};
    // design/gpu-programs.md, Program Events.
    static const std::vector<const char*> program = {"program", "change", "host", "code", "message", "line", "column"};
    // design/instrument-interface.md, Events.
    static const std::vector<const char*> state = {"state", "value", "previous", "writer"};
    static const std::vector<const char*> cursor = {"change", "step", "first", "latest", "playing", "speed"};
    switch (kind) {
        case EventKind::pick: return pick;
        case EventKind::select: return select;
        case EventKind::control: return control;
        case EventKind::run: return run;
        case EventKind::spec: return spec;
        case EventKind::approval: return approval;
        case EventKind::program: return program;
        case EventKind::state: return state;
        case EventKind::cursor: return cursor;
    }
    return pick;
}

nlohmann::json event_json(const Event& event) {
    json out = json::object();
    if (event.fields.is_object())
        for (const char* key : kind_keys(event.kind)) {
            const auto it = event.fields.find(key);
            if (it != event.fields.end()) out[key] = *it;
        }
    // A cleared selection names no entity.
    if (event.kind == EventKind::select && !out.contains("entities")) out["entities"] = json::array();
    out["seq"] = event.seq;
    out["time"] = event.time;
    out["kind"] = event_kind_name(event.kind);
    out["panel"] = event.panel.empty() ? json(nullptr) : json(event.panel);
    out["view"] = event.view.empty() ? json(nullptr) : json(event.view);
    out["origin"] = event.origin;
    return out;
}

struct EventRing::State {
    std::size_t capacity = kCapacity;
    mutable std::mutex mutex;
    mutable std::condition_variable changed;
    std::deque<Event> events;    // oldest first, at most `capacity`
    std::uint64_t next = 1;      // the seq of the next push
    bool closed = false;
    std::vector<const EventFilter*> waiting;   // the filters of the callers blocked in wait()

    // The cursor a read starts after: a cursor beyond the newest event reads
    // from the oldest event kept (see the top of this file).
    std::uint64_t start(std::uint64_t cursor) const { return cursor >= next ? 0 : cursor; }
    std::uint64_t oldest() const { return events.empty() ? next : events.front().seq; }
};

EventRing::EventRing(std::size_t capacity) : state_(std::make_unique<State>()) {
    state_->capacity = std::max<std::size_t>(1, capacity);
}

EventRing::~EventRing() { close(); }

std::uint64_t EventRing::push(Event event) {
    std::uint64_t seq = 0;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->closed) return 0;
        seq = state_->next++;
        event.seq = seq;
        event.time = command_utc_timestamp(std::chrono::system_clock::now());
        state_->events.push_back(std::move(event));
        while (state_->events.size() > state_->capacity) state_->events.pop_front();
    }
    state_->changed.notify_all();
    return seq;
}

EventRead EventRing::read(std::uint64_t cursor, const EventFilter& filter, std::size_t limit) const {
    EventRead out;
    std::lock_guard<std::mutex> lock(state_->mutex);
    const std::uint64_t from = state_->start(cursor);
    const std::uint64_t oldest = state_->oldest();
    if (from + 1 < oldest) out.lost = oldest - from - 1;
    out.cursor = std::max(from, oldest - 1);
    for (const Event& e : state_->events) {
        if (e.seq <= from) continue;
        if (matches(e, filter)) {
            if (out.events.size() >= limit) {
                out.more = true;
                break;
            }
            out.events.push_back(e);
        }
        out.cursor = e.seq;
    }
    return out;
}

bool EventRing::wait(std::uint64_t cursor, const EventFilter& filter,
                     std::chrono::steady_clock::time_point deadline) const {
    return *wait_if_room(cursor, filter, deadline, static_cast<std::size_t>(-1));
}

std::optional<bool> EventRing::wait_if_room(std::uint64_t cursor, const EventFilter& filter,
                                            std::chrono::steady_clock::time_point deadline,
                                            std::size_t max_waiting) const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    std::uint64_t scanned = state_->start(cursor);
    auto found = [&] {
        // Events dropped before this reader saw them may have matched: the
        // read that follows reports them as lost.
        if (scanned + 1 < state_->oldest()) return true;
        for (auto it = state_->events.rbegin(); it != state_->events.rend() && it->seq > scanned; ++it)
            if (matches(*it, filter)) return true;
        scanned = state_->next - 1;
        return false;
    };
    if (found()) return true;
    if (state_->closed) return false;
    if (state_->waiting.size() >= max_waiting) return std::nullopt;
    state_->waiting.push_back(&filter);
    bool match = false;
    for (;;) {
        if (state_->changed.wait_until(lock, deadline) == std::cv_status::timeout) {
            match = found();
            break;
        }
        if (found()) {
            match = true;
            break;
        }
        if (state_->closed) break;
    }
    state_->waiting.erase(std::find(state_->waiting.begin(), state_->waiting.end(), &filter));
    return match;
}

std::size_t EventRing::waiting(std::optional<EventKind> kind) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::size_t n = 0;
    for (const EventFilter* f : state_->waiting)
        if (!kind || f->kinds.empty() || std::find(f->kinds.begin(), f->kinds.end(), *kind) != f->kinds.end()) ++n;
    return n;
}

std::uint64_t EventRing::latest() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->next - 1;
}

void EventRing::close() {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->closed = true;
    }
    state_->changed.notify_all();
    Callers& c = callers();
    std::unique_lock<std::mutex> lock(c.mutex);
    c.left.wait(lock, [&] {
        const auto it = c.count.find(this);
        return it == c.count.end() || it->second == 0;
    });
    c.count.erase(this);
}

CommandOutput events_wait_command(const CommandContext& context, EventRing& ring) {
    const WaitArgs w = wait_args(context.invocation.args);
    EventRead r = ring.read(w.cursor, w.filter, w.limit);
    json events = json::array();
    for (const Event& e : r.events) events.push_back(event_json(e));
    return CommandOutput::ok(
        {{"events", std::move(events)}, {"cursor", r.cursor}, {"lost", r.lost}, {"more", r.more}});
}

EventWaitHold::EventWaitHold(const EventRing& ring) : ring_(&ring) {
    std::lock_guard<std::mutex> lock(callers().mutex);
    ++callers().count[ring_];
}

EventWaitHold::~EventWaitHold() {
    Callers& c = callers();
    {
        std::lock_guard<std::mutex> lock(c.mutex);
        --c.count[ring_];
    }
    c.left.notify_all();
}

CommandResult wait_for_events(CommandQueue& queue, EventRing& ring, CommandInvocation invocation) {
    const EventWaitHold caller(ring);

    const WaitArgs w = wait_args(invocation.args);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(w.timeout_s));
    CommandResult first = queue.submit(invocation).get();
    if (!first.ok || w.timeout_s <= 0.0 || !first.result.is_object()) return first;
    const auto events = first.result.find("events");
    if ((events != first.result.end() && events->is_array() && !events->empty()) ||
        first.result.value("lost", std::uint64_t(0)) > 0)
        return first;
    // Wait for a match after everything the first answer looked at, then ask
    // the same question again, which answers at once.
    if (!ring.wait_if_room(first.result.value("cursor", w.cursor), w.filter, deadline, kMaxEventWaits))
        return CommandResult::failure(
            first.id,
            Error(kEventsTooManyWaits,
                  std::to_string(kMaxEventWaits) +
                      " calls of events.wait wait already, the most at once; call again after one answers, or with "
                      "timeout_s 0, which answers at once.",
                  "timeout_s"));
    invocation.args["timeout_s"] = 0;
    return queue.submit(std::move(invocation)).get();
}

void note_journal_event(EventSink& sink, const JournalEvent& event, const std::string& applet) {
    if (!event.ok) return;
    const bool history = event.event == "undo" || event.event == "redo";
    if (!history && !(event.event == "do" && changes_open_spec(event.name))) return;
    Event e;
    e.kind = EventKind::spec;
    e.origin = command_origin_name(event.origin);
    e.panel = history ? std::string() : spec_event_panel(event);
    e.fields = {{"command", event.name}, {"event", event.event}};
    if (!applet.empty()) e.fields["applet"] = applet;
    sink.push(std::move(e));
}

}  // namespace caliper_builder
