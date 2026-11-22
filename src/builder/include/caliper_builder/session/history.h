#pragma once
// caliper_builder: the recent steps that the app keeps for the sources of
// the open spec, and the time cursor over them
// (design/instrument-interface.md, section Time Transport).
//
// The app keeps each source's value after its longest reference-free prefix
// of steps, one value for each step, up to the source's depth (`history`,
// or 64 for a live tensor source without `from`). All kept steps of the open
// spec share a budget, and the largest rings shorten first. A new
// attachment, a change of a source's chain and the opening of a spec clear
// the source's kept steps.
//
// The cursor is the built-in state "cursor": the step the panels show. It is
// live when a spec opens and then moves to each new step; held, it stays.
//
// Threads: the live catalog appends on the frame thread, commands read and
// move the cursor on the command thread; every member function takes its own
// lock. The session creates one StepHistory (Session::history()), and
// history.cpp defines it.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_builder {

inline constexpr std::int64_t kDefaultLiveDepth = 64;
inline constexpr std::uint64_t kHistoryBudgetBytes = std::uint64_t(512) << 20;

// The fields of the cursor (design/instrument-interface.md, Cursor).
struct CursorState {
    std::int64_t step = 0;
    std::int64_t first = 0;     // the oldest step that any source keeps
    std::int64_t latest = 0;    // the newest step that the app received
    bool live = true;
    bool playing = false;
    double speed = 10;          // steps per second of play, 0.1 to 1,000
};
nlohmann::json cursor_json(const CursorState& cursor);

// What one source keeps, for cursor.get.
struct KeptSource {
    std::string id;
    std::int64_t first = 0, last = 0;     // the oldest and newest kept step
    std::int64_t steps = 0;               // how many it keeps now
    std::int64_t requested = 0;           // its depth before the budget
    std::uint64_t bytes = 0;
};

class StepHistory {
public:
    StepHistory();
    ~StepHistory();
    StepHistory(const StepHistory&) = delete;
    StepHistory& operator=(const StepHistory&) = delete;

    // Sets each source's depth and history point from the spec, keeps what
    // still fits and clears what changed.
    void configure(const spec::Spec& spec);
    // Clears every source, for a new attachment.
    void clear();

    // A new value of a live view at a step: the stamp's step, else the run's
    // step at arrival. Each source whose root is `catalog_id` keeps it.
    void append(const std::string& catalog_id, std::int64_t step, const TensorData& value);

    // The newest kept value of `source` at or before `step`.
    std::optional<std::pair<std::int64_t, TensorData>> at(const std::string& source, std::int64_t step) const;
    // The kept values of `source` from `lo` to `hi`, both included, oldest first.
    std::vector<std::pair<std::int64_t, TensorData>> window(const std::string& source, std::int64_t lo,
                                                            std::int64_t hi) const;
    std::vector<KeptSource> kept() const;
    // The union of the kept steps of every source, for a step back or forward.
    std::vector<std::int64_t> kept_steps() const;

    CursorState cursor() const;
    // Moves the cursor as cursor.set does; a step outside the kept steps
    // goes to the nearest kept step. Returns the cursor it took.
    CursorState set_cursor(const CursorState& wanted);
    // Moves the cursor by `count` kept steps.
    CursorState step_cursor(std::int64_t count);
    // Advances a playing cursor by the frame's time; at the newest step it
    // becomes live. Called once a frame.
    void tick(double seconds);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_builder
