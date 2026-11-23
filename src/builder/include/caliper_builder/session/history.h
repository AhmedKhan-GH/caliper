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
//
// The step history also keeps the versions of each published resource of
// the open spec's programs, under "<program>/<resource>"
// (design/programmable-graphics.md, Kept Versions), beside the sources'
// kept steps, so that the step cursor scrubs them too.
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
// The most that one source keeps: a quarter of the budget, 128 MiB. A raw
// activation of 512 by 4096 values (8 MiB a step) thus keeps 16 steps and
// leaves the rest of the budget to the other sources; a reduce step before
// the source's history point keeps more.
inline constexpr std::uint64_t kHistorySourceBytes = kHistoryBudgetBytes / 4;

// The kept versions of one published resource: `keep` of the piece's
// resource, 256 when absent and at most 4,096.
inline constexpr std::uint32_t kResourceKeepDefault = 256;
inline constexpr std::uint32_t kResourceKeepMax = 4096;
// All resources together keep at most half the budget, and one resource at
// most kHistorySourceBytes; the sources share the rest. Over that half, the
// largest rings thin first, then rings drop their oldest versions down to
// one each, and a version that still does not fit is not kept.
inline constexpr std::uint64_t kHistoryResourceBytes = kHistoryBudgetBytes / 2;
// The resources that the history keeps for one program, as many as a piece
// has, and for all programs together, as many as 16 programs of a producer
// (CALIPER_NET_MAX_PROGRAMS) publish. A version of another resource is not kept.
inline constexpr std::size_t kResourceRingsPerProgram = spec::piece_limits::kResources;
inline constexpr std::size_t kResourceRingsMax = 16 * kResourceRingsPerProgram;

// One field of a resource's element as program.read gives it: its WGSL type
// (f32, i32, u32, vec2<f32>, vec4<f32>, vec2<u32> or vec4<u32>) and its byte
// offset in the element.
struct ResourceField {
    std::string name;
    std::string type;
    std::uint32_t offset = 0;
};

// One version of a published resource: the valid elements that a derive pass
// wrote at a step of the run, as RESOURCE carried them.
struct ResourceVersion {
    std::int64_t step = 0;
    std::uint32_t count = 0;                 // valid elements
    std::uint32_t stride = 0;                // bytes of one element
    std::uint32_t width = 0, height = 0;     // a texture's size; 0 for a buffer
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;   // count * stride bytes, shared with readers
    // The history's number of the version, which differs for a version that
    // replaces another at the same step (append_resource() sets it).
    std::uint64_t serial = 0;
    std::uint64_t nbytes() const { return bytes ? bytes->size() : 0; }
};

// What one resource keeps, for program.read.
struct KeptResource {
    std::string key;                         // "<program>/<resource>"
    std::int64_t first = 0, last = 0;        // the oldest and newest kept step
    std::int64_t versions = 0;
    std::uint32_t keep = kResourceKeepDefault;
    std::int64_t stride = 0;                 // the thinning stride in steps; 0 while every version is kept
    std::uint64_t bytes = 0;
    std::vector<ResourceField> fields;       // the element's fields; empty when not known
};

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
    // Why it keeps fewer steps than it requests, naming the source and the
    // reduce step that keeps more; empty when it keeps them all.
    std::string note;
};

class StepHistory {
public:
    StepHistory();
    ~StepHistory();
    StepHistory(const StepHistory&) = delete;
    StepHistory& operator=(const StepHistory&) = delete;

    // Sets each source's depth and history point from the spec, keeps what
    // still fits and clears what changed. The versions of a resource of a
    // program that the spec no longer has, or that pins another piece, are
    // dropped.
    void configure(const spec::Spec& spec);
    // Clears every source, for a new attachment.
    void clear();

    // A new value of a live view at a step: the stamp's step, else the run's
    // step at arrival. Each source whose root is `catalog_id` keeps it; a
    // value of the newest step replaces it. A step below the newest step of
    // the same view starts a new epoch: every source forgets its kept steps
    // after that step before the value is kept, and a cursor held after it
    // goes live.
    void append(const std::string& catalog_id, std::int64_t step, const TensorData& value);
    // The newest value of a live series at a step. A series keeps its own
    // ring of values without their steps, so only a source over it that
    // states `history` keeps these, as 0-d values, and a window over that
    // source gives each value with its step.
    void append_series(const std::string& catalog_id, std::int64_t step, double value);

    // A new version of the resource `resource` of the program `program`, an
    // instance of a piece in the open spec (else the version is not kept), up
    // to kResourceRingsPerProgram resources of a program and kResourceRingsMax
    // of all programs; the version gets a new serial.
    // `keep` is the piece's `keep` for it (0: kResourceKeepDefault), and
    // `fields` its element's fields. A version at a step below the newest
    // step of the same resource starts a new epoch, as append() does. When
    // the ring holds `keep` versions or kHistorySourceBytes, every second
    // kept version is dropped, and from then on a version is kept only when
    // its step is at least twice the former stride after the newest kept
    // one; the newest version is always there until a later one replaces it.
    // Thus the kept versions span the whole run evenly.
    void append_resource(const std::string& program, const std::string& resource, std::uint32_t keep,
                         std::vector<ResourceField> fields, ResourceVersion version);
    // The newest kept version of "<program>/<resource>" at or before `step`.
    std::optional<ResourceVersion> resource_at(const std::string& key, std::int64_t step) const;
    // What each resource keeps, by key.
    std::vector<KeptResource> kept_resources() const;

    // The newest kept value of `source` at or before `step`.
    std::optional<std::pair<std::int64_t, TensorData>> at(const std::string& source, std::int64_t step) const;
    // The kept values of `source` from `lo` to `hi`, both included, oldest first.
    std::vector<std::pair<std::int64_t, TensorData>> window(const std::string& source, std::int64_t lo,
                                                            std::int64_t hi) const;
    std::vector<KeptSource> kept() const;
    // The union of the kept steps of every source and every resource, for a
    // step back or forward.
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
    void append_to(const std::string& catalog_id, std::int64_t step, const TensorData& value, bool declared_only);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_builder
