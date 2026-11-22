#pragma once
// The app's host of stages (spec_stages.h, design/gpu-programs.md, Viewer Stage Interface and
// Time Watchdog): its class, the compile of one stage program and the time watchdog, which the
// spec runtime and the tests use. app_stage_host() gives the same object as a StageHost.
//
// For each stage of the open spec, the host runs the sandbox checks in their order
// (caliper_builder/programs/sandbox.h): the declaration, the scene panel, the provenance, the
// source, the compile of the prelude and the source, and the stage's bindings. It then composes
// the program of each entity with a stage for points and for lines (the prelude, the material and
// vertex sources, the built-in program's WGSL and stage_entries()), validates it and translates
// its three entry points to MSL with caliper-wgsl, and gives the builds to the panel's
// StageSource, from which each scene view makes its programs. A refusal names the code, the
// message, and the line and column in the stage's own source; it is the stage's status, a
// `program` event and the line the panel shows, never an error of the app.
//
// Threads: reconcile(), source() and message() on the frame thread; status() on any thread.
#include "scene/stage_cache.h"

#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/program_host.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_host {

// The outcome of composing one stage program.
struct StageCompile {
    std::shared_ptr<const caliper_scene::StageBuild> build;   // null on a refusal
    caliper_builder::programs::Verdict verdict;               // the refusal, its line and column in the stage's source
    int culprit = -1;                                         // the source the refusal is in: 0 material, 1 vertex, -1 none
};

// Composes, validates and translates the program of one entity for one primitive (points or
// lines) with the given material and vertex sources (either may be null, not both). Without the
// caliper-wgsl library the verdict is program.compiler_unavailable.
//
// `params` names the f32 fields of the uniform `params` that the stages read (their params, in
// name order): the composed module declares them after the stage sources (sandbox.h,
// stage_params_declaration()), and the build binds them at @group(3) @binding(0).
StageCompile compile_stage_program(caliper_scene::Primitive primitive, const std::string* material,
                                   const std::string* vertex, const std::string& label,
                                   const std::vector<std::string>& params = {});

// Joins per-entry MSL texts of one module into one text: their common leading lines once, then
// what follows in each. caliper-wgsl translates one entry point at a time, and the graphics
// layer makes a program from one MSL text with all of its entry points.
std::string merge_msl(const std::vector<std::string>& texts);

// The time watchdog of one scene panel with stages: when the GPU time of a frame with its stages
// is above kLimitMs for kFrames frames in a row, and above the panel's time without its stages,
// the stages stop. Without a time of the panel without its stages it asks for one frame without
// them (probe) before it decides. A frame with the stages that the GPU has not completed after
// kHangMs stops them at once: its stage may not end, and the GPU cannot stop it.
class StageWatchdog {
public:
    static constexpr double kLimitMs = 16.7;
    static constexpr uint32_t kFrames = 30;
    static constexpr double kHangMs = 1000.0;
    enum class Action { none, probe, stop };
    // Frames completed: their mean GPU time, their count, and whether they drew the stages.
    Action frames(double gpu_ms, uint32_t count, bool staged);
    // The oldest frame with the stages that has not completed has waited this long.
    static Action pending(double waited_ms) { return waited_ms > kHangMs ? Action::stop : Action::none; }
    // Forgets what it measured (new stages).
    void reset() { *this = StageWatchdog(); }
    uint32_t over() const { return over_; }
    double baseline_ms() const { return baseline_ms_; }
    bool probing() const { return probing_; }

private:
    uint32_t over_ = 0;          // frames with the stages over the limit, in a row
    double last_ms_ = 0;         // the last such frame's time
    double baseline_ms_ = -1;    // the panel's last time without its stages; < 0 unknown
    bool probing_ = false;
};

// The sandbox checks the host runs before its own compile (sandbox.h, steps 1 to 6). The app
// runs the sandbox's; a test may give others to reach the host's own steps in a tree whose
// sandbox is not built.
struct StageChecks {
    caliper_builder::programs::Verdict (*declaration)(const caliper_builder::spec::Stage&) = nullptr;
    caliper_builder::programs::Verdict (*provenance)(const caliper_builder::spec::Provenance&, std::string_view,
                                                     std::string_view) = nullptr;
    caliper_builder::programs::Verdict (*source)(std::string_view, std::size_t) = nullptr;
    caliper_builder::programs::Compiled (*compile)(std::string_view) = nullptr;
    caliper_builder::programs::Verdict (*stage)(const caliper_builder::spec::Stage&,
                                                const caliper_builder::programs::Reflection&) = nullptr;
    // The sandbox's own functions.
    static StageChecks sandbox();
};

// The app's stage host.
class AppStageHost final : public caliper_builder::StageHost {
public:
    explicit AppStageHost(StageChecks checks = StageChecks::sandbox());
    ~AppStageHost() override;

    void reconcile(const caliper_builder::spec::Spec& spec) override;
    std::vector<caliper_builder::ProgramStatus> status() const override;

    // Where the host pushes its `program` events; null pushes none. The spec runtime gives the
    // session's ring.
    void set_events(caliper_builder::EventSink* sink);
    // The stages of a scene panel, for its scene view; null when the panel has none that compiled.
    // The object lives as long as the host.
    caliper_scene::StageSource* source(const std::string& panel);
    // The line a panel shows over its picture: its first stage that a check refused or the
    // watchdog stopped, with the code, the line and the column; empty when none.
    std::string message(const std::string& panel) const;
    // What the references of the stages' params read (the session's values, ValueStore::value()
    // in the app); null gives each reference its default. The panels evaluate them at each frame
    // (StageSource::params()), thus a new value reaches the uniform at the next frame.
    void set_evaluate(std::function<nlohmann::json(const caliper_builder::spec::Reference&)> evaluate);
    // The status items of the instances whose draw half the scene panels run, with half "draw"
    // and `draw` set; status() gives them after the stages, so that program.status shows each
    // draw half (caliper_builder::StageHost). Any thread.
    void set_drawn(std::function<std::vector<caliper_builder::ProgramStatus>()> drawn);

private:
    struct Record;
    class Panel;
    friend class Panel;
    void push(const caliper_builder::ProgramStatus& status, const char* change);

    StageChecks checks_;
    mutable std::mutex mutex_;                       // records_ and the panels' counters
    std::vector<Record> records_;                    // the last reconciled spec's stages, in its order
    std::map<std::string, std::unique_ptr<Panel>> panels_;
    caliper_builder::EventSink* events_ = nullptr;
    std::function<nlohmann::json(const caliper_builder::spec::Reference&)> evaluate_;
    std::function<std::vector<caliper_builder::ProgramStatus>()> drawn_;   // under mutex_
};

// The host behind app_stage_host(), as its class.
AppStageHost* app_stage_host_impl();

}  // namespace caliper_host
