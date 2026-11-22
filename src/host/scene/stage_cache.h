#pragma once
// The stage programs of a scene view (design/gpu-programs.md, Viewer Stage Interface).
//
// The spec runtime's stage host checks and translates each stage of a scene panel once, into a
// StageBuild that names no backend: the composed WGSL, the merged MSL of the entry points of
// stage_entries() and the binding table. Each scene view draws on its own backend of the
// graphics layer, thus each view keeps a StageCache, the StageLookup that its compiles use. The
// cache makes a build into a caliper_gfx::Program on the view's backend at its first use, keeps
// it while the panel's StageSource gives the same build, and releases it after the frame that
// stopped using it, or with the view. The layer keeps a released program's pipelines working
// until their last submit completed.
//
// Threads: the frame thread, as the compiler. Nothing here throws.
#include "gfx/gfx.h"
#include "scene/resources.h"
#include "scene/stages.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_scene {

// One stage program of one entity and primitive, before any backend.
struct StageBuild {
    Primitive primitive = Primitive::Points;
    ProgramId builtin = ProgramId::scene_points;    // the built-in program it replaces, with the same bindings
    std::string label;                              // for the layer's messages
    std::string wgsl;                               // the composed module
    std::string msl;                                // vs_stage, fs_stage and fs_stage_pick in one MSL text
    std::vector<caliper_gfx::BindingDesc> bindings; // the built-in program's, with the stage bits and indices
                                                    // of the composed module
    bool material = false;                          // a material stage is in it
    bool vertex = false;                            // a vertex stage is in it
    bool animated = false;                          // a stage reads time_s or step: the view draws each frame
    uint64_t source_hash = 0;                       // of the composed WGSL
    // The names of the stages' params, the f32 fields of the uniform `params` in struct order at
    // @group(3) @binding(0) (sandbox.h, kStageParamsGroup); empty when the program reads none.
    std::vector<std::string> params;
};

// The stages of one scene panel, which its scene view asks on the frame thread.
class StageSource {
public:
    virtual ~StageSource() = default;
    // The build of the entity at `path` for this primitive, or null to keep the built-in program.
    virtual std::shared_ptr<const StageBuild> build(std::string_view path, Primitive primitive) const = 0;
    // Changes whenever a build of the panel appears, changes or goes, and when the time watchdog
    // turns the panel's stages off or on: the view then draws again.
    virtual uint64_t revision() const = 0;
    // The seconds since the panel's stages were compiled: MaterialIn.time_s and VertexIn.time_s.
    virtual float time_s() const = 0;
    // The view's layer refused to make `build` into a program, with the layer's message.
    virtual void refused(const StageBuild& build, const std::string& message) = 0;
    // The GPU time of frames the view completed since the last call: their mean in milliseconds,
    // their count, and whether they drew a stage program. The time watchdog's input.
    virtual void frames_done(double gpu_ms, uint32_t frames, bool staged) = 0;
    // The bytes of the uniform `params` of the entity at `path` now (StageProgram::params): the
    // values of its stages' params, evaluated at this frame; empty when it has none.
    virtual std::vector<uint8_t> params(std::string_view path) const {
        (void)path;
        return {};
    }
};

// The StageLookup of one scene view over its panel's StageSource, which makes and keeps the
// programs on the view's backend.
class StageCache final : public StageLookup {
public:
    explicit StageCache(caliper_gfx::Backend& backend);
    ~StageCache() override;
    StageCache(const StageCache&) = delete;
    StageCache& operator=(const StageCache&) = delete;

    // The panel's stages from now on; null draws none. The view sets it before each render.
    void set_source(StageSource* source) { source_ = source; }
    StageSource* source() const { return source_; }
    // Starts a compile: find() marks what it gives as used.
    void begin_frame() { ++frame_; }
    // After a compile: releases the programs that it did not use.
    void end_frame();
    // Whether the last compile drew a build that reads the time or the step.
    bool animated() const;
    // The programs the cache holds (tests).
    size_t size() const { return entries_.size(); }
    // Whether the view's backend binds StageProgram::params as the uniform of group 3. A build
    // with params draws only then; else it keeps the built-in program and its source hears why.
    void set_params_bound(bool bound) { params_bound_ = bound; }

    const StageProgram* find(std::string_view path, Primitive primitive) const override;

private:
    struct Entry {
        std::shared_ptr<const StageBuild> build;
        StageProgram program;
        bool ok = false;           // the layer made the program
        uint64_t used = 0;         // the frame that last used it
    };
    caliper_gfx::Backend& backend_;
    StageSource* source_ = nullptr;
    uint64_t frame_ = 0;
    bool params_bound_ = false;
    mutable std::map<std::string, Entry> entries_;   // by path, a NUL and the primitive
};

}  // namespace caliper_scene
