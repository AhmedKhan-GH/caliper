// live_points.h: the points live view as a scene of the scene core (design/renderer-core.md,
// "The points view and ImPlot3D"; migration step 7).
//
// The view's payload is one points batch on the root entity: the positions and the optional values
// or labels are imported layer buffers at their byte offsets, which the generated points program
// reads as instance streams. A payload above the scene model's 4,194,304 points a batch continues
// on child entities "c00001", "c00002" and so on, at later offsets of the same buffers, so that it
// draws in the same order. Flat points take the view's grey; values map through the shared colormap
// cache (src/host/gfx/colormap_keys.h); labels, which the producer sends as f32, map through the
// same keys over the range 0 to 255 with a table that holds the fixed 16-colour palette at every
// level, as the old view's 256-entry table did, so that labels above 255, below 0 and NaN keep
// their colours and the payload is read in place. A selection draws every point dimmed toward the
// ground, then the selected points in their own colours on the entity "selection". Points are
// squares of 7, 5.5, 4 or 3 logical pixels by count (caliper_scene::default_point_size). In 3-D the
// points draw the view's depth cues (CompileOptions): nearer points larger within a clamp and the
// points behind the camera's target dimmed towards the far side of the fitted bounding sphere, the
// selected points of a selection at full strength.
//
// The camera is the scene core's: an orbit about +y in 3-D, a pan and zoom with y up in 2-D, fitted
// by caliper_scene::fit at the default three-quarter view (camera_defaults.h), composed in double
// and handed to the GPU as f32 once. The gestures keep the old view's rates and bounds. The scene core and the layer draw into an offscreen colour target,
// with depth in 3-D, which the view's window shows as an ImGui image.
//
// Frame thread only. Nothing here waits for the GPU except read_pixels(), which tests call.
#pragma once

#include "gfx/gfx.h"
#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/scene.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caliper_scene { class BackendResources; }

namespace caliper_live_views {

// The arrays of one set of points: positions (N, 3) f32 and an optional (N,) f32 aux, in layer
// buffers at byte offsets; pos_cpu is a CPU view of the positions (bounds without a box), or null.
struct PointsArrays {
    caliper_gfx::Buffer pos;
    uint64_t pos_offset = 0;
    const void* pos_cpu = nullptr;
    caliper_gfx::Buffer aux;
    uint64_t aux_offset = 0;
    uint64_t count = 0;
};

enum class PointsColour : uint8_t { Flat, Values, Classes };

struct PointsLook {
    int dims = 3;                          // 2: x and y of each (x, y, 0) row
    PointsColour colour = PointsColour::Flat;
    int32_t colormap = 0;                  // values: CALIPER_CMAP_*
    float vmin = 0.0f, vmax = 1.0f;        // values: the range
    bool has_bbox = false;
    float bbox[6] = {};                    // xmin, xmax, ymin, ymax, zmin, zmax (the producer's stats)
    float size_px = 0.0f;                  // logical pixels of a point's side
    uint32_t ground = 0xff000000u;         // the plot's clear colour, which a selection dims toward
};

class PointsScene {
public:
    // backend: borrowed, it outlives the scene. programs: the backend's table of generated programs.
    PointsScene(caliper_gfx::Backend& backend, const caliper_gfx::ProgramDesc* programs);
    ~PointsScene();
    PointsScene(const PointsScene&) = delete;
    PointsScene& operator=(const PointsScene&) = delete;

    bool init(std::string* why);
    caliper_gfx::Backend& gfx() { return gfx_; }

    // The points to draw, and, with a selection, the selected points drawn over the dimmed rest.
    // A change of dims changes the camera model. False with *why when the scene store refused it.
    bool set(const PointsArrays& all, const PointsArrays* selected, const PointsLook& look, std::string* why);
    int dims() const { return dims_; }

    // ---- the camera, with the old view's gestures (logical pixels of the view) ----------------
    void fit(double aspect);               // to the scene's bounds, at the default angles in 3-D
    void orbit(float dx, float dy);
    void turn(double radians);             // the azimuth (the rotate tool)
    void set_angles(double azimuth, double elevation);   // 3-D: the orbit's angles (tests)
    double radius() const { return radius_; }            // the fitted bounding sphere's radius
    void pan(float dx, float dy, float view_height);
    void zoom(float wheel, float mx, float my, double aspect);   // m: the pointer, -1 to 1
    const caliper_scene::ViewCamera& camera() const { return camera_; }
    // The f32 clip-from-world matrix of a width x height target, as the draws get it.
    bool mvp(uint32_t width, uint32_t height, float out[16]) const;

    // ---- drawing ---------------------------------------------------------------------------------
    // The depth cues of 3-D points (on by default): the size relative to the size at the orbit's
    // target, and the fade from the target's distance to the far side of the fitted sphere.
    void set_depth_cues(bool on) { cues_ = on; }
    bool depth_cues() const { return cues_; }
    // The cues' values for the current camera (CompileOptions), all zero when they are off or in 2-D.
    void cue_values(float* att_ref, float* fade_w0, float* fade_w1) const;
    // Compiles and submits one frame into a target of width x height physical pixels, cleared to
    // clear_rgba; the target is made or remade to that size. *serial: the submit, which reads the
    // points' buffers. False with *why when the scene did not compile or the layer refused.
    bool render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba,
                caliper_gfx::Serial* serial, std::string* why);
    bool has_target() const { return bool(color_); }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint64_t imgui_texture() const;
    caliper_gfx::Texture color() const { return color_; }
    // The colour target, tightly packed RGBA8 (tests; waits for the GPU).
    bool read_pixels(std::vector<uint8_t>* rgba, std::string* why);
    // Retires completed work; the layer's statistics since the last call.
    caliper_gfx::Stats poll();

    // The draws of the last frame (tests), kept while keep_draws is on.
    void keep_draws(bool on) { keep_draws_ = on; }
    const caliper_gfx::DrawList& last_draws() const { return last_list_; }
    const caliper_scene::DrawMap& last_map() const { return last_map_; }
    const caliper_scene::SceneTree& tree() const { return tree_; }

    // The colormap ids of the view's tables beyond the built-in colormaps: the palette at every
    // level, and a flag for a table dimmed toward the ground.
    static constexpr uint32_t kPaletteTable = 0x10000u;
    static constexpr uint32_t kDimmed = 0x20000u;

    struct Tables;   // the colormap source's state (live_points.cpp)

private:
    caliper_gfx::Backend& gfx_;
    const caliper_gfx::ProgramDesc* programs_;
    std::shared_ptr<Tables> tables_;
    std::unique_ptr<caliper_scene::BackendResources> res_;
    caliper_scene::SceneTree tree_;
    caliper_scene::SceneCompiler compiler_;
    caliper_scene::ViewCamera camera_ = caliper_scene::Orbit{};
    int dims_ = 3;
    double radius_ = 1.0;                  // the fitted bounds' radius: the zoom's bounds and the fade
    bool cues_ = true;

    caliper_gfx::Texture color_, depth_;
    uint32_t width_ = 0, height_ = 0;

    bool keep_draws_ = false;
    caliper_gfx::DrawList last_list_;
    caliper_scene::DrawMap last_map_;
};

// The packed colour dimmed toward the ground as the points view dims the unselected points
// (0.28 of the colour, 0.72 of the ground, rounded; alpha 255).
uint32_t dimmed_rgba(uint32_t rgba, uint32_t ground);

}  // namespace caliper_live_views
