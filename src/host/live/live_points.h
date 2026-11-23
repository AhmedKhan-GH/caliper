// live_points.h: the points live view as a scene of the scene core (design/renderer-core.md,
// "The points view and ImPlot3D"; migration step 7).
//
// The view writes points@1 resources from the payload. Each instance holds at most 4,194,304
// points. Further instances draw the remaining points, followed by the selected points.
// The view packs colors with the shared colormap keys and palette. It passes camera matrices,
// point sizes, and depth cues to each piece. A payload in private GPU storage first returns
// through asynchronous texture readback. Missing or stopped pieces report an error.
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
#include "scene/scene.h"

#include <cstdint>
#include <memory>
#include <string>


namespace caliper_live_views {

// The arrays of one set of points: positions (N, 3) f32 and an optional (N,) f32 aux, in layer
// buffers at byte offsets; pos_cpu is a CPU view of the positions (bounds without a box), or null.
struct PointsArrays {
    caliper_gfx::Buffer pos;
    uint64_t pos_offset = 0;
    const void* pos_cpu = nullptr;
    caliper_gfx::Buffer aux;
    uint64_t aux_offset = 0;
    const void* aux_cpu = nullptr;         // a CPU view of aux, or null
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

    bool drew_pieces() const;
    const std::string& error() const { return error_; }
    const std::string& error_detail() const { return error_detail_; }   // the layer's or runtime's words
    // A payload that the CPU cannot read is on its way back from the GPU: render() again later.
    bool pending() const;

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
    // Puts back a camera that a press on the picture saved, when the press ends as a click
    // (live/camera_drag.h).
    void set_camera(const caliper_scene::ViewCamera& camera) { camera_ = camera; }
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

    const caliper_scene::SceneTree& tree() const { return tree_; }

    // The colormap ids of the view's tables beyond the built-in colormaps: the palette at every
    // level, and a flag for a table dimmed toward the ground.
    static constexpr uint32_t kPaletteTable = 0x10000u;
    static constexpr uint32_t kDimmed = 0x20000u;

    struct Tables;   // the colormap source's state (live_points.cpp)

private:
    caliper_gfx::Backend& gfx_;
    std::shared_ptr<Tables> tables_;
    caliper_scene::SceneTree tree_;
    caliper_scene::ViewCamera camera_ = caliper_scene::Orbit{};
    int dims_ = 3;
    double radius_ = 1.0;                  // the fitted bounds' radius: the zoom's bounds and the fade
    bool cues_ = true;

    caliper_gfx::Texture color_, depth_;
    uint32_t width_ = 0, height_ = 0;


    // points@1's instances, their resources and the readback of a payload (live_points.cpp).
    struct Pieces;
    std::unique_ptr<Pieces> pieces_;
    std::string error_;
    std::string error_detail_;
    void set_pieces(const PointsArrays& all, const PointsArrays* selected, const PointsLook& look);
    void add_rows(const PointsArrays& a, bool selection, bool dimmed, const PointsLook& look);
    bool render_pieces(float pixel_scale, uint32_t clear_rgba, caliper_gfx::Serial* serial, std::string* why);
};

// The packed colour dimmed toward the ground as the points view dims the unselected points
// (0.28 of the colour, 0.72 of the ground, rounded; alpha 255).
uint32_t dimmed_rgba(uint32_t rgba, uint32_t ground);

}  // namespace caliper_live_views
