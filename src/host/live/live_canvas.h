#pragma once
// A GPU-drawn 3-D view in an inspector pane: an offscreen geometry view the renderer draws into, buffers imported
// for its sources, and Model 3D's orbit camera (left drag rotates, right drag or shift-drag moves, the wheel zooms
// at the pointer). The Embedding and Trajectories panes draw through it; their overlays use its projection.
#include "live/model3d_geometry.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace caliper_host { class HostRenderer; }

namespace caliper_live {

class GeomCanvas {
public:
    struct Frame {
        model3d::Rect rect;               // the view on screen, in ImGui coordinates
        model3d::Matrices m;
        bool hovered = false;             // the pointer is over the view and not dragging it
        float mouse_x = 0, mouse_y = 0;
    };
    GeomCanvas() = default;
    GeomCanvas(const GeomCanvas&) = delete;
    GeomCanvas& operator=(const GeomCanvas&) = delete;
    ~GeomCanvas();
    // Binds the renderer; a different one releases the view and every import, and bumps generation().
    void attach(caliper_host::HostRenderer* renderer);
    // The renderer draws geometry from imported Metal buffers.
    bool ready() const;
    // Counts resets: a source uploaded under an older generation is gone and must be uploaded again.
    int generation() const { return generation_; }
    // A GPU buffer holding these bytes, imported for geometry; replaces the buffer `id` held. 0 on failure.
    uint64_t upload(uint64_t& id, const void* bytes, size_t length);
    caliper_host::HostRenderer* renderer() const { return renderer_; }
    void invalidate() { dirty_ = true; }
    model3d::Camera camera;
    // Lays the view out over the rest of the window, takes the pointer, and calls render(view, matrices) to redraw
    // when the camera, the size or the content changed. Overlays drawn after it may take the pointer first.
    Frame draw(const char* id, const std::function<bool(uint64_t view, const model3d::Matrices&)>& render);
    // Where (x, y, z) lands on screen in frame f; false behind the camera.
    static bool project(const Frame& f, float x, float y, float z, float& sx, float& sy);

private:
    void release();
    caliper_host::HostRenderer* renderer_ = nullptr;
    std::vector<uint64_t> imports_;
    uint64_t target_ = 0;
    int width_ = 0, height_ = 0, generation_ = 0;
    double aspect_ = 0;
    bool dirty_ = true, failed_ = false;
    unsigned drag_ = 0;
};

} // namespace caliper_live
