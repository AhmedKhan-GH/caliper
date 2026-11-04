#pragma once

#include "live/model3d_geometry.h"

#include <memory>
#include <string>
#include <vector>

namespace caliper_host { class HostRenderer; }
namespace caliper_live {

using ModelTensor = model3d::Tensor;

namespace model3d {
// A small row-major atlas. Folded row blocks run from left to right; both
// output sides stay at most 128 pixels, including legacy thumbnail replies.
struct ThumbnailAtlas { int width = 0, height = 0; std::vector<float> values; };
bool thumbnail_atlas(const Tensor& tensor, int rows, int cols,
                     const std::vector<float>& values, ThumbnailAtlas* output);
struct ViewState { Camera camera; Rect viewport, labels; };
struct FrameBand { Vec3 center; double width = 0, height = 0; };
// Coplanar, contained quad bands for a frame of approximately pixel_width.
// Edge-on projections use bounded thickness rather than widening the panel.
std::array<FrameBand, 4> frame_bands(const Panel& panel, const Camera& camera,
                                    Rect viewport, double pixel_width);
} // namespace model3d

// Owns only bounded thumbnail samples, render resources and a camera. Requests
// remain in the inspector's serial scheduler; this class never opens a socket.
class Model3D {
public:
    Model3D();
    ~Model3D();
    Model3D(const Model3D&) = delete;
    Model3D& operator=(const Model3D&) = delete;

    void set_tensors(const std::vector<ModelTensor>& tensors,
                     const std::string& output_label = "Output (tied)");
    bool set_thumbnail(const std::string& name, int rows, int cols,
                       const std::vector<float>& values, double minimum, double maximum,
                       uint64_t revision = 0);
    void invalidate(const std::string& name, bool edited);
    // Returns one as-yet-unrequested thumbnail while this window is visible.
    // Marked requested until set_thumbnail/thumbnail_failed/invalidate.
    std::string next_thumbnail();
    void thumbnail_failed(const std::string& name);
    // Returns the tensor selected by a click, or an empty string. Null renderer
    // keeps the UI testable and explicitly reports missing renderer support.
    // When offer_trace is set the toolbar shows a Trace switch; a click sets trace_clicked.
    bool offer_trace = false, trace_clicked = false;
    std::string draw(caliper_host::HostRenderer* renderer, float x, float y,
                     float width, float height, const std::string& selected);
    // The camera and exact drawing/picking viewport, available without a GPU.
    model3d::ViewState view_state() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace caliper_live
