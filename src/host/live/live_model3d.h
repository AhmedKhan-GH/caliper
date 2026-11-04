#pragma once

#include "live/model3d_geometry.h"

#include <memory>
#include <string>
#include <vector>

namespace caliper_host { class HostRenderer; }
namespace caliper_live {

using ModelTensor = model3d::Tensor;

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
    std::string draw(caliper_host::HostRenderer* renderer, float x, float y,
                     float width, float height, const std::string& selected);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace caliper_live
