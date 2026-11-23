#pragma once
#include "scene/camera.h"
#include "scene/scene.h"

namespace caliper_scene {
struct LabelAnchor {
    EntityId entity;
    Primitive primitive = Primitive::Points;
    uint64_t item = 0;
    float x = 0, y = 0;                         // physical pixels, top-left origin
    float depth = 0;                            // distance from the eye; 0 for flat overlays
    std::string text;
    uint8_t pin = 0;                            // 0 placed; pinned at the anchor with the text's 1 start, 2 end, 3 middle
};

// Label anchors of visible CPU rows, nearest first.
std::vector<LabelAnchor> project_labels(const SceneTree& tree, const ViewCamera& camera, uint32_t width, uint32_t height);
}
