#pragma once

#include <imgui.h>
#include <vector>
#include <string>
#include <cstdint>

struct LayerActivation {
    float mean = 0, stddev = 0, min_val = 0, max_val = 0;
    std::string shape;
    bool valid = false;
};

struct InferenceOverlay {
    std::vector<LayerActivation> layers;
    std::string sample_id;
    float probs[2] = {0, 0};
    int result_class = -1;
    bool valid = false;
};

struct ModelNode {
    std::string name;
    std::string type;
    std::vector<std::string> lines;
    std::string shape_out;
    float x, y, w, h;
    ImU32 fill;
    ImU32 border;
    ImU32 header_fill;
    int64_t param_count = 0;
};

struct ModelEdge {
    int from, to;
    std::string label;
};

struct StageGroup {
    float y_top, y_bot;
    std::string label;
};

class ModelVisualizer {
public:
    void build_repnet_graph();
    void draw(ImVec2 avail_size, const InferenceOverlay* overlay = nullptr);

    int hovered_node() const { return hovered_; }
    int selected_node() const { return selected_; }
    const ModelNode* get_node(int idx) const;
    int node_count() const { return (int)nodes_.size(); }

private:
    std::vector<ModelNode> nodes_;
    std::vector<ModelEdge> edges_;
    std::vector<StageGroup> stages_;

    int hovered_ = -1, selected_ = -1;
    bool built_ = false;
};
