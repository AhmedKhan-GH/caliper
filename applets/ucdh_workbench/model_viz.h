#pragma once

#include <imgui.h>
#include <vector>
#include <string>
#include <cstdint>

struct ModelNode {
    std::string name;
    std::string type;     // conv, attention, fusion, pool, linear, dropout, input
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
    void draw(ImVec2 avail_size);

    int hovered_node() const { return hovered_; }
    int selected_node() const { return selected_; }
    const ModelNode* get_node(int idx) const;
    const std::vector<ModelNode>& nodes() const { return nodes_; }

private:
    std::vector<ModelNode> nodes_;
    std::vector<ModelEdge> edges_;
    std::vector<StageGroup> stages_;

    float scroll_y_ = 0;
    int hovered_ = -1, selected_ = -1;
    bool built_ = false;
};
