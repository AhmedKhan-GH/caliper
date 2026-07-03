#pragma once

#include "verilog_parser.h"
#include <imgui.h>

struct GraphLayout {
    struct NodePos {
        float x, y;
        int layer;
    };
    std::vector<NodePos> positions;
    float total_width = 0;
    float total_height = 0;
    bool valid = false;
};

GraphLayout compute_layout(const CircuitGraph& graph, float node_w = 80, float node_h = 40, float h_gap = 40, float v_gap = 30);

ImU32 cell_type_color(const std::string& cell_type);
ImU32 power_heatmap_color(float normalized_value);
