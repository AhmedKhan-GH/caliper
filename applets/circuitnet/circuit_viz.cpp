#include "circuit_viz.h"

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <cmath>

GraphLayout compute_layout(const CircuitGraph& graph, float node_w, float node_h, float h_gap, float v_gap) {
    GraphLayout layout;
    if (graph.gates.empty()) return layout;

    int n = (int)graph.gates.size();
    layout.positions.resize(n);

    // Topological layering via BFS from sources
    std::vector<int> in_degree(n, 0);
    std::vector<std::vector<int>> adj(n);

    for (auto& e : graph.edges) {
        if (e.from_gate >= 0 && e.from_gate < n && e.to_gate >= 0 && e.to_gate < n) {
            adj[e.from_gate].push_back(e.to_gate);
            in_degree[e.to_gate]++;
        }
    }

    std::queue<int> q;
    std::vector<int> layer(n, 0);

    for (int i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            q.push(i);
            layer[i] = 0;
        }
    }

    int max_layer = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            layer[v] = std::max(layer[v], layer[u] + 1);
            max_layer = std::max(max_layer, layer[v]);
            in_degree[v]--;
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }
    }

    // Count nodes per layer
    std::vector<int> layer_count(max_layer + 1, 0);
    for (int i = 0; i < n; i++) {
        layer_count[layer[i]]++;
    }

    // Assign positions
    std::vector<int> layer_idx(max_layer + 1, 0);
    float max_width = 0;

    for (int i = 0; i < n; i++) {
        int l = layer[i];
        int idx = layer_idx[l]++;
        int count = layer_count[l];

        layout.positions[i].layer = l;
        layout.positions[i].x = l * (node_w + h_gap);
        layout.positions[i].y = idx * (node_h + v_gap) - (count - 1) * (node_h + v_gap) * 0.5f;

        max_width = std::max(max_width, layout.positions[i].x + node_w);
    }

    layout.total_width = max_width;

    float min_y = 1e9f, max_y = -1e9f;
    for (auto& p : layout.positions) {
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y + node_h);
    }
    layout.total_height = max_y - min_y;

    // Shift so min_y = 0
    for (auto& p : layout.positions) {
        p.y -= min_y;
    }

    layout.valid = true;
    return layout;
}

ImU32 cell_type_color(const std::string& cell_type) {
    // Color by gate family
    if (cell_type.find("NAND") != std::string::npos) return IM_COL32(255, 100, 100, 200);
    if (cell_type.find("NOR") != std::string::npos)  return IM_COL32(100, 200, 255, 200);
    if (cell_type.find("AND") != std::string::npos)  return IM_COL32(100, 255, 100, 200);
    if (cell_type.find("OR") != std::string::npos)   return IM_COL32(180, 100, 255, 200);
    if (cell_type.find("XOR") != std::string::npos)  return IM_COL32(255, 200, 50, 200);
    if (cell_type.find("INV") != std::string::npos)  return IM_COL32(200, 200, 200, 200);
    if (cell_type.find("BUF") != std::string::npos)  return IM_COL32(150, 255, 150, 200);
    if (cell_type.find("DFF") != std::string::npos)  return IM_COL32(255, 180, 50, 200);
    if (cell_type.find("MX") != std::string::npos)   return IM_COL32(50, 200, 200, 200);
    return IM_COL32(180, 180, 180, 200);
}

ImU32 power_heatmap_color(float normalized_value) {
    float t = std::clamp(normalized_value, 0.0f, 1.0f);
    // Blue -> Green -> Yellow -> Red
    int r, g, b;
    if (t < 0.33f) {
        float s = t / 0.33f;
        r = 0;
        g = (int)(s * 200);
        b = (int)((1 - s) * 255);
    } else if (t < 0.66f) {
        float s = (t - 0.33f) / 0.33f;
        r = (int)(s * 255);
        g = 200;
        b = 0;
    } else {
        float s = (t - 0.66f) / 0.34f;
        r = 255;
        g = (int)((1 - s) * 200);
        b = 0;
    }
    return IM_COL32(r, g, b, 220);
}
