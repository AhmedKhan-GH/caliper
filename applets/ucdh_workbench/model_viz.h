#pragma once

#include <GL/glew.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <cstdint>

struct ModelNode {
    std::string name;
    std::string type;     // conv, attention, fusion, pool, linear, dropout, input
    std::vector<std::string> lines;
    std::string shape_out;
    float x, y, w, h;
    ImVec4 color;
    ImVec4 border;
    int64_t param_count = 0;
};

struct ModelEdge {
    int from, to;
    std::string label;
};

struct StageGroup {
    float x, y, w, h;
    std::string label;
    ImVec4 color;
};

class ModelVisualizer {
public:
    ModelVisualizer();
    ~ModelVisualizer();

    void init();
    void cleanup();
    void build_repnet_graph();
    void render(int width, int height);
    void render_labels(ImVec2 canvas_pos, ImVec2 canvas_size);
    void handle_input(ImVec2 canvas_pos, ImVec2 canvas_size);
    void fit_view(int viewport_w, int viewport_h);

    GLuint texture() const { return fbo_color_; }
    bool is_initialized() const { return initialized_; }
    int hovered_node() const { return hovered_; }
    int selected_node() const { return selected_; }
    const ModelNode* get_node(int idx) const;

private:
    void ensure_fbo(int w, int h);
    void render_stage_groups(const glm::mat4& proj);
    void render_edges(const glm::mat4& proj);
    void render_nodes(const glm::mat4& proj);

    GLuint fbo_ = 0, fbo_color_ = 0;
    int fbo_w_ = 0, fbo_h_ = 0;

    GLuint node_prog_ = 0, edge_prog_ = 0;
    GLuint quad_vao_ = 0, quad_vbo_ = 0;
    GLuint line_vao_ = 0, line_vbo_ = 0;

    std::vector<ModelNode> nodes_;
    std::vector<ModelEdge> edges_;
    std::vector<StageGroup> stages_;

    float cam_x_ = 0, cam_y_ = 0, zoom_ = 1.0f;
    bool dragging_ = false;
    ImVec2 drag_start_{};
    float drag_cam_x_ = 0, drag_cam_y_ = 0;

    int hovered_ = -1, selected_ = -1;
    bool initialized_ = false;
    bool needs_fit_ = true;
};
