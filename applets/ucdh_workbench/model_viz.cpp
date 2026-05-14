#include "model_viz.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// SHADERS
// ============================================================================

static const char* VS_NODE = R"(
#version 330 core
layout(location=0) in vec2 a_pos;

uniform vec4  u_rect;
uniform mat4  u_proj;

out vec2 v_uv;
out vec2 v_size;

void main() {
    v_uv   = a_pos;
    v_size = u_rect.zw;
    vec2 world = u_rect.xy + a_pos * u_rect.zw;
    gl_Position = u_proj * vec4(world, 0.0, 1.0);
}
)";

static const char* FS_NODE = R"(
#version 330 core
in vec2 v_uv;
in vec2 v_size;

uniform vec4  u_fill;
uniform vec4  u_border_color;
uniform float u_radius;
uniform float u_border_width;
uniform float u_hovered;
uniform float u_selected;

out vec4 frag;

float roundedBoxSDF(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec2  p    = (v_uv - 0.5) * v_size;
    vec2  half = v_size * 0.5;
    float d    = roundedBoxSDF(p, half, u_radius);

    float aa    = 1.5;
    float alpha = 1.0 - smoothstep(-aa, aa, d);

    float border_inner = -u_border_width;
    float is_border    = smoothstep(border_inner - aa, border_inner + aa, d);

    vec4 fill = u_fill;
    if (u_hovered > 0.5)
        fill = mix(fill, vec4(1.0), 0.12);

    vec4 border = u_border_color;
    if (u_selected > 0.5)
        border = vec4(1.0, 0.85, 0.3, 1.0);

    vec4 color = mix(fill, border, is_border);
    color.a *= alpha;
    if (color.a < 0.01) discard;
    frag = color;
}
)";

static const char* VS_EDGE = R"(
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;

uniform mat4 u_proj;
out vec4 v_color;

void main() {
    gl_Position = u_proj * vec4(a_pos, 0.0, 1.0);
    v_color = a_color;
}
)";

static const char* FS_EDGE = R"(
#version 330 core
in vec4 v_color;
out vec4 frag;

void main() { frag = v_color; }
)";

// ============================================================================
// GL HELPERS
// ============================================================================

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[model_viz] Shader compile error (%s):\n%s\n",
                     type == GL_VERTEX_SHADER ? "VS" : "FS", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint link_program(const char* vs_src, const char* fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[model_viz] Link error:\n%s\n", log);
        glDeleteProgram(p);
        p = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ============================================================================
// LAYOUT CONSTANTS
// ============================================================================

static constexpr float NODE_W       = 340.0f;
static constexpr float NODE_H       = 56.0f;
static constexpr float NODE_GAP     = 28.0f;
static constexpr float GRAPH_PAD    = 60.0f;
static constexpr float NODE_RADIUS  = 10.0f;
static constexpr float BORDER_W     = 2.0f;
static constexpr float STAGE_GAP    = 20.0f;
static constexpr float STAGE_MARGIN = 16.0f;
static constexpr float ARROW_W      = 5.0f;
static constexpr float ARROW_H      = 9.0f;

// ============================================================================
// NODE COLORS
// ============================================================================

static ImVec4 fill_for(const std::string& t) {
    if (t == "conv")      return {0.15f, 0.36f, 0.82f, 1.0f};
    if (t == "attention")  return {0.78f, 0.44f, 0.02f, 1.0f};
    if (t == "fusion")     return {0.02f, 0.52f, 0.38f, 1.0f};
    if (t == "pool")       return {0.44f, 0.20f, 0.80f, 1.0f};
    if (t == "linear")     return {0.28f, 0.25f, 0.78f, 1.0f};
    if (t == "dropout")    return {0.26f, 0.30f, 0.36f, 1.0f};
    return {0.22f, 0.27f, 0.34f, 1.0f};
}

static ImVec4 border_for(const std::string& t) {
    if (t == "conv")      return {0.23f, 0.51f, 0.96f, 1.0f};
    if (t == "attention")  return {0.96f, 0.62f, 0.07f, 1.0f};
    if (t == "fusion")     return {0.06f, 0.73f, 0.51f, 1.0f};
    if (t == "pool")       return {0.55f, 0.36f, 0.96f, 1.0f};
    if (t == "linear")     return {0.39f, 0.40f, 0.95f, 1.0f};
    if (t == "dropout")    return {0.42f, 0.48f, 0.55f, 1.0f};
    return {0.45f, 0.52f, 0.60f, 1.0f};
}

// ============================================================================
// LIFECYCLE
// ============================================================================

ModelVisualizer::ModelVisualizer()  = default;
ModelVisualizer::~ModelVisualizer() { cleanup(); }

void ModelVisualizer::init() {
    if (initialized_) return;

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::fprintf(stderr, "[model_viz] glewInit failed: %s\n",
                     glewGetErrorString(err));
        return;
    }

    node_prog_ = link_program(VS_NODE, FS_NODE);
    edge_prog_ = link_program(VS_EDGE, FS_EDGE);
    if (!node_prog_ || !edge_prog_) {
        std::fprintf(stderr, "[model_viz] Failed to compile shaders\n");
        return;
    }

    float quad[] = {0,0, 1,0, 1,1, 0,0, 1,1, 0,1};
    glGenVertexArrays(1, &quad_vao_);
    glGenBuffers(1, &quad_vbo_);
    glBindVertexArray(quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    glGenVertexArrays(1, &line_vao_);
    glGenBuffers(1, &line_vbo_);
    glBindVertexArray(line_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    initialized_ = true;
    build_repnet_graph();
}

void ModelVisualizer::cleanup() {
    if (node_prog_) { glDeleteProgram(node_prog_); node_prog_ = 0; }
    if (edge_prog_) { glDeleteProgram(edge_prog_); edge_prog_ = 0; }
    if (quad_vao_)  { glDeleteVertexArrays(1, &quad_vao_); quad_vao_ = 0; }
    if (quad_vbo_)  { glDeleteBuffers(1, &quad_vbo_);      quad_vbo_ = 0; }
    if (line_vao_)  { glDeleteVertexArrays(1, &line_vao_); line_vao_ = 0; }
    if (line_vbo_)  { glDeleteBuffers(1, &line_vbo_);      line_vbo_ = 0; }
    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &fbo_color_);
        fbo_ = 0; fbo_color_ = 0;
    }
    initialized_ = false;
}

// ============================================================================
// FBO
// ============================================================================

void ModelVisualizer::ensure_fbo(int w, int h) {
    if (fbo_ && fbo_w_ == w && fbo_h_ == h) return;

    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &fbo_color_);
    }
    fbo_w_ = w;
    fbo_h_ = h;

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    glGenTextures(1, &fbo_color_);
    glBindTexture(GL_TEXTURE_2D, fbo_color_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, fbo_color_, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ============================================================================
// GRAPH CONSTRUCTION
// ============================================================================

void ModelVisualizer::build_repnet_graph() {
    nodes_.clear();
    edges_.clear();
    stages_.clear();

    float cx = GRAPH_PAD + NODE_W * 0.5f;
    float y  = GRAPH_PAD;

    auto add = [&](const char* name, const char* type, const char* detail,
                   int64_t params, float pre_gap = 0.0f) -> int {
        y += pre_gap;
        int idx = (int)nodes_.size();
        nodes_.push_back({name, type, detail,
                          cx - NODE_W * 0.5f, y, NODE_W, NODE_H,
                          fill_for(type), border_for(type), params});
        y += NODE_H + NODE_GAP;
        return idx;
    };

    int n_input = add("Input", "input", "(B, 12, T)  12-lead ECG", 0);

    // ── Stage 0 ──
    float s0_top = y;
    int n_c0 = add("PerLeadConvBlock", "conv",
                   "12x Conv1d(1->32, k=7) + BN + ReLU + Skip + Pool/2",
                   7744, STAGE_GAP);
    int n_a0 = add("CrossLeadAttention", "attention",
                   "MHA(dim=32, heads=4) + LayerNorm + Gate", 5344);
    float s0_bot = y - NODE_GAP;

    // ── Stage 1 ──
    float s1_top = y;
    int n_c1 = add("PerLeadConvBlock", "conv",
                   "12x Conv1d(32->64, k=5) + BN + ReLU + Skip + Pool/2",
                   33344, STAGE_GAP);
    int n_a1 = add("CrossLeadAttention", "attention",
                   "MHA(dim=64, heads=4) + LayerNorm + Gate", 20928);
    float s1_bot = y - NODE_GAP;

    // ── Stage 2 ──
    float s2_top = y;
    int n_c2 = add("PerLeadConvBlock", "conv",
                   "12x Conv1d(64->128, k=3) + BN + ReLU + Skip + Pool/2",
                   83072, STAGE_GAP);
    int n_a2 = add("CrossLeadAttention", "attention",
                   "MHA(dim=128, heads=4) + LayerNorm + Gate", 82816);
    float s2_bot = y - NODE_GAP;

    // ── Head ──
    int n_fuse = add("Fusion", "fusion",
                     "Conv1d(1536->128, k=1) + BN + ReLU", 196992, STAGE_GAP);
    int n_gap  = add("Global Avg Pool", "pool", "AdaptiveAvgPool1d(1)", 0);
    int n_drop = add("Dropout", "dropout", "p = 0.064", 0);
    int n_fc   = add("Linear", "linear", "128 -> 2  (PE vs Normal)", 258);
    int n_out  = add("Output", "input", "Logits (B, 2)", 0);

    // ── Edges ──
    auto edge = [&](int a, int b) { edges_.push_back({a, b}); };
    edge(n_input, n_c0);
    edge(n_c0, n_a0); edge(n_a0, n_c1);
    edge(n_c1, n_a1); edge(n_a1, n_c2);
    edge(n_c2, n_a2); edge(n_a2, n_fuse);
    edge(n_fuse, n_gap); edge(n_gap, n_drop);
    edge(n_drop, n_fc);  edge(n_fc, n_out);

    // ── Stage groups ──
    float sg_x = GRAPH_PAD - STAGE_MARGIN;
    float sg_w = NODE_W + STAGE_MARGIN * 2;
    ImVec4 sg_col = {0.25f, 0.35f, 0.55f, 0.12f};
    stages_.push_back({sg_x, s0_top - STAGE_MARGIN, sg_w,
                       s0_bot - s0_top + STAGE_MARGIN * 2, "Stage 0", sg_col});
    stages_.push_back({sg_x, s1_top - STAGE_MARGIN, sg_w,
                       s1_bot - s1_top + STAGE_MARGIN * 2, "Stage 1", sg_col});
    stages_.push_back({sg_x, s2_top - STAGE_MARGIN, sg_w,
                       s2_bot - s2_top + STAGE_MARGIN * 2, "Stage 2", sg_col});

    cam_x_ = cx;
    cam_y_ = y * 0.5f;
    needs_fit_ = true;
}

// ============================================================================
// RENDERING
// ============================================================================

void ModelVisualizer::render(int width, int height) {
    if (!initialized_ || width <= 0 || height <= 0) return;

    if (needs_fit_) {
        fit_view(width, height);
        needs_fit_ = false;
    }

    ensure_fbo(width, height);

    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    GLint prev_vp[4];
    glGetIntegerv(GL_VIEWPORT, prev_vp);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, width, height);
    glClearColor(0.07f, 0.07f, 0.11f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float hw = (float)width  / (2.0f * zoom_);
    float hh = (float)height / (2.0f * zoom_);
    glm::mat4 proj = glm::ortho(cam_x_ - hw, cam_x_ + hw,
                                cam_y_ + hh, cam_y_ - hh,
                                -1.0f, 1.0f);

    render_stage_groups(proj);
    render_edges(proj);
    render_nodes(proj);

    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
    glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
}

void ModelVisualizer::render_stage_groups(const glm::mat4& proj) {
    glUseProgram(node_prog_);
    glBindVertexArray(quad_vao_);
    glUniformMatrix4fv(glGetUniformLocation(node_prog_, "u_proj"),
                       1, GL_FALSE, glm::value_ptr(proj));
    glUniform1f(glGetUniformLocation(node_prog_, "u_radius"), 14.0f);
    glUniform1f(glGetUniformLocation(node_prog_, "u_border_width"), 1.0f);
    glUniform1f(glGetUniformLocation(node_prog_, "u_hovered"), 0.0f);
    glUniform1f(glGetUniformLocation(node_prog_, "u_selected"), 0.0f);

    for (const auto& sg : stages_) {
        glUniform4f(glGetUniformLocation(node_prog_, "u_rect"),
                    sg.x, sg.y, sg.w, sg.h);
        glUniform4f(glGetUniformLocation(node_prog_, "u_fill"),
                    sg.color.x, sg.color.y, sg.color.z, sg.color.w);
        glUniform4f(glGetUniformLocation(node_prog_, "u_border_color"),
                    sg.color.x * 1.5f, sg.color.y * 1.5f,
                    sg.color.z * 1.5f, sg.color.w * 1.8f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

void ModelVisualizer::render_edges(const glm::mat4& proj) {
    if (edges_.empty()) return;

    struct V { float x, y, r, g, b, a; };
    std::vector<V> lines, tris;
    lines.reserve(edges_.size() * 2);
    tris.reserve(edges_.size() * 3);

    const float cr = 0.40f, cg = 0.46f, cb = 0.55f, ca = 0.65f;

    for (const auto& e : edges_) {
        const auto& src = nodes_[e.from];
        const auto& dst = nodes_[e.to];
        float x0 = src.x + src.w * 0.5f, y0 = src.y + src.h;
        float x1 = dst.x + dst.w * 0.5f, y1 = dst.y;

        lines.push_back({x0, y0, cr, cg, cb, ca});
        lines.push_back({x1, y1 - ARROW_H, cr, cg, cb, ca});

        tris.push_back({x1,           y1,           cr, cg, cb, ca});
        tris.push_back({x1 - ARROW_W, y1 - ARROW_H, cr, cg, cb, ca});
        tris.push_back({x1 + ARROW_W, y1 - ARROW_H, cr, cg, cb, ca});
    }

    glUseProgram(edge_prog_);
    glUniformMatrix4fv(glGetUniformLocation(edge_prog_, "u_proj"),
                       1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(line_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo_);

    if (!lines.empty()) {
        glBufferData(GL_ARRAY_BUFFER,
                     (GLsizeiptr)(lines.size() * sizeof(V)),
                     lines.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, (GLsizei)lines.size());
    }
    if (!tris.empty()) {
        glBufferData(GL_ARRAY_BUFFER,
                     (GLsizeiptr)(tris.size() * sizeof(V)),
                     tris.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)tris.size());
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

void ModelVisualizer::render_nodes(const glm::mat4& proj) {
    glUseProgram(node_prog_);
    glBindVertexArray(quad_vao_);
    glUniformMatrix4fv(glGetUniformLocation(node_prog_, "u_proj"),
                       1, GL_FALSE, glm::value_ptr(proj));
    glUniform1f(glGetUniformLocation(node_prog_, "u_radius"), NODE_RADIUS);
    glUniform1f(glGetUniformLocation(node_prog_, "u_border_width"), BORDER_W);

    for (int i = 0; i < (int)nodes_.size(); i++) {
        const auto& n = nodes_[i];
        glUniform4f(glGetUniformLocation(node_prog_, "u_rect"),
                    n.x, n.y, n.w, n.h);
        glUniform4f(glGetUniformLocation(node_prog_, "u_fill"),
                    n.color.x, n.color.y, n.color.z, n.color.w);
        glUniform4f(glGetUniformLocation(node_prog_, "u_border_color"),
                    n.border.x, n.border.y, n.border.z, n.border.w);
        glUniform1f(glGetUniformLocation(node_prog_, "u_hovered"),
                    i == hovered_ ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(node_prog_, "u_selected"),
                    i == selected_ ? 1.0f : 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

// ============================================================================
// TEXT LABELS (rendered via ImGui overlay, not in FBO)
// ============================================================================

void ModelVisualizer::render_labels(ImVec2 cp, ImVec2 cs) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(cp, ImVec2(cp.x + cs.x, cp.y + cs.y), true);

    auto to_screen = [&](float gx, float gy) -> ImVec2 {
        return {cp.x + cs.x * 0.5f + (gx - cam_x_) * zoom_,
                cp.y + cs.y * 0.5f + (gy - cam_y_) * zoom_};
    };

    // Stage labels
    for (const auto& sg : stages_) {
        ImVec2 p = to_screen(sg.x + 6.0f, sg.y + 4.0f);
        if (zoom_ > 0.35f)
            dl->AddText(p, IM_COL32(160, 180, 220, 100), sg.label.c_str());
    }

    // Node labels
    for (const auto& n : nodes_) {
        ImVec2 tl = to_screen(n.x, n.y);
        float sw = n.w * zoom_;
        float sh = n.h * zoom_;

        if (tl.x + sw < cp.x || tl.x > cp.x + cs.x) continue;
        if (tl.y + sh < cp.y || tl.y > cp.y + cs.y) continue;
        if (sh < 18.0f) continue;

        float pad = 10.0f;
        dl->AddText(ImVec2(tl.x + pad, tl.y + 6.0f),
                    IM_COL32(255, 255, 255, 230), n.name.c_str());

        if (sh >= 38.0f && !n.detail.empty()) {
            dl->AddText(ImVec2(tl.x + pad, tl.y + 22.0f),
                        IM_COL32(200, 205, 215, 150), n.detail.c_str());
        }

        if (sh >= 50.0f && n.param_count > 0) {
            char buf[64];
            if (n.param_count >= 1000000)
                std::snprintf(buf, sizeof(buf), "%.1fM params", n.param_count / 1e6);
            else if (n.param_count >= 1000)
                std::snprintf(buf, sizeof(buf), "%.1fK params", n.param_count / 1e3);
            else
                std::snprintf(buf, sizeof(buf), "%lld params", (long long)n.param_count);

            ImVec2 ts = ImGui::CalcTextSize(buf);
            dl->AddText(ImVec2(tl.x + sw - ts.x - pad, tl.y + 6.0f),
                        IM_COL32(180, 190, 210, 120), buf);
        }
    }

    dl->PopClipRect();
}

// ============================================================================
// INPUT
// ============================================================================

void ModelVisualizer::handle_input(ImVec2 cp, ImVec2 cs) {
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 m = io.MousePos;

    bool in = m.x >= cp.x && m.x < cp.x + cs.x &&
              m.y >= cp.y && m.y < cp.y + cs.y;

    if (!in && !dragging_) { hovered_ = -1; return; }

    float mx = cam_x_ + (m.x - cp.x - cs.x * 0.5f) / zoom_;
    float my = cam_y_ + (m.y - cp.y - cs.y * 0.5f) / zoom_;

    hovered_ = -1;
    for (int i = (int)nodes_.size() - 1; i >= 0; i--) {
        const auto& n = nodes_[i];
        if (mx >= n.x && mx < n.x + n.w && my >= n.y && my < n.y + n.h) {
            hovered_ = i;
            break;
        }
    }

    if (in && ImGui::IsMouseClicked(0) && hovered_ < 0)
        selected_ = -1;
    else if (in && ImGui::IsMouseClicked(0) && hovered_ >= 0)
        selected_ = hovered_;

    // Pan: middle mouse or Alt+left
    bool pan_btn = ImGui::IsMouseDown(2) ||
                   (ImGui::IsMouseDown(0) && io.KeyAlt);
    if (pan_btn && !dragging_ && in) {
        dragging_   = true;
        drag_start_ = m;
        drag_cam_x_ = cam_x_;
        drag_cam_y_ = cam_y_;
    }
    if (dragging_) {
        if (pan_btn) {
            cam_x_ = drag_cam_x_ - (m.x - drag_start_.x) / zoom_;
            cam_y_ = drag_cam_y_ - (m.y - drag_start_.y) / zoom_;
        } else {
            dragging_ = false;
        }
    }

    // Zoom with scroll wheel
    if (in && std::abs(io.MouseWheel) > 0.01f) {
        float old = zoom_;
        zoom_ *= (1.0f + io.MouseWheel * 0.12f);
        zoom_  = std::clamp(zoom_, 0.15f, 5.0f);
        float f = 1.0f - old / zoom_;
        cam_x_ += (mx - cam_x_) * f;
        cam_y_ += (my - cam_y_) * f;
    }
}

// ============================================================================
// FIT VIEW
// ============================================================================

void ModelVisualizer::fit_view(int vw, int vh) {
    if (nodes_.empty() || vw <= 0 || vh <= 0) return;

    float min_x = nodes_[0].x, max_x = nodes_[0].x + nodes_[0].w;
    float min_y = nodes_[0].y, max_y = nodes_[0].y + nodes_[0].h;
    for (const auto& n : nodes_) {
        min_x = std::min(min_x, n.x);
        max_x = std::max(max_x, n.x + n.w);
        min_y = std::min(min_y, n.y);
        max_y = std::max(max_y, n.y + n.h);
    }

    float gw = max_x - min_x + GRAPH_PAD * 2;
    float gh = max_y - min_y + GRAPH_PAD * 2;

    cam_x_ = (min_x + max_x) * 0.5f;
    cam_y_ = (min_y + max_y) * 0.5f;
    zoom_  = std::min((float)vw / gw, (float)vh / gh) * 0.92f;
    zoom_  = std::clamp(zoom_, 0.15f, 5.0f);
}

// ============================================================================
// QUERY
// ============================================================================

const ModelNode* ModelVisualizer::get_node(int idx) const {
    if (idx < 0 || idx >= (int)nodes_.size()) return nullptr;
    return &nodes_[idx];
}
