#include "intro_screen.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ============================================================================
// Applet registry
// ============================================================================

namespace {

struct AppletCard {
    const char* name;
    const char* tagline;
    const char* description;
    const char* tag;
    bool        available;
    AppletKind  kind;
    ImVec4      accent;
};

const AppletCard kApplets[] = {
    { "ECG Explorer",    "12-lead waveform viewer",
      "Stream 12-lead ECG recordings, run baseline-wander "
      "correction and z-score normalization, and inspect per-lead "
      "statistics in a linked multi-plot view.",
      "ECG", true,  AppletKind::ECGExplorer,
      ImVec4(0.45f, 0.72f, 1.00f, 1.0f) },      // soft blue

    { "Spectral Analyzer", "Frequency-domain view",
      "FFT and spectrogram surface for windowed segments of the "
      "signal. Compare spectral power across leads.",
      "FFT", false, AppletKind::None,
      ImVec4(0.70f, 0.62f, 1.00f, 1.0f) },      // periwinkle

    { "Rhythm Classifier", "Arrhythmia inference",
      "Run a convolutional model over ECG segments to classify "
      "rhythm. Overlays per-beat attribution.",
      "CNN", false, AppletKind::None,
      ImVec4(0.88f, 0.55f, 0.92f, 1.0f) },      // light violet

    { "Dataset Inspector", "Corpus-level overview",
      "Browse the full recording corpus with distribution plots, "
      "label histograms, and quick-filter tools.",
      "SET", false, AppletKind::None,
      ImVec4(0.55f, 0.75f, 0.95f, 1.0f) },      // silver-blue
};
constexpr int kNumApplets = (int)(sizeof(kApplets) / sizeof(kApplets[0]));

// ============================================================================
// Shader utilities
// ============================================================================

GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[intro] Shader compile error (%s):\n%s\n",
                     type == GL_VERTEX_SHADER ? "VS" : "FS", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

GLuint link_program(const char* vs_src, const char* fs_src) {
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
        std::fprintf(stderr, "[intro] Program link error:\n%s\n", log);
        glDeleteProgram(p);
        p = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ============================================================================
// GLSL
// ============================================================================

// Fullscreen quad VS reused by nebula, bright-pass, blur, and composite.
const char* VS_FSQUAD = R"(
#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

// Globe: shared vertex shader for nodes (GL_POINTS) and edges (GL_LINES).
// Transforms by model*viewProj and passes rotated-world z for front/back fade.
const char* VS_GLOBE = R"(
#version 330 core
layout(location = 0) in vec3 a_pos;

uniform mat4  u_model;
uniform mat4  u_viewProj;
uniform float u_pxScale;
uniform float u_pointSize;
uniform int   u_isPoint;

out float v_facing;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    vec4 cp    = u_viewProj * world;
    gl_Position = cp;
    v_facing = world.z;                         // camera sits at +Z → +z faces cam
    if (u_isPoint == 1) {
        gl_PointSize = clamp(u_pxScale * u_pointSize / max(cp.w, 0.1),
                             2.0, 14.0);
    }
}
)";

const char* FS_NODE = R"(
#version 330 core
in float v_facing;
out vec4 frag;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(uv, uv);
    if (d2 > 1.0) discard;
    float halo = pow(1.0 - d2, 2.0);
    float core = pow(1.0 - d2, 8.0);

    float f = smoothstep(-0.9, 0.5, v_facing);   // fade back-facing
    f = mix(0.22, 1.0, f);

    vec3 col = vec3(0.55, 0.80, 1.00) * (halo * 0.6 + core * 2.0);
    float a  = (halo * 0.7 + core * 1.0) * f;
    frag = vec4(col * f, a);
}
)";

const char* FS_EDGE = R"(
#version 330 core
in float v_facing;
out vec4 frag;

void main() {
    float f = smoothstep(-0.9, 0.5, v_facing);
    float a = mix(0.06, 0.55, f);
    vec3 col = vec3(0.28, 0.55, 0.95);
    frag = vec4(col * a, a);
}
)";

const char* FS_BRIGHT = R"(
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_scene;
uniform float u_threshold;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float f = max(lum - u_threshold, 0.0) / max(lum, 1e-4);
    frag = vec4(c * f, 1.0);
}
)";

const char* FS_BLUR = R"(
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_step;
void main() {
    float w0 = 0.227027;
    float w1 = 0.194595;
    float w2 = 0.121622;
    float w3 = 0.054054;
    float w4 = 0.016216;
    vec3 acc = texture(u_src, v_uv).rgb * w0;
    acc += texture(u_src, v_uv + u_step * 1.0).rgb * w1;
    acc += texture(u_src, v_uv - u_step * 1.0).rgb * w1;
    acc += texture(u_src, v_uv + u_step * 2.0).rgb * w2;
    acc += texture(u_src, v_uv - u_step * 2.0).rgb * w2;
    acc += texture(u_src, v_uv + u_step * 3.0).rgb * w3;
    acc += texture(u_src, v_uv - u_step * 3.0).rgb * w3;
    acc += texture(u_src, v_uv + u_step * 4.0).rgb * w4;
    acc += texture(u_src, v_uv - u_step * 4.0).rgb * w4;
    frag = vec4(acc, 1.0);
}
)";

const char* FS_COMPOSITE = R"(
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    vec3 col = scene + bloom * 1.1;

    // Filmic tonemap + approximate gamma
    col = col / (col + vec3(1.0));
    col = pow(col, vec3(1.0 / 2.2));

    // Vignette
    vec2 vp = v_uv - 0.5;
    float vig = 1.0 - dot(vp, vp) * 1.25;
    col *= clamp(vig, 0.55, 1.0);

    frag = vec4(col, 1.0);
}
)";

// ============================================================================
// FBOs
// ============================================================================

struct FBO {
    GLuint fb = 0;
    GLuint color = 0;
    GLuint depth = 0;
    int    w = 0, h = 0;
    bool   hdr = false;
    bool   has_depth = false;
};

bool create_fbo(FBO& f, int w, int h, bool hdr, bool depth) {
    f.w = w; f.h = h; f.hdr = hdr; f.has_depth = depth;

    glGenFramebuffers(1, &f.fb);
    glBindFramebuffer(GL_FRAMEBUFFER, f.fb);

    glGenTextures(1, &f.color);
    glBindTexture(GL_TEXTURE_2D, f.color);
    // HDR attachments use R11F_G11F_B10F (4 B/px) rather than RGBA16F (8 B/px):
    // all consumers read .rgb and write alpha=1, so the alpha channel and the
    // extra 5 bits/channel of mantissa are pure waste.
    if (hdr) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R11F_G11F_B10F, w, h, 0,
                     GL_RGB, GL_FLOAT, nullptr);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, f.color, 0);

    if (depth) {
        glGenRenderbuffers(1, &f.depth);
        glBindRenderbuffer(GL_RENDERBUFFER, f.depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, f.depth);
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "[intro] FBO incomplete: 0x%x\n", status);
        return false;
    }
    return true;
}

void destroy_fbo(FBO& f) {
    if (f.color) glDeleteTextures(1, &f.color);
    if (f.depth) glDeleteRenderbuffers(1, &f.depth);
    if (f.fb)    glDeleteFramebuffers(1, &f.fb);
    f = FBO();
}

} // namespace

// ============================================================================
// Scene state
// ============================================================================

struct IntroScreen::State {
    // Programs
    GLuint prog_node      = 0;   // shared VS_GLOBE + FS_NODE
    GLuint prog_edge      = 0;   // shared VS_GLOBE + FS_EDGE
    GLuint prog_bright    = 0;
    GLuint prog_blur      = 0;
    GLuint prog_composite = 0;

    // Geometry
    GLuint node_vao = 0, node_vbo = 0;
    int    num_nodes = 0;
    GLuint edge_vao = 0, edge_vbo = 0;
    int    num_edge_verts = 0;
    GLuint fs_vao = 0, fs_vbo = 0;

    // FBOs
    FBO scene;
    FBO bloom_a;
    FBO bloom_b;
    int fbo_w = 0, fbo_h = 0;

    // Time / camera
    double t_sim         = 0.0;
    double prev_time     = 0.0;
    float  camera_angle  = 0.0f;

    // UI
    AppletKind chosen_kind = AppletKind::None;

    bool ok = false;
};

// ============================================================================
// Geometry builders
// ============================================================================

namespace {

void build_fs_quad(GLuint& vao, GLuint& vbo) {
    float verts[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1,  1,
    };
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float),
                          (void*)0);
    glBindVertexArray(0);
}

// Build a unit-radius UV-sphere globe.
//   seg_lat = number of latitude segments (pole-to-pole).
//   seg_lon = number of longitude segments around the equator.
//
// Nodes sit at every (lat, lon) grid intersection plus a single pole node at
// each end. Edges are strictly meridians (constant lon, connecting adjacent
// latitudes including through the poles) and parallels (constant lat,
// wrapping around), so the pattern is fully regular.
void build_globe(GLuint& node_vao, GLuint& node_vbo,
                 GLuint& edge_vao, GLuint& edge_vbo,
                 int& out_num_nodes, int& out_num_edge_verts,
                 int seg_lat, int seg_lon) {
    std::vector<float> pos;
    auto add_node = [&](float phi, float theta) {
        float sp = std::sin(phi);
        pos.push_back(sp * std::cos(theta));
        pos.push_back(std::cos(phi));
        pos.push_back(sp * std::sin(theta));
    };

    // North pole
    const int north = 0;
    add_node(0.0f, 0.0f);

    // Interior rings 1..seg_lat-1
    std::vector<int> ring_start(seg_lat + 1);
    ring_start[0] = north;
    for (int i = 1; i < seg_lat; i++) {
        ring_start[i] = (int)(pos.size() / 3);
        float phi = (float)i * (float)M_PI / (float)seg_lat;
        for (int j = 0; j < seg_lon; j++) {
            float theta = (float)j * 2.0f * (float)M_PI / (float)seg_lon;
            add_node(phi, theta);
        }
    }

    // South pole
    const int south = (int)(pos.size() / 3);
    ring_start[seg_lat] = south;
    add_node((float)M_PI, 0.0f);

    out_num_nodes = (int)(pos.size() / 3);

    // --- Node VBO ---
    glGenVertexArrays(1, &node_vao);
    glBindVertexArray(node_vao);
    glGenBuffers(1, &node_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, node_vbo);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(float),
                 pos.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);

    // --- Edge VBO (GL_LINES pairs) ---
    std::vector<float> edges;
    auto push_edge = [&](int a, int b) {
        edges.push_back(pos[a * 3 + 0]);
        edges.push_back(pos[a * 3 + 1]);
        edges.push_back(pos[a * 3 + 2]);
        edges.push_back(pos[b * 3 + 0]);
        edges.push_back(pos[b * 3 + 1]);
        edges.push_back(pos[b * 3 + 2]);
    };

    // Meridians: north pole → ring 1 → ring 2 → … → south pole, one per lon.
    for (int j = 0; j < seg_lon; j++) {
        push_edge(north, ring_start[1] + j);
        for (int i = 1; i < seg_lat - 1; i++) {
            push_edge(ring_start[i] + j, ring_start[i + 1] + j);
        }
        push_edge(ring_start[seg_lat - 1] + j, south);
    }
    // Parallels: wrap around each interior ring.
    for (int i = 1; i < seg_lat; i++) {
        for (int j = 0; j < seg_lon; j++) {
            int a = ring_start[i] + j;
            int b = ring_start[i] + (j + 1) % seg_lon;
            push_edge(a, b);
        }
    }

    out_num_edge_verts = (int)edges.size() / 3;

    glGenVertexArrays(1, &edge_vao);
    glBindVertexArray(edge_vao);
    glGenBuffers(1, &edge_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, edge_vbo);
    glBufferData(GL_ARRAY_BUFFER, edges.size() * sizeof(float),
                 edges.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);
    glBindVertexArray(0);
}

} // namespace

// ============================================================================
// IntroScreen
// ============================================================================

bool IntroScreen::initialize() {
    s_ = new State();

    s_->prog_node      = link_program(VS_GLOBE,  FS_NODE);
    s_->prog_edge      = link_program(VS_GLOBE,  FS_EDGE);
    s_->prog_bright    = link_program(VS_FSQUAD, FS_BRIGHT);
    s_->prog_blur      = link_program(VS_FSQUAD, FS_BLUR);
    s_->prog_composite = link_program(VS_FSQUAD, FS_COMPOSITE);

    if (!s_->prog_node || !s_->prog_edge ||
        !s_->prog_bright || !s_->prog_blur || !s_->prog_composite) {
        std::fprintf(stderr, "[intro] Shader program creation failed\n");
        return false;
    }

    build_fs_quad(s_->fs_vao, s_->fs_vbo);
    build_globe(s_->node_vao, s_->node_vbo,
                s_->edge_vao, s_->edge_vbo,
                s_->num_nodes, s_->num_edge_verts,
                /*seg_lat=*/14, /*seg_lon=*/24);

    s_->fbo_w = s_->fbo_h = 0;
    s_->prev_time = glfwGetTime();
    s_->ok = true;
    return true;
}

void IntroScreen::update(GLFWwindow* /*window*/) {
    if (!s_) return;

    double now = glfwGetTime();
    double dt  = now - s_->prev_time;
    s_->prev_time = now;

    s_->t_sim        += dt;
    s_->camera_angle += (float)dt * 0.06f;
}

void IntroScreen::render_3d(int fb_w, int fb_h) {
    if (!s_ || !s_->ok || fb_w <= 0 || fb_h <= 0) return;

    // Cap internal render resolution. On 4K/5K retina the scene FBO would
    // otherwise consume 60–120+ MB before we even draw anything; the starfield
    // composites up to the default framebuffer via linear filtering with
    // no visible quality loss. Bloom runs at half of the (capped) size.
    constexpr int kMaxInternalH = 1440;
    int scene_w = fb_w, scene_h = fb_h;
    if (scene_h > kMaxInternalH) {
        scene_w = (int)((long long)fb_w * kMaxInternalH / fb_h);
        scene_h = kMaxInternalH;
    }

    if (scene_w != s_->fbo_w || scene_h != s_->fbo_h) {
        destroy_fbo(s_->scene);
        destroy_fbo(s_->bloom_a);
        destroy_fbo(s_->bloom_b);
        // No depth attachment — nebula + stars both run with depth disabled.
        create_fbo(s_->scene,   scene_w,     scene_h,     true, false);
        create_fbo(s_->bloom_a, scene_w / 2, scene_h / 2, true, false);
        create_fbo(s_->bloom_b, scene_w / 2, scene_h / 2, true, false);
        s_->fbo_w = scene_w;
        s_->fbo_h = scene_h;
    }

    // Camera + rotating model
    float aspect = (float)fb_w / (float)fb_h;
    glm::mat4 proj = glm::perspective(glm::radians(42.0f), aspect, 0.1f, 50.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.2f),
                                 glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 vp = proj * view;

    glm::mat4 model(1.0f);
    model = glm::rotate(model, (float)s_->t_sim * 0.28f,
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, 0.35f * std::sin((float)s_->t_sim * 0.18f),
                        glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, 0.35f, glm::vec3(1.0f, 0.0f, 0.0f));

    // --- Scene pass ---
    glBindFramebuffer(GL_FRAMEBUFFER, s_->scene.fb);
    glViewport(0, 0, s_->scene.w, s_->scene.h);
    glClearColor(0.02f, 0.03f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    auto set_common = [&](GLuint prog) {
        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog, "u_model"),
                           1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(prog, "u_viewProj"),
                           1, GL_FALSE, glm::value_ptr(vp));
        glUniform1f(glGetUniformLocation(prog, "u_pxScale"),
                    (float)fb_h * 0.5f);
    };

    // 1) Edges
    set_common(s_->prog_edge);
    glUniform1i(glGetUniformLocation(s_->prog_edge, "u_isPoint"), 0);
    glLineWidth(1.2f);
    glBindVertexArray(s_->edge_vao);
    glDrawArrays(GL_LINES, 0, s_->num_edge_verts);

    // 2) Nodes
    set_common(s_->prog_node);
    glUniform1i(glGetUniformLocation(s_->prog_node, "u_isPoint"), 1);
    glUniform1f(glGetUniformLocation(s_->prog_node, "u_pointSize"), 7.0f);
    glBindVertexArray(s_->node_vao);
    glDrawArrays(GL_POINTS, 0, s_->num_nodes);

    // --- Bloom: bright pass ---
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, s_->bloom_a.fb);
    glViewport(0, 0, s_->bloom_a.w, s_->bloom_a.h);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(s_->prog_bright);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s_->scene.color);
    glUniform1i(glGetUniformLocation(s_->prog_bright, "u_scene"), 0);
    glUniform1f(glGetUniformLocation(s_->prog_bright, "u_threshold"), 0.55f);
    glBindVertexArray(s_->fs_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    auto blur_pass = [&](FBO& src, FBO& dst, float dx, float dy) {
        glBindFramebuffer(GL_FRAMEBUFFER, dst.fb);
        glViewport(0, 0, dst.w, dst.h);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(s_->prog_blur);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, src.color);
        glUniform1i(glGetUniformLocation(s_->prog_blur, "u_src"), 0);
        glUniform2f(glGetUniformLocation(s_->prog_blur, "u_step"),
                    dx / (float)src.w, dy / (float)src.h);
        glBindVertexArray(s_->fs_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    };

    blur_pass(s_->bloom_a, s_->bloom_b, 1.0f, 0.0f);
    blur_pass(s_->bloom_b, s_->bloom_a, 0.0f, 1.0f);
    blur_pass(s_->bloom_a, s_->bloom_b, 1.0f, 0.0f);
    blur_pass(s_->bloom_b, s_->bloom_a, 0.0f, 1.0f);

    // --- Composite ---
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fb_w, fb_h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(s_->prog_composite);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s_->scene.color);
    glUniform1i(glGetUniformLocation(s_->prog_composite, "u_scene"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, s_->bloom_a.color);
    glUniform1i(glGetUniformLocation(s_->prog_composite, "u_bloom"), 1);
    glBindVertexArray(s_->fs_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(0);
    glUseProgram(0);
}

// ============================================================================
// UI
// ============================================================================

namespace {

// --- Applet row -------------------------------------------------------------
// Returns true iff clicked AND the applet is launchable.
bool draw_applet_row(int idx, const AppletCard& a, float row_h) {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x - 6.0f; // room for scrollbar

    ImGui::PushID(idx);
    ImGui::InvisibleButton("##row", ImVec2(w, row_h));
    bool hov = ImGui::IsItemHovered();
    bool clicked = a.available && ImGui::IsItemClicked();
    ImGui::PopID();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p0 = pos;
    ImVec2 p1 = ImVec2(pos.x + w, pos.y + row_h);

    ImU32 bg_base = IM_COL32( 8, 12, 24, 140);
    ImU32 bg_hov  = IM_COL32(22, 28, 48, 220);
    ImU32 bg = hov ? bg_hov : bg_base;
    dl->AddRectFilled(p0, p1, bg, 6.0f);

    ImU32 accent_u = IM_COL32(
        (int)(a.accent.x * 255), (int)(a.accent.y * 255),
        (int)(a.accent.z * 255),
        a.available ? 255 : 110);

    // Left accent stripe
    dl->AddRectFilled(ImVec2(p0.x, p0.y + 4.0f),
                      ImVec2(p0.x + 3.0f, p1.y - 4.0f),
                      accent_u, 2.0f);

    // Border on hover
    if (hov) {
        dl->AddRect(p0, p1,
                    IM_COL32((int)(a.accent.x * 255),
                             (int)(a.accent.y * 255),
                             (int)(a.accent.z * 255),
                             a.available ? 220 : 140),
                    6.0f, 0, 1.5f);
    }

    // Tag glyph (left)
    dl->AddText(nullptr, 19.0f, ImVec2(p0.x + 14.0f, p0.y + 12.0f),
                a.available ? accent_u : IM_COL32(130, 140, 160, 200),
                a.tag);

    // Name
    dl->AddText(nullptr, 15.0f, ImVec2(p0.x + 64.0f, p0.y + 10.0f),
                a.available ? IM_COL32(240, 248, 255, 255)
                            : IM_COL32(170, 180, 200, 220),
                a.name);

    // Tagline
    dl->AddText(nullptr, 11.0f, ImVec2(p0.x + 64.0f, p0.y + 31.0f),
                IM_COL32(140, 160, 195, 220), a.tagline);

    // Status chip on the right
    const char* chip = a.available ? "READY" : "SOON";
    ImU32 chip_col = a.available ? IM_COL32(120, 180, 255, 255)
                                 : IM_COL32(210, 170, 110, 255);
    float chip_w = a.available ? 44.0f : 38.0f;
    ImVec2 cp0(p1.x - chip_w - 10.0f, p0.y + 12.0f);
    ImVec2 cp1(p1.x - 10.0f,           p0.y + 27.0f);
    dl->AddRect(cp0, cp1, chip_col, 3.0f);
    dl->AddText(nullptr, 10.0f,
                ImVec2(cp0.x + 6.0f, cp0.y + 2.0f), chip_col, chip);

    return clicked;
}

} // namespace

void IntroScreen::draw_ui(int /*win_w*/, int /*win_h*/) {
    if (!s_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,  ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##IntroOverlay", nullptr,
        ImGuiWindowFlags_NoTitleBar  | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove      | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();

    // ── Centered CALIPER title (default font, scaled) ──
    ImGui::SetWindowFontScale(7.0f);
    const char* title = "C A L I P E R";
    ImVec2 t_sz = ImGui::CalcTextSize(title);
    float title_y = (ws.y - t_sz.y) * 0.5f - t_sz.y * 0.35f;
    ImGui::SetCursorPos(ImVec2((ws.x - t_sz.x) * 0.5f, title_y));
    ImGui::TextColored(ImVec4(0.80f, 0.90f, 1.00f, 1.00f), "%s", title);

    ImGui::SetWindowFontScale(1.6f);
    const char* sub = "precision signal instrumentation";
    ImVec2 s_sz = ImGui::CalcTextSize(sub);
    ImGui::SetCursorPos(ImVec2((ws.x - s_sz.x) * 0.5f,
                               title_y + t_sz.y + 18.0f));
    ImGui::TextColored(ImVec4(0.55f, 0.68f, 0.88f, 0.85f), "%s", sub);
    ImGui::SetWindowFontScale(1.0f);

    // ── Left applet scroller ──
    float col_x = 24.0f;
    float col_y = 28.0f;
    float col_w = 320.0f;
    float col_h = ws.y - 56.0f;

    // Header + divider (drawn via draw list so we don't disturb cursor flow)
    dl->AddText(nullptr, 12.0f,
                ImVec2(wp.x + col_x + 6.0f, wp.y + col_y),
                IM_COL32(130, 160, 210, 230), "INSTRUMENTS");
    dl->AddLine(ImVec2(wp.x + col_x + 6.0f,         wp.y + col_y + 18.0f),
                ImVec2(wp.x + col_x + col_w - 6.0f, wp.y + col_y + 18.0f),
                IM_COL32(70, 90, 130, 140), 1.0f);

    ImGui::SetCursorPos(ImVec2(col_x, col_y + 28.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg,     ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab,
                          ImVec4(0.25f, 0.30f, 0.45f, 0.60f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 8.0f));

    ImGui::BeginChild("##applet_scroll",
                      ImVec2(col_w, col_h - 28.0f),
                      false,
                      ImGuiWindowFlags_NoBackground);

    const float row_h = 60.0f;
    for (int i = 0; i < kNumApplets; i++) {
        if (draw_applet_row(i, kApplets[i], row_h)) {
            s_->chosen_kind = kApplets[i].kind;
            launch_requested_ = true;
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void IntroScreen::cleanup() {
    if (!s_) return;

    if (s_->prog_node)      glDeleteProgram(s_->prog_node);
    if (s_->prog_edge)      glDeleteProgram(s_->prog_edge);
    if (s_->prog_bright)    glDeleteProgram(s_->prog_bright);
    if (s_->prog_blur)      glDeleteProgram(s_->prog_blur);
    if (s_->prog_composite) glDeleteProgram(s_->prog_composite);

    if (s_->node_vbo) glDeleteBuffers(1, &s_->node_vbo);
    if (s_->node_vao) glDeleteVertexArrays(1, &s_->node_vao);
    if (s_->edge_vbo) glDeleteBuffers(1, &s_->edge_vbo);
    if (s_->edge_vao) glDeleteVertexArrays(1, &s_->edge_vao);
    if (s_->fs_vbo)   glDeleteBuffers(1, &s_->fs_vbo);
    if (s_->fs_vao)   glDeleteVertexArrays(1, &s_->fs_vao);

    destroy_fbo(s_->scene);
    destroy_fbo(s_->bloom_a);
    destroy_fbo(s_->bloom_b);

    delete s_;
    s_ = nullptr;
}

AppletKind IntroScreen::selected_applet() const {
    return s_ ? s_->chosen_kind : AppletKind::None;
}
