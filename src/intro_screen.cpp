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

// Hopf fiber: per-vertex colored line segments stored as 4D points on S³
// plus RGB. The VS applies a time-varying 4D rotation inside S³, then
// stereographic-projects to R³. This gives the classic "fibers flow through
// each other" animation since a generic SO(4) element is not a Hopf
// symmetry — it genuinely permutes fibers.
const char* VS_HOPF = R"(
#version 330 core
layout(location = 0) in vec4 a_s3;
layout(location = 1) in vec3 a_color;

uniform mat4  u_model;
uniform mat4  u_viewProj;
uniform mat4  u_s3_rot;                         // rotation applied in S³
uniform float u_scale;                          // post-projection scale

out vec3  v_color;
out float v_facing;
out float v_safe;                               // 1 = far from singularity, 0 = at it

void main() {
    vec4 p = u_s3_rot * a_s3;                   // rotate on S³
    float denom = 1.0 - p.w;                    // stereographic pole at w=+1
    float sd = sign(denom) * max(abs(denom), 0.035);
    vec3  r3 = p.xyz / sd * u_scale;

    vec4 world = u_model * vec4(r3, 1.0);
    gl_Position = u_viewProj * world;
    v_color  = a_color;
    v_facing = world.z;                         // camera at +Z ⇒ +z faces cam
    v_safe   = smoothstep(0.04, 0.22, abs(denom));
}
)";

const char* FS_HOPF = R"(
#version 330 core
in vec3  v_color;
in float v_facing;
in float v_safe;
out vec4 frag;

void main() {
    float f = smoothstep(-1.4, 0.9, v_facing);
    float intensity = mix(0.18, 1.45, f) * v_safe;
    frag = vec4(v_color * intensity, intensity);
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
    GLuint prog_fiber     = 0;   // VS_HOPF + FS_HOPF
    GLuint prog_bright    = 0;
    GLuint prog_blur      = 0;
    GLuint prog_composite = 0;

    // Geometry
    GLuint fiber_vao = 0, fiber_vbo = 0;
    int    num_fiber_verts = 0;
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

void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    float hp = h * 6.0f;
    float c  = v * s;
    float x  = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    float m  = v - c;
    float rp = 0, gp = 0, bp = 0;
    if      (hp < 1.0f) { rp = c; gp = x; bp = 0; }
    else if (hp < 2.0f) { rp = x; gp = c; bp = 0; }
    else if (hp < 3.0f) { rp = 0; gp = c; bp = x; }
    else if (hp < 4.0f) { rp = 0; gp = x; bp = c; }
    else if (hp < 5.0f) { rp = x; gp = 0; bp = c; }
    else                { rp = c; gp = 0; bp = x; }
    r = rp + m; g = gp + m; b = bp + m;
}

// Build a Hopf fibration as raw S³ points — projection is deferred to the
// vertex shader so we can rotate in 4D per frame.
//
// Each (θ, φ) on S² lifts to a great circle in S³ parameterized by ψ:
//   (x, y, z, w) = (cos(θ/2) cosψ, cos(θ/2) sinψ,
//                   sin(θ/2) cos(ψ+φ), sin(θ/2) sin(ψ+φ))
//
// Color is HSV with hue = φ/2π, value modulated by sin(θ) so equatorial
// rings glow brighter than polar ones.
void build_hopf(GLuint& vao, GLuint& vbo, int& out_num_verts) {
    struct V { float x, y, z, w, r, g, b; };
    std::vector<V> verts;

    const int N_RINGS = 11;           // concentric tori (θ layers)
    const int N_PHI   = 28;           // fibers per torus (φ spokes)
    const int N_SEG   = 128;          // per-fiber sample density

    // θ sampled across the upper hemisphere; the singularity at θ=π is
    // handled by the shader, but keeping θ moderate avoids excessive
    // stretching under rotation.
    const float theta_min = 0.08f * (float)M_PI;
    const float theta_max = 0.72f * (float)M_PI;

    verts.reserve((size_t)N_RINGS * N_PHI * N_SEG * 2);
    std::vector<V> fiber(N_SEG);

    for (int ri = 0; ri < N_RINGS; ri++) {
        float t     = (N_RINGS == 1) ? 0.5f
                                     : (float)ri / (float)(N_RINGS - 1);
        float theta = theta_min + t * (theta_max - theta_min);
        float ct2   = std::cos(theta * 0.5f);
        float st2   = std::sin(theta * 0.5f);

        for (int pi = 0; pi < N_PHI; pi++) {
            float phi = (float)pi * 2.0f * (float)M_PI / (float)N_PHI;

            float h = phi / (2.0f * (float)M_PI);
            float v = 0.62f + 0.38f * std::sin(theta);
            float r, g, b;
            hsv_to_rgb(h, 0.92f, v, r, g, b);

            for (int si = 0; si < N_SEG; si++) {
                float psi = (float)si * 2.0f * (float)M_PI / (float)N_SEG;
                fiber[si] = {
                    ct2 * std::cos(psi),
                    ct2 * std::sin(psi),
                    st2 * std::cos(psi + phi),
                    st2 * std::sin(psi + phi),
                    r, g, b
                };
            }

            for (int si = 0; si < N_SEG; si++) {
                verts.push_back(fiber[si]);
                verts.push_back(fiber[(si + 1) % N_SEG]);
            }
        }
    }

    out_num_verts = (int)verts.size();

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(V),
                 verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(V),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(V),
                          (void*)(4 * sizeof(float)));
    glBindVertexArray(0);
}

} // namespace

// ============================================================================
// IntroScreen
// ============================================================================

bool IntroScreen::initialize() {
    s_ = new State();

    s_->prog_fiber     = link_program(VS_HOPF,   FS_HOPF);
    s_->prog_bright    = link_program(VS_FSQUAD, FS_BRIGHT);
    s_->prog_blur      = link_program(VS_FSQUAD, FS_BLUR);
    s_->prog_composite = link_program(VS_FSQUAD, FS_COMPOSITE);

    if (!s_->prog_fiber ||
        !s_->prog_bright || !s_->prog_blur || !s_->prog_composite) {
        std::fprintf(stderr, "[intro] Shader program creation failed\n");
        return false;
    }

    build_fs_quad(s_->fs_vao, s_->fs_vbo);
    build_hopf(s_->fiber_vao, s_->fiber_vbo, s_->num_fiber_verts);

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

    // Outer rigid spin (slower now that the internal rotation carries motion).
    glm::mat4 model(1.0f);
    model = glm::rotate(model, (float)s_->t_sim * 0.12f,
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, 0.42f, glm::vec3(1.0f, 0.0f, 0.0f));

    // Internal S³ rotation. We compose two different plane-rotations
    // (y–w and x–z) at incommensurate rates so the fibration never
    // retraces a closed orbit — the visual is a perpetual flow.
    float a1 = (float)s_->t_sim * 0.30f;   // rotation in the (y, w) plane
    float a2 = (float)s_->t_sim * 0.17f;   // rotation in the (x, z) plane
    glm::mat4 r_yw(1.0f);
    r_yw[1][1] =  std::cos(a1); r_yw[3][1] =  std::sin(a1);
    r_yw[1][3] = -std::sin(a1); r_yw[3][3] =  std::cos(a1);
    glm::mat4 r_xz(1.0f);
    r_xz[0][0] =  std::cos(a2); r_xz[2][0] =  std::sin(a2);
    r_xz[0][2] = -std::sin(a2); r_xz[2][2] =  std::cos(a2);
    glm::mat4 s3_rot = r_yw * r_xz;

    // --- Scene pass ---
    glBindFramebuffer(GL_FRAMEBUFFER, s_->scene.fb);
    glViewport(0, 0, s_->scene.w, s_->scene.h);
    glClearColor(0.02f, 0.03f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glUseProgram(s_->prog_fiber);
    glUniformMatrix4fv(glGetUniformLocation(s_->prog_fiber, "u_model"),
                       1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(s_->prog_fiber, "u_viewProj"),
                       1, GL_FALSE, glm::value_ptr(vp));
    glUniformMatrix4fv(glGetUniformLocation(s_->prog_fiber, "u_s3_rot"),
                       1, GL_FALSE, glm::value_ptr(s3_rot));
    glUniform1f(glGetUniformLocation(s_->prog_fiber, "u_scale"), 0.22f);
    glLineWidth(1.4f);
    glBindVertexArray(s_->fiber_vao);
    glDrawArrays(GL_LINES, 0, s_->num_fiber_verts);

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

    // ── Top-banner CALIPER title (drawn via draw list so we can drop a
    //    shadow that stays readable against the colored fibration) ──
    const char* title = "C A L I P E R";

    ImGui::SetWindowFontScale(5.0f);
    ImVec2 t_sz     = ImGui::CalcTextSize(title);
    float  title_fsz = ImGui::GetFontSize();
    ImGui::SetWindowFontScale(1.0f);

    float  title_x = wp.x + (ws.x - t_sz.x) * 0.5f;
    float  title_y = wp.y + ws.y * 0.055f;

    // Soft dark backdrop bar: gives text a consistent reading surface while
    // still letting the fibration breathe through.
    float band_pad_x = 36.0f;
    float band_pad_y = 12.0f;
    ImVec2 band_p0(title_x - band_pad_x,          title_y - band_pad_y);
    ImVec2 band_p1(title_x + t_sz.x + band_pad_x, title_y + t_sz.y + band_pad_y);
    dl->AddRectFilled(band_p0, band_p1, IM_COL32(4, 6, 14, 130), 10.0f);

    // Drop shadow then glyphs.
    ImFont* font = ImGui::GetFont();
    dl->AddText(font, title_fsz, ImVec2(title_x + 2.0f, title_y + 3.0f),
                IM_COL32(0, 0, 0, 200), title);
    dl->AddText(font, title_fsz, ImVec2(title_x, title_y),
                IM_COL32(210, 230, 255, 255), title);

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

    if (s_->prog_fiber)     glDeleteProgram(s_->prog_fiber);
    if (s_->prog_bright)    glDeleteProgram(s_->prog_bright);
    if (s_->prog_blur)      glDeleteProgram(s_->prog_blur);
    if (s_->prog_composite) glDeleteProgram(s_->prog_composite);

    if (s_->fiber_vbo) glDeleteBuffers(1, &s_->fiber_vbo);
    if (s_->fiber_vao) glDeleteVertexArrays(1, &s_->fiber_vao);
    if (s_->fs_vbo)    glDeleteBuffers(1, &s_->fs_vbo);
    if (s_->fs_vao)    glDeleteVertexArrays(1, &s_->fs_vao);

    destroy_fbo(s_->scene);
    destroy_fbo(s_->bloom_a);
    destroy_fbo(s_->bloom_b);

    delete s_;
    s_ = nullptr;
}

AppletKind IntroScreen::selected_applet() const {
    return s_ ? s_->chosen_kind : AppletKind::None;
}
