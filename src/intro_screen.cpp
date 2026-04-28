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
#include <random>

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
uniform float     u_bloom_mix;
void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    vec3 col = scene + bloom * u_bloom_mix;

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

// Live parameters. Anything in `render` is cheap (uniforms only);
// changes to anything in `geom` require rebuilding the VBO.
struct HopfParams {
    // --- render (cheap / per-frame uniforms) ---
    float speed            = 1.35f;  // global time multiplier (0 = pause)
    float rate_s3_yw       = -0.059f; // rot rate in (y,w) plane, rad/sec
    float rate_s3_xz       = 0.033f; // rot rate in (x,z) plane, rad/sec
    float rate_outer_spin  = -0.001f; // outer Y-axis spin rate, rad/sec
    float static_tilt      = -0.89012f; // fixed X-axis tilt (rad) ≈ -51°
    float scale            = 0.266f; // post-stereographic scale
    float fov_deg          = 30.6f;
    float cam_dist         = 3.39f;
    float line_width       = 1.40f;
    float bloom_threshold  = 0.66f;
    float bloom_mix        = 0.71f;

    // --- geometry (rebuild on change) ---
    int   n_rings       = 5;
    int   n_phi         = 32;
    int   n_seg         = 128;
    float theta_min_frac = 0.11f;   // fraction of π
    float theta_max_frac = 0.83f;
    float saturation    = 0.83f;
    float hue_offset    = 0.30f;    // palette rotation, [0,1)
    float value_base    = 0.59f;    // HSV-V base before sin(θ) modulation
    float value_swing   = 0.49f;    // HSV-V sin(θ) amplitude
};

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
    double prev_time     = 0.0;
    float  camera_angle  = 0.0f;

    // Angle accumulators (so changing rate sliders doesn't jump — each
    // integrates independently from its current rate).
    float ang_outer = 0.0f;
    float ang_s3_yw = 0.0f;
    float ang_s3_xz = 0.0f;

    // Live params + geometry-rebuild flag
    HopfParams p;
    bool geom_dirty = false;

    // Last non-zero speed, restored when un-pausing via the toggle button.
    float saved_speed = 1.0f;

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
void build_hopf(const HopfParams& p,
                GLuint& vao, GLuint& vbo, int& out_num_verts) {
    struct V { float x, y, z, w, r, g, b; };
    std::vector<V> verts;

    const int N_RINGS = std::max(1, p.n_rings);
    const int N_PHI   = std::max(3, p.n_phi);
    const int N_SEG   = std::max(8, p.n_seg);

    // θ sampled across the upper hemisphere; the singularity at θ=π is
    // handled by the shader, but keeping θ moderate avoids excessive
    // stretching under rotation.
    float tmin = std::min(p.theta_min_frac, p.theta_max_frac);
    float tmax = std::max(p.theta_min_frac, p.theta_max_frac);
    const float theta_min = tmin * (float)M_PI;
    const float theta_max = tmax * (float)M_PI;

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

            float h = phi / (2.0f * (float)M_PI) + p.hue_offset;
            h -= std::floor(h);
            float v = p.value_base + p.value_swing * std::sin(theta);
            v = std::max(0.0f, std::min(1.0f, v));
            float r, g, b;
            hsv_to_rgb(h, p.saturation, v, r, g, b);

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

// Roll a fresh aesthetically-biased configuration. Ranges are tighter than the
// full slider extents so the output stays within visually pleasing territory.
void randomize_params(HopfParams& p) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    auto rnd  = [&](float lo, float hi) { return lo + u01(rng) * (hi - lo); };
    auto rndi = [&](int lo, int hi)     { return lo + (int)(u01(rng) * (float)(hi - lo + 1)); };

    // Animation
    p.speed           = rnd(0.5f, 2.0f);
    p.rate_s3_yw      = rnd(-0.6f, 0.6f);
    p.rate_s3_xz      = rnd(-0.6f, 0.6f);
    p.rate_outer_spin = rnd(-0.3f, 0.3f);
    p.static_tilt     = rnd(-1.0f, 1.0f);

    // Projection / style
    p.scale      = rnd(0.12f, 0.35f);
    p.line_width = rnd(1.0f, 2.2f);

    // Bloom
    p.bloom_threshold = rnd(0.3f, 0.9f);
    p.bloom_mix       = rnd(0.6f, 1.6f);

    // Geometry
    p.n_rings        = rndi(5, 18);
    p.n_phi          = rndi(12, 36);
    p.theta_min_frac = rnd(0.04f, 0.15f);
    p.theta_max_frac = rnd(0.55f, 0.85f);
    p.hue_offset     = u01(rng);
    p.saturation     = rnd(0.7f, 1.0f);
    p.value_base     = rnd(0.5f, 0.75f);
    p.value_swing    = rnd(0.25f, 0.5f);
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
    build_hopf(s_->p, s_->fiber_vao, s_->fiber_vbo, s_->num_fiber_verts);

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

    // Speed-scaled advance. Each rotation has its own angle accumulator so
    // that changing a rate slider mid-flight just changes the slope — the
    // fibration never jumps.
    const auto& p = s_->p;
    float sdt = (float)dt * p.speed;
    s_->ang_outer += sdt * p.rate_outer_spin;
    s_->ang_s3_yw += sdt * p.rate_s3_yw;
    s_->ang_s3_xz += sdt * p.rate_s3_xz;
    s_->camera_angle += (float)dt * 0.06f;
}

void IntroScreen::render_3d(int fb_w, int fb_h) {
    if (!s_ || !s_->ok || fb_w <= 0 || fb_h <= 0) return;

    // Rebuild the fibration geometry if a geometry slider changed since
    // last frame. Cheap enough (~1ms for default sizes) to do inline.
    if (s_->geom_dirty) {
        if (s_->fiber_vbo) glDeleteBuffers(1, &s_->fiber_vbo);
        if (s_->fiber_vao) glDeleteVertexArrays(1, &s_->fiber_vao);
        s_->fiber_vbo = s_->fiber_vao = 0;
        build_hopf(s_->p, s_->fiber_vao, s_->fiber_vbo, s_->num_fiber_verts);
        s_->geom_dirty = false;
    }

    const auto& p = s_->p;

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
    glm::mat4 proj = glm::perspective(glm::radians(p.fov_deg), aspect, 0.1f, 50.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, p.cam_dist),
                                 glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 vp = proj * view;

    // Outer rigid spin + fixed tilt.
    glm::mat4 model(1.0f);
    model = glm::rotate(model, s_->ang_outer,
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, p.static_tilt, glm::vec3(1.0f, 0.0f, 0.0f));

    // Internal S³ rotation. We compose two different plane-rotations
    // (y–w and x–z) at incommensurate rates so the fibration never
    // retraces a closed orbit — the visual is a perpetual flow.
    float a1 = s_->ang_s3_yw;  // (y, w) plane
    float a2 = s_->ang_s3_xz;  // (x, z) plane
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
    glUniform1f(glGetUniformLocation(s_->prog_fiber, "u_scale"), p.scale);
    glLineWidth(p.line_width);
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
    glUniform1f(glGetUniformLocation(s_->prog_bright, "u_threshold"), p.bloom_threshold);
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
    glUniform1f(glGetUniformLocation(s_->prog_composite, "u_bloom_mix"), p.bloom_mix);
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

    // Applet-list height — only as tall as the actual applet rows.
    const float applet_row_h = 60.0f;
    const float applet_list_h =
        (float)kNumApplets * applet_row_h
        + (float)(kNumApplets - 1) * 8.0f    // ItemSpacing.y, pushed below
        + 12.0f;                             // small visual padding

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
                      ImVec2(col_w, applet_list_h),
                      false,
                      ImGuiWindowFlags_NoBackground);

    for (int i = 0; i < kNumApplets; i++) {
        if (draw_applet_row(i, kApplets[i], applet_row_h)) {
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

    // ── Fibration parameters panel (anchored below applets, collapsible) ──
    // Sized to the tallest tab (Geometry, 2 cols × 5 rows). ImGui's title-bar
    // chevron gives us collapse-to-title-bar for free; NoMove/NoResize keep
    // it anchored as a fixed panel rather than a loose window.
    const float panel_w = col_w;
    const float panel_h = 360.0f;
    ImGui::SetNextWindowPos(
        ImVec2(vp->WorkPos.x + col_x,
               vp->WorkPos.y + col_y + 28.0f + applet_list_h + 14.0f),
        ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panel_w, panel_h), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.82f);

    bool panel_open = ImGui::Begin("Parameters", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    if (panel_open) {
        HopfParams& p = s_->p;
        bool geom_changed = false;

        // Tighten the gap between a label and its slider.
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,  ImVec2(6.0f, 2.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,  ImVec2(4.0f, 6.0f));

        auto lbl = [](const char* t) {
            ImGui::TextDisabled("%s", t);
            ImGui::SetNextItemWidth(-1);
        };

        if (ImGui::BeginTabBar("##fib_tabs", ImGuiTabBarFlags_FittingPolicyResizeDown)) {
            if (ImGui::BeginTabItem("Animation")) {
                if (ImGui::BeginTable("##anim_tbl", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();
                    lbl("speed");
                    ImGui::SliderFloat("##speed", &p.speed, 0.0f, 3.0f, "%.2fx");
                    ImGui::TableNextColumn();
                    lbl("outer spin");
                    ImGui::SliderFloat("##outer_spin", &p.rate_outer_spin, -1.0f, 1.0f, "%.3f");

                    ImGui::TableNextColumn();
                    lbl("y-w rate");
                    ImGui::SliderFloat("##yw_rate", &p.rate_s3_yw, -1.5f, 1.5f, "%.3f");
                    ImGui::TableNextColumn();
                    lbl("tilt");
                    ImGui::SliderAngle("##tilt", &p.static_tilt, -180.0f, 180.0f);

                    ImGui::TableNextColumn();
                    lbl("x-z rate");
                    ImGui::SliderFloat("##xz_rate", &p.rate_s3_xz, -1.5f, 1.5f, "%.3f");
                    ImGui::TableNextColumn();
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Camera")) {
                if (ImGui::BeginTable("##cam_tbl", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();
                    lbl("FOV");
                    ImGui::SliderFloat("##fov", &p.fov_deg, 15.0f, 90.0f, "%.1f deg");
                    ImGui::TableNextColumn();
                    lbl("cam dist");
                    ImGui::SliderFloat("##cam_dist", &p.cam_dist, 1.5f, 10.0f, "%.2f");

                    ImGui::TableNextColumn();
                    lbl("scale");
                    ImGui::SliderFloat("##scale", &p.scale, 0.02f, 1.0f, "%.3f");
                    ImGui::TableNextColumn();
                    lbl("line width");
                    ImGui::SliderFloat("##line_w", &p.line_width, 0.5f, 4.0f, "%.2f");
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Bloom")) {
                if (ImGui::BeginTable("##bloom_tbl", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();
                    lbl("threshold");
                    ImGui::SliderFloat("##bthresh", &p.bloom_threshold, 0.0f, 2.0f, "%.2f");
                    ImGui::TableNextColumn();
                    lbl("mix");
                    ImGui::SliderFloat("##bmix", &p.bloom_mix, 0.0f, 3.0f, "%.2f");
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Geometry")) {
                if (ImGui::BeginTable("##geom_tbl", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextColumn();
                    lbl("rings");
                    if (ImGui::SliderInt("##rings", &p.n_rings, 1, 40))  geom_changed = true;
                    ImGui::TableNextColumn();
                    lbl("hue offset");
                    if (ImGui::SliderFloat("##hue", &p.hue_offset, 0.0f, 1.0f, "%.2f"))  geom_changed = true;

                    ImGui::TableNextColumn();
                    lbl("fibers");
                    if (ImGui::SliderInt("##fibers", &p.n_phi, 3, 120))  geom_changed = true;
                    ImGui::TableNextColumn();
                    lbl("saturation");
                    if (ImGui::SliderFloat("##sat", &p.saturation, 0.0f, 1.0f, "%.2f"))  geom_changed = true;

                    ImGui::TableNextColumn();
                    lbl("segments");
                    if (ImGui::SliderInt("##segs", &p.n_seg, 8, 512))  geom_changed = true;
                    ImGui::TableNextColumn();
                    lbl("value base");
                    if (ImGui::SliderFloat("##vbase", &p.value_base, 0.0f, 1.0f, "%.2f"))  geom_changed = true;

                    ImGui::TableNextColumn();
                    lbl("theta min / pi");
                    if (ImGui::SliderFloat("##tmin", &p.theta_min_frac, 0.0f, 1.0f, "%.2f"))  geom_changed = true;
                    ImGui::TableNextColumn();
                    lbl("value swing");
                    if (ImGui::SliderFloat("##vswing", &p.value_swing, 0.0f, 1.0f, "%.2f"))  geom_changed = true;

                    ImGui::TableNextColumn();
                    lbl("theta max / pi");
                    if (ImGui::SliderFloat("##tmax", &p.theta_max_frac, 0.0f, 1.0f, "%.2f"))  geom_changed = true;
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::PopStyleVar(2);

        // Persistent action rows — full-width playback toggle on top,
        // Default / Randomize split underneath. Anchored to bottom of panel,
        // visible from any tab.
        const float btn_h    = ImGui::GetFrameHeight();
        const float pad      = ImGui::GetStyle().WindowPadding.y;
        const float gap_y    = ImGui::GetStyle().ItemSpacing.y;
        const float gap_x    = ImGui::GetStyle().ItemSpacing.x;
        const float half     = (ImGui::GetContentRegionAvail().x - gap_x) * 0.5f;
        const float rows_h   = btn_h * 2.0f + gap_y;
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - rows_h - pad);

        const bool paused = (p.speed == 0.0f);
        const char* lbl_pp = paused ? "Play" : "Pause";
        if (ImGui::Button(lbl_pp, ImVec2(-1, btn_h))) {
            if (paused) {
                p.speed = (s_->saved_speed > 0.0f) ? s_->saved_speed : 1.0f;
            } else {
                s_->saved_speed = p.speed;
                p.speed = 0.0f;
            }
        }
        if (ImGui::Button("Default", ImVec2(half, btn_h))) {
            p = HopfParams{};
            s_->saved_speed = 1.0f;
            geom_changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Randomize", ImVec2(half, btn_h))) {
            randomize_params(p);
            geom_changed = true;
        }

        if (geom_changed) s_->geom_dirty = true;
    }
    ImGui::End();
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
