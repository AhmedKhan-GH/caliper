#pragma once

// Header-only orbit-camera math, hoisted verbatim from the TwinScope donor
// applet (git show codex/twinscope-implementation:applets/twin_scope/
// twin_scope.cpp, the V3/look_at/perspective block) so that draw AND pick share
// ONE eye computation (design §8.f). The math is byte-identical to the donor;
// only the two convenience helpers (orbit_eye, cursor_ray) are factored out of
// the donor's draw_ui usage so the applet stops duplicating them.

#include <cmath>
#include <cstring>

namespace caliper::adapters {

inline constexpr float kOrbitPi = 3.14159265358979323846f;

struct V3 { float x, y, z; };

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline V3 normalized(V3 a) {
    const float length = std::sqrt(dot(a, a));
    // {0,1,0} degenerate fallback (donor): a zero-length vector normalizes to
    // world-up so look_at / ray construction never emit NaNs.
    return length > 1e-8f ? a * (1.f / length) : V3{0.f, 1.f, 0.f};
}

// Right-handed lookAt, column-major (GLSL layout).
inline void look_at(V3 eye, V3 at, V3 up, float* m) {
    const V3 f = normalized(at - eye);
    const V3 s = normalized(cross(f, up));
    const V3 u = cross(s, f);
    const float value[16] = {
        s.x, u.x, -f.x, 0.f, s.y, u.y, -f.y, 0.f,
        s.z, u.z, -f.z, 0.f, -dot(s, eye), -dot(u, eye), dot(f, eye), 1.f};
    std::memcpy(m, value, sizeof(value));
}

// Vulkan-style perspective (z in [0,1]; +y-up NDC comes from the backend's
// negative-viewport convention, so no Y flip here).
inline void perspective(float fovy, float aspect, float near_z, float far_z,
                        float* m) {
    const float f = 1.f / std::tan(fovy * 0.5f);
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far_z / (near_z - far_z);
    m[11] = -1.f;
    m[14] = (near_z * far_z) / (near_z - far_z);
}

// Eye position for an orbit camera (donor draw_ui, ~line 490): azimuth sweeps
// the horizontal ring, elevation lifts toward world-up, distance is the radius.
inline V3 orbit_eye(float azimuth, float elevation, float distance, V3 target) {
    const float ce = std::cos(elevation);
    return target + V3{distance * ce * std::cos(azimuth),
                       distance * std::sin(elevation),
                       distance * ce * std::sin(azimuth)};
}

// Normalized world-space ray through an NDC point (donor pick, ~line 572):
// ndc_x/ndc_y in [-1,1] with +y up. fov_deg is the vertical field of view, and
// the basis is the same one look_at builds, so a ray cast here lands where the
// draw put the pixel.
inline V3 cursor_ray(V3 eye, V3 target, float fov_deg, float aspect,
                     float ndc_x, float ndc_y) {
    const V3 forward = normalized(target - eye);
    const V3 right = normalized(cross(forward, {0.f, 1.f, 0.f}));
    const V3 up = cross(right, forward);
    const float tangent = std::tan(fov_deg * kOrbitPi / 360.f);
    return normalized(forward + right * (ndc_x * tangent * aspect) +
                      up * (ndc_y * tangent));
}

}  // namespace caliper::adapters
