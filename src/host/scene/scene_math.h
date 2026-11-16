// scene_math.h: the double-precision vectors and 4 x 4 matrices of the scene core.
//
// Matrices are column-major (element (row r, column c) at m[c * 4 + r]) and compose as
// clip = P * V * M * p, the conventions of design/renderer-core.md ("Conventions"). The scene
// core composes in double and rounds a draw's matrix to f32 once.
#pragma once

#include <cmath>
#include <cstdint>

namespace caliper_scene {

struct Vec3 {
    double x = 0, y = 0, z = 0;
    double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
    const double l = length(a);
    return l > 0 ? a * (1.0 / l) : Vec3{0, 0, 0};
}
inline bool finite(Vec3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

struct Mat4 {
    double m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   // identity
    double& at(int row, int col) { return m[col * 4 + row]; }
    double at(int row, int col) const { return m[col * 4 + row]; }
    static Mat4 identity() { return Mat4{}; }
    static Mat4 zero() {
        Mat4 r;
        for (double& v : r.m) v = 0;
        return r;
    }
    static Mat4 from(const double src[16]) {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = src[i];
        return r;
    }
    static Mat4 from(const float src[16]) {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = src[i];
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r = Mat4::zero();
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

inline bool finite(const Mat4& a) {
    for (double v : a.m)
        if (!std::isfinite(v)) return false;
    return true;
}

// p with w = 1, without the division by w (the scene's matrices are affine).
inline Vec3 transform_point(const Mat4& a, Vec3 p) {
    return {a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12],
            a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13],
            a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14]};
}
inline Vec3 transform_vector(const Mat4& a, Vec3 v) {
    return {a.m[0] * v.x + a.m[4] * v.y + a.m[8] * v.z,
            a.m[1] * v.x + a.m[5] * v.y + a.m[9] * v.z,
            a.m[2] * v.x + a.m[6] * v.y + a.m[10] * v.z};
}
// The full product with w = 1, returning (x, y, z, w).
inline void transform_h(const Mat4& a, Vec3 p, double out[4]) {
    for (int r = 0; r < 4; ++r)
        out[r] = a.m[r] * p.x + a.m[4 + r] * p.y + a.m[8 + r] * p.z + a.m[12 + r];
}

inline Mat4 translation(Vec3 t) {
    Mat4 r;
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    return r;
}
inline Mat4 scaling(Vec3 s) {
    Mat4 r;
    r.m[0] = s.x;
    r.m[5] = s.y;
    r.m[10] = s.z;
    return r;
}

// General inverse by cofactors; false (and r untouched) when singular or not finite.
inline bool inverse(const Mat4& a, Mat4* r) {
    const double* m = a.m;
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!(std::fabs(det) > 0) || !std::isfinite(det)) return false;
    const double k = 1.0 / det;
    for (int i = 0; i < 16; ++i) r->m[i] = inv[i] * k;
    return true;
}

// A unit quaternion (x, y, z, w) as a rotation matrix. The caller normalises.
inline Mat4 rotation_from_quaternion(double x, double y, double z, double w) {
    Mat4 r;
    r.at(0, 0) = 1 - 2 * (y * y + z * z);
    r.at(1, 0) = 2 * (x * y + w * z);
    r.at(2, 0) = 2 * (x * z - w * y);
    r.at(0, 1) = 2 * (x * y - w * z);
    r.at(1, 1) = 1 - 2 * (x * x + z * z);
    r.at(2, 1) = 2 * (y * z + w * x);
    r.at(0, 2) = 2 * (x * z + w * y);
    r.at(1, 2) = 2 * (y * z - w * x);
    r.at(2, 2) = 1 - 2 * (x * x + y * y);
    return r;
}

// The right-handed look-at of the points view (src/host/live/live_views_draw.cpp, look_at):
// view space looks along -z with +y up and +x right.
inline Mat4 look_at(Vec3 eye, Vec3 at, Vec3 up) {
    const Vec3 f = normalize(at - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r;
    r.at(0, 0) = s.x;  r.at(0, 1) = s.y;  r.at(0, 2) = s.z;  r.at(0, 3) = -dot(s, eye);
    r.at(1, 0) = u.x;  r.at(1, 1) = u.y;  r.at(1, 2) = u.z;  r.at(1, 3) = -dot(u, eye);
    r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z; r.at(2, 3) = dot(f, eye);
    return r;
}

// The points view's perspective with clip z in [0, 1], near at 0 (right-handed view space).
inline Mat4 perspective_z01(double fovy, double aspect, double zn, double zf) {
    const double f = 1.0 / std::tan(fovy * 0.5);
    Mat4 r = Mat4::zero();
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = zf / (zn - zf);
    r.m[11] = -1.0;
    r.m[14] = (zn * zf) / (zn - zf);
    return r;
}

// Columns of transpose(inverse(upper 3 x 3 of a)), the normal matrix of the geometry service
// (src/host/renderer/metal_renderer.mm, normal_matrix_columns): identity when singular.
inline void normal_matrix_columns(const Mat4& a, float c0[4], float c1[4], float c2[4]) {
    const double a00 = a.m[0], a01 = a.m[4], a02 = a.m[8];
    const double a10 = a.m[1], a11 = a.m[5], a12 = a.m[9];
    const double a20 = a.m[2], a21 = a.m[6], a22 = a.m[10];
    const double det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) +
                       a02 * (a10 * a21 - a11 * a20);
    if (!(std::fabs(det) >= 1e-12) || !std::isfinite(det)) {
        c0[0] = 1; c0[1] = 0; c0[2] = 0; c0[3] = 0;
        c1[0] = 0; c1[1] = 1; c1[2] = 0; c1[3] = 0;
        c2[0] = 0; c2[1] = 0; c2[2] = 1; c2[3] = 0;
        return;
    }
    const double k = 1.0 / det;
    c0[0] = float((a11 * a22 - a12 * a21) * k);
    c0[1] = float(-(a01 * a22 - a02 * a21) * k);
    c0[2] = float((a01 * a12 - a02 * a11) * k);
    c0[3] = 0;
    c1[0] = float(-(a10 * a22 - a12 * a20) * k);
    c1[1] = float((a00 * a22 - a02 * a20) * k);
    c1[2] = float(-(a00 * a12 - a02 * a10) * k);
    c1[3] = 0;
    c2[0] = float((a10 * a21 - a11 * a20) * k);
    c2[1] = float(-(a00 * a21 - a01 * a20) * k);
    c2[2] = float((a00 * a11 - a01 * a10) * k);
    c2[3] = 0;
}

inline void to_f32(const Mat4& a, float out[16]) {
    for (int i = 0; i < 16; ++i) out[i] = float(a.m[i]);
}

inline constexpr double kPi = 3.14159265358979323846;

}  // namespace caliper_scene
