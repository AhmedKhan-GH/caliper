#pragma once
// colormap_ref.h: declarations for the CPU reference colormap LUTs and
// conversion functions (defined in src/host/tensor_bridge.cpp).
//
// Kept in src/host/gfx/ so graphics-layer headers (colormap_keys.h,
// colormap_tables.h) and renderers (webgpu_renderer.cpp) depend downward on
// gfx without including the host-level tensor_bridge.h.
#include <caliper/services/tensor_bridge_v1.h>

#include <cstdint>

namespace caliper_host {

// Built-in 256-entry RGBA8 colormap LUTs (packed r | g<<8 | b<<16 | a<<24).
// Returned by value-stable pointer; nullptr for an out-of-range colormap id.
const uint32_t* colormap_lut(int32_t colormap);

// Colormap a row-major (h,w) f32 buffer through lut256 into tightly-packed
// RGBA8 dst (w*h*4 bytes):
//   idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5   (truncated)
// with vmax==vmin -> t=0, and NaN or non-finite span -> index 0.
void map_f32_to_rgba8(const float* src, int w, int h,
                      const uint32_t* lut256, float vmin, float vmax,
                      uint8_t* dst);

// Exact float comparison thresholds for GPU shaders whose division is not exact.
// Writes T[0..255] (T[0] unused) and returns mode: 0 degenerate/non-finite,
// 1 increasing, 2 decreasing.
uint32_t colormap_thresholds(float vmin, float vmax, float* T);

// Expand a row-major (h,w,c) u8 buffer (c in 1..4) into tightly-packed RGBA8.
void expand_u8_to_rgba8(const uint8_t* src, int w, int h, int c, uint8_t* dst);

}  // namespace caliper_host
