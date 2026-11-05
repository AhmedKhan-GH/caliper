// CUDA twins of colormap.comp and the u8 blit (docs/vulkan-update-costs.md W2,
// D25). The renderer launches these on the producer's stream; they read the
// tensor in place and write the texture's VkImage through a surface object.
//
// The output must stay BYTE-IDENTICAL to the CPU reference map_f32_to_rgba8
// (§16), so -fmad=false is load-bearing: a contracted t*255+0.5 rounds
// differently from the reference. NaN handling matches the reference too.
//
// The host links no toolkit (D11): the build embeds the checked-in PTX. After
// editing this file, regenerate it from a developer prompt with:
//   nvcc -ptx -m64 -arch=sm_75 -fmad=false -allow-unsupported-compiler
//        -o cmap_direct.ptx cmap_direct.cu
// (the override is for the VS version check only; PTX has no host code)

#include <cuda_runtime.h>

struct Lut256 { unsigned int v[256]; };

extern "C" __global__ void caliper_cmap_f32(const char* src,
                                            cudaSurfaceObject_t dst,
                                            int w, int h,
                                            long long stride_x, long long stride_y,
                                            float vmin, float vmax,
                                            Lut256 lut) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float v = *(const float*)(src + y * stride_y + x * stride_x);
    int idx = 0;
    if (v == v) {
        const float denom = vmax - vmin;
        float t = (denom != 0.0f) ? (v - vmin) / denom : 0.0f;
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;
        // A NaN t (infinite vmin/vmax) lands on index 0, as x64's cast and
        // clamp do in the reference; CUDA's cast already gives 0.
        idx = (int)(t * 255.0f + 0.5f);
        if (idx < 0) idx = 0;
        else if (idx > 255) idx = 255;
    }
    surf2Dwrite(lut.v[idx], dst, x * 4, y);
}

extern "C" __global__ void caliper_blit_rgba8(const char* src,
                                              cudaSurfaceObject_t dst,
                                              int w, int h,
                                              long long stride_x, long long stride_y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned char* p = (const unsigned char*)(src + y * stride_y + x * stride_x);
    const unsigned int packed = (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
                                ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
    surf2Dwrite(packed, dst, x * 4, y);
}
