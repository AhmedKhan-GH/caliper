// The native colormap index of float32 values, for web/test/fixtures/lut_index.json:
// for each input line "<v> <vmin> <vmax>" (float32 bit patterns in hex) it prints
// the index map_f32_to_rgba8 (src/host/tensor_bridge.cpp) gives v for the range,
// through an identity table. make_lut_index.py compiles it against the native
// build's libcaliper.a and runs it.
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace caliper_host {
void map_f32_to_rgba8(const float* src, int w, int h, const uint32_t* lut256, float vmin, float vmax, uint8_t* dst);
}

static float from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

int main() {
    uint32_t ident[256];
    for (uint32_t i = 0; i < 256; ++i) ident[i] = i;
    unsigned v, lo, hi;
    while (std::scanf("%x %x %x", &v, &lo, &hi) == 3) {
        const float x = from_bits(v);
        uint8_t out[4];
        caliper_host::map_f32_to_rgba8(&x, 1, 1, ident, from_bits(lo), from_bits(hi), out);
        std::printf("%d\n", out[0]);
    }
    return 0;
}
