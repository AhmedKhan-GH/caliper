// caliper_colormap_cases <file>: writes the boundary cases of tests/shaders/colormap_cases.h for
// the browser rows (web/test/shaders.spec.ts and web/test/colormap.spec.ts), so that the browser
// compares the generated WGSL and the TypeScript fallback with vectors from the C++ itself: for
// each range, the Cmap block from the shared derivation (src/host/gfx/colormap_keys.h), the
// inputs, and the RGBA that the reference map_f32_to_rgba8 gives each, with the test table.
//
// The file, little-endian: "CMAPCAS3", u32 ranges, u32 bytes of a block, then per range u32 name
// bytes, the name, the block (the generated Cmap struct: the range, the flip and one 16-byte entry
// per level, its first key, the next level's first key and its colour), u32 inputs, the inputs
// (u32 bits) and the expected (u32 packed RGBA).
//
// Plain C++17 with no GPU. Outside CMake (the web workflow) it builds with one command:
//   c++ -std=c++17 -O2 -Isrc/host -Isdk/include -Itests tests/shaders/colormap_cases_tool.cpp \
//       src/host/tensor_bridge.cpp -o caliper_colormap_cases
#include "shaders/colormap_cases.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: caliper_colormap_cases <file>\n");
        return 2;
    }
    FILE* out = std::fopen(argv[1], "wb");
    if (out == nullptr) {
        std::fprintf(stderr, "caliper_colormap_cases: cannot write %s\n", argv[1]);
        return 1;
    }
    const auto rs = colormap_cases::ranges();
    const std::vector<uint32_t> table = colormap_cases::lut();
    const uint32_t n = uint32_t(rs.size()), block_bytes = uint32_t(sizeof(caliper_gfx::shaders::Cmap));
    std::fwrite("CMAPCAS3", 1, 8, out);
    std::fwrite(&n, 4, 1, out);
    std::fwrite(&block_bytes, 4, 1, out);
    size_t total = 0;
    for (size_t i = 0; i < rs.size(); ++i) {
        const auto& r = rs[i];
        const caliper_gfx::shaders::Cmap block = colormap_cases::block(r, table);
        const std::vector<uint32_t> in = colormap_cases::inputs(r, block, colormap_cases::seed(i));
        const std::vector<uint32_t> want = colormap_cases::expected(r, in, table);
        const uint32_t name = uint32_t(std::strlen(r.name)), count = uint32_t(in.size());
        std::fwrite(&name, 4, 1, out);
        std::fwrite(r.name, 1, name, out);
        std::fwrite(&block, sizeof block, 1, out);
        std::fwrite(&count, 4, 1, out);
        std::fwrite(in.data(), 4, in.size(), out);
        std::fwrite(want.data(), 4, want.size(), out);
        total += in.size();
    }
    const bool ok = std::fclose(out) == 0;
    std::printf("%s: %zu inputs over %u ranges\n", argv[1], total, n);
    return ok ? 0 : 1;
}
