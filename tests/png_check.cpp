// caliper_png_check <file.png> <width> <height>: checks that the file is a PNG
// of that size that is not one flat colour, for the `caliper shot` test
// (tests/shot_run.cmake). Prints the size and the number of distinct colours,
// and exits with 0 when every check passes and 1 otherwise.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: caliper_png_check <file.png> <width> <height>\n");
        return 2;
    }
    const int want_w = std::atoi(argv[2]);
    const int want_h = std::atoi(argv[3]);
    int w = 0, h = 0, channels = 0;
    unsigned char* px = stbi_load(argv[1], &w, &h, &channels, 4);
    if (!px) {
        std::fprintf(stderr, "%s: not a readable PNG (%s)\n", argv[1], stbi_failure_reason());
        return 1;
    }
    std::unordered_set<uint32_t> colours;
    for (size_t i = 0; i < (size_t)w * (size_t)h; ++i) {
        const unsigned char* p = px + i * 4;
        colours.insert((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
                       (uint32_t)p[3] << 24);
    }
    stbi_image_free(px);
    std::printf("%s: %dx%d, %zu colours\n", argv[1], w, h, colours.size());
    if (w != want_w || h != want_h) {
        std::fprintf(stderr, "%s is %dx%d, not %dx%d\n", argv[1], w, h, want_w, want_h);
        return 1;
    }
    if (colours.size() < 2) {
        std::fprintf(stderr, "%s is one flat colour\n", argv[1]);
        return 1;
    }
    return 0;
}
