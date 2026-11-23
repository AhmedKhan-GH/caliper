#pragma once
// The reference pictures of the match tests (tests/*_match): what the scene core's built-in draw
// gave for each case before it was deleted, as PNG files under tests/match_references/<suite>/.
// Each library piece is compared with them at the tolerances that its test used against the
// built-in draw, thus the pieces stay pinned to the pictures that the built-in drew.
//
// The test target names its suite (CALIPER_MATCH_SUITE), and ctest gives the folder of the
// references in the environment (CALIPER_MATCH_REFERENCES). The pictures of one kind are numbered
// in the order in which the test binary asks for them, so a test binary runs whole, as ctest runs
// it. picture(kind, w, h, &pixels) records pixels as the next picture when
// CALIPER_MATCH_REFERENCE_WRITE names a folder (this is how the references were taken, from the
// built-in draw, before its deletion) and otherwise gives them back; picture(kind, w, h, nullptr)
// reads the next picture. A picture is width x height texels of 4 bytes, RGBA8 colours or the bits
// of a 32-bit depth, stored losslessly in the PNG's four channels.
//
// The references change only when a piece's picture is meant to change: a person replaces the
// files with the piece's own pictures, which the tests leave in their capture folders, and the
// change is reviewed like code.
//
// The including file has included <stb_image_write.h> (every match rig writes its captures).
#ifndef STBI_INCLUDE_STB_IMAGE_H   // a file that reads PNG files itself has its own stb_image
#define STBI_ONLY_PNG
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#ifndef CALIPER_MATCH_SUITE
#error "the test target defines CALIPER_MATCH_SUITE, its folder under tests/match_references"
#endif

namespace match_reference {

inline std::string next_name(const char* kind) {
    static std::map<std::string, int> counts;
    char b[32];
    std::snprintf(b, sizeof b, "-%03d.png", counts[kind]++);
    return std::string(CALIPER_MATCH_SUITE) + "/" + kind + b;
}

inline std::vector<std::uint8_t> picture(const char* kind, std::uint32_t width, std::uint32_t height,
                                         const std::vector<std::uint8_t>* drawn) {
    const std::string name = next_name(kind);
    if (drawn) {
        REQUIRE(drawn->size() == std::size_t(width) * height * 4);
        if (const char* dir = std::getenv("CALIPER_MATCH_REFERENCE_WRITE"); dir && *dir) {
            const std::filesystem::path path = std::filesystem::path(dir) / name;
            std::filesystem::create_directories(path.parent_path());
            CHECK(stbi_write_png(path.string().c_str(), int(width), int(height), 4, drawn->data(), int(width) * 4) != 0);
        }
        return *drawn;
    }
    const char* root = std::getenv("CALIPER_MATCH_REFERENCES");
    REQUIRE_MESSAGE((root && *root), "CALIPER_MATCH_REFERENCES names no folder");
    const std::string path = std::string(root) + "/" + name;
    int w = 0, h = 0, channels = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &channels, 4);
    REQUIRE_MESSAGE(px != nullptr, "no reference picture ", path);
    REQUIRE_MESSAGE((std::uint32_t(w) == width && std::uint32_t(h) == height), path, " is ", w, " x ", h);
    std::vector<std::uint8_t> out(px, px + std::size_t(w) * h * 4);
    stbi_image_free(px);
    return out;
}

}  // namespace match_reference
