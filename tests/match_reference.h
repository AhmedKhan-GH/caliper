#pragma once
// The match tests compare library pieces with saved images from the removed renderer.
// PNG channels preserve RGBA8 colors or the bits of Depth32Float values.
// Each suite runs as a whole because image numbers follow its fixture order.
// The adjacent metadata stores camera matrices and picks from the same original draw.
// The reference files are read-only during tests. README.md describes their provenance.

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
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifndef CALIPER_MATCH_SUITE
#error "the test target defines CALIPER_MATCH_SUITE, its folder under tests/match_references"
#endif

namespace match_reference {

inline std::map<std::string, std::string>& names() {
    static std::map<std::string, std::string> value;
    return value;
}

inline const std::string& last_name(const char* kind) {
    REQUIRE(names().count(kind) == 1);
    return names().at(kind);
}

inline const char* root() {
    const char* value = std::getenv("CALIPER_MATCH_REFERENCES");
    REQUIRE_MESSAGE((value && *value), "CALIPER_MATCH_REFERENCES names no folder");
    return value;
}

inline const nlohmann::json& metadata(const std::string& name) {
    static const nlohmann::json saved = [] {
        const auto file = std::filesystem::path(root()) / CALIPER_MATCH_SUITE / "metadata.json";
        std::ifstream in(file);
        REQUIRE_MESSAGE(in.good(), "no reference metadata ", file.string());
        return nlohmann::json::parse(in);
    }();
    REQUIRE_MESSAGE(saved.contains(name), "no metadata for ", name);
    return saved.at(name);
}


inline std::string next_name(const char* kind) {
    static std::map<std::string, int> counts;
    char b[32];
    std::snprintf(b, sizeof b, "-%03d.png", counts[kind]++);
    return std::string(CALIPER_MATCH_SUITE) + "/" + kind + b;
}

inline const std::string& os_build() {
    static const std::string build = [] {
#ifdef __APPLE__
        char value[256] = {};
        std::size_t size = sizeof value;
        if (sysctlbyname("kern.osversion", value, &size, nullptr, 0) == 0 && size > 0 && size <= sizeof value)
            return std::string(value, size - 1);
#endif
        return std::string();
    }();
    return build;
}

inline const char* profile(const char* adapter, const std::string& build) {
    if (adapter && std::strcmp(adapter, "Apple Paravirtual device") == 0 && build == "24H232")
        return "metal-paravirtual-24H232";
    return "";
}

inline std::vector<std::uint8_t> picture(const char* kind, std::uint32_t width, std::uint32_t height,
                                         const std::vector<std::uint8_t>* drawn, const char* adapter = "") {
    REQUIRE_MESSAGE(drawn == nullptr, "reference images are read-only");
    const std::string name = next_name(kind);
    names()[kind] = name;
    std::filesystem::path file = std::filesystem::path(root()) / name;
    // Select the captured environment before decoding pixels. Unlisted cases retain their base image.
    const char* selected = profile(adapter, os_build());
    if (*selected) {
        const auto directory = std::filesystem::path(root()) / "profiles" / selected;
        std::ifstream manifest(directory / "provenance.json");
        REQUIRE_MESSAGE(manifest.good(), "no reference profile manifest ", directory.string());
        const auto provenance = nlohmann::json::parse(manifest);
        for (const auto& entry : provenance.at("cases")) {
            if (entry.at("file").get<std::string>() != name) continue;
            file = directory / name;
            MESSAGE("reference profile ", std::string(selected), ": ", file.string());
            break;
        }
    }
    const std::string path = file.string();
    int w = 0, h = 0, channels = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &channels, 4);
    REQUIRE_MESSAGE(px != nullptr, "no reference picture ", path);
    REQUIRE_MESSAGE((std::uint32_t(w) == width && std::uint32_t(h) == height), path, " is ", w, " x ", h);
    std::vector<std::uint8_t> out(px, px + std::size_t(w) * h * 4);
    stbi_image_free(px);
    return out;
}

}  // namespace match_reference
