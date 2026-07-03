// Shared read-only helpers for Training Lab golden-file tests.
// Bin format (little-endian): int32 ndim, int32 dims[ndim], float32 data[prod(dims)].
#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace golden {

// Directory containing the golden artifacts. Override with GOLDEN_DIR env var.
inline std::string dir() {
    if (const char* e = std::getenv("GOLDEN_DIR")) return e;
    return "applets/repnet_demo/tests/golden";
}
inline std::string path(const std::string& rel) { return dir() + "/" + rel; }

// Load a .bin tensor (float32) written by export_training_lab_goldens.py.
inline torch::Tensor load_bin(const std::string& rel) {
    std::ifstream f(path(rel), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open golden bin: " + path(rel));
    int32_t ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), 4);
    std::vector<int64_t> shape(ndim);
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        int32_t d = 0;
        f.read(reinterpret_cast<char*>(&d), 4);
        shape[i] = d;
        n *= d;
    }
    std::vector<float> data(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("short read on golden bin: " + path(rel));
    return torch::from_blob(data.data(), shape, torch::kFloat32).clone();
}

// Read a whole text file (e.g. a JSON golden) into a string.
inline std::string load_text(const std::string& rel) {
    std::ifstream f(path(rel));
    if (!f) throw std::runtime_error("cannot open golden text: " + path(rel));
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// Minimal test harness: CHECK(cond, msg); main returns nonzero if any failed.
struct Harness {
    int failures = 0;
    int checks = 0;
    void check(bool cond, const std::string& msg) {
        ++checks;
        if (!cond) {
            ++failures;
            std::fprintf(stderr, "  FAIL: %s\n", msg.c_str());
        }
    }
    int report(const char* name) {
        std::fprintf(stderr, "[%s] %d/%d checks passed%s\n", name,
                     checks - failures, checks, failures ? " -- FAILED" : " -- OK");
        return failures ? 1 : 0;
    }
};

}  // namespace golden
