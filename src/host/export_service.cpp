// The host's PNG writers: an atomic PNG file write and an in-memory encode.
// The single STB_IMAGE_WRITE_IMPLEMENTATION TU in the tree (README pins v1.16).
#include "export_service.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
// The pinned stb_image_write calls sprintf, which the macOS SDK marks deprecated; the warning is about the
// third-party code, which stays as it is pinned.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <stb_image_write.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <cstdio>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace caliper_host {
namespace {

// Every refusal below says why when the caller asked (`why` may be null).
bool refuse(std::string* why, std::string text) {
    if (why) *why = std::move(text);
    return false;
}

// Encode tightly-packed top-down RGBA8 to `file` as PNG; on failure the
// partial file is removed. stb writes row 0 first, and the renderer readback
// already hands back row 0 = top of the image (stride = w*4).
bool encode_png_file(const fs::path& file, const uint8_t* rgba, uint32_t w,
                     uint32_t h, std::string* why) {
    const int rc = stbi_write_png(file.string().c_str(), static_cast<int>(w),
                                  static_cast<int>(h), 4, rgba,
                                  static_cast<int>(w * 4));
    if (rc != 0) return true;
    std::error_code ec;
    fs::remove(file, ec);
    return refuse(why, "cannot write the PNG " + file.string());
}

std::string rename_failure(const fs::path& from, const fs::path& to,
                           const std::error_code& ec) {
    return "cannot rename " + from.string() + " onto " + to.string() + ": " +
           ec.message();
}

fs::path with_suffix(const fs::path& p, const char* suffix) {
    fs::path out = p;
    out += suffix;
    return out;
}

}  // namespace

bool export_write_png_atomic(const std::string& path,
                             const uint8_t* rgba, uint32_t w, uint32_t h,
                             std::string* why) {
    if (!rgba || w == 0 || h == 0) return refuse(why, "no pixels to write");
    fs::path target(path);
    fs::path tmp = with_suffix(target, ".caliper_tmp");
    if (!encode_png_file(tmp, rgba, w, h, why)) return false;
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        return refuse(why, rename_failure(tmp, target, ec));
    }
    return true;
}

bool export_encode_png(const uint8_t* rgba, uint32_t w, uint32_t h, std::string* out) {
    if (!rgba || w == 0 || h == 0 || !out) return false;
    out->clear();
    auto append = [](void* ctx, void* data, int size) {
        static_cast<std::string*>(ctx)->append(static_cast<const char*>(data), (size_t)size);
    };
    return stbi_write_png_to_func(append, out, static_cast<int>(w), static_cast<int>(h), 4, rgba,
                                  static_cast<int>(w * 4)) != 0;
}

}  // namespace caliper_host
