// caliper.export.v1 — PNG encode + atomic writes + provenance sidecar builder.
// The single STB_IMAGE_WRITE_IMPLEMENTATION TU in the tree (README pins v1.16).
#include "export_service.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace fs = std::filesystem;

namespace caliper_host {
namespace {

void json_escape(const std::string& s, std::string& out) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

void append_quoted(const std::string& s, std::string& out) {
    out += '"';
    json_escape(s, out);
    out += '"';
}

// %.9g round-trips a float32 exactly; keeps identity matrices as "1"/"0".
void append_float(float v, std::string& out) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    out += buf;
}

void append_mat16(const float* m, std::string& out) {
    out += '[';
    for (int i = 0; i < 16; ++i) {
        if (i) out += ", ";
        if (m) append_float(m[i], out);
        else   out += '0';
    }
    out += ']';
}

// Every refusal below says why when the caller asked (`why` may be null).
bool refuse(std::string* why, std::string text) {
    if (why) *why = std::move(text);
    return false;
}

// Write `n` bytes to `file` (created/truncated); on failure the partial file
// is removed. The staging half of every atomic write below.
bool write_file(const fs::path& file, const void* data, size_t n,
                std::string* why) {
    {
        std::ofstream f(file, std::ios::binary | std::ios::trunc);
        if (!f) return refuse(why, "cannot create " + file.string());
        if (n) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        f.flush();
        if (f.good()) return true;
    }
    std::error_code ec;
    fs::remove(file, ec);
    return refuse(why, "cannot write " + file.string());
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

// Write `bytes` to a sibling temp file then rename onto `path`. Nothing is
// created AT `path` unless the whole write+rename succeeds.
bool write_bytes_atomic(const std::string& path, const void* data, size_t n,
                        std::string* why) {
    fs::path target(path);
    fs::path tmp = with_suffix(target, ".caliper_tmp");
    if (!write_file(tmp, data, n, why)) return false;
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        return refuse(why, rename_failure(tmp, target, ec));
    }
    return true;
}

}  // namespace

std::string export_build_sidecar_json(const ExportProvenance& p) {
    std::string o;
    o.reserve(512);
    o += "{\n";
    o += "  \"caliper\": {\n";
    o += "    \"version\": ";     append_quoted(p.version, o);     o += ",\n";
    o += "    \"git_commit\": ";  append_quoted(p.git_commit, o);  o += ",\n";
    o += "    \"backend\": ";     append_quoted(p.backend, o);     o += ",\n";
    o += "    \"platform\": ";    append_quoted(p.platform, o);    o += "\n";
    o += "  },\n";
    o += "  \"timestamp_utc\": "; append_quoted(p.timestamp_utc, o); o += ",\n";
    o += "  \"width\": " + std::to_string(p.width) + ",\n";
    o += "  \"height\": " + std::to_string(p.height) + ",\n";
    o += "  \"clear_rgba\": " + std::to_string(p.clear_rgba) + ",\n";
    o += "  \"camera\": {\n";
    o += "    \"view\": "; append_mat16(p.view16, o); o += ",\n";
    o += "    \"proj\": "; append_mat16(p.proj16, o); o += "\n";
    o += "  },\n";
    o += "  \"draw_count\": " + std::to_string(p.draw_count) + ",\n";
    o += "  \"colormaps\": [";
    for (size_t i = 0; i < p.colormaps.size(); ++i) {
        if (i) o += ", ";
        o += std::to_string(p.colormaps[i]);
    }
    o += "],\n";
    if (p.is_sequence) {
        o += "  \"frame_count\": " + std::to_string(p.frame_count) + ",\n";
    }
    o += "  \"state\": ";
    if (p.state_json) o += p.state_json;   // verbatim (caller-owned JSON)
    else              o += "null";
    o += "\n";
    o += "}\n";
    return o;
}

std::string export_utc_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

const char* export_platform_string() {
#if defined(__APPLE__)
    return "macos";
#elif defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

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

bool export_write_text_atomic(const std::string& path, const std::string& text,
                              std::string* why) {
    return write_bytes_atomic(path, text.data(), text.size(), why);
}

bool export_write_png_with_sidecar_atomic(const std::string& png_path,
                                          const uint8_t* rgba, uint32_t w,
                                          uint32_t h,
                                          const std::string& sidecar_path,
                                          const std::string& json,
                                          std::string* why) {
    if (!rgba || w == 0 || h == 0) return refuse(why, "no pixels to write");
    const fs::path png(png_path), side(sidecar_path);
    std::error_code ec;
    // A directory can't be replaced by rename, and must not be set aside as a
    // "previous PNG" either — refuse before anything moves.
    if (fs::is_directory(png, ec))
        return refuse(why, png.string() + " is a directory");
    const fs::path png_tmp  = with_suffix(png,  ".caliper_tmp");
    const fs::path side_tmp = with_suffix(side, ".caliper_tmp");
    const fs::path png_bak  = with_suffix(png,  ".caliper_bak");
    auto discard = [&] {
        std::error_code e;
        fs::remove(png_tmp, e);
        fs::remove(side_tmp, e);
    };

    // Stage both. A failure here has touched nothing the user can see.
    if (!encode_png_file(png_tmp, rgba, w, h, why)) return false;
    if (!write_file(side_tmp, json.data(), json.size(), why)) {
        discard();
        return false;
    }

    // Commit: previous PNG aside, new PNG in, sidecar in, backup gone. Every
    // failure path puts the previous PNG back — the E1 fix "rolled back" by
    // deleting whatever sat at png_path, i.e. the user's last good figure.
    const bool had_png = fs::exists(png, ec);
    if (had_png) {
        fs::rename(png, png_bak, ec);
        if (ec) { discard(); return refuse(why, rename_failure(png, png_bak, ec)); }
    }
    auto restore_previous = [&] {
        std::error_code e;
        if (had_png) fs::rename(png_bak, png, e);
        else         fs::remove(png, e);
    };
    fs::rename(png_tmp, png, ec);
    if (ec) {
        const std::string text = rename_failure(png_tmp, png, ec);
        restore_previous();
        discard();
        return refuse(why, text);
    }
    fs::rename(side_tmp, side, ec);
    if (ec) {
        const std::string text = rename_failure(side_tmp, side, ec);
        restore_previous();
        discard();
        return refuse(why, text);
    }
    if (had_png) fs::remove(png_bak, ec);
    return true;
}

}  // namespace caliper_host
