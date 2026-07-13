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

// Write `bytes` to a sibling temp file then rename onto `path`. Nothing is
// created AT `path` unless the whole write+rename succeeds.
bool write_bytes_atomic(const std::string& path, const void* data, size_t n) {
    fs::path target(path);
    fs::path tmp = target;
    tmp += ".caliper_tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        if (n) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        f.flush();
        if (!f.good()) {
            f.close();
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        return false;
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
                             const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (!rgba || w == 0 || h == 0) return false;
    fs::path target(path);
    fs::path tmp = target;
    tmp += ".caliper_tmp";
    // Tightly-packed rows (stride = w*4), top-down: stb writes row 0 first, and
    // the renderer readback already hands back row 0 = top of the image.
    const int rc = stbi_write_png(tmp.string().c_str(), static_cast<int>(w),
                                  static_cast<int>(h), 4, rgba,
                                  static_cast<int>(w * 4));
    if (rc == 0) {
        std::error_code ec;
        fs::remove(tmp, ec);
        return false;
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2);
        return false;
    }
    return true;
}

bool export_write_text_atomic(const std::string& path, const std::string& text) {
    return write_bytes_atomic(path, text.data(), text.size());
}

}  // namespace caliper_host
