// Small codecs of the mcp module (codec.h).
#include "codec.h"

#include <filesystem>
#include <fstream>
#include <system_error>

namespace caliper_builder::mcp::detail {

namespace {
const char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
}  // namespace

std::string base64_encode(std::string_view bytes) {
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const std::uint32_t v = (std::uint32_t(std::uint8_t(bytes[i])) << 16) |
                                (std::uint32_t(std::uint8_t(bytes[i + 1])) << 8) | std::uint8_t(bytes[i + 2]);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += kAlphabet[v & 63];
    }
    const std::size_t rest = bytes.size() - i;
    if (rest == 1) {
        const std::uint32_t v = std::uint32_t(std::uint8_t(bytes[i])) << 16;
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += "==";
    } else if (rest == 2) {
        const std::uint32_t v = (std::uint32_t(std::uint8_t(bytes[i])) << 16) | (std::uint32_t(std::uint8_t(bytes[i + 1])) << 8);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

bool base64_decode(std::string_view text, std::string* out) {
    out->clear();
    if (text.size() % 4 != 0) return false;
    out->reserve(text.size() / 4 * 3);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        const bool last = i + 4 == text.size();
        int v[4];
        int pad = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = text[i + k];
            if (c == '=' && last && k >= 2) {
                v[k] = 0;
                ++pad;
                continue;
            }
            if (pad > 0) return false;   // a character after padding
            v[k] = b64_value(c);
            if (v[k] < 0) return false;
        }
        const std::uint32_t w = (std::uint32_t(v[0]) << 18) | (std::uint32_t(v[1]) << 12) | (std::uint32_t(v[2]) << 6) |
                                std::uint32_t(v[3]);
        out->push_back(char((w >> 16) & 0xff));
        if (pad < 2) out->push_back(char((w >> 8) & 0xff));
        if (pad < 1) out->push_back(char(w & 0xff));
    }
    return true;
}

bool png_size(std::string_view b, std::uint32_t* width, std::uint32_t* height) {
    static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    if (b.size() < 24) return false;
    for (int i = 0; i < 8; ++i)
        if (std::uint8_t(b[i]) != sig[i]) return false;
    if (b.substr(12, 4) != "IHDR") return false;
    auto be32 = [&](std::size_t at) {
        return (std::uint32_t(std::uint8_t(b[at])) << 24) | (std::uint32_t(std::uint8_t(b[at + 1])) << 16) |
               (std::uint32_t(std::uint8_t(b[at + 2])) << 8) | std::uint32_t(std::uint8_t(b[at + 3]));
    };
    *width = be32(16);
    *height = be32(20);
    return *width > 0 && *height > 0;
}

bool json_depth_ok(std::string_view text, int max_depth) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (const char c : text) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '[' || c == '{') {
            if (++depth > max_depth) return false;
        } else if (c == ']' || c == '}') {
            if (depth > 0) --depth;
        }
    }
    return true;
}

bool read_small_file(const std::string& path, std::size_t max_bytes, std::string* out, std::string* why) {
    std::error_code ec;
    const auto status = std::filesystem::status(path, ec);
    if (ec || !std::filesystem::exists(status)) {
        *why = "it does not exist";
        return false;
    }
    if (!std::filesystem::is_regular_file(status)) {
        *why = "it is not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        *why = "its size cannot be read";
        return false;
    }
    if (size > max_bytes) {
        *why = "it is larger than " + std::to_string(max_bytes) + " bytes";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *why = "it cannot be opened";
        return false;
    }
    out->assign(std::size_t(size), '\0');
    in.read(out->data(), std::streamsize(size));
    out->resize(std::size_t(in.gcount()));
    // The file may have grown since its size was read; anything past the
    // bound is not read.
    if (in.peek() != std::char_traits<char>::eof()) {
        *why = "it grew past " + std::to_string(max_bytes) + " bytes while it was read";
        out->clear();
        return false;
    }
    return true;
}

void redact(std::string* text, const std::string& secret) {
    if (secret.size() < 8) return;
    std::size_t pos = 0;
    while ((pos = text->find(secret, pos)) != std::string::npos) {
        text->replace(pos, secret.size(), "<redacted>");
        pos += 10;
    }
}

}  // namespace caliper_builder::mcp::detail
