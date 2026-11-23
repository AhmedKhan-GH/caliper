#include "util.h"

#include <caliper_builder/loaders/paths.h>

#include <atomic>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

namespace caliper_builder::session_detail {

namespace fs = std::filesystem;

std::string cut_utf8(const std::string& text, std::size_t max) {
    if (text.size() <= max) return text;
    std::size_t end = max;
    // Step back over continuation bytes so the cut lands on a boundary.
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) --end;
    return text.substr(0, end);
}

void fix_nonfinite(QueryResult& result) {
    for (std::size_t c = 0; c < result.columns.size(); ++c) {
        const std::string& type = result.columns[c].type;
        if (type != "DOUBLE" && type != "FLOAT" && type != "REAL" && type != "FLOAT4" && type != "FLOAT8") continue;
        for (auto& row : result.rows) {
            if (!row.is_array() || c >= row.size() || !row[c].is_string()) continue;
            const std::string& s = row[c].get_ref<const std::string&>();
            if (s == "NaN") row[c] = "nan";
            else if (s == "Infinity") row[c] = "inf";
            else if (s == "-Infinity") row[c] = "-inf";
        }
    }
}

std::string utc_text(std::int64_t seconds) {
    const std::time_t t = static_cast<std::time_t>(seconds);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

std::optional<std::string> file_mtime_utc(const std::string& path) {
#ifdef _WIN32
    struct _stat64 st;
    if (_wstat64(fs::u8path(path).wstring().c_str(), &st) != 0) return std::nullopt;
#else
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return std::nullopt;
#endif
    return utc_text(static_cast<std::int64_t>(st.st_mtime));
}

Error read_file_bounded(const std::string& path, std::size_t max, std::string* out, const std::string& too_large_code) {
    std::error_code ec;
    const fs::path p = fs::u8path(path);
    const auto size = fs::file_size(p, ec);
    if (ec) return Error("session.io", "The file could not be read.", path);
    if (size > max)
        return Error(too_large_code, "The file is larger than its limit of " + std::to_string(max) + " bytes.", path);
    std::ifstream in(p, std::ios::binary);
    if (!in) return Error("session.io", "The file could not be opened.", path);
    std::string text(static_cast<std::size_t>(size), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (static_cast<std::size_t>(in.gcount()) != text.size()) return Error("session.io", "The file could not be read.", path);
    // A file that grew since its size was taken is refused rather than read in part.
    if (in.peek() != std::ifstream::traits_type::eof())
        return Error(too_large_code, "The file changed while it was read.", path);
    *out = std::move(text);
    return {};
}

Error write_file_atomic(const std::string& path, const std::string& text) {
    static std::atomic<unsigned> counter{0};
    const fs::path target = fs::u8path(path);
    std::error_code ec;
    if (touches_protected_file(path))
        return Error("session.protected_file", "The file is, or leads to, a protected settings file, so it was not written.",
                     path);
    fs::create_directories(target.parent_path(), ec);
#ifdef _WIN32
    const long pid = static_cast<long>(_getpid());
#else
    const long pid = static_cast<long>(::getpid());
#endif
    const fs::path temp = target.parent_path() / fs::u8path("." + target.filename().u8string() + ".tmp-" +
                                                            std::to_string(pid) + "-" + std::to_string(counter++));
    // A name that is already there, such as a link, is removed, not written through.
    fs::remove(temp, ec);
    if (touches_protected_file(temp.u8string()))
        return Error("session.protected_file", "The file is, or leads to, a protected settings file, so it was not written.",
                     path);
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) return Error("session.io", "The file could not be written.", path);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if (!out) {
            out.close();
            fs::remove(temp, ec);
            return Error("session.io", "The file could not be written.", path);
        }
    }
    fs::rename(temp, target, ec);
    if (ec) {
        // Windows refuses to rename over an existing file in some cases.
        std::error_code ec2;
        fs::remove(target, ec2);
        fs::rename(temp, target, ec2);
        if (ec2) {
            fs::remove(temp, ec2);
            return Error("session.io", "The file could not be put in place.", path);
        }
    }
    return {};
}

}  // namespace caliper_builder::session_detail
