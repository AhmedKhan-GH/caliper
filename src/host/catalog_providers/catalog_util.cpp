#include "catalog_util.h"

#include <cmath>
#include <cstdio>
#include <ctime>

namespace caliper_host::catalog_detail {

namespace {

bool valid_utf8(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t n = 0;
        if (c < 0x80) n = 1;
        else if ((c & 0xE0) == 0xC0 && c >= 0xC2) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0 && c <= 0xF4) n = 4;
        else return false;
        if (i + n > s.size()) return false;
        for (std::size_t k = 1; k < n; ++k)
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        i += n;
    }
    return true;
}

bool has_control(const std::string& s) {
    for (unsigned char c : s)
        if (c < 0x20 || c == 0x7F) return true;
    return false;
}

}  // namespace

std::string cut_utf8(const std::string& text, std::size_t max) {
    if (text.size() <= max) return text;
    std::size_t n = max;
    while (n > 0 && (static_cast<unsigned char>(text[n]) & 0xC0) == 0x80) --n;
    return text.substr(0, n);
}

bool is_catalog_name(const std::string& text) {
    return !text.empty() && text.size() <= caliper_builder::CatalogLimits::max_name && valid_utf8(text) &&
           !has_control(text) && text.front() != ' ' && text.back() != ' ';
}

std::optional<std::string> origin_value(const std::string& text) {
    if (text.empty() || !valid_utf8(text) || has_control(text)) return std::nullopt;
    return cut_utf8(text, caliper_builder::CatalogLimits::max_origin_value);
}

std::string title_or(const std::string& text, const std::string& fallback) {
    if (is_catalog_name(text)) return text;
    const std::string cut = cut_utf8(text, caliper_builder::CatalogLimits::max_name);
    return is_catalog_name(cut) ? cut : fallback;
}

std::string utc_text_ms(std::chrono::system_clock::time_point t) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(t.time_since_epoch()).count();
    const std::time_t secs = static_cast<std::time_t>(ms / 1000);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms % 1000));
    return buf;
}

nlohmann::json float_cell(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return v;
}

caliper_builder::CatalogListResult filter_entries(const std::vector<caliper_builder::CatalogEntry>& all,
                                                  const caliper_builder::CatalogFilter& filter) {
    caliper_builder::CatalogListResult out;
    for (const auto& e : all) {
        if (filter.kind && e.kind != *filter.kind) continue;
        if (e.name.compare(0, filter.prefix.size(), filter.prefix) != 0) continue;
        if (out.entries.size() >= filter.limit) {
            out.truncated = true;
            break;
        }
        out.entries.push_back(e);
    }
    return out;
}

}  // namespace caliper_host::catalog_detail
