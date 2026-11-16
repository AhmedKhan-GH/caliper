// catalog_util.h: small helpers shared by the host's catalog providers
// (live_catalog.h, feed_catalog.h, metrics_catalog.h). Host-internal.
#pragma once

#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/catalog/catalog_entry.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_host::catalog_detail {

// Cuts UTF-8 text to at most `max` bytes at a character boundary.
std::string cut_utf8(const std::string& text, std::size_t max);

// True when `text` is a valid catalog name (1 to 256 bytes of UTF-8, no
// control character, no space at either end), so that an id made from it
// passes the catalog's strict check.
bool is_catalog_name(const std::string& text);

// An origin value: the text cut to 1024 bytes, or nothing when it is empty or
// holds a control character or bad UTF-8.
std::optional<std::string> origin_value(const std::string& text);

// A title from `text`, else `fallback` (which must be a valid name).
std::string title_or(const std::string& text, const std::string& fallback);

// "2026-11-15T12:00:00.123Z" for a wall-clock time.
std::string utc_text_ms(std::chrono::system_clock::time_point t);

// A floating-point cell: the number, or "nan", "inf" and "-inf".
nlohmann::json float_cell(double v);

// The entries of `all` (sorted by id) that match the filter, at most
// filter.limit of them, with `truncated` set when more matched.
caliper_builder::CatalogListResult filter_entries(const std::vector<caliper_builder::CatalogEntry>& all,
                                                  const caliper_builder::CatalogFilter& filter);

}  // namespace caliper_host::catalog_detail
