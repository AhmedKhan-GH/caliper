#pragma once
// Small helpers shared by the session module's sources. Not a public header.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace caliper_builder::session_detail {

// Cuts a UTF-8 string to at most `max` bytes at a character boundary.
std::string cut_utf8(const std::string& text, std::size_t max);

// Writes the non-finite values of floating-point columns as the catalog and
// the command table spell them: "nan", "inf" and "-inf" (the store writes
// "NaN", "Infinity" and "-Infinity").
void fix_nonfinite(QueryResult& result);

// "2026-11-15T12:00:00Z" for a time in seconds since the epoch.
std::string utc_text(std::int64_t seconds);

// The modification time of a file, as utc_text(), or nothing.
std::optional<std::string> file_mtime_utc(const std::string& path);

// Reads a whole file of at most `max` bytes. Fails with `too_large_code` when
// it is larger, and with session.io when it cannot be read.
Error read_file_bounded(const std::string& path, std::size_t max, std::string* out,
                        const std::string& too_large_code = "session.too_large");

// Writes a file through a temporary name in the same folder and a rename, so
// a reader sees the old text or the new one, never a mix.
Error write_file_atomic(const std::string& path, const std::string& text);

}  // namespace caliper_builder::session_detail
