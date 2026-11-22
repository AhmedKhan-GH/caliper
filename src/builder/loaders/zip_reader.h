#pragma once
// A small, strict reader of zip archives for the npz reader and the torch
// archive check: the central directory (with zip64 records), stored and
// deflated members, and CRC-32 checks. It trusts no size in the archive: the
// directory must lie before its end record, each member's data must lie before
// the directory, members may not overlap, and every count is bounded by
// ReaderLimits. Multi-disk archives and encrypted members are refused.
// Internal to the loaders module.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder {
namespace detail {

struct ZipEntry {
    std::string name;
    uint16_t method = 0;  // 0 stored, 8 deflate
    uint16_t flags = 0;
    uint32_t crc32 = 0;
    uint64_t compressed_size = 0;
    uint64_t size = 0;  // uncompressed
    uint64_t local_offset = 0;
    uint64_t data_offset = 0;  // where the member's data starts
};

// Reads the central directory and each member's local header.
Error read_zip_directory(InputFile& file, const ReaderLimits& limits, std::vector<ZipEntry>* entries);

// Reads a member's uncompressed bytes (stored or deflate), refusing more than
// `max_bytes`, and checks its CRC-32.
Error read_zip_member(InputFile& file, const ZipEntry& entry, uint64_t max_bytes, std::vector<uint8_t>* out);

// Reads the first `n` uncompressed bytes of a member (fewer if it is shorter).
Error read_zip_member_prefix(InputFile& file, const ZipEntry& entry, uint64_t n, std::vector<uint8_t>* out);

// Whether this build can inflate deflated members (it has zlib).
bool zip_has_deflate();

}  // namespace detail
}  // namespace caliper_builder
