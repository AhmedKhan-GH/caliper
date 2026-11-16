// A strict zip reader for npz archives and the torch archive check.
#include "zip_reader.h"

#include <algorithm>
#include <cstring>

#ifdef CALIPER_BUILDER_HAVE_ZLIB
#include <zlib.h>
#endif

namespace caliper_builder {
namespace detail {
namespace {

constexpr uint32_t kEndSig = 0x06054b50;
constexpr uint32_t kEnd64Sig = 0x06064b50;
constexpr uint32_t kLocator64Sig = 0x07064b50;
constexpr uint32_t kCentralSig = 0x02014b50;
constexpr uint32_t kLocalSig = 0x04034b50;

Error malformed(const std::string& where, const std::string& what) {
    return Error("loader.malformed", "The zip archive is malformed: " + what + ".", where);
}

uint32_t crc32_of(const uint8_t* data, std::size_t n) {
    static uint32_t table[256];
    static bool ready = [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return true;
    }();
    (void)ready;
    uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

#ifdef CALIPER_BUILDER_HAVE_ZLIB
// Inflates a raw deflate stream from the file into out, which has room for
// exactly `want` bytes. With `prefix_only` it stops once out is full.
Error inflate_member(InputFile& file, const ZipEntry& e, std::vector<uint8_t>* out, uint64_t want, bool prefix_only) {
    z_stream zs{};
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return Error("loader.io", "zlib could not start.", file.path());
    std::vector<uint8_t> chunk(1 << 20);
    uint64_t in_left = e.compressed_size;
    uint64_t in_pos = e.data_offset;
    uint64_t produced = 0;
    Error err;
    bool done = false;
    while (!done) {
        if (zs.avail_in == 0 && in_left > 0) {
            const uint64_t n = std::min<uint64_t>(in_left, chunk.size());
            if ((err = file.read(in_pos, n, chunk.data()))) break;
            in_pos += n;
            in_left -= n;
            zs.next_in = chunk.data();
            zs.avail_in = static_cast<uInt>(n);
        }
        uint8_t spare = 0;
        const uint64_t room = want - produced;
        // With no room left, offer one spare byte: any output means the member is longer than declared.
        zs.next_out = room ? out->data() + produced : &spare;
        zs.avail_out = room ? static_cast<uInt>(std::min<uint64_t>(room, 1u << 30)) : 1;
        const uInt before = zs.avail_out;
        const int rc = inflate(&zs, Z_NO_FLUSH);
        const uInt wrote = before - zs.avail_out;
        if (room == 0 && wrote > 0) {
            if (prefix_only) {
                done = true;
                break;
            }
            err = malformed(file.path(), "a member inflates to more than its declared size");
            break;
        }
        produced += wrote;
        if (rc == Z_STREAM_END) {
            done = true;
            break;
        }
        if (prefix_only && produced == want) {
            done = true;
            break;
        }
        if (rc == Z_BUF_ERROR && zs.avail_in == 0 && in_left == 0) {
            err = Error("loader.truncated", "A deflated member ends before its stream does.", file.path());
            break;
        }
        if (rc != Z_OK && rc != Z_BUF_ERROR) {
            err = malformed(file.path(), "a deflated member is corrupt");
            break;
        }
    }
    inflateEnd(&zs);
    if (err) return err;
    if (produced != want) {
        if (prefix_only) {
            out->resize(static_cast<std::size_t>(produced));
            return {};
        }
        return malformed(file.path(), "a member inflates to fewer bytes than its declared size");
    }
    return {};
}
#endif

}  // namespace

bool zip_has_deflate() {
#ifdef CALIPER_BUILDER_HAVE_ZLIB
    return true;
#else
    return false;
#endif
}

Error read_zip_directory(InputFile& f, const ReaderLimits& limits, std::vector<ZipEntry>* entries) {
    const std::string& where = f.path();
    const uint64_t size = f.size();
    if (size < 22) return malformed(where, "it is too short to hold an end record");
    const uint64_t tail_len = std::min<uint64_t>(size, 22 + 65535);
    std::vector<uint8_t> tail;
    if (Error e = f.read_vec(size - tail_len, tail_len, &tail, tail_len)) return e;
    int64_t found = -1;
    for (int64_t i = static_cast<int64_t>(tail_len) - 22; i >= 0; --i) {
        const uint8_t* p = tail.data() + i;
        if (le32(p) == kEndSig && static_cast<uint64_t>(i) + 22 + le16(p + 20) == tail_len) {
            found = i;
            break;
        }
    }
    if (found < 0) return malformed(where, "it has no end record");
    const uint8_t* end = tail.data() + found;
    const uint64_t end_pos = size - tail_len + static_cast<uint64_t>(found);
    uint64_t disk = le16(end + 4), cd_disk = le16(end + 6), n_disk = le16(end + 8), n_total = le16(end + 10);
    uint64_t cd_size = le32(end + 12), cd_offset = le32(end + 16);
    uint64_t cd_limit = end_pos;
    if (n_total == 0xFFFF || n_disk == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_offset == 0xFFFFFFFFu) {
        if (end_pos < 20) return malformed(where, "its zip64 locator is missing");
        uint8_t loc[20];
        if (Error e = f.read(end_pos - 20, 20, loc)) return e;
        if (le32(loc) != kLocator64Sig) return malformed(where, "its zip64 locator is missing");
        if (le32(loc + 16) > 1 || le32(loc + 4) != 0) {
            return Error("loader.unsupported", "Zip archives that span several disks are not supported.", where);
        }
        const uint64_t rec = le64(loc + 8);
        if (!fits(rec, 56, end_pos - 20)) return malformed(where, "its zip64 end record lies outside the archive");
        uint8_t z[56];
        if (Error e = f.read(rec, 56, z)) return e;
        if (le32(z) != kEnd64Sig) return malformed(where, "its zip64 end record is missing");
        disk = le32(z + 16);
        cd_disk = le32(z + 20);
        n_disk = le64(z + 24);
        n_total = le64(z + 32);
        cd_size = le64(z + 40);
        cd_offset = le64(z + 48);
        cd_limit = rec;
    }
    if (disk != 0 || cd_disk != 0 || n_disk != n_total) {
        return Error("loader.unsupported", "Zip archives that span several disks are not supported.", where);
    }
    if (n_total > limits.max_zip_members) {
        return Error("loader.too_large", "The archive has more members than the reader's limit.", where);
    }
    if (cd_size > limits.max_zip_directory) {
        return Error("loader.too_large", "The archive's directory is larger than the reader's limit.", where);
    }
    if (!fits(cd_offset, cd_size, cd_limit)) return malformed(where, "its directory lies outside the archive");
    if (n_total > cd_size / 46) return malformed(where, "its directory is too small for its member count");
    std::vector<uint8_t> cd;
    if (Error e = f.read_vec(cd_offset, cd_size, &cd, limits.max_zip_directory)) return e;

    std::vector<ZipEntry> out;
    out.reserve(static_cast<std::size_t>(n_total));
    uint64_t pos = 0;
    for (uint64_t k = 0; k < n_total; ++k) {
        if (!fits(pos, 46, cd_size)) return malformed(where, "its directory is cut short");
        const uint8_t* p = cd.data() + pos;
        if (le32(p) != kCentralSig) return malformed(where, "a directory entry has the wrong signature");
        ZipEntry e;
        e.flags = le16(p + 8);
        e.method = le16(p + 10);
        e.crc32 = le32(p + 16);
        e.compressed_size = le32(p + 20);
        e.size = le32(p + 24);
        const uint64_t name_len = le16(p + 28), extra_len = le16(p + 30), comment_len = le16(p + 32);
        uint64_t disk_start = le16(p + 34);
        e.local_offset = le32(p + 42);
        const uint64_t var_len = name_len + extra_len + comment_len;
        if (!fits(pos + 46, var_len, cd_size)) return malformed(where, "a directory entry is cut short");
        e.name.assign(reinterpret_cast<const char*>(p + 46), static_cast<std::size_t>(name_len));
        if (e.name.find('\0') != std::string::npos) return malformed(where, "a member name holds a NUL byte");
        // The zip64 extra field holds, in order, each value whose 32-bit field is all ones.
        const uint8_t* x = p + 46 + name_len;
        uint64_t xpos = 0;
        while (xpos + 4 <= extra_len) {
            const uint16_t id = le16(x + xpos), len = le16(x + xpos + 2);
            if (xpos + 4 + len > extra_len) return malformed(where, "an extra field is cut short");
            if (id == 0x0001) {
                const uint8_t* d = x + xpos + 4;
                uint64_t dpos = 0;
                auto take = [&](uint64_t* v) -> bool {
                    if (dpos + 8 > len) return false;
                    *v = le64(d + dpos);
                    dpos += 8;
                    return true;
                };
                if (e.size == 0xFFFFFFFFu && !take(&e.size)) return malformed(where, "a zip64 field is cut short");
                if (e.compressed_size == 0xFFFFFFFFu && !take(&e.compressed_size)) {
                    return malformed(where, "a zip64 field is cut short");
                }
                if (e.local_offset == 0xFFFFFFFFu && !take(&e.local_offset)) {
                    return malformed(where, "a zip64 field is cut short");
                }
                if (disk_start == 0xFFFF) {
                    if (dpos + 4 > len) return malformed(where, "a zip64 field is cut short");
                    disk_start = le32(d + dpos);
                }
            }
            xpos += 4 + len;
        }
        if (e.size == 0xFFFFFFFFu || e.compressed_size == 0xFFFFFFFFu || e.local_offset == 0xFFFFFFFFu) {
            // A real value of exactly 0xFFFFFFFF needs zip64 too, so an all-ones value without it is malformed.
            return malformed(where, "a member lacks the zip64 field its sizes need");
        }
        if (disk_start != 0) return Error("loader.unsupported", "Zip archives that span several disks are not supported.", where);
        out.push_back(std::move(e));
        pos += 46 + var_len;
    }
    // Local headers: each lies before the directory and gives where the data starts.
    for (ZipEntry& e : out) {
        uint8_t lh[30];
        if (!fits(e.local_offset, 30, cd_offset)) return malformed(where, "a member's header lies outside the archive");
        if (Error err = f.read(e.local_offset, 30, lh)) return err;
        if (le32(lh) != kLocalSig) return malformed(where, "a member's header has the wrong signature");
        e.data_offset = e.local_offset + 30 + le16(lh + 26) + le16(lh + 28);
        if (!fits(e.data_offset, e.compressed_size, cd_offset)) {
            return malformed(where, "a member's data lies outside the archive");
        }
    }
    // Members may not overlap, which closes the overlapping-member zip bomb.
    std::vector<const ZipEntry*> sorted;
    for (const ZipEntry& e : out) sorted.push_back(&e);
    std::sort(sorted.begin(), sorted.end(),
              [](const ZipEntry* a, const ZipEntry* b) { return a->local_offset < b->local_offset; });
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i - 1]->data_offset + sorted[i - 1]->compressed_size > sorted[i]->local_offset) {
            return malformed(where, "two members overlap");
        }
    }
    *entries = std::move(out);
    return {};
}

Error read_zip_member(InputFile& f, const ZipEntry& e, uint64_t max_bytes, std::vector<uint8_t>* out) {
    const std::string& where = f.path();
    if (e.flags & 1) return Error("loader.unsupported", "The member " + e.name + " is encrypted.", where);
    if (e.size > max_bytes) {
        return Error("loader.too_large", "The member " + e.name + " is larger than the reader's limit.", where);
    }
    if (e.method == 0) {
        if (e.compressed_size != e.size) return malformed(where, "a stored member's two sizes differ");
        if (Error err = f.read_vec(e.data_offset, e.size, out, max_bytes)) return err;
    } else if (e.method == 8) {
#ifdef CALIPER_BUILDER_HAVE_ZLIB
        try {
            out->resize(static_cast<std::size_t>(e.size));
        } catch (...) {
            return Error("loader.too_large", "Memory for the member could not be allocated.", where);
        }
        if (Error err = inflate_member(f, e, out, e.size, false)) return err;
#else
        return Error("loader.unsupported", "This build has no zlib, so it cannot inflate the member " + e.name + ".",
                     where);
#endif
    } else {
        return Error("loader.unsupported",
                     "The member " + e.name + " uses compression method " + std::to_string(e.method) +
                         "; only stored and deflate are supported.",
                     where);
    }
    if (crc32_of(out->data(), out->size()) != e.crc32) {
        return malformed(where, "the member " + e.name + " fails its CRC-32 check");
    }
    return {};
}

Error read_zip_member_prefix(InputFile& f, const ZipEntry& e, uint64_t n, std::vector<uint8_t>* out) {
    const std::string& where = f.path();
    if (e.flags & 1) return Error("loader.unsupported", "The member " + e.name + " is encrypted.", where);
    const uint64_t want = std::min(n, e.size);
    if (e.method == 0) {
        if (e.compressed_size != e.size) return malformed(where, "a stored member's two sizes differ");
        return f.read_vec(e.data_offset, want, out, want);
    }
    if (e.method == 8) {
#ifdef CALIPER_BUILDER_HAVE_ZLIB
        out->resize(static_cast<std::size_t>(want));
        return inflate_member(f, e, out, want, true);
#else
        return Error("loader.unsupported", "This build has no zlib, so it cannot inflate the member " + e.name + ".",
                     where);
#endif
    }
    return Error("loader.unsupported",
                 "The member " + e.name + " uses compression method " + std::to_string(e.method) +
                     "; only stored and deflate are supported.",
                 where);
}

}  // namespace detail
}  // namespace caliper_builder
