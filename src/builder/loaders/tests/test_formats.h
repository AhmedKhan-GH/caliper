#pragma once
// Builders of small test files in code: .npy files with any header, zip
// archives with stored or deflated members (and zip64 records or deliberate
// faults), safetensors files and WAV files.
#include "test_util.h"

#ifdef CALIPER_BUILDER_HAVE_ZLIB
#include <zlib.h>
#endif

#include <string>
#include <vector>

namespace loaders_test {

inline std::vector<uint8_t> make_npy(const std::string& dict, const std::vector<uint8_t>& data, int version = 1) {
    std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', static_cast<uint8_t>(version), 0};
    std::string header = dict;
    const std::size_t prefix = version == 1 ? 10 : 12;
    std::size_t total = prefix + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';
    if (version == 1) {
        put<uint16_t>(out, static_cast<uint16_t>(header.size()));
    } else {
        put<uint32_t>(out, static_cast<uint32_t>(header.size()));
    }
    append(out, header);
    append(out, data);
    return out;
}

inline uint32_t test_crc32(const std::vector<uint8_t>& data) {
    uint32_t c = 0xFFFFFFFFu;
    for (uint8_t b : data) {
        c ^= b;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    }
    return c ^ 0xFFFFFFFFu;
}

#ifdef CALIPER_BUILDER_HAVE_ZLIB
inline std::vector<uint8_t> raw_deflate(const std::vector<uint8_t>& in) {
    z_stream zs{};
    deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
    std::vector<uint8_t> out(deflateBound(&zs, static_cast<uLong>(in.size())) + 16);
    zs.next_in = const_cast<Bytef*>(in.data());
    zs.avail_in = static_cast<uInt>(in.size());
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(out.size());
    deflate(&zs, Z_FINISH);
    out.resize(zs.total_out);
    deflateEnd(&zs);
    return out;
}
#else
inline std::vector<uint8_t> raw_deflate(const std::vector<uint8_t>& in) { return in; }
#endif

struct ZipMember {
    std::string name;
    std::vector<uint8_t> data;
    bool deflate = false;
    int method = -1;  // overrides the method field when >= 0
    uint16_t flags = 0;
    bool bad_crc = false;
};

struct ZipOptions {
    bool zip64_extras = false;  // sizes and offsets in zip64 extra fields, as numpy writes
    bool zip64_end = false;     // a zip64 end record and locator
    bool overlap = false;       // the second member's directory entry points at the first
};

inline std::vector<uint8_t> make_zip(const std::vector<ZipMember>& members, ZipOptions opt = {}) {
    struct Rec {
        std::string name;
        uint16_t method, flags;
        uint32_t crc;
        uint64_t csize, usize, offset;
    };
    std::vector<uint8_t> out;
    std::vector<Rec> recs;
    for (const ZipMember& m : members) {
        const std::vector<uint8_t> comp = m.deflate ? raw_deflate(m.data) : m.data;
        Rec r;
        r.name = m.name;
        r.method = static_cast<uint16_t>(m.method >= 0 ? m.method : (m.deflate ? 8 : 0));
        r.flags = m.flags;
        r.crc = test_crc32(m.data) ^ (m.bad_crc ? 1u : 0u);
        r.csize = comp.size();
        r.usize = m.data.size();
        r.offset = out.size();
        put<uint32_t>(out, 0x04034b50);
        put<uint16_t>(out, opt.zip64_extras ? 45 : 20);
        put<uint16_t>(out, r.flags);
        put<uint16_t>(out, r.method);
        put<uint32_t>(out, 0);  // time and date
        put<uint32_t>(out, r.crc);
        put<uint32_t>(out, opt.zip64_extras ? 0xFFFFFFFFu : static_cast<uint32_t>(r.csize));
        put<uint32_t>(out, opt.zip64_extras ? 0xFFFFFFFFu : static_cast<uint32_t>(r.usize));
        put<uint16_t>(out, static_cast<uint16_t>(r.name.size()));
        put<uint16_t>(out, opt.zip64_extras ? 20 : 0);
        append(out, r.name);
        if (opt.zip64_extras) {
            put<uint16_t>(out, 1);
            put<uint16_t>(out, 16);
            put<uint64_t>(out, r.usize);
            put<uint64_t>(out, r.csize);
        }
        append(out, comp);
        recs.push_back(r);
    }
    if (opt.overlap && recs.size() >= 2) recs[1].offset = recs[0].offset;
    const uint64_t cd_offset = out.size();
    for (const Rec& r : recs) {
        put<uint32_t>(out, 0x02014b50);
        put<uint16_t>(out, 45);
        put<uint16_t>(out, opt.zip64_extras ? 45 : 20);
        put<uint16_t>(out, r.flags);
        put<uint16_t>(out, r.method);
        put<uint32_t>(out, 0);
        put<uint32_t>(out, r.crc);
        put<uint32_t>(out, opt.zip64_extras ? 0xFFFFFFFFu : static_cast<uint32_t>(r.csize));
        put<uint32_t>(out, opt.zip64_extras ? 0xFFFFFFFFu : static_cast<uint32_t>(r.usize));
        put<uint16_t>(out, static_cast<uint16_t>(r.name.size()));
        put<uint16_t>(out, opt.zip64_extras ? 28 : 0);
        put<uint16_t>(out, 0);  // comment
        put<uint16_t>(out, 0);  // disk
        put<uint16_t>(out, 0);  // internal attributes
        put<uint32_t>(out, 0);  // external attributes
        put<uint32_t>(out, opt.zip64_extras ? 0xFFFFFFFFu : static_cast<uint32_t>(r.offset));
        append(out, r.name);
        if (opt.zip64_extras) {
            put<uint16_t>(out, 1);
            put<uint16_t>(out, 24);
            put<uint64_t>(out, r.usize);
            put<uint64_t>(out, r.csize);
            put<uint64_t>(out, r.offset);
        }
    }
    const uint64_t cd_size = out.size() - cd_offset;
    if (opt.zip64_end) {
        const uint64_t rec = out.size();
        put<uint32_t>(out, 0x06064b50);
        put<uint64_t>(out, 44);
        put<uint16_t>(out, 45);
        put<uint16_t>(out, 45);
        put<uint32_t>(out, 0);
        put<uint32_t>(out, 0);
        put<uint64_t>(out, recs.size());
        put<uint64_t>(out, recs.size());
        put<uint64_t>(out, cd_size);
        put<uint64_t>(out, cd_offset);
        put<uint32_t>(out, 0x07064b50);
        put<uint32_t>(out, 0);
        put<uint64_t>(out, rec);
        put<uint32_t>(out, 1);
    }
    put<uint32_t>(out, 0x06054b50);
    put<uint16_t>(out, 0);
    put<uint16_t>(out, 0);
    put<uint16_t>(out, opt.zip64_end ? 0xFFFF : static_cast<uint16_t>(recs.size()));
    put<uint16_t>(out, opt.zip64_end ? 0xFFFF : static_cast<uint16_t>(recs.size()));
    put<uint32_t>(out, opt.zip64_end ? 0xFFFFFFFFu : static_cast<uint32_t>(cd_size));
    put<uint32_t>(out, opt.zip64_end ? 0xFFFFFFFFu : static_cast<uint32_t>(cd_offset));
    put<uint16_t>(out, 0);
    return out;
}

inline std::vector<uint8_t> make_safetensors(const std::string& header, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> out;
    put<uint64_t>(out, header.size());
    append(out, header);
    append(out, data);
    return out;
}

struct WavSpec {
    uint16_t format = 1;
    uint16_t channels = 1;
    uint32_t rate = 8000;
    uint16_t bits = 16;
    bool extensible = false;
    uint16_t sub_format = 1;
    bool bad_guid = false;
    int64_t block_align = -1;  // overrides when >= 0
    int64_t data_size = -1;    // overrides the data chunk's size field when >= 0
    bool list_chunk = false;   // an odd-sized chunk before "fmt "
};

inline std::vector<uint8_t> make_wav(const WavSpec& s, const std::vector<uint8_t>& samples) {
    std::vector<uint8_t> body;
    append(body, std::string("WAVE"));
    if (s.list_chunk) {
        append(body, std::string("LIST"));
        put<uint32_t>(body, 3);
        append(body, std::string("abc"));
        body.push_back(0);  // the pad byte of an odd-sized chunk
    }
    append(body, std::string("fmt "));
    put<uint32_t>(body, s.extensible ? 40 : 16);
    put<uint16_t>(body, s.extensible ? 0xFFFE : s.format);
    put<uint16_t>(body, s.channels);
    put<uint32_t>(body, s.rate);
    const uint16_t align = static_cast<uint16_t>(s.block_align >= 0 ? s.block_align : s.channels * (s.bits / 8));
    put<uint32_t>(body, s.rate * align);
    put<uint16_t>(body, align);
    put<uint16_t>(body, s.bits);
    if (s.extensible) {
        put<uint16_t>(body, 22);
        put<uint16_t>(body, s.bits);
        put<uint32_t>(body, 0);
        put<uint16_t>(body, s.sub_format);
        const uint8_t tail[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
                                  static_cast<uint8_t>(s.bad_guid ? 0x00 : 0x71)};
        append(body, tail, sizeof tail);
    }
    append(body, std::string("data"));
    put<uint32_t>(body, static_cast<uint32_t>(s.data_size >= 0 ? s.data_size : static_cast<int64_t>(samples.size())));
    append(body, samples);
    std::vector<uint8_t> out;
    append(out, std::string("RIFF"));
    put<uint32_t>(out, static_cast<uint32_t>(body.size()));
    append(out, body);
    return out;
}

}  // namespace loaders_test
