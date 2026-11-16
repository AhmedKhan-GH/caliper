// The WAV reader: RIFF chunks, PCM 8, 16, 24 and 32 bit, IEEE float 32 and 64
// bit, and WAVE_FORMAT_EXTENSIBLE, into [frames, channels] with rate_hz.
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"

#include <cstring>

namespace caliper_builder {

using namespace detail;

namespace {

constexpr uint16_t kPcm = 1;
constexpr uint16_t kFloat = 3;
constexpr uint16_t kExtensible = 0xFFFE;
// The bytes 2 to 15 of the KSDATAFORMAT_SUBTYPE GUIDs; bytes 0 and 1 hold the format.
const uint8_t kGuidTail[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
constexpr int kMaxChunks = 4096;

}  // namespace

TensorReadResult read_wav(const std::string& path, const ReaderLimits& limits) {
    TensorReadResult result;
    InputFile f;
    if ((result.error = f.open(path))) return result;
    uint8_t riff[12];
    if (f.size() < 12) {
        result.error = Error("loader.truncated", "The file is too short to be a WAV file.", path);
        return result;
    }
    if ((result.error = f.read(0, 12, riff))) return result;
    if (std::memcmp(riff + 8, "WAVE", 4) != 0 || std::memcmp(riff, "RIFF", 4) != 0) {
        if (std::memcmp(riff, "RIFX", 4) == 0 || std::memcmp(riff, "RF64", 4) == 0) {
            result.error = Error("loader.unsupported", "Big-endian (RIFX) and RF64 WAV files are not supported.", path);
        } else {
            result.error = Error("loader.malformed", "The file is not a RIFF WAVE file.", path);
        }
        return result;
    }
    bool have_fmt = false, have_data = false;
    uint16_t format = 0, channels = 0, block_align = 0, bits = 0, valid_bits = 0;
    uint32_t rate = 0;
    uint64_t data_offset = 0, data_size = 0;
    uint64_t pos = 12;
    for (int k = 0; k < kMaxChunks && fits(pos, 8, f.size()); ++k) {
        uint8_t head[8];
        if ((result.error = f.read(pos, 8, head))) return result;
        const uint64_t size = le32(head + 4);
        const uint64_t body = pos + 8;
        if (std::memcmp(head, "fmt ", 4) == 0 && !have_fmt) {
            if (size < 16) {
                result.error = Error("loader.malformed", "The WAV format chunk is too short.", path);
                return result;
            }
            uint8_t fmt[40] = {};
            const uint64_t n = size < 40 ? size : 40;
            if ((result.error = f.read(body, n, fmt))) return result;
            format = le16(fmt);
            channels = le16(fmt + 2);
            rate = le32(fmt + 4);
            block_align = le16(fmt + 12);
            bits = le16(fmt + 14);
            valid_bits = bits;
            if (format == kExtensible) {
                if (n < 40 || le16(fmt + 16) < 22) {
                    result.error = Error("loader.malformed", "The WAV extensible format chunk is too short.", path);
                    return result;
                }
                valid_bits = le16(fmt + 18);
                if (std::memcmp(fmt + 26, kGuidTail, sizeof kGuidTail) != 0) {
                    result.error = Error("loader.unsupported", "The WAV sub-format is not PCM or IEEE float.", path);
                    return result;
                }
                format = le16(fmt + 24);
            }
            have_fmt = true;
        } else if (std::memcmp(head, "data", 4) == 0 && !have_data) {
            data_offset = body;
            data_size = size;
            if (!fits(body, size, f.size())) {
                data_size = f.size() - body;
                result.warnings.push_back("The data chunk claims more bytes than the file holds; read the " +
                                          std::to_string(data_size) + " bytes that are there.");
            }
            have_data = true;
        }
        if (have_fmt && have_data) break;
        if (!fits(body, size + (size & 1), f.size())) break;
        pos = body + size + (size & 1);
    }
    if (!have_fmt || !have_data) {
        result.error = Error("loader.malformed", "The WAV file lacks a format chunk or a data chunk.", path);
        return result;
    }
    DType dtype;
    if (format == kPcm && bits == 8) dtype = DType::u8;
    else if (format == kPcm && bits == 16) dtype = DType::i16;
    else if (format == kPcm && (bits == 24 || bits == 32)) dtype = DType::i32;
    else if (format == kFloat && bits == 32) dtype = DType::f32;
    else if (format == kFloat && bits == 64) dtype = DType::f64;
    else {
        result.error = Error("loader.unsupported",
                             "The WAV encoding (format " + std::to_string(format) + ", " + std::to_string(bits) +
                                 " bits) is not supported; PCM 8, 16, 24 and 32 bit and float 32 and 64 bit are.",
                             path);
        return result;
    }
    if (channels < 1 || channels > limits.max_wav_channels || rate == 0) {
        result.error = Error("loader.malformed", "The WAV file has an invalid channel count or sample rate.", path);
        return result;
    }
    const uint64_t sample_bytes = bits / 8;
    if (block_align != channels * sample_bytes) {
        result.error = Error("loader.malformed", "The WAV block size does not match its channels and sample size.", path);
        return result;
    }
    const uint64_t frames = data_size / block_align;
    if (data_size % block_align != 0) {
        result.warnings.push_back("Ignored a partial frame at the end of the data chunk.");
    }
    ReadTensor t;
    t.name = file_stem(path);
    t.rate_hz = static_cast<double>(rate);
    if (valid_bits != 0 && valid_bits < bits) {
        t.warnings.push_back("samples carry " + std::to_string(valid_bits) + " valid bits in " + std::to_string(bits) +
                             "-bit containers");
    }
    const std::vector<int64_t> shape = {static_cast<int64_t>(frames), static_cast<int64_t>(channels)};
    if (bits != 24) {
        TensorResult m = TensorData::map_file(path, data_offset, dtype, shape);
        if ((result.error = m.error)) return result;
        t.data = std::move(m.tensor);
    } else {
        const uint64_t samples = frames * channels;
        if (samples > limits.max_derived_bytes / 4) {
            result.error = Error("loader.too_large", "The converted audio exceeds the reader's limit.", path);
            return result;
        }
        std::vector<uint8_t> raw;
        if ((result.error = f.read_vec(data_offset, samples * 3, &raw, limits.max_derived_bytes))) return result;
        TensorResult o = TensorData::allocate(DType::i32, shape);
        if ((result.error = o.error)) return result;
        uint8_t* out = o.tensor.mutable_data();
        for (uint64_t i = 0; i < samples; ++i) {
            const uint8_t* p = raw.data() + 3 * i;
            uint32_t v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                         (static_cast<uint32_t>(p[2]) << 16);
            if (v & 0x800000u) v |= 0xFF000000u;
            std::memcpy(out + 4 * i, &v, 4);
        }
        t.data = std::move(o.tensor);
        t.derived = true;
        t.warnings.push_back("converted 24-bit samples to i32");
    }
    result.tensors.push_back(std::move(t));
    return result;
}

}  // namespace caliper_builder
