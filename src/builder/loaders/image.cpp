// The image reader: PNG and JPEG through stb_image, into u8 [H, W, C].
//
// stb_image is compiled here with static linkage (STB_IMAGE_STATIC), so it
// cannot clash with another copy in the host, and with only its PNG and JPEG
// decoders. The dimensions are read and checked before anything is decoded.
#include <caliper_builder/loaders/tensor_readers.h>

#include "file_io.h"

#include <climits>
#include <cstring>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 32768
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace caliper_builder {

using namespace detail;

TensorReadResult read_image(const std::string& path, int channels, const ReaderLimits& limits) {
    TensorReadResult result;
    if (channels < 0 || channels > 4) {
        result.error = Error("loader.bad_value", "`channels` is 0, 1, 2, 3 or 4.", path);
        return result;
    }
    InputFile f;
    if ((result.error = f.open(path))) return result;
    if (f.size() > limits.max_image_file || f.size() > static_cast<uint64_t>(INT_MAX)) {
        result.error = Error("loader.too_large", "The image file is larger than the reader's limit.", path);
        return result;
    }
    std::vector<uint8_t> bytes;
    if ((result.error = f.read_vec(0, f.size(), &bytes, limits.max_image_file))) return result;
    static const uint8_t png[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    const bool is_png = bytes.size() >= 8 && std::memcmp(bytes.data(), png, 8) == 0;
    const bool is_jpeg = bytes.size() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF;
    if (!is_png && !is_jpeg) {
        result.error = Error("loader.malformed", "The file is not a PNG or a JPEG image.", path);
        return result;
    }
    // A PNG states its size in the IHDR chunk, which must come first; check it before stb reads anything.
    if (is_png && bytes.size() >= 24 && std::memcmp(bytes.data() + 12, "IHDR", 4) == 0) {
        auto be32 = [&](std::size_t i) {
            return (static_cast<uint64_t>(bytes[i]) << 24) | (static_cast<uint64_t>(bytes[i + 1]) << 16) |
                   (static_cast<uint64_t>(bytes[i + 2]) << 8) | static_cast<uint64_t>(bytes[i + 3]);
        };
        const uint64_t pw = be32(16), ph = be32(20);
        if (pw > limits.max_image_side || ph > limits.max_image_side || pw * ph > limits.max_image_pixels) {
            result.error = Error("loader.too_large", "The image's size exceeds the reader's limits.", path);
            return result;
        }
    }
    const int len = static_cast<int>(bytes.size());
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory(bytes.data(), len, &w, &h, &comp)) {
        const std::string reason = stbi_failure_reason() ? stbi_failure_reason() : "unknown";
        result.error = reason == "too large"
                           ? Error("loader.too_large", "The image's size exceeds the decoder's limits.", path)
                           : Error("loader.malformed", "The image header could not be read: " + reason + ".", path);
        return result;
    }
    const uint64_t uw = static_cast<uint64_t>(w), uh = static_cast<uint64_t>(h);
    if (w < 1 || h < 1 || uw > limits.max_image_side || uh > limits.max_image_side ||
        uw * uh > limits.max_image_pixels) {
        result.error = Error("loader.too_large", "The image's size exceeds the reader's limits.", path);
        return result;
    }
    const int out_channels = channels ? channels : comp;
    if (out_channels < 1 || out_channels > 4 || uw * uh * static_cast<uint64_t>(out_channels) > limits.max_derived_bytes) {
        result.error = Error("loader.too_large", "The decoded image would exceed the reader's limit.", path);
        return result;
    }
    const bool sixteen = stbi_is_16_bit_from_memory(bytes.data(), len) != 0;
    int w2 = 0, h2 = 0, comp2 = 0;
    stbi_uc* pixels = stbi_load_from_memory(bytes.data(), len, &w2, &h2, &comp2, out_channels);
    if (!pixels) {
        const std::string reason = stbi_failure_reason() ? stbi_failure_reason() : "unknown";
        result.error = Error("loader.malformed", "The image could not be decoded: " + reason + ".", path);
        return result;
    }
    if (w2 != w || h2 != h) {
        stbi_image_free(pixels);
        result.error = Error("loader.malformed", "The image's size changed while it was decoded.", path);
        return result;
    }
    const std::size_t n = static_cast<std::size_t>(uw * uh * static_cast<uint64_t>(out_channels));
    std::vector<uint8_t> data;
    try {
        data.assign(pixels, pixels + n);
    } catch (...) {
        stbi_image_free(pixels);
        result.error = Error("loader.too_large", "Memory for the image could not be allocated.", path);
        return result;
    }
    stbi_image_free(pixels);
    TensorResult t = TensorData::from_bytes(DType::u8, {h, w, out_channels}, std::move(data));
    if ((result.error = t.error)) return result;
    ReadTensor rt;
    rt.name = file_stem(path);
    rt.data = std::move(t.tensor);
    rt.derived = true;
    if (sixteen) rt.warnings.push_back("converted 16-bit samples to 8 bits");
    result.tensors.push_back(std::move(rt));
    return result;
}

}  // namespace caliper_builder
