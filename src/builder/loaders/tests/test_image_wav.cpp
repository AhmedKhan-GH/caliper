// Tests of the image and WAV readers and the HDF5 refusal, on files made in
// code: images through stb_image_write, WAV files byte by byte.
#include <caliper_builder/loaders/tensor_readers.h>

#include <doctest/doctest.h>

#include <cstdlib>

#include "test_formats.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif
#include <stb_image_write.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

using namespace caliper_builder;
using namespace loaders_test;

namespace {

std::vector<double> values(const TensorData& t) {
    std::vector<double> v;
    for (uint64_t i = 0; i < t.numel(); ++i) v.push_back(t.element(i));
    return v;
}

void collect(void* context, void* data, int size) {
    auto* out = static_cast<std::vector<uint8_t>*>(context);
    append(*out, data, static_cast<std::size_t>(size));
}

std::vector<uint8_t> png(int w, int h, int comp, const std::vector<uint8_t>& pixels) {
    std::vector<uint8_t> out;
    stbi_write_png_to_func(collect, &out, w, h, comp, pixels.data(), w * comp);
    return out;
}

std::vector<uint8_t> be32(uint32_t v) {
    return {static_cast<uint8_t>(v >> 24), static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 8),
            static_cast<uint8_t>(v)};
}

// A PNG signature, an IHDR chunk that claims w x h, and an empty IDAT chunk.
std::vector<uint8_t> png_header_only(uint32_t w, uint32_t h) {
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    append(out, be32(13));
    append(out, std::string("IHDR"));
    append(out, be32(w));
    append(out, be32(h));
    append(out, std::vector<uint8_t>{8, 2, 0, 0, 0});
    append(out, be32(0));
    append(out, be32(0));
    append(out, std::string("IDAT"));
    append(out, be32(0));
    return out;
}

}  // namespace

TEST_CASE("PNG decodes exactly, with channel conversion") {
    const std::string dir = fresh_dir("image_png");
    // 3 wide, 2 high, RGB.
    const std::vector<uint8_t> pixels = {255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    write_bytes(dir + "/rgb.png", png(3, 2, 3, pixels));
    TensorReadResult r = read_image(dir + "/rgb.png");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].name == "rgb");
    CHECK(r.tensors[0].data.dtype() == DType::u8);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 3, 3});
    CHECK(r.tensors[0].derived);
    std::vector<double> want(pixels.begin(), pixels.end());
    CHECK(values(r.tensors[0].data) == want);

    r = read_image(dir + "/rgb.png", 4);
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 3, 4});
    CHECK(r.tensors[0].data.element(3) == 255);  // alpha
    r = read_image(dir + "/rgb.png", 1);
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 3, 1});

    write_bytes(dir + "/grey.png", png(2, 2, 1, {0, 64, 128, 255}));
    r = read_image(dir + "/grey.png");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 2, 1});
    CHECK(values(r.tensors[0].data) == std::vector<double>{0, 64, 128, 255});
}

TEST_CASE("JPEG decodes close to its source") {
    const std::string dir = fresh_dir("image_jpeg");
    std::vector<uint8_t> pixels;
    for (int i = 0; i < 64; ++i) append(pixels, std::vector<uint8_t>{200, 100, 50});
    std::vector<uint8_t> jpg;
    stbi_write_jpg_to_func(collect, &jpg, 8, 8, 3, pixels.data(), 100);
    write_bytes(dir + "/flat.jpg", jpg);
    TensorReadResult r = read_image(dir + "/flat.jpg");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{8, 8, 3});
    CHECK(std::abs(r.tensors[0].data.element(0) - 200) <= 4);
    CHECK(std::abs(r.tensors[0].data.element(1) - 100) <= 4);
    CHECK(std::abs(r.tensors[0].data.element(2) - 50) <= 4);
}

TEST_CASE("the image reader refuses hostile files") {
    const std::string dir = fresh_dir("image_hostile");
    write_bytes(dir + "/x.png", {'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'a', 'g', 'e'});
    CHECK(read_image(dir + "/x.png").error.code == "loader.malformed");
    write_bytes(dir + "/huge.png", png_header_only(17000, 17000));  // passes stb's own check, not the pixel limit
    Error e = read_image(dir + "/huge.png").error;
    CHECK_MESSAGE(e.code == "loader.too_large", e.message);
    write_bytes(dir + "/huger.png", png_header_only(30000, 30000));  // refused by stb as too large
    e = read_image(dir + "/huger.png").error;
    CHECK_MESSAGE(e.code == "loader.too_large", e.message);
    write_bytes(dir + "/wide.png", png_header_only(100000, 1));
    CHECK(read_image(dir + "/wide.png").error.code != "");
    std::vector<uint8_t> good = png(16, 16, 3, std::vector<uint8_t>(16 * 16 * 3, 7));
    write_bytes(dir + "/cut.png", std::vector<uint8_t>(good.begin(), good.begin() + static_cast<long>(good.size() * 6 / 10)));
    CHECK(read_image(dir + "/cut.png").error.code == "loader.malformed");
    write_bytes(dir + "/ok.png", good);
    CHECK(read_image(dir + "/ok.png", 5).error.code == "loader.bad_value");
    ReaderLimits limits;
    limits.max_image_side = 8;
    CHECK(read_image(dir + "/ok.png", 0, limits).error.code == "loader.too_large");
    limits = ReaderLimits{};
    limits.max_image_file = 10;
    CHECK(read_image(dir + "/ok.png", 0, limits).error.code == "loader.too_large");
    limits = ReaderLimits{};
    limits.max_derived_bytes = 100;
    CHECK(read_image(dir + "/ok.png", 0, limits).error.code == "loader.too_large");
}

TEST_CASE("WAV encodings") {
    const std::string dir = fresh_dir("wav");
    SUBCASE("16-bit PCM stereo is mapped with its rate") {
        WavSpec s;
        s.channels = 2;
        s.rate = 44100;
        s.list_chunk = true;
        write_bytes(dir + "/a.wav", make_wav(s, raw(std::vector<int16_t>{1, -1, 2, -2, 32767, -32768})));
        TensorReadResult r = read_wav(dir + "/a.wav");
        REQUIRE_MESSAGE(r.error.ok(), r.error.message);
        const ReadTensor& t = r.tensors[0];
        CHECK(t.name == "a");
        CHECK(t.data.dtype() == DType::i16);
        CHECK(t.data.shape() == std::vector<int64_t>{3, 2});
        CHECK(t.data.is_mapped());
        CHECK(*t.rate_hz == 44100);
        CHECK(values(t.data) == std::vector<double>{1, -1, 2, -2, 32767, -32768});
    }
    SUBCASE("8-bit PCM is unsigned") {
        WavSpec s;
        s.bits = 8;
        write_bytes(dir + "/b.wav", make_wav(s, {0, 128, 255}));
        TensorReadResult r = read_wav(dir + "/b.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::u8);
        CHECK(values(r.tensors[0].data) == std::vector<double>{0, 128, 255});
    }
    SUBCASE("24-bit PCM is sign-extended into i32") {
        WavSpec s;
        s.bits = 24;
        write_bytes(dir + "/c.wav", make_wav(s, {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00}));
        TensorReadResult r = read_wav(dir + "/c.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::i32);
        CHECK(r.tensors[0].derived);
        CHECK(values(r.tensors[0].data) == std::vector<double>{-1, 8388607, -8388608, 1});
    }
    SUBCASE("32-bit PCM and float 32 and 64") {
        WavSpec s;
        s.bits = 32;
        write_bytes(dir + "/d.wav", make_wav(s, raw(std::vector<int32_t>{-5, 2000000000})));
        TensorReadResult r = read_wav(dir + "/d.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::i32);
        CHECK(values(r.tensors[0].data) == std::vector<double>{-5, 2000000000});
        s.format = 3;
        write_bytes(dir + "/e.wav", make_wav(s, raw(std::vector<float>{0.5f, -0.25f})));
        r = read_wav(dir + "/e.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::f32);
        CHECK(values(r.tensors[0].data) == std::vector<double>{0.5, -0.25});
        s.bits = 64;
        write_bytes(dir + "/f.wav", make_wav(s, raw(std::vector<double>{0.125})));
        r = read_wav(dir + "/f.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::f64);
    }
    SUBCASE("WAVE_FORMAT_EXTENSIBLE with PCM and float sub-formats") {
        WavSpec s;
        s.extensible = true;
        s.channels = 2;
        write_bytes(dir + "/g.wav", make_wav(s, raw(std::vector<int16_t>{3, 4})));
        TensorReadResult r = read_wav(dir + "/g.wav");
        REQUIRE_MESSAGE(r.error.ok(), r.error.message);
        CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{1, 2});
        s.sub_format = 3;
        s.bits = 32;
        s.channels = 1;
        write_bytes(dir + "/h.wav", make_wav(s, raw(std::vector<float>{1.5f})));
        r = read_wav(dir + "/h.wav");
        REQUIRE(r.error.ok());
        CHECK(r.tensors[0].data.dtype() == DType::f32);
    }
}

TEST_CASE("the WAV reader refuses hostile files") {
    const std::string dir = fresh_dir("wav_hostile");
    auto code = [&](const WavSpec& s, const std::vector<uint8_t>& samples) {
        write_bytes(dir + "/x.wav", make_wav(s, samples));
        return read_wav(dir + "/x.wav");
    };
    WavSpec s;
    s.format = 2;  // ADPCM
    CHECK(code(s, {0, 0}).error.code == "loader.unsupported");
    s = WavSpec{};
    s.extensible = true;
    s.bad_guid = true;
    CHECK(code(s, {0, 0}).error.code == "loader.unsupported");
    s = WavSpec{};
    s.block_align = 3;
    CHECK(code(s, {0, 0}).error.code == "loader.malformed");
    s = WavSpec{};
    s.channels = 0;
    CHECK(code(s, {}).error.code == "loader.malformed");
    s = WavSpec{};
    s.rate = 0;
    CHECK(code(s, {0, 0}).error.code == "loader.malformed");
    s = WavSpec{};
    s.bits = 12;
    CHECK(code(s, {0, 0}).error.code == "loader.unsupported");
    // A data chunk that claims more than the file holds is cut to what is there.
    s = WavSpec{};
    s.data_size = 1000000;
    TensorReadResult r = code(s, raw(std::vector<int16_t>{1, 2, 3}));
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{3, 1});
    CHECK(r.warnings.size() == 1);
    // A partial frame is dropped.
    s = WavSpec{};
    s.channels = 2;
    r = code(s, raw(std::vector<int16_t>{1, 2, 3}));
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{1, 2});

    write_bytes(dir + "/rifx.wav", {'R', 'I', 'F', 'X', 0, 0, 0, 0, 'W', 'A', 'V', 'E'});
    CHECK(read_wav(dir + "/rifx.wav").error.code == "loader.unsupported");
    write_bytes(dir + "/riff.wav", {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'A', 'V', 'I', ' '});
    CHECK(read_wav(dir + "/riff.wav").error.code == "loader.malformed");
    write_bytes(dir + "/nodata.wav", {'R', 'I', 'F', 'F', 4, 0, 0, 0, 'W', 'A', 'V', 'E'});
    CHECK(read_wav(dir + "/nodata.wav").error.code == "loader.malformed");
    write_bytes(dir + "/short.wav", {'R', 'I', 'F'});
    CHECK(read_wav(dir + "/short.wav").error.code == "loader.truncated");
    ReaderLimits limits;
    limits.max_wav_channels = 1;
    s = WavSpec{};
    s.channels = 2;
    write_bytes(dir + "/y.wav", make_wav(s, raw(std::vector<int16_t>{1, 2})));
    CHECK(read_wav(dir + "/y.wav", limits).error.code == "loader.malformed");
}

TEST_CASE("HDF5 is a refusal in this build") {
    CHECK_FALSE(hdf5_available());
    Error e = read_hdf5("any.h5").error;
    CHECK(e.code == "loader.hdf5_not_available");
    CHECK(e.message.find("Python loader") != std::string::npos);
}
