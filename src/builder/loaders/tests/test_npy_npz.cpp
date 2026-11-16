// Tests of the .npy and .npz readers on numpy-written fixtures and on files
// built in code, including hostile headers and archives.
#include <caliper_builder/loaders/tensor_readers.h>

#include <doctest/doctest.h>

#include <cmath>

#include "test_formats.h"

using namespace caliper_builder;
using namespace loaders_test;

namespace {

const std::string kFixtures = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/loaders/tests/fixtures/";

std::vector<double> values(const TensorData& t) {
    std::vector<double> v;
    for (uint64_t i = 0; i < t.numel(); ++i) v.push_back(t.element(i));
    return v;
}

const ReadTensor* find(const TensorReadResult& r, const std::string& name) {
    for (const ReadTensor& t : r.tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

std::string npy_file(const std::string& dir, const std::string& name, const std::string& dict,
                     const std::vector<uint8_t>& data, int version = 1) {
    const std::string path = dir + "/" + name;
    write_bytes(path, make_npy(dict, data, version));
    return path;
}

}  // namespace

TEST_CASE("npy fixtures written by numpy") {
    TensorReadResult r = read_npy(kFixtures + "numpy_fortran_f8.npy");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].name == "numpy_fortran_f8");
    CHECK(r.tensors[0].data.dtype() == DType::f64);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 3});
    CHECK(values(r.tensors[0].data) == std::vector<double>{0, 1, 2, 3, 4, 5});
    CHECK(r.tensors[0].derived);

    r = read_npy(kFixtures + "numpy_big_endian_i4.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.dtype() == DType::i32);
    CHECK(values(r.tensors[0].data) == std::vector<double>{1, -2, 3, 0x01020304});
    CHECK(r.tensors[0].derived);

    r = read_npy(kFixtures + "numpy_f2.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.dtype() == DType::f16);
    CHECK(r.tensors[0].data.is_mapped());
    CHECK_FALSE(r.tensors[0].derived);
    const std::vector<double> h = values(r.tensors[0].data);
    CHECK(h[0] == 1.0);
    CHECK(h[1] == -0.5);
    CHECK(h[2] == 65504.0);
    CHECK(std::isinf(h[3]));

    r = read_npy(kFixtures + "numpy_scalar_u2.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape().empty());
    CHECK(values(r.tensors[0].data) == std::vector<double>{513});

    r = read_npy(kFixtures + "numpy_v2_bool.npy");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.tensors[0].data.dtype() == DType::boolean);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{1, 3});
    CHECK(values(r.tensors[0].data) == std::vector<double>{1, 0, 1});

    r = read_npy(kFixtures + "numpy_v3_u1.npy");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 2, 1});
    CHECK(values(r.tensors[0].data) == std::vector<double>{0, 1, 2, 3});
}

TEST_CASE("npy built in code: Fortran order of rank 3, big-endian f16 and write_npy") {
    const std::string dir = fresh_dir("npy_code");
    // A [2, 3, 2] array stored in Fortran order: element (i, j, k) sits at i + 2j + 6k.
    std::vector<float> fortran(12);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k) fortran[i + 2 * j + 6 * k] = static_cast<float>(100 * i + 10 * j + k);
    std::string p = npy_file(dir, "f.npy", "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3, 2), }",
                             raw(fortran));
    TensorReadResult r = read_npy(p);
    REQUIRE(r.error.ok());
    CHECK(values(r.tensors[0].data) == std::vector<double>{0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121});

    p = npy_file(dir, "h.npy", "{'descr': '>f2', 'fortran_order': False, 'shape': (2,), }", {0x3C, 0x00, 0xB8, 0x00});
    r = read_npy(p);
    REQUIRE(r.error.ok());
    CHECK(values(r.tensors[0].data) == std::vector<double>{1.0, -0.5});

    // Python 2 long literals and a trailing comma are accepted.
    p = npy_file(dir, "l.npy", "{\"descr\": \"|u1\", \"fortran_order\": False, \"shape\": (2L, 1L,)}", {7, 8});
    r = read_npy(p);
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{2, 1});

    TensorResult t = TensorData::from_bytes(DType::f32, {2, 3}, raw(std::vector<float>{1, 2, 3, 4, 5, 6}));
    REQUIRE(t.error.ok());
    uint64_t offset = 0;
    REQUIRE(write_npy(dir + "/w.npy", t.tensor, &offset).ok());
    CHECK(offset % 64 == 0);
    r = read_npy(dir + "/w.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.is_mapped());
    CHECK(r.tensors[0].data.file_offset() == offset);
    CHECK(values(r.tensors[0].data) == std::vector<double>{1, 2, 3, 4, 5, 6});

    TensorResult s = TensorData::from_bytes(DType::i64, {}, raw(std::vector<int64_t>{-7}));
    REQUIRE(write_npy(dir + "/s.npy", s.tensor).ok());
    r = read_npy(dir + "/s.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape().empty());
    CHECK(values(r.tensors[0].data) == std::vector<double>{-7});
    TensorResult one = TensorData::from_bytes(DType::boolean, {3}, {1, 0, 1});
    REQUIRE(write_npy(dir + "/b.npy", one.tensor).ok());
    r = read_npy(dir + "/b.npy");
    REQUIRE(r.error.ok());
    CHECK(r.tensors[0].data.shape() == std::vector<int64_t>{3});
}

TEST_CASE("npy refuses hostile and unsupported headers") {
    const std::string dir = fresh_dir("npy_hostile");
    const std::string ok_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
    const std::vector<uint8_t> two = raw(std::vector<float>{1, 2});
    auto code = [&](const std::string& dict, const std::vector<uint8_t>& data = {}, int version = 1) {
        return read_npy(npy_file(dir, "x.npy", dict, data, version)).error.code;
    };
    CHECK(code("{'descr': '|O', 'fortran_order': False, 'shape': (2,), }", std::vector<uint8_t>(16)) ==
          "loader.pickle_refused");
    CHECK(code("{'descr': [('a', '<f4')], 'fortran_order': False, 'shape': (2,), }") == "loader.unsupported");
    CHECK(code("{'descr': '<c8', 'fortran_order': False, 'shape': (2,), }") == "loader.unsupported");
    CHECK(code("{'descr': '<U4', 'fortran_order': False, 'shape': (2,), }") == "loader.unsupported");
    CHECK(code("{'descr': '<f3', 'fortran_order': False, 'shape': (2,), }") == "loader.unsupported");
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), 'extra': 1, }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'shape': (2,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (2,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'fortran_order': maybe, 'shape': (2,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (-2,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f\\x34', 'fortran_order': False, 'shape': (2,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), } junk") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999,), }") == "loader.malformed");
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1, 1, 1, 1, 1, 1, 1), }") == "tensor.rank");
    CHECK(code("{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296), }") == "tensor.too_large");
    // A shape that claims more than the file holds.
    CHECK(code("{'descr': '<f4', 'fortran_order': False, 'shape': (1099511627776,), }", two) == "loader.truncated");
    CHECK(code("{'descr': '>f4', 'fortran_order': False, 'shape': (3,), }", two) == "loader.truncated");
    // Bad magic, versions and header lengths.
    write_bytes(dir + "/m.npy", {'N', 'U', 'M', 'P', 'Y', 0, 1, 0, 0, 0});
    CHECK(read_npy(dir + "/m.npy").error.code == "loader.malformed");
    write_bytes(dir + "/v.npy", {0x93, 'N', 'U', 'M', 'P', 'Y', 4, 0, 10, 0, 0, 0});
    CHECK(read_npy(dir + "/v.npy").error.code == "loader.unsupported");
    write_bytes(dir + "/len.npy", {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0xFF, 0xFF, '{'});
    CHECK(read_npy(dir + "/len.npy").error.code == "loader.truncated");
    write_bytes(dir + "/big.npy", {0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 0xFF, 0xFF, 0xFF, 0xFF, '{'});
    CHECK(read_npy(dir + "/big.npy").error.code == "loader.too_large");
    write_bytes(dir + "/short.npy", {0x93, 'N', 'U'});
    CHECK(read_npy(dir + "/short.npy").error.code == "loader.malformed");
    CHECK(read_npy(dir + "/missing.npy").error.code == "loader.io");
    // A derived tensor larger than the limit is refused before it is allocated.
    ReaderLimits small;
    small.max_derived_bytes = 4;
    CHECK(read_npy(npy_file(dir, "be.npy", "{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }", two), small)
              .error.code == "loader.too_large");
    // Trailing bytes are ignored with a warning.
    std::vector<uint8_t> extra = two;
    extra.push_back(0);
    TensorReadResult r = read_npy(npy_file(dir, "t.npy", ok_dict, extra));
    REQUIRE(r.error.ok());
    CHECK(r.warnings.size() == 1);
}

TEST_CASE("npz fixtures written by numpy, stored and compressed") {
    for (const char* name : {"numpy_savez.npz", "numpy_savez_compressed.npz"}) {
        CAPTURE(name);
        TensorReadResult r = read_npz(kFixtures + name);
        REQUIRE_MESSAGE(r.error.ok(), r.error.message);
        REQUIRE(r.tensors.size() == 3);
        const ReadTensor* a = find(r, "a");
        const ReadTensor* b = find(r, "b");
        const ReadTensor* c = find(r, "c");
        REQUIRE(a);
        REQUIRE(b);
        REQUIRE(c);
        CHECK(a->data.dtype() == DType::f32);
        CHECK(a->data.shape() == std::vector<int64_t>{2, 3});
        CHECK(values(a->data) == std::vector<double>{0, 0.5, 1, 1.5, 2, 2.5});
        CHECK(b->data.dtype() == DType::i64);
        CHECK(values(b->data) == std::vector<double>{-1, 0, 7, 1099511627776.0});
        CHECK(c->data.dtype() == DType::boolean);
        CHECK(c->data.shape().empty());
        const bool compressed = std::string(name).find("compressed") != std::string::npos;
        CHECK(a->derived == compressed);
        CHECK(a->data.is_mapped() == !compressed);
    }
    TensorReadResult r = read_npz(kFixtures + "numpy_savez.npz", [](const std::string& n) { return n == "b"; });
    REQUIRE(r.error.ok());
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].name == "b");
}

TEST_CASE("npz archives built in code") {
    const std::string dir = fresh_dir("npz_code");
    const std::vector<uint8_t> x =
        make_npy("{'descr': '<i2', 'fortran_order': False, 'shape': (3,), }", raw(std::vector<int16_t>{1, -2, 3}));
    const std::vector<uint8_t> be =
        make_npy("{'descr': '>i2', 'fortran_order': False, 'shape': (2,), }", {0x01, 0x02, 0xFF, 0xFE});
    auto read = [&](const std::vector<ZipMember>& members, ZipOptions opt = {}) {
        write_bytes(dir + "/a.npz", make_zip(members, opt));
        return read_npz(dir + "/a.npz");
    };

    SUBCASE("zip64 extras and a zip64 end record") {
        for (ZipOptions opt : {ZipOptions{true, false, false}, ZipOptions{false, true, false}, ZipOptions{true, true, false}}) {
            TensorReadResult r = read({{"x.npy", x}, {"be.npy", be}}, opt);
            REQUIRE_MESSAGE(r.error.ok(), r.error.message);
            REQUIRE(r.tensors.size() == 2);
            CHECK(values(find(r, "x")->data) == std::vector<double>{1, -2, 3});
            CHECK(values(find(r, "be")->data) == std::vector<double>{258, -2});
            CHECK(find(r, "be")->derived);
        }
    }
#ifdef CALIPER_BUILDER_HAVE_ZLIB
    SUBCASE("deflated members and a CRC mismatch") {
        ZipMember m{"x.npy", x, true};
        TensorReadResult r = read({m});
        REQUIRE(r.error.ok());
        CHECK(values(r.tensors[0].data) == std::vector<double>{1, -2, 3});
        m.bad_crc = true;
        CHECK(read({m}).error.code == "loader.malformed");
        ZipMember b{"be.npy", be, true};
        r = read({b});
        REQUIRE(r.error.ok());
        CHECK(values(r.tensors[0].data) == std::vector<double>{258, -2});
    }
#endif
    SUBCASE("other members are skipped with a warning") {
        TensorReadResult r = read({{"x.npy", x}, {"notes.txt", {'h', 'i'}}, {"sub/", {}}});
        REQUIRE(r.error.ok());
        CHECK(r.tensors.size() == 1);
        CHECK(r.warnings.size() == 1);
    }
    SUBCASE("pickles inside the archive refuse the file") {
        CHECK(read({{"x.npy", x}, {"archive/data.pkl", {0x80, 0x02, '.'}}}).error.code == "loader.pickle_refused");
        CHECK(read({{"data.pkl", {0x80, 0x02, '.'}}}).error.code == "loader.pickle_refused");
        CHECK(read({{"x.npy", x}, {"model.bin", {0x80, 0x04, 0x95}}}).error.code == "loader.pickle_refused");
        CHECK(read({{"x.pickle", {'(', '.'}}}).error.code == "loader.pickle_refused");
        const std::vector<uint8_t> obj = make_npy("{'descr': '|O', 'fortran_order': False, 'shape': (1,), }",
                                                  {0x80, 0x02, ']', '.'});
        Error e = read({{"x.npy", x}, {"obj.npy", obj}}).error;
        CHECK(e.code == "loader.pickle_refused");
        CHECK(e.message.find("obj.npy") != std::string::npos);
        CHECK(e.message.find("safetensors") != std::string::npos);
        // A filter does not skip the check.
        write_bytes(dir + "/f.npz", make_zip({{"x.npy", x}, {"obj.npy", obj}}));
        CHECK(read_npz(dir + "/f.npz", [](const std::string& n) { return n == "x"; }).error.code ==
              "loader.pickle_refused");
    }
    SUBCASE("hostile archives") {
        CHECK(read({{"x.npy", x}, {"x.npy", x}}).error.code == "loader.malformed");
        CHECK(read({{"x.npy", x}, {"y.npy", x}}, ZipOptions{false, false, true}).error.code == "loader.malformed");
        ZipMember bz{"x.npy", x};
        bz.method = 12;
        CHECK(read({bz}).error.code == "loader.unsupported");
        ZipMember enc{"x.npy", x};
        enc.flags = 1;
        CHECK(read({enc}).error.code == "loader.unsupported");
        const std::vector<uint8_t> short_npy =
            make_npy("{'descr': '<i2', 'fortran_order': False, 'shape': (30,), }", raw(std::vector<int16_t>{1, 2}));
        CHECK(read({{"s.npy", short_npy}}).error.code == "loader.truncated");

        std::vector<uint8_t> zip = make_zip({{"x.npy", x}});
        write_bytes(dir + "/cut.npz", std::vector<uint8_t>(zip.begin(), zip.begin() + static_cast<long>(zip.size() / 2)));
        CHECK(read_npz(dir + "/cut.npz").error.code == "loader.malformed");
        write_bytes(dir + "/tiny.npz", {'P', 'K'});
        CHECK(read_npz(dir + "/tiny.npz").error.code == "loader.malformed");

        // An end record that claims a directory larger than the file, or more members than fit.
        std::vector<uint8_t> lie = zip;
        const std::size_t end = lie.size() - 22;
        lie[end + 12] = 0xFF;
        lie[end + 13] = 0xFF;
        write_bytes(dir + "/lie.npz", lie);
        CHECK(read_npz(dir + "/lie.npz").error.code == "loader.malformed");
        lie = zip;
        lie[end + 10] = 0xF0;
        lie[end + 8] = 0xF0;
        write_bytes(dir + "/count.npz", lie);
        CHECK(read_npz(dir + "/count.npz").error.code == "loader.malformed");
        ReaderLimits few;
        few.max_zip_members = 1;
        write_bytes(dir + "/two.npz", make_zip({{"x.npy", x}, {"y.npy", x}}));
        CHECK(read_npz(dir + "/two.npz", {}, few).error.code == "loader.too_large");
        few = ReaderLimits{};
        few.max_tensors = 1;
        CHECK(read_npz(dir + "/two.npz", {}, few).error.code == "loader.too_large");
        // A local header offset past the directory.
        lie = zip;
        const std::size_t cd = lie.size() - 22 - (46 + 5);
        lie[cd + 42] = 0x00;
        lie[cd + 43] = 0x10;
        write_bytes(dir + "/off.npz", lie);
        CHECK(read_npz(dir + "/off.npz").error.code == "loader.malformed");
    }
}
