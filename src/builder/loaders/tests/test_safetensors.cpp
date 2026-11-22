// Tests of the safetensors reader on a fixture written by the safetensors
// package and on hostile files built in code.
#include <caliper_builder/loaders/tensor_readers.h>

#include <doctest/doctest.h>

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

}  // namespace

TEST_CASE("safetensors fixture written by the safetensors package") {
    TensorReadResult r = read_safetensors(kFixtures + "real.safetensors");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    REQUIRE(r.tensors.size() == 3);
    const ReadTensor* w = find(r, "weight");
    REQUIRE(w);
    CHECK(w->data.dtype() == DType::f32);
    CHECK(w->data.shape() == std::vector<int64_t>{2, 3});
    CHECK(w->data.is_mapped());
    CHECK_FALSE(w->derived);
    CHECK(values(w->data) == std::vector<double>{0, 0.5, 1, 1.5, 2, 2.5});
    CHECK(values(find(r, "bias")->data) == std::vector<double>{0.25, -1.0});
    CHECK(find(r, "bias")->data.dtype() == DType::f16);
    CHECK(values(find(r, "steps")->data) == std::vector<double>{-1, 0, 7, 1099511627776.0});

    r = read_safetensors(kFixtures + "real.safetensors", [](const std::string& n) { return n == "bias"; });
    REQUIRE(r.error.ok());
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].name == "bias");
}

TEST_CASE("safetensors converts bf16 and skips f8") {
    const std::string dir = fresh_dir("st_dtypes");
    std::vector<uint8_t> data = raw(std::vector<uint16_t>{0x3F80, 0xC040});  // bf16 1, -3
    data.push_back(0x38);                                                     // one f8 value
    const std::string header =
        R"({"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"e":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[4,5]},"__metadata__":{"k":"v"}})";
    write_bytes(dir + "/a.safetensors", make_safetensors(header, data));
    TensorReadResult r = read_safetensors(dir + "/a.safetensors");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].data.dtype() == DType::f32);
    CHECK(values(r.tensors[0].data) == std::vector<double>{1, -3});
    CHECK(r.tensors[0].derived);
    CHECK(r.tensors[0].warnings.size() == 1);
    CHECK(r.warnings.size() == 1);

    // Empty tensors and a header padded with spaces are fine.
    write_bytes(dir + "/e.safetensors", make_safetensors(R"({"z":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]}}   )", {}));
    r = read_safetensors(dir + "/e.safetensors");
    REQUIRE_MESSAGE(r.error.ok(), r.error.message);
    CHECK(r.tensors[0].data.numel() == 0);
}

TEST_CASE("safetensors refuses hostile headers") {
    const std::string dir = fresh_dir("st_hostile");
    const std::vector<uint8_t> eight = raw(std::vector<float>{1, 2});
    auto code = [&](const std::string& header, const std::vector<uint8_t>& data, const ReaderLimits& limits = {}) {
        write_bytes(dir + "/x.safetensors", make_safetensors(header, data));
        return read_safetensors(dir + "/x.safetensors", {}, limits).error.code;
    };
    const std::string good = R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    CHECK(code(good, eight) == "");
    CHECK(code(R"([1,2])", eight) == "loader.malformed");
    CHECK(code(R"( {"a":1})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"x":1}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":7,"shape":[2],"data_offsets":[0,8]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2.5],"data_offsets":[0,8]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,2],"data_offsets":[0,8]}})", eight) == "tensor.rank");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,800]}})", eight) == "loader.truncated");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", eight) == "loader.malformed");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,8]}})", eight) ==
          "tensor.too_large");
    CHECK(code(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
               eight) == "loader.malformed");  // overlap (and the last four bytes are unused)
    CHECK(code(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", eight) == "loader.malformed");  // a hole
    CHECK(code(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", eight) == "loader.malformed");  // leftover
    CHECK(code(R"({"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", eight) ==
          "loader.malformed");
    std::string deep = R"({"a":)";
    for (int i = 0; i < 200; ++i) deep += "[";
    for (int i = 0; i < 200; ++i) deep += "]";
    deep += "}";
    CHECK(code(deep, {}) == "loader.malformed");
    ReaderLimits one;
    one.max_tensors = 1;
    CHECK(code(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
               eight, one) == "loader.too_large");

    // Header lengths that lie.
    std::vector<uint8_t> lie;
    put<uint64_t>(lie, 1ULL << 40);
    append(lie, std::string("{}"));
    write_bytes(dir + "/big.safetensors", lie);
    CHECK(read_safetensors(dir + "/big.safetensors").error.code == "loader.too_large");
    lie.clear();
    put<uint64_t>(lie, 100);
    append(lie, std::string("{}"));
    write_bytes(dir + "/short.safetensors", lie);
    CHECK(read_safetensors(dir + "/short.safetensors").error.code == "loader.truncated");
    write_bytes(dir + "/tiny.safetensors", {1, 2, 3});
    CHECK(read_safetensors(dir + "/tiny.safetensors").error.code == "loader.truncated");
    ReaderLimits limit;
    limit.max_safetensors_header = 4;
    write_bytes(dir + "/x.safetensors", make_safetensors(good, eight));
    CHECK(read_safetensors(dir + "/x.safetensors", {}, limit).error.code == "loader.too_large");
}
