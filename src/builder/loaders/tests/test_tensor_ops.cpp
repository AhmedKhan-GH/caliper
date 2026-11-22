// Tests of TensorData and the tensor transforms, against values computed by hand.
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/loaders/tensor_ops.h>

#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "test_util.h"

using namespace caliper_builder;
using nlohmann::json;
using loaders_test::raw;

namespace {

TensorData f32(std::vector<int64_t> shape, std::vector<float> values) {
    TensorResult r = TensorData::from_bytes(DType::f32, std::move(shape), raw(values));
    REQUIRE(r.error.ok());
    return r.tensor;
}

TensorData f64(std::vector<int64_t> shape, std::vector<double> values) {
    TensorResult r = TensorData::from_bytes(DType::f64, std::move(shape), raw(values));
    REQUIRE(r.error.ok());
    return r.tensor;
}

TensorTransform parse(const json& j) {
    TensorTransform t;
    Error e = transform_from_json(j, &t);
    REQUIRE_MESSAGE(e.ok(), e.code << ": " << e.message);
    return t;
}

TensorData apply(const TensorData& in, const json& j) {
    TensorResult r = apply_transform(in, parse(j));
    REQUIRE_MESSAGE(r.error.ok(), r.error.code << ": " << r.error.message);
    return r.tensor;
}

std::vector<double> values(const TensorData& t) {
    std::vector<double> v;
    for (uint64_t i = 0; i < t.numel(); ++i) v.push_back(t.element(i));
    return v;
}

void check_close(const std::vector<double>& got, const std::vector<double>& want, double eps = 1e-5) {
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(want[i])) {
            CHECK_MESSAGE(std::isnan(got[i]), "index " << i);
        } else {
            CHECK_MESSAGE(got[i] == doctest::Approx(want[i]).epsilon(eps), "index " << i);
        }
    }
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

TEST_CASE("dtype names and sizes round-trip") {
    for (const char* name : {"u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "f32", "f64", "bool"}) {
        DType d;
        REQUIRE(dtype_from_name(name, &d));
        CHECK(std::string(dtype_name(d)) == name);
    }
    DType d;
    CHECK_FALSE(dtype_from_name("bf16", &d));
    CHECK(dtype_size(DType::f16) == 2);
    CHECK(dtype_size(DType::boolean) == 1);
    CHECK(dtype_size(DType::i64) == 8);
}

TEST_CASE("half and bfloat16 conversion is exact") {
    CHECK(half_to_float(0x3C00) == 1.0f);
    CHECK(half_to_float(0x4000) == 2.0f);
    CHECK(half_to_float(0xB800) == -0.5f);
    CHECK(half_to_float(0x7BFF) == 65504.0f);
    CHECK(half_to_float(0x0001) == std::ldexp(1.0f, -24));  // the smallest subnormal
    CHECK(half_to_float(0x03FF) == std::ldexp(1023.0f, -24));
    CHECK(std::isinf(half_to_float(0x7C00)));
    CHECK(half_to_float(0xFC00) < 0);
    CHECK(std::isnan(half_to_float(0x7E00)));
    CHECK(std::signbit(half_to_float(0x8000)));
    CHECK(bfloat16_to_float(0x3F80) == 1.0f);
    CHECK(bfloat16_to_float(0xC040) == -3.0f);
}

TEST_CASE("check_shape refuses hostile shapes") {
    uint64_t numel = 0, nbytes = 0;
    CHECK(check_shape({2, 3}, DType::f32, &numel, &nbytes).ok());
    CHECK(numel == 6);
    CHECK(nbytes == 24);
    CHECK(check_shape({}, DType::f64, &numel, &nbytes).ok());
    CHECK(numel == 1);
    CHECK(check_shape({1, 1, 1, 1, 1, 1, 1, 1, 1}, DType::u8, nullptr, nullptr).code == "tensor.rank");
    CHECK(check_shape({-1}, DType::u8, nullptr, nullptr).code == "tensor.shape");
    CHECK(check_shape({1LL << 62, 1LL << 62}, DType::u8, nullptr, nullptr).code == "tensor.too_large");
    CHECK(check_shape({1LL << 62}, DType::f64, nullptr, nullptr).code == "tensor.too_large");
    CHECK(check_shape({0, 1LL << 62, 1LL << 62}, DType::u8, &numel, nullptr).ok());
    CHECK(numel == 0);
    CHECK(TensorData::from_bytes(DType::f32, {2}, std::vector<uint8_t>(7)).error.code == "tensor.size");
}

TEST_CASE("map_file maps at an unaligned offset and refuses a short file") {
    const std::string dir = loaders_test::fresh_dir("map_file");
    std::vector<uint8_t> bytes(7, 0xAB);
    loaders_test::append(bytes, raw(std::vector<float>{1.5f, -2.0f, 3.25f}));
    loaders_test::write_bytes(dir + "/t.bin", bytes);
    TensorResult r = TensorData::map_file(dir + "/t.bin", 7, DType::f32, {3});
    REQUIRE(r.error.ok());
    CHECK(r.tensor.is_mapped());
    CHECK(r.tensor.file_offset() == 7);
    check_close(values(r.tensor), {1.5, -2.0, 3.25});
    CHECK(r.tensor.mutable_data() == nullptr);
    CHECK(TensorData::map_file(dir + "/t.bin", 8, DType::f32, {3}).error.code == "loader.truncated");
    CHECK(TensorData::map_file(dir + "/t.bin", 1u << 30, DType::f32, {1}).error.code == "loader.truncated");
    CHECK(TensorData::map_file(dir + "/missing.bin", 0, DType::f32, {1}).error.code == "loader.io");
    CHECK(TensorData::map_file(dir, 0, DType::f32, {1}).error.code == "loader.io");
    TensorResult empty = TensorData::map_file(dir + "/t.bin", 19, DType::f32, {0, 4});
    REQUIRE(empty.error.ok());
    CHECK(empty.tensor.numel() == 0);
    CHECK(empty.tensor.data() == nullptr);
}

TEST_CASE("element_at reads only inside the tensor, for an index from outside it") {
    const TensorData t = f32({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(t.element_at(0) == std::optional<double>(1));
    CHECK(t.element_at(5) == std::optional<double>(6));
    CHECK_FALSE(t.element_at(6).has_value());
    CHECK_FALSE(t.element_at(-1).has_value());
    CHECK_FALSE(t.element_at(std::numeric_limits<int64_t>::min()).has_value());
    CHECK(t.element_at(1, 2) == std::optional<double>(6));
    CHECK_FALSE(t.element_at(2, 0).has_value());
    CHECK_FALSE(t.element_at(0, 3).has_value());
    CHECK_FALSE(t.element_at(-1, 1).has_value());
    CHECK_FALSE(f32({6}, {1, 2, 3, 4, 5, 6}).element_at(0, 0).has_value());   // not of rank 2
    CHECK_FALSE(TensorData().element_at(0).has_value());
}

TEST_CASE("a slice with a step near the int64_t limit counts one row, without overflow") {
    const TensorData t = f32({2, 3}, {1, 2, 3, 4, 5, 6});
    const TensorData s = apply(t, {{"op", "slice"}, {"axis", 1}, {"start", 1}, {"step", std::numeric_limits<int64_t>::max()}});
    CHECK(s.shape() == std::vector<int64_t>{2, 1});
    check_close(values(s), {2, 5});
}

TEST_CASE("slice follows Python semantics") {
    const TensorData t = f32({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData s = apply(t, {{"op", "slice"}, {"axis", 1}, {"start", 1}});
    CHECK(s.shape() == std::vector<int64_t>{2, 2});
    CHECK(s.dtype() == DType::f32);
    check_close(values(s), {2, 3, 5, 6});

    s = apply(t, {{"op", "slice"}, {"axis", -1}, {"index", -1}});
    CHECK(s.shape() == std::vector<int64_t>{2});
    check_close(values(s), {3, 6});

    s = apply(t, {{"op", "slice"}, {"axis", 1}, {"step", 2}});
    check_close(values(s), {1, 3, 4, 6});

    s = apply(t, {{"op", "slice"}, {"axis", 1}, {"stop", -1}});
    check_close(values(s), {1, 2, 4, 5});

    s = apply(t, {{"op", "slice"}, {"axis", 0}, {"start", -100}, {"stop", 100}});
    check_close(values(s), {1, 2, 3, 4, 5, 6});

    s = apply(t, {{"op", "slice"}, {"axis", 0}, {"index", 1}});
    CHECK(s.shape() == std::vector<int64_t>{3});
    check_close(values(s), {4, 5, 6});

    SUBCASE("an empty result") {
        s = apply(t, {{"op", "slice"}, {"axis", 0}, {"start", 5}});
        CHECK(s.shape() == std::vector<int64_t>{0, 3});
        CHECK(s.numel() == 0);
        s = apply(t, {{"op", "slice"}, {"axis", 1}, {"start", 2}, {"stop", 1}});
        CHECK(s.shape() == std::vector<int64_t>{2, 0});
        TensorStats st = tensor_stats(s);
        CHECK(st.count == 0);
        CHECK_FALSE(st.min.has_value());
        CHECK(st.to_json()["min"].is_null());
    }
    SUBCASE("errors") {
        CHECK(apply_transform(t, parse({{"op", "slice"}, {"axis", 1}, {"index", 3}})).error.code ==
              "transform.index_out_of_range");
        CHECK(apply_transform(t, parse({{"op", "slice"}, {"axis", 1}, {"index", -4}})).error.code ==
              "transform.index_out_of_range");
        CHECK(apply_transform(t, parse({{"op", "slice"}, {"axis", 2}})).error.code == "transform.axis_out_of_range");
        CHECK(apply_transform(f32({}, {1}), parse({{"op", "slice"}, {"axis", 0}})).error.code ==
              "transform.axis_out_of_range");
    }
}

TEST_CASE("slice keeps integer dtypes") {
    TensorResult r = TensorData::from_bytes(DType::i16, {4}, raw(std::vector<int16_t>{-1, 2, -3, 4}));
    REQUIRE(r.error.ok());
    TensorData s = apply(r.tensor, {{"op", "slice"}, {"axis", 0}, {"start", 1}, {"step", 2}});
    CHECK(s.dtype() == DType::i16);
    check_close(values(s), {2, 4});
}

TEST_CASE("reduce computes each function") {
    const TensorData t = f32({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData r = apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}});
    CHECK(r.dtype() == DType::f32);
    CHECK(r.shape() == std::vector<int64_t>{3});
    check_close(values(r), {2.5, 3.5, 4.5});

    r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "sum"}, {"keepdims", true}});
    CHECK(r.shape() == std::vector<int64_t>{2, 1});
    check_close(values(r), {6, 15});

    r = apply(t, {{"op", "reduce"}, {"axis", json::array({0, -1})}, {"fn", "max"}});
    CHECK(r.shape().empty());
    check_close(values(r), {6});

    r = apply(t, {{"op", "reduce"}, {"axis", json::array({1, 0})}, {"fn", "min"}, {"keepdims", true}});
    CHECK(r.shape() == std::vector<int64_t>{1, 1});
    check_close(values(r), {1});

    r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "var"}});
    check_close(values(r), {2.0 / 3.0, 2.0 / 3.0});
    r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "std"}});
    check_close(values(r), {0.816497, 0.816497});
    r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "norm"}});
    check_close(values(r), {std::sqrt(14.0), std::sqrt(77.0)});

    r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "argmax"}});
    CHECK(r.dtype() == DType::i64);
    check_close(values(r), {2, 2});
    r = apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "argmin"}});
    check_close(values(r), {0, 0, 0});
}

TEST_CASE("reduce over a middle axis of a rank-3 tensor") {
    // shape [2, 2, 2]: values 0..7; the mean over axis 1 averages pairs two apart.
    const TensorData t = f32({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
    TensorData r = apply(t, {{"op", "reduce"}, {"axis", 1}, {"fn", "mean"}});
    CHECK(r.shape() == std::vector<int64_t>{2, 2});
    check_close(values(r), {1, 2, 5, 6});
    r = apply(t, {{"op", "reduce"}, {"axis", json::array({0, 2})}, {"fn", "sum"}});
    CHECK(r.shape() == std::vector<int64_t>{2});
    check_close(values(r), {0 + 1 + 4 + 5, 2 + 3 + 6 + 7});
}

TEST_CASE("entropy is in nats over a normalised distribution") {
    const TensorData t = f32({4, 2}, {0.5f, 0.5f, 0.25f, 0.75f, 0.0f, 1.0f, 0.0f, 0.0f});
    TensorData r = apply(t, {{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}});
    CHECK(r.dtype() == DType::f32);
    check_close(values(r), {std::log(2.0), -(0.25 * std::log(0.25) + 0.75 * std::log(0.75)), 0.0, kNaN});

    // Unnormalised counts: four equal values give ln 4.
    r = apply(f32({4}, {3, 3, 3, 3}), {{"op", "reduce"}, {"axis", 0}, {"fn", "entropy"}});
    check_close(values(r), {std::log(4.0)});

    // Negative values contribute nothing; NaN propagates.
    r = apply(f32({3}, {-1, 1, 1}), {{"op", "reduce"}, {"axis", 0}, {"fn", "entropy"}});
    check_close(values(r), {std::log(2.0)});
    r = apply(f32({2}, {static_cast<float>(kNaN), 1}), {{"op", "reduce"}, {"axis", 0}, {"fn", "entropy"}});
    check_close(values(r), {kNaN});

    // Two axes form one distribution.
    r = apply(f32({1, 2, 2}, {1, 1, 1, 1}), {{"op", "reduce"}, {"axis", json::array({1, 2})}, {"fn", "entropy"}});
    check_close(values(r), {std::log(4.0)});
}

TEST_CASE("f16 input widens and f64 input stays f64") {
    std::vector<uint16_t> halves = {0x3C00, 0x4000, 0xB800};  // 1, 2, -0.5
    TensorResult h = TensorData::from_bytes(DType::f16, {3}, raw(halves));
    REQUIRE(h.error.ok());
    check_close(values(h.tensor), {1, 2, -0.5});
    TensorData r = apply(h.tensor, {{"op", "reduce"}, {"axis", 0}, {"fn", "sum"}});
    CHECK(r.dtype() == DType::f32);
    check_close(values(r), {2.5});
    TensorData n = apply(h.tensor, {{"op", "normalise"}, {"mode", "minmax"}});
    CHECK(n.dtype() == DType::f32);
    check_close(values(n), {0.6, 1.0, 0.0});

    TensorData d = f64({2}, {1e300, 1e300});
    r = apply(d, {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}});
    CHECK(r.dtype() == DType::f64);
    check_close(values(r), {1e300});
}

TEST_CASE("integer and bool input reduce to f32") {
    TensorResult u = TensorData::from_bytes(DType::u8, {4}, {0, 255, 1, 2});
    REQUIRE(u.error.ok());
    TensorData r = apply(u.tensor, {{"op", "reduce"}, {"axis", 0}, {"fn", "max"}});
    CHECK(r.dtype() == DType::f32);
    check_close(values(r), {255});
    TensorResult b = TensorData::from_bytes(DType::boolean, {3}, {1, 0, 1});
    REQUIRE(b.error.ok());
    r = apply(b.tensor, {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}});
    check_close(values(r), {2.0 / 3.0});
}

TEST_CASE("NaN propagates and empty reductions have defined results") {
    const TensorData t = f32({3}, {1, static_cast<float>(kNaN), 3});
    check_close(values(apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "max"}})), {kNaN});
    check_close(values(apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "min"}})), {kNaN});
    check_close(values(apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}})), {kNaN});
    check_close(values(apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "argmax"}})), {1});
    check_close(values(apply(t, {{"op", "reduce"}, {"axis", 0}, {"fn", "argmin"}})), {1});

    const TensorData empty = f32({2, 0}, {});
    TensorData r = apply(empty, {{"op", "reduce"}, {"axis", 1}, {"fn", "sum"}});
    CHECK(r.shape() == std::vector<int64_t>{2});
    check_close(values(r), {0, 0});
    check_close(values(apply(empty, {{"op", "reduce"}, {"axis", 1}, {"fn", "mean"}})), {kNaN, kNaN});
    check_close(values(apply(empty, {{"op", "reduce"}, {"axis", 1}, {"fn", "norm"}})), {0, 0});
    check_close(values(apply(empty, {{"op", "reduce"}, {"axis", 1}, {"fn", "max"}})), {kNaN, kNaN});
    CHECK(apply_transform(empty, parse({{"op", "reduce"}, {"axis", 1}, {"fn", "argmax"}})).error.code ==
          "transform.empty_reduce");
    // Reducing the non-empty axis of an empty tensor gives an empty result.
    r = apply(empty, {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}});
    CHECK(r.shape() == std::vector<int64_t>{0});
    CHECK(r.numel() == 0);
}

TEST_CASE("normalise rescales the whole tensor or along an axis") {
    const TensorData t = f32({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData n = apply(t, {{"op", "normalise"}, {"mode", "minmax"}});
    CHECK(n.shape() == t.shape());
    check_close(values(n), {0, 0.2, 0.4, 0.6, 0.8, 1.0});

    n = apply(t, {{"op", "normalise"}, {"mode", "zscore"}, {"axis", 1}});
    check_close(values(n), {-1.224745, 0, 1.224745, -1.224745, 0, 1.224745});

    n = apply(t, {{"op", "normalise"}, {"mode", "softmax"}, {"axis", -1}});
    check_close(values(n), {0.0900306, 0.2447285, 0.6652410, 0.0900306, 0.2447285, 0.6652410});

    n = apply(t, {{"op", "normalise"}, {"mode", "l2"}, {"axis", 1}});
    check_close(values(n), {1 / std::sqrt(14.0), 2 / std::sqrt(14.0), 3 / std::sqrt(14.0), 4 / std::sqrt(77.0),
                            5 / std::sqrt(77.0), 6 / std::sqrt(77.0)});

    // Softmax over axis 0 pairs the rows.
    n = apply(t, {{"op", "normalise"}, {"mode", "softmax"}, {"axis", 0}});
    const double a = 1 / (1 + std::exp(3.0));
    check_close(values(n), {a, a, a, 1 - a, 1 - a, 1 - a});

    // A constant input or a zero vector maps to 0.
    const TensorData c = f32({3}, {2, 2, 2});
    check_close(values(apply(c, {{"op", "normalise"}, {"mode", "minmax"}})), {0, 0, 0});
    check_close(values(apply(c, {{"op", "normalise"}, {"mode", "zscore"}})), {0, 0, 0});
    check_close(values(apply(f32({2}, {0, 0}), {{"op", "normalise"}, {"mode", "l2"}})), {0, 0});
    check_close(values(apply(f32({2}, {1, static_cast<float>(kNaN)}), {{"op", "normalise"}, {"mode", "minmax"}})),
                {kNaN, kNaN});

    // A scalar normalises as a whole.
    check_close(values(apply(f32({}, {5}), {{"op", "normalise"}, {"mode", "softmax"}})), {1});
}

TEST_CASE("transform records are parsed strictly") {
    TensorTransform t;
    CHECK(transform_from_json(json::array(), &t).code == "transform.bad_record");
    CHECK(transform_from_json({{"axis", 0}}, &t).code == "transform.missing_key");
    CHECK(transform_from_json({{"op", "flip"}}, &t).code == "transform.unknown_op");
    Error e = transform_from_json({{"op", "slice"}, {"axis", 0}, {"stride", 2}}, &t);
    CHECK(e.code == "transform.unknown_key");
    CHECK(e.where == "stride");
    CHECK(transform_from_json({{"op", "slice"}}, &t).code == "transform.missing_key");
    CHECK(transform_from_json({{"op", "slice"}, {"axis", 0}, {"step", 0}}, &t).code == "transform.bad_value");
    CHECK(transform_from_json({{"op", "slice"}, {"axis", 0}, {"index", 0}, {"start", 1}}, &t).code ==
          "transform.bad_value");
    CHECK(transform_from_json({{"op", "slice"}, {"axis", json::array({0})}}, &t).code == "transform.bad_value");
    CHECK(transform_from_json({{"op", "slice"}, {"axis", 1.5}}, &t).code == "transform.bad_value");
    CHECK(transform_from_json({{"op", "reduce"}, {"axis", 0}}, &t).code == "transform.missing_key");
    CHECK(transform_from_json({{"op", "reduce"}, {"axis", 0}, {"fn", "median"}}, &t).code == "transform.bad_value");
    CHECK(transform_from_json({{"op", "reduce"}, {"axis", json::array({0, 1})}, {"fn", "argmax"}}, &t).code ==
          "transform.bad_value");
    CHECK(transform_from_json({{"op", "reduce"}, {"axis", json::array()}, {"fn", "sum"}}, &t).code ==
          "transform.bad_value");
    CHECK(transform_from_json({{"op", "reduce"}, {"axis", 0}, {"fn", "sum"}, {"keepdims", 1}}, &t).code ==
          "transform.bad_value");
    CHECK(transform_from_json({{"op", "normalise"}, {"mode", "rank"}}, &t).code == "transform.bad_value");
    CHECK(apply_transform(f32({2, 2}, {1, 2, 3, 4}), parse({{"op", "reduce"}, {"axis", json::array({0, -2})},
                                                            {"fn", "sum"}}))
              .error.code == "transform.duplicate_axis");

    for (const json& j : {json{{"op", "slice"}, {"axis", -1}, {"start", 1}, {"stop", 5}, {"step", 2}},
                          json{{"op", "slice"}, {"axis", 0}, {"index", -1}},
                          json{{"op", "reduce"}, {"axis", json::array({2, 3})}, {"fn", "entropy"}, {"keepdims", true}},
                          json{{"op", "reduce"}, {"axis", 1}, {"fn", "mean"}},
                          json{{"op", "normalise"}, {"mode", "softmax"}, {"axis", -1}},
                          json{{"op", "normalise"}, {"mode", "l2"}}}) {
        CHECK(transform_to_json(parse(j)) == j);
    }
}

TEST_CASE("apply_transforms chains and names the failing transform") {
    const TensorData t = f32({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::vector<TensorTransform> chain = {parse({{"op", "slice"}, {"axis", 0}, {"index", 1}}),
                                          parse({{"op", "reduce"}, {"axis", 1}, {"fn", "sum"}})};
    TensorResult r = apply_transforms(t, chain);
    REQUIRE(r.error.ok());
    CHECK(r.tensor.shape() == std::vector<int64_t>{2});
    check_close(values(r.tensor), {7 + 8 + 9, 10 + 11 + 12});

    chain.push_back(parse({{"op", "reduce"}, {"axis", 1}, {"fn", "sum"}}));
    r = apply_transforms(t, chain);
    CHECK(r.error.code == "transform.axis_out_of_range");
    CHECK(r.error.where == "transform[2].axis");

    std::vector<int64_t> shape;
    DType dtype;
    REQUIRE(transform_result_shape({12, 12, 64, 64}, DType::f16,
                                   parse({{"op", "reduce"}, {"axis", 3}, {"fn", "entropy"}}), &shape, &dtype)
                .ok());
    CHECK(shape == std::vector<int64_t>{12, 12, 64});
    CHECK(dtype == DType::f32);
}

TEST_CASE("statistics count non-finite values and skip them") {
    const float inf = std::numeric_limits<float>::infinity();
    TensorStats s = tensor_stats(f32({6}, {1, 2, 3, 4, static_cast<float>(kNaN), -inf}));
    CHECK(s.count == 6);
    CHECK(s.nonfinite == 2);
    CHECK(*s.min == 1);
    CHECK(*s.max == 4);
    CHECK(*s.mean == doctest::Approx(2.5));
    CHECK(*s.std == doctest::Approx(std::sqrt(1.25)));
    const json j = s.to_json();
    CHECK(j["count"] == 6);
    CHECK(j["nonfinite"] == 2);

    std::vector<uint16_t> halves = {0x7C00, 0x3C00};  // inf, 1
    TensorResult h = TensorData::from_bytes(DType::f16, {2}, raw(halves));
    s = tensor_stats(h.tensor);
    CHECK(s.nonfinite == 1);
    CHECK(*s.mean == 1);

    s = tensor_stats(f32({2}, {static_cast<float>(kNaN), inf}));
    CHECK(s.nonfinite == 2);
    CHECK_FALSE(s.mean.has_value());
}

TEST_CASE("the sample slice is bounded per axis and in total") {
    std::vector<float> big(3 * 100 * 100);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<float>(i);
    json s = tensor_sample(f32({3, 100, 100}, big));
    CHECK(s["shape"] == json::array({1, 64, 64}));
    CHECK(s["start"] == json::array({0, 0, 0}));
    CHECK(s["values"].size() == 1);
    CHECK(s["values"][0].size() == 64);
    CHECK(s["values"][0][1][2] == 102.0);

    s = tensor_sample(f32({2, 3}, {1, 2, 3, 4, 5, 6}));
    CHECK(s["shape"] == json::array({2, 3}));
    CHECK(s["values"] == json::parse("[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"));

    s = tensor_sample(f32({200}, std::vector<float>(200, 1.0f)));
    CHECK(s["shape"] == json::array({64}));

    s = tensor_sample(f32({3}, {static_cast<float>(kNaN), std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}));
    CHECK(s["values"] == json::array({"NaN", "Infinity", "-Infinity"}));

    s = tensor_sample(f32({}, {7}));
    CHECK(s["values"] == 7.0);
    CHECK(s["shape"] == json::array());

    TensorResult u = TensorData::from_bytes(DType::u64, {1}, raw(std::vector<uint64_t>{18446744073709551615ULL}));
    CHECK(tensor_sample(u.tensor)["values"][0] == 18446744073709551615ULL);
    TensorResult b = TensorData::from_bytes(DType::boolean, {2}, {1, 0});
    CHECK(tensor_sample(b.tensor)["values"] == json::array({true, false}));

    s = tensor_sample(f32({0, 5}, {}));
    CHECK(s["shape"] == json::array({0, 5}));
    CHECK(s["values"] == json::array());
}
