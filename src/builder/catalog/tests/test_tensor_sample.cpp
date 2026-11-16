// Tests of sample_tensor(): statistics, slices, every dtype, non-finite
// values, the stride beyond 2^26 elements and hostile sizes.
#include <caliper_builder/catalog/tensor_sample.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using caliper_builder::CatalogDtype;
using nlohmann::json;
namespace cb = caliper_builder;

namespace {

template <typename T>
std::vector<std::uint8_t> bytes_of(const std::vector<T>& values) {
    std::vector<std::uint8_t> out(values.size() * sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), values.data(), out.size());  // the test host is little-endian
    return out;
}

}  // namespace

TEST_CASE("a small f32 tensor gives exact statistics and its whole slice") {
    const std::vector<float> v = {1, 2, 3, 4, 5, 6};
    const auto b = bytes_of(v);
    const auto r = cb::sample_tensor(CatalogDtype::f32, {2, 3}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.stats.count == 6);
    CHECK(r.stats.sampled == 6);
    CHECK(r.stats.exact);
    CHECK(r.stats.nonfinite == 0);
    CHECK(*r.stats.min == 1.0);
    CHECK(*r.stats.max == 6.0);
    CHECK(*r.stats.mean == doctest::Approx(3.5));
    CHECK(*r.stats.std == doctest::Approx(std::sqrt(17.5 / 6.0)));
    CHECK(r.slice.start == std::vector<std::int64_t>{0, 0});
    CHECK(r.slice.shape == std::vector<std::int64_t>{2, 3});
    CHECK(r.slice.values == json::array({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST_CASE("a slice box is read in row-major order from its start") {
    std::vector<std::int32_t> v(4 * 5 * 6);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::int32_t>(i);
    const auto b = bytes_of(v);
    const auto r = cb::sample_tensor(CatalogDtype::i32, {4, 5, 6}, b.data(), b.size(), {1, 2, 3}, {2, 2, 2});
    REQUIRE(r.error.ok());
    // element [a][b][c] = a*30 + b*6 + c
    CHECK(r.slice.values == json::array({1 * 30 + 2 * 6 + 3, 1 * 30 + 2 * 6 + 4, 1 * 30 + 3 * 6 + 3,
                                         1 * 30 + 3 * 6 + 4, 2 * 30 + 2 * 6 + 3, 2 * 30 + 2 * 6 + 4,
                                         2 * 30 + 3 * 6 + 3, 2 * 30 + 3 * 6 + 4}));
    CHECK(r.slice.values[0].is_number_integer());
}

TEST_CASE("the default box fits 64 per axis and 4096 values") {
    CHECK(cb::default_slice_count({100}) == std::vector<std::int64_t>{64});
    CHECK(cb::default_slice_count({10}) == std::vector<std::int64_t>{10});
    CHECK(cb::default_slice_count({100, 100}) == std::vector<std::int64_t>{64, 64});
    CHECK(cb::default_slice_count({100, 100, 100}) == std::vector<std::int64_t>{16, 16, 16});
    CHECK(cb::default_slice_count({12, 12, 64, 64}) == std::vector<std::int64_t>{8, 8, 8, 8});
    CHECK(cb::default_slice_count({9, 9, 9, 9, 9}) == std::vector<std::int64_t>{5, 5, 5, 5, 5});
    CHECK(cb::default_slice_count({9, 9, 9, 9, 9, 9, 9, 9}) == std::vector<std::int64_t>(8, 2));
    for (std::size_t rank = 1; rank <= 8; ++rank) {
        const auto count = cb::default_slice_count(std::vector<std::int64_t>(rank, 1000));
        std::int64_t product = 1;
        for (auto c : count) {
            CHECK(c <= 64);
            product *= c;
        }
        CHECK(product <= 4096);
    }
}

TEST_CASE("f16 values are decoded exactly") {
    CHECK(cb::catalog_half_to_float(0x3C00) == 1.0f);
    CHECK(cb::catalog_half_to_float(0xC000) == -2.0f);
    CHECK(cb::catalog_half_to_float(0x7BFF) == 65504.0f);
    CHECK(cb::catalog_half_to_float(0x0001) == std::ldexp(1.0f, -24));  // the smallest subnormal
    CHECK(cb::catalog_half_to_float(0x0400) == std::ldexp(1.0f, -14));  // the smallest normal
    CHECK(cb::catalog_half_to_float(0x8000) == 0.0f);
    CHECK(std::signbit(cb::catalog_half_to_float(0x8000)));
    CHECK(std::isinf(cb::catalog_half_to_float(0x7C00)));
    CHECK(std::isnan(cb::catalog_half_to_float(0x7E00)));

    const std::vector<std::uint16_t> h = {0x3C00, 0x7C00, 0x7E00, 0xFC00};
    const auto b = bytes_of(h);
    const auto r = cb::sample_tensor(CatalogDtype::f16, {4}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values == json::array({1.0, "inf", "nan", "-inf"}));
    CHECK(r.stats.nonfinite == 3);
    CHECK(*r.stats.min == 1.0);
}

TEST_CASE("statistics cover the finite values and are null when there are none") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> v = {nan, 2.0, inf, 4.0};
    auto b = bytes_of(v);
    auto r = cb::sample_tensor(CatalogDtype::f64, {4}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.stats.nonfinite == 2);
    CHECK(*r.stats.mean == 3.0);
    CHECK(*r.stats.std == 1.0);

    v = {nan, inf};
    b = bytes_of(v);
    r = cb::sample_tensor(CatalogDtype::f64, {2}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.stats.nonfinite == 2);
    CHECK(!r.stats.min);
    CHECK(!r.stats.mean);
    CHECK(!r.stats.std);
}

TEST_CASE("integer, unsigned and bool values keep their exact form") {
    const std::vector<std::uint64_t> u = {0, std::numeric_limits<std::uint64_t>::max()};
    auto b = bytes_of(u);
    auto r = cb::sample_tensor(CatalogDtype::u64, {2}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values[1].get<std::uint64_t>() == std::numeric_limits<std::uint64_t>::max());

    const std::vector<std::int64_t> i = {std::numeric_limits<std::int64_t>::min(), -1};
    b = bytes_of(i);
    r = cb::sample_tensor(CatalogDtype::i64, {2}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values[0].get<std::int64_t>() == std::numeric_limits<std::int64_t>::min());

    const std::vector<std::int8_t> s8 = {-128, 127};
    b = bytes_of(s8);
    r = cb::sample_tensor(CatalogDtype::i8, {2}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values == json::array({-128, 127}));

    const std::vector<std::uint16_t> u16 = {65535, 1};
    b = bytes_of(u16);
    r = cb::sample_tensor(CatalogDtype::u16, {2}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values == json::array({65535, 1}));

    const std::vector<std::uint8_t> flags = {0, 1, 7};
    r = cb::sample_tensor(CatalogDtype::boolean, {3}, flags.data(), flags.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values == json::array({false, true, true}));
    CHECK(*r.stats.max == 1.0);
    CHECK(*r.stats.mean == doctest::Approx(2.0 / 3.0));
}

TEST_CASE("a rank-0 tensor and an empty axis") {
    const std::vector<float> one = {2.5f};
    auto b = bytes_of(one);
    auto r = cb::sample_tensor(CatalogDtype::f32, {}, b.data(), b.size());
    REQUIRE(r.error.ok());
    CHECK(r.slice.values == json::array({2.5}));
    CHECK(r.slice.shape.empty());
    CHECK(r.stats.count == 1);

    r = cb::sample_tensor(CatalogDtype::f32, {0, 5}, nullptr, 0);
    REQUIRE(r.error.ok());
    CHECK(r.stats.count == 0);
    CHECK(!r.stats.min);
    CHECK(r.slice.values.empty());
    CHECK(r.slice.shape == std::vector<std::int64_t>{0, 5});
}

TEST_CASE("bytes that do not match the shape are refused before anything is read") {
    const std::vector<float> v = {1, 2, 3};
    const auto b = bytes_of(v);
    CHECK(cb::sample_tensor(CatalogDtype::f32, {4}, b.data(), b.size()).error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f32, {3}, b.data(), b.size() - 1).error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f64, {3}, b.data(), b.size()).error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f32, {-3}, b.data(), b.size()).error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f32, {3}, nullptr, b.size()).error.code == "catalog.bad_tensor");
    // A shape whose byte size overflows is refused without a read.
    CHECK(cb::sample_tensor(CatalogDtype::f64, {INT64_C(1) << 31, INT64_C(1) << 31, 4}, b.data(), b.size())
              .error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f64, {INT64_C(1) << 40, INT64_C(1) << 20}, b.data(), 8)
              .error.code == "catalog.bad_tensor");
    CHECK(cb::sample_tensor(CatalogDtype::f32, std::vector<std::int64_t>(9, 1), b.data(), 4).error.code ==
          "catalog.bad_tensor");
}

TEST_CASE("start and count outside their bounds are refused") {
    std::vector<float> v(100 * 100);
    const auto b = bytes_of(v);
    auto code = [&](std::vector<std::int64_t> start, std::vector<std::int64_t> count) {
        return cb::sample_tensor(CatalogDtype::f32, {100, 100}, b.data(), b.size(), start, count).error.code;
    };
    CHECK(code({0, 0}, {64, 64}) == "");
    CHECK(code({36, 36}, {64, 64}) == "");
    CHECK(code({37, 0}, {64, 1}) == "catalog.bad_request");  // past the end
    CHECK(code({0}, {}) == "catalog.bad_request");           // one entry per axis
    CHECK(code({}, {1, 1, 1}) == "catalog.bad_request");
    CHECK(code({-1, 0}, {}) == "catalog.bad_request");
    CHECK(code({100, 0}, {}) == "catalog.bad_request");
    CHECK(code({0, 0}, {65, 1}) == "catalog.bad_request");   // more than 64 on an axis
    CHECK(code({0, 0}, {0, 1}) == "catalog.bad_request");    // 0 only on an empty axis
    CHECK(code({0, 0}, {64, 64}) == "");

    std::vector<float> w(64 * 64 * 2);
    const auto c = bytes_of(w);
    CHECK(cb::sample_tensor(CatalogDtype::f32, {64, 64, 2}, c.data(), c.size(), {}, {64, 64, 2}).error.code ==
          "catalog.bad_request");  // 8192 values
}

TEST_CASE("statistics of a tensor beyond 2^26 elements use a stride") {
    const std::size_t n = (std::size_t(1) << 26) + 1;
    std::vector<std::uint8_t> big(n, 3);
    big[0] = 1;
    big[n - 1] = 200;  // index 2^26 is even, so the stride of 2 reads it
    const auto r = cb::sample_tensor(CatalogDtype::u8, {static_cast<std::int64_t>(n)}, big.data(), big.size());
    REQUIRE(r.error.ok());
    CHECK(r.stats.count == n);
    CHECK(r.stats.sampled == (n + 1) / 2);
    CHECK_FALSE(r.stats.exact);
    CHECK(*r.stats.min == 1.0);
    CHECK(*r.stats.max == 200.0);
    CHECK(r.slice.values.size() == 64);
}
