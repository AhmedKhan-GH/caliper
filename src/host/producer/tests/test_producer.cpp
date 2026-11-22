// Tests of the producer runtime's C interface (caliper_producer.h) on every
// platform: the ABI version and the error handling. The runtime's cases on the
// Metal device are in test_producer_metal.mm.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "producer/caliper_producer.h"

#include <string>

TEST_CASE("the producer interface states its ABI version and frees an empty error") {
    CHECK(caliper_producer_abi_version() == CALIPER_PRODUCER_ABI_VERSION);
    caliper_producer_error e{};
    caliper_producer_error_free(&e);
    caliper_producer_error_free(nullptr);
    caliper_producer_destroy(nullptr);
}

TEST_CASE("a refused call fills the error with a stable code and a sentence") {
    caliper_producer* rt = nullptr;
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_create(nullptr, &rt, &e);
    if (s == CALIPER_PRODUCER_OK) {
        caliper_producer_destroy(rt);
        return;
    }
    CHECK(rt == nullptr);
    CHECK(e.status == s);
    REQUIRE(e.code);
    CHECK(std::string(e.code).rfind("program.", 0) == 0);
    REQUIRE(e.message);
    caliper_producer_error_free(&e);
    CHECK(e.message == nullptr);
}
