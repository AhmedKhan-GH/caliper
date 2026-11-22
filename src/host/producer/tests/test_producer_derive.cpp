// Tests of the derive half's C interface on every platform
// (producer/caliper_producer_derive.h, design/programmable-graphics.md): a call
// without a runtime or with a NULL pointer is refused with a code and leaves
// nothing behind. The cases on the Metal device are in
// test_producer_derive_metal.mm.
#include <doctest/doctest.h>

#include "producer/caliper_producer_derive.h"

#include <cstring>

TEST_CASE("the derive half's functions refuse a call without a runtime") {
    caliper_producer_derive_desc desc{};
    desc.id = "arcs";
    uint32_t program = 7;
    caliper_producer_error error{};
    CHECK(caliper_producer_compile_derive(nullptr, &desc, &program, &error) != CALIPER_PRODUCER_OK);
    CHECK(program == 0);
    REQUIRE(error.code != nullptr);
    CHECK(std::strlen(error.code) > 0);
    caliper_producer_error_free(&error);

    caliper_producer_output out{};
    out.count = 3;
    caliper_producer_error error2{};
    CHECK(caliper_producer_read_resource(nullptr, 1, "arcs", &out, &error2) != CALIPER_PRODUCER_OK);
    CHECK(out.count == 0);
    REQUIRE(error2.code != nullptr);
    caliper_producer_error_free(&error2);
}
