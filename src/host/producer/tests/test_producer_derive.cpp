// Tests of the derive half's C interface (producer/caliper_producer_derive.h,
// design/programmable-graphics.md): until the derive half is built, both
// functions answer CALIPER_PRODUCER_UNSUPPORTED with program.not_built and
// leave nothing behind.
#include <doctest/doctest.h>

#include "producer/caliper_producer_derive.h"

#include <cstring>

TEST_CASE("the derive half's functions answer program.not_built") {
    caliper_producer_derive_desc desc{};
    desc.id = "arcs";
    uint32_t program = 7;
    caliper_producer_error error{};
    CHECK(caliper_producer_compile_derive(nullptr, &desc, &program, &error) == CALIPER_PRODUCER_UNSUPPORTED);
    CHECK(program == 0);
    REQUIRE(error.code != nullptr);
    CHECK(std::strcmp(error.code, "program.not_built") == 0);
    caliper_producer_error_free(&error);

    caliper_producer_output out{};
    out.count = 3;
    caliper_producer_error error2{};
    CHECK(caliper_producer_read_resource(nullptr, 1, "arcs", &out, &error2) == CALIPER_PRODUCER_UNSUPPORTED);
    CHECK(out.count == 0);
    CHECK(std::strcmp(error2.code, "program.not_built") == 0);
    caliper_producer_error_free(&error2);
}
