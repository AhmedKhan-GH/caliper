#include <doctest/doctest.h>
#include <caliper/adapters/alloc_registry.hpp>

using caliper::adapters::AllocRegistry;

TEST_CASE("AllocRegistry: interval lookup with offset and extent bounds") {
    AllocRegistry r;
    int h1 = 0, h2 = 0;
    r.add(0x1000, 0x1000, &h1);              // [0x1000, 0x2000)
    r.add(0x8000, 0x0800, &h2);              // [0x8000, 0x8800)

    auto hit = r.find(reinterpret_cast<void*>(0x1200), 0x100);
    REQUIRE(hit.has_value());
    CHECK(hit->os_handle == &h1);
    CHECK(hit->offset == 0x200);
    CHECK(hit->size == 0x1000);
    CHECK(hit->base == 0x1000);

    CHECK(r.find(reinterpret_cast<void*>(0x1000), 0x1000).has_value());  // exact fit
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x1F00), 0x200).has_value()); // spills out
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x0FFF), 1).has_value());     // below
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x2000), 1).has_value());     // end is exclusive
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x3000), 1).has_value());     // gap

    r.remove(0x1000);
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x1200), 1).has_value());
    CHECK(r.find(reinterpret_cast<void*>(0x8400), 0x100)->os_handle == &h2);
}
