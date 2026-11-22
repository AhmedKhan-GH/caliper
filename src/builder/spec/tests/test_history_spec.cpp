// The spec tests of kept steps, the window step and the transport control
// (design/instrument-interface.md, Time Transport).
#include <doctest/doctest.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include "helpers.h"

using namespace spec_test;

namespace {

const char* kHead = "caliper_spec = 1\n[applet]\nid = \"h\"\n";

cs::ValidationReport check(const std::string& body) { return cs::check_toml(std::string(kHead) + body, fixture_resolver()); }

const std::string kLive = "[[source]]\nid = \"act\"\ncatalog = \"live:attn.weights\"\n";

}  // namespace

TEST_CASE("history is read on a live tensor source and refused elsewhere") {
    cs::ValidationReport r = check(kLive.substr(0, kLive.size() - 1) + "\nhistory = 2048\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
    CHECK_FALSE(has(r.diagnostics, "spec.not_built"));

    r = check("[[source]]\nid = \"act\"\ncatalog = \"live:attn.weights\"\nhistory = 70000\n");
    CHECK(has(r.diagnostics, "spec.out_of_range", "source[0].history"));
    const cs::Diagnostic* d = find(r.diagnostics, "spec.out_of_range");
    REQUIRE(d);
    CHECK(d->line == 7);

    r = check("[[source]]\nid = \"w\"\ncatalog = \"tensor:acts\"\nhistory = 10\n");
    CHECK(has(r.diagnostics, "spec.history_not_allowed", "source[0].history"));

    // A derived source of a live tensor may keep steps.
    r = check(kLive + "[[source]]\nid = \"mean\"\nfrom = \"act\"\nhistory = 8\n"
                      "transform = [{ op = \"reduce\", fn = \"mean\", axis = 0 }]\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
}

TEST_CASE("a window stacks the kept steps of its from source") {
    cs::ValidationReport r =
        check(kLive.substr(0, kLive.size() - 1) +
              "\ntransform = [{ op = \"reduce\", fn = \"mean\", axis = [0, 1] }]\n"
              "[[source]]\nid = \"recent\"\nfrom = \"act\"\ntransform = [{ op = \"window\", steps = 16 }]\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
    REQUIRE(r.sources.size() == 2);
    REQUIRE(r.sources[1].shape);
    CHECK(*r.sources[1].shape == std::vector<std::int64_t>{-1, 64, 64});

    r = check(kLive + "[[source]]\nid = \"range\"\nfrom = \"act\"\n"
                      "transform = [{ op = \"window\", start = 10, stop = 20 }]\n");
    INFO(dump(r.diagnostics));
    CHECK(r.ok());
}

TEST_CASE("a window's refusals") {
    // Not the first step.
    cs::ValidationReport r = check(kLive + "[[source]]\nid = \"x\"\nfrom = \"act\"\n"
                                           "transform = [{ op = \"reduce\", fn = \"mean\", axis = 0 },"
                                           " { op = \"window\", steps = 4 }]\n");
    CHECK(has(r.diagnostics, "spec.bad_transform", "source[1].transform[1]"));
    // No from.
    r = check("[[source]]\nid = \"x\"\ncatalog = \"live:attn.weights\"\ntransform = [{ op = \"window\" }]\n");
    CHECK(has(r.diagnostics, "spec.bad_transform", "source[0].transform[0]"));
    // steps beside start.
    r = check(kLive + "[[source]]\nid = \"x\"\nfrom = \"act\"\ntransform = [{ op = \"window\", steps = 4, start = 1 }]\n");
    CHECK(has(r.diagnostics, "spec.bad_transform", "source[1].transform[0]"));
    // steps out of range.
    r = check(kLive + "[[source]]\nid = \"x\"\nfrom = \"act\"\ntransform = [{ op = \"window\", steps = 0 }]\n");
    CHECK(has(r.diagnostics, "spec.out_of_range", "source[1].transform[0].steps"));
    // A from source that keeps no steps.
    r = check(kLive + "[[source]]\nid = \"mean\"\nfrom = \"act\"\n"
                      "[[source]]\nid = \"x\"\nfrom = \"mean\"\ntransform = [{ op = \"window\" }]\n");
    CHECK(has(r.diagnostics, "spec.no_history", "source[2].transform[0]"));
    // A from source whose own steps read a reference.
    r = check("[[state]]\nid = \"token\"\ntype = \"integer\"\n"
              "[[source]]\nid = \"act\"\ncatalog = \"live:attn.weights\"\n"
              "transform = [{ op = \"slice\", axis = 0, index = { state = \"token\" } }]\n"
              "[[source]]\nid = \"x\"\nfrom = \"act\"\ntransform = [{ op = \"window\" }]\n");
    CHECK(has(r.diagnostics, "spec.reference_in_history", "source[1].transform[0]"));
}

TEST_CASE("a spec has at most one transport, with a speed from 0.1 to 1,000") {
    const std::string transport = "[[control]]\nid = \"t\"\nkind = \"transport\"\nbind = \"cursor\"\n";
    cs::ValidationReport r = check(kLive + transport + "[[control]]\nid = \"u\"\nkind = \"transport\"\nbind = \"cursor\"\n");
    CHECK(has(r.diagnostics, "spec.too_many", "control[1]"));
    CHECK_FALSE(has(r.diagnostics, "spec.too_many", "control[0]"));
    r = check(kLive + transport + "options = { speed = 0.01, keys = false }\n");
    CHECK(has(r.diagnostics, "spec.out_of_range", "control[0].options.speed"));
    CHECK_FALSE(has(r.diagnostics, "spec.not_built"));
    r = check(kLive + "[[control]]\nid = \"s\"\nkind = \"slider\"\nbind = \"value\"\noptions = { speed = 2 }\n");
    CHECK(has(r.diagnostics, "spec.unknown_option", "control[0].options.speed"));
}
