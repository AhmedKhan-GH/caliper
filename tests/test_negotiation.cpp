#include <doctest/doctest.h>
#include "negotiation.h"
using namespace caliper_host;

static AppletManifest base() {
    AppletManifest m;
    m.id = "a.b"; m.name = "A"; m.version = "1.0.0";
    m.abi_epoch = 2; m.min_host = "0.6.0";
    m.required_services = {"caliper.ui.v1"};
    return m;
}
static HostCaps host() {
    return HostCaps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
}

TEST_CASE("negotiate: compatible applet passes") {
    auto n = negotiate(base(), host());
    CHECK(n.ok);
    CHECK(n.reason.empty());
}

TEST_CASE("negotiate: epoch mismatch → friendly reason") {
    auto m = base(); m.abi_epoch = 1;
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason ==
        "Built for ABI epoch 1; this host speaks 2 — check for an applet update.");
}

TEST_CASE("negotiate: min_host newer than host → refuse") {
    auto m = base(); m.min_host = "9.9.9";
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Requires host 9.9.9 or newer; this host is 0.6.0.");
}

TEST_CASE("negotiate: missing required service → refuse, first missing named") {
    auto m = base();
    m.required_services = {"caliper.ui.v1", "caliper.jobs.v1", "caliper.metrics.v1"};
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason ==
        "Requires a capability this host doesn't have: caliper.jobs.v1.");
}

TEST_CASE("negotiate: empty min_host means no floor") {
    auto m = base(); m.min_host.clear();
    CHECK(negotiate(m, host()).ok);
}

TEST_CASE("semver_cmp is numeric, not lexical") {
    CHECK(semver_cmp("0.6.0",  "0.6.0")  == 0);
    CHECK(semver_cmp("0.6.0",  "0.10.0") <  0);   // lexical would say >
    CHECK(semver_cmp("1.0.0",  "0.9.9")  >  0);
    CHECK(semver_cmp("0.6.1",  "0.6.0")  >  0);
}
