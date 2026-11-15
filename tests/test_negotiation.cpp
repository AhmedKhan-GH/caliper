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

// The versioning checks (docs/explanations/compatibility.md) and their order:
// epoch, min_host, UI stack, libtorch, services, data schema.
static HostCaps torch_host() {
    HostCaps c = host();
    c.torch_version = "2.5.1";
    c.ui_stack_hash = "aaaaaaaaaaaaaaaaaaaaaaaa";
    return c;
}

TEST_CASE("negotiate: libtorch must match exactly") {
    auto m = base(); m.torch = "2.5.1";
    CHECK(negotiate(m, torch_host()).ok);
    m.torch = "2.4.0";
    auto n = negotiate(m, torch_host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Built against libtorch 2.4.0; this host provides 2.5.1.");
}

TEST_CASE("negotiate: a host without libtorch refuses a torch applet") {
    auto m = base(); m.torch = "2.5.1";
    auto n = negotiate(m, host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Built against libtorch 2.5.1; this host provides no libtorch.");
    m.torch.clear();
    CHECK(negotiate(m, host()).ok);   // no key, no check
}

TEST_CASE("negotiate: UI-stack hash must match when both sides have one") {
    auto m = base();
    m.ui_stack = "aaaaaaaaaaaaaaaaaaaaaaaa";
    CHECK(negotiate(m, torch_host()).ok);
    m.ui_stack = "bbbbbbbbbbbbbbbbbbbbbbbb";
    auto n = negotiate(m, torch_host());
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Built against UI stack bbbbbbbbbbbb; this host uses "
                      "aaaaaaaaaaaa — rebuild the applet against this host's SDK.");
    // An empty host hash disables the check; so does a manifest without one.
    CHECK(negotiate(m, host()).ok);
    m.ui_stack.clear();
    CHECK(negotiate(m, torch_host()).ok);
}

TEST_CASE("negotiate: data schema lower than the stored one is refused") {
    auto m = base(); m.data_schema = 2;
    CHECK(negotiate(m, host(), 0).ok);   // nothing recorded
    CHECK(negotiate(m, host(), 1).ok);   // the applet migrates upward
    CHECK(negotiate(m, host(), 2).ok);
    auto n = negotiate(m, host(), 3);
    CHECK_FALSE(n.ok);
    CHECK(n.reason == "Stored data is schema 3; this applet reads schema 2.");
    m.data_schema = 0;                   // no key, no check
    CHECK(negotiate(m, host(), 3).ok);
}

TEST_CASE("negotiate: check order is epoch, min_host, UI stack, torch, services, data") {
    auto m = base();
    m.abi_epoch = 1; m.min_host = "9.9.9"; m.ui_stack = "cccccccccccccccc";
    m.torch = "1.0.0"; m.required_services = {"caliper.nope.v1"}; m.data_schema = 1;
    const HostCaps c = torch_host();
    CHECK(negotiate(m, c, 5).reason.find("ABI epoch") != std::string::npos);
    m.abi_epoch = 2;
    CHECK(negotiate(m, c, 5).reason.find("Requires host") != std::string::npos);
    m.min_host.clear();
    CHECK(negotiate(m, c, 5).reason.find("UI stack") != std::string::npos);
    m.ui_stack.clear();
    CHECK(negotiate(m, c, 5).reason.find("libtorch") != std::string::npos);
    m.torch.clear();
    CHECK(negotiate(m, c, 5).reason.find("capability") != std::string::npos);
    m.required_services.clear();
    CHECK(negotiate(m, c, 5).reason.find("Stored data") != std::string::npos);
    m.data_schema = 5;
    CHECK(negotiate(m, c, 5).ok);
}

TEST_CASE("host_caps: advertises the epoch, the version and every served id") {
    const HostCaps c = host_caps();
    CHECK(c.abi_epoch == 2);
    CHECK(c.version == "0.6.0");
    CHECK(c.services.count("caliper.ui.v1") == 1);
    CHECK(c.services == host_service_ids());
    CHECK(c.torch_version == host_torch_version());
    CHECK(c.ui_stack_hash == host_ui_stack_hash());
}
