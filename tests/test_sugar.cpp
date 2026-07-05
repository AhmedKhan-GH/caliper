#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <stdexcept>
#include <string>

namespace {
bool g_throw_in_frame = false;

class TinyApplet final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        host.log_info("tiny.on_init");
        return true;
    }
    void on_frame(const caliper::Frame& f) override {
        if (g_throw_in_frame) throw std::runtime_error("boom");
        last_w_ = f.fb_width;
    }
    void on_cleanup() override { if (host_) host_->log_info("tiny.on_cleanup"); }
    int last_w_ = 0;
private:
    caliper::Host* host_ = nullptr;
};
} // namespace

CALIPER_APPLET(TinyApplet,
    .id       = "dev.caliper.tiny",
    .version  = "0.1.0",
    .name     = "Tiny",
    .summary  = "sugar test applet",
    .tag      = "Test",
    .services = {CALIPER_LOG_V1})

TEST_CASE("sugar: macro-generated descriptor matches meta") {
    const CaliperAppletDescriptor* d = caliper_applet_descriptor();
    REQUIRE(d != nullptr);
    CHECK(d->struct_size == sizeof(CaliperAppletDescriptor));
    CHECK(d->abi_epoch == CALIPER_ABI_EPOCH);
    CHECK(std::string(d->id) == "dev.caliper.tiny");
    CHECK(std::string(d->version) == "0.1.0");
    CHECK(std::string(d->name) == "Tiny");
    CHECK(std::string(d->tag) == "Test");
    REQUIRE(d->required_services != nullptr);
    CHECK(std::string(d->required_services[0]) == "caliper.log.v1");
    CHECK(d->required_services[1] == nullptr);          // NULL-terminated
    CHECK(d->api.struct_size == sizeof(CaliperAppletAPI));
    REQUIRE(d->api.create); REQUIRE(d->api.destroy); REQUIRE(d->api.initialize);
    REQUIRE(d->api.frame);  REQUIRE(d->api.cleanup);
}

TEST_CASE("sugar: lifecycle bridges to the class through the C table") {
    caliper::testing::FixtureHost fx;
    const auto* d = caliper_applet_descriptor();
    void* self = d->api.create();
    REQUIRE(self != nullptr);
    CHECK(d->api.initialize(self, fx.host()));
    CHECK(fx.log_contains("tiny.on_init"));

    CaliperFrameInfo fi{};
    fi.struct_size = sizeof fi; fi.fb_width = 640; fi.fb_height = 480;
    fi.dpi_scale = 2.0f;
    d->api.frame(self, &fi);

    d->api.cleanup(self);
    CHECK(fx.log_contains("tiny.on_cleanup"));
    d->api.destroy(self);
}

TEST_CASE("sugar: exceptions never cross the C boundary") {
    caliper::testing::FixtureHost fx;
    const auto* d = caliper_applet_descriptor();
    void* self = d->api.create();
    REQUIRE(d->api.initialize(self, fx.host()));
    g_throw_in_frame = true;
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    d->api.frame(self, &fi);                    // must not terminate/propagate
    g_throw_in_frame = false;
    CHECK(fx.log_contains("unhandled exception in on_frame"));
    d->api.cleanup(self);
    d->api.destroy(self);
}
