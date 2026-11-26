// SceneView status tests use the CPU recording backend. It does not rasterize.
#include <doctest/doctest.h>

#include "draw/draw_host.h"
#include "draw/draw_runtime_impl.h"
#include "gfx/recording_backend.h"
#include "live/live_scene.h"

#include <cstring>

namespace cd = caliper_draw;
namespace cs = caliper_builder::spec;
namespace gfx = caliper_gfx;

namespace {

cs::Piece status_piece() {
    cs::Piece p;
    p.name = "status_frame";
    p.params = {{"selected_node", cs::ParamType::i32, 0}};
    cs::Resource resource;
    resource.id = "anchors";
    resource.layout.type_name = "Value";
    resource.layout.fields = {{"value", "f32"}};
    resource.count = 1;
    p.resources.push_back(resource);
    resource.id = "colors";
    p.resources.push_back(resource);
    cs::Derive derive;
    derive.passes = {cs::DerivePass{"place", {1, 1, 1}, {"anchors", "colors"}}};
    p.derive = derive;
    cs::Draw draw;
    draw.camera = cs::CameraMode::none;
    cs::DrawPass pass;
    pass.fragment = "status_fs";
    draw.passes = {pass};
    p.draw = draw;
    p.files = {{"derive.wgsl", "@compute @workgroup_size(1) fn place() {}\n"},
               {"draw.wgsl", "@fragment fn status_fs(in: CaliperFullscreen) -> @location(0) vec4<f32> {\n"
                             "    return vec4<f32>(f32(params.selected_node), anchors[0].value, colors[0].value, 1.0);\n}\n"}};
    return p;
}

struct Sink final : caliper_builder::EventSink {
    std::vector<caliper_builder::Event> events;
    std::uint64_t push(caliper_builder::Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

// Count callbacks but retain the real DrawHost resource delivery and status cache.
struct CountedFeed final : cd::DrawFeed {
    cd::DrawFeed& feed;
    unsigned frames = 0, reports = 0;
    explicit CountedFeed(cd::DrawFeed& f) : feed(f) {}
    std::uint64_t revision() const override { return feed.revision(); }
    std::vector<cd::PanelProgram> programs() const override { return feed.programs(); }
    cd::DrawChecks checks() const override { return feed.checks(); }
    void frame(cd::DrawRuntime& rt, cd::FrameInputs* in) override {
        ++frames;
        feed.frame(rt, in);
    }
    void status(const cd::DrawRuntime& rt) override {
        ++reports;
        feed.status(rt);
    }
};

}  // namespace

TEST_CASE("scene status describes the last encoded frame before completion") {
    cd::DrawHost host;
    Sink sink;
    host.set_events(&sink);
    host.set_resolver([](const cs::Spec&, const cs::Program&, cs::Piece* out) {
        *out = status_piece();
        return cd::Verdict{};
    });
    cs::Program program;
    program.id = "status";
    program.panel = "panel";
    cs::PieceRef ref;
    ref.name = "status_frame";
    ref.version = 1;
    ref.from = cs::PieceFrom::workspace;
    program.piece = ref;
    cs::Spec spec;
    spec.programs = {program};
    host.reconcile(spec);

    std::int64_t color_step = 2;
    std::uint64_t color_serial = 1;
    float color = 0.25f;
    // The provider reuses its storage on the next call, as DrawHost::Resources permits.
    float delivery = 0;
    std::vector<std::string> delivered;
    host.set_resources([&](const std::string& id, const std::string& resource, std::int64_t,
                           cd::ResourceVersion* out, std::uint64_t* epoch) {
        CHECK(id == "status");
        delivered.push_back(resource);
        delivery = resource == "anchors" ? 0.5f : color;
        out->resource = resource;
        out->step = resource == "anchors" ? 10 : color_step;
        out->serial = resource == "anchors" ? 1 : color_serial;
        out->count = 1;
        out->bytes = reinterpret_cast<const std::uint8_t*>(&delivery);
        out->size = sizeof(delivery);
        *epoch = 1;
        return true;
    });

    CountedFeed feed(*host.feed("panel"));
    auto caps = gfx::RecordingBackend::default_caps();
    caps.backend = "webgpu";  // Accept WGSL on the CPU recording backend.
    caps.bits |= gfx::kCapTimestamps;
    gfx::RecordingBackend backend(caps);
    backend.set_auto_complete(false);
    caliper_live_views::SceneView view(backend, nullptr);
    std::string why;
    REQUIRE(view.init(&why));
    view.set_draw_feed(&feed);
    REQUIRE_MESSAGE(view.render(16, 16, 1, 0, &why), why);
    REQUIRE(view.draw_runtimes().size() == 1);
    const auto& rt = *view.draw_runtimes().front();
    INFO(cd::draw_status_json(rt.status()).dump());
    REQUIRE(rt.status().state == cd::DrawState::running);
    REQUIRE(rt.status().frames == 1);
    const auto before = host.status().at("status");
    REQUIRE(before.frames == 1);
    const auto versions = rt.internals().versions;
    const auto revision = feed.revision();
    REQUIRE(sink.events.size() == 1);
    CHECK(sink.events.front().fields.at("change") == "compiled");
    const unsigned reports = feed.reports;
    std::int32_t selected = 0;

    SUBCASE("parameter change") {
        selected = 3;
        spec.programs[0].params["selected_node"] = double(selected);
        host.reconcile(spec);
    }
    SUBCASE("non-maximum resource step change") {
        color_step = 3;
        ++color_serial;
        color = 0.75f;
    }
    SUBCASE("frame count without input changes") {}
    SUBCASE("resource replacement at the same step") {
        ++color_serial;
        color = 0.75f;
    }

    CHECK(feed.revision() == revision);
    delivered.clear();
    REQUIRE_MESSAGE(view.render(16, 16, 1, 0, &why), why);
    const auto encoded = rt.status();
    REQUIRE(encoded.frames == 2);
    CHECK(encoded.state == before.state);
    CHECK(encoded.resource_step == before.resource_step);
    CHECK(encoded.frame_ms == before.frame_ms);
    CHECK(encoded.params.at("selected_node") == selected);
    CHECK(encoded.resource_steps.at("colors") == color_step);
    CHECK(delivered == std::vector<std::string>{"anchors", "colors"});
    CHECK(feed.frames == 2);
    CHECK(rt.internals().versions == versions + (color_serial - 1));
    REQUIRE(rt.internals().slots.size() == 2);
    for (const auto& slot : rt.internals().slots) {
        const float expected = slot.id == "anchors" ? 0.5f : color;
        const auto resource = backend.buffer_bytes(slot.buffer);
        REQUIRE(resource.size() >= sizeof(expected));
        CHECK(std::memcmp(resource.data(), &expected, sizeof(expected)) == 0);
    }
    const auto bytes = backend.buffer_bytes(rt.internals().params);
    REQUIRE(bytes.size() >= sizeof(selected));
    CHECK(std::memcmp(bytes.data(), &selected, sizeof(selected)) == 0);
    // Inspect the cache immediately, before another frame or a completion poll.
    const auto cached = cd::draw_status_json(host.status().at("status"));
    CHECK(cached.at("params") == encoded.params);
    CHECK(cached.at("resource_steps") == encoded.resource_steps);
    CHECK(cached.at("frames") == encoded.frames);
    CHECK(feed.reports == reports + 1);
    CHECK(sink.events.size() == 1);

    view.poll();
    CHECK(feed.reports == reports + 1);
    CHECK(sink.events.size() == 1);

    // Synthetic completion updates timing once without another encode or state event.
    backend.set_gpu_ms(0.25);
    backend.finish(backend.submitted().back().serial);
    view.poll();
    CHECK(host.status().at("status").frame_ms == doctest::Approx(0.25));
    CHECK(host.status().at("status").frames == 2);
    CHECK(feed.reports == reports + 2);
    view.poll();
    CHECK(feed.reports == reports + 2);
    CHECK(sink.events.size() == 1);

    // A stop still reaches the cache and emits one event with the last encoded values.
    REQUIRE_MESSAGE(view.render(16, 16, 1, 0, &why), why);
    backend.set_gpu_ms(1500);
    backend.finish(backend.submitted().back().serial);
    view.poll();
    const auto stopped = host.status().at("status");
    CHECK(stopped.state == cd::DrawState::stopped);
    CHECK(stopped.frames == 3);
    CHECK(stopped.params == encoded.params);
    CHECK(stopped.resource_steps == encoded.resource_steps);
    CHECK(feed.reports == reports + 4);
    REQUIRE(sink.events.size() == 2);
    CHECK(sink.events.back().fields.at("change") == "stopped");
    view.poll();
    CHECK(feed.reports == reports + 4);
    REQUIRE_MESSAGE(view.render(16, 16, 1, 0, &why), why);
    CHECK(cd::draw_status_json(host.status().at("status")) == cd::draw_status_json(stopped));
    CHECK(rt.internals().versions == versions + (color_serial - 1));
    CHECK(sink.events.size() == 2);
}
