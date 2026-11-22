// The spec runtime tests of the transport control and the cursor line
// (design/instrument-interface.md, Transport Control): a headless ImGui
// context with the mouse and the keys driven through its input queue, a
// step history with kept steps and a recorder of events. The suite "gpu"
// draws a held transport beside a paused run, and a held series with its
// cursor line, on the offscreen canvas.
#include <doctest/doctest.h>

#include "spec_runtime/spec_transport.h"

#include "spec_runtime/spec_offscreen.h"
#include "export_service.h"
#include "spec_test_canvas.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/format.h>

#include "implot.h"

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using caliper_host::SpecInstrumentHost;
using caliper_host::SpecLook;

namespace {

struct Recorder final : cb::EventSink {
    std::vector<cb::Event> events;
    std::uint64_t push(cb::Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
    std::vector<std::string> changes() const {
        std::vector<std::string> out;
        for (const auto& e : events) out.push_back(e.fields.value("change", ""));
        return out;
    }
};

struct PausedRun final : cb::RunControl {
    cb::RunState run;
    cb::RunState state() override { return run; }
    cb::Error set_param(const std::string&, const nlohmann::json&) override { return {}; }
    cb::Error run_action(const std::string&) override { return {}; }
    cb::Error pause(cb::RunPauseOp, std::int64_t) override { return {}; }
};

const char* kSpec =
    "caliper_spec = 1\n[applet]\nid = \"t\"\n"
    "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n"
    "[[control]]\nid = \"time\"\nkind = \"transport\"\nbind = \"cursor\"\n";

cb::TensorData one(float v) {
    cb::TensorResult r = cb::TensorData::allocate(cb::DType::f32, {1});
    REQUIRE(r.error.ok());
    *reinterpret_cast<float*>(r.tensor.mutable_data()) = v;
    return r.tensor;
}

// The session state the transport moves: a spec, its kept steps 10 to 50
// and the value store.
struct State {
    spec::Spec spec;
    cb::StepHistory history;
    cb::ValueStore values;
    Recorder events;
    State() {
        spec::ParseResult r = spec::parse_toml(kSpec);
        REQUIRE(r.ok());
        spec = r.spec;
        history.configure(spec);
        values.reset(spec);
        for (int step = 10; step <= 50; step += 10) history.append("live:act", step, one(float(step)));
    }
    SpecInstrumentHost host() { return SpecInstrumentHost{&events, nullptr, &values, &history, &spec}; }
};

// A headless ImGui context, one window over the display, and the mouse.
struct Ui {
    ImGuiContext* ctx;
    Ui() {
        ctx = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(800, 200);
        io.DeltaTime = 1.0f / 60.0f;
        frame([] {});
    }
    ~Ui() { ImGui::DestroyContext(ctx); }
    void frame(const std::function<void()>& inside) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(800, 200));
        ImGui::Begin("transport", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        inside();
        ImGui::End();
        ImGui::Render();
    }
    int click(float x, float y, const std::function<bool()>& draw) {
        int changes = 0;
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(x, y);
        frame([&] { changes += draw(); });
        io.AddMouseButtonEvent(0, true);
        frame([&] { changes += draw(); });
        io.AddMouseButtonEvent(0, false);
        frame([&] { changes += draw(); });
        frame([&] { changes += draw(); });
        return changes;
    }
    int key(ImGuiKey k, const std::function<bool()>& handle) {
        int changes = 0;
        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(k, true);
        frame([&] { changes += handle(); });
        io.AddKeyEvent(k, false);
        frame([&] { changes += handle(); });
        return changes;
    }
};

const ImRect kRow(0, 10, 700, 40);   // the buttons are 30 high: "<" is x 0 to 30

// A pixel of the accent's warm hue, also where antialiasing blends a thin
// line or a glyph into the ground: red well above blue. The ground, the
// bright text and the blue series line all fail it.
bool warm(uint32_t p) {
    const int r = int(p & 0xff), b = int((p >> 16) & 0xff);
    return r > 90 && r - b > 40;
}

}  // namespace

TEST_CASE("spec transport: the buttons hold, step, play and go live, and each move adds one cursor event") {
    State s;
    Ui ui;
    SpecInstrumentHost host = s.host();
    const spec::Control& t = s.spec.controls.front();
    auto draw = [&] { return caliper_host::draw_transport_control(t, kRow, SpecLook{}, host); };
    CHECK(s.history.cursor().live);
    // "<" steps back from the live step and holds the view.
    CHECK(ui.click(15, 25, draw) == 1);
    CHECK_FALSE(s.history.cursor().live);
    CHECK(s.history.cursor().step == 40);
    // The play button stands after "<" and a gap: it plays now.
    CHECK(ui.click(50, 25, draw) == 1);
    CHECK(s.history.cursor().playing);
    // A click on it again holds the view.
    CHECK(ui.click(50, 25, draw) == 1);
    CHECK_FALSE(s.history.cursor().playing);
    CHECK(s.events.changes() == std::vector<std::string>{"moved", "playing", "held"});
    for (const auto& e : s.events.events) CHECK(e.origin == "ui");
    // A hold never pauses the run: the transport has no run control here and
    // needs none.
}

TEST_CASE("spec transport: the keys step, go to the first step and go live, unless a text field holds them") {
    State s;
    Ui ui;
    SpecInstrumentHost host = s.host();
    auto keys = [&] { return caliper_host::handle_transport_keys(s.spec, host); };
    CHECK(ui.key(ImGuiKey_LeftArrow, keys) == 1);
    CHECK(s.history.cursor().step == 40);
    CHECK(ui.key(ImGuiKey_Home, keys) == 1);
    CHECK(s.history.cursor().step == 10);
    CHECK(ui.key(ImGuiKey_RightArrow, keys) == 1);
    CHECK(s.history.cursor().step == 20);
    CHECK(ui.key(ImGuiKey_End, keys) == 1);
    CHECK(s.history.cursor().live);
    CHECK(s.history.cursor().step == 50);
    // keys = false turns them off.
    s.spec.controls.front().options.keys = false;
    CHECK(ui.key(ImGuiKey_LeftArrow, keys) == 0);
    CHECK(s.history.cursor().live);
}

TEST_CASE("spec transport: play moves the cursor once a frame at its speed and goes live at the newest step") {
    State s;
    SpecInstrumentHost host = s.host();
    cb::CursorState want = s.history.cursor();
    want.live = false;
    want.step = 10;
    want.playing = true;
    want.speed = 20;
    s.history.set_cursor(want);
    caliper_host::tick_transport(host, 0.5);
    CHECK(s.history.cursor().step == 20);
    caliper_host::tick_transport(host, 2.0);
    CHECK(s.history.cursor().live);
    CHECK(s.events.events.empty());   // play and live moves add no event
}

TEST_SUITE("gpu") {
TEST_CASE("spec transport: a held transport beside a paused run, and a held series with its cursor line") {
    caliper_host::SpecOffscreen* c = spec_test::canvas();
    if (!c) return;
    State s;
    PausedRun run;
    run.run.attached = true;
    run.run.state = "paused";
    run.run.step = 50;
    SpecInstrumentHost host = s.host();
    host.run = &run;
    cb::CursorState want = s.history.cursor();
    want.live = false;
    want.step = 30;
    s.history.set_cursor(want);
    const SpecLook look = c->look();
    const spec::Control& t = s.spec.controls.front();
    const float row = caliper_host::spec_control_height(t, look);
    const int w = 720, h = (int)row;
    const spec_test::Image bar = spec_test::render(w, h, [&](const ImRect& area) {
        caliper_host::draw_spec_control(t, area, look, host);
    });
    REQUIRE(bar.w == w);
    const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP");
    if (dir && *dir)
        caliper_host::export_write_png_atomic(std::string(dir) + "/transport-held-paused.png", bar.px.data(),
                                              (uint32_t)w, (uint32_t)h);
    // The held step is drawn in the accent colour right of the buttons, and the
    // buttons in bright text.
    CHECK(bar.count(warm, w / 3, 0, w, h).n > 20);
    CHECK(bar.count(spec_test::bright).n > 0);

    // A series of the steps 0 to 50 with the cursor held at 30.
    std::vector<double> xs, ys;
    for (int i = 0; i <= 50; ++i) {
        xs.push_back(i);
        ys.push_back(1.0 / (1.0 + 0.1 * i));
    }
    spec::Panel loss;
    loss.id = "loss";
    loss.primitive = spec::Primitive::series;
    const int pw = 480, ph = 300;
    const spec_test::Image plot = spec_test::render(pw, ph, [&](const ImRect& area) {
        ImGui::SetCursorScreenPos(area.Min);
        if (ImPlot::BeginPlot("##loss", area.GetSize(), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus)) {
            ImPlot::SetupAxesLimits(0, 50, 0, 1.1, ImPlotCond_Always);
            ImPlot::PlotLine("loss", xs.data(), ys.data(), (int)xs.size());
            caliper_host::draw_cursor_line(loss, host);
            ImPlot::EndPlot();
        }
    });
    REQUIRE(plot.w == pw);
    if (dir && *dir)
        caliper_host::export_write_png_atomic(std::string(dir) + "/series-cursor-line.png", plot.px.data(),
                                              (uint32_t)pw, (uint32_t)ph);
    // The line is a vertical run of accent pixels three fifths along the plot.
    const auto line = plot.count(warm);
    REQUIRE(line.n > 100);
    CHECK(line.cx > 0.45 * pw);
    CHECK(line.cx < 0.75 * pw);
}
}  // TEST_SUITE("gpu")
