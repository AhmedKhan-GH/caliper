// Tests of the control widgets of the spec runtime (spec_instruments.h,
// design/instruments.md, Control Table): a headless ImGui context with the
// mouse driven through its input queue, a fake run control and a recorder of
// events. A slider bound to param:lr sends the value the person drags to and
// pushes one control event when they let go; a button queues its action or
// pauses the run; a widget without a script or without control sends nothing.
// The suite "gpu" draws a slider row and a button row on the offscreen canvas.
#include "spec_runtime/spec_instruments.h"
#include "spec_runtime/spec_offscreen.h"

#include "export_service.h"

#include "spec_test_canvas.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using caliper_host::SpecInstrumentHost;
using caliper_host::SpecLook;
using nlohmann::json;

namespace {

struct FakeRun final : cb::RunControl {
    cb::RunState run;
    std::vector<std::pair<std::string, json>> sent;   // (op or param, value)
    cb::RunState state() override { return run; }
    cb::Error set_param(const std::string& name, const json& value) override {
        sent.push_back({"set " + name, value});
        return {};
    }
    cb::Error run_action(const std::string& name) override {
        sent.push_back({"run " + name, json()});
        return {};
    }
    cb::Error pause(cb::RunPauseOp op, std::int64_t) override {
        sent.push_back({op == cb::RunPauseOp::pause ? "pause" : "resume", json()});
        return {};
    }
    bool supported = false;
    std::vector<cb::InterventionPoint> pts;
    std::vector<cb::Intervention> ivs;
    bool interventions_supported() override { return supported; }
    std::vector<cb::InterventionPoint> intervention_points() override { return pts; }
    std::vector<cb::Intervention> interventions() override { return ivs; }
    cb::Error intervene_set(const cb::Intervention& i) override {
        sent.push_back({"ablate " + i.point, i.op});
        return {};
    }
    cb::Error intervene_clear(const std::string& point) override {
        sent.push_back({"clear " + point, json()});
        return {};
    }
};

struct Recorder final : cb::EventSink {
    std::vector<cb::Event> events;
    std::uint64_t push(cb::Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

cb::RunState training() {
    cb::RunState s;
    s.attached = true;
    s.script = "train.py";
    s.control = true;
    s.state = "running";
    s.step = 3;
    cb::RunParam lr;
    lr.name = "lr";
    lr.type = "number";
    lr.value = 0.01;
    lr.min = 1e-5;
    lr.max = 0.1;
    lr.log = true;
    s.params = {lr};
    cb::RunAction save;
    save.name = "save";
    s.actions = {save};
    return s;
}

spec::Control make(const std::string& id, spec::ControlKind kind, const std::string& bind) {
    spec::Control c;
    c.id = id;
    c.kind = kind;
    c.bind = bind;
    return c;
}

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
        // A new window is hidden in its first frame and takes no mouse in the
        // next one, so the window is drawn once before any click.
        frame([] {});
    }
    ~Ui() { ImGui::DestroyContext(ctx); }
    // One frame that draws `inside` in a window over the display.
    void frame(const std::function<void()>& inside) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(800, 200));
        ImGui::Begin("controls", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        inside();
        ImGui::End();
        ImGui::Render();
    }
    // A click at (x, y): the move, the press and the release in three frames;
    // returns how many frames reported a change.
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
};

const ImRect kRow(0, 0, 400, 30);

}  // namespace

TEST_CASE("spec controls: a control leaf takes a row of the look's height") {
    spec::Control slider = make("lr", spec::ControlKind::slider, "param:lr");
    const SpecLook look;
    CHECK(caliper_host::spec_control_height(slider, look) == 30.0f);
    SpecLook big;
    big.text_scale = 2.0f;
    CHECK(caliper_host::spec_control_height(slider, big) == 60.0f);
}

TEST_CASE("spec controls: a slider bound to param:lr sends the dragged value and pushes one control event") {
    Ui ui;
    FakeRun run;
    run.run = training();
    Recorder rec;
    SpecInstrumentHost host{&rec, &run};
    const SpecLook look;
    spec::Control lr = make("lr", spec::ControlKind::slider, "param:lr");
    lr.title = "Learning rate";
    auto draw = [&] { return caliper_host::draw_spec_control(lr, kRow, look, host); };
    ui.frame([&] { CHECK_FALSE(draw()); });
    CHECK(run.sent.empty());

    // A click near the right end sets a value near the parameter's maximum.
    CHECK(ui.click(390, 15, draw) >= 1);
    REQUIRE_FALSE(run.sent.empty());
    CHECK(run.sent.back().first == "set lr");
    const double v = run.sent.back().second.get<double>();
    CHECK(v > 0.05);
    CHECK(v <= 0.1);
    REQUIRE(rec.events.size() == 1);
    const cb::Event& e = rec.events[0];
    CHECK(e.kind == cb::EventKind::control);
    CHECK(e.panel == "lr");
    CHECK(e.origin == "ui");
    CHECK(e.fields == json{{"control", "lr"}, {"bind", "param:lr"}, {"value", v}});

    // The options narrow the range.
    lr.options.max = 0.02;
    rec.events.clear();
    ui.click(390, 15, draw);
    REQUIRE_FALSE(run.sent.empty());
    CHECK(run.sent.back().second.get<double>() <= 0.02);
    CHECK(rec.events.size() == 1);
}

TEST_CASE("spec controls: a slider clamps the script's own range and value to finite numbers it can send") {
    Ui ui;
    FakeRun run;
    run.run = training();
    cb::RunParam& p = run.run.params[0];
    p.type = "integer";
    p.min = -1e300;
    p.max = 1e300;
    p.log = false;
    p.value = 5;
    SpecInstrumentHost host{nullptr, &run};
    const SpecLook look;
    spec::Control lr = make("huge", spec::ControlKind::slider, "param:lr");
    auto draw = [&] { return caliper_host::draw_spec_control(lr, kRow, look, host); };
    ui.frame([&] { draw(); });
    CHECK(ui.click(390, 15, draw) >= 1);
    REQUIRE_FALSE(run.sent.empty());
    REQUIRE(run.sent.back().second.is_number_integer());
    CHECK(run.sent.back().second.get<std::int64_t>() > 0);
    CHECK(run.sent.back().second.get<std::int64_t>() <= 1000000000000000);

    // A value that is not a number, and no range: the slider sends finite numbers.
    p.type = "number";
    p.min.reset();
    p.max.reset();
    p.value = std::numeric_limits<double>::quiet_NaN();
    run.sent.clear();
    lr.id = "nan";
    ui.click(390, 15, draw);
    REQUIRE_FALSE(run.sent.empty());
    CHECK(std::isfinite(run.sent.back().second.get<double>()));
}

TEST_CASE("spec controls: a slider without a script, without control or without its parameter sends nothing") {
    Ui ui;
    FakeRun run;
    Recorder rec;
    const SpecLook look;
    spec::Control lr = make("lr", spec::ControlKind::slider, "param:lr");
    SpecInstrumentHost none{&rec, nullptr};
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(lr, kRow, look, none); });
    SpecInstrumentHost host{&rec, &run};
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(lr, kRow, look, host); });   // not attached
    run.run = training();
    run.run.control = false;
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(lr, kRow, look, host); });
    run.run.control = true;
    spec::Control momentum = make("momentum", spec::ControlKind::slider, "param:momentum");
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(momentum, kRow, look, host); });
    CHECK(run.sent.empty());
    CHECK(rec.events.empty());
}

TEST_CASE("spec controls: a slider bound to value holds its value and pushes it") {
    Ui ui;
    Recorder rec;
    SpecInstrumentHost host{&rec, nullptr};
    const SpecLook look;
    spec::Control w = make("weight", spec::ControlKind::slider, "value");
    w.options.min = 0.0;
    w.options.max = 10.0;
    w.options.step = 1.0;
    w.options.value = 2.0;
    auto draw = [&] { return caliper_host::draw_spec_control(w, kRow, look, host); };
    ui.click(390, 15, draw);
    REQUIRE(rec.events.size() == 1);
    const double v = rec.events[0].fields["value"].get<double>();
    CHECK(v >= 9.0);
    CHECK(v == std::round(v));   // on its step
}

TEST_CASE("spec controls: a button queues its action, and run:pause pauses a run and resumes a paused one") {
    Ui ui;
    FakeRun run;
    run.run = training();
    Recorder rec;
    SpecInstrumentHost host{&rec, &run};
    const SpecLook look;
    spec::Control save = make("save", spec::ControlKind::button, "action:save");
    CHECK(ui.click(200, 15, [&] { return caliper_host::draw_spec_control(save, kRow, look, host); }) == 1);
    REQUIRE(run.sent.size() == 1);
    CHECK(run.sent[0].first == "run save");
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].fields == json{{"control", "save"}, {"bind", "action:save"}, {"value", true}});

    spec::Control pause = make("pause", spec::ControlKind::button, "run:pause");
    auto draw = [&] { return caliper_host::draw_spec_control(pause, kRow, look, host); };
    ui.click(200, 15, draw);
    run.run.state = "paused";
    ui.click(200, 15, draw);
    REQUIRE(run.sent.size() == 3);
    CHECK(run.sent[1].first == "pause");
    CHECK(run.sent[2].first == "resume");
    CHECK(rec.events.back().fields["value"] == "resume");

    // Before the first step there is nothing to pause; an unknown action is disabled.
    run.run.state = "none";
    run.run.step.reset();
    ui.click(200, 15, draw);
    spec::Control eval = make("eval", spec::ControlKind::button, "action:evaluate");
    ui.click(200, 15, [&] { return caliper_host::draw_spec_control(eval, kRow, look, host); });
    CHECK(run.sent.size() == 3);
}

TEST_CASE("spec controls: a button bound to an action shows the action it runs") {
    cb::RunState st = training();
    spec::Control wipe = make("wipe", spec::ControlKind::button, "action:save");
    wipe.title = "Evaluate";
    CHECK(caliper_host::spec_button_label(wipe, st) == "Evaluate (action:save)");
    st.actions[0].queued = true;
    CHECK(caliper_host::spec_button_label(wipe, st) == "Evaluate (action:save)  \xC2\xB7 queued");
    CHECK(caliper_host::spec_button_label(make("save", spec::ControlKind::button, "action:save"), cb::RunState{}) ==
          "save (action:save)");
    spec::Control pause = make("pause", spec::ControlKind::button, "run:pause");
    CHECK(caliper_host::spec_button_label(pause, st) == "Pause");
    st.state = "paused";
    CHECK(caliper_host::spec_button_label(pause, st) == "Resume");
}

// ---- drawn on the offscreen canvas (suite "gpu": caliper_spec_runtime_gpu) ----------------

TEST_SUITE("gpu") {
TEST_CASE("spec controls: a slider row and a button row in the house look") {
    caliper_host::SpecOffscreen* c = spec_test::canvas();
    if (!c) return;
    FakeRun run;
    run.run = training();
    Recorder rec;
    SpecInstrumentHost host{&rec, &run};
    spec::Control lr = make("lr", spec::ControlKind::slider, "param:lr");
    lr.title = "Learning rate";
    spec::Control save = make("save", spec::ControlKind::button, "action:save");
    save.title = "Save checkpoint";
    const SpecLook look = c->look();
    const float row = caliper_host::spec_control_height(lr, look);
    const int w = 480, h = (int)(2 * row);
    const spec_test::Image img = spec_test::render(w, h, [&](const ImRect& area) {
        caliper_host::draw_spec_control(lr, ImRect(area.Min, ImVec2(area.Max.x, area.Min.y + row)), look, host);
        caliper_host::draw_spec_control(save, ImRect(ImVec2(area.Min.x, area.Min.y + row), area.Max), look, host);
    });
    REQUIRE(img.w == w);
    if (const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP"); dir && *dir)
        caliper_host::export_write_png_atomic(std::string(dir) + "/controls-slider-button.png", img.px.data(),
                                              (uint32_t)w, (uint32_t)h);
    // The grab of lr = 0.01 on its log range 1e-5 to 0.1 stands three
    // quarters along the slider, which takes the right part of the row.
    const auto grab = img.count(spec_test::near(look.accent, 8), 0, 0, w, (int)row);
    REQUIRE(grab.n > 0);
    CHECK(grab.cx > 0.6 * w);
    // The title and the button's label are drawn: bright text in both rows.
    CHECK(img.count(spec_test::bright, 0, 0, w / 3, (int)row).n > 0);
    CHECK(img.count(spec_test::bright, 0, (int)row, w, h).n > 0);
    CHECK(run.sent.empty());   // drawing alone sends nothing
}
}  // TEST_SUITE("gpu")

TEST_CASE("spec controls: a toggle bound to intervene:<point> ablates the point while it is on") {
    Ui ui;
    FakeRun run;
    run.run = training();
    run.supported = true;
    cb::InterventionPoint attn;
    attn.point = attn.module = "blocks.0.attn";
    run.pts = {attn};
    Recorder rec;
    SpecInstrumentHost host{&rec, &run};
    const SpecLook look;
    spec::Control t = make("ablate", spec::ControlKind::toggle, "intervene:blocks.0.attn");
    auto draw = [&] { return caliper_host::draw_spec_control(t, kRow, look, host); };
    CHECK(ui.click(390, 15, draw) == 1);
    REQUIRE(run.sent.size() == 1);
    CHECK(run.sent[0] == std::pair<std::string, json>{"ablate blocks.0.attn", "ablate"});
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].kind == cb::EventKind::control);
    CHECK(rec.events[0].origin == "ui");
    CHECK(rec.events[0].fields == json{{"control", "ablate"}, {"bind", "intervene:blocks.0.attn"}, {"value", true}});

    // The script applies it; the next click clears it.
    cb::Intervention iv;
    iv.point = "blocks.0.attn";
    iv.since_step = 4;
    iv.state = "active";
    run.ivs = {iv};
    CHECK(ui.click(390, 15, draw) == 1);
    REQUIRE(run.sent.size() == 2);
    CHECK(run.sent[1].first == "clear blocks.0.attn");
    CHECK(rec.events.back().fields["value"] == false);
    run.ivs.clear();

    // A script without interventions, an unknown point and no script send nothing.
    run.supported = false;
    ui.click(390, 15, draw);
    run.supported = true;
    spec::Control other = make("other", spec::ControlKind::toggle, "intervene:blocks.9");
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(other, kRow, look, host); });
    run.run.attached = false;
    ui.click(390, 15, draw);
    CHECK(run.sent.size() == 2);
}

TEST_CASE("spec controls: a toggle bound to a boolean parameter sets it") {
    Ui ui;
    FakeRun run;
    run.run = training();
    cb::RunParam aug;
    aug.name = "augment";
    aug.type = "boolean";
    aug.value = false;
    run.run.params.push_back(aug);
    Recorder rec;
    SpecInstrumentHost host{&rec, &run};
    const SpecLook look;
    spec::Control t = make("aug", spec::ControlKind::toggle, "param:augment");
    CHECK(ui.click(390, 15, [&] { return caliper_host::draw_spec_control(t, kRow, look, host); }) == 1);
    REQUIRE(run.sent.size() == 1);
    CHECK(run.sent[0] == std::pair<std::string, json>{"set augment", true});
    // A parameter that is not boolean takes no toggle.
    spec::Control lr = make("lr", spec::ControlKind::toggle, "param:lr");
    ui.click(390, 15, [&] { return caliper_host::draw_spec_control(lr, kRow, look, host); });
    CHECK(run.sent.size() == 1);
}
