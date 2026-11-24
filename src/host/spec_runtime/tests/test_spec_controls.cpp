// Tests of the control widgets of the spec runtime (spec_instruments.h,
// design/instruments.md, Control Table): a headless ImGui context with the
// mouse driven through its input queue, a fake run control and a recorder of
// events. A slider bound to param:lr sends the value the person drags to and
// pushes one control event when they let go; a button queues its action or
// pauses the run; a widget without a script or without control sends nothing.
// The suite "gpu" draws a slider row and a button row on the offscreen canvas.
//
// The kinds of design/instrument-interface.md: a range drags its grabs and
// keeps its value in the session's values, a segmented group chooses on a
// click and on its key while no text field holds the keyboard, a text field
// applies on Return, a picker sets a choice from its menu, a toggle bound to a
// state holds a boolean, a pick writes its state, and the app's single keys
// are reserved shortcuts. The suite "gpu" draws one row of each new kind.
#include "spec_runtime/spec_instruments.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_shortcuts.h"

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/validate.h>

#include "export_service.h"

#include "spec_test_canvas.h"

#include <doctest/doctest.h>

#include <imgui_internal.h>   // the tooltip window of a hover

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
    save.title = "Save Snapshot";
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

TEST_CASE("spec controls: action buttons show plain labels and their current status") {
    cb::RunState st = training();
    st.actions[0].name = "apply_clamp";
    spec::Control clamp = make("clamp", spec::ControlKind::button, "action:apply_clamp");
    clamp.title = "Apply Clamp";
    CHECK(caliper_host::spec_button_label(clamp, st) == "Apply Clamp");
    st.actions[0].queued = true;
    CHECK(caliper_host::spec_button_label(clamp, st) == "Apply Clamp  \xC2\xB7 queued");
    st.actions[0].running = true;
    CHECK(caliper_host::spec_button_label(clamp, st) == "Apply Clamp  \xC2\xB7 running");
    st.actions[0].queued = false;
    st.actions[0].running = false;
    CHECK(caliper_host::spec_button_label(clamp, st) == "Apply Clamp");
    st.actions.clear();
    CHECK(caliper_host::spec_button_label(clamp, st) == "Apply Clamp");
    CHECK(caliper_host::spec_button_label(clamp, cb::RunState{}) == "Apply Clamp");
    CHECK(caliper_host::spec_button_label(make("save", spec::ControlKind::button, "action:save"), cb::RunState{}) ==
          "save");
    spec::Control pause = make("pause", spec::ControlKind::button, "run:pause");
    CHECK(caliper_host::spec_button_label(pause, st) == "Pause");
    st.state = "paused";
    CHECK(caliper_host::spec_button_label(pause, st) == "Resume");
}

TEST_CASE("spec controls: drawn button text excludes binding identifiers") {
    Ui ui;
    FakeRun run;
    run.run = training();
    run.run.actions[0].name = "apply_clamp";
    SpecInstrumentHost host{nullptr, &run};
    const SpecLook look;
    spec::Control clamp = make("clamp", spec::ControlKind::button, "action:apply_clamp");
    clamp.title = "Apply Clamp";
    const auto shown = [&] {
        std::string text;
        ui.frame([&] {
            ImGui::LogToBuffer();
            caliper_host::draw_spec_control(clamp, kRow, look, host);
            text = ui.ctx->LogBuffer.c_str();
            ImGui::LogFinish();
        });
        CHECK(text.find("Apply Clamp") != std::string::npos);
        CHECK(text.find("action:") == std::string::npos);
        CHECK(text.find("apply_clamp") == std::string::npos);
        return text;
    };
    shown();
    run.run.actions[0].queued = true;
    CHECK(shown().find("queued") != std::string::npos);
    run.run.actions[0].running = true;
    CHECK(shown().find("running") != std::string::npos);
    run.run.control = false;
    shown();
    run.run.attached = false;
    CHECK(shown().find("running") == std::string::npos);
    CHECK(run.sent.empty());
}

// ---- drawn on the offscreen canvas (suite "gpu": caliper_spec_runtime_gpu) ----------------

TEST_CASE("spec controls: a readout and a title wider than their part of the row end in an ellipsis and show whole on a hover") {
    Ui ui;
    Recorder rec;
    SpecInstrumentHost host{&rec, nullptr};
    const SpecLook look;
    spec::Control now = make("now", spec::ControlKind::readout, "");
    now.title = "Now";
    spec::BoundText text;
    text.format = "Step 1,024 of a long run, with a readout text that does not fit a row of 400 pixels";
    now.options.bound_text = text;
    // The tooltip window of this frame, which ImGui names ##Tooltip_00.
    const auto tooltip = [] {
        ImGuiWindow* w = ImGui::FindWindowByName("##Tooltip_00");
        return w != nullptr && w->Active;
    };
    const auto hover = [&](float x) {
        bool shown = false;
        ImGui::GetIO().AddMousePosEvent(x, 15);
        ui.frame([&] { caliper_host::draw_spec_control(now, kRow, look, host); });
        ui.frame([&] {
            caliper_host::draw_spec_control(now, kRow, look, host);
            shown = tooltip();
        });
        return shown;
    };
    CHECK(hover(300));
    // A readout that fits needs no tooltip, nor does a title that fits.
    now.options.bound_text->format = "Step 1,024";
    CHECK_FALSE(hover(300));
    CHECK_FALSE(hover(12));
    // A title cut at its share of the row shows whole on a hover over it.
    now.title = "A title much longer than a third of the row";
    CHECK(hover(12));
    CHECK(rec.events.empty());
}

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

// ---- the kinds that hold a value (design/instrument-interface.md, Controls) ----------------

namespace {

// A spec with the states measure (text), span (a range) and flag (a boolean)
// and the given controls, as the session's values read it.
spec::Spec instrument_spec(std::vector<spec::Control> controls) {
    spec::Spec s;
    s.version = 1;
    s.applet.id = "bench";
    spec::StateVar measure;
    measure.id = "measure";
    measure.type = spec::StateType::text;
    measure.choices = {"norm", "mean"};
    spec::StateVar span;
    span.id = "span";
    span.type = spec::StateType::range;
    span.min = 0.0;
    span.max = 100.0;
    span.default_value = json::array({20.0, 60.0});
    spec::StateVar flag;
    flag.id = "flag";
    flag.type = spec::StateType::boolean;
    spec::StateVar token;
    token.id = "token";
    token.type = spec::StateType::integer;
    token.min = 0.0;
    token.max = 63.0;
    s.states = {measure, span, flag, token};
    s.controls = std::move(controls);
    return s;
}

spec::Choice choice(const std::string& value, const std::string& label, const std::string& key = {}) {
    spec::Choice c;
    c.value = value;
    c.label = label;
    if (!key.empty()) c.key = key;
    c.table = true;
    return c;
}

// A drag from (x0, y) to (x1, y): the press, two moves and the release.
void drag(Ui& ui, float x0, float x1, float y, const std::function<bool()>& draw) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(x0, y);
    ui.frame([&] { draw(); });
    io.AddMouseButtonEvent(0, true);
    ui.frame([&] { draw(); });
    io.AddMousePosEvent((x0 + x1) * 0.5f, y);
    ui.frame([&] { draw(); });
    io.AddMousePosEvent(x1, y);
    ui.frame([&] { draw(); });
    io.AddMouseButtonEvent(0, false);
    ui.frame([&] { draw(); });
    ui.frame([&] { draw(); });
}

// A key pressed and released, with the frame of each drawing `draw`.
void press(Ui& ui, ImGuiKey key, const std::function<void()>& draw) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(key, true);
    ui.frame(draw);
    io.AddKeyEvent(key, false);
    ui.frame(draw);
}

std::vector<cb::Event> of_kind(const Recorder& rec, cb::EventKind kind) {
    std::vector<cb::Event> out;
    for (const auto& e : rec.events)
        if (e.kind == kind) out.push_back(e);
    return out;
}

}  // namespace

TEST_CASE("spec controls: a range drags its nearer grab, keeps the session's value and pushes one event") {
    Ui ui;
    Recorder rec;
    spec::Control win = make("win", spec::ControlKind::range, "value");
    win.options.min = 0.0;
    win.options.max = 100.0;
    win.options.step = 1.0;
    win.options.value_other = json::array({20.0, 60.0});
    spec::Control span = make("span", spec::ControlKind::range, "state:span");
    const spec::Spec sp = instrument_spec({win, span});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look;
    // The high grab is nearer the right end of the track; a drag there takes it to the top.
    drag(ui, 380, 399, 15, [&] { return caliper_host::draw_spec_control(win, kRow, look, host); });
    const json v = values.control_value("win");
    REQUIRE(v.is_array());
    CHECK(v[0].get<double>() == 20.0);
    CHECK(v[1].get<double>() >= 95.0);
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].kind == cb::EventKind::control);
    CHECK(rec.events[0].fields["value"] == v);

    // Bound to a state: one control event and one state event, whose previous
    // value is the value before the drag.
    rec.events.clear();
    drag(ui, 380, 399, 15, [&] { return caliper_host::draw_spec_control(span, kRow, look, host); });
    const auto states = of_kind(rec, cb::EventKind::state);
    REQUIRE(states.size() == 1);
    CHECK(states[0].fields["previous"] == json::array({20.0, 60.0}));
    CHECK(states[0].fields["writer"] == "span");
    CHECK(of_kind(rec, cb::EventKind::control).size() == 1);
}

TEST_CASE("spec controls: a segmented group chooses on a click, and on its key while no text field holds the keyboard") {
    Ui ui;
    Recorder rec;
    spec::Control group = make("measure", spec::ControlKind::segmented, "state:measure");
    group.options.choices = {choice("norm", "Norm", "n"), choice("mean", "Mean", "m")};
    spec::Control note = make("note", spec::ControlKind::text, "value");
    const spec::Spec sp = instrument_spec({group, note});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look;
    const ImRect second(0, 40, 400, 70);
    auto draw = [&] {
        caliper_host::handle_instrument_keys(sp, host);
        caliper_host::draw_spec_control(group, kRow, look, host);
        return caliper_host::draw_spec_control(note, second, look, host);
    };
    // A click on the right button chooses Mean.
    ui.click(390, 15, [&] { draw(); return false; });
    CHECK(values.control_value("measure") == "mean");
    REQUIRE(of_kind(rec, cb::EventKind::state).size() == 1);
    CHECK(of_kind(rec, cb::EventKind::state)[0].fields["writer"] == "measure");
    // The key n chooses Norm; its control event carries the key.
    rec.events.clear();
    press(ui, ImGuiKey_N, [&] { draw(); });
    CHECK(values.control_value("measure") == "norm");
    const auto controls = of_kind(rec, cb::EventKind::control);
    REQUIRE(controls.size() == 1);
    CHECK(controls[0].fields["key"] == "n");
    // While the text field holds the keyboard, m types into it and leaves the group.
    ui.click(300, 55, [&] { draw(); return false; });
    REQUIRE(ImGui::GetIO().WantTextInput);
    press(ui, ImGuiKey_M, [&] { draw(); });
    CHECK(values.control_value("measure") == "norm");
}

TEST_CASE("spec controls: a group with multiple turns each choice on and off") {
    Ui ui;
    Recorder rec;
    spec::Control layers = make("layers", spec::ControlKind::segmented, "value");
    layers.options.multiple = true;
    layers.options.choices = {choice("a", "A"), choice("b", "B"), choice("c", "C")};
    const spec::Spec sp = instrument_spec({layers});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(layers, kRow, look, host); };
    CHECK(values.control_value("layers") == json::array());
    ui.click(385, 15, draw);   // C
    ui.click(100, 15, draw);   // A, the first third of the widget
    const json v = values.control_value("layers");
    INFO(v.dump() << " after " << rec.events.size() << " events");
    REQUIRE(v.is_array());
    CHECK(v.size() == 2);
    CHECK(v.back() == "c");   // in the order of the choices
    ui.click(385, 15, draw);   // C off
    CHECK(values.control_value("layers").size() == 1);
    CHECK(of_kind(rec, cb::EventKind::control).size() == 3);
}

TEST_CASE("spec controls: a text field applies its text on Return, cut at max_length") {
    Ui ui;
    Recorder rec;
    spec::Control note = make("note", spec::ControlKind::text, "value");
    note.options.max_length = 5;
    note.options.placeholder = "Label";
    const spec::Spec sp = instrument_spec({note});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(note, kRow, look, host); };
    ui.click(300, 15, draw);
    ImGui::GetIO().AddInputCharactersUTF8("layer seven");
    ui.frame([&] { draw(); });
    CHECK(values.control_value("note") == "");   // nothing applies while the person types
    press(ui, ImGuiKey_Enter, [&] { draw(); });
    CHECK(values.control_value("note") == "layer");
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].fields["value"] == "layer");
}

TEST_CASE("spec controls: a picker sets a choice from its menu, and a toggle bound to a state holds a boolean") {
    Ui ui;
    Recorder rec;
    spec::Control pick = make("pick", spec::ControlKind::picker, "state:measure");
    pick.options.choices = {choice("norm", "Norm"), choice("mean", "Mean")};
    spec::Control flag = make("flag", spec::ControlKind::toggle, "state:flag");
    const spec::Spec sp = instrument_spec({pick, flag});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(pick, kRow, look, host); };
    ui.click(300, 15, draw);   // opens the menu below the row
    ui.click(300, 57, draw);   // the second item
    CHECK(values.control_value("pick") == "mean");
    CHECK(of_kind(rec, cb::EventKind::state).size() == 1);
    // The toggle: a click turns the boolean state on.
    rec.events.clear();
    ui.click(200, 15, [&] { return caliper_host::draw_spec_control(flag, kRow, look, host); });
    CHECK(values.control_value("flag") == true);
    CHECK(of_kind(rec, cb::EventKind::state).size() == 1);
}

TEST_CASE("spec controls: a pick records its fields in the session and writes the panel's state") {
    spec::Spec sp = instrument_spec({});
    spec::Panel world;
    world.id = "world";
    world.primitive = spec::Primitive::scene;
    world.options.select = spec::StateWrite{"token", "column", std::string("terrain")};
    sp.panels.push_back(world);
    cb::ValueStore values;
    values.reset(sp);
    Recorder rec;
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    cb::Event pick;
    pick.kind = cb::EventKind::pick;
    pick.panel = "world";
    pick.fields = json{{"entity", "terrain"}, {"fields", {{"entity", "terrain"}, {"row", 3}, {"column", 7}}}};
    caliper_host::record_spec_pick("world", pick, host);
    spec::Reference token;
    token.target = "token";
    CHECK(values.value(token) == 7);
    REQUIRE(values.last_pick("world"));
    CHECK(values.last_pick("world")->row == std::optional<std::int64_t>(3));
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].kind == cb::EventKind::state);
    // A pick on another entity writes nothing.
    pick.fields = json{{"entity", "loss"}, {"index", 2}};
    caliper_host::record_spec_pick("world", pick, host);
    CHECK(values.value(token) == 7);
}

TEST_CASE("spec controls: the single keys of the app's views are reserved shortcuts") {
    // The live views use F (fit the camera) and U (turn the up axis) without a
    // modifier (src/host/live/live_views_draw.cpp); the transport takes Space,
    // the arrows, Home and End, which the grammar of a shortcut never reaches.
    const std::string head = "caliper_spec = 1\n[applet]\nid = \"a\"\n[[state]]\nid = \"m\"\ntype = \"text\"\n"
                             "choices = [\"x\", \"y\"]\n[[control]]\nid = \"g\"\nkind = \"segmented\"\n"
                             "bind = \"state:m\"\noptions = { choices = [{ value = \"x\", key = \"";
    for (const char* key : {"f", "u", "shift+f", "shift+u"}) {
        CAPTURE(key);
        const auto report = cb::spec::check_toml(head + key + "\" }, { value = \"y\", key = \"y\" }] }\n");
        bool reserved = false;
        for (const auto& d : report.diagnostics) reserved = reserved || d.code == "spec.reserved_shortcut";
        CHECK(reserved);
    }
    const auto free = cb::spec::check_toml(head + "n\" }, { value = \"y\", key = \"m\" }] }\n");
    for (const auto& d : free.diagnostics) CHECK(d.code != "spec.reserved_shortcut");
}

TEST_SUITE("gpu") {
TEST_CASE("spec controls: a range, a picker, a text field, a segmented group and a readout in the house look") {
    caliper_host::SpecOffscreen* c = spec_test::canvas();
    if (!c) return;
    Recorder rec;
    spec::Control win = make("window", spec::ControlKind::range, "value");
    win.title = "Steps";
    win.options.min = 0.0;
    win.options.max = 3000.0;
    win.options.value_other = json::array({952.0, 3000.0});
    spec::Control cmap = make("cmap", spec::ControlKind::picker, "value");
    cmap.title = "Colormap";
    cmap.options.choices = {choice("viridis", "Viridis"), choice("magma", "Magma")};
    spec::Control note = make("note", spec::ControlKind::text, "value");
    note.title = "Note";
    note.options.placeholder = "A label for this run";
    spec::Control measure = make("measure", spec::ControlKind::segmented, "state:measure");
    measure.title = "Measure";
    measure.options.choices = {choice("norm", "Norm", "n"), choice("mean", "Mean", "m")};
    spec::Control now = make("now", spec::ControlKind::readout, "");
    now.title = "Now";
    spec::BoundText text;
    text.format = "Step {step:,d}  Token {token:d}";
    spec::Reference step;
    step.target = "cursor";
    step.field = "step";
    spec::Reference token;
    token.target = "token";
    text.fields = {{"step", step}, {"token", token}};
    now.options.bound_text = text;
    const spec::Spec sp = instrument_spec({win, cmap, note, measure, now});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look = c->look();
    const float row = caliper_host::spec_control_height(win, look);
    const std::vector<const spec::Control*> rows = {&win, &cmap, &note, &measure, &now};
    const int w = 560, h = (int)(row * rows.size());
    const spec_test::Image img = spec_test::render(w, h, [&](const ImRect& area) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const float y = area.Min.y + row * (float)i;
            caliper_host::draw_spec_control(*rows[i], ImRect(ImVec2(area.Min.x, y), ImVec2(area.Max.x, y + row)), look,
                                            host);
        }
    });
    REQUIRE(img.w == w);
    if (const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP"); dir && *dir)
        caliper_host::export_write_png_atomic(std::string(dir) + "/controls-instrument.png", img.px.data(),
                                              (uint32_t)w, (uint32_t)h);
    const int r = (int)row;
    // The range's span [952, 3000] fills the right two thirds of its track in the accent colour.
    const auto span = img.count(spec_test::near(look.accent, 8), 0, 0, w, r);
    REQUIRE(span.n > 0);
    CHECK(span.cx > 0.55 * w);
    // The segmented group's first button (Norm, on) is filled with the accent colour, the second is not.
    const auto on = img.count(spec_test::near(look.accent, 8), 0, 3 * r, w, 4 * r);
    REQUIRE(on.n > 0);
    CHECK(on.cx < 0.75 * w);
    // Each row draws its title in bright text.
    for (int i = 0; i < (int)rows.size(); ++i) {
        CAPTURE(i);
        CHECK(img.count(spec_test::bright, 0, i * r, w / 3, (i + 1) * r).n > 0);
    }
    CHECK(rec.events.empty());   // drawing alone sends nothing
}
TEST_CASE("spec controls: texts wider than a narrow row end in an ellipsis in the house look") {
    caliper_host::SpecOffscreen* c = spec_test::canvas();
    if (!c) return;
    Recorder rec;
    spec::Control now = make("now", spec::ControlKind::readout, "");
    now.title = "Loss";
    spec::BoundText text;
    text.format = "Loss 2.3071 at step 1,024 of the run, averaged over the last 64 batches";
    now.options.bound_text = text;
    spec::Control measure = make("measure", spec::ControlKind::segmented, "state:measure");
    measure.title = "Measure";
    measure.options.choices = {choice("norm", "Mean absolute activation"), choice("mean", "Signed mean")};
    spec::Control win = make("window", spec::ControlKind::range, "value");
    win.title = "Window of training steps";
    win.options.min = 0.0;
    win.options.max = 3000000.0;
    win.options.value_other = json::array({952113.0, 2999871.0});
    const spec::Spec sp = instrument_spec({now, measure, win});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, nullptr, &values, nullptr, &sp};
    const SpecLook look = c->look();
    const float row = caliper_host::spec_control_height(now, look);
    const std::vector<const spec::Control*> rows = {&now, &measure, &win};
    const int w = 300, h = (int)(row * rows.size());
    const spec_test::Image img = spec_test::render(w, h, [&](const ImRect& area) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const float y = area.Min.y + row * (float)i;
            caliper_host::draw_spec_control(*rows[i], ImRect(ImVec2(area.Min.x, y), ImVec2(area.Max.x, y + row)), look,
                                            host);
        }
    });
    REQUIRE(img.w == w);
    if (const char* dir = std::getenv("CALIPER_SPEC_TEST_DUMP"); dir && *dir)
        caliper_host::export_write_png_atomic(std::string(dir) + "/controls-cut-text.png", img.px.data(), (uint32_t)w,
                                              (uint32_t)h);
    // The readout's text reaches into the row's right part and stops before its edge.
    const int r = (int)row;
    CHECK(img.count(spec_test::bright, w / 2, 0, w - 8, r).n > 0);
    CHECK(img.count(spec_test::bright, w - 4, 0, w, r).n == 0);
    CHECK(rec.events.empty());
}
}  // TEST_SUITE("gpu")

TEST_CASE("spec controls: a text parameter sends UTF-8 on Return") {
    Ui ui;
    Recorder rec;
    FakeRun run;
    run.run = training();
    cb::RunParam prompt;
    prompt.name = "prompt"; prompt.type = "text"; prompt.value = "";
    run.run.params.push_back(prompt);
    auto control = make("prompt", spec::ControlKind::text, "param:prompt");
    const auto sp = instrument_spec({control});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, &run, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(control, kRow, look, host); };
    ui.click(300, 15, draw);
    ImGui::GetIO().AddInputCharactersUTF8("A café");
    ui.frame([&] { draw(); });
    CHECK(run.sent.empty());
    press(ui, ImGuiKey_Enter, [&] { draw(); });
    REQUIRE(run.sent.size() == 1);
    CHECK(run.sent[0] == std::make_pair(std::string("set prompt"), json("A café")));
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].fields["value"] == "A café");
}

TEST_CASE("spec controls: a text parameter preserves a long current value") {
    Ui ui;
    Recorder rec;
    FakeRun run;
    run.run = training();
    std::string text;
    for (unsigned i = 0; i < 2048; ++i) text += "\xc3\xa9";
    cb::RunParam prompt;
    prompt.name = "prompt"; prompt.type = "text"; prompt.value = text;
    run.run.params.push_back(prompt);
    auto control = make("long_prompt", spec::ControlKind::text, "param:prompt");
    bool accepts = true;
    SUBCASE("default bound") {}
    SUBCASE("explicit smaller bound") { control.options.max_length = 1024; accepts = false; }
    const auto sp = instrument_spec({control});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, &run, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(control, kRow, look, host); };
    ui.click(300, 15, draw);
    press(ui, ImGuiKey_Enter, [&] { draw(); });
    if (accepts) {
        REQUIRE(run.sent.size() == 1);
        CHECK(run.sent[0].second == text);
    } else CHECK(run.sent.empty());
    CHECK(run.run.params.back().value == text);
}

TEST_CASE("spec controls: an oversized parameter edit is refused without a truncated send") {
    Ui ui;
    Recorder rec;
    FakeRun run;
    run.run = training();
    cb::RunParam prompt;
    prompt.name = "prompt"; prompt.type = "text"; prompt.value = "";
    run.run.params.push_back(prompt);
    auto control = make("oversized_prompt", spec::ControlKind::text, "param:prompt");
    const auto sp = instrument_spec({control});
    cb::ValueStore values;
    values.reset(sp);
    SpecInstrumentHost host{&rec, &run, &values, nullptr, &sp};
    const SpecLook look;
    auto draw = [&] { return caliper_host::draw_spec_control(control, kRow, look, host); };
    ui.click(300, 15, draw);
    ImGui::GetIO().AddInputCharactersUTF8(std::string(4097, 'a').c_str());
    ui.frame([&] { draw(); });
    press(ui, ImGuiKey_Enter, [&] { draw(); });
    CHECK(run.sent.empty());
    CHECK(rec.events.empty());
    CHECK(run.run.params.back().value == "");
}
