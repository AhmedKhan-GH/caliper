// Parsing and writing the keys that the instrument interface adds to spec
// version 1 (design/instrument-interface.md): state, history, the window,
// surface and path steps, entities, annotations, binding references, the new
// options, the new control kinds, bound texts, layout shares and the params of
// programs and stages.
#include <set>
#include <string>

#include <doctest/doctest.h>

#include <caliper_builder/spec/validate.h>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

// One document that uses every new key, with the examples of the design page.
const char* kInstrumentDocument = R"toml(caliper_spec = 1

[applet]
id = "terrain-demo"
title = "Terrain Demo"

[[source]]
id = "act"
catalog = "live:blocks.0.attn.act"
history = 2048
  [[source.transform]]
  op = "reduce"
  axis = 2
  fn = "norm"

[[source]]
id = "terrain"
from = "act"
  [[source.transform]]
  op = "window"
  start = { control = "window", field = "lo" }
  stop = { control = "window", field = "hi" }
  [[source.transform]]
  op = "surface"
  max_cells = 32768
  pool = "mean"

[[source]]
id = "recent"
from = "act"
  [[source.transform]]
  op = "window"
  steps = { state = "recent" }
  [[source.transform]]
  op = "slice"
  axis = 0
  index = { state = "token", default = 0 }
  start = 1
  stop = { control = "stop" }
  step = 2

[[source]]
id = "orbit"
from = "act"
  [[source.transform]]
  op = "path"
  columns = [0, 1, 2]
  max_points = 65536

[[source]]
id = "loss"
sql = "SELECT step, loss FROM run42_metrics ORDER BY step"

[[state]]
id = "token"
title = "Token"
type = "integer"
default = 0
min = 0.0
max = 63.0

[[state]]
id = "measure"
type = "text"
default = "norm"
choices = ["norm", "mean"]

[[state]]
id = "palette"
type = "text"
choices = ["viridis", "magma"]

[[state]]
id = "recent"
type = "integer"
default = 256

[[state]]
id = "span"
type = "range"
default = [0.0, 10.0]

[[panel]]
id = "world"
primitive = "scene"
title = "Activation Terrain"
options = { axes = true, box = [3.0, 1.0, 0.8], exclude = ["debug/**"], legend = "auto", scale_bar = true, select = { entity = "terrain", field = "column", state = "token" }, up = "+z", view_camera = "overview", views = [{ azimuth = 30.0, distance = 4.0, elevation = 20.0, name = "overview", target = [0.0, 1.0, 0.0] }, { azimuth = 0.0, distance = 2.0, elevation = 90.0, name = "top", target = [0.0, 0.0, 0.0] }] }
  [[panel.entity]]
  path = "terrain"
  draw = "surface"
  bind = "terrain"
  options = { colormap = "viridis", highlight = { along = "column", state = "token" } }
  [[panel.entity]]
  path = "loss"
  draw = "lines"
  title = "Loss"
  bind = { x = "loss_window.step", z = { state = "measure" } }
  options = { mode = "strip", own = ["z"], width_px = 2.0, y = "max" }
  pose = { position = [0.0, 1.0, 0.0], rotation = [0.0, 0.0, 0.0, 1.0], scale = [1.0, 1.0, 2.0] }
  [[panel.annotation]]
  id = "window"
  kind = "band"
  axis = "x"
  range = { control = "window" }
  text = "Window"
  [[panel.annotation]]
  id = "picked"
  kind = "marker"
  at = { entity = "terrain", selection = "world" }
  text = "Token {token:d}"
  fields = { token = { state = "token" } }
  emphasis = "muted"
  [[panel.annotation]]
  id = "limit"
  kind = "threshold"
  axis = "y"
  value = { state = "token" }
  [[panel.annotation]]
  id = "note"
  kind = "label"
  at = [1.0, 2.0, 3.0]
  text = "Note"
  leader = false

[[panel]]
id = "detail"
primitive = "heatmap"
bind = { values = { state = "measure" } }
options = { colormap = { state = "palette" }, legend = true, range = { state = "span" }, select = { field = "column", state = "token" } }

[[panel]]
id = "loss-plot"
primitive = "series"
bind = { x = "loss.step", y = ["loss.loss"] }
options = { legend = false, x_range = { control = "window" }, y_range = [0.0, 1.0] }

[[panel]]
id = "picture"
primitive = "image"
bind = { values = "act" }
options = { legend = "auto", scale_bar = true }

[[panel]]
id = "now-text"
primitive = "text"
bind = { value = "loss.loss" }
options = { fields = { loss = "loss.loss", step = { field = "step", state = "cursor" } }, format = "Step {step:d} Loss {loss:.3f}" }

[[control]]
id = "window"
kind = "range"
title = "Steps"
bind = "value"
options = { follow = true, max = { field = "latest", state = "cursor" }, min = { field = "first", state = "cursor" }, min_span = 16.0, value = [0, 2048] }

[[control]]
id = "stop"
kind = "slider"
bind = "state:token"
options = { max = { control = "window", field = "hi" }, min = 0.0, step = 1.0, value = 4.0 }

[[control]]
id = "measure"
kind = "segmented"
bind = "state:measure"
options = { choices = [{ key = "n", label = "Norm", value = "norm" }, { key = "m", label = "Mean", value = "mean" }], multiple = false, value = "norm" }

[[control]]
id = "pick"
kind = "picker"
bind = "value"
options = { choices = ["a", "b"], value = "a" }

[[control]]
id = "pick-column"
kind = "picker"
bind = "value"
options = { choices_from = "loss.step" }

[[control]]
id = "name"
kind = "text"
bind = "state:measure"
options = { max_length = 32, placeholder = "Measure" }

[[control]]
id = "flag"
kind = "toggle"
bind = "value"
options = { value = true }

[[control]]
id = "now"
kind = "readout"
options = { fields = { loss = "loss.loss", step = { field = "step", state = "cursor" }, token = { state = "token" } }, format = "Step {step:,d}  Loss {loss:.3f}  Token {token:d}" }

[[control]]
id = "transport"
kind = "transport"
bind = "cursor"
options = { keys = true, speed = 10.0 }

[[program]]
id = "stat"
wgsl = "@compute @workgroup_size(1) fn main() {}"
inputs = [{ source = "live:x", var = "a" }]
output = { kind = "tensor", shape = [16] }
params = { gain = { control = "stop" }, level = 0.5 }

[[stage]]
panel = "world"
entity = "spec/terrain"
kind = "material"
params = { level = { control = "stop" } }
wgsl = "fn material(in: MaterialIn) -> vec4<f32> { return in.color; }"

[layout]
split = "vertical"
shares = [6.0, 3.0, 1.0]
children = [{ split = "horizontal", shares = [3.0, 2.0], children = [{ panel = "world", min = [48, 20] }, "detail"] }, "loss-plot", { split = "horizontal", children = ["window", "measure", "transport", "now"] }]
)toml";

cs::ParseResult parse(const std::string& text) { return cs::parse_toml(text); }

}  // namespace

TEST_CASE("a version 1 document with every new key reads into the expected values") {
    auto r = parse(kInstrumentDocument);
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    const cs::Spec& s = r.spec;
    CHECK(s.version == 1);
    CHECK(cs::required_version(s) == 1);

    REQUIRE(s.sources.size() == 5);
    CHECK(s.sources[0].history == std::optional<std::int64_t>(2048));
    const cs::Transform& window = s.sources[1].transforms[0];
    CHECK(window.op == cs::TransformOp::window);
    CHECK_FALSE(window.start.has_value());
    CHECK(window.references.at("start").kind == cs::Reference::Kind::control);
    CHECK(window.references.at("start").target == "window");
    CHECK(window.references.at("start").field == std::optional<std::string>("lo"));
    CHECK(window.references.at("stop").field == std::optional<std::string>("hi"));
    const cs::Transform& surface = s.sources[1].transforms[1];
    CHECK(surface.op == cs::TransformOp::surface);
    CHECK(surface.max_cells == std::optional<std::int64_t>(32768));
    CHECK(surface.pool == std::optional<std::string>("mean"));
    CHECK(s.sources[2].transforms[0].references.at("steps").target == "recent");
    const cs::Transform& slice = s.sources[2].transforms[1];
    CHECK_FALSE(slice.index.has_value());
    CHECK(slice.references.at("index").target == "token");
    CHECK(slice.references.at("index").fallback == json(0));
    CHECK(slice.start == std::optional<std::int64_t>(1));
    CHECK(slice.references.count("start") == 0);
    CHECK(slice.references.at("stop").target == "stop");
    const cs::Transform& path = s.sources[3].transforms[0];
    CHECK(path.op == cs::TransformOp::path);
    CHECK(path.columns == std::vector<std::int64_t>{0, 1, 2});
    CHECK(path.max_points == std::optional<std::int64_t>(65536));

    REQUIRE(s.states.size() == 5);
    CHECK(s.find_state("token") == &s.states[0]);
    CHECK(s.find_state("nothing") == nullptr);
    CHECK(s.states[0].type == cs::StateType::integer);
    CHECK(s.states[0].title == std::optional<std::string>("Token"));
    CHECK(s.states[0].default_value == json(0));
    CHECK(s.states[0].max == std::optional<double>(63.0));
    CHECK(s.states[1].choices == std::vector<json>{"norm", "mean"});
    CHECK(cs::state_initial_value(s.states[2]) == json("viridis"));
    CHECK(s.states[4].default_value == json::array({0.0, 10.0}));

    const cs::Panel& world = s.panels[0];
    CHECK(world.options.legend == json("auto"));
    CHECK(world.options.box == json::array({3.0, 1.0, 0.8}));
    REQUIRE(world.options.views.size() == 2);
    CHECK(world.options.views[0].name == "overview");
    CHECK(world.options.views[0].target == std::array<double, 3>{0, 1, 0});
    CHECK(world.options.views[1].elevation == 90.0);
    REQUIRE(world.options.select.has_value());
    CHECK(world.options.select->state == "token");
    CHECK(world.options.select->field == "column");
    CHECK(world.options.select->entity == std::optional<std::string>("terrain"));
    CHECK(world.options.scene.scale_bar == std::optional<bool>(true));
    CHECK(world.options.scene.view_camera == json("overview"));
    REQUIRE(world.entities.size() == 2);
    CHECK(world.entities[0].path == "terrain");
    CHECK(world.entities[0].draw == cs::DrawKind::surface);
    CHECK(world.entities[0].source == std::optional<std::string>("terrain"));
    CHECK(world.entities[0].options.at("highlight").at("state") == "token");
    CHECK_FALSE(world.entities[0].pose.has_value());
    CHECK(world.entities[1].title == std::optional<std::string>("Loss"));
    CHECK_FALSE(world.entities[1].source.has_value());
    CHECK(std::get<std::string>(world.entities[1].slots.at("x")) == "loss_window.step");
    CHECK(std::get<cs::Reference>(world.entities[1].slots.at("z")).target == "measure");
    REQUIRE(world.entities[1].pose.has_value());
    CHECK(world.entities[1].pose->scale == std::array<double, 3>{1, 1, 2});
    REQUIRE(world.annotations.size() == 4);
    CHECK(world.annotations[0].kind == cs::AnnotationKind::band);
    CHECK(world.annotations[0].range == json{{"control", "window"}});
    REQUIRE(world.annotations[0].text.has_value());
    CHECK(world.annotations[0].text->format == "Window");
    CHECK(world.annotations[1].emphasis == std::optional<std::string>("muted"));
    CHECK(world.annotations[1].at == (json{{"entity", "terrain"}, {"selection", "world"}}));
    CHECK(std::get<cs::Reference>(world.annotations[1].text->fields.at("token")).target == "token");
    CHECK(world.annotations[2].value == json{{"state", "token"}});
    CHECK(world.annotations[3].at == json::array({1.0, 2.0, 3.0}));
    CHECK(world.annotations[3].leader == std::optional<bool>(false));

    const cs::Panel& detail = s.panels[1];
    REQUIRE(detail.bind.at("values").reference.has_value());
    CHECK(detail.bind.at("values").bindings.empty());
    CHECK(detail.bind.at("values").reference->target == "measure");
    CHECK_FALSE(detail.options.colormap.has_value());
    CHECK(detail.options.references.at("colormap").target == "palette");
    CHECK(detail.options.references.at("range").target == "span");
    CHECK(detail.options.legend == json(true));
    CHECK(detail.options.set_names() == std::vector<std::string>{"colormap", "legend", "range", "select"});
    const cs::Panel& plot = s.panels[2];
    CHECK_FALSE(plot.options.x_range.has_value());
    CHECK(plot.options.references.at("x_range").target == "window");
    CHECK(plot.options.y_range == std::optional<std::array<double, 2>>(std::array<double, 2>{0.0, 1.0}));
    CHECK(plot.options.legend == json(false));
    CHECK(s.panels[3].options.scene.scale_bar == std::optional<bool>(true));
    const cs::Panel& text = s.panels[4];
    REQUIRE(text.options.bound_text.has_value());
    CHECK(text.options.bound_text->format == "Step {step:d} Loss {loss:.3f}");
    CHECK(std::get<std::string>(text.options.bound_text->fields.at("loss")) == "loss.loss");
    CHECK(std::get<cs::Reference>(text.options.bound_text->fields.at("step")).field == std::optional<std::string>("step"));

    REQUIRE(s.controls.size() == 9);
    const cs::Control& range = s.controls[0];
    CHECK(range.kind == cs::ControlKind::range);
    CHECK_FALSE(range.options.min.has_value());
    CHECK(range.options.references.at("min").field == std::optional<std::string>("first"));
    CHECK(range.options.references.at("max").field == std::optional<std::string>("latest"));
    CHECK(range.options.value_other == json::array({0, 2048}));
    CHECK_FALSE(range.options.value.has_value());
    CHECK(range.options.min_span == std::optional<double>(16.0));
    CHECK(range.options.follow == std::optional<bool>(true));
    const cs::Control& slider = s.controls[1];
    CHECK(slider.options.value == std::optional<double>(4.0));
    CHECK(slider.options.min == std::optional<double>(0.0));
    CHECK(slider.options.references.at("max").target == "window");
    CHECK(cs::parse_control_target(slider.bind) ==
          std::optional<cs::ControlTarget>(cs::ControlTarget{cs::ControlTarget::Kind::state, "token"}));
    const cs::Control& segmented = s.controls[2];
    CHECK(segmented.kind == cs::ControlKind::segmented);
    REQUIRE(segmented.options.choices.size() == 2);
    CHECK(segmented.options.choices[0].table);
    CHECK(segmented.options.choices[0].value == json("norm"));
    CHECK(segmented.options.choices[0].label == std::optional<std::string>("Norm"));
    CHECK(segmented.options.choices[0].key == std::optional<std::string>("n"));
    CHECK(segmented.options.multiple == std::optional<bool>(false));
    CHECK(segmented.options.value_other == json("norm"));
    CHECK_FALSE(s.controls[3].options.choices[0].table);
    CHECK(s.controls[4].options.choices_from == std::optional<std::string>("loss.step"));
    CHECK(s.controls[5].options.placeholder == std::optional<std::string>("Measure"));
    CHECK(s.controls[5].options.max_length == std::optional<std::int64_t>(32));
    CHECK(s.controls[6].options.value_other == json(true));
    const cs::Control& readout = s.controls[7];
    CHECK(readout.kind == cs::ControlKind::readout);
    CHECK(readout.bind.empty());
    REQUIRE(readout.options.bound_text.has_value());
    CHECK(readout.options.bound_text->fields.size() == 3);
    const cs::Control& transport = s.controls[8];
    CHECK(transport.kind == cs::ControlKind::transport);
    CHECK(cs::parse_control_target(transport.bind) ==
          std::optional<cs::ControlTarget>(cs::ControlTarget{cs::ControlTarget::Kind::cursor, ""}));
    CHECK(transport.options.speed == std::optional<double>(10.0));
    CHECK(transport.options.keys == std::optional<bool>(true));

    REQUIRE(s.programs.size() == 1);
    CHECK(s.programs[0].output.kind == cs::OutputKind::tensor);
    CHECK(s.programs[0].output.shape == std::vector<std::int64_t>{16});
    CHECK(std::get<double>(s.programs[0].params.at("level")) == 0.5);
    CHECK(std::get<cs::Reference>(s.programs[0].params.at("gain")).target == "stop");
    CHECK_FALSE(s.programs[0].builtin.has_value());
    REQUIRE(s.stages.size() == 1);
    CHECK(std::get<cs::Reference>(s.stages[0].params.at("level")).target == "stop");

    REQUIRE(s.layout.has_value());
    CHECK(s.layout->shares == std::vector<double>{6, 3, 1});
    CHECK_FALSE(s.layout->ratio.has_value());
    const cs::LayoutNode& leaf = s.layout->children[0].children[0];
    CHECK(leaf.kind == cs::LayoutNode::Kind::panel);
    CHECK(leaf.panel == "world");
    CHECK(leaf.min == std::optional<std::array<std::int64_t, 2>>(std::array<std::int64_t, 2>{48, 20}));
    CHECK_FALSE(s.layout->children[0].children[1].min.has_value());
}

TEST_CASE("the document with every new key round trips through canonical TOML") {
    auto first = parse(kInstrumentDocument);
    REQUIRE_MESSAGE(first.ok(), dump(first.diagnostics));
    const std::string text = cs::to_toml(first.spec);
    CHECK(text.find("caliper_spec = 1\n") == 0);
    CHECK(text.find("\n[[state]]\nid = \"token\"\n") != std::string::npos);
    CHECK(text.find("[[state]]") < text.find("[[panel]]"));
    CHECK(text.find("[[panel]]\nid = \"world\"") < text.find("  [[panel.entity]]\n  path = \"terrain\"\n"));
    CHECK(text.find("  [[panel.entity]]") < text.find("  [[panel.annotation]]\n  id = \"window\"\n"));
    CHECK(text.find("  [[panel.annotation]]") < text.find("[[panel]]\nid = \"detail\""));
    auto second = parse(text);
    REQUIRE_MESSAGE(second.ok(), (dump(second.diagnostics) + text));
    CHECK(second.spec == first.spec);
    CHECK(cs::to_toml(second.spec) == text);  // the writer reaches a fixed point at once
}

TEST_CASE("the document with every new key round trips through canonical JSON") {
    auto parsed = parse(kInstrumentDocument);
    REQUIRE(parsed.ok());
    const json j = cs::to_json(parsed.spec);
    auto from_value = cs::from_json(j);
    REQUIRE_MESSAGE(from_value.ok(), dump(from_value.diagnostics));
    CHECK(from_value.spec == parsed.spec);

    const std::string text = cs::to_json_text(parsed.spec);
    auto from_text = cs::from_json_text(text);
    REQUIRE_MESSAGE(from_text.ok(), dump(from_text.diagnostics));
    CHECK(from_text.spec == parsed.spec);
    CHECK(cs::to_json_text(from_text.spec) == text);
    CHECK(json::parse(text) == j);

    // The JSON form of the new keys.
    CHECK(j.at("state").size() == 5);
    CHECK(j.at("source")[1].at("transform")[0].at("start") == (json{{"control", "window"}, {"field", "lo"}}));
    CHECK(j.at("panel")[0].at("entity").size() == 2);
    CHECK(j.at("panel")[0].at("annotation").size() == 4);
    CHECK(j.at("panel")[1].at("bind").at("values") == json{{"state", "measure"}});
    CHECK(j.at("control")[7].count("bind") == 0);
    CHECK(j.at("layout").at("children")[0].at("children")[0] ==
          (json{{"panel", "world"}, {"min", json::array({48, 20})}}));
    CHECK(j.at("program")[0].at("output") == (json{{"kind", "tensor"}, {"shape", json::array({16})}}));
}

TEST_CASE("the item decoders and writers of state, entity and annotation agree with the document") {
    auto parsed = parse(kInstrumentDocument);
    REQUIRE(parsed.ok());
    const cs::Spec& s = parsed.spec;

    const json state = cs::to_json(s.states[1]);
    auto state_back = cs::state_from_json(state);
    REQUIRE_MESSAGE(state_back.ok(), dump(state_back.diagnostics));
    CHECK(state_back.value == s.states[1]);

    const json entity = cs::to_json(s.panels[0].entities[1]);
    auto entity_back = cs::entity_from_json(entity, "panel[0].entity[1]");
    REQUIRE_MESSAGE(entity_back.ok(), dump(entity_back.diagnostics));
    CHECK(entity_back.value == s.panels[0].entities[1]);

    const json annotation = cs::to_json(s.panels[0].annotations[1]);
    auto annotation_back = cs::annotation_from_json(annotation, "panel[0].annotation[1]");
    REQUIRE_MESSAGE(annotation_back.ok(), dump(annotation_back.diagnostics));
    CHECK(annotation_back.value == s.panels[0].annotations[1]);

    auto bad = cs::entity_from_json(json{{"path", "a"}, {"draw", "mesh"}, {"bind", "src"}}, "panel[0].entity[0]");
    CHECK_FALSE(bad.ok());
    CHECK(has(bad.diagnostics, "spec.bad_value", "panel[0].entity[0].draw"));
    auto missing = cs::state_from_json(json{{"id", "t"}});
    CHECK(has(missing.diagnostics, "spec.missing_key", "state.type"));
}

TEST_CASE("a version 0 document keeps the new keys unknown") {
    const std::string head = "caliper_spec = 0\n[applet]\nid = \"a\"\n";
    auto state = parse(head + "[[state]]\nid = \"t\"\ntype = \"number\"\n");
    CHECK(has(state.diagnostics, "spec.unknown_key", "state"));

    auto history = parse(head + "[[source]]\nid = \"s\"\ncatalog = \"live:x\"\nhistory = 4\n");
    CHECK(has(history.diagnostics, "spec.unknown_key", "source[0].history"));

    auto window = parse(head + "[[source]]\nid = \"s\"\nfrom = \"t\"\n  [[source.transform]]\n  op = \"window\"\n");
    CHECK(has(window.diagnostics, "spec.unknown_transform", "source[0].transform[0].op"));

    auto reference = parse(head + "[[source]]\nid = \"s\"\nfrom = \"t\"\n  [[source.transform]]\n  op = \"slice\"\n"
                                  "  axis = 0\n  index = { state = \"t\" }\n");
    CHECK(has(reference.diagnostics, "spec.wrong_type", "source[0].transform[0].index"));
    CHECK_FALSE(has(reference.diagnostics, "spec.bad_reference"));

    auto option = parse(head + "[[panel]]\nid = \"p\"\nprimitive = \"series\"\nbind = { x = \"a.x\", y = \"a.y\" }\n"
                               "options = { legend = true }\n");
    CHECK(has(option.diagnostics, "spec.unknown_option", "panel[0].options.legend"));

    auto entity = parse(head + "[[panel]]\nid = \"p\"\nprimitive = \"series\"\n  [[panel.entity]]\n  path = \"a\"\n");
    CHECK(has(entity.diagnostics, "spec.unknown_key", "panel[0].entity"));

    auto layout = parse(head + "[layout]\nsplit = \"horizontal\"\nshares = [1, 1]\nchildren = [\"a\", \"b\"]\n");
    CHECK(has(layout.diagnostics, "spec.unknown_key", "layout.shares"));

    // The same state table as version 1 reads.
    CHECK(parse("caliper_spec = 1\n[applet]\nid = \"a\"\n[[state]]\nid = \"t\"\ntype = \"number\"\n").ok());
}

TEST_CASE("a table that looks like a reference but is not one gives spec.bad_reference") {
    const std::string head = "caliper_spec = 1\n[applet]\nid = \"a\"\n";
    auto slice = [&](const std::string& index) {
        return parse(head + "[[source]]\nid = \"s\"\nfrom = \"t\"\n  [[source.transform]]\n  op = \"slice\"\n"
                            "  axis = 0\n  index = " +
                     index + "\n")
            .diagnostics;
    };
    CHECK(has(slice("{ state = 3 }"), "spec.bad_reference", "source[0].transform[0].index"));
    CHECK(has(slice("{ control = \"a\", state = \"b\" }"), "spec.bad_reference", "source[0].transform[0].index"));
    CHECK(has(slice("{ state = \"a\", extra = 1 }"), "spec.bad_reference", "source[0].transform[0].index"));
    CHECK(has(slice("{ state = \"a\", field = 1 }"), "spec.bad_reference", "source[0].transform[0].index"));
    CHECK(has(slice("{ state = \"a\", default = { state = \"b\" } }"), "spec.bad_reference",
              "source[0].transform[0].index"));
    CHECK(find(slice("{ state = 3 }"), "spec.bad_reference")->message.find("reference") != std::string::npos);
    CHECK(slice("{ state = \"a\" }").empty());
    // A table that has no input key is not a reference, and has the wrong type as before.
    CHECK(has(slice("{ field = \"a\" }"), "spec.wrong_type", "source[0].transform[0].index"));

    auto panel = [&](const std::string& options) {
        return parse(head + "[[panel]]\nid = \"p\"\nprimitive = \"heatmap\"\nbind = { values = \"s\" }\n"
                            "options = { " +
                     options + " }\n")
            .diagnostics;
    };
    CHECK(has(panel("colormap = { selection = 1 }"), "spec.bad_reference", "panel[0].options.colormap"));
    CHECK(has(panel("range = { control = \"a\", state = \"b\" }"), "spec.bad_reference", "panel[0].options.range"));
    CHECK(panel("range = { control = \"a\", field = \"lo\" }").empty());

    auto bound = parse(head + "[[panel]]\nid = \"p\"\nprimitive = \"heatmap\"\nbind = { values = { state = 4 } }\n");
    CHECK(has(bound.diagnostics, "spec.bad_reference", "panel[0].bind.values"));
    auto control = parse(head + "[[control]]\nid = \"c\"\nkind = \"slider\"\nbind = \"value\"\n"
                                "options = { min = { control = true } }\n");
    CHECK(has(control.diagnostics, "spec.bad_reference", "control[0].options.min"));
    auto fields = parse(head + "[[control]]\nid = \"c\"\nkind = \"readout\"\n"
                               "options = { format = \"{a}\", fields = { a = { state = 1 } } }\n");
    CHECK(has(fields.diagnostics, "spec.bad_reference", "control[0].options.fields.a"));
    auto params = parse(head + "[[stage]]\npanel = \"w\"\nentity = \"spec/a\"\nkind = \"material\"\nwgsl = \"x\"\n"
                               "params = { level = { control = 2 } }\n");
    CHECK(has(params.diagnostics, "spec.bad_reference", "stage[0].params.level"));
}

TEST_CASE("the types of the new keys are checked") {
    const std::string head = "caliper_spec = 1\n[applet]\nid = \"a\"\n";
    auto diagnostics = [&](const std::string& body) { return parse(head + body).diagnostics; };
    CHECK(has(diagnostics("[[state]]\nid = \"t\"\ntype = \"vector\"\n"), "spec.bad_value", "state[0].type"));
    CHECK(has(diagnostics("[[state]]\nid = \"t\"\n"), "spec.missing_key", "state[0].type"));
    CHECK(has(diagnostics("[[state]]\nid = \"t\"\ntype = \"text\"\nchoices = \"a\"\n"), "spec.wrong_type",
              "state[0].choices"));
    CHECK(has(parse("caliper_spec = 1\nstate = 3\n[applet]\nid = \"a\"\n").diagnostics, "spec.wrong_type", "state"));
    CHECK(has(diagnostics("[[source]]\nid = \"s\"\ncatalog = \"live:x\"\nhistory = \"many\"\n"), "spec.wrong_type",
              "source[0].history"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"scene\"\noptions = { views = [{ name = \"v\" }] }\n"),
              "spec.missing_key", "panel[0].options.views[0].target"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"scene\"\noptions = { legend = \"always\" }\n"),
              "spec.bad_value", "panel[0].options.legend"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"scene\"\noptions = { box = [1, 2] }\n"),
              "spec.wrong_type", "panel[0].options.box"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"table\"\noptions = { legend = true }\n"),
              "spec.unknown_option", "panel[0].options.legend"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"text\"\noptions = { fields = { a = \"b\" } }\n"),
              "spec.missing_key", "panel[0].options.format"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"scene\"\n  [[panel.entity]]\n  path = \"a\"\n"
                          "  bind = \"s\"\n"),
              "spec.missing_key", "panel[0].entity[0].draw"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"scene\"\n  [[panel.entity]]\n  path = \"a\"\n"
                          "  draw = \"points\"\n  bind = \"s\"\n  pose = { position = [1, 2] }\n"),
              "spec.wrong_type", "panel[0].entity[0].pose.position"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"series\"\n  [[panel.annotation]]\n  id = \"n\"\n"
                          "  kind = \"arrow\"\n"),
              "spec.bad_value", "panel[0].annotation[0].kind"));
    CHECK(has(diagnostics("[[panel]]\nid = \"p\"\nprimitive = \"series\"\n  [[panel.annotation]]\n  id = \"n\"\n"
                          "  kind = \"label\"\n  size = 3\n"),
              "spec.unknown_key", "panel[0].annotation[0].size"));
    CHECK(has(diagnostics("[[control]]\nid = \"c\"\nkind = \"picker\"\nbind = \"value\"\noptions = { choices = [] }\n"),
              "spec.bad_value", "control[0].options.choices"));
    CHECK(has(diagnostics("[[control]]\nid = \"c\"\nkind = \"picker\"\nbind = \"value\"\n"
                          "options = { choices = [{ label = \"A\" }] }\n"),
              "spec.missing_key", "control[0].options.choices[0].value"));
    CHECK(has(diagnostics("[[control]]\nid = \"c\"\nkind = \"transport\"\nbind = \"cursor\"\n"
                          "options = { speed = \"fast\" }\n"),
              "spec.wrong_type", "control[0].options.speed"));
    CHECK(has(diagnostics("[[control]]\nid = \"c\"\nkind = \"slider\"\nbind = \"value\"\noptions = { speeds = 1 }\n"),
              "spec.unknown_option", "control[0].options.speeds"));
    CHECK(has(diagnostics("[[control]]\nid = \"c\"\nkind = \"slider\"\n"), "spec.missing_key", "control[0].bind"));
    CHECK(has(diagnostics("[layout]\nsplit = \"horizontal\"\nshares = [1, \"a\"]\nchildren = [\"a\", \"b\"]\n"),
              "spec.wrong_type", "layout.shares[1]"));
    CHECK(has(diagnostics("[layout]\nsplit = \"horizontal\"\nchildren = [{ panel = \"a\", min = [4] }, \"b\"]\n"),
              "spec.wrong_type", "layout.children[0].min"));
    CHECK(has(diagnostics("[[program]]\nid = \"p\"\nwgsl = \"x\"\ninputs = []\noutput = { kind = \"tensor\" }\n"),
              "spec.missing_key", "program[0].output.shape"));
    CHECK(has(diagnostics("[[stage]]\npanel = \"w\"\nentity = \"spec/a\"\nkind = \"material\"\nwgsl = \"x\"\n"
                          "params = { level = \"high\" }\n"),
              "spec.wrong_type", "stage[0].params.level"));
}

TEST_CASE("a spec built in code needs version 1 for each new key") {
    auto base = [] {
        cs::Spec s;
        s.applet.id = "a";
        return s;
    };
    CHECK(cs::required_version(base()) == 0);

    cs::Spec with_state = base();
    with_state.states.push_back(cs::StateVar{});
    CHECK(cs::required_version(with_state) == 1);

    cs::Spec with_history = base();
    with_history.sources.emplace_back();
    with_history.sources[0].history = 4;
    CHECK(cs::required_version(with_history) == 1);

    cs::Spec with_reference = base();
    with_reference.sources.emplace_back();
    with_reference.sources[0].transforms.emplace_back();
    CHECK(cs::required_version(with_reference) == 0);
    with_reference.sources[0].transforms[0].references["index"] = cs::Reference{};
    CHECK(cs::required_version(with_reference) == 1);

    cs::Spec with_legend = base();
    with_legend.panels.emplace_back();
    with_legend.panels[0].primitive = cs::Primitive::series;
    CHECK(cs::required_version(with_legend) == 0);
    with_legend.panels[0].options.legend = true;
    CHECK(cs::required_version(with_legend) == 1);

    cs::Spec with_shares = base();
    with_shares.layout = cs::LayoutNode::of_panel("p");
    CHECK(cs::required_version(with_shares) == 0);
    with_shares.layout->min = std::array<std::int64_t, 2>{4, 4};
    CHECK(cs::required_version(with_shares) == 1);
}

TEST_CASE("the validator refuses each new key that is not built yet with spec.not_built") {
    auto parsed = parse(kInstrumentDocument);
    REQUIRE_MESSAGE(parsed.ok(), dump(parsed.diagnostics));
    const cs::ValidationReport report = cs::validate(parsed.spec);
    std::set<std::string> refused;
    for (const auto& d : report.diagnostics) {
        CHECK(d.code != "spec.internal");
        if (d.code == "spec.not_built") refused.insert(d.path);
    }
    // One refusal at least for state, an entity, an annotation, a window step and an operator.
    CHECK(refused.count("state") == 1);
    bool entity = false, annotation = false;
    for (const std::string& p : refused) {
        entity = entity || p.find(".entity") != std::string::npos;
        annotation = annotation || p.find(".annotation") != std::string::npos;
    }
    CHECK(entity);
    CHECK(annotation);
}
