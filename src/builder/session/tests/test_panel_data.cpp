// Tests of panel.data, one panel's data for the browser's spec page
// (panel_data.h): the dashboard's three panels from a saved applet and from
// the open spec, the version and `known`, the native runtime's bounds on each
// primitive, the states and errors, spec.get of a saved applet, and the viewer
// origin, which runs reads only.
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>

#include "helpers.h"
#include "parquet_writer.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

std::vector<unsigned char> unbase64(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<unsigned char> out;
    unsigned int acc = 0;
    int bits = 0;
    for (char c : s) {
        const int v = val(c);
        if (v < 0) continue;
        acc = (acc << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// An array of the result as doubles, checking its dtype and shape.
std::vector<double> numbers(const json& a, const std::string& dtype, const std::vector<std::int64_t>& shape) {
    REQUIRE(a.is_object());
    CHECK(a["dtype"] == dtype);
    CHECK(a["shape"].get<std::vector<std::int64_t>>() == shape);
    const auto bytes = unbase64(a["base64"].get<std::string>());
    std::vector<double> out;
    if (dtype == "f32") {
        REQUIRE(bytes.size() % 4 == 0);
        for (std::size_t i = 0; i < bytes.size(); i += 4) {
            float f;
            std::memcpy(&f, bytes.data() + i, 4);
            out.push_back(f);
        }
    } else {
        REQUIRE(bytes.size() % 8 == 0);
        for (std::size_t i = 0; i < bytes.size(); i += 8) {
            double d;
            std::memcpy(&d, bytes.data() + i, 8);
            out.push_back(d);
        }
    }
    return out;
}

const char* const kDashboard = R"toml(caliper_spec = 0
[applet]
id = "dash"
title = "Run 42"
[[source]]
id = "metrics"
catalog = "table:run42_metrics"
[[source]]
id = "attn"
catalog = "tensor:attn_weights"
  [[source.transform]]
  op = "slice"
  axis = 0
  index = 0
  [[source.transform]]
  op = "reduce"
  axis = 0
  fn = "mean"
  [[source.transform]]
  op = "normalise"
  mode = "minmax"
[[panel]]
id = "curves"
primitive = "series"
title = "Loss and accuracy"
bind = { x = "metrics.step", y = ["metrics.loss", "metrics.accuracy"] }
options = { y_log = true }
[[panel]]
id = "attention"
primitive = "heatmap"
bind = { values = "attn" }
options = { colormap = "magma", range = [0.0, 1.0] }
[[panel]]
id = "rows"
primitive = "table"
bind = { rows = "metrics" }
options = { columns = ["step", "loss"] }
[layout]
split = "vertical"
ratio = 0.65
children = [{ split = "horizontal", children = ["curves", "attention"] }, "rows"]
)toml";

struct Fixture {
    std::string ws;
    std::unique_ptr<Session> s;
    explicit Fixture(const std::string& name) {
        ws = fresh_dir(name) + "/workspace";
        // step 0..9, loss 1 / (step + 1), accuracy step / 10 (0 at step 0,
        // which a log axis cannot show).
        write_parquet(ws + "/data/run42_metrics.parquet",
                      "SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, "
                      "CAST(range AS DOUBLE) / 10 AS accuracy FROM range(10)");
        std::vector<float> w(96);
        for (int i = 0; i < 96; ++i) w[i] = static_cast<float>(i);
        write_safetensors_f32(ws + "/data/attn.safetensors", "weights", {2, 3, 4, 4}, w);
        s = open_session(ws);
        ok(*s, "data.import", {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}});
        ok(*s, "spec.new", {{"id", "scratch"}});
    }
};

json panel_data(Session& s, json args, CommandOrigin origin = CommandOrigin::ui) {
    return ok(s, "panel.data", std::move(args), origin);
}

}  // namespace

TEST_CASE("panel.data gives the dashboard's three panels from the saved applet, bounded and versioned") {
    Fixture f("dashboard");
    ok(*f.s, "spec.set_text", {{"text", kDashboard}});
    ok(*f.s, "applet.save", json::object());
    // Another spec is open now; the saved applet is still what applet reads.
    ok(*f.s, "spec.set_text", {{"text", "caliper_spec = 0\n[applet]\nid = \"other\"\n"}});

    const json got = ok(*f.s, "spec.get", {{"applet", "dash"}});
    CHECK(got["id"] == "dash");
    CHECK(got["path"] == "applets/dash/dash.spec.toml");
    CHECK(got["dirty"] == false);
    CHECK(got["json"]["panel"].size() == 3);
    CHECK(ok(*f.s, "spec.get")["id"] == "other");

    // The series: x from the step column, the accuracy of step 0 left out by
    // the log axis but counted, and its last value.
    const json curves = panel_data(*f.s, {{"applet", "dash"}, {"panel", "curves"}});
    CHECK(curves["primitive"] == "series");
    CHECK(curves["unchanged"] == false);
    CHECK(curves["error"].is_null());
    CHECK(curves["state"].is_null());
    CHECK(curves["version"].get<std::string>().size() == 16);
    const json& d = curves["data"];
    CHECK(d["x_index"] == false);
    CHECK(d["steps"] == 10);
    REQUIRE(d["lines"].size() == 2);
    CHECK(d["lines"][0]["binding"] == "metrics.loss");
    CHECK(d["lines"][0]["row"].is_null());
    CHECK(d["lines"][0]["count"] == 10);
    CHECK(numbers(d["lines"][0]["x"], "f64", {10}) == std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    const auto loss = numbers(d["lines"][0]["y"], "f64", {10});
    for (int i = 0; i < 10; ++i) CHECK(loss[i] == doctest::Approx(1.0 / (i + 1)));
    CHECK(d["lines"][1]["binding"] == "metrics.accuracy");
    CHECK(numbers(d["lines"][1]["x"], "f64", {9}).front() == 1.0);
    CHECK(d["lines"][1]["last"].get<double>() == doctest::Approx(0.9));

    // The heatmap: layer 0, the mean over heads, minmax: (4r + c) / 15.
    const json heat = panel_data(*f.s, {{"applet", "dash"}, {"panel", "attention"}});
    CHECK(heat["data"]["source_rows"] == 4);
    CHECK(heat["data"]["stride_rows"] == 1);
    const auto v = numbers(heat["data"]["values"], "f32", {4, 4});
    for (int i = 0; i < 16; ++i) CHECK(v[i] == doctest::Approx(i / 15.0));

    // The table: the chosen columns, with their types, and every row.
    const json rows = panel_data(*f.s, {{"applet", "dash"}, {"panel", "rows"}});
    CHECK(rows["data"]["columns"] == json::parse(R"([{"name":"step","type":"BIGINT"},{"name":"loss","type":"DOUBLE"}])"));
    CHECK(rows["data"]["rows"].size() == 10);
    CHECK(rows["data"]["rows"][3][0] == 3);
    CHECK(rows["data"]["total_rows"] == 10);
    CHECK(rows["data"]["truncated"] == false);

    // known: unchanged without data until the spec or the catalog changes.
    const std::string version = rows["version"];
    const json same = panel_data(*f.s, {{"applet", "dash"}, {"panel", "rows"}, {"known", version}});
    CHECK(same["unchanged"] == true);
    CHECK(same["data"].is_null());
    CHECK(same["version"] == version);
    CHECK(panel_data(*f.s, {{"applet", "dash"}, {"panel", "curves"}, {"known", version}})["unchanged"] == false);
    // A loader run moves the catalog's revision.
    const json loaders = ok(*f.s, "loader.list");
    ok(*f.s, "loader.run", {{"id", loaders["loaders"][0]["id"]}});
    const json after = panel_data(*f.s, {{"applet", "dash"}, {"panel", "rows"}, {"known", version}});
    CHECK(after["unchanged"] == false);
    CHECK(after["version"] != version);

    // Errors of the arguments.
    CHECK(err(*f.s, "panel.data", {{"applet", "dash"}, {"panel", "nope"}}) == "panel.not_found");
    CHECK(err(*f.s, "panel.data", {{"applet", "gone"}, {"panel", "rows"}}) == "applet.not_found");
    CHECK(err(*f.s, "spec.get", {{"applet", "gone"}}) == "applet.not_found");
    // Without applet, the open spec, which has no such panel.
    CHECK(err(*f.s, "panel.data", {{"panel", "rows"}}) == "panel.not_found");
}

TEST_CASE("panel.data keeps the native runtime's bounds on every primitive") {
    Fixture f("bounds");
    // 10,000 steps, 1,500 rows, a 3,000 x 10 grid, 150,000 points with class
    // labels, a long text cell.
    write_parquet(f.ws + "/data/long.parquet",
                  "SELECT range AS step, sin(range / 100.0) AS v, CAST(range % 5 AS INTEGER) AS label, "
                  "repeat('x', 300) AS note FROM range(10000)");
    std::vector<float> big(30000);
    for (int i = 0; i < 30000; ++i) big[i] = static_cast<float>(i);
    write_safetensors_f32(f.ws + "/data/grid.safetensors", "g", {3000, 10}, big);
    std::vector<float> cloud(300000);
    for (int i = 0; i < 300000; ++i) cloud[i] = static_cast<float>(i % 1000);
    write_safetensors_f32(f.ws + "/data/cloud.safetensors", "p", {150000, 2}, cloud);
    ok(*f.s, "data.import", {{"paths", {"data/long.parquet", "data/grid.safetensors", "data/cloud.safetensors"}}});
    ok(*f.s, "spec.set_text", {{"text", R"toml(caliper_spec = 0
[applet]
id = "b"
[[source]]
id = "long"
catalog = "table:long"
[[source]]
id = "grid"
catalog = "tensor:grid_g"
[[source]]
id = "cloud"
catalog = "tensor:cloud_p"
[[panel]]
id = "line"
primitive = "series"
bind = { y = "long.v" }
[[panel]]
id = "heat"
primitive = "heatmap"
bind = { values = "grid" }
[[panel]]
id = "pts"
primitive = "points"
bind = { positions = "cloud" }
[[panel]]
id = "cls"
primitive = "points"
bind = { x = "long.step", y = "long.v", color = "long.label" }
[[panel]]
id = "hist"
primitive = "histogram"
bind = { values = "grid" }
options = { bins = 10 }
[[panel]]
id = "tab"
primitive = "table"
bind = { rows = "long" }
[[panel]]
id = "note"
primitive = "text"
bind = { value = "long.v" }
options = { text = "Last value" }
)toml"}});

    const json line = panel_data(*f.s, {{"panel", "line"}});
    CHECK(line["data"]["x_index"] == true);
    CHECK(line["data"]["lines"][0]["count"] == 10000);
    const auto ys = numbers(line["data"]["lines"][0]["y"], "f64",
                            {line["data"]["lines"][0]["y"]["shape"][0].get<std::int64_t>()});
    CHECK(ys.size() <= 2 * 2048 + 2);
    CHECK(ys.size() > 2048);
    REQUIRE(line["bounds"].size() == 1);
    CHECK(line["bounds"][0].get<std::string>().rfind("series long.v decimated from 10000 to ", 0) == 0);

    const json heat = panel_data(*f.s, {{"panel", "heat"}});
    CHECK(heat["data"]["source_rows"] == 3000);
    CHECK(heat["data"]["stride_rows"] == 2);
    CHECK(heat["data"]["stride_cols"] == 1);
    const auto hv = numbers(heat["data"]["values"], "f32", {1500, 10});
    CHECK(hv[10] == 20.0);   // row 1 of the sample is row 2 of the grid
    CHECK(heat["bounds"][0] == "heatmap sampled from 3000x10 to 1500x10");

    const json pts = panel_data(*f.s, {{"panel", "pts"}});
    CHECK(pts["data"]["total"] == 150000);
    CHECK(pts["data"]["stride"] == 2);
    CHECK(pts["data"]["three_d"] == false);
    CHECK(numbers(pts["data"]["x"], "f32", {75000}).size() == 75000);
    CHECK(pts["data"]["color"].is_null());

    // A colour column of an integer type: the page applies the class-label rule.
    const json cls = panel_data(*f.s, {{"panel", "cls"}});
    CHECK(cls["data"]["color_integer"] == true);
    const auto labels = numbers(cls["data"]["color"], "f32", {10000});
    CHECK(labels[7] == 2.0);

    // A histogram sends its counts, never the values.
    const json hist = panel_data(*f.s, {{"panel", "hist"}});
    CHECK(hist["data"]["counts"].size() == 10);
    CHECK(hist["data"]["counts"][0] == 3000.0);
    CHECK(hist["data"]["lo"] == 0.0);
    CHECK(hist["data"]["hi"] == 29999.0);
    CHECK(hist["data"]["total"] == 30000.0);

    // A table: 1,000 rows, a text cell cut at 256 characters and an ellipsis.
    const json tab = panel_data(*f.s, {{"panel", "tab"}});
    CHECK(tab["data"]["rows"].size() == 1000);
    CHECK(tab["data"]["total_rows"] == 10000);
    const std::string note = tab["data"]["rows"][0][3];
    CHECK(note == std::string(256, 'x') + "\xE2\x80\xA6");
    CHECK(tab["bounds"][0] == "table cut from 10000 to 1000 rows");

    // A text panel: the last value only.
    const json text = panel_data(*f.s, {{"panel", "note"}});
    CHECK(text["data"]["value"].get<double>() == doctest::Approx(std::sin(9999 / 100.0)));
}

TEST_CASE("panel.data reports a state or the resolver's error in place of data") {
    Fixture f("states");
    ok(*f.s, "spec.set_text", {{"text", R"toml(caliper_spec = 0
[applet]
id = "st"
[[source]]
id = "m"
catalog = "table:run42_metrics"
[[source]]
id = "w"
catalog = "tensor:attn_weights"
[[source]]
id = "q"
sql = "SELECT step FROM run42_metrics WHERE step > 100"
[[source]]
id = "live"
catalog = "live:run/attn"
[[panel]]
id = "none"
primitive = "histogram"
bind = { values = "q.step" }
[[panel]]
id = "off"
primitive = "heatmap"
bind = { values = "live" }
)toml"}});
    // Data that does not fit its slot, which the validator lets through
    // when the shapes are not known: a heatmap of a rank-1 tensor.
    spec::Spec s = f.s->open_spec()->spec;
    spec::Panel heat;
    heat.id = "flat";
    heat.primitive = spec::Primitive::heatmap;
    s.panels.push_back(heat);
    PanelData d;
    d.panel = "flat";
    d.primitive = spec::Primitive::heatmap;
    SlotData slot;
    slot.slot = "values";
    slot.binding = slot.source = "w";
    std::vector<uint8_t> bytes(12, 0);
    auto made = TensorData::from_bytes(DType::f32, {3}, bytes);
    REQUIRE(made.error.ok());
    slot.tensor = made.tensor;
    d.slots.push_back(slot);
    const json flat = panel_data_result(s, d, 0);
    CHECK(flat["state"] == json{{"kind", "mismatch"}, {"detail", "w"}});
    CHECK(flat["data"].is_null());
    const json none = panel_data(*f.s, {{"panel", "none"}});
    CHECK(none["state"] == json{{"kind", "empty"}, {"detail", "q.step"}});
    const json off = panel_data(*f.s, {{"panel", "off"}});
    CHECK(off["error"]["code"] == "resolve.unavailable");
    CHECK(off["data"].is_null());
}

TEST_CASE("the viewer origin runs reads and is refused every command that changes the workspace") {
    Fixture f("viewer");
    ok(*f.s, "spec.set_text", {{"text", kDashboard}});
    ok(*f.s, "applet.save", json::object());
    const std::string journal = read_text(f.ws + "/journal/commands.jsonl");
    CHECK(panel_data(*f.s, {{"applet", "dash"}, {"panel", "rows"}}, CommandOrigin::viewer)["data"]["rows"].size() == 10);
    CHECK(ok(*f.s, "spec.get", {{"applet", "dash"}}, CommandOrigin::viewer)["id"] == "dash");
    CHECK(ok(*f.s, "applet.list", json::object(), CommandOrigin::viewer)["applets"].size() == 1);
    for (const auto& call : std::vector<std::pair<std::string, json>>{
             {"panel.remove", {{"id", "rows"}}},
             {"spec.set_text", {{"text", "caliper_spec = 0\n[applet]\nid = \"x\"\n"}}},
             {"applet.save", {{"overwrite", true}}},
             {"history.undo", json::object()},
             {"loader.delete", {{"id", "run42_metrics"}}}}) {
        const std::string& name = call.first;
        CAPTURE(name);
        const auto r = invoke(*f.s, name, call.second, CommandOrigin::viewer);
        CHECK_FALSE(r.ok);
        CHECK(r.error.code == "command.not_allowed");
        CHECK(r.error.where == "origin");
    }
    // Nothing changed and nothing was journaled.
    CHECK(f.s->open_spec()->spec.panels.size() == 3);
    CHECK(read_text(f.ws + "/journal/commands.jsonl") == journal);
    CHECK(command_origin_from_name("viewer") == CommandOrigin::viewer);
    CHECK(std::string(command_origin_name(CommandOrigin::viewer)) == "viewer");
}

TEST_CASE("the version of a panel's data follows the spec, the catalog's revision and the generation") {
    const spec::Spec a = parse_fixture("dashboard.spec.toml");
    spec::Spec b = a;
    b.applet.title = "Run 43";
    const std::string v = panel_data_version(a, "rows", 1, 0);
    CHECK(v.size() == 16);
    CHECK(panel_data_version(a, "rows", 1, 0) == v);
    CHECK(panel_data_version(a, "curves", 1, 0) != v);
    CHECK(panel_data_version(b, "rows", 1, 0) != v);
    CHECK(panel_data_version(a, "rows", 2, 0) != v);
    CHECK(panel_data_version(a, "rows", 1, 7) != v);
    const json u = panel_data_unchanged(a, "rows", v);
    CHECK(u["unchanged"] == true);
    CHECK(u["primitive"] == "table");
}
