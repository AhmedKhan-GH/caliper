// Validation: the fixtures, shape propagation through `from` and transforms,
// the slot rules, the layout, the warnings and every error code.
#include <ostream>
#include <string>

#include <doctest/doctest.h>

#include <caliper_builder/spec/patch.h>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

// A spec around a body of sources and panels, with an instrument table.
std::string spec_text(const std::string& body) { return "caliper_spec = 2\n[instrument]\nid = \"t\"\n" + body; }

cs::ValidationReport check(const std::string& body) { return cs::check_toml(spec_text(body), fixture_resolver()); }

// The same, with root keys (such as `layout = "p"`) before the first table.
cs::ValidationReport check_root(const std::string& root, const std::string& body) {
    return cs::check_toml("caliper_spec = 2\n" + root + "\n[instrument]\nid = \"t\"\n" + body, fixture_resolver());
}

// A tabs node nested `levels` times around the panel p.
std::string nested_tabs(int levels) {
    std::string node = "\"p\"";
    for (int i = 0; i < levels; ++i) node = "{ tabs = [" + node + "] }";
    return node;
}

const char* kMetrics = "[[source]]\nid = \"m\"\ncatalog = \"table:run42_metrics\"\n";
const char* kAttn = "[[source]]\nid = \"a\"\ncatalog = \"tensor:attn_weights\"\n";  // f16 [12, 12, 64, 64]

std::string panel(const std::string& id, const std::string& primitive, const std::string& bind,
                  const std::string& options = "") {
    std::string s = "[[panel]]\nid = \"" + id + "\"\nprimitive = \"" + primitive + "\"\n";
    if (!bind.empty()) s += "bind = { " + bind + " }\n";
    if (!options.empty()) s += "options = { " + options + " }\n";
    return s;
}

const cs::SourceInfo& info(const cs::ValidationReport& r, const std::string& id) {
    for (const auto& s : r.sources)
        if (s.id == id) return s;
    static cs::SourceInfo none;
    return none;
}

}  // namespace

TEST_CASE("every valid fixture validates with and without a resolver") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto bare = cs::check_toml(fixture(name));
        CHECK_MESSAGE(bare.diagnostics.empty(), dump(bare.diagnostics));
        auto resolved = cs::check_toml(fixture(name), fixture_resolver());
        CHECK_MESSAGE(resolved.diagnostics.empty(), dump(resolved.diagnostics));
    }
}

TEST_CASE("shapes propagate through from and the transforms") {
    auto e = cs::check_toml(fixture("attention-entropy.spec.toml"), fixture_resolver());
    REQUIRE(e.ok());
    CHECK(info(e, "attn").shape == std::optional<std::vector<std::int64_t>>({12, 12, 64, 64}));
    CHECK(info(e, "entropy").kind == cs::DataKind::tensor);
    CHECK(info(e, "entropy").shape == std::optional<std::vector<std::int64_t>>({12, 12}));
    CHECK(info(e, "entropy").dtype == "f32");
    CHECK(info(e, "entropy").wire == std::optional<bool>(true));
    CHECK(info(e, "loss").kind == cs::DataKind::tabular);
    CHECK_FALSE(info(e, "loss").columns.has_value());

    auto d = cs::check_toml(fixture("dashboard.spec.toml"), fixture_resolver());
    REQUIRE(d.ok());
    CHECK(info(d, "attn").shape == std::optional<std::vector<std::int64_t>>({64, 64}));
    CHECK(info(d, "attn").dtype == "f32");  // f16 reduced to f32
    CHECK(info(d, "metrics").columns->size() == 5);

    auto p = cs::check_toml(fixture("every-primitive.spec.toml"), fixture_resolver());
    REQUIRE(p.ok());
    // acts [32, 128, 768] f64: slice 0:-1:2 of 32 gives 16, then the norm over axes 0 and -1.
    CHECK(info(p, "acts").shape == std::optional<std::vector<std::int64_t>>({128}));
    CHECK(info(p, "acts").dtype == "f64");
    CHECK(info(p, "probs").shape == std::optional<std::vector<std::int64_t>>({128}));

    const json j = cs::to_json(info(e, "entropy"));
    CHECK(j == json::parse(R"({"id":"entropy","kind":"tensor","columns":null,"dtype":"f32","shape":[12,12],"wire":true})"));
}

TEST_CASE("each transform's result shape follows the contracts") {
    auto shape_of = [](const std::string& transforms) {
        auto r = check(std::string(kAttn) + transforms + panel("p", "histogram", "values = \"a\""));
        REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
        return info(r, "a");
    };
    using Shape = std::optional<std::vector<std::int64_t>>;
    CHECK(shape_of("  [[source.transform]]\n  op = \"slice\"\n  axis = -1\n  start = -5\n  step = 2\n").shape ==
          Shape({12, 12, 64, 3}));
    CHECK(shape_of("  [[source.transform]]\n  op = \"slice\"\n  axis = 1\n  start = 10\n  stop = 2\n").shape ==
          Shape({12, 0, 64, 64}));
    CHECK(shape_of("  [[source.transform]]\n  op = \"slice\"\n  axis = 1\n  index = -1\n").shape ==
          Shape({12, 64, 64}));
    auto kept = shape_of("  [[source.transform]]\n  op = \"reduce\"\n  axis = [1, 3]\n  fn = \"max\"\n  keepdims = true\n");
    CHECK(kept.shape == Shape({12, 1, 64, 1}));
    CHECK(kept.dtype == "f32");
    auto arg = shape_of("  [[source.transform]]\n  op = \"reduce\"\n  axis = 0\n  fn = \"argmax\"\n");
    CHECK(arg.shape == Shape({12, 64, 64}));
    CHECK(arg.dtype == "i64");
    auto norm = shape_of("  [[source.transform]]\n  op = \"normalise\"\n  mode = \"zscore\"\n  axis = 2\n");
    CHECK(norm.shape == Shape({12, 12, 64, 64}));
    CHECK(norm.dtype == "f32");
}

TEST_CASE("transforms that do not fit the shape are refused") {
    auto t = [](const std::string& transform) { return check(std::string(kAttn) + transform); };
    CHECK(has(t("  [[source.transform]]\n  op = \"slice\"\n  axis = 1\n  index = 12\n").diagnostics,
              "spec.bad_transform", "source[0].transform[0].index"));
    CHECK(has(t("  [[source.transform]]\n  op = \"reduce\"\n  axis = 4\n  fn = \"sum\"\n").diagnostics,
              "spec.bad_transform", "source[0].transform[0].axis"));
    CHECK(has(t("  [[source.transform]]\n  op = \"reduce\"\n  axis = [1, -3]\n  fn = \"sum\"\n").diagnostics,
              "spec.bad_transform", "source[0].transform[0].axis[1]"));
    CHECK(has(t("  [[source.transform]]\n  op = \"reduce\"\n  axis = [1, 1]\n  fn = \"sum\"\n").diagnostics,
              "spec.bad_transform"));
    CHECK(has(t("  [[source.transform]]\n  op = \"reduce\"\n  axis = [0, 1]\n  fn = \"argmin\"\n").diagnostics,
              "spec.bad_transform", "source[0].transform[0].axis"));
    CHECK(has(t("  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n  index = 1\n  stop = 3\n").diagnostics,
              "spec.bad_transform", "source[0].transform[0]"));
    CHECK(has(t("  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n  step = 0\n").diagnostics,
              "spec.out_of_range", "source[0].transform[0].step"));
    CHECK(has(t("  [[source.transform]]\n  op = \"slice\"\n  axis = 9\n").diagnostics, "spec.out_of_range"));
    CHECK(has(t("  [[source.transform]]\n  op = \"normalise\"\n  mode = \"l2\"\n  axis = 5\n").diagnostics,
              "spec.bad_transform"));
    CHECK(has(t("  [[source.transform]]\n  op = \"reduce\"\n  axis = 0\n  fn = \"median\"\n").diagnostics,
              "spec.bad_value", "source[0].transform[0].fn"));
    CHECK(has(t("  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n  fn = \"sum\"\n").diagnostics,
              "spec.unknown_key", "source[0].transform[0].fn"));
    CHECK(has(t("  [[source.transform]]\n  op = \"normalise\"\n").diagnostics, "spec.missing_key",
              "source[0].transform[0].mode"));
    // On a tabular source, any transform is refused, once.
    auto tab = check(std::string(kMetrics) + "  [[source.transform]]\n  op = \"normalise\"\n  mode = \"l2\"\n"
                                             "  [[source.transform]]\n  op = \"normalise\"\n  mode = \"l2\"\n");
    CHECK(has(tab.diagnostics, "spec.bad_transform", "source[0].transform[0]"));
    CHECK_FALSE(has(tab.diagnostics, "spec.bad_transform", "source[0].transform[1]"));
    // Without a resolver the shape is not known, so only the shape-free rules apply.
    auto bare = cs::check_toml(spec_text(std::string(kAttn) + "  [[source.transform]]\n  op = \"slice\"\n  axis = 7\n"
                                                              "  index = 100\n"));
    CHECK_FALSE(has(bare.diagnostics, "spec.bad_transform"));
}

TEST_CASE("the slots of each primitive take the data they need") {
    const std::string m = kMetrics, a = kAttn;
    // series
    CHECK(check(m + panel("p", "series", "x = \"m.time\", y = [\"m.loss\", \"m.accuracy\"]")).ok());
    CHECK(has(check(m + panel("p", "series", "y = \"m.note\"")).diagnostics, "spec.slot_mismatch", "panel[0].bind.y"));
    CHECK(has(check(m + panel("p", "series", "y = \"m\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(m + a + panel("p", "series", "y = \"a\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(m + a + panel("p", "series", "x = \"m.step\", y = [\"m.loss\", \"a\"]")).diagnostics,
              "spec.slot_mismatch", "panel[0].bind.y[1]"));
    CHECK(has(check(m + panel("p", "series", "x = \"m.step\", y = \"m.lss\"")).diagnostics, "spec.unknown_column",
              "panel[0].bind.y"));
    CHECK(has(check(a + panel("p", "series", "y = \"a.loss\"")).diagnostics, "spec.bad_binding"));
    const std::string row = "[[source]]\nid = \"row\"\nfrom = \"a\"\n"
                            "  [[source.transform]]\n  op = \"reduce\"\n  axis = [0, 1, 2]\n  fn = \"mean\"\n";
    const std::string xs = "[[source]]\nid = \"xs\"\nfrom = \"a\"\n"
                           "  [[source.transform]]\n  op = \"reduce\"\n  axis = [0, 1, 3]\n  fn = \"mean\"\n"
                           "  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n  stop = 10\n";
    CHECK(check(a + row + panel("p", "series", "x = \"row\", y = \"row\"")).ok());
    CHECK(has(check(a + row + xs + panel("p", "series", "x = \"xs\", y = \"row\"")).diagnostics, "spec.slot_mismatch",
              "panel[0].bind.y"));
    CHECK(has(check(m + panel("p", "series", "x = \"m.step\", y = \"m.loss\"") + "[[source]]\nid = \"q\"\nsql = \"SELECT 1\"\n" +
                    panel("p2", "series", "x = \"m.step\", y = \"q.v\""))
                  .diagnostics,
              "spec.slot_mismatch", "panel[1].bind.y"));
    CHECK(has(check(m + panel("p", "heatmap", "values = [\"m.loss\"]")).diagnostics, "spec.wrong_type",
              "panel[0].bind.values"));
    CHECK(has(check(m + panel("p", "series", "y = []")).diagnostics, "spec.missing_slot", "panel[0].bind.y"));

    // heatmap and image
    CHECK(has(check(a + panel("p", "heatmap", "values = \"a\"")).diagnostics, "spec.slot_mismatch",
              "panel[0].bind.values"));
    CHECK(has(check(m + panel("p", "heatmap", "values = \"m.loss\"")).diagnostics, "spec.slot_mismatch"));
    const std::string img = "[[source]]\nid = \"i\"\ncatalog = \"tensor:sample_png\"\n";
    CHECK(check(img + panel("p", "image", "values = \"i\"")).ok());
    CHECK(has(check(a + panel("p", "image", "values = \"a\"")).diagnostics, "spec.slot_mismatch"));

    // points
    const std::string cloud = "[[source]]\nid = \"c\"\ncatalog = \"tensor:embedding\"\n"
                              "[[source]]\nid = \"l\"\ncatalog = \"tensor:embedding_labels\"\n";
    CHECK(check(cloud + panel("p", "points", "positions = \"c\", color = \"l\", size = \"l\"")).ok());
    CHECK(has(check(cloud + panel("p", "points", "positions = \"c\", x = \"l\"")).diagnostics, "spec.slot_mismatch",
              "panel[0].bind.positions"));
    CHECK(has(check(cloud + panel("p", "points", "positions = \"l\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(cloud + row + panel("p", "points", "positions = \"c\", color = \"row\"") + a).diagnostics,
              "spec.slot_mismatch", "panel[0].bind.color"));
    CHECK(has(check(m + cloud + panel("p", "points", "x = \"m.step\", y = \"l\"")).diagnostics, "spec.slot_mismatch",
              "panel[0].bind.y"));
    CHECK(has(check(m + panel("p", "points", "x = \"m.step\"")).diagnostics, "spec.missing_slot",
              "panel[0].bind.y"));
    CHECK(check(m + panel("p", "points", "x = \"m.step\", y = \"m.loss\", z = \"m.accuracy\"")).ok());

    // histogram
    CHECK(check(a + panel("p", "histogram", "values = \"a\"")).ok());
    CHECK(has(check(m + panel("p", "histogram", "values = \"m.note\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(m + panel("p", "histogram", "values = \"m\"")).diagnostics, "spec.slot_mismatch"));

    // table
    CHECK(has(check(m + panel("p", "table", "rows = \"m.step\"")).diagnostics, "spec.slot_mismatch",
              "panel[0].bind.rows"));
    CHECK(has(check(a + panel("p", "table", "rows = \"a\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(m + panel("p", "table", "rows = \"m\"", "columns = [\"step\", \"stp\"]")).diagnostics,
              "spec.unknown_column", "panel[0].options.columns[1]"));
    CHECK(has(check(m + panel("p", "table", "rows = \"m\"", "columns = [\"step\", \"step\"]")).diagnostics,
              "spec.bad_value", "panel[0].options.columns[1]"));
    CHECK(has(check(m + panel("p", "table", "rows = \"m\"", "columns = []")).diagnostics, "spec.bad_value"));

    // text
    const std::string one = "[[source]]\nid = \"one\"\nfrom = \"a\"\n"
                            "  [[source.transform]]\n  op = \"reduce\"\n  axis = [0, 1, 2, 3]\n  fn = \"max\"\n"
                            "  keepdims = true\n";
    CHECK(check(a + one + panel("p", "text", "value = \"one\"")).ok());
    CHECK(check(m + panel("p", "text", "value = \"m.note\"")).ok());
    CHECK(has(check(a + panel("p", "text", "value = \"a\"")).diagnostics, "spec.slot_mismatch"));
    CHECK(has(check(panel("p", "text", "")).diagnostics, "spec.missing_slot", "panel[0].bind"));
}

TEST_CASE("options are checked for each primitive") {
    const std::string a = kAttn;
    CHECK(has(check(a + panel("p", "heatmap", "values = \"a\"", "bins = 3")).diagnostics, "spec.unknown_option",
              "panel[0].options.bins"));
    CHECK(has(check(a + panel("p", "histogram", "values = \"a\"", "bins = 0")).diagnostics, "spec.out_of_range",
              "panel[0].options.bins"));
    CHECK(has(check(a + panel("p", "histogram", "values = \"a\"", "range = [1, 1]")).diagnostics, "spec.out_of_range",
              "panel[0].options.range"));
    CHECK(has(check(a + panel("p", "histogram", "values = \"a\"", "range = [1]")).diagnostics, "spec.wrong_type"));
    CHECK(has(check(a + panel("p", "histogram", "values = \"a\"", "x_label = \"a\\nb\"")).diagnostics,
              "spec.bad_value", "panel[0].options.x_label"));
    CHECK(has(check(a + panel("p", "image", "values = \"a\"", "colormap = \"jet\"")).diagnostics, "spec.bad_value",
              "panel[0].options.colormap"));
    CHECK(has(check(a + panel("p", "histogram", "values = \"a\"", "x_label = \"" + std::string(300, 'l') + "\"")).diagnostics,
              "spec.string_too_long", "panel[0].options.x_label"));
}

TEST_CASE("the layout places each panel once") {
    const std::string body = std::string(kAttn) + panel("p", "histogram", "values = \"a\"") +
                             panel("q", "histogram", "values = \"a\"");
    CHECK(check(body + "[layout]\ntabs = [\"p\", \"q\"]\n").ok());
    CHECK(check_root("layout = { split = \"vertical\", children = [\"p\", \"q\"] }", body).ok());
    auto unplaced = check_root("layout = \"p\"", body);
    CHECK(unplaced.ok());
    CHECK(has(unplaced.diagnostics, "spec.unplaced_panel", "panel[1]"));
    CHECK(has(check(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\", \"p\"]\n").diagnostics,
              "spec.duplicate_placement", "layout.children[1]"));
    CHECK(has(check(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\", \"r\"]\n").diagnostics,
              "spec.unknown_panel", "layout.children[1]"));
    CHECK(has(check(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\"]\n").diagnostics, "spec.bad_layout",
              "layout.children"));
    CHECK(has(check(body + "[layout]\ntabs = []\n").diagnostics, "spec.bad_layout", "layout.tabs"));
    CHECK(has(check(body + "[layout]\nsplit = \"vertical\"\nratio = 0.5\nchildren = [\"p\", \"q\", { tabs = [\"p\"] }]\n")
                  .diagnostics,
              "spec.bad_layout", "layout.ratio"));
    CHECK(has(check(body + "[layout]\nsplit = \"vertical\"\nratio = 1.5\nchildren = [\"p\", \"q\"]\n").diagnostics,
              "spec.out_of_range", "layout.ratio"));
    CHECK(has(check(body + "[layout]\nsplit = \"diagonal\"\nchildren = [\"p\", \"q\"]\n").diagnostics,
              "spec.bad_value", "layout.split"));
    CHECK(has(check(body + "[layout]\ntabs = [\"p\", \"q\"]\nratio = 0.5\n").diagnostics, "spec.unknown_key",
              "layout.ratio"));
    // A path of eight nodes from the root to a panel is the deepest layout.
    CHECK(check_root("layout = " + nested_tabs(7), body).ok());
    CHECK(check(body + "[layout]\ntabs = [" + nested_tabs(6) + "]\n").ok());
    auto too_deep = check(body + "[layout]\ntabs = [" + nested_tabs(7) + "]\n");
    CHECK(has(too_deep.diagnostics, "spec.too_deep"));
}

TEST_CASE("warnings leave a spec valid") {
    cs::SourceResolver offline = [](const std::string& id) -> std::optional<cs::ResolvedSource> {
        if (id != "live:attn.weights") return std::nullopt;
        cs::ResolvedSource r;
        r.kind = "live";
        r.dtype = "f32";
        r.shape = std::vector<std::int64_t>{12, 12, 64, 64};
        r.wire = false;
        return r;
    };
    auto r = cs::check_toml(fixture("attention-entropy.spec.toml"), offline);
    CHECK(r.ok());
    CHECK(has(r.diagnostics, "spec.not_portable", "source[0].catalog"));
    const cs::Diagnostic* d = find(r.diagnostics, "spec.not_portable");
    REQUIRE(d);
    CHECK(d->severity == cs::Severity::warning);
    CHECK(d->line == 10);  // located in the text

    auto unknown = check(std::string("[[source]]\nid = \"u\"\ncatalog = \"tensor:not_loaded_yet\"\n") +
                         panel("p", "histogram", "values = \"u\""));
    CHECK(unknown.ok());
    CHECK(has(unknown.diagnostics, "spec.unresolved_source", "source[0].catalog"));

    auto unused = check(std::string(kAttn) + kMetrics + panel("p", "histogram", "values = \"a\""));
    CHECK(unused.ok());
    CHECK(has(unused.diagnostics, "spec.unused_source", "source[1]"));
    CHECK_FALSE(has(unused.diagnostics, "spec.unused_source", "source[0]"));
}

TEST_CASE("a resolver that throws is reported, and one that contradicts the id is not trusted") {
    cs::SourceResolver throwing = [](const std::string&) -> std::optional<cs::ResolvedSource> {
        throw std::runtime_error("catalog offline");
    };
    auto r = cs::check_toml(fixture("dashboard.spec.toml"), throwing);
    CHECK(has(r.diagnostics, "spec.internal", "source[0].catalog"));

    cs::SourceResolver liar = [](const std::string&) -> std::optional<cs::ResolvedSource> {
        cs::ResolvedSource s;
        s.kind = "table";  // the id says tensor
        s.columns = std::vector<cs::ResolvedColumn>{{"a", "INTEGER"}};
        s.wire = true;
        return s;
    };
    auto l = cs::check_toml(spec_text(std::string(kAttn) + panel("p", "histogram", "values = \"a\"")), liar);
    CHECK(l.ok());
    CHECK(info(l, "a").kind == cs::DataKind::tensor);
    CHECK_FALSE(info(l, "a").columns.has_value());
}

TEST_CASE("validation checks a spec built in code") {
    cs::Spec s;
    s.instrument.id = "built";
    s.instrument.title = std::string("bad \xC3 byte");
    cs::Source src;
    src.id = "a";
    src.catalog = "tensor:a";
    cs::Transform t;
    t.op = cs::TransformOp::slice;
    t.axis = {0};
    t.fn = "sum";  // not a key of a slice
    src.transforms.push_back(t);
    s.sources.push_back(src);
    cs::Panel p;
    p.id = "p";
    p.primitive = cs::Primitive::heatmap;
    p.bind["values"] = cs::SlotValue{{"a", "a"}, true};
    p.options.bins = 4;
    s.panels.push_back(p);
    cs::LayoutNode root;
    root.kind = cs::LayoutNode::Kind::split;
    root.ratio = std::nan("");
    root.children = {cs::LayoutNode::of_panel("p"), cs::LayoutNode::of_panel("p")};
    root.children[0].title = "a panel node has no title";
    s.layout = root;
    s.x = json::array();
    auto r = cs::validate(s);
    CHECK(has(r.diagnostics, "spec.bad_value", "instrument.title"));
    CHECK(has(r.diagnostics, "spec.unknown_key", "source[0].transform[0].fn"));
    CHECK(has(r.diagnostics, "spec.wrong_type", "panel[0].bind.values"));
    CHECK(has(r.diagnostics, "spec.unknown_option", "panel[0].options.bins"));
    CHECK(has(r.diagnostics, "spec.out_of_range", "layout.ratio"));
    CHECK(has(r.diagnostics, "spec.bad_layout", "layout.children[0]"));
    CHECK(has(r.diagnostics, "spec.wrong_type", "x"));

    cs::Spec later;
    later.version = 3;
    auto v = cs::validate(later);
    REQUIRE(v.diagnostics.size() == 1);
    CHECK(v.diagnostics[0].code == "spec.unsupported_version");
}

TEST_CASE("validation errors of a TOML text point into the text") {
    auto r = check(std::string(kAttn) + "\n\n" + panel("p", "heatmap", "values = \"nope\""));
    const cs::Diagnostic* d = find(r.diagnostics, "spec.unknown_source");
    REQUIRE(d);
    CHECK(d->path == "panel[0].bind.values");
    CHECK(d->line == 12);
    CHECK(cs::to_json(*d) == json{{"severity", "error"},
                                  {"code", "spec.unknown_source"},
                                  {"message", d->message},
                                  {"path", "panel[0].bind.values"},
                                  {"line", 12},
                                  {"column", d->column}});
}

TEST_CASE("every documented code is reported at least once") {
    std::set<std::string> seen;
    auto note = [&seen](const std::vector<cs::Diagnostic>& ds) {
        for (const auto& d : ds) seen.insert(d.code);
    };
    const std::string a = kAttn, m = kMetrics;
    const std::string hist = panel("p", "histogram", "values = \"a\"");
    const std::vector<std::string> bodies = {
        "title = 3\n",                                                       // wrong_type
        "titel = \"x\"\n",                                                   // unknown_key
        "title = \"" + std::string(300, 't') + "\"\n",                       // string_too_long
        a + panel("p", "pie", ""),                                           // unknown_primitive
        a + "  [[source.transform]]\n  op = \"transpose\"\n",                // unknown_transform
        a + panel("p", "heatmap", "y = \"a\""),                              // unknown_slot
        a + panel("p", "heatmap", "values = \"a\"", "bins = 2"),             // unknown_option
        a + "[[panel]]\nid = \"p\"\n",                                       // missing_key
        a + hist + panel("p", "histogram", "values = \"a\""),                // duplicate_id
        "[[source]]\nid = \"s\"\ncatalog = \"tensor:a\"\nsql = \"SELECT 1\"\n",  // source_form
        "[[source]]\nid = \"s\"\nfrom = \"nope\"\n",                         // unknown_source
        "[[source]]\nid = \"s\"\nfrom = \"r\"\n[[source]]\nid = \"r\"\nfrom = \"s\"\n",  // cycle
        a + panel("p", "histogram", "values = \"a.b\""),                     // bad_binding
        m + panel("p", "histogram", "values = \"m.lss\""),                   // unknown_column
        a + panel("p", "heatmap", ""),                                       // missing_slot
        a + panel("p", "heatmap", "values = \"a\""),                         // slot_mismatch
        m + "  [[source.transform]]\n  op = \"normalise\"\n  mode = \"l2\"\n",  // bad_transform
        a + hist + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\"]\n",   // bad_layout
        a + hist + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\", \"z\"]\n",  // unknown_panel
        a + hist + "[layout]\nsplit = \"vertical\"\nchildren = [\"p\", \"p\"]\n",  // duplicate_placement
        a + hist + "[layout]\nsplit = \"vertical\"\nratio = 2.0\nchildren = [\"p\", \"p\"]\n",  // out_of_range
        a + panel("p", "histogram", "values = \"a\"", "x_label = \"\\u0001\""),  // bad_value
        "[[source]]\nid = \"u\"\ncatalog = \"tensor:unknown\"\n" + panel("p", "histogram", "values = \"u\""),
        a + m + hist + "[layout]\ntabs = [\"p\"]\n" + panel("q", "histogram", "values = \"a\""),  // unused, unplaced
    };
    for (const std::string& body : bodies) note(check(body).diagnostics);
    note(cs::check_toml("caliper_spec = = 2").diagnostics);                       // syntax
    note(cs::check_toml(std::string(cs::limits::kTextBytes + 1, '#')).diagnostics);  // too_large
    note(cs::check_toml("caliper_spec = 3\n").diagnostics);                       // unsupported_version
    {
        cs::Spec s;
        s.instrument.id = "Bad";
        note(cs::validate(s).diagnostics);  // bad_id
    }
    {
        std::string many;
        for (int i = 0; i < 65; ++i) many += panel("p" + std::to_string(i), "text", "", "text = \"t\"");
        note(check(many).diagnostics);  // too_many
    }
    note(check(a + hist + "[layout]\ntabs = [" + nested_tabs(7) + "]\n").diagnostics);  // too_deep
    {
        cs::SourceResolver no_wire = [](const std::string&) -> std::optional<cs::ResolvedSource> {
            cs::ResolvedSource r;
            r.kind = "tensor";
            r.wire = false;
            return r;
        };
        note(cs::check_toml(spec_text(a + hist), no_wire).diagnostics);  // not_portable
        cs::SourceResolver throwing = [](const std::string&) -> std::optional<cs::ResolvedSource> {
            throw std::runtime_error("offline");
        };
        note(cs::check_toml(spec_text(a + hist), throwing).diagnostics);  // internal
    }
    {
        auto base = cs::parse_toml(spec_text(a + hist));
        REQUIRE(base.ok());
        note(cs::patch_panel(base.spec, "nope", json::object()).diagnostics);           // not_found
        note(cs::patch_panel(base.spec, "p", json{{"id", "renamed"}}).diagnostics);    // bad_patch
    }

    const std::vector<std::string> documented = {
        "spec.too_large",       "spec.syntax",          "spec.too_deep",          "spec.too_many",
        "spec.string_too_long", "spec.wrong_type",      "spec.out_of_range",      "spec.bad_value",
        "spec.unknown_key",     "spec.missing_key",     "spec.unsupported_version", "spec.unknown_primitive",
        "spec.unknown_transform", "spec.unknown_slot",  "spec.unknown_option",    "spec.bad_id",
        "spec.duplicate_id",    "spec.source_form",     "spec.unknown_source",    "spec.cycle",
        "spec.bad_binding",     "spec.unknown_column",  "spec.missing_slot",      "spec.slot_mismatch",
        "spec.bad_transform",   "spec.bad_layout",      "spec.unknown_panel",     "spec.duplicate_placement",
        "spec.not_found",       "spec.bad_patch",       "spec.internal",          "spec.not_portable",
        "spec.unresolved_source", "spec.unused_source", "spec.unplaced_panel"};
    for (const std::string& code : documented) {
        CAPTURE(code);
        CHECK(seen.count(code) == 1);
    }
    // And nothing undocumented.
    for (const std::string& code : seen) {
        CAPTURE(code);
        CHECK(std::find(documented.begin(), documented.end(), code) != documented.end());
    }
}

TEST_CASE("a cycle is reported once, and sources fed by it are unknown") {
    auto r = check("[[source]]\nid = \"a\"\nfrom = \"b\"\n[[source]]\nid = \"b\"\nfrom = \"c\"\n"
                   "[[source]]\nid = \"c\"\nfrom = \"b\"\n[[source]]\nid = \"self\"\nfrom = \"self\"\n" +
                   panel("p", "histogram", "values = \"a\"") + panel("q", "histogram", "values = \"self\""));
    int cycles = 0;
    for (const auto& d : r.diagnostics) cycles += d.code == "spec.cycle";
    CHECK(cycles == 2);
    CHECK(info(r, "a").kind == cs::DataKind::unknown);
    // A long chain of from resolves without recursion.
    std::string chain = std::string(kAttn);
    std::string prev = "a";
    for (int i = 0; i < 60; ++i) {
        const std::string id = "s" + std::to_string(i);
        chain += "[[source]]\nid = \"" + id + "\"\nfrom = \"" + prev + "\"\n";
        prev = id;
    }
    auto c = check(chain + panel("p", "histogram", "values = \"" + prev + "\""));
    CHECK_MESSAGE(c.ok(), dump(c.diagnostics));
    CHECK(info(c, prev).shape == std::optional<std::vector<std::int64_t>>({12, 12, 64, 64}));
    CHECK_FALSE(has(c.diagnostics, "spec.unused_source"));
}

TEST_CASE("a full report keeps its errors before its warnings") {
    // 300 sources the catalog does not know: one too_many error, then a
    // warning for each, which fills the report. The panels' errors come after.
    std::string body;
    for (int i = 0; i < 300; ++i)
        body += "[[source]]\nid = \"s" + std::to_string(i) + "\"\ncatalog = \"tensor:missing" + std::to_string(i) +
                "\"\n";
    for (int i = 0; i < 10; ++i) body += panel("p" + std::to_string(i), "histogram", "values = \"nope\"");
    auto r = check(body);
    CHECK(r.diagnostics.size() == cs::limits::kDiagnostics);
    CHECK_FALSE(r.ok());
    int errors = 0;
    for (const auto& d : r.diagnostics) errors += d.is_error();
    CHECK(errors == 11);
    CHECK(has(r.diagnostics, "spec.unknown_source", "panel[9].bind.values"));
}

// ---- scene panels and controls (design/instruments.md) ----------------------

namespace {

// A version 2 spec around a body of panels, controls and a layout.
cs::ValidationReport check_v2(const std::string& body, const cs::SourceResolver& resolver = fixture_resolver()) {
    return cs::check_toml("caliper_spec = 2\n[instrument]\nid = \"t\"\n" + body, resolver);
}

std::string control(const std::string& id, const std::string& kind, const std::string& bind,
                    const std::string& options = "") {
    std::string s = "[[control]]\nid = \"" + id + "\"\nkind = \"" + kind + "\"\nbind = \"" + bind + "\"\n";
    if (!options.empty()) s += "options = { " + options + " }\n";
    return s;
}

const char* kScene = "[[panel]]\nid = \"w\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n";

}  // namespace

TEST_CASE("the instruments fixture is valid, and a scene the script does not send is a warning") {
    auto known = cs::check_toml(fixture("instruments.spec.toml"), [](const std::string& id) {
        std::optional<cs::ResolvedSource> r;
        if (id == "live:model") r = cs::ResolvedSource{"live", std::nullopt, "", std::nullopt, false};
        return r;
    });
    CHECK_MESSAGE(known.ok(), dump(known.diagnostics));
    CHECK(known.diagnostics.empty());
    auto offline = cs::check_toml(fixture("instruments.spec.toml"), fixture_resolver());
    CHECK(offline.ok());
    CHECK(has(offline.diagnostics, "spec.unresolved_source", "panel[0].bind.scene"));
    CHECK(cs::check_toml(fixture("instruments.spec.toml")).diagnostics.empty());
}

TEST_CASE("the control error codes") {
    // spec.unknown_control_kind is the parser's.
    CHECK(has(check_v2(control("k", "knob", "value")).diagnostics, "spec.unknown_control_kind", "control[0].kind"));
    // Every kind is drawn, so none gives spec.unsupported_control any more.
    for (const char* kind : {"range", "picker", "text", "segmented", "transport", "readout"}) {
        CAPTURE(kind);
        CHECK_FALSE(has(check_v2(control("k", kind, "value")).diagnostics, "spec.unsupported_control",
                        "control[0].kind"));
    }
    CHECK(has(check_v2(control("k", "slider", "action:save")).diagnostics, "spec.control_mismatch",
              "control[0].bind"));
    CHECK(has(check_v2(control("k", "slider", "run:pause")).diagnostics, "spec.control_mismatch",
              "control[0].bind"));
    CHECK(has(check_v2(control("k", "button", "param:lr")).diagnostics, "spec.control_mismatch",
              "control[0].bind"));
    CHECK(has(check_v2(control("k", "button", "value")).diagnostics, "spec.control_mismatch", "control[0].bind"));
    CHECK(has(check_v2(control("k", "button", "run:pause", "min = 0")).diagnostics, "spec.control_mismatch",
              "control[0].options.min"));
    for (const std::string& ok : {control("k", "slider", "param:lr", "min = 0.001, max = 1, log = true, step = 0.01"),
                                  control("k", "slider", "value", "min = 0, max = 10, value = 3"),
                                  control("k", "button", "action:save"), control("k", "button", "run:pause")}) {
        CAPTURE(ok);
        auto r = check_v2(ok);
        CHECK_MESSAGE(r.ok(), dump(r.diagnostics));
    }
}

TEST_CASE("controls and scene panels use standard error codes for their other errors") {
    auto first = [](const cs::ValidationReport& r) { return dump(r.diagnostics); };
    for (const char* bind : {"param:", "intervene:", "run:step", "lr"}) {
        CAPTURE(bind);
        CHECK(has(check_v2(control("k", "slider", bind)).diagnostics, "spec.bad_binding", "control[0].bind"));
    }
    CHECK(has(check_v2(control("k", "slider", "param:" + std::string(64, 'a'))).diagnostics, "spec.bad_binding",
              "control[0].bind"));
    auto scene_bind = [&](const std::string& b) {
        return check_v2("[[panel]]\nid = \"w\"\nprimitive = \"scene\"\nbind = { scene = \"" + b + "\" }\n");
    };
    CHECK(has(scene_bind("model").diagnostics, "spec.bad_binding", "panel[0].bind.scene"));
    CHECK(has(scene_bind("live:").diagnostics, "spec.bad_binding", "panel[0].bind.scene"));
    CHECK(has(scene_bind("tensor:model").diagnostics, "spec.bad_binding", "panel[0].bind.scene"));
    CHECK(has(check_v2("[[panel]]\nid = \"w\"\nprimitive = \"scene\"\n").diagnostics, "spec.missing_slot",
              "panel[0].bind.scene"));
    // A scene panel where an instance of a piece draws needs no scene: it holds its programs alone.
    const std::string alone =
        "[[panel]]\nid = \"w\"\nprimitive = \"scene\"\n"
        "[[program]]\nid = \"arcs\"\npanel = \"w\"\nbind = { attn = \"live:attn\" }\n"
        "piece = { name = \"arcs\", version = 1, from = \"workspace\", hash = \"sha256:" + std::string(64, 'a') + "\" }\n";
    CHECK_MESSAGE(!has(check_v2(alone).diagnostics, "spec.missing_slot", "panel[0].bind.scene"),
                  first(check_v2(alone)));
    CHECK(has(check_v2(std::string(kScene) + control("w", "button", "run:pause")).diagnostics, "spec.duplicate_id",
              "control[0].id"));
    CHECK(has(check_v2(control("k", "button", "run:pause") + control("k", "button", "action:a")).diagnostics,
              "spec.duplicate_id", "control[1].id"));
    CHECK(has(check_v2(control("Bad", "button", "run:pause")).diagnostics, "spec.bad_id", "control[0].id"));
    CHECK(has(check_v2(control("k", "slider", "value", "min = 1, max = 1")).diagnostics, "spec.out_of_range",
              "control[0].options.min"));
    CHECK(has(check_v2(control("k", "slider", "value", "step = 0")).diagnostics, "spec.out_of_range",
              "control[0].options.step"));
    CHECK(has(check_v2(control("k", "slider", "param:lr", "log = true, min = 0")).diagnostics, "spec.out_of_range",
              "control[0].options.min"));
    CHECK(has(check_v2(control("k", "slider", "value", "min = 0, max = 1e308")).diagnostics, "spec.out_of_range",
              "control[0].options.max"));
    CHECK(has(check_v2(control("k", "slider", "value", "value = -2e15")).diagnostics, "spec.out_of_range",
              "control[0].options.value"));
    CHECK(has(check_v2(control("k", "slider", "value", "step = 1e300")).diagnostics, "spec.out_of_range",
              "control[0].options.step"));
    std::string many;
    for (int i = 0; i < 65; ++i) many += control("c" + std::to_string(i), "button", "run:pause");
    CHECK_MESSAGE(has(check_v2(many).diagnostics, "spec.too_many", "control"), first(check_v2(many)));
    std::string patterns;
    for (int i = 0; i < 65; ++i) patterns += std::string(i ? ", " : "") + "\"m" + std::to_string(i) + "\"";
    CHECK(has(check_v2(std::string(kScene) + "options = { include = [" + patterns + "] }\n").diagnostics,
              "spec.too_many", "panel[0].options.include"));
    CHECK(has(check_v2(std::string(kScene) + "options = { view_camera = { elevation = 90 } }\n").diagnostics,
              "spec.out_of_range", "panel[0].options.view_camera.elevation"));
    CHECK(has(check_v2(std::string(kScene) + "options = { view_camera = { distance = -1 } }\n").diagnostics,
              "spec.out_of_range", "panel[0].options.view_camera.distance"));
    CHECK(has(check_v2(std::string(kScene) + "options = { exclude = [\"\"] }\n").diagnostics, "spec.bad_value",
              "panel[0].options.exclude[0]"));
}

TEST_CASE("a layout leaf names a panel or a control") {
    const std::string body = std::string(kScene) + control("lr", "slider", "param:lr") + control("go", "button", "run:pause");
    auto placed = check_v2(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"w\", { split = \"horizontal\", "
                                  "children = [\"lr\", \"go\"] }]\n");
    CHECK_MESSAGE(placed.ok(), dump(placed.diagnostics));
    CHECK_FALSE(has(placed.diagnostics, "spec.unplaced_panel"));
    auto unplaced = check_v2(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"w\", \"lr\"]\n");
    CHECK(unplaced.ok());
    CHECK(has(unplaced.diagnostics, "spec.unplaced_panel", "control[1]"));
    auto unknown = check_v2(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"w\", \"nothing\"]\n");
    CHECK(has(unknown.diagnostics, "spec.unknown_panel", "layout.children[1]"));
    auto twice = check_v2(body + "[layout]\nsplit = \"vertical\"\nchildren = [\"w\", \"lr\", \"go\", \"lr\"]\n");
    CHECK(has(twice.diagnostics, "spec.duplicate_placement", "layout.children[3]"));
    // Without a layout, nothing is unplaced.
    CHECK(check_v2(body, {}).diagnostics.empty());
}

TEST_CASE("a Spec with a retired or unsupported version fails validation with spec.unsupported_version") {
    auto parsed = cs::parse_toml(fixture("instruments.spec.toml"));
    REQUIRE(parsed.ok());
    for (std::int64_t bad_version : {0, 1, 3}) {
        CAPTURE(bad_version);
        cs::Spec s = parsed.spec;
        s.version = bad_version;
        auto r = cs::validate(s);
        REQUIRE(r.diagnostics.size() == 1);
        CHECK(r.diagnostics[0].code == "spec.unsupported_version");
    }
}
