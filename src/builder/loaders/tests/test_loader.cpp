// Tests of the loader format (parsing, validation, the canonical writer), of
// path confinement and globs, and of reader inference.
#include <caliper_builder/loaders/inference.h>
#include <caliper_builder/loaders/loader.h>
#include <caliper_builder/loaders/paths.h>

#include <doctest/doctest.h>

#include <filesystem>

#include "test_formats.h"

using namespace caliper_builder;
using namespace loaders_test;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const char* kTabular = R"(caliper_loader = 0

[loader]
id = "run42-metrics"
title = "Run 42 metrics"

[input]
path = "data/run42/*.parquet"
reader = "parquet"

[reader]
union_by_name = true

[[output]]
name = "run42_metrics"
mode = "view"
sql = "SELECT step, loss FROM input WHERE step > 0"

[[output]]
name = "run42_all"
mode = "table"
)";

const char* kTensor = R"(caliper_loader = 0

[loader]
id = "run42-weights"
description = "Attention weights."

[input]
path = "checkpoints/run42.safetensors"

[[output]]
name = "attn_entropy"
tensor = "layers.0.attn.weights"
  [[output.transform]]
  op = "reduce"
  axis = -1
  fn = "entropy"
  [[output.transform]]
  op = "slice"
  axis = 0
  start = 1
  stop = -1

[[output]]
name = "w_"
tensor = "*"
)";

Error parse_error(const std::string& text) { return parse_loader(text).error; }

}  // namespace

TEST_CASE("a tabular recipe parses into its fields") {
    LoaderParse p = parse_loader(kTabular);
    REQUIRE_MESSAGE(p.error.ok(), p.error.code << " " << p.error.where << ": " << p.error.message);
    const Loader& l = p.loader;
    CHECK(l.id == "run42-metrics");
    CHECK(l.title == "Run 42 metrics");
    CHECK(l.path == "data/run42/*.parquet");
    REQUIRE(l.reader);
    CHECK(*l.reader == ReaderKind::parquet);
    CHECK(l.reader_options == json{{"union_by_name", true}});
    REQUIRE(l.outputs.size() == 2);
    CHECK(l.outputs[0].name == "run42_metrics");
    CHECK(*l.outputs[0].mode == "view");
    CHECK(*l.outputs[0].sql == "SELECT step, loss FROM input WHERE step > 0");
    CHECK(*l.outputs[1].mode == "table");
    CHECK_FALSE(l.outputs[1].sql);
}

TEST_CASE("a tensor recipe parses its transforms") {
    LoaderParse p = parse_loader(kTensor);
    REQUIRE_MESSAGE(p.error.ok(), p.error.code << " " << p.error.where << ": " << p.error.message);
    const Loader& l = p.loader;
    CHECK_FALSE(l.reader);
    CHECK(*declared_reader(l) == ReaderKind::safetensors);
    CHECK(l.description == "Attention weights.");
    REQUIRE(l.outputs[0].transforms.size() == 2);
    CHECK(l.outputs[0].transforms[0].op == TransformOp::reduce);
    CHECK(l.outputs[0].transforms[0].fn == ReduceFn::entropy);
    CHECK(l.outputs[0].transforms[0].axes == std::vector<int64_t>{-1});
    CHECK(*l.outputs[0].transforms[1].stop == -1);
    CHECK(*l.outputs[1].tensor == "*");
}

TEST_CASE("the canonical writer round-trips") {
    for (const char* text : {kTabular, kTensor}) {
        LoaderParse a = parse_loader(text);
        REQUIRE(a.error.ok());
        const std::string written = write_loader(a.loader);
        LoaderParse b = parse_loader(written);
        REQUIRE_MESSAGE(b.error.ok(), b.error.message << "\n" << written);
        CHECK(a.loader.to_json() == b.loader.to_json());
        CHECK(write_loader(b.loader) == written);
        LoaderParse c = loader_from_json(a.loader.to_json());
        REQUIRE(c.error.ok());
        CHECK(c.loader.to_json() == a.loader.to_json());
    }
    Loader l;
    l.id = "odd";
    l.title = "Quotes \" and \\ backslashes";
    l.description = "Two\nlines\twith a tab";
    l.path = "data/x \"y\".csv";
    l.reader_options = {{"delimiter", "\t"}, {"null", "NA"}, {"skip", 2}, {"header", false}};
    LoaderOutput o;
    o.name = "x";
    o.sql = "SELECT *\nFROM input -- all\n";
    l.outputs.push_back(o);
    LoaderParse p = parse_loader(write_loader(l));
    REQUIRE_MESSAGE(p.error.ok(), p.error.message << "\n" << write_loader(l));
    CHECK(p.loader.to_json() == l.to_json());
}

TEST_CASE("parse errors carry the line and column and name the key") {
    Error e = parse_error("caliper_loader = 0\n\n[loader]\nid = \"a\"\ncolour = \"red\"\n");
    CHECK(e.code == "loader.unknown_key");
    CHECK(e.where == "5:1");
    CHECK(e.message.find("loader.colour") != std::string::npos);

    e = parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.csv\"\n[[output]]\nname = \"t\"\nmode = \"copy\"\n");
    CHECK(e.code == "loader.bad_value");
    CHECK(e.where == "8:1");
    CHECK(e.message.find("output[0].mode") != std::string::npos);

    e = parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.npy\"\n[[output]]\nname = \"t\"\n"
                    "[[output.transform]]\nop = \"reduce\"\naxis = 0\nfn = \"median\"\n");
    CHECK(e.code == "transform.bad_value");
    CHECK(e.where == "11:1");
    CHECK(e.message.find("output[0].transform[0].fn") != std::string::npos);

    e = parse_error("caliper_loader = 0\n[loader\nid = \"a\"\n");
    CHECK(e.code == "loader.parse");
    CHECK(e.where.rfind("2:", 0) == 0);
}

TEST_CASE("the loader format refuses what it does not define") {
    const std::string head = "caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.csv\"\n";
    CHECK(parse_error("[loader]\nid = \"a\"\n").code == "loader.missing_key");
    CHECK(parse_error("caliper_loader = 1\n").code == "loader.unsupported");
    CHECK(parse_error("caliper_loader = \"0\"\n").code == "loader.unsupported");
    CHECK(parse_error(head).code == "loader.missing_key");  // no output
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"A\"\n[input]\npath = \"x.csv\"\n[[output]]\nname = \"t\"\n").code ==
          "loader.bad_id");
    CHECK(parse_error("caliper_loader = 0\n[loader]\ntitle = \"x\"\n").code == "loader.missing_key");
    CHECK(parse_error(head + "[[output]]\nname = \"Bad Name\"\n").code == "loader.bad_id");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\n[[output]]\nname = \"t\"\n").code == "loader.name_collision");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntensor = \"w\"\n").code == "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\n[[output.transform]]\nop = \"normalise\"\nmode = \"l2\"\n").code ==
          "loader.bad_value");
    CHECK(parse_error(head + "[reader]\ndelim = \",\"\n[[output]]\nname = \"t\"\n").code == "loader.unknown_key");
    CHECK(parse_error(head + "[reader]\nskip = -1\n[[output]]\nname = \"t\"\n").code == "loader.bad_value");
    CHECK(parse_error(head + "[reader]\nskip = [1]\n[[output]]\nname = \"t\"\n").code == "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ncreated = 2026-11-15\n").code == "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\nsql = 3\n").code == "loader.bad_value");
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x\"\nreader = \"excel\"\n"
                      "[[output]]\nname = \"t\"\n")
              .code == "loader.bad_value");
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"\"\n[[output]]\nname = \"t\"\n").code ==
          "loader.bad_value");
    // Tensor readers: no mode or sql; a single-tensor file takes no `tensor`; a multi-tensor file needs one.
    const std::string npy = "caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.npy\"\n";
    CHECK(parse_error(npy + "[[output]]\nname = \"t\"\nsql = \"SELECT 1\"\n").code == "loader.bad_value");
    CHECK(parse_error(npy + "[[output]]\nname = \"t\"\nmode = \"table\"\n").code == "loader.bad_value");
    CHECK(parse_error(npy + "[[output]]\nname = \"t\"\ntensor = \"x\"\n").code == "loader.bad_value");
    const std::string st = "caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.safetensors\"\n";
    CHECK(parse_error(st + "[[output]]\nname = \"t\"\n").code == "loader.missing_key");
    CHECK(parse_error(st + "[[output]]\nname = \"t\"\ntensor = \"\"\n").code == "loader.bad_value");
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.png\"\n[reader]\nchannels = 7\n"
                      "[[output]]\nname = \"t\"\n")
              .code == "loader.bad_value");
    // An undeclared reader (an unknown extension) defers those checks to the run.
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.bin\"\n[[output]]\nname = \"t\"\n"
                      "sql = \"SELECT 1\"\n")
              .ok());
    // The python reader needs its script and outputs that select a tensor or a table
    // (test_python_loader.cpp has the rest).
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.pt\"\nreader = \"python\"\n"
                      "[[output]]\nname = \"t\"\n")
              .code == "loader.missing_key");
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.pt\"\nreader = \"python\"\n"
                      "[reader]\nscript = \"s.py\"\n[[output]]\nname = \"t\"\ntensor = \"*\"\n")
              .ok());
    // Limits.
    CHECK(parse_error(std::string(kMaxLoaderText + 1, ' ')).code == "loader.too_large");
    std::string many = head;
    for (int i = 0; i < 257; ++i) many += "[[output]]\nname = \"t" + std::to_string(i) + "\"\n";
    CHECK(parse_error(many).code == "loader.too_large");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\nsql = \"" + std::string(kMaxLoaderSql + 1, 'x') + "\"\n").code ==
          "loader.too_large");
    std::string deep = "caliper_loader = 0\nx = ";
    for (int i = 0; i < 40; ++i) deep += "[";
    for (int i = 0; i < 40; ++i) deep += "]";
    CHECK_FALSE(parse_error(deep).ok());
    // JSON input reports key paths.
    LoaderParse j = loader_from_json(json::parse(R"({"caliper_loader": 0, "loader": {"id": "a"},
        "input": {"path": "x.csv"}, "output": [{"name": "t", "mode": "copy"}]})"));
    CHECK(j.error.code == "loader.bad_value");
    CHECK(j.error.where == "output[0].mode");
    CHECK(loader_from_json(json::array()).error.code == "loader.parse");
}

TEST_CASE("ids and source names") {
    CHECK(valid_id("a"));
    CHECK(valid_id("run42-metrics_x"));
    CHECK_FALSE(valid_id(""));
    CHECK_FALSE(valid_id("1a"));
    CHECK_FALSE(valid_id("A"));
    CHECK_FALSE(valid_id("a.b"));
    CHECK_FALSE(valid_id(std::string(65, 'a')));
    CHECK(valid_id(std::string(64, 'a')));
    CHECK(make_id("Run 42.v2") == "run_42_v2");
    CHECK(make_id("2024-data") == "f_2024-data");
    CHECK(make_id("2024-data", true) == "f_2024_data");
    CHECK(make_id("layers.0.attn.weight") == "layers_0_attn_weight");
    CHECK(make_id("") == "f_");
    const std::string long_name = make_id(std::string(100, 'x') + "A");
    CHECK(long_name.size() == 64);
    CHECK(valid_id(long_name));
    CHECK(long_name != make_id(std::string(100, 'x') + "B"));
    CHECK(long_name == make_id(std::string(100, 'x') + "A"));
}

TEST_CASE("glob matching and confinement") {
    CHECK(glob_match("*.csv", "a.csv"));
    CHECK_FALSE(glob_match("*.csv", "a.csvx"));
    CHECK(glob_match("a?c", "abc"));
    CHECK(glob_match("*", "abc"));
    CHECK_FALSE(glob_match("*", ".hidden"));
    CHECK(glob_match(".*", ".hidden"));
    CHECK(glob_match("*a*b*", "xxaxxbxx"));
    CHECK_FALSE(glob_match("*a*b", "xxaxxbxx"));
    CHECK(path_within("/a/b/c", "/a/b"));
    CHECK(path_within("/a/b", "/a/b"));
    CHECK(path_within("/a/b/c", "/a/b/"));
    CHECK_FALSE(path_within("/a/bc", "/a/b"));
    CHECK_FALSE(path_within("/a", "/a/b"));

    const std::string root = fresh_dir("paths");
    const std::string ws = root + "/ws";
    for (const char* f : {"a.csv", "b.csv", ".hidden.csv", "sub/c.csv", "deep/x/y/z.csv", "notes.txt"}) {
        write_text(ws + "/" + f, "x\n1\n");
    }
    write_text(root + "/outside/secret.csv", "s\n1\n");
    write_text(root + "/granted/g.csv", "g\n1\n");
    write_text(root + "/granted_file.csv", "g\n1\n");
    write_text(root + "/ws-sibling/s.csv", "s\n1\n");
    AllowedRoots roots{ws, {root + "/granted"}, {root + "/granted_file.csv"}};

    auto files = [&](const std::string& pattern, const GlobLimits& limits = {}) {
        ExpandResult r = expand_input(pattern, roots, limits);
        REQUIRE_MESSAGE(r.error.ok(), pattern << ": " << r.error.code << " " << r.error.message);
        std::vector<std::string> rel;
        for (const std::string& f : r.files) rel.push_back(display_path(f, ws));
        return rel;
    };
    auto code = [&](const std::string& pattern, const GlobLimits& limits = {}) {
        return expand_input(pattern, roots, limits).error.code;
    };
    CHECK(files("a.csv") == std::vector<std::string>{"a.csv"});
    CHECK(files("./sub/c.csv") == std::vector<std::string>{"sub/c.csv"});
    CHECK(files("*.csv") == std::vector<std::string>{"a.csv", "b.csv"});
    CHECK(files("?.csv") == std::vector<std::string>{"a.csv", "b.csv"});
    CHECK(files("**/*.csv") == std::vector<std::string>{"a.csv", "b.csv", "deep/x/y/z.csv", "sub/c.csv"});
    CHECK(files("deep/**/z.csv") == std::vector<std::string>{"deep/x/y/z.csv"});
    CHECK(files("*/c.csv") == std::vector<std::string>{"sub/c.csv"});
    CHECK(files(".hidden.csv") == std::vector<std::string>{".hidden.csv"});
    CHECK(files(ws + "/b.csv") == std::vector<std::string>{"b.csv"});
    CHECK(files(root + "/granted/*.csv").size() == 1);
    CHECK(files(root + "/granted_file.csv").size() == 1);

    CHECK(code("../outside/secret.csv") == "loader.path_not_allowed");
    CHECK(code("sub/../a.csv") == "loader.path_not_allowed");
    CHECK(code(root + "/outside/secret.csv") == "loader.path_not_allowed");
    CHECK(code(root + "/outside/*.csv") == "loader.path_not_allowed");
    CHECK(code(root + "/ws-sibling/s.csv") == "loader.path_not_allowed");
    CHECK(code("missing.csv") == "loader.no_match");
    CHECK(code("*.parquet") == "loader.no_match");
    CHECK(code("nofolder/*.csv") == "loader.no_match");
    CHECK(code("sub") == "loader.no_match");  // a folder is not a file
    CHECK(code("**") == "loader.bad_value");
    CHECK(code("") == "loader.bad_value");
    CHECK(code("a\n.csv") == "loader.bad_value");
    GlobLimits few;
    few.max_matches = 1;
    CHECK(code("*.csv", few) == "loader.too_many_files");
    few = GlobLimits{};
    few.max_visited = 3;
    CHECK(code("**/*.csv", few) == "loader.too_many_files");
    few = GlobLimits{};
    few.max_depth = 1;
    CHECK(code("**/*.csv", few) == "loader.too_many_files");
    CHECK(expand_input("a.csv", AllowedRoots{}).error.code == "loader.bad_value");
#ifndef _WIN32
    fs::create_symlink(root + "/outside/secret.csv", ws + "/link.csv");
    fs::create_directory_symlink(root + "/outside", ws + "/linkdir");
    CHECK(code("link.csv") == "loader.path_not_allowed");
    CHECK(code("*.csv") == "loader.path_not_allowed");  // the link matches the glob
    CHECK(code("linkdir/secret.csv") == "loader.path_not_allowed");
    CHECK(code("linkdir/*.csv") == "loader.path_not_allowed");
    fs::remove(ws + "/link.csv");
    // The ** walk does not enter a linked folder.
    CHECK(files("**/*.csv") == std::vector<std::string>{"a.csv", "b.csv", "deep/x/y/z.csv", "sub/c.csv"});
#endif
}

TEST_CASE("reader inference: extension first, then the bytes") {
    const std::string dir = fresh_dir("infer");
    write_text(dir + "/a.csv", "x,y\n1,2\n");
    CHECK(infer_reader(dir + "/a.csv").reader == ReaderKind::csv);
    CHECK(infer_reader(dir + "/a.csv").from_extension);
    write_bytes(dir + "/fake.csv", {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0});
    Error e = infer_reader(dir + "/fake.csv").error;
    CHECK(e.code == "loader.reader_mismatch");
    CHECK(e.message.find("png") != std::string::npos);
    write_bytes(dir + "/zip.npy", make_zip({{"a.npy", {}}}));
    CHECK(infer_reader(dir + "/zip.npy").error.code == "loader.reader_mismatch");
    const std::vector<uint8_t> npy = make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }", raw(std::vector<float>{1}));
    write_bytes(dir + "/noext", npy);
    ReaderInference r = infer_reader(dir + "/noext");
    REQUIRE(r.error.ok());
    CHECK(r.reader == ReaderKind::npy);
    CHECK_FALSE(r.from_extension);
    write_bytes(dir + "/w.bin", make_safetensors(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", raw(std::vector<float>{1})));
    CHECK(infer_reader(dir + "/w.bin").reader == ReaderKind::safetensors);
    write_text(dir + "/notes.txt", "hello");
    CHECK(infer_reader(dir + "/notes.txt").error.code == "loader.no_reader");
    write_bytes(dir + "/model.bin", {0x80, 0x02, '.'});
    CHECK(infer_reader(dir + "/model.bin").error.code == "loader.pickle_refused");
    write_bytes(dir + "/m.pt", npy);
    CHECK(infer_reader(dir + "/m.pt").error.code == "loader.pickle_refused");
    // A .npy whose bytes identify nothing goes to the npy reader, which reports it.
    write_text(dir + "/odd.npy", "not numpy");
    CHECK(infer_reader(dir + "/odd.npy").reader == ReaderKind::npy);
    CHECK(infer_reader(dir + "/missing.csv").error.code == "loader.io");
}

TEST_CASE("default loaders for dropped files") {
    const std::string ws = fresh_dir("infer_loader");
    write_text(ws + "/data/Run 42.csv", "x\n1\n");
    LoaderInference l = infer_loader(ws + "/data/Run 42.csv", ws, {});
    REQUIRE(l.error.ok());
    CHECK(l.loader.id == "run_42");
    CHECK(l.loader.path == "data/Run 42.csv");
    CHECK(*l.loader.reader == ReaderKind::csv);
    CHECK(l.loader.outputs[0].name == "run_42");
    CHECK_FALSE(l.loader.outputs[0].tensor);
    CHECK(l.loader.title == "Run 42.csv");
    CHECK(parse_loader(write_loader(l.loader)).error.ok());

    l = infer_loader(ws + "/data/Run 42.csv", ws, {"run_42", "run_42-2"});
    CHECK(l.loader.id == "run_42-3");

    write_bytes(ws + "/model-v2.safetensors",
                make_safetensors(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", raw(std::vector<float>{1})));
    l = infer_loader(ws + "/model-v2.safetensors", ws, {});
    REQUIRE(l.error.ok());
    CHECK(l.loader.id == "model-v2");
    CHECK(l.loader.outputs[0].name == "model_v2_");
    CHECK(*l.loader.outputs[0].tensor == "*");
    CHECK(parse_loader(write_loader(l.loader)).error.ok());

    const std::string other = fresh_dir("infer_loader_outside");
    write_text(other + "/x.csv", "x\n1\n");
    l = infer_loader(other + "/x.csv", ws, {});
    REQUIRE(l.error.ok());
    CHECK(l.loader.path == other + "/x.csv");
}
