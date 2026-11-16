// Tests of the Python loader: its options and output rules, the interpreter and
// its environment, the manifest checks on hostile folders written in code, and
// whole runs of real scripts through LoaderStore with the interpreter named by
// CALIPER_BUILDER_TEST_PYTHON. When that is empty, the tests that run Python
// say so and return, and the others still run. POSIX only, like process
// spawning.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/loaders/python_loader.h>
#include <caliper_builder/loaders/store.h>

#include <doctest/doctest.h>

#include "test_util.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <sys/types.h>
#include <unistd.h>

using namespace caliper_builder;
using namespace loaders_test;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const std::string kPython = CALIPER_BUILDER_TEST_PYTHON;
const std::string kFixtures = CALIPER_BUILDER_PYTHON_FIXTURES;

bool have_python() {
    if (kPython.empty()) {
        MESSAGE("Skipped: CALIPER_BUILDER_TEST_PYTHON is not set, so this test does not run Python.");
        return false;
    }
    return true;
}

// Runs the test interpreter on a short program; true when it exits with 0.
bool python_ok(const std::string& code) {
    acp::SpawnOptions so;
    so.argv = {kPython, "-I", "-c", code};
    acp::RunLimits limits;
    limits.timeout_ms = 120000;
    return acp::run_process(so, limits).status.success();
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

Error parse_error(const std::string& text) { return parse_loader(text).error; }

const std::string kPythonHead =
    "caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.pt\"\nreader = \"python\"\n";

// A workspace with a store, scripts and Python loaders.
struct Workspace {
    std::string root;  // holds ws/ and files outside it
    std::string ws;
    std::unique_ptr<LoaderStore> store;
    StoreOptions options;

    explicit Workspace(const std::string& name) {
        root = fresh_dir("python_" + name);
        ws = root + "/ws";
        fs::create_directories(ws);
        options.python.kill_grace_ms = 500;
        open();
    }
    void open() {
        store.reset();
        StoreOpen o = LoaderStore::open(ws, options);
        REQUIRE_MESSAGE(o.error.ok(), o.error.code << ": " << o.error.message);
        store = std::move(o.store);
    }
    void file(const std::string& rel, const std::string& text) { write_text(ws + "/" + rel, text); }
    // Saves a Python loader that runs `script` on `input` with the test interpreter.
    Error loader(const std::string& id, const std::string& input, const std::string& script,
                 const std::string& outputs, const std::string& reader_extra = "") {
        const std::string text = "caliper_loader = 0\n[loader]\nid = \"" + id + "\"\n[input]\npath = \"" + input +
                                 "\"\nreader = \"python\"\n[reader]\nscript = \"" + script + "\"\npython = \"" +
                                 kPython + "\"\n" + reader_extra + outputs;
        LoaderParse p = parse_loader(text);
        if (p.error) return p.error;
        return store->save(p.loader);
    }
    RunResult run(const std::string& id) {
        RunOptions o;
        o.allow_python = true;
        return store->run(id, o);
    }
    RunResult run_ok(const std::string& id) {
        RunResult r = run(id);
        REQUIRE_MESSAGE(r.error.ok(), id << ": " << r.error.code << " " << r.error.where << ": " << r.error.message);
        return r;
    }
    LoadedSource source(const std::string& id) {
        LoaderStore::SourceGet g = store->describe(id);
        REQUIRE_MESSAGE(g.error.ok(), id << ": " << g.error.message);
        return g.source;
    }
    json rows(const std::string& sql) {
        QueryResult q = store->query(sql);
        REQUIRE_MESSAGE(q.error.ok(), q.error.message);
        return q.rows;
    }
};

bool has_warning(const LoadedSource& s, const std::string& text) {
    for (const std::string& w : s.warnings) {
        if (w.find(text) != std::string::npos) return true;
    }
    return false;
}

// A folder with one .npy tensor, for the manifest tests.
struct ManifestFolder {
    std::string dir;
    uint64_t npy_bytes = 0;

    explicit ManifestFolder(const std::string& name) {
        dir = fresh_dir("manifest_" + name) + "/out";
        fs::create_directories(dir);
        TensorResult t = TensorData::from_bytes(DType::f32, {2, 3}, std::vector<uint8_t>(24, 0));
        REQUIRE(t.error.ok());
        REQUIRE(write_npy(dir + "/w.npy", t.tensor).ok());
        npy_bytes = fs::file_size(dir + "/w.npy");
    }
    json tensor_entry() const {
        return {{"name", "w"}, {"kind", "tensor"}, {"file", "w.npy"}, {"bytes", npy_bytes}, {"dtype", "f32"},
                {"shape", {2, 3}}};
    }
    json manifest(const std::vector<json>& outputs) const {
        return {{"caliper_python_loader", 0}, {"outputs", json(outputs)}, {"warnings", json::array()}};
    }
    void write(const json& m) const { write_text(dir + "/manifest.json", m.dump()); }
    Error check(const json& m, const PythonOutputLimits& limits = {}) const {
        write(m);
        return read_python_manifest(dir, limits).error;
    }
};

}  // namespace

TEST_CASE("the options and outputs of a Python loader are checked") {
    PythonReaderOptions o;
    CHECK(parse_python_options(json::object(), &o).code == "loader.missing_key");
    REQUIRE(parse_python_options({{"script", "s.py"}}, &o).ok());
    CHECK(o.script == "s.py");
    CHECK(o.python.empty());
    CHECK(o.timeout_s == 120);
    REQUIRE(parse_python_options({{"script", "s.py"}, {"python", "/usr/bin/python3"}, {"timeout_s", 3600}}, &o).ok());
    CHECK(o.timeout_s == 3600);
    CHECK(parse_python_options({{"script", "s.py"}, {"python", "python3.12"}}, &o).ok());
    CHECK(parse_python_options({{"script", "s.py"}, {"interpreter", "x"}}, &o).code == "loader.unknown_key");
    CHECK(parse_python_options({{"script", ""}}, &o).code == "loader.bad_value");
    CHECK(parse_python_options({{"script", 3}}, &o).code == "loader.bad_value");
    CHECK(parse_python_options({{"script", "s\n.py"}}, &o).code == "loader.bad_value");
    for (const json& t : {json(0), json(3601), json("60"), json(1.5)}) {
        CHECK(parse_python_options({{"script", "s.py"}, {"timeout_s", t}}, &o).code == "loader.bad_value");
    }
    // A relative interpreter path is refused; a command name is looked up on PATH.
    for (const char* p : {".venv/bin/python", "./python3", "bin/python"}) {
        Error e = parse_python_options({{"script", "s.py"}, {"python", p}}, &o);
        CHECK(e.code == "loader.bad_value");
        CHECK(e.where == "reader.python");
    }

    const std::string head = kPythonHead + "[reader]\nscript = \"s.py\"\n";
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntensor = \"w\"\n").ok());
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntable = \"m\"\nmode = \"table\"\nsql = \"SELECT 1\"\n").ok());
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\n").code == "loader.missing_key");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntensor = \"w\"\ntable = \"m\"\n").code == "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntensor = \"w\"\nsql = \"SELECT 1\"\n").code ==
          "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntable = \"m\"\n  [[output.transform]]\n  op = \"reduce\"\n"
                             "  axis = 0\n  fn = \"mean\"\n")
              .code == "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntable = \"*\"\nsql = \"SELECT 1\"\n").code ==
          "loader.bad_value");
    CHECK(parse_error(head + "[[output]]\nname = \"t\"\ntable = \"\"\n").code == "loader.bad_value");
    CHECK(parse_error(kPythonHead + "[reader]\nscript = \"s.py\"\npython = \"venv/bin/python\"\n[[output]]\n"
                                    "name = \"t\"\ntensor = \"*\"\n")
              .code == "loader.bad_value");
    // Only a Python loader selects a table.
    CHECK(parse_error("caliper_loader = 0\n[loader]\nid = \"a\"\n[input]\npath = \"x.csv\"\n[[output]]\nname = \"t\"\n"
                      "table = \"m\"\n")
              .code == "loader.bad_value");

    // The canonical writer keeps `table` and the reader options.
    LoaderParse p = parse_loader(head + "timeout_s = 30\n[[output]]\nname = \"t\"\ntable = \"m\"\n");
    REQUIRE(p.error.ok());
    const std::string text = write_loader(p.loader);
    CHECK(text.find("table = \"m\"") != std::string::npos);
    LoaderParse again = parse_loader(text);
    REQUIRE(again.error.ok());
    CHECK(again.loader.to_json() == p.loader.to_json());
    CHECK(loader_from_json(p.loader.to_json()).error.ok());
}

TEST_CASE("the interpreter is resolved and gets a scrubbed environment") {
    CHECK(resolve_python("venv/bin/python").error.code == "loader.bad_value");
    CHECK(resolve_python("/no/such/folder/python3").error.code == "loader.python_not_available");
    CHECK(resolve_python("caliper-no-such-python-interpreter").error.code == "loader.python_not_available");

    setenv("CALIPER_PYLOADER_TEST_SECRET", "not-for-the-child", 1);
    PythonInterpreter fake;
    fake.path = "/usr/bin/python3";
    for (const auto& [name, value] : python_environment(fake)) {
        CHECK(name != "CALIPER_PYLOADER_TEST_SECRET");
        CHECK((name == "PATH" || name == "HOME" || name == "TMPDIR" || name == "LANG"));
        if (name == "PATH") {
            for (const char* rel : {"::", ":.:"}) CHECK(value.find(rel) == std::string::npos);
        }
    }
    fake.venv = "/opt/env";
    bool saw_venv = false;
    for (const auto& [name, value] : python_environment(fake)) {
        if (name == "VIRTUAL_ENV") saw_venv = value == "/opt/env";
        if (name == "PATH") CHECK(value.rfind("/opt/env/bin:", 0) == 0);
    }
    CHECK(saw_venv);

    if (!have_python()) return;
    PythonInterpreter py = resolve_python(kPython);
    REQUIRE(py.error.ok());
    CHECK(fs::u8path(py.path).is_absolute());
    // The test interpreter lies in a virtual environment made by `uv venv`.
    CHECK(fs::exists(fs::u8path(py.venv) / "pyvenv.cfg"));
}

TEST_CASE("the embedded loader_host.py is the file in python/caliper") {
    CHECK(embedded_loader_host() == read_text(CALIPER_BUILDER_LOADER_HOST_PY));
    CHECK(embedded_loader_host().find("def torch_weights") != std::string::npos);
}

TEST_CASE("read_python_manifest refuses hostile folders") {
    ManifestFolder f("hostile");
    // A valid manifest.
    json good = f.manifest({f.tensor_entry()});
    good["outputs"][0]["title"] = "layers.0.w";
    good["outputs"][0]["warnings"] = {"loaded weights-only from a pickle file"};
    good["warnings"] = {"a run warning"};
    f.write(good);
    PythonManifestRead ok = read_python_manifest(f.dir);
    REQUIRE_MESSAGE(ok.error.ok(), ok.error.message);
    REQUIRE(ok.manifest.outputs.size() == 1);
    CHECK(ok.manifest.outputs[0].title == "layers.0.w");
    CHECK(*ok.manifest.outputs[0].dtype == DType::f32);
    CHECK(*ok.manifest.outputs[0].shape == std::vector<int64_t>{2, 3});
    CHECK(ok.manifest.warnings == std::vector<std::string>{"a run warning"});
    CHECK(ok.manifest.to_json()["outputs"][0]["bytes"] == f.npy_bytes);

    auto with = [&](const std::string& key, const json& value) {
        json m = f.manifest({f.tensor_entry()});
        if (value.is_null()) m["outputs"][0].erase(key);
        else m["outputs"][0][key] = value;
        return f.check(m);
    };
    // Path tricks in the file name.
    for (const char* name : {"../w.npy", "/etc/w.npy", "sub/w.npy", ".w.npy", "w.npy/", "..", "w\\x.npy", ""}) {
        CHECK_MESSAGE(with("file", name).code == "loader.path_not_allowed", name);
    }
    CHECK(with("file", "w.txt").code == "loader.malformed");
    CHECK(with("file", "missing.npy").code == "loader.malformed");
    // Names, kinds and types.
    for (const json& name : {json("W"), json("1w"), json("a.b"), json(""), json(std::string(65, 'a')), json(3)}) {
        CHECK(with("name", name).code == "loader.malformed");
    }
    CHECK(with("kind", "blob").code == "loader.malformed");
    CHECK(with("kind", "table").code == "loader.malformed");  // a table is no .npy and has no dtype
    CHECK(with("dtype", "object").code == "loader.malformed");
    CHECK(with("dtype", nullptr).code == "loader.malformed");
    CHECK(with("shape", {1, 2, 3, 4, 5, 6, 7, 8, 9}).code == "loader.malformed");
    CHECK(with("shape", {-1}).code == "loader.malformed");
    CHECK(with("shape", {100, 100}).code == "loader.malformed");  // more bytes than the file holds
    CHECK(with("extra", 1).code == "loader.malformed");
    CHECK(with("title", "two\nlines").code == "loader.malformed");
    CHECK(with("warnings", {"bad\x01text"}).code == "loader.malformed");
    CHECK(with("warnings", json(std::vector<std::string>(65, "w"))).code == "loader.too_large");
    // Sizes: a huge declared size, a huge shape, a size that is not the file's.
    CHECK(with("bytes", 1ULL << 62).code == "loader.too_large");
    CHECK(with("bytes", -1).code == "loader.malformed");
    CHECK(with("bytes", 1.5).code == "loader.malformed");
    CHECK(with("bytes", f.npy_bytes + 1).code == "loader.malformed");
    CHECK(with("shape", {int64_t(1) << 40, int64_t(1) << 40}).code == "loader.too_large");
    PythonOutputLimits small;
    small.max_file_bytes = 64;
    CHECK(f.check(f.manifest({f.tensor_entry()}), small).code == "loader.too_large");
    small = {};
    small.max_total_bytes = f.npy_bytes + 10;
    fs::copy_file(f.dir + "/w.npy", f.dir + "/v.npy");
    json two = f.tensor_entry();
    two["name"] = "v";
    two["file"] = "v.npy";
    CHECK(f.check(f.manifest({f.tensor_entry(), two}), small).code == "loader.too_large");
    small = {};
    small.max_outputs = 1;
    CHECK(f.check(f.manifest({f.tensor_entry(), two}), small).code == "loader.too_large");
    CHECK(f.check(f.manifest({f.tensor_entry(), f.tensor_entry()})).code == "loader.malformed");
    two["name"] = "w";
    two["kind"] = "tensor";
    CHECK(f.check(f.manifest({f.tensor_entry(), two})).code == "loader.malformed");  // the same name twice
    // The manifest itself.
    CHECK(f.check(json::array()).code == "loader.malformed");
    json bad = f.manifest({f.tensor_entry()});
    bad["caliper_python_loader"] = 1;
    CHECK(f.check(bad).code == "loader.unsupported");
    bad = f.manifest({f.tensor_entry()});
    bad["extra"] = true;
    CHECK(f.check(bad).code == "loader.malformed");
    bad = f.manifest({f.tensor_entry()});
    bad["outputs"] = "w.npy";
    CHECK(f.check(bad).code == "loader.malformed");
    write_text(f.dir + "/manifest.json", "{\"caliper_python_loader\": 0, ");
    CHECK(read_python_manifest(f.dir).error.code == "loader.malformed");
    write_text(f.dir + "/manifest.json", std::string(100000, '[') + std::string(100000, ']'));
    CHECK(read_python_manifest(f.dir).error.code == "loader.malformed");
    write_text(f.dir + "/manifest.json", std::string((1 << 20) + 1, ' '));
    CHECK(read_python_manifest(f.dir).error.code == "loader.too_large");
    fs::remove(f.dir + "/manifest.json");
    CHECK(read_python_manifest(f.dir).error.code == "loader.malformed");

    // Symbolic links and hard links.
    const std::string outside = fs::path(f.dir).parent_path().string() + "/outside.npy";
    fs::copy_file(f.dir + "/w.npy", outside);
    fs::create_symlink(outside, f.dir + "/link.npy");
    json link = f.tensor_entry();
    link["file"] = "link.npy";
    CHECK(f.check(f.manifest({link})).code == "loader.path_not_allowed");
    fs::create_hard_link(outside, f.dir + "/hard.npy");
    link["file"] = "hard.npy";
    CHECK(f.check(f.manifest({link})).code == "loader.path_not_allowed");
    f.write(good);
    fs::create_symlink(f.dir + "/manifest.json", f.dir + "/../m.json");
    fs::rename(f.dir + "/manifest.json", f.dir + "/real.json");
    fs::create_symlink(f.dir + "/real.json", f.dir + "/manifest.json");
    CHECK(read_python_manifest(f.dir).error.code == "loader.path_not_allowed");
    // The folder itself may not be a link.
    const std::string folder_link = fs::path(f.dir).parent_path().string() + "/out-link";
    fs::create_directory_symlink(f.dir, folder_link);
    CHECK(read_python_manifest(folder_link).error.code == "loader.path_not_allowed");
}

TEST_CASE("the typed CSV header is checked") {
    const std::string dir = fresh_dir("typed_csv");
    write_text(dir + "/ok.csv", "\"step:BIGINT\",\"a:b:VARCHAR\",\"x \"\"q\"\":DOUBLE\",\"f:BOOLEAN\"\n1,\"s\",0.5,true\n");
    TypedCsvHeader h = read_typed_csv_header(dir + "/ok.csv");
    REQUIRE_MESSAGE(h.error.ok(), h.error.message);
    REQUIRE(h.columns.size() == 4);
    CHECK(h.columns[0].name == "step");
    CHECK(h.columns[1].name == "a:b");
    CHECK(h.columns[1].type == "VARCHAR");
    CHECK(h.columns[2].name == "x \"q\"");
    CHECK(h.columns[3].type == "BOOLEAN");
    for (const char* text : {"step:BIGINT\n1\n", "\"step:INTEGER\"\n", "\"step\"\n", "\"step:BIGINT\"",
                             "\"step:BIGINT\",\"step:DOUBLE\"\n", "\":BIGINT\"\n", "\"step:BIGINT\";\"x:DOUBLE\"\n",
                             "\"unterminated:BIGINT\n", ""}) {
        write_text(dir + "/bad.csv", text);
        CHECK_MESSAGE(read_typed_csv_header(dir + "/bad.csv").error.code == "loader.malformed", text);
    }
    PythonOutputLimits limits;
    limits.max_csv_header_bytes = 16;
    write_text(dir + "/long.csv", "\"a_long_column_name:BIGINT\"\n1\n");
    CHECK(read_typed_csv_header(dir + "/long.csv", limits).error.code == "loader.malformed");
    limits = {};
    limits.max_csv_columns = 1;
    CHECK(read_typed_csv_header(dir + "/ok.csv", limits).error.code == "loader.too_large");
}

TEST_CASE("a Python loader runs only with approval, and a missing interpreter is reported") {
    Workspace w("approval");
    w.file("in.bin", "data");
    w.file("s.py", "def load(path, out):\n    out.tensor('w', [1])\n");
    const std::string text = "caliper_loader = 0\n[loader]\nid = \"py\"\n[input]\npath = \"in.bin\"\nreader = "
                             "\"python\"\n[reader]\nscript = \"s.py\"\n[[output]]\nname = \"w\"\ntensor = \"w\"\n";
    REQUIRE(w.store->save(parse_loader(text).loader).ok());
    RunResult r = w.store->run("py");
    CHECK(r.error.code == "loader.python_needs_approval");
    CHECK(w.store->get("py").record.last_status == "never");
    CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/py"));
    RunOptions denied;
    denied.allow_python = false;
    CHECK(w.store->run("py", denied).error.code == "loader.python_needs_approval");

    // A missing interpreter, by path and by name.
    for (const char* python : {"/no/such/folder/python3", "caliper-no-such-python-interpreter"}) {
        const std::string t = "caliper_loader = 0\n[loader]\nid = \"missing\"\n[input]\npath = \"in.bin\"\nreader = "
                              "\"python\"\n[reader]\nscript = \"s.py\"\npython = \"" +
                              std::string(python) + "\"\n[[output]]\nname = \"w\"\ntensor = \"w\"\n";
        REQUIRE(w.store->save(parse_loader(t).loader).ok());
        RunOptions allow;
        allow.allow_python = true;
        CHECK(w.store->run("missing", allow).error.code == "loader.python_not_available");
        CHECK(w.store->get("missing").record.last_error_code == "loader.python_not_available");
    }

    // The script must lie inside the workspace.
    const std::string base = "caliper_loader = 0\n[loader]\nid = \"out\"\n[input]\npath = \"in.bin\"\nreader = "
                             "\"python\"\n[reader]\nscript = ";
    const std::string tail = "\n[[output]]\nname = \"w\"\ntensor = \"w\"\n";
    CHECK(w.store->save(parse_loader(base + "\"../s.py\"" + tail).loader).code == "loader.path_not_allowed");
    CHECK(w.store->save(parse_loader(base + "\"" + w.root + "/s.py\"" + tail).loader).code ==
          "loader.path_not_allowed");
    write_text(w.root + "/outside.py", "def load(path, out):\n    pass\n");
    fs::create_symlink(w.root + "/outside.py", w.ws + "/linked.py");
    CHECK(w.store->save(parse_loader(base + "\"linked.py\"" + tail).loader).code == "loader.path_not_allowed");
}

TEST_CASE("a Python loader without numpy is not available") {
    if (!have_python()) return;
    Workspace w("no_numpy");
    // A virtual environment of the test interpreter's Python, without numpy.
    acp::SpawnOptions so;
    so.argv = {kPython, "-I", "-m", "venv", "--without-pip", w.root + "/bare"};
    acp::RunLimits limits;
    limits.timeout_ms = 120000;
    acp::RunResult made = acp::run_process(so, limits);
    REQUIRE_MESSAGE(made.status.success(), made.err);
    w.file("in.bin", "data");
    w.file("s.py", "def load(path, out):\n    out.tensor('w', [1])\n");
    const std::string text = "caliper_loader = 0\n[loader]\nid = \"py\"\n[input]\npath = \"in.bin\"\nreader = "
                             "\"python\"\n[reader]\nscript = \"s.py\"\npython = \"" +
                             w.root + "/bare/bin/python\"\n[[output]]\nname = \"w\"\ntensor = \"w\"\n";
    REQUIRE(w.store->save(parse_loader(text).loader).ok());
    RunResult r = w.run("py");
    CHECK(r.error.code == "loader.python_not_available");
    CHECK(r.error.message.find("numpy") != std::string::npos);
}

TEST_CASE("tensors of each dtype round-trip through a Python loader") {
    if (!have_python()) return;
    Workspace w("dtypes");
    w.file("in.bin", "data");
    w.file("scripts/dtypes.py", R"PY(import numpy as np


def load(path, out):
    out.tensor("b", np.array([True, False, True]))
    out.tensor("u8", np.array([0, 255], dtype=np.uint8))
    out.tensor("u16", np.array([1, 65535], dtype=">u2"))
    out.tensor("u32", np.array([0, 2**32 - 1], dtype=np.uint32))
    out.tensor("u64", np.array([0, 2**64 - 1], dtype=np.uint64))
    out.tensor("i8", np.array([-128, 127], dtype=np.int8))
    out.tensor("i16", np.array([-32768, 32767], dtype=np.int16))
    out.tensor("i32", np.array([-2**31, 2**31 - 1], dtype=">i4"))
    out.tensor("i64", np.array([-2**63, 2**63 - 1], dtype=np.int64))
    out.tensor("f16", np.array([1.5, -2.0], dtype=np.float16))
    out.tensor("f32", np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)))
    out.tensor("f64", np.array([[0.1, np.nan], [np.inf, -0.0]]))
    out.tensor("scalar", np.float64(2.5))
    out.tensor("empty", np.zeros((0, 3), dtype=np.int32))
    out.tensor("nested", [[1, 2], [3, 4]])
)PY");
    REQUIRE(w.loader("dt", "in.bin", "scripts/dtypes.py", "[[output]]\nname = \"t_\"\ntensor = \"*\"\n").ok());
    RunResult r = w.run_ok("dt");
    CHECK(r.sources.size() == 15);
    struct Want {
        const char* name;
        DType dtype;
        std::vector<int64_t> shape;
    };
    const Want wants[] = {{"b", DType::boolean, {3}},  {"u8", DType::u8, {2}},   {"u16", DType::u16, {2}},
                          {"u32", DType::u32, {2}},    {"u64", DType::u64, {2}}, {"i8", DType::i8, {2}},
                          {"i16", DType::i16, {2}},    {"i32", DType::i32, {2}}, {"i64", DType::i64, {2}},
                          {"f16", DType::f16, {2}},    {"f32", DType::f32, {2, 3}}, {"f64", DType::f64, {2, 2}},
                          {"scalar", DType::f64, {}}, {"empty", DType::i32, {0, 3}}, {"nested", DType::i64, {2, 2}}};
    for (const Want& want : wants) {
        const LoadedSource s = w.source(std::string("tensor:t_") + want.name);
        CHECK_MESSAGE(*s.dtype == want.dtype, want.name);
        CHECK_MESSAGE(*s.shape == want.shape, want.name);
        CHECK(s.tensor == want.name);
        CHECK_FALSE(s.derived);
        CHECK(s.file == "in.bin");
        CHECK(s.data_file.find("/loader-outputs/dt/run-1/") != std::string::npos);
    }
    auto element = [&](const char* name, uint64_t i) {
        TensorResult t = w.store->load_tensor(std::string("tensor:t_") + name);
        REQUIRE(t.error.ok());
        return t.tensor.element(i);
    };
    CHECK(element("b", 1) == 0);
    CHECK(element("u8", 1) == 255);
    CHECK(element("u16", 0) == 1);  // big-endian input, stored little-endian
    CHECK(element("u16", 1) == 65535);
    CHECK(element("u32", 1) == 4294967295.0);
    CHECK(element("i8", 0) == -128);
    CHECK(element("i32", 0) == -2147483648.0);
    CHECK(element("f16", 0) == 1.5);
    CHECK(element("f32", 1) == 1);  // Fortran input, stored in row-major order
    CHECK(element("f32", 3) == 3);
    CHECK(std::isnan(element("f64", 1)));
    CHECK(std::isinf(element("f64", 2)));
    CHECK(element("scalar", 0) == 2.5);
    CHECK(element("nested", 3) == 4);
    TensorResult i64 = w.store->load_tensor("tensor:t_i64");
    int64_t v[2];
    std::memcpy(v, i64.tensor.data(), sizeof v);
    CHECK(v[0] == INT64_MIN);
    CHECK(v[1] == INT64_MAX);
    TensorResult u64 = w.store->load_tensor("tensor:t_u64");
    uint64_t u[2];
    std::memcpy(u, u64.tensor.data(), sizeof u);
    CHECK(u[1] == UINT64_MAX);
}

TEST_CASE("tables reach the catalog through Parquet and through typed CSV") {
    if (!have_python()) return;
    const bool parquet = python_ok("import pyarrow.parquet");
    if (!parquet) MESSAGE("pyarrow is not installed for the test interpreter, so only the CSV path is tested.");
    Workspace w("tables");
    w.file("in.bin", "data");
    const std::string script = R"PY(import math


def load(path, out):
    out.table("rows", [
        {"i": 1, "x": 0.5, "flag": True, "s": 'a,"b"'},
        {"i": 2, "x": math.nan, "flag": False, "s": ""},
        {"i": 3, "x": math.inf, "flag": None, "s": "line\nbreak"},
        {"i": None, "x": -1, "flag": True, "s": None, "extra": "only here"},
    ])
    out.table("cols", {"step": [1, 2, 3], "loss": [0.5, 0.25, 0.125]})
    out.warn("tables written")
)PY";
    w.file("scripts/parquet/tables.py", script);
    w.file("scripts/csv/tables.py", script);
    // The script's folder comes first on sys.path, so this hides pyarrow from the host.
    w.file("scripts/csv/pyarrow.py", "raise ImportError('pyarrow is hidden for this test')\n");
    const std::string outputs = "[[output]]\nname = \"$_\"\ntable = \"*\"\n"
                                "[[output]]\nname = \"$_small\"\ntable = \"cols\"\nmode = \"table\"\n"
                                "sql = \"SELECT step FROM input WHERE loss < 0.4\"\n";
    auto with_prefix = [&](const std::string& p) {
        std::string s = outputs;
        for (std::size_t at; (at = s.find('$')) != std::string::npos;) s.replace(at, 1, p);
        return s;
    };
    REQUIRE(w.loader("csv", "in.bin", "scripts/csv/tables.py", with_prefix("c")).ok());
    w.run_ok("csv");
    std::vector<std::string> prefixes = {"c"};
    if (parquet) {
        REQUIRE(w.loader("pq", "in.bin", "scripts/parquet/tables.py", with_prefix("p")).ok());
        w.run_ok("pq");
        prefixes.push_back("p");
        CHECK(w.source("table:p_rows").bytes > 0);
        CHECK(fs::exists(w.ws + "/loader-outputs/pq/run-1/rows.parquet"));
    }
    CHECK(fs::exists(w.ws + "/loader-outputs/csv/run-1/rows.csv"));
    for (const std::string& p : prefixes) {
        INFO("prefix " << p);
        const LoadedSource rows = w.source("table:" + p + "_rows");
        REQUIRE(rows.columns);
        REQUIRE(rows.columns->size() == 5);
        CHECK((*rows.columns)[0].type == "BIGINT");
        CHECK((*rows.columns)[1].type == "DOUBLE");
        CHECK((*rows.columns)[2].type == "BOOLEAN");
        CHECK((*rows.columns)[3].type == "VARCHAR");
        CHECK((*rows.columns)[4].name == "extra");
        CHECK(*rows.rows == 4);
        CHECK(rows.file == "in.bin");
        CHECK(rows.table_mode == "view");
        CHECK(has_warning(rows, "tables written"));
        const json got = w.rows("SELECT i, x, flag, s, extra FROM " + p + "_rows ORDER BY i NULLS LAST");
        CHECK(got == json::parse(R"([[1, 0.5, true, "a,\"b\"", null], [2, "NaN", false, "", null],
                                     [3, "Infinity", null, "line\nbreak", null], [null, -1.0, true, null, "only here"]])"));
        CHECK(w.rows("SELECT step, loss FROM " + p + "_cols ORDER BY step") ==
              json::parse("[[1, 0.5], [2, 0.25], [3, 0.125]]"));
        CHECK(w.rows("SELECT step FROM " + p + "_small ORDER BY step") == json::parse("[[2], [3]]"));
        CHECK(w.source("table:" + p + "_small").table_mode == "table");
    }
}

TEST_CASE("a pickle the native readers refuse loads weights-only through a Python loader") {
    if (!have_python()) return;
    Workspace w("torch_route");
    const std::string long_key = "encoder.blocks.11.attention.output.dense.weight_with_a_rather_long_suffix_here";
    const json checkpoint = {
        {"__record__", w.root + "/record.json"},
        {"layers.0.attn.weight", {{"dtype", "float32"}, {"shape", {2, 2}}, {"data", {1, 2, 3, 4}}}},
        {"layers.0.attn.bias", {{"dtype", "bfloat16"}, {"shape", {2}}, {"data", {0.5, 1.5}}}},
        {"nested", {{"inner", {{"dtype", "int64"}, {"shape", {1}}, {"data", {7}}}}}},
        {long_key, {{"dtype", "float64"}, {"shape", {1}}, {"data", {2.5}}}},
        {"epoch", 3}};
    std::vector<uint8_t> bytes = {0x80, 0x04};
    append(bytes, checkpoint.dump());
    write_bytes(w.ws + "/ckpt/model.pt", bytes);

    // The native route refuses the file, and its message names this route.
    std::vector<LoaderStore::ImportItem> items = w.store->import_files({w.ws + "/ckpt/model.pt"});
    REQUIRE(items.size() == 1);
    CHECK(items[0].error.code == "loader.pickle_refused");
    CHECK(items[0].error.message.find("reader = \"python\"") != std::string::npos);
    CHECK(items[0].error.message.find("out.torch_weights") != std::string::npos);
    LoaderParse native = parse_loader("caliper_loader = 0\n[loader]\nid = \"native\"\n[input]\npath = "
                                      "\"ckpt/model.pt\"\nreader = \"safetensors\"\n[[output]]\nname = \"n\"\n"
                                      "tensor = \"*\"\n");
    REQUIRE(w.store->save(native.loader).ok());
    CHECK(w.store->run("native").error.code == "loader.pickle_refused");

    // The Python route, with the test torch beside the script.
    fs::create_directories(w.ws + "/scripts/torch_route");
    fs::copy_file(kFixtures + "/torch.py", w.ws + "/scripts/torch_route/torch.py");
    w.file("scripts/torch_route/load_ckpt.py", "def load(path, out):\n    out.torch_weights(path)\n");
    REQUIRE(w.loader("ckpt", "ckpt/model.pt", "scripts/torch_route/load_ckpt.py",
                     "[[output]]\nname = \"ckpt_\"\ntensor = \"*\"\n")
                .ok());
    RunResult r = w.run_ok("ckpt");
    CHECK(r.sources.size() == 4);
    const json record = json::parse(read_text(w.root + "/record.json"));
    CHECK(record == json::parse(R"({"map_location": "cpu", "weights_only": true, "other": []})"));

    const LoadedSource weight = w.source("tensor:ckpt_layers_0_attn_weight");
    CHECK(weight.title == "layers.0.attn.weight");
    CHECK(*weight.dtype == DType::f32);
    CHECK(*weight.shape == std::vector<int64_t>{2, 2});
    CHECK(has_warning(weight, "loaded weights-only from a pickle file"));
    CHECK(has_warning(weight, "1 value of the checkpoint is not a tensor and was skipped."));
    CHECK(weight.file == "ckpt/model.pt");
    CHECK(w.store->load_tensor(weight.id).tensor.element(3) == 4);
    CHECK(weight.to_json()["warnings"].size() == 2);

    const LoadedSource bias = w.source("tensor:ckpt_layers_0_attn_bias");
    CHECK(*bias.dtype == DType::f32);
    CHECK(has_warning(bias, "converted from bf16 to f32"));
    CHECK(has_warning(bias, "loaded weights-only from a pickle file"));
    CHECK(*w.source("tensor:ckpt_nested_inner").dtype == DType::i64);
    // Python's make_id and the library's agree, so the names of the two routes match.
    bool found_long = false;
    for (const LoadedSource& s : w.store->sources("tensor", "ckpt_")) {
        if (s.title == long_key) {
            found_long = true;
            CHECK(s.tensor == make_id(long_key));
            CHECK(s.name == make_id("ckpt_" + make_id(long_key)));
        }
    }
    CHECK(found_long);
}

TEST_CASE("a real torch checkpoint loads weights-only when torch is installed") {
    if (!have_python()) return;
    if (!python_ok("import torch")) {
        MESSAGE("Skipped: torch is not installed for the test interpreter.");
        return;
    }
    Workspace w("real_torch");
    acp::SpawnOptions so;
    so.argv = {kPython, "-I", "-c",
               "import sys, torch; torch.save({'w': torch.arange(6.).reshape(2, 3), 'h': torch.ones(2, "
               "dtype=torch.bfloat16)}, sys.argv[1])",
               w.ws + "/model.pt"};
    acp::RunLimits limits;
    limits.timeout_ms = 120000;
    REQUIRE(acp::run_process(so, limits).status.success());
    w.file("s.py", "def load(path, out):\n    out.torch_weights(path)\n");
    REQUIRE(w.loader("real", "model.pt", "s.py", "[[output]]\nname = \"m_\"\ntensor = \"*\"\n").ok());
    w.run_ok("real");
    const LoadedSource s = w.source("tensor:m_w");
    CHECK(*s.shape == std::vector<int64_t>{2, 3});
    CHECK(has_warning(s, "loaded weights-only from a pickle file"));
    CHECK(w.store->load_tensor("tensor:m_w").tensor.element(5) == 5);
    CHECK(*w.source("tensor:m_h").dtype == DType::f32);
}

TEST_CASE("a script that fails, hangs or prints a lot is contained") {
    if (!have_python()) return;
    Workspace w("failures");
    w.file("data/in.bin", "data");
    w.file("good.py", "def load(path, out):\n    out.tensor('w', [1.0, 2.0])\n");
    REQUIRE(w.loader("good", "data/in.bin", "good.py", "[[output]]\nname = \"w\"\ntensor = \"w\"\n").ok());
    w.run_ok("good");

    // A script that raises keeps the previous sources whole.
    w.file("good.py", "def load(path, out):\n    out.tensor('w', [9.0])\n    raise ValueError('broken on purpose')\n");
    RunResult r = w.run("good");
    CHECK(r.error.code == "loader.python_failed");
    CHECK(r.error.message.find("ValueError: broken on purpose") != std::string::npos);
    CHECK(w.store->get("good").record.last_status == "error");
    CHECK(w.store->load_tensor("tensor:w").tensor.element(1) == 2.0);
    CHECK(fs::exists(w.ws + "/loader-outputs/good/run-1"));
    CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/good/tmp-2"));

    // The writer refuses an object array, and a script without load() fails.
    w.file("object.py", "import numpy as np\n\ndef load(path, out):\n    out.tensor('o', np.array([{}], dtype=object))\n");
    REQUIRE(w.loader("object", "data/in.bin", "object.py", "[[output]]\nname = \"o\"\ntensor = \"o\"\n").ok());
    r = w.run("object");
    CHECK(r.error.code == "loader.python_failed");
    CHECK(r.error.message.find("object dtype") != std::string::npos);
    w.file("noload.py", "x = 1\n");
    REQUIRE(w.loader("noload", "data/in.bin", "noload.py", "[[output]]\nname = \"n\"\ntensor = \"*\"\n").ok());
    CHECK(w.run("noload").error.message.find("load(path, out)") != std::string::npos);

    // A script that hangs is stopped at its time limit with its child processes.
    w.file("hang.py", R"PY(import os
import subprocess
import sys
import time


def load(path, out):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(os.path.join(os.path.dirname(path), "child.pid"), "w") as f:
        f.write(str(child.pid))
    time.sleep(600)
)PY");
    REQUIRE(w.loader("hang", "data/in.bin", "hang.py", "[[output]]\nname = \"h\"\ntensor = \"*\"\n",
                     "timeout_s = 2\n")
                .ok());
    const auto start = std::chrono::steady_clock::now();
    r = w.run("hang");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(r.error.code == "loader.python_timeout");
    CHECK(seconds < 15);
    const std::string pid_text = read_text(w.ws + "/data/child.pid");
    REQUIRE_FALSE(pid_text.empty());
    const pid_t child = static_cast<pid_t>(std::stol(pid_text));
    bool gone = false;
    for (int i = 0; i < 100 && !gone; ++i) {
        gone = kill(child, 0) != 0 && errno == ESRCH;
        if (!gone) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK_MESSAGE(gone, "the script's child process outlived the time limit");
    if (!gone) kill(child, SIGKILL);
    CHECK(fs::is_empty(w.ws + "/loader-outputs/hang"));

    // A script that prints megabytes to both streams still loads.
    w.file("loud.py", R"PY(import sys


def load(path, out):
    line = "x" * 1023 + "\n"
    for _ in range(8 * 1024):
        sys.stdout.write(line)
        sys.stderr.write(line)
    out.tensor("ok", [1, 2, 3])
)PY");
    REQUIRE(w.loader("loud", "data/in.bin", "loud.py", "[[output]]\nname = \"ok\"\ntensor = \"ok\"\n").ok());
    w.run_ok("loud");
    CHECK(*w.source("tensor:ok").shape == std::vector<int64_t>{3});
}

TEST_CASE("a manifest that a script forges is checked like any hostile input") {
    if (!have_python()) return;
    Workspace w("forged");
    w.file("in.bin", "data");
    write_text(w.root + "/outside.csv", "\"s:VARCHAR\"\n\"secret\"\n");
    // Each script writes its own files and manifest.json, then leaves before the host writes one.
    const std::string head = "import json\nimport os\n\n\ndef load(path, out):\n";
    const std::string tail = "    with open('manifest.json', 'w') as f:\n"
                             "        json.dump({'caliper_python_loader': 0, 'outputs': outputs}, f)\n"
                             "    os._exit(0)\n";
    struct Case {
        const char* name;
        const char* body;
        const char* code;
    };
    const Case cases[] = {
        {"tricks",
         "    with open('../escape.npy', 'wb') as f:\n        f.write(b'x')\n"
         "    outputs = [{'name': 'e', 'kind': 'tensor', 'file': '../escape.npy', 'bytes': 1, 'dtype': 'u8', "
         "'shape': [1]}]\n",
         "loader.path_not_allowed"},
        {"absolute",
         "    outputs = [{'name': 'e', 'kind': 'table', 'file': os.path.abspath('x.csv'), 'bytes': 1}]\n",
         "loader.path_not_allowed"},
        {"object",
         "    header = \"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }\"\n"
         "    header = header + ' ' * (117 - len(header)) + '\\n'\n"
         "    with open('o.npy', 'wb') as f:\n"
         "        f.write(b'\\x93NUMPY\\x01\\x00' + len(header).to_bytes(2, 'little') + header.encode() + bytes(8))\n"
         "    outputs = [{'name': 'o', 'kind': 'tensor', 'file': 'o.npy', 'bytes': os.path.getsize('o.npy'), "
         "'dtype': 'u8', 'shape': [1]}]\n",
         "loader.pickle_refused"},
        {"symlink",
         "    target = os.path.join(os.path.dirname(os.path.dirname(path)), 'outside.csv')\n"
         "    os.symlink(target, 'x.csv')\n"
         "    outputs = [{'name': 'x', 'kind': 'table', 'file': 'x.csv', 'bytes': os.path.getsize(target)}]\n",
         "loader.path_not_allowed"},
        {"huge",
         "    with open('h.npy', 'wb') as f:\n        f.write(b'x')\n"
         "    outputs = [{'name': 'h', 'kind': 'tensor', 'file': 'h.npy', 'bytes': 2 ** 62, 'dtype': 'f32', "
         "'shape': [2 ** 30, 2 ** 30]}]\n",
         "loader.too_large"},
        {"mismatch",
         "    import numpy as np\n    np.save('m.npy', np.zeros(4, dtype=np.float32), allow_pickle=False)\n"
         "    outputs = [{'name': 'm', 'kind': 'tensor', 'file': 'm.npy', 'bytes': os.path.getsize('m.npy'), "
         "'dtype': 'u8', 'shape': [2]}]\n",
         "loader.malformed"},
        {"header",
         "    with open('t.csv', 'w') as f:\n        f.write('a,b\\n1,2\\n')\n"
         "    outputs = [{'name': 't', 'kind': 'table', 'file': 't.csv', 'bytes': os.path.getsize('t.csv')}]\n",
         "loader.malformed"},
    };
    for (const Case& c : cases) {
        INFO("case " << c.name);
        const std::string file = std::string(c.name) + ".py";
        w.file(file, head + c.body + tail);
        REQUIRE(w.loader(c.name, "in.bin", file, "[[output]]\nname = \"x_\"\ntensor = \"*\"\n"
                                                 "[[output]]\nname = \"t_\"\ntable = \"*\"\n")
                    .ok());
        RunResult r = w.run(c.name);
        CHECK_MESSAGE(r.error.code == c.code, r.error.code << ": " << r.error.message);
        CHECK(w.store->sources("", "x_").empty());
        CHECK(w.store->sources("", "t_").empty());
        CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/" + c.name + "/run-1"));
    }
}

TEST_CASE("the script sees a scrubbed environment") {
    if (!have_python()) return;
    setenv("CALIPER_PYLOADER_TEST_SECRET", "not-for-the-child", 1);
    Workspace w("environment");
    w.file("in.bin", "data");
    w.file("env.py", "import os\n\ndef load(path, out):\n    out.table('env', {'name': sorted(os.environ)})\n"
                     "    out.table('cwd', {'dir': [os.getcwd()]})\n");
    REQUIRE(w.loader("env", "in.bin", "env.py", "[[output]]\nname = \"env\"\ntable = \"env\"\nmode = \"table\"\n"
                                                "[[output]]\nname = \"cwd\"\ntable = \"cwd\"\nmode = \"table\"\n")
                .ok());
    w.run_ok("env");
    std::vector<std::string> names;
    for (const json& row : w.rows("SELECT name FROM env")) names.push_back(row[0].get<std::string>());
    CHECK(std::find(names.begin(), names.end(), "CALIPER_PYLOADER_TEST_SECRET") == names.end());
    CHECK(std::find(names.begin(), names.end(), "PATH") != names.end());
    CHECK(std::find(names.begin(), names.end(), "VIRTUAL_ENV") != names.end());
    for (const std::string& n : names) {
        const bool allowed = n == "PATH" || n == "HOME" || n == "TMPDIR" || n == "LANG" || n == "VIRTUAL_ENV" ||
                             n == "LC_CTYPE" || n == "__CF_USER_TEXT_ENCODING";
        CHECK_MESSAGE(allowed, "the child saw " << n);
    }
    // The working folder is the run's output folder, which is gone after the run.
    const std::string cwd = w.rows("SELECT dir FROM cwd")[0][0].get<std::string>();
    CHECK(cwd.find("/loader-outputs/env/tmp-1/out") != std::string::npos);
    CHECK_FALSE(fs::exists(cwd));
}

TEST_CASE("Python sources survive a restart and a rebuild without running Python again") {
    if (!have_python()) return;
    Workspace w("restart");
    w.file("data/in.bin", "data");
    w.file("count.py", R"PY(import os


def load(path, out):
    with open(os.path.join(os.path.dirname(path), "runs.txt"), "a") as f:
        f.write("run\n")
    out.tensor("w", [[1.5, 2.5], [3.5, 4.5]])
    out.table("m", [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}])
    out.tensor("unused", [0])
)PY");
    REQUIRE(w.loader("count", "data/in.bin", "count.py",
                     "[[output]]\nname = \"w\"\ntensor = \"w\"\n"
                     "[[output]]\nname = \"m\"\ntable = \"m\"\n"
                     "[[output]]\nname = \"w_mean\"\ntensor = \"w\"\n  [[output.transform]]\n  op = \"reduce\"\n"
                     "  axis = 0\n  fn = \"mean\"\n")
                .ok());
    RunResult r = w.run_ok("count");
    CHECK(r.sources.size() == 3);
    REQUIRE(r.warnings.size() == 1);
    CHECK(r.warnings[0].find("unused") != std::string::npos);
    CHECK(w.source("tensor:w_mean").derived);
    CHECK(fs::exists(w.ws + "/tensors/w_mean.npy"));
    const std::string kept = w.ws + "/loader-outputs/count/run-1";
    CHECK(fs::exists(kept + "/manifest.json"));
    CHECK(fs::exists(kept + "/w.npy"));
    CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/count/tmp-1"));
    auto runs = [&] { return read_text(w.ws + "/data/runs.txt"); };
    CHECK(runs() == "run\n");

    auto check_sources = [&] {
        CHECK(w.store->load_tensor("tensor:w").tensor.element(3) == 4.5);
        CHECK(w.store->load_tensor("tensor:w_mean").tensor.element(1) == 3.5);
        CHECK(w.rows("SELECT loss FROM m ORDER BY step") == json::parse("[[0.5], [0.25]]"));
        CHECK(w.store->get("count").record.last_status == "ok");
    };
    // A restart keeps the rows; a deleted database is rebuilt from the kept outputs.
    w.open();
    check_sources();
    w.store.reset();
    fs::remove(w.ws + "/workspace.duckdb");
    w.open();
    check_sources();
    CHECK(runs() == "run\n");

    // A second run replaces the kept outputs.
    w.run_ok("count");
    CHECK(runs() == "run\nrun\n");
    CHECK(fs::exists(w.ws + "/loader-outputs/count/run-2/manifest.json"));
    CHECK_FALSE(fs::exists(kept));

    // A kept output edited by hand is checked again when the database is rebuilt.
    write_text(w.ws + "/loader-outputs/count/run-2/w.npy", "short");
    w.store.reset();
    fs::remove(w.ws + "/workspace.duckdb");
    w.open();
    CHECK(w.store->get("count").record.last_error_code == "loader.malformed");
    CHECK(w.store->sources().empty());

    // Without kept outputs, a rebuilt Python loader has no sources and never ran.
    fs::remove_all(w.ws + "/loader-outputs/count");
    w.store.reset();
    fs::remove(w.ws + "/workspace.duckdb");
    w.open();
    CHECK(w.store->get("count").record.last_status == "never");
    CHECK(w.store->sources().empty());
    CHECK(runs() == "run\nrun\n");

    // Removing the loader removes its kept outputs.
    w.run_ok("count");
    CHECK(fs::exists(w.ws + "/loader-outputs/count"));
    REQUIRE(w.store->remove("count").ok());
    CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/count"));
    CHECK_FALSE(fs::exists(w.ws + "/tensors/w_mean.npy"));
}

TEST_CASE("a Python loader's outputs are selected by name") {
    if (!have_python()) return;
    Workspace w("selection");
    w.file("a.bin", "a");
    w.file("b.bin", "b");
    w.file("s.py", "def load(path, out):\n    out.tensor('w', [1])\n    out.table('m', {'x': [1]})\n");
    REQUIRE(w.loader("tn", "a.bin", "s.py", "[[output]]\nname = \"t\"\ntensor = \"absent\"\n").ok());
    CHECK(w.run("tn").error.code == "loader.tensor_not_found");
    REQUIRE(w.loader("tb", "a.bin", "s.py", "[[output]]\nname = \"t\"\ntable = \"w\"\n").ok());
    CHECK(w.run("tb").error.code == "loader.table_not_found");
    // One file only.
    REQUIRE(w.loader("glob", "*.bin", "s.py", "[[output]]\nname = \"w\"\ntensor = \"w\"\n").ok());
    CHECK(w.run("glob").error.code == "loader.bad_value");
    // A source name another loader owns.
    REQUIRE(w.loader("first", "a.bin", "s.py", "[[output]]\nname = \"same\"\ntable = \"m\"\n").ok());
    w.run_ok("first");
    REQUIRE(w.loader("second", "b.bin", "s.py", "[[output]]\nname = \"same\"\ntable = \"m\"\n").ok());
    CHECK(w.run("second").error.code == "loader.source_exists");
    CHECK(w.source("table:same").loader == "first");
    // A loader that becomes a native one drops its kept outputs once it runs.
    REQUIRE(fs::exists(w.ws + "/loader-outputs/first/run-1"));
    w.file("same.csv", "x\n1\n");
    REQUIRE(w.store->save(parse_loader("caliper_loader = 0\n[loader]\nid = \"first\"\n[input]\npath = \"same.csv\"\n"
                                       "[[output]]\nname = \"same\"\n")
                              .loader)
                .ok());
    REQUIRE(w.store->run("first").error.ok());
    CHECK_FALSE(fs::exists(w.ws + "/loader-outputs/first"));
    CHECK(w.source("table:same").file == "same.csv");
    // A script outside the workspace through a link, at run time.
    write_text(w.root + "/outside.py", "def load(path, out):\n    pass\n");
    REQUIRE(w.loader("late", "a.bin", "late.py", "[[output]]\nname = \"w\"\ntensor = \"*\"\n").ok());
    fs::create_symlink(w.root + "/outside.py", w.ws + "/late.py");
    CHECK(w.run("late").error.code == "loader.path_not_allowed");
}
