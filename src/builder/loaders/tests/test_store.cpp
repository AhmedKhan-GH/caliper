// Tests of LoaderStore: the end-to-end path from dropped files to catalog
// sources and back after reopening, saving and removing loaders, grants,
// failed runs, name clashes, and every reader through a run.
#include <caliper_builder/loaders/store.h>

#include <doctest/doctest.h>

#include <filesystem>

#include "../db.h"
#include "test_formats.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif
#include <stb_image_write.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

using namespace caliper_builder;
using namespace loaders_test;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

// Writes a Parquet file through DuckDB's COPY, in an in-memory database that may write to `dir`.
void write_parquet(const std::string& dir, const std::string& file, const std::string& select) {
    std::unique_ptr<detail::Db> db;
    detail::DbOptions o;
    o.allowed_directories = {dir};
    REQUIRE(detail::Db::open("", o, &db).ok());
    Error e = db->execute("COPY (" + select + ") TO " + detail::sql_literal(dir + "/" + file) + " (FORMAT parquet)");
    REQUIRE_MESSAGE(e.ok(), e.message);
}

std::unique_ptr<LoaderStore> open_store(const std::string& ws) {
    StoreOpen o = LoaderStore::open(ws);
    REQUIRE_MESSAGE(o.error.ok(), o.error.code << ": " << o.error.message);
    return std::move(o.store);
}

const LoadedSource* find(const std::vector<LoadedSource>& v, const std::string& id) {
    for (const LoadedSource& s : v) {
        if (s.id == id) return &s;
    }
    return nullptr;
}

RunResult run_ok(LoaderStore& store, const std::string& id) {
    RunResult r = store.run(id);
    REQUIRE_MESSAGE(r.error.ok(), id << ": " << r.error.code << " " << r.error.where << ": " << r.error.message);
    return r;
}

Loader loader_from(const std::string& text) {
    LoaderParse p = parse_loader(text);
    REQUIRE_MESSAGE(p.error.ok(), p.error.code << " " << p.error.where << ": " << p.error.message);
    return p.loader;
}

void collect(void* context, void* data, int size) {
    append(*static_cast<std::vector<uint8_t>*>(context), data, static_cast<std::size_t>(size));
}

}  // namespace

TEST_CASE("end to end: a Parquet file and a safetensors file become sources that survive a reopen") {
    const std::string ws = fresh_dir("e2e") + "/workspace";
    fs::create_directories(ws + "/data");
    write_parquet(ws + "/data", "metrics.parquet", "SELECT range AS step, 1.0 / (range + 1) AS loss FROM range(5)");
    write_bytes(ws + "/data/model.safetensors",
                make_safetensors(R"({"layers.0.attn":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
                                 R"("bias":{"dtype":"F16","shape":[2],"data_offsets":[24,28]}})",
                                 [] {
                                     std::vector<uint8_t> d = raw(std::vector<float>{1, 2, 3, 4, 5, 6});
                                     append(d, raw(std::vector<uint16_t>{0x3C00, 0xB800}));
                                     return d;
                                 }()));
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        std::vector<LoaderStore::ImportItem> items =
            store->import_files({ws + "/data/metrics.parquet", ws + "/data/model.safetensors"});
        REQUIRE(items.size() == 2);
        for (const auto& item : items) REQUIRE_MESSAGE(item.error.ok(), item.path << ": " << item.error.message);
        CHECK(items[0].loader_id == "metrics");
        CHECK(items[1].loader_id == "model");
        CHECK(fs::exists(ws + "/loaders/metrics.loader.toml"));
        CHECK(fs::exists(ws + "/loaders/model.loader.toml"));

        RunResult table = run_ok(*store, "metrics");
        REQUIRE(table.sources.size() == 1);
        RunResult tensors = run_ok(*store, "model");
        CHECK(tensors.sources.size() == 2);

        LoaderStore::SourceGet t = store->describe("table:metrics");
        REQUIRE(t.error.ok());
        const json entry = t.source.to_json();
        CHECK(entry["kind"] == "table");
        CHECK(entry["origin"] == json{{"loader", "metrics"}, {"file", "data/metrics.parquet"}});
        CHECK(entry["columns"] == json::parse(R"([{"name": "step", "type": "BIGINT"}, {"name": "loss", "type": "DOUBLE"}])"));
        CHECK(entry["rows"] == 5);
        CHECK(entry["dtype"].is_null());
        CHECK(entry["shape"].is_null());
        CHECK(entry["wire"] == true);
        CHECK(entry["updated_at"].get<std::string>().size() == 20);
        for (const char* key : {"id", "kind", "name", "title", "origin", "columns", "rows", "dtype", "shape", "rate_hz",
                                "bytes", "wire", "warnings", "updated_at"}) {
            CHECK_MESSAGE(entry.contains(key), key);
        }
        CHECK(entry.size() == 14);
        CHECK(t.source.table_mode == "view");

        LoaderStore::SourceGet w = store->describe("tensor:model_layers_0_attn");
        REQUIRE(w.error.ok());
        CHECK(w.source.to_json()["dtype"] == "f32");
        CHECK(w.source.to_json()["shape"] == json::array({2, 3}));
        CHECK(w.source.to_json()["bytes"] == 24);
        CHECK(w.source.title == "layers.0.attn");
        CHECK(w.source.tensor == "layers.0.attn");
        CHECK_FALSE(w.source.derived);

        LoaderStore::Sample s = store->sample("table:metrics", 2);
        REQUIRE(s.error.ok());
        CHECK(s.value["rows"] == json::parse("[[0, 1.0], [1, 0.5]]"));
        CHECK(s.value["truncated"] == true);
        s = store->sample("tensor:model_layers_0_attn");
        REQUIRE(s.error.ok());
        CHECK(s.value["stats"]["min"] == 1.0);
        CHECK(s.value["stats"]["max"] == 6.0);
        CHECK(s.value["stats"]["mean"] == 3.5);
        CHECK(s.value["slice"]["values"] == json::parse("[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"));
        QueryResult q = store->query("SELECT sum(step) FROM metrics");
        REQUIRE(q.error.ok());
        CHECK(q.rows == json::parse("[[10]]"));
    }
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        std::vector<LoadedSource> all = store->sources();
        CHECK(all.size() == 3);
        CHECK(find(all, "table:metrics"));
        CHECK(find(all, "tensor:model_bias"));
        CHECK(store->sources("tensor").size() == 2);
        CHECK(store->sources("", "model_l").size() == 1);
        CHECK(store->sources("", "", 1).size() == 1);
        TensorResult bias = store->load_tensor("tensor:model_bias");
        REQUIRE(bias.error.ok());
        CHECK(bias.tensor.element(1) == -0.5);
        LoaderStore::Sample s = store->sample("table:metrics", 10);
        REQUIRE(s.error.ok());
        CHECK(s.value["rows"].size() == 5);
        std::vector<LoaderRecord> loaders = store->list();
        REQUIRE(loaders.size() == 2);
        CHECK(loaders[0].last_status == "ok");
        CHECK(loaders[0].recipe.find("caliper_loader = 0") == 0);
        CHECK(store->describe("table:nothing").error.code == "catalog.not_found");
        CHECK(store->load_tensor("table:metrics").error.code == "catalog.not_a_tensor");
    }
    // The database is rebuilt from the workspace files.
    fs::remove(ws + "/workspace.duckdb");
    fs::remove(ws + "/workspace.duckdb.wal");
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        CHECK(store->sources().size() == 3);
        LoaderStore::Sample s = store->sample("tensor:model_layers_0_attn");
        REQUIRE(s.error.ok());
        CHECK(s.value["stats"]["max"] == 6.0);
        CHECK(store->query("SELECT count(*) FROM metrics").rows == json::parse("[[5]]"));
    }
}

TEST_CASE("save, get, run with SQL and transforms, and remove") {
    const std::string ws = fresh_dir("store_crud");
    write_text(ws + "/runs/a.csv", "step,loss\n0,2.0\n1,1.0\n");
    write_text(ws + "/runs/b.csv", "step,loss\n2,0.5\n");
    write_bytes(ws + "/w.npy", make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }",
                                        raw(std::vector<float>{1, 1, 1, 3})));
    std::unique_ptr<LoaderStore> store = open_store(ws);
    Loader csv = loader_from(R"(caliper_loader = 0
[loader]
id = "runs"
[input]
path = "runs/*.csv"
[[output]]
name = "late"
sql = "SELECT step, loss FROM input WHERE step > 0;"
[[output]]
name = "order"
mode = "table"
)");
    REQUIRE(store->save(csv).ok());
    LoaderStore::LoaderGet got = store->get("runs");
    REQUIRE(got.error.ok());
    CHECK(got.loader.to_json() == csv.to_json());
    CHECK(got.record.last_status == "never");
    RunResult r = run_ok(*store, "runs");
    REQUIRE(r.sources.size() == 2);
    CHECK(find(r.sources, "table:late")->rows == 2);
    const LoadedSource* order = find(r.sources, "table:order");
    CHECK(order->rows == 3);
    CHECK(order->table_mode == "table");
    CHECK(order->warnings.size() == 1);  // "order" is an SQL keyword
    CHECK(order->file == "runs/*.csv");
    CHECK(store->query("SELECT count(*) FROM \"order\"").rows == json::parse("[[3]]"));

    // A third file appears in the view when the loader runs again.
    write_text(ws + "/runs/c.csv", "step,loss\n3,0.25\n");
    run_ok(*store, "runs");
    CHECK(store->describe("table:late").source.rows == 3);

    Loader npy = loader_from(R"(caliper_loader = 0
[loader]
id = "w"
[input]
path = "w.npy"
[[output]]
name = "w_raw"
[[output]]
name = "w_entropy"
  [[output.transform]]
  op = "reduce"
  axis = 1
  fn = "entropy"
)");
    REQUIRE(store->save(npy).ok());
    r = run_ok(*store, "w");
    const LoadedSource* raw_src = find(r.sources, "tensor:w_raw");
    const LoadedSource* ent = find(r.sources, "tensor:w_entropy");
    REQUIRE(raw_src);
    REQUIRE(ent);
    CHECK_FALSE(raw_src->derived);
    CHECK(ent->derived);
    CHECK(ent->shape == std::vector<int64_t>{2});
    CHECK(fs::exists(ws + "/tensors/w_entropy.npy"));
    CHECK_FALSE(fs::exists(ws + "/tensors/w_entropy.npy.tmp"));
    TensorResult t = store->load_tensor("tensor:w_entropy");
    REQUIRE(t.error.ok());
    CHECK(t.tensor.element(0) == doctest::Approx(std::log(2.0)));
    CHECK(t.tensor.element(1) == doctest::Approx(-(0.25 * std::log(0.25) + 0.75 * std::log(0.75))));

    REQUIRE(store->remove("w").ok());
    CHECK_FALSE(fs::exists(ws + "/tensors/w_entropy.npy"));
    CHECK_FALSE(fs::exists(ws + "/loaders/w.loader.toml"));
    CHECK(store->describe("tensor:w_raw").error.code == "catalog.not_found");
    REQUIRE(store->remove("runs").ok());
    CHECK_FALSE(store->query("SELECT * FROM late").error.ok());
    CHECK(store->sources().empty());
    CHECK(store->list().empty());
    CHECK(store->remove("runs").code == "loader.not_found");
    CHECK(store->run("runs").error.code == "loader.not_found");
    CHECK(store->get("Bad Id").error.code == "loader.not_found");
}

TEST_CASE("a failed run keeps the previous sources and records the error") {
    const std::string ws = fresh_dir("store_fail");
    write_text(ws + "/m.csv", "step,loss\n0,1\n");
    std::unique_ptr<LoaderStore> store = open_store(ws);
    REQUIRE(store->save(loader_from(R"(caliper_loader = 0
[loader]
id = "m"
[input]
path = "m.csv"
[[output]]
name = "m_loss"
mode = "table"
sql = "SELECT loss FROM input"
)")).ok());
    run_ok(*store, "m");
    write_text(ws + "/m.csv", "step,other\n0,1\n");
    RunResult r = store->run("m");
    CHECK(r.error.code == "loader.sql");
    CHECK(store->describe("table:m_loss").error.ok());
    CHECK(store->query("SELECT count(*) FROM m_loss").rows == json::parse("[[1]]"));
    LoaderStore::LoaderGet got = store->get("m");
    CHECK(got.record.last_status == "error");
    CHECK(got.record.last_error_code == "loader.sql");
    CHECK_FALSE(got.record.last_run_at.empty());

    // A loader whose SQL is not a SELECT fails the same way.
    REQUIRE(store->save(loader_from(R"(caliper_loader = 0
[loader]
id = "bad"
[input]
path = "m.csv"
[[output]]
name = "bad_out"
sql = "DROP TABLE m_loss"
)")).ok());
    r = store->run("bad");
    CHECK(r.error.code == "loader.sql_not_select");
    CHECK(r.error.where == "output[0].sql");
    CHECK(store->query("SELECT count(*) FROM m_loss").rows == json::parse("[[1]]"));
}

TEST_CASE("source names belong to one loader") {
    const std::string ws = fresh_dir("store_clash");
    write_text(ws + "/a.csv", "x\n1\n");
    std::unique_ptr<LoaderStore> store = open_store(ws);
    const std::string recipe = "caliper_loader = 0\n[loader]\nid = \"%\"\n[input]\npath = \"a.csv\"\n[[output]]\nname = \"same\"\n";
    for (const char* id : {"one", "two"}) {
        std::string text = recipe;
        text.replace(text.find('%'), 1, id);
        REQUIRE(store->save(loader_from(text)).ok());
    }
    run_ok(*store, "one");
    RunResult r = store->run("two");
    CHECK(r.error.code == "loader.source_exists");
    CHECK(store->describe("table:same").source.loader == "one");
}

TEST_CASE("grants for files outside the workspace") {
    const std::string root = fresh_dir("store_grants");
    const std::string ws = root + "/ws";
    write_text(root + "/dropped/run.csv", "x\n1\n2\n");
    write_text(root + "/dropped/other.csv", "x\n9\n");
    write_text(root + "/folder/f.csv", "x\n5\n");
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        // A loader cannot name an outside file the user did not grant.
        Loader l = loader_from("caliper_loader = 0\n[loader]\nid = \"o\"\n[input]\npath = \"" + root +
                               "/dropped/other.csv\"\n[[output]]\nname = \"o\"\n");
        CHECK(store->save(l).code == "loader.path_not_allowed");
        CHECK(store->import_files({root + "/dropped/other.csv"}, false)[0].error.code == "loader.path_not_allowed");
        // Dropping a file grants it.
        std::vector<LoaderStore::ImportItem> items = store->import_files({root + "/dropped/run.csv"});
        REQUIRE(items[0].error.ok());
        CHECK(store->granted_files() == std::vector<std::string>{fs::canonical(root + "/dropped/run.csv").string()});
        RunResult r = run_ok(*store, items[0].loader_id);
        CHECK(r.sources[0].file == fs::canonical(root + "/dropped/run.csv").string());
        CHECK(store->query("SELECT count(*) FROM run").rows == json::parse("[[2]]"));
        // The SQL of a query still cannot read the file next to it.
        CHECK(store->query("SELECT * FROM read_csv(" + detail::sql_literal(root + "/dropped/other.csv") + ")").error.code ==
              "loader.path_not_allowed");
        REQUIRE(store->grant(root + "/folder").ok());
        CHECK(store->query("SELECT * FROM read_csv(" + detail::sql_literal(root + "/folder/f.csv") + ")").error.ok());
        CHECK(store->grant("/").code == "loader.path_not_allowed");
        CHECK(store->grant(root + "/nothing").code == "loader.io");
        CHECK(store->grant(ws).ok());  // inside the workspace: nothing to record
        CHECK(store->granted_folders().size() == 1);
    }
    std::string text;
    {
        std::ifstream in(ws + "/inputs.toml");
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    CHECK(text.find("caliper_inputs = 0") == 0);
    std::unique_ptr<LoaderStore> store = open_store(ws);
    CHECK(store->granted_files().size() == 1);
    CHECK(store->granted_folders().size() == 1);
    CHECK(store->query("SELECT count(*) FROM run").rows == json::parse("[[2]]"));
}

TEST_CASE("every reader through a run") {
    const std::string ws = fresh_dir("store_readers");
    write_text(ws + "/t.tsv", "a\tb\n1\tx\n");
    write_text(ws + "/r.json", R"([{"a": 1}, {"a": 2}])");
    write_text(ws + "/r.jsonl", "{\"a\": 1}\n");
    write_bytes(ws + "/n.npz", make_zip({{"x.npy", make_npy("{'descr': '<i2', 'fortran_order': False, 'shape': (2,), }",
                                                              raw(std::vector<int16_t>{1, 2}))},
                                         {"y.npy", make_npy("{'descr': '>i2', 'fortran_order': False, 'shape': (1,), }",
                                                            {0x01, 0x00})}}));
    WavSpec spec;
    spec.channels = 2;
    spec.rate = 16000;
    write_bytes(ws + "/s.wav", make_wav(spec, raw(std::vector<int16_t>{1, 2, 3, 4})));
    for (const char* name : {"img_a", "img_b"}) {
        std::vector<uint8_t> png;
        const std::vector<uint8_t> px = {10, 20, 30, 40};
        stbi_write_png_to_func(collect, &png, 2, 2, 1, px.data(), 2);
        write_bytes(ws + "/images/" + name + ".png", png);
    }
    write_bytes(ws + "/m.arrow", {'A', 'R', 'R', 'O', 'W', '1', 0, 0});
    write_bytes(ws + "/d.h5", {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'});
    write_bytes(ws + "/model.pt", {0x80, 0x02, '.'});
    write_text(ws + "/notes.txt", "hello");

    std::unique_ptr<LoaderStore> store = open_store(ws);
    std::vector<LoaderStore::ImportItem> items = store->import_files(
        {ws + "/t.tsv", ws + "/r.json", ws + "/r.jsonl", ws + "/n.npz", ws + "/s.wav", ws + "/m.arrow", ws + "/d.h5",
         ws + "/model.pt", ws + "/notes.txt", ws + "/missing.csv"});
    REQUIRE(items.size() == 10);
    for (int i = 0; i < 7; ++i) REQUIRE_MESSAGE(items[i].error.ok(), items[i].path << ": " << items[i].error.message);
    CHECK(items[7].error.code == "loader.pickle_refused");
    CHECK(items[8].error.code == "loader.no_reader");
    CHECK(items[9].error.code == "loader.io");
    CHECK(items[1].loader_id == "r");
    CHECK(items[2].loader_id == "r-2");

    CHECK(run_ok(*store, "t").sources[0].columns->size() == 2);
    CHECK(run_ok(*store, "r").sources[0].rows == 2);
    // Both JSON loaders name their table "r", so the second one clashes.
    CHECK(store->run("r-2").error.code == "loader.source_exists");
    RunResult npz = run_ok(*store, "n");
    CHECK(npz.sources.size() == 2);
    CHECK(find(npz.sources, "tensor:n_x"));
    CHECK(find(npz.sources, "tensor:n_y")->derived);
    CHECK(store->load_tensor("tensor:n_y").tensor.element(0) == 256);
    RunResult wav = run_ok(*store, "s");
    CHECK(*wav.sources[0].rate_hz == 16000);
    CHECK(wav.sources[0].to_json()["rate_hz"] == 16000.0);
    CHECK(wav.sources[0].shape == std::vector<int64_t>{2, 2});
    CHECK(store->run("m").error.code == "loader.arrow_not_available");
    CHECK(store->run("d").error.code == "loader.hdf5_not_available");

    // A glob of images makes one tensor per file.
    REQUIRE(store->save(loader_from("caliper_loader = 0\n[loader]\nid = \"imgs\"\n[input]\npath = \"images/*.png\"\n"
                                    "[reader]\nchannels = 3\n[[output]]\nname = \"img\"\n"))
                .ok());
    RunResult imgs = run_ok(*store, "imgs");
    REQUIRE(imgs.sources.size() == 2);
    CHECK(imgs.sources[0].id == "tensor:img_img_a");
    CHECK(imgs.sources[0].shape == std::vector<int64_t>{2, 2, 3});
    CHECK(imgs.sources[0].title == "img (img_a.png)");

    // A Python loader runs only with the caller's approval (test_python_loader.cpp has the rest).
    REQUIRE(store->save(loader_from("caliper_loader = 0\n[loader]\nid = \"py\"\n[input]\npath = \"t.tsv\"\n"
                                    "reader = \"python\"\n[reader]\nscript = \"s.py\"\n[[output]]\nname = \"py\"\n"
                                    "table = \"*\"\n"))
                .ok());
    CHECK(store->run("py").error.code == "loader.python_needs_approval");
    // A named tensor that the file lacks.
    REQUIRE(store->save(loader_from("caliper_loader = 0\n[loader]\nid = \"nz\"\n[input]\npath = \"n.npz\"\n"
                                    "[[output]]\nname = \"nz\"\ntensor = \"absent\"\n"))
                .ok());
    CHECK(store->run("nz").error.code == "loader.tensor_not_found");
    // A pickle that sneaks in under a reader's extension is still refused.
    write_bytes(ws + "/sneaky.npy", {0x80, 0x04, 0x95, 0, 0});
    REQUIRE(store->save(loader_from("caliper_loader = 0\n[loader]\nid = \"sneaky\"\n[input]\npath = \"sneaky.npy\"\n"
                                    "reader = \"npy\"\n[[output]]\nname = \"sneaky\"\n"))
                .ok());
    CHECK(store->run("sneaky").error.code == "loader.pickle_refused");
}

TEST_CASE("the rows follow the loader files") {
    const std::string ws = fresh_dir("store_sync");
    write_text(ws + "/a.csv", "x\n1\n");
    { open_store(ws); }
    // A loader file written by hand, and a broken one.
    write_text(ws + "/loaders/hand.loader.toml",
               "# written by hand\ncaliper_loader = 0\n[loader]\nid = \"hand\"\n[input]\npath = \"a.csv\"\n[[output]]\nname = \"hand\"\n");
    write_text(ws + "/loaders/broken.loader.toml", "caliper_loader = 0\n[loader\n");
    write_text(ws + "/loaders/other.loader.toml",
               "caliper_loader = 0\n[loader]\nid = \"different\"\n[input]\npath = \"a.csv\"\n[[output]]\nname = \"o\"\n");
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        std::vector<LoaderRecord> list = store->list();
        REQUIRE(list.size() == 3);
        CHECK(list[0].id == "broken");
        CHECK(list[0].last_status == "error");
        CHECK(list[0].last_error_code == "loader.parse");
        CHECK(list[1].id == "hand");
        CHECK(list[1].last_status == "never");
        CHECK(list[1].recipe.find("# written by hand") == 0);
        CHECK(list[2].last_error_code == "loader.bad_id");
        run_ok(*store, "hand");
        CHECK(store->run("broken").error.code == "loader.parse");
    }
    fs::remove(ws + "/loaders/hand.loader.toml");
    {
        std::unique_ptr<LoaderStore> store = open_store(ws);
        CHECK(store->list().size() == 2);
        CHECK(store->sources().empty());
        CHECK_FALSE(store->query("SELECT * FROM hand").error.ok());
    }
    // rebuild() runs every loader that parses.
    write_text(ws + "/loaders/hand.loader.toml",
               "caliper_loader = 0\n[loader]\nid = \"hand\"\n[input]\npath = \"a.csv\"\n[[output]]\nname = \"hand\"\n");
    std::unique_ptr<LoaderStore> store = open_store(ws);
    REQUIRE(store->rebuild().ok());
    CHECK(store->describe("table:hand").error.ok());
    CHECK(store->get("hand").record.last_status == "ok");
}
