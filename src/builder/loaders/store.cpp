// LoaderStore: saved loaders, their runs and their outputs in a workspace.
#include <caliper_builder/loaders/store.h>

#include <caliper_builder/loaders/inference.h>
#include <caliper_builder/loaders/paths.h>
#include <caliper_builder/loaders/pickle.h>
#include <caliper_builder/loaders/tensor_ops.h>

#include "db.h"
#include "file_io.h"
#include "tabular.h"

#include <toml++/toml.hpp>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace caliper_builder {

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace detail;

namespace {

const char* kLoaderSuffix = ".loader.toml";
constexpr std::size_t kMaxInputsText = 4u << 20;

std::string now_utc() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

Error read_text(const fs::path& p, std::size_t max, std::string* out) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(p, ec);
    if (ec) return Error("loader.io", "The file could not be read.", p.u8string());
    if (size > max) return Error("loader.too_large", "The file is larger than its limit.", p.u8string());
    std::ifstream in(p, std::ios::binary);
    if (!in) return Error("loader.io", "The file could not be opened.", p.u8string());
    std::ostringstream s;
    s << in.rdbuf();
    *out = s.str();
    return {};
}

Error protected_error(const fs::path& p) {
    return Error("loader.protected_file", "The file is, or leads to, a protected settings file, so it was not written.",
                 p.u8string());
}

// Removes a file or a folder with its contents, unless that would change a
// protected file (paths.h).
void remove_unprotected(const fs::path& p) {
    if (touches_protected_file(p.u8string())) return;
    std::error_code ec;
    fs::remove_all(p, ec);
}

Error write_text_atomic(const fs::path& p, const std::string& text) {
    const fs::path tmp = fs::u8path(p.u8string() + ".tmp");
    std::error_code rec;
    fs::remove(tmp, rec);   // a name already there, such as a link, is removed, not written through
    if (touches_protected_file(p.u8string()) || touches_protected_file(tmp.u8string())) return protected_error(p);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return Error("loader.io", "The file could not be created.", p.u8string());
        out << text;
        out.close();
        if (!out) return Error("loader.io", "The file could not be written.", p.u8string());
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) {
        fs::remove(p, ec);
        fs::rename(tmp, p, ec);
        if (ec) return Error("loader.io", "The file could not be replaced.", p.u8string());
    }
    return {};
}

std::string lit(const std::string& s) { return sql_literal(s); }
std::string lit_or_null(const std::string& s) { return s.empty() ? "NULL" : sql_literal(s); }

const char* kSourceColumns =
    "\"id\", \"kind\", \"name\", \"title\", \"loader\", \"file\", \"columns\", \"rows\", \"dtype\", \"shape\", "
    "\"rate_hz\", \"bytes\", \"wire\", \"warnings\", \"updated_at\", \"table_mode\", \"tensor\", \"data_file\", "
    "\"data_offset\", \"derived\"";

std::string str_at(const json& row, std::size_t i) { return row[i].is_string() ? row[i].get<std::string>() : ""; }

LoadedSource source_from_row(const json& row) {
    LoadedSource s;
    s.id = str_at(row, 0);
    s.kind = str_at(row, 1);
    s.name = str_at(row, 2);
    s.title = str_at(row, 3);
    s.loader = str_at(row, 4);
    s.file = str_at(row, 5);
    if (row[6].is_string()) {
        const json cols = json::parse(row[6].get<std::string>(), nullptr, false);
        if (cols.is_array()) {
            std::vector<ColumnInfo> list;
            for (const json& c : cols) {
                if (c.is_object() && c.value("name", json()).is_string() && c.value("type", json()).is_string()) {
                    list.push_back({c["name"].get<std::string>(), c["type"].get<std::string>()});
                }
            }
            s.columns = std::move(list);
        }
    }
    if (row[7].is_number_integer()) s.rows = row[7].get<int64_t>();
    DType d;
    if (row[8].is_string() && dtype_from_name(row[8].get<std::string>(), &d)) s.dtype = d;
    if (row[9].is_string()) {
        const json shape = json::parse(row[9].get<std::string>(), nullptr, false);
        if (shape.is_array()) {
            std::vector<int64_t> dims;
            for (const json& v : shape) {
                if (v.is_number_integer()) dims.push_back(v.get<int64_t>());
            }
            s.shape = std::move(dims);
        }
    }
    if (row[10].is_number()) s.rate_hz = row[10].get<double>();
    if (row[11].is_number_integer()) s.bytes = row[11].get<int64_t>();
    s.wire = row[12].is_boolean() ? row[12].get<bool>() : true;
    if (row[13].is_string()) {
        const json w = json::parse(row[13].get<std::string>(), nullptr, false);
        if (w.is_array()) {
            for (const json& v : w) {
                if (v.is_string()) s.warnings.push_back(v.get<std::string>());
            }
        }
    }
    s.updated_at = str_at(row, 14);
    s.table_mode = str_at(row, 15);
    s.tensor = str_at(row, 16);
    s.data_file = str_at(row, 17);
    if (row[18].is_number_integer()) s.data_offset = row[18].get<int64_t>();
    s.derived = row[19].is_boolean() && row[19].get<bool>();
    return s;
}

std::string source_values(const LoadedSource& s) {
    std::string columns = "NULL", shape = "NULL";
    if (s.columns) {
        json c = json::array();
        for (const ColumnInfo& col : *s.columns) c.push_back(col.to_json());
        columns = lit(dump(c));
    }
    if (s.shape) shape = lit(dump(json(*s.shape)));
    std::string rate = "NULL";
    if (s.rate_hz) {
        std::ostringstream o;
        o.precision(17);
        o << *s.rate_hz;
        rate = o.str();
    }
    return "(" + lit(s.id) + ", " + lit(s.kind) + ", " + lit(s.name) + ", " + lit(s.title) + ", " + lit(s.loader) +
           ", " + lit(s.file) + ", " + columns + ", " + (s.rows ? std::to_string(*s.rows) : "NULL") + ", " +
           (s.dtype ? lit(dtype_name(*s.dtype)) : "NULL") + ", " + shape + ", " + rate + ", " +
           (s.bytes ? std::to_string(*s.bytes) : "NULL") + ", " + (s.wire ? "true" : "false") + ", " +
           lit(dump(json(s.warnings))) + ", " + lit(s.updated_at) + ", " + lit_or_null(s.table_mode) + ", " +
           lit_or_null(s.tensor) + ", " + lit_or_null(s.data_file) + ", " + std::to_string(s.data_offset) + ", " +
           (s.derived ? "true" : "false") + ")";
}

LoaderRecord record_from_row(const json& row) {
    LoaderRecord r;
    r.id = str_at(row, 0);
    r.title = str_at(row, 1);
    r.recipe = str_at(row, 2);
    r.created_at = str_at(row, 3);
    r.updated_at = str_at(row, 4);
    r.last_run_at = str_at(row, 5);
    r.last_status = str_at(row, 6);
    r.last_error_code = str_at(row, 7);
    r.last_error_message = str_at(row, 8);
    return r;
}

const char* kLoaderColumns =
    "\"id\", \"title\", \"recipe\", \"created_at\", \"updated_at\", \"last_run_at\", \"last_status\", "
    "\"last_error_code\", \"last_error_message\"";

// A derived tensor written under a temporary name, renamed once the run commits.
struct PendingFile {
    fs::path temp;
    fs::path final;
};

}  // namespace

json LoaderRecord::to_json() const {
    auto opt = [](const std::string& s) { return s.empty() ? json(nullptr) : json(s); };
    return {{"id", id},
            {"title", title},
            {"recipe", recipe},
            {"created_at", created_at},
            {"updated_at", updated_at},
            {"last_run_at", opt(last_run_at)},
            {"last_status", last_status},
            {"last_error_code", opt(last_error_code)},
            {"last_error_message", opt(last_error_message)}};
}

json LoadedSource::to_json() const {
    json j;
    j["id"] = id;
    j["kind"] = kind;
    j["name"] = name;
    j["title"] = title;
    j["origin"] = {{"loader", loader}, {"file", file}};
    if (columns) {
        json c = json::array();
        for (const ColumnInfo& col : *columns) c.push_back(col.to_json());
        j["columns"] = c;
    } else {
        j["columns"] = nullptr;
    }
    j["rows"] = rows ? json(*rows) : json(nullptr);
    j["dtype"] = dtype ? json(dtype_name(*dtype)) : json(nullptr);
    j["shape"] = shape ? json(*shape) : json(nullptr);
    j["rate_hz"] = rate_hz ? json(*rate_hz) : json(nullptr);
    j["bytes"] = bytes ? json(*bytes) : json(nullptr);
    j["wire"] = wire;
    j["warnings"] = warnings;
    j["updated_at"] = updated_at;
    return j;
}

json LoadedSource::storage_json() const {
    auto opt = [](const std::string& s) { return s.empty() ? json(nullptr) : json(s); };
    return {{"table_mode", opt(table_mode)},
            {"tensor", opt(tensor)},
            {"data_file", opt(data_file)},
            {"data_offset", data_offset},
            {"derived", derived}};
}

struct LoaderStore::Impl {
    std::mutex mutex;
    std::string workspace;
    StoreOptions options;
    std::vector<std::string> granted_files;
    std::vector<std::string> granted_folders;
    std::unique_ptr<Db> db;

    fs::path ws() const { return fs::u8path(workspace); }
    fs::path loader_file(const std::string& id) const { return ws() / "loaders" / fs::u8path(id + kLoaderSuffix); }

    AllowedRoots roots() const { return AllowedRoots{workspace, granted_folders, granted_files}; }

    Error open_db() {
        db.reset();
        DbOptions o;
        o.allowed_directories = {workspace};
        for (const std::string& f : granted_folders) o.allowed_directories.push_back(f);
        o.allowed_files = granted_files;
        o.temp_directory = (ws() / "workspace.duckdb.tmp").u8string();
        o.threads = options.threads;
        o.memory_limit = options.memory_limit;
        if (touches_protected_file((ws() / "workspace.duckdb").u8string())) return protected_error(ws() / "workspace.duckdb");
        return Db::open((ws() / "workspace.duckdb").u8string(), o, &db);
    }

    Error ensure_schema() {
        return db->execute(
            "CREATE SCHEMA IF NOT EXISTS caliper;"
            "CREATE TABLE IF NOT EXISTS caliper.loaders (\"id\" VARCHAR PRIMARY KEY, \"title\" VARCHAR, "
            "\"recipe\" VARCHAR, \"created_at\" VARCHAR, \"updated_at\" VARCHAR, \"last_run_at\" VARCHAR, "
            "\"last_status\" VARCHAR, \"last_error_code\" VARCHAR, \"last_error_message\" VARCHAR);"
            "CREATE TABLE IF NOT EXISTS caliper.sources (\"id\" VARCHAR PRIMARY KEY, \"kind\" VARCHAR, "
            "\"name\" VARCHAR, \"title\" VARCHAR, \"loader\" VARCHAR, \"file\" VARCHAR, \"columns\" VARCHAR, "
            "\"rows\" BIGINT, \"dtype\" VARCHAR, \"shape\" VARCHAR, \"rate_hz\" DOUBLE, \"bytes\" BIGINT, "
            "\"wire\" BOOLEAN, \"warnings\" VARCHAR, \"updated_at\" VARCHAR, \"table_mode\" VARCHAR, "
            "\"tensor\" VARCHAR, \"data_file\" VARCHAR, \"data_offset\" BIGINT, \"derived\" BOOLEAN);");
    }

    bool schema_exists() {
        int64_t n = 0;
        return !db->scalar("SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'caliper'", &n) &&
               n > 0;
    }

    // inputs.toml: caliper_inputs = 0, files = [...], folders = [...]
    Error read_inputs() {
        const fs::path p = ws() / "inputs.toml";
        std::error_code ec;
        if (!fs::exists(p, ec)) return {};
        std::string text;
        if (Error e = read_text(p, kMaxInputsText, &text)) return e;
        toml::table t;
#if TOML_EXCEPTIONS
        try {
            t = toml::parse(text);
        } catch (const toml::parse_error& e) {
            return Error("loader.parse", "inputs.toml is not valid TOML: " + std::string(e.description()) + ".",
                         std::to_string(e.source().begin.line) + ":" + std::to_string(e.source().begin.column));
        }
#else
        toml::parse_result parsed = toml::parse(text);
        if (!parsed) return Error("loader.parse", "inputs.toml is not valid TOML.", p.u8string());
        t = std::move(parsed).table();
#endif
        for (auto&& [k, v] : t) {
            const std::string key(k.str());
            if (key == "caliper_inputs") {
                if (v.value<int64_t>().value_or(-1) != 0) {
                    return Error("loader.unsupported", "This Caliper reads inputs.toml version 0 only.", "inputs.toml");
                }
            } else if (key == "files" || key == "folders") {
                const toml::array* a = v.as_array();
                if (!a) return Error("loader.bad_value", "`" + key + "` in inputs.toml is a list of paths.", "inputs.toml");
                for (const toml::node& n : *a) {
                    std::optional<std::string> s = n.value<std::string>();
                    if (!s || !fs::u8path(*s).is_absolute()) {
                        return Error("loader.bad_value", "`" + key + "` in inputs.toml lists absolute paths.", "inputs.toml");
                    }
                    (key == "files" ? granted_files : granted_folders).push_back(*s);
                }
            } else {
                return Error("loader.unknown_key", "The key `" + key + "` is not part of inputs.toml.", "inputs.toml");
            }
        }
        return {};
    }

    Error write_inputs() {
        auto list = [](const std::vector<std::string>& v) {
            std::string out = "[";
            for (std::size_t i = 0; i < v.size(); ++i) {
                std::string q = "\"";
                for (char c : v[i]) {
                    if (c == '"' || c == '\\') q += '\\';
                    q += c;
                }
                out += (i ? ",\n  " : "\n  ") + q + "\"";
            }
            return out + (v.empty() ? "]" : ",\n]");
        };
        return write_text_atomic(ws() / "inputs.toml", "caliper_inputs = 0\nfiles = " + list(granted_files) +
                                                           "\nfolders = " + list(granted_folders) + "\n");
    }

    std::vector<LoadedSource> select_sources(const std::string& where, const std::string& tail = "") {
        json rows;
        std::vector<LoadedSource> out;
        if (db->rows(std::string("SELECT ") + kSourceColumns + " FROM caliper.sources" +
                         (where.empty() ? "" : " WHERE " + where) + " ORDER BY \"id\"" + tail,
                     &rows)) {
            return out;
        }
        for (const json& row : rows) out.push_back(source_from_row(row));
        return out;
    }

    bool loader_row(const std::string& id, LoaderRecord* out) {
        json rows;
        if (db->rows(std::string("SELECT ") + kLoaderColumns + " FROM caliper.loaders WHERE \"id\" = " + lit(id), &rows) ||
            rows.empty()) {
            return false;
        }
        if (out) *out = record_from_row(rows[0]);
        return true;
    }

    Error upsert_loader(const std::string& id, const std::string& title, const std::string& recipe,
                        const std::string& now) {
        return db->execute(
            "INSERT INTO caliper.loaders (\"id\", \"title\", \"recipe\", \"created_at\", \"updated_at\", "
            "\"last_status\") VALUES (" +
            lit(id) + ", " + lit(title) + ", " + lit(recipe) + ", " + lit(now) + ", " + lit(now) +
            ", 'never') ON CONFLICT (\"id\") DO UPDATE SET \"title\" = excluded.\"title\", \"recipe\" = "
            "excluded.\"recipe\", \"updated_at\" = excluded.\"updated_at\"");
    }

    Error set_status(const std::string& id, const std::string& now, const Error& e) {
        return db->execute("UPDATE caliper.loaders SET \"last_run_at\" = " + lit(now) + ", \"last_status\" = " +
                           (e ? std::string("'error'") : std::string("'ok'")) + ", \"last_error_code\" = " +
                           lit_or_null(e.code) + ", \"last_error_message\" = " + lit_or_null(e.message) +
                           " WHERE \"id\" = " + lit(id));
    }

    // Drops the relations of a loader's table sources, deletes its rows, and
    // removes its derived tensor files.
    Error remove_sources_of(const std::string& id) {
        const std::vector<LoadedSource> old = select_sources("\"loader\" = " + lit(id));
        for (const LoadedSource& s : old) {
            if (s.kind == "table") {
                if (Error e = db->drop_relation(s.name)) return e;
            }
        }
        if (Error e = db->execute("DELETE FROM caliper.sources WHERE \"loader\" = " + lit(id))) return e;
        for (const LoadedSource& s : old) {
            if (s.derived && !s.data_file.empty()) {
                std::error_code ec;
                const fs::path p = fs::u8path(s.data_file);
                if (path_within(p.u8string(), (ws() / "tensors").u8string())) fs::remove(p, ec);
            }
        }
        return {};
    }

    std::vector<std::string> loader_file_ids() {
        std::vector<std::string> ids;
        std::error_code ec;
        for (fs::directory_iterator it(ws() / "loaders", ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().u8string();
            const std::string suffix = kLoaderSuffix;
            if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                it->is_regular_file(ec)) {
                ids.push_back(name.substr(0, name.size() - suffix.size()));
            }
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    // Inserts or refreshes the row of a loader file, keeping its run status.
    Error index_loader_file(const std::string& id, const std::string& now, bool* parsed_ok) {
        std::string text;
        if (Error e = read_text(loader_file(id), kMaxLoaderText, &text)) return e;
        LoaderParse parsed = parse_loader(text);
        std::string title = id;
        Error problem = parsed.error;
        if (!problem && parsed.loader.id != id) {
            problem = Error("loader.bad_id", "The loader's id differs from its file name.", loader_file(id).u8string());
        }
        if (!problem) title = parsed.loader.title.empty() ? parsed.loader.id : parsed.loader.title;
        if (Error e = upsert_loader(id, title, text, now)) return e;
        if (problem) {
            if (Error e = set_status(id, now, problem)) return e;
        }
        if (parsed_ok) *parsed_ok = !problem;
        return {};
    }

    // Brings the rows in line with the files: rows without a file go, files
    // without a row (or with changed text) are indexed.
    Error sync() {
        const std::vector<std::string> files = loader_file_ids();
        json rows;
        if (Error e = db->rows("SELECT \"id\", \"recipe\" FROM caliper.loaders", &rows)) return e;
        std::map<std::string, std::string> recipes;
        for (const json& r : rows) recipes[str_at(r, 0)] = str_at(r, 1);
        const std::string now = now_utc();
        for (const auto& [id, recipe] : recipes) {
            if (std::find(files.begin(), files.end(), id) == files.end()) {
                if (Error e = remove_sources_of(id)) return e;
                if (Error e = db->execute("DELETE FROM caliper.loaders WHERE \"id\" = " + lit(id))) return e;
                remove_python_outputs(id);
            }
        }
        for (const std::string& id : files) {
            if (!valid_id(id)) continue;
            auto it = recipes.find(id);
            std::string text;
            if (it != recipes.end() && !read_text(loader_file(id), kMaxLoaderText, &text) && text == it->second) continue;
            if (Error e = index_loader_file(id, now, nullptr)) return e;
        }
        return {};
    }

    Error rebuild() {
        for (const LoadedSource& s : select_sources("\"kind\" = 'table'")) {
            if (Error e = db->drop_relation(s.name)) return e;
        }
        if (Error e = db->execute("DELETE FROM caliper.sources; DELETE FROM caliper.loaders;")) return e;
        const std::string now = now_utc();
        const std::vector<std::string> ids = loader_file_ids();
        for (const std::string& id : ids) {
            if (!valid_id(id)) continue;
            bool ok = false;
            if (Error e = index_loader_file(id, now, &ok)) return e;
            if (ok) run(id, RunOptions{}, nullptr, true);
        }
        // Kept outputs of loaders that are gone, or are no longer Python loaders, are removed.
        std::vector<fs::path> orphans;
        std::error_code oec;
        for (fs::directory_iterator it(ws() / "loader-outputs", oec), end; !oec && it != end; it.increment(oec)) {
            const std::string name = it->path().filename().u8string();
            bool keep = false;
            std::string text;
            if (std::find(ids.begin(), ids.end(), name) != ids.end() &&
                !read_text(loader_file(name), kMaxLoaderText, &text)) {
                LoaderParse parsed = parse_loader(text);
                keep = parsed.error || (parsed.loader.reader && *parsed.loader.reader == ReaderKind::python);
            }
            if (!keep) orphans.push_back(it->path());
        }
        for (const fs::path& p : orphans) remove_unprotected(p);
        // Derived tensors that no source uses any more are removed.
        std::set<std::string> used;
        for (const LoadedSource& s : select_sources("\"derived\"")) used.insert(s.data_file);
        std::error_code ec;
        for (fs::directory_iterator it(ws() / "tensors", ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code rec;
            const std::string p = fs::canonical(it->path(), rec).u8string();
            if (!rec && !used.count(p) && it->is_regular_file(rec) && !touches_protected_file(it->path().u8string()))
                fs::remove(it->path(), rec);
        }
        return {};
    }

    // One tensor source: its transforms applied, and its bytes either mapped
    // where they lie or written as a derived tensor under a temporary name.
    Error add_tensor(const Loader& l, std::size_t oi, const LoaderOutput& o, const std::string& name,
                     const std::string& title, const ReadTensor& t, const std::string& shown,
                     const std::vector<std::string>& extra_warnings, const std::string& now,
                     std::set<std::string>* names, std::vector<LoadedSource>* out, std::vector<PendingFile>* pending) {
        if (!names->insert(name).second) {
            return Error("loader.name_collision", "Two tensors of the loader become the source tensor:" + name + ".");
        }
        if (out->size() >= options.max_sources_per_run) {
            return Error("loader.too_large", "The run makes more sources than its limit of " +
                                                 std::to_string(options.max_sources_per_run) + ".");
        }
        TensorData data = t.data;
        bool derived = t.derived;
        if (!o.transforms.empty()) {
            TensorResult tr = apply_transforms(data, o.transforms);
            if (tr.error) {
                tr.error.where = "output[" + std::to_string(oi) + "]." + tr.error.where;
                return tr.error;
            }
            data = std::move(tr.tensor);
            derived = true;
        }
        LoadedSource s;
        s.id = "tensor:" + name;
        s.kind = "tensor";
        s.name = name;
        s.title = title;
        s.loader = l.id;
        s.file = shown;
        s.dtype = data.dtype();
        s.shape = data.shape();
        s.rate_hz = t.rate_hz;
        s.bytes = static_cast<int64_t>(data.nbytes());
        s.warnings = t.warnings;
        s.warnings.insert(s.warnings.end(), extra_warnings.begin(), extra_warnings.end());
        s.updated_at = now;
        s.tensor = t.name;
        s.derived = derived;
        if (derived) {
            const fs::path final = ws() / "tensors" / fs::u8path(name + ".npy");
            const fs::path temp = fs::u8path(final.u8string() + ".tmp");
            std::error_code rec;
            fs::remove(temp, rec);   // a name already there, such as a link, is removed, not written through
            if (touches_protected_file(final.u8string()) || touches_protected_file(temp.u8string()))
                return protected_error(final);
            uint64_t offset = 0;
            if (Error e = write_npy(temp.u8string(), data, &offset)) return e;
            pending->push_back({temp, final});
            s.data_file = final.u8string();
            s.data_offset = static_cast<int64_t>(offset);
        } else {
            s.data_file = data.file();
            s.data_offset = static_cast<int64_t>(data.file_offset());
        }
        out->push_back(std::move(s));
        return {};
    }

    // One table source: a view or table named `name` over the files of `read`.
    Error add_table(const Loader& l, std::size_t i, const LoaderOutput& o, const std::string& name,
                    const std::string& title, const TabularRead& read, int64_t bytes,
                    const std::vector<std::string>& extra_warnings, const std::string& now,
                    std::vector<LoadedSource>* made) {
        if (!db->relation_type(name).empty() || !select_sources("\"id\" = " + lit("table:" + name)).empty()) {
            return Error("loader.source_exists",
                         "A table or view named " + name + " exists already and is not this loader's.",
                         "output[" + std::to_string(i) + "].name");
        }
        if (made->size() >= options.max_sources_per_run) {
            return Error("loader.too_large", "The run makes more sources than its limit of " +
                                                 std::to_string(options.max_sources_per_run) + ".");
        }
        TableMade table;
        if (Error e = make_table_output(*db, read, TableOutput{name, o.mode && *o.mode == "table", o.sql}, &table)) {
            if (!e.where.empty() && e.where.rfind("output", 0) != 0) {
                e.where = "output[" + std::to_string(i) + "]." + e.where;
            }
            return e;
        }
        LoadedSource s;
        s.id = "table:" + name;
        s.kind = "table";
        s.name = name;
        s.title = title;
        s.loader = l.id;
        s.file = l.path;
        s.columns = table.columns;
        s.rows = table.rows;
        s.bytes = bytes;
        s.updated_at = now;
        s.table_mode = o.mode.value_or("view");
        s.warnings = extra_warnings;
        if (db->needs_quotes(name)) {
            s.warnings.push_back("The name " + name + " is an SQL keyword; quote it in SQL, as \"" + name + "\".");
        }
        made->push_back(std::move(s));
        return {};
    }

    // The tensor sources of one loader, with derived files written to temporary names.
    Error tensor_outputs(const Loader& l, ReaderKind reader, const std::vector<std::string>& files,
                         const std::string& now, std::vector<LoadedSource>* out, std::vector<PendingFile>* pending,
                         std::vector<std::string>* warnings) {
        const bool multi = files.size() > 1;
        bool want_all = false;
        std::set<std::string> wanted;
        for (const LoaderOutput& o : l.outputs) {
            if (!o.tensor || *o.tensor == "*") want_all = true;
            else wanted.insert(*o.tensor);
        }
        TensorFilter filter;
        if (!want_all) filter = [&wanted](const std::string& n) { return wanted.count(n) > 0; };
        std::set<std::string> names;
        for (const std::string& file : files) {
            TensorReadResult read;
            const ReaderLimits& limits = options.limits;
            switch (reader) {
                case ReaderKind::npy: read = read_npy(file, limits); break;
                case ReaderKind::npz: read = read_npz(file, filter, limits); break;
                case ReaderKind::safetensors: read = read_safetensors(file, filter, limits); break;
                case ReaderKind::image:
                    read = read_image(file, static_cast<int>(l.reader_options.value("channels", 0)), limits);
                    break;
                case ReaderKind::wav: read = read_wav(file, limits); break;
                case ReaderKind::hdf5:
                    read = read_hdf5(file, l.reader_options.value("dataset", std::string()), filter, limits);
                    break;
                default: return Error("loader.bad_value", "The reader does not make tensors.");
            }
            if (read.error) return read.error;
            const std::string shown = display_path(file, workspace);
            for (const std::string& w : read.warnings) warnings->push_back(shown + ": " + w);
            const std::string stem = fs::u8path(file).stem().u8string();
            const std::string filename = fs::u8path(file).filename().u8string();
            for (std::size_t oi = 0; oi < l.outputs.size(); ++oi) {
                const LoaderOutput& o = l.outputs[oi];
                const bool star = o.tensor && *o.tensor == "*";
                std::vector<const ReadTensor*> chosen;
                for (const ReadTensor& t : read.tensors) {
                    if (!o.tensor || star || t.name == *o.tensor) chosen.push_back(&t);
                }
                if (o.tensor && !star && chosen.empty()) {
                    return Error("loader.tensor_not_found", "The file has no tensor named " + *o.tensor + ".", shown);
                }
                for (const ReadTensor* t : chosen) {
                    std::string name;
                    std::string title;
                    if (star) {
                        name = make_id(o.name + (multi ? stem + "_" : "") + t->name);
                        title = multi ? t->name + " (" + filename + ")" : t->name;
                    } else {
                        name = multi ? make_id(o.name + "_" + stem) : o.name;
                        title = (o.title.empty() ? o.name : o.title) + (multi ? " (" + filename + ")" : "");
                    }
                    if (Error e = add_tensor(l, oi, o, name, title, *t, shown, {}, now, &names, out, pending)) return e;
                }
            }
        }
        return {};
    }

    // ---- Python loaders: their folders under loader-outputs/<id>/ ----

    fs::path python_root(const std::string& id) const { return ws() / "loader-outputs" / fs::u8path(id); }

    // The run folders of a Python loader: run-<n> holds a run's checked
    // outputs, tmp-<n> a run in progress.
    struct RunFolder {
        uint64_t n = 0;
        fs::path path;
        bool kept = false;
    };
    std::vector<RunFolder> python_folders(const std::string& id) const {
        std::vector<RunFolder> found;
        std::error_code ec;
        for (fs::directory_iterator it(python_root(id), ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().u8string();
            const bool kept = name.rfind("run-", 0) == 0;
            if (!kept && name.rfind("tmp-", 0) != 0) continue;
            const std::string digits = name.substr(4);
            if (digits.empty() || digits.size() > 18 ||
                !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) {
                continue;
            }
            std::error_code sec;
            const fs::file_status st = fs::symlink_status(it->path(), sec);
            if (sec || fs::is_symlink(st) || !fs::is_directory(st)) continue;
            found.push_back({std::stoull(digits), it->path(), kept});
        }
        return found;
    }

    // The newest run-<n> folder with a manifest, or an empty path.
    fs::path latest_kept(const std::string& id) const {
        fs::path best;
        uint64_t best_n = 0;
        for (const RunFolder& f : python_folders(id)) {
            std::error_code ec;
            if (f.kept && (best.empty() || f.n > best_n) && fs::is_regular_file(f.path / "manifest.json", ec)) {
                best = f.path;
                best_n = f.n;
            }
        }
        return best;
    }

    // Removes every run-<n> folder but `keep`, and with `stale` every tmp-<n> too.
    void prune_python(const std::string& id, const fs::path& keep, bool stale) {
        for (const RunFolder& f : python_folders(id)) {
            if ((f.kept && f.path != keep) || (!f.kept && stale)) remove_unprotected(f.path);
        }
    }

    void remove_python_outputs(const std::string& id) {
        if (!valid_id(id)) return;
        remove_unprotected(python_root(id));
    }

    // A Python loader's script: a regular file inside the workspace (not a
    // grant), with no ".." segment and no symbolic link that leads out.
    Error resolve_script(const std::string& script, bool must_exist, std::string* canonical) const {
        fs::path p = fs::u8path(script);
        for (const fs::path& part : p) {
            if (part == "..") {
                return Error("loader.path_not_allowed", "The script path of a Python loader may not use `..`.",
                             "reader.script");
            }
        }
        if (!p.is_absolute()) p = ws() / p;
        std::error_code ec;
        const fs::path c = fs::canonical(p, ec);
        if (ec) {
            if (must_exist) {
                return Error("loader.io", "The Python loader's script " + script + " does not exist.", "reader.script");
            }
            if (!path_within(p.lexically_normal().u8string(), workspace)) {
                return Error("loader.path_not_allowed", "The script of a Python loader must lie inside the workspace.",
                             "reader.script");
            }
            return {};
        }
        if (!path_within(c.u8string(), workspace)) {
            return Error("loader.path_not_allowed", "The script of a Python loader must lie inside the workspace.",
                         "reader.script");
        }
        if (!fs::is_regular_file(c, ec)) {
            return Error("loader.io", "The Python loader's script " + script + " is not a regular file.",
                         "reader.script");
        }
        if (canonical) *canonical = c.u8string();
        return {};
    }

    // The sources of a Python loader from the checked outputs in `kept`: each
    // [[output]] selects tensors or tables of the manifest by name or "*", and
    // the native readers read them.
    Error python_sources(const Loader& l, const fs::path& kept, const PythonManifest& m, const std::string& now,
                         std::vector<LoadedSource>* made, std::vector<PendingFile>* pending,
                         std::vector<std::string>* warnings) {
        std::set<std::string> tensor_names, table_names;
        std::vector<bool> used(m.outputs.size(), false);
        for (std::size_t oi = 0; oi < l.outputs.size(); ++oi) {
            const LoaderOutput& o = l.outputs[oi];
            const bool want_tensor = o.tensor.has_value();
            const std::string kind = want_tensor ? "tensor" : "table";
            const std::string& wanted = want_tensor ? *o.tensor : *o.table;
            const bool star = wanted == "*";
            bool found = false;
            for (std::size_t mi = 0; mi < m.outputs.size(); ++mi) {
                const PythonOutput& mo = m.outputs[mi];
                if (mo.kind != kind || !(star || mo.name == wanted)) continue;
                found = true;
                used[mi] = true;
                std::error_code ec;
                const fs::path c = fs::canonical(kept / fs::u8path(mo.file), ec);
                if (ec || !path_within(c.u8string(), workspace)) {
                    return Error("loader.path_not_allowed", "A Python loader's output lies outside the workspace.",
                                 mo.file);
                }
                const std::string file = c.u8string();
                if (Error e = check_not_pickle(file, options.limits)) return e;
                std::vector<std::string> extra = mo.warnings;
                extra.insert(extra.end(), m.warnings.begin(), m.warnings.end());
                const std::string name = star ? make_id(o.name + mo.name) : o.name;
                const std::string title =
                    star ? (mo.title.empty() ? mo.name : mo.title) : (o.title.empty() ? o.name : o.title);
                if (want_tensor) {
                    TensorReadResult read = read_npy(file, options.limits);
                    if (read.error) return read.error;
                    if (read.tensors.size() != 1 || !mo.dtype || !mo.shape ||
                        read.tensors[0].data.dtype() != *mo.dtype || read.tensors[0].data.shape() != *mo.shape) {
                        return Error("loader.malformed",
                                     "The tensor file " + mo.file + " does not hold the dtype and shape that the "
                                                                    "Python loader's manifest declares.",
                                     mo.file);
                    }
                    ReadTensor t = read.tensors[0];
                    t.name = mo.name;
                    if (Error e = add_tensor(l, oi, o, name, title, t, l.path, extra, now, &tensor_names, made, pending)) {
                        return e;
                    }
                } else {
                    TabularRead read{ReaderKind::parquet, {file}, json::object(), {}};
                    if (fs::u8path(mo.file).extension() == ".csv") {
                        TypedCsvHeader header = read_typed_csv_header(file, options.python_limits);
                        if (header.error) return header.error;
                        read.reader = ReaderKind::csv;
                        read.csv_columns = std::move(header.columns);
                    }
                    if (!table_names.insert(name).second) {
                        return Error("loader.name_collision", "Two tables of the loader become the source table:" +
                                                                  name + ".");
                    }
                    if (Error e = add_table(l, oi, o, name, title, read, static_cast<int64_t>(mo.bytes), extra, now,
                                            made)) {
                        return e;
                    }
                }
            }
            if (!star && !found) {
                return Error(want_tensor ? "loader.tensor_not_found" : "loader.table_not_found",
                             "The Python loader's script wrote no " + kind + " named " + wanted + ".",
                             "output[" + std::to_string(oi) + "]." + kind);
            }
        }
        for (std::size_t mi = 0; mi < m.outputs.size(); ++mi) {
            if (!used[mi]) {
                warnings->push_back("The script wrote the " + m.outputs[mi].kind + " " + m.outputs[mi].name +
                                    ", which no output of the loader selects.");
            }
        }
        return {};
    }

    // ---- runs ----

    RunResult failed(const std::string& id, const std::string& now, Error e) {
        set_status(id, now, e);
        RunResult res;
        res.error = std::move(e);
        return res;
    }

    using MakeSources =
        std::function<Error(std::vector<LoadedSource>*, std::vector<PendingFile>*, std::vector<std::string>*)>;

    // Replaces a loader's sources in one transaction with those `make` creates,
    // then renames the derived tensors into place and deletes the previous ones.
    // A failure rolls everything back and keeps the previous sources whole.
    RunResult commit_run(const Loader& l, const std::string& now, const MakeSources& make) {
        RunResult res;
        const std::string& id = l.id;
        std::vector<LoadedSource> made;
        std::vector<PendingFile> pending;
        auto discard = [&] {
            std::error_code ec;
            for (const PendingFile& p : pending) fs::remove(p.temp, ec);
        };
        const std::vector<LoadedSource> old = select_sources("\"loader\" = " + lit(id));
        if (Error e = db->begin()) return failed(id, now, e);
        Error err = [&]() -> Error {
            for (const LoadedSource& s : old) {
                if (s.kind == "table") {
                    if (Error e = db->drop_relation(s.name)) return e;
                }
            }
            if (Error e = db->execute("DELETE FROM caliper.sources WHERE \"loader\" = " + lit(id))) return e;
            if (Error e = make(&made, &pending, &res.warnings)) return e;
            for (const LoadedSource& s : made) {
                if (s.kind == "tensor" && !select_sources("\"id\" = " + lit(s.id)).empty()) {
                    return Error("loader.source_exists", "The source " + s.id + " belongs to another loader.");
                }
            }
            for (const LoadedSource& s : made) {
                if (Error e = db->execute(std::string("INSERT INTO caliper.sources (") + kSourceColumns +
                                          ") VALUES " + source_values(s))) {
                    return e;
                }
            }
            return set_status(id, now, Error());
        }();
        if (err) {
            db->rollback();
            discard();
            return failed(id, now, err);
        }
        if (Error e = db->commit()) {
            db->rollback();
            discard();
            return failed(id, now, e);
        }
        std::set<std::string> kept;
        for (const PendingFile& p : pending) {
            std::error_code ec;
            fs::rename(p.temp, p.final, ec);
            if (ec) {
                fs::remove(p.final, ec);
                fs::rename(p.temp, p.final, ec);
            }
            kept.insert(p.final.u8string());
        }
        for (const LoadedSource& s : old) {
            if (s.derived && !kept.count(s.data_file) && path_within(s.data_file, (ws() / "tensors").u8string())) {
                std::error_code ec;
                fs::remove(fs::u8path(s.data_file), ec);
            }
        }
        res.sources = std::move(made);
        return res;
    }

    // A Python loader. With `reuse_kept` (a rebuild) the script does not run:
    // the sources are read back from the newest kept outputs, if there are any.
    // Otherwise the script runs in tmp-<n>/out with the store's lock released,
    // its checked files move to run-<n>, and the sources are read from there.
    RunResult run_python(const Loader& l, const std::string& now, const RunOptions& run_options,
                         std::unique_lock<std::mutex>* lock, bool reuse_kept) {
        RunResult res;
        const std::string id = l.id;
        if (Error e = check_loader_for_reader(l, ReaderKind::python)) return failed(id, now, e);
        PythonReaderOptions po;
        if (Error e = parse_python_options(l.reader_options, &po)) return failed(id, now, e);
        auto ingest = [&](const fs::path& kept) {
            PythonManifestRead m = read_python_manifest(kept.u8string(), options.python_limits);
            if (m.error) return failed(id, now, m.error);
            return commit_run(l, now, [&](std::vector<LoadedSource>* made, std::vector<PendingFile>* pending,
                                          std::vector<std::string>* warnings) {
                return python_sources(l, kept, m.manifest, now, made, pending, warnings);
            });
        };
        if (reuse_kept) {
            const fs::path kept = latest_kept(id);
            if (kept.empty()) return res;
            RunResult r = ingest(kept);
            if (r.error.ok()) prune_python(id, kept, true);
            return r;
        }
        if (!run_options.allow_python) {
            res.error = Error("loader.python_needs_approval",
                              "Running the Python loader " + id +
                                  " runs its script with your privileges, so each run needs your approval.",
                              id);
            return res;
        }
        ExpandResult expanded = expand_input(l.path, roots());
        if (expanded.error) return failed(id, now, expanded.error);
        if (expanded.files.size() != 1) {
            return failed(id, now,
                          Error("loader.bad_value",
                                "A Python loader reads one file, and `input.path` matches " +
                                    std::to_string(expanded.files.size()) + ".",
                                "input.path"));
        }
        std::string script;
        if (Error e = resolve_script(po.script, true, &script)) return failed(id, now, e);
        PythonInterpreter python = resolve_python(po.python);
        if (python.error) return failed(id, now, python.error);

        // A fresh folder for this run: tmp-<n>/host and tmp-<n>/out.
        std::error_code ec;
        const fs::path root = python_root(id);
        fs::create_directories(root, ec);
        uint64_t n = 1;
        for (const RunFolder& f : python_folders(id)) n = std::max(n, f.n + 1);
        fs::path tmp;
        for (int tries = 0; tries < 64; ++tries, ++n) {
            const fs::path candidate = root / fs::u8path("tmp-" + std::to_string(n));
            if (fs::create_directory(candidate, ec) && !ec) {
                tmp = candidate;
                break;
            }
        }
        if (tmp.empty()) {
            return failed(id, now, Error("loader.io", "A folder for the Python loader's run could not be created.",
                                         root.u8string()));
        }
        const fs::path out_dir = tmp / "out";
        const fs::path host_dir = tmp / "host";
        auto cleanup = [&] { remove_unprotected(tmp); };
        fs::create_directory(out_dir, ec);
        if (!ec) fs::create_directory(host_dir, ec);
        std::string host = options.python.host_script;
        if (!ec && host.empty()) {
            host = (host_dir / "loader_host.py").u8string();
            if (Error e = write_text_atomic(host_dir / "loader_host.py", embedded_loader_host())) {
                cleanup();
                return failed(id, now, e);
            }
        }
        if (ec) {
            cleanup();
            return failed(id, now, Error("loader.io", "A folder for the Python loader's run could not be created.",
                                         tmp.u8string()));
        }
        PythonScriptRun args;
        args.python = python;
        args.host_script = host;
        args.script = script;
        args.input = expanded.files[0];
        args.output_dir = fs::canonical(out_dir, ec).u8string();
        args.timeout_s = po.timeout_s;
        if (ec) {
            cleanup();
            return failed(id, now, Error("loader.io", "The Python loader's output folder could not be resolved.",
                                         out_dir.u8string()));
        }

        if (lock) lock->unlock();
        PythonScriptResult ran = run_python_script(args, options.python, options.python_limits);
        if (lock) lock->lock();

        if (ran.error) {
            cleanup();
            return failed(id, now, ran.error);
        }
        if (!loader_row(id, nullptr)) {
            cleanup();
            res.error = Error("loader.not_found", "The loader " + id + " was removed while its script ran.");
            return res;
        }
        // The checked files move to run-<n>, with the manifest as the host wrote it.
        const fs::path kept = root / fs::u8path("run-" + std::to_string(n));
        if (!fs::create_directory(kept, ec) || ec) {
            cleanup();
            return failed(id, now, Error("loader.io", "The Python loader's outputs could not be kept.", kept.u8string()));
        }
        Error moved;
        for (const PythonOutput& o : ran.manifest.outputs) {
            if (touches_protected_file((kept / fs::u8path(o.file)).u8string())) {
                moved = protected_error(kept / fs::u8path(o.file));
                break;
            }
            fs::rename(out_dir / fs::u8path(o.file), kept / fs::u8path(o.file), ec);
            if (ec) {
                moved = Error("loader.io", "The Python loader's output " + o.file + " could not be kept.", o.file);
                break;
            }
        }
        if (!moved) moved = write_text_atomic(kept / "manifest.json", ran.manifest.to_json().dump(1));
        cleanup();
        if (moved) {
            remove_unprotected(kept);
            return failed(id, now, moved);
        }
        RunResult r = ingest(kept);
        if (r.error) {
            remove_unprotected(kept);
            return r;
        }
        prune_python(id, kept, false);
        return r;
    }

    // Runs a loader. `lock` is the store's lock, which a Python loader releases
    // while its script runs; `reuse_kept` is set by rebuild().
    RunResult run(const std::string& id, const RunOptions& run_options, std::unique_lock<std::mutex>* lock,
                  bool reuse_kept) {
        RunResult res;
        const std::string now = now_utc();
        if (!valid_id(id) || !loader_row(id, nullptr)) {
            res.error = Error("loader.not_found", "There is no saved loader " + id + ".");
            return res;
        }
        auto fail = [&](Error e) { return failed(id, now, std::move(e)); };
        std::string text;
        if (Error e = read_text(loader_file(id), kMaxLoaderText, &text)) return fail(e);
        LoaderParse parsed = parse_loader(text);
        if (parsed.error) return fail(parsed.error);
        const Loader& l = parsed.loader;
        if (l.id != id) return fail(Error("loader.bad_id", "The loader's id differs from its file name."));
        // A Python loader names its reader, and its input is handed to the
        // script, never to a native reader, so the pickle check does not apply.
        if (l.reader && *l.reader == ReaderKind::python) return run_python(l, now, run_options, lock, reuse_kept);

        ExpandResult expanded = expand_input(l.path, roots());
        if (expanded.error) return fail(expanded.error);
        for (const std::string& f : expanded.files) {
            if (Error e = check_not_pickle(f, options.limits)) return fail(e);
        }
        ReaderKind reader;
        if (l.reader) {
            reader = *l.reader;
        } else {
            ReaderInference first = infer_reader(expanded.files[0], options.limits);
            if (first.error) return fail(first.error);
            reader = first.reader;
            for (std::size_t i = 1; i < expanded.files.size(); ++i) {
                ReaderInference other = infer_reader(expanded.files[i], options.limits);
                if (other.error) return fail(other.error);
                if (other.reader != reader) {
                    return fail(Error("loader.reader_mismatch",
                                      "The files of the glob call for different readers; name the reader in the loader.",
                                      expanded.files[i]));
                }
            }
        }
        if (Error e = check_loader_for_reader(l, reader)) return fail(e);

        RunResult r = commit_run(l, now, [&](std::vector<LoadedSource>* made, std::vector<PendingFile>* pending,
                                             std::vector<std::string>* warnings) -> Error {
            if (!reader_is_tabular(reader)) return tensor_outputs(l, reader, expanded.files, now, made, pending, warnings);
            uint64_t total = 0;
            for (const std::string& f : expanded.files) {
                std::error_code ec;
                total += static_cast<uint64_t>(fs::file_size(fs::u8path(f), ec));
            }
            const TabularRead read{reader, expanded.files, l.reader_options, {}};
            for (std::size_t i = 0; i < l.outputs.size(); ++i) {
                const LoaderOutput& o = l.outputs[i];
                if (Error e = add_table(l, i, o, o.name, o.title.empty() ? o.name : o.title, read,
                                        static_cast<int64_t>(total), {}, now, made)) {
                    return e;
                }
            }
            return {};
        });
        // A loader that was a Python loader before no longer needs its kept outputs.
        if (r.error.ok()) remove_python_outputs(id);
        return r;
    }
};

LoaderStore::LoaderStore() : impl_(std::make_unique<Impl>()) {}
LoaderStore::~LoaderStore() = default;

StoreOpen LoaderStore::open(const std::string& workspace, const StoreOptions& options) {
    StoreOpen r;
    std::unique_ptr<LoaderStore> store(new LoaderStore());
    Impl& im = *store->impl_;
    im.options = options;
    std::error_code ec;
    fs::create_directories(fs::u8path(workspace), ec);
    const fs::path canonical = fs::canonical(fs::u8path(workspace), ec);
    if (ec || !fs::is_directory(canonical, ec)) {
        r.error = Error("loader.io", "The workspace folder could not be created or opened.", workspace);
        return r;
    }
    im.workspace = canonical.u8string();
    fs::create_directories(im.ws() / "loaders", ec);
    fs::create_directories(im.ws() / "tensors", ec);
    if ((r.error = im.read_inputs())) return r;
    const bool existed = fs::exists(im.ws() / "workspace.duckdb", ec);
    if ((r.error = im.open_db())) return r;
    const bool had_schema = im.schema_exists();
    if ((r.error = im.ensure_schema())) return r;
    r.error = (existed && had_schema) ? im.sync() : im.rebuild();
    if (r.error) return r;
    r.store = std::move(store);
    return r;
}

const std::string& LoaderStore::workspace() const { return impl_->workspace; }

Error LoaderStore::grant(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl& im = *impl_;
    std::error_code ec;
    const fs::path canonical = fs::canonical(fs::u8path(path), ec);
    if (ec) return Error("loader.io", "The path to grant does not exist.", path);
    const std::string c = canonical.u8string();
    if (path_within(c, im.workspace)) return {};
    if (fs::is_directory(canonical, ec)) {
        if (canonical == canonical.root_path()) {
            return Error("loader.path_not_allowed", "The root folder cannot be granted.", c);
        }
        if (std::find(im.granted_folders.begin(), im.granted_folders.end(), c) != im.granted_folders.end()) return {};
        im.granted_folders.push_back(c);
    } else if (fs::is_regular_file(canonical, ec)) {
        if (std::find(im.granted_files.begin(), im.granted_files.end(), c) != im.granted_files.end()) return {};
        im.granted_files.push_back(c);
    } else {
        return Error("loader.io", "Only a regular file or a folder can be granted.", c);
    }
    if (Error e = im.write_inputs()) return e;
    return im.open_db();
}

std::vector<std::string> LoaderStore::granted_files() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->granted_files;
}

std::vector<std::string> LoaderStore::granted_folders() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->granted_folders;
}

Error LoaderStore::save(const Loader& loader) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl& im = *impl_;
    LoaderParse checked = loader_from_json(loader.to_json());
    if (checked.error) return checked.error;
    ExpandResult expanded = expand_input(loader.path, im.roots());
    if (expanded.error && expanded.error.code != "loader.no_match") return expanded.error;
    if (checked.loader.reader && *checked.loader.reader == ReaderKind::python) {
        PythonReaderOptions po;
        if (Error e = parse_python_options(checked.loader.reader_options, &po)) return e;
        if (Error e = im.resolve_script(po.script, false, nullptr)) return e;
    }
    const std::string text = write_loader(checked.loader);
    if (Error e = write_text_atomic(im.loader_file(loader.id), text)) return e;
    return im.upsert_loader(loader.id, loader.title.empty() ? loader.id : loader.title, text, now_utc());
}

LoaderStore::LoaderGet LoaderStore::get(const std::string& id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    LoaderGet r;
    if (!valid_id(id) || !impl_->loader_row(id, &r.record)) {
        r.error = Error("loader.not_found", "There is no saved loader " + id + ".");
        return r;
    }
    LoaderParse parsed = parse_loader(r.record.recipe);
    r.error = parsed.error;
    r.loader = std::move(parsed.loader);
    return r;
}

std::vector<LoaderRecord> LoaderStore::list() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    json rows;
    std::vector<LoaderRecord> out;
    if (impl_->db->rows(std::string("SELECT ") + kLoaderColumns + " FROM caliper.loaders ORDER BY \"id\"", &rows)) {
        return out;
    }
    for (const json& row : rows) out.push_back(record_from_row(row));
    return out;
}

Error LoaderStore::remove(const std::string& id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl& im = *impl_;
    std::error_code ec;
    const bool has_file = valid_id(id) && fs::exists(im.loader_file(id), ec);
    if (!valid_id(id) || (!im.loader_row(id, nullptr) && !has_file)) {
        return Error("loader.not_found", "There is no saved loader " + id + ".");
    }
    if (Error e = im.db->begin()) return e;
    Error e = im.remove_sources_of(id);
    if (!e) e = im.db->execute("DELETE FROM caliper.loaders WHERE \"id\" = " + lit(id));
    if (e) {
        im.db->rollback();
        return e;
    }
    if ((e = im.db->commit())) return e;
    fs::remove(im.loader_file(id), ec);
    im.remove_python_outputs(id);
    return {};
}

RunResult LoaderStore::run(const std::string& id, const RunOptions& options) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    return impl_->run(id, options, &lock, false);
}

std::vector<LoaderStore::ImportItem> LoaderStore::import_files(const std::vector<std::string>& paths,
                                                               bool grant_outside) {
    std::vector<ImportItem> items;
    for (const std::string& path : paths) {
        ImportItem item;
        item.path = path;
        std::error_code ec;
        const fs::path canonical = fs::canonical(fs::u8path(path), ec);
        if (ec || !fs::is_regular_file(canonical, ec)) {
            item.error = Error("loader.io", "The dropped path is not a readable file.", path);
            items.push_back(std::move(item));
            continue;
        }
        const std::string c = canonical.u8string();
        if (!path_within(c, workspace())) {
            if (!grant_outside) {
                item.error = Error("loader.path_not_allowed", "The file lies outside the workspace and is not granted.", c);
                items.push_back(std::move(item));
                continue;
            }
            if ((item.error = grant(c))) {
                items.push_back(std::move(item));
                continue;
            }
        }
        std::vector<std::string> taken;
        for (const LoaderRecord& r : list()) taken.push_back(r.id);
        LoaderInference inferred = infer_loader(c, workspace(), taken, impl_->options.limits);
        if ((item.error = inferred.error)) {
            items.push_back(std::move(item));
            continue;
        }
        if ((item.error = save(inferred.loader))) {
            items.push_back(std::move(item));
            continue;
        }
        item.loader_id = inferred.loader.id;
        items.push_back(std::move(item));
    }
    return items;
}

std::vector<LoadedSource> LoaderStore::sources(const std::string& kind, const std::string& prefix, std::size_t limit) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string where;
    if (!kind.empty()) where = "\"kind\" = " + lit(kind);
    if (!prefix.empty()) where += (where.empty() ? "" : " AND ") + std::string("starts_with(\"name\", ") + lit(prefix) + ")";
    return impl_->select_sources(where, limit ? " LIMIT " + std::to_string(limit) : "");
}

LoaderStore::SourceGet LoaderStore::describe(const std::string& source_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    SourceGet r;
    std::vector<LoadedSource> found = impl_->select_sources("\"id\" = " + lit(source_id));
    if (found.empty()) {
        r.error = Error("catalog.not_found", "There is no source " + source_id + ".");
        return r;
    }
    r.source = std::move(found[0]);
    return r;
}

TensorResult LoaderStore::load_tensor(const std::string& source_id) {
    SourceGet got = describe(source_id);
    if (got.error) return {got.error, {}};
    const LoadedSource& s = got.source;
    if (s.kind != "tensor" || !s.dtype || !s.shape) {
        return {Error("catalog.not_a_tensor", "The source " + source_id + " is not a tensor."), {}};
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::error_code ec;
    const std::string canonical = fs::canonical(fs::u8path(s.data_file), ec).u8string();
    if (ec) return {Error("loader.io", "The tensor's file is missing; run its loader again.", s.data_file), {}};
    if (!path_allowed(canonical, impl_->roots())) {
        return {Error("loader.path_not_allowed", "The tensor's file lies outside the workspace and the grants.",
                      canonical),
                {}};
    }
    return TensorData::map_file(canonical, static_cast<uint64_t>(s.data_offset), *s.dtype, *s.shape);
}

LoaderStore::Sample LoaderStore::sample(const std::string& source_id, uint64_t limit) {
    Sample r;
    SourceGet got = describe(source_id);
    if ((r.error = got.error)) return r;
    const LoadedSource& s = got.source;
    if (s.kind == "table") {
        if (limit < 1) limit = 1;
        if (limit > 1000) limit = 1000;
        QueryResult q = query("SELECT * FROM " + sql_identifier(s.name), limit, 10000);
        if ((r.error = q.error)) return r;
        json cols = json::array();
        for (const ColumnInfo& c : q.columns) cols.push_back(c.to_json());
        r.value = {{"id", s.id}, {"columns", cols}, {"rows", q.rows}, {"truncated", q.truncated}};
        return r;
    }
    TensorResult t = load_tensor(source_id);
    if ((r.error = t.error)) return r;
    r.value = {{"id", s.id},
               {"dtype", dtype_name(t.tensor.dtype())},
               {"shape", t.tensor.shape()},
               {"stats", tensor_stats(t.tensor).to_json()},
               {"slice", tensor_sample(t.tensor)}};
    return r;
}

QueryResult LoaderStore::query(const std::string& sql, uint64_t row_limit, uint32_t timeout_ms) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->db->query(sql, row_limit, timeout_ms);
}

Error LoaderStore::rebuild() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->rebuild();
}

}  // namespace caliper_builder
