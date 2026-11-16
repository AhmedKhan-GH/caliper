// The Python loader: its options, the interpreter and its environment, the run
// of loader_host.py through acp::run_process, and the checks on what it writes.
#include <caliper_builder/loaders/python_loader.h>

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/loaders/loader.h>

#include "file_io.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace caliper_builder {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// python/caliper/loader_host.py, as bytes; the build generates the include.
const unsigned char kLoaderHost[] = {
#include "loader_host_py.inc"
};

#ifdef _WIN32
constexpr char kPathListSeparator = ';';
#else
constexpr char kPathListSeparator = ':';
#endif

const char* kCsvTypes[] = {"BIGINT", "DOUBLE", "BOOLEAN", "VARCHAR"};

bool has_control(const std::string& s) {
    for (unsigned char c : s) {
        if (c < 0x20 || c == 0x7F) return true;
    }
    return false;
}

std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// The absolute entries of a PATH value: an empty or relative entry would make
// the lookup depend on the working folder.
std::string absolute_entries(const std::string& list) {
    std::string out;
    std::size_t start = 0;
    while (start <= list.size()) {
        const std::size_t end = list.find(kPathListSeparator, start);
        const std::string entry = list.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!entry.empty() && fs::u8path(entry).is_absolute()) {
            if (!out.empty()) out += kPathListSeparator;
            out += entry;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

// Text from the child for a message: control characters become spaces, runs of
// spaces collapse, and the text is cut to `limit` bytes.
std::string clean_text(const std::string& s, std::size_t limit) {
    std::string out;
    for (unsigned char c : s) {
        const char ch = (c < 0x20 || c == 0x7F) ? ' ' : static_cast<char>(c);
        if (ch == ' ' && (out.empty() || out.back() == ' ')) continue;
        out += ch;
        if (out.size() >= limit) break;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// The last line of text that is not blank.
std::string last_line(const std::string& text) {
    std::size_t end = text.size();
    while (end > 0) {
        const std::size_t start = text.rfind('\n', end - 1);
        const std::size_t from = start == std::string::npos ? 0 : start + 1;
        const std::string line = clean_text(text.substr(from, end - from), 300);
        if (!line.empty()) return line;
        if (start == std::string::npos) break;
        end = start;
    }
    return {};
}

// Reads a small regular file that is not a symbolic link, within `max` bytes.
Error read_small_file(const fs::path& p, uint64_t max, std::string* out, const std::string& where) {
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(p, ec);
    if (ec || !fs::exists(st)) return Error("loader.malformed", "The Python loader wrote no " + where + ".", where);
    if (fs::is_symlink(st) || !fs::is_regular_file(st)) {
        return Error("loader.path_not_allowed", where + " is not a regular file.", where);
    }
    const uintmax_t size = fs::file_size(p, ec);
    if (ec) return Error("loader.io", "The size of " + where + " could not be read.", where);
    if (size > max) return Error("loader.too_large", where + " is larger than its limit.", where);
    std::ifstream in(p, std::ios::binary);
    if (!in) return Error("loader.io", where + " could not be opened.", where);
    // At most the size checked above, even if the file grows meanwhile.
    out->assign(static_cast<std::size_t>(size), '\0');
    in.read(&(*out)[0], static_cast<std::streamsize>(size));
    out->resize(static_cast<std::size_t>(in.gcount()));
    return {};
}

Error parse_small_json(const std::string& text, int max_depth, json* out, const std::string& where) {
    if (detail::json_depth(reinterpret_cast<const uint8_t*>(text.data()), text.size(), max_depth) > max_depth) {
        return Error("loader.malformed", where + " is nested too deeply.", where);
    }
    *out = json::parse(text, nullptr, false);
    if (out->is_discarded()) return Error("loader.malformed", where + " is not valid JSON.", where);
    return {};
}

bool plain_file_name(const std::string& name) {
    if (name.empty() || name.size() > 128) return false;
    for (std::size_t i = 0; i < name.size(); ++i) {
        const char c = name[i];
        const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!(word || (i > 0 && (c == '.' || c == '-')))) return false;
    }
    return true;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct ManifestContext {
    const PythonOutputLimits& limits;

    Error malformed(const std::string& what, const std::string& where) const {
        return Error("loader.malformed", "The Python loader's manifest.json is malformed: " + what + ".", where);
    }

    Error text_list(const json& v, const std::string& where, std::vector<std::string>* out) const {
        if (!v.is_array()) return malformed("`" + where + "` is a list of strings", where);
        if (v.size() > limits.max_warnings) {
            return Error("loader.too_large", "manifest.json has more than " + std::to_string(limits.max_warnings) +
                                                 " warnings in `" + where + "`.",
                         where);
        }
        for (const json& w : v) {
            if (!w.is_string()) return malformed("`" + where + "` is a list of strings", where);
            const std::string s = w.get<std::string>();
            if (s.size() > limits.max_text_chars * 4 || has_control(s)) {
                return malformed("a warning in `" + where + "` is too long or holds a control character", where);
            }
            out->push_back(s);
        }
        return {};
    }

    Error output(const json& o, const std::string& where, PythonOutput* out) const {
        if (!o.is_object()) return malformed("`" + where + "` is an object", where);
        for (auto it = o.begin(); it != o.end(); ++it) {
            static const char* keys[] = {"name", "kind", "file", "bytes", "title", "warnings", "dtype", "shape"};
            if (std::find_if(std::begin(keys), std::end(keys), [&](const char* k) { return it.key() == k; }) ==
                std::end(keys)) {
                return malformed("the key `" + where + "." + it.key() + "` is unknown", where + "." + it.key());
            }
        }
        auto str = [&](const char* key, std::string* dst) -> Error {
            auto it = o.find(key);
            if (it == o.end() || !it->is_string()) {
                return malformed("`" + where + "." + key + "` is a required string", where + "." + key);
            }
            *dst = it->get<std::string>();
            return {};
        };
        if (Error e = str("name", &out->name)) return e;
        if (!valid_id(out->name)) {
            return malformed("the name `" + clean_text(out->name, 80) + "` does not match [a-z][a-z0-9_-]{0,63}",
                             where + ".name");
        }
        if (Error e = str("kind", &out->kind)) return e;
        if (out->kind != "tensor" && out->kind != "table") {
            return malformed("`" + where + ".kind` is \"tensor\" or \"table\"", where + ".kind");
        }
        if (Error e = str("file", &out->file)) return e;
        if (!plain_file_name(out->file)) {
            return Error("loader.path_not_allowed",
                         "The Python loader's manifest names the file `" + clean_text(out->file, 200) +
                             "`, which is not a plain file name inside its output folder.",
                         where + ".file");
        }
        const bool tensor = out->kind == "tensor";
        if (tensor ? !ends_with(out->file, ".npy") : !(ends_with(out->file, ".parquet") || ends_with(out->file, ".csv"))) {
            return malformed(std::string("a ") + out->kind + " is " + (tensor ? "an .npy file" : "a .parquet or .csv file"),
                             where + ".file");
        }
        auto bytes = o.find("bytes");
        if (bytes == o.end() || !(bytes->is_number_unsigned() || bytes->is_number_integer()) ||
            (bytes->is_number_integer() && bytes->get<int64_t>() < 0)) {
            return malformed("`" + where + ".bytes` is a required size in bytes", where + ".bytes");
        }
        out->bytes = bytes->get<uint64_t>();
        if (out->bytes > limits.max_file_bytes) {
            return Error("loader.too_large",
                         "The Python loader's output " + out->file + " is larger than the limit of " +
                             std::to_string(limits.max_file_bytes) + " bytes.",
                         where + ".bytes");
        }
        if (auto t = o.find("title"); t != o.end()) {
            if (!t->is_string() || t->get<std::string>().size() > 1024 || has_control(t->get<std::string>())) {
                return malformed("`" + where + ".title` is a short line of text", where + ".title");
            }
            out->title = t->get<std::string>();
        }
        if (auto w = o.find("warnings"); w != o.end()) {
            if (Error e = text_list(*w, where + ".warnings", &out->warnings)) return e;
        }
        auto dtype = o.find("dtype");
        auto shape = o.find("shape");
        if (!tensor) {
            if (dtype != o.end() || shape != o.end()) return malformed("a table has no dtype or shape", where);
            return {};
        }
        DType d;
        if (dtype == o.end() || !dtype->is_string() || !dtype_from_name(dtype->get<std::string>(), &d)) {
            return malformed("`" + where + ".dtype` is a catalog dtype", where + ".dtype");
        }
        out->dtype = d;
        if (shape == o.end() || !shape->is_array() || shape->size() > static_cast<std::size_t>(kMaxTensorRank)) {
            return malformed("`" + where + ".shape` is a list of at most 8 sizes", where + ".shape");
        }
        std::vector<int64_t> dims;
        for (const json& v : *shape) {
            if (!v.is_number_integer() || v.get<int64_t>() < 0) {
                return malformed("`" + where + ".shape` holds sizes of 0 or more", where + ".shape");
            }
            dims.push_back(v.get<int64_t>());
        }
        uint64_t numel = 0, nbytes = 0;
        if (Error e = check_shape(dims, d, &numel, &nbytes)) {
            return Error(e.code == "tensor.too_large" ? "loader.too_large" : "loader.malformed",
                         "The Python loader's manifest declares a shape that is too large or invalid.",
                         where + ".shape");
        }
        if (nbytes > out->bytes) {
            return malformed("the shape of " + out->file + " needs more bytes than the file holds", where + ".shape");
        }
        out->shape = std::move(dims);
        return {};
    }
};

Error error_from_record(const fs::path& folder, const PythonOutputLimits& limits, std::string* message,
                        std::string* type) {
    std::string text;
    if (Error e = read_small_file(folder / "error.json", limits.max_error_bytes, &text, "error.json")) return e;
    json j;
    if (Error e = parse_small_json(text, limits.max_json_depth, &j, "error.json")) return e;
    if (!j.is_object() || !j.contains("error") || !j["error"].is_object()) {
        return Error("loader.malformed", "error.json has no error.", "error.json");
    }
    const json& err = j["error"];
    if (err.contains("message") && err["message"].is_string()) {
        *message = clean_text(err["message"].get<std::string>(), 1000);
    }
    if (err.contains("type") && err["type"].is_string()) *type = clean_text(err["type"].get<std::string>(), 100);
    return {};
}

// An interpreter is an absolute path or a command name; a relative path would
// depend on the working folder.
Error check_python_name(const std::string& name) {
    const bool separator = name.find('/') != std::string::npos
#ifdef _WIN32
                           || name.find('\\') != std::string::npos
#endif
        ;
    if (name.empty() || has_control(name) || (separator && !fs::u8path(name).is_absolute())) {
        return Error("loader.bad_value",
                     "`reader.python` is an absolute path or a command name looked up on PATH; a relative path is "
                     "refused.",
                     "reader.python");
    }
    return {};
}

}  // namespace

Error parse_python_options(const json& options, PythonReaderOptions* out) {
    PythonReaderOptions r;
    if (!options.is_object()) {
        return Error("loader.missing_key", "A Python loader needs `reader.script`, the script that loads the file.",
                     "reader");
    }
    for (auto it = options.begin(); it != options.end(); ++it) {
        const std::string& key = it.key();
        const json& v = it.value();
        const std::string where = "reader." + key;
        if (key == "script" || key == "python") {
            if (!v.is_string() || v.get<std::string>().empty() || v.get<std::string>().size() > 4096 ||
                has_control(v.get<std::string>())) {
                return Error("loader.bad_value", "`" + where + "` is a path of at most 4096 characters.", where);
            }
            (key == "script" ? r.script : r.python) = v.get<std::string>();
        } else if (key == "timeout_s") {
            if (!v.is_number_integer() || v.get<int64_t>() < 1 || v.get<int64_t>() > kPythonMaxTimeoutS) {
                return Error("loader.bad_value", "`reader.timeout_s` is a whole number of seconds from 1 to 3600.",
                             where);
            }
            r.timeout_s = static_cast<int>(v.get<int64_t>());
        } else {
            return Error("loader.unknown_key", "The python reader has no option `" + key + "`.", where);
        }
    }
    if (r.script.empty()) {
        return Error("loader.missing_key", "A Python loader needs `reader.script`, the script that loads the file.",
                     "reader");
    }
    if (!r.python.empty()) {
        if (Error e = check_python_name(r.python)) return e;
    }
    if (out) *out = std::move(r);
    return {};
}

PythonInterpreter resolve_python(const std::string& configured) {
    PythonInterpreter r;
    const std::string name = configured.empty() ? "python3" : configured;
    if ((r.error = check_python_name(name))) return r;
    if (fs::u8path(name).is_absolute()) {
        r.path = acp::find_executable(fs::u8path(name).lexically_normal().u8string(), "");
        if (r.path.empty()) {
            r.error = Error("loader.python_not_available",
                            "There is no Python interpreter that can run at " + name + ".", "reader.python");
            return r;
        }
    } else {
        r.path = acp::find_executable(name, absolute_entries(env_or_empty("PATH")));
        if (r.path.empty()) {
            r.error = Error("loader.python_not_available",
                            name + " was not found on PATH; install Python 3 with numpy, or name the interpreter in "
                                   "`reader.python`.",
                            "reader.python");
            return r;
        }
    }
    std::error_code ec;
    const fs::path bin = fs::u8path(r.path).parent_path();
    if (bin.filename() == "bin" && fs::is_regular_file(bin.parent_path() / "pyvenv.cfg", ec)) {
        r.venv = bin.parent_path().u8string();
    }
    return r;
}

std::vector<std::pair<std::string, std::string>> python_environment(const PythonInterpreter& python) {
    std::vector<std::pair<std::string, std::string>> env;
    std::string path = absolute_entries(env_or_empty("PATH"));
    if (path.empty()) path = "/usr/bin:/bin";
    if (!python.venv.empty()) {
        path = (fs::u8path(python.venv) / "bin").u8string() + kPathListSeparator + path;
        env.emplace_back("VIRTUAL_ENV", python.venv);
    }
    env.emplace_back("PATH", path);
    for (const char* name : {"HOME", "TMPDIR", "LANG"}) {
        const char* v = std::getenv(name);
        if (v && *v) env.emplace_back(name, v);
    }
    return env;
}

const std::string& embedded_loader_host() {
    static const std::string text(reinterpret_cast<const char*>(kLoaderHost), sizeof kLoaderHost);
    return text;
}

json PythonManifest::to_json() const {
    json outs = json::array();
    for (const PythonOutput& o : outputs) {
        json j = {{"name", o.name}, {"kind", o.kind}, {"file", o.file}, {"bytes", o.bytes}};
        if (!o.title.empty()) j["title"] = o.title;
        if (!o.warnings.empty()) j["warnings"] = o.warnings;
        if (o.dtype) j["dtype"] = dtype_name(*o.dtype);
        if (o.shape) j["shape"] = *o.shape;
        outs.push_back(std::move(j));
    }
    return {{"caliper_python_loader", 0}, {"outputs", outs}, {"warnings", warnings}};
}

PythonManifestRead read_python_manifest(const std::string& folder, const PythonOutputLimits& limits) {
    PythonManifestRead r;
    const fs::path dir = fs::u8path(folder);
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(dir, ec);
    if (ec || fs::is_symlink(st) || !fs::is_directory(st)) {
        r.error = Error("loader.path_not_allowed", "The Python loader's output folder is not a plain folder.", folder);
        return r;
    }
    std::string text;
    if ((r.error = read_small_file(dir / "manifest.json", limits.max_manifest_bytes, &text, "manifest.json"))) return r;
    json j;
    if ((r.error = parse_small_json(text, limits.max_json_depth, &j, "manifest.json"))) return r;
    const ManifestContext ctx{limits};
    if (!j.is_object()) {
        r.error = ctx.malformed("it is not an object", "manifest.json");
        return r;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() != "caliper_python_loader" && it.key() != "outputs" && it.key() != "warnings") {
            r.error = ctx.malformed("the key `" + it.key() + "` is unknown", it.key());
            return r;
        }
    }
    auto version = j.find("caliper_python_loader");
    if (version == j.end() || !version->is_number_integer() || version->get<int64_t>() != 0) {
        r.error = Error("loader.unsupported", "This Caliper reads Python loader manifests of version 0 only.",
                        "caliper_python_loader");
        return r;
    }
    auto outputs = j.find("outputs");
    if (outputs == j.end() || !outputs->is_array()) {
        r.error = ctx.malformed("`outputs` is a required list", "outputs");
        return r;
    }
    if (outputs->size() > limits.max_outputs) {
        r.error = Error("loader.too_large",
                        "The Python loader wrote more than " + std::to_string(limits.max_outputs) + " outputs.",
                        "outputs");
        return r;
    }
    if (auto w = j.find("warnings"); w != j.end()) {
        if ((r.error = ctx.text_list(*w, "warnings", &r.manifest.warnings))) return r;
    }
    std::set<std::string> names, files;
    uint64_t total = 0;
    for (std::size_t i = 0; i < outputs->size(); ++i) {
        const std::string where = "outputs[" + std::to_string(i) + "]";
        PythonOutput o;
        if ((r.error = ctx.output((*outputs)[i], where, &o))) return r;
        if (!names.insert(o.kind + ":" + o.name).second) {
            r.error = ctx.malformed("two " + o.kind + "s are named " + o.name, where + ".name");
            return r;
        }
        if (!files.insert(o.file).second) {
            r.error = ctx.malformed("two outputs share the file " + o.file, where + ".file");
            return r;
        }
        if (o.bytes > limits.max_total_bytes - total) {
            r.error = Error("loader.too_large", "The Python loader's outputs are larger than the limit of " +
                                                    std::to_string(limits.max_total_bytes) + " bytes in all.",
                            where + ".bytes");
            return r;
        }
        total += o.bytes;
        const fs::path p = dir / fs::u8path(o.file);
        const fs::file_status fst = fs::symlink_status(p, ec);
        if (ec || !fs::exists(fst)) {
            r.error = ctx.malformed("the file " + o.file + " is missing", where + ".file");
            return r;
        }
        if (fs::is_symlink(fst) || !fs::is_regular_file(fst)) {
            r.error = Error("loader.path_not_allowed",
                            "The Python loader's output " + o.file + " is a symbolic link or not a regular file.",
                            where + ".file");
            return r;
        }
        if (fs::hard_link_count(p, ec) != 1 || ec) {
            r.error = Error("loader.path_not_allowed",
                            "The Python loader's output " + o.file + " has more than one hard link.", where + ".file");
            return r;
        }
        const uintmax_t size = fs::file_size(p, ec);
        if (ec || size != o.bytes) {
            r.error = ctx.malformed("the file " + o.file + " does not have the size the manifest declares",
                                    where + ".bytes");
            return r;
        }
        r.manifest.outputs.push_back(std::move(o));
    }
    return r;
}

TypedCsvHeader read_typed_csv_header(const std::string& path, const PythonOutputLimits& limits) {
    TypedCsvHeader r;
    detail::InputFile f;
    if ((r.error = f.open(path))) return r;
    std::vector<uint8_t> head;
    const uint64_t n = std::min<uint64_t>(f.size(), limits.max_csv_header_bytes);
    if ((r.error = f.read_vec(0, n, &head, limits.max_csv_header_bytes))) return r;
    auto malformed = [&](const std::string& what) {
        return Error("loader.malformed", "The typed CSV header is malformed: " + what + ".", path);
    };
    std::vector<std::string> cells;
    std::size_t i = 0;
    bool ended = false;
    while (i < head.size() && !ended) {
        if (head[i] != '"') {
            r.error = malformed("every header cell is quoted");
            return r;
        }
        ++i;
        std::string cell;
        bool closed = false;
        while (i < head.size()) {
            if (head[i] == '"') {
                if (i + 1 < head.size() && head[i + 1] == '"') {
                    cell += '"';
                    i += 2;
                    continue;
                }
                ++i;
                closed = true;
                break;
            }
            cell += static_cast<char>(head[i++]);
        }
        if (!closed) break;
        cells.push_back(std::move(cell));
        if (cells.size() > limits.max_csv_columns) {
            r.error = Error("loader.too_large", "The table has more columns than the limit.", path);
            return r;
        }
        if (i < head.size() && head[i] == ',') {
            ++i;
        } else if (i < head.size() && head[i] == '\n') {
            ended = true;
        } else if (i + 1 < head.size() && head[i] == '\r' && head[i + 1] == '\n') {
            ended = true;
        } else {
            break;
        }
    }
    if (!ended) {
        r.error = malformed("it does not end with a line break within " + std::to_string(limits.max_csv_header_bytes) +
                            " bytes, or a cell is not followed by a comma");
        return r;
    }
    std::set<std::string> seen;
    for (const std::string& cell : cells) {
        const std::size_t colon = cell.rfind(':');
        if (colon == std::string::npos || colon == 0) {
            r.error = malformed("a cell is not \"<name>:<type>\"");
            return r;
        }
        ColumnInfo c{cell.substr(0, colon), cell.substr(colon + 1)};
        if (std::find(std::begin(kCsvTypes), std::end(kCsvTypes), c.type) == std::end(kCsvTypes)) {
            r.error = malformed("the type " + clean_text(c.type, 40) + " is not BIGINT, DOUBLE, BOOLEAN or VARCHAR");
            return r;
        }
        if (has_control(c.name) || c.name.size() > 1024) {
            r.error = malformed("a column name is too long or holds a control character");
            return r;
        }
        if (!seen.insert(c.name).second) {
            r.error = malformed("two columns are named " + clean_text(c.name, 80));
            return r;
        }
        r.columns.push_back(std::move(c));
    }
    return r;
}

PythonScriptResult run_python_script(const PythonScriptRun& run, const PythonProcessOptions& process,
                                     const PythonOutputLimits& limits) {
    PythonScriptResult r;
    if (run.python.error) {
        r.error = run.python.error;
        return r;
    }
    acp::SpawnOptions so;
    so.argv = {run.python.path, "-I",          run.host_script, "--script", run.script,
               "--input",       run.input,     "--output",      run.output_dir};
    so.inherit_environment = false;
    for (const auto& [name, value] : python_environment(run.python)) so.env.push_back({name, value});
    so.cwd = run.output_dir;
    so.new_process_group = true;
    acp::RunLimits rl;
    const int timeout_s = std::min(std::max(run.timeout_s, 1), kPythonMaxTimeoutS);
    rl.timeout_ms = timeout_s * 1000;
    rl.max_output_bytes = process.max_output_bytes;
    rl.kill_grace_ms = process.kill_grace_ms;
    const acp::RunResult pr = acp::run_process(so, rl);
    r.log = pr.err.size() > 4096 ? pr.err.substr(pr.err.size() - 4096) : pr.err;
    const std::string where = fs::u8path(run.script).filename().u8string();

    if (pr.error) {
        if (pr.error.code == "process.not_found" || pr.error.code == "process.unsupported") {
            r.error = Error("loader.python_not_available",
                            pr.error.code == "process.unsupported"
                                ? "This platform cannot run a Python loader yet."
                                : "The Python interpreter " + run.python.path + " could not be started.",
                            "reader.python");
        } else {
            r.error = Error("loader.python_failed", "The Python loader could not be started: " + pr.error.message,
                            where);
        }
        return r;
    }
    if (pr.timed_out) {
        r.error = Error("loader.python_timeout",
                        "The Python loader ran past its time limit of " + std::to_string(timeout_s) +
                            " seconds and was stopped with its child processes.",
                        where);
        return r;
    }
    const fs::path out_dir = fs::u8path(run.output_dir);
    if (!pr.status.success()) {
        std::string message, type;
        error_from_record(out_dir, limits, &message, &type);
        if (pr.status.exited && pr.status.code == 3 && type == "numpy_missing") {
            r.error = Error("loader.python_not_available",
                            message.empty() ? "numpy is not installed for " + run.python.path + "." : message + ".",
                            "reader.python");
            return r;
        }
        if (message.empty()) message = last_line(pr.err);
        if (message.empty()) message = "its process ended with " + pr.status.describe();
        r.error = Error("loader.python_failed", "The Python loader's script failed: " + message, where);
        return r;
    }
    PythonManifestRead m = read_python_manifest(run.output_dir, limits);
    if ((r.error = m.error)) return r;
    r.manifest = std::move(m.manifest);
    return r;
}

}  // namespace caliper_builder
