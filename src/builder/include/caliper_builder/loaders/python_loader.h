#pragma once
// The Python loader: a loader whose reader is "python" runs the user's script
// out of process, and Caliper reads back only files of formats its own readers
// already read safely. docs/plans/loaders.md ("The Python loader") is the
// reference; LoaderStore::run() is the caller.
//
// The recipe names the script and, optionally, the interpreter and a limit:
//   [input]  reader = "python"
//   [reader] script = "scripts/load_ckpt.py"   a file inside the workspace
//            python = "/abs/path/to/python3"   default: python3 on PATH; a bare
//                                              name is looked up on PATH, and a
//                                              relative path is refused
//            timeout_s = 120                   1 to 3600
//
// The script defines `def load(path, out)` and writes through `out`, an object
// of python/caliper/loader_host.py, which this library embeds and writes next to
// each run. run_python_script() runs
//   <python> -I loader_host.py --script <script> --input <file> --output <folder>
// through acp::run_process with a clean environment (PATH, HOME, TMPDIR, LANG,
// plus VIRTUAL_ENV and the environment's bin folder first on PATH when the
// interpreter lies in a virtual environment), the empty output folder as its
// working folder, the time limit (after which the process group is killed),
// and at most 1 MiB kept of each output stream.
//
// read_python_manifest() then treats the folder as hostile: manifest.json is at
// most 1 MiB and nested at most 64 deep; unknown keys are refused; every name
// matches the id grammar and is unique per kind; every file is a plain name
// ("[A-Za-z0-9_][A-Za-z0-9_.-]*", at most 128 characters, ending in .npy for a
// tensor and .parquet or .csv for a table) of a regular file in the folder that
// is not a symbolic link and has one hard link; its declared size equals its
// real size; and the counts and sizes stay within PythonOutputLimits.
//
// Running a script runs the user's code with the user's privileges, so
// LoaderStore::run() refuses a Python loader unless the caller passes
// RunOptions::allow_python, which the command layer sets after the user's
// approval (loader.python_needs_approval).
//
// Error codes: loader.python_not_available (no interpreter, no numpy, or a
// platform without process spawning), loader.python_failed,
// loader.python_timeout, loader.malformed, loader.path_not_allowed,
// loader.too_large, loader.bad_value, loader.unknown_key, loader.missing_key,
// loader.io. Process spawning is POSIX only for now; on Windows every run fails
// with loader.python_not_available. No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/tensor.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caliper_builder {

constexpr int kPythonDefaultTimeoutS = 120;
constexpr int kPythonMaxTimeoutS = 3600;

// The [reader] table of a Python loader.
struct PythonReaderOptions {
    std::string script;  // as written: relative to the workspace, or absolute inside it
    std::string python;  // empty: python3 on PATH
    int timeout_s = kPythonDefaultTimeoutS;
};

// Checks and reads the [reader] table: `script` (required), `python` and
// `timeout_s`; any other key is loader.unknown_key. `where` is "reader.<key>".
Error parse_python_options(const nlohmann::json& options, PythonReaderOptions* out);

// The interpreter a run uses.
struct PythonInterpreter {
    Error error;
    std::string path;  // absolute
    std::string venv;  // the virtual environment it lies in, or empty
};

// Resolves `reader.python`: empty means "python3"; a name without a separator
// is looked up on the absolute entries of this process's PATH; an absolute path
// must be an executable file; a relative path is loader.bad_value. A missing
// interpreter is loader.python_not_available. A virtual environment is
// recognised by the pyvenv.cfg file above the interpreter's bin folder.
PythonInterpreter resolve_python(const std::string& configured);

// The child's whole environment: PATH (the absolute entries of this process's
// PATH, after the virtual environment's bin folder), HOME, TMPDIR and LANG when
// this process has them, and VIRTUAL_ENV for an interpreter in one.
std::vector<std::pair<std::string, std::string>> python_environment(const PythonInterpreter& python);

// The text of python/caliper/loader_host.py, embedded at build time.
const std::string& embedded_loader_host();

// Bounds on what a run may hand back.
struct PythonOutputLimits {
    uint64_t max_manifest_bytes = 1ULL << 20;
    int max_json_depth = 64;
    uint64_t max_outputs = 10000;
    uint64_t max_file_bytes = 4ULL << 30;    // one output file
    uint64_t max_total_bytes = 16ULL << 30;  // all of them
    std::size_t max_warnings = 64;           // per output, and for the run
    std::size_t max_text_chars = 1024;       // one warning; a title has at most 256
    uint64_t max_error_bytes = 64ULL << 10;  // error.json
    uint64_t max_csv_header_bytes = 1ULL << 20;
    std::size_t max_csv_columns = 10000;
};

// One output in manifest.json.
struct PythonOutput {
    std::string name;  // the id the script gave it
    std::string kind;  // "tensor" or "table"
    std::string file;  // a plain file name in the folder
    uint64_t bytes = 0;
    std::string title;                         // optional; a torch key path, for example
    std::vector<std::string> warnings;
    std::optional<DType> dtype;                // tensors: checked against the file
    std::optional<std::vector<int64_t>> shape; // tensors: checked against the file
};

// manifest.json:
//   {"caliper_python_loader": 0,
//    "outputs": [{"name": "w", "kind": "tensor", "file": "w.npy", "bytes": 152,
//                 "dtype": "f32", "shape": [3, 2], "title": "layer.w",
//                 "warnings": ["loaded weights-only from a pickle file"]},
//                {"name": "m", "kind": "table", "file": "m.parquet", "bytes": 701}],
//    "warnings": ["a warning of the run, carried by every source"]}
struct PythonManifest {
    std::vector<PythonOutput> outputs;
    std::vector<std::string> warnings;

    nlohmann::json to_json() const;
};

struct PythonManifestRead {
    Error error;
    PythonManifest manifest;
};

// Reads and checks <folder>/manifest.json and the files it lists, as above.
PythonManifestRead read_python_manifest(const std::string& folder, const PythonOutputLimits& limits = {});

// The columns of a CSV table written by loader_host.py, from its typed header:
// every cell is "<name>:<type>" with the type BIGINT, DOUBLE, BOOLEAN or
// VARCHAR, quoted with '"'. Fails with loader.malformed for any other header.
struct TypedCsvHeader {
    Error error;
    std::vector<ColumnInfo> columns;
};
TypedCsvHeader read_typed_csv_header(const std::string& path, const PythonOutputLimits& limits = {});

// How a run's process is started.
struct PythonProcessOptions {
    std::string host_script;                  // empty: write the embedded copy next to the run
    std::size_t max_output_bytes = 1u << 20;  // kept of each output stream
    int kill_grace_ms = 2000;
};

// One run of a script.
struct PythonScriptRun {
    PythonInterpreter python;
    std::string host_script;  // the loader_host.py to run
    std::string script;       // canonical
    std::string input;        // canonical
    std::string output_dir;   // an empty folder
    int timeout_s = kPythonDefaultTimeoutS;
};

struct PythonScriptResult {
    Error error;
    PythonManifest manifest;
    std::string log;  // the end of the script's standard error, at most 4 KiB, for a person
};

// Runs the host on the script and reads the manifest of a run that succeeded.
// A failed script is loader.python_failed with the exception from error.json
// (or the end of standard error); a run past its limit is loader.python_timeout;
// exit code 3 (numpy missing) and an interpreter that cannot start are
// loader.python_not_available.
PythonScriptResult run_python_script(const PythonScriptRun& run, const PythonProcessOptions& process = {},
                                     const PythonOutputLimits& limits = {});

}  // namespace caliper_builder
