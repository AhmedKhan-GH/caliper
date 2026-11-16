#pragma once
// LoaderStore: the loaders of one workspace, their runs and their outputs.
//
// The workspace folder holds the source of truth as text:
//   loaders/<id>.loader.toml   one saved loader each (loader.h)
//   inputs.toml                the files and folders the user granted
//   tensors/<name>.npy         derived tensors
// and workspace.duckdb holds the views and tables that loaders made plus two
// tables, caliper.loaders (one row per loader: the recipe text, timestamps and
// the last run's status) and caliper.sources (one row per output, the fields
// of LoadedSource). rebuild() recreates every row from the text files by
// running each loader again; open() rebuilds a database that is missing or has
// no caliper schema, and otherwise brings the rows in line with the files.
//
// Safety (docs/plans/loaders.md, "Safety"): every input path must lie in the
// workspace or in a grant; the database is opened with external access off,
// the allowed paths set to the workspace and the grants, autoload and
// autoinstall off and the configuration locked; every SQL text from a loader
// or a query must be exactly one SELECT. Only a user action should call
// grant(); a grant reopens the database with the new allowed paths.
//
// One LoaderStore per workspace per process. Every method locks one mutex, so
// the store can be shared between threads; a long query blocks the others
// until its time limit. No method throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/loaded_source.h>
#include <caliper_builder/loaders/loader.h>
#include <caliper_builder/loaders/query.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/loaders/tensor_readers.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_builder {

struct StoreOptions {
    int threads = 2;
    std::string memory_limit = "1GB";
    ReaderLimits limits;
    uint64_t max_sources_per_run = 10000;
};

// A row of caliper.loaders. Times are UTC ISO 8601 text; last_run_at and the
// error fields are empty when they are NULL.
struct LoaderRecord {
    std::string id;
    std::string title;
    std::string recipe;
    std::string created_at;
    std::string updated_at;
    std::string last_run_at;
    std::string last_status;  // "never", "ok" or "error"
    std::string last_error_code;
    std::string last_error_message;

    nlohmann::json to_json() const;
};

struct RunResult {
    Error error;
    std::vector<LoadedSource> sources;
    std::vector<std::string> warnings;  // about the files, such as skipped members
};

class LoaderStore;

struct StoreOpen {
    Error error;
    std::unique_ptr<LoaderStore> store;
};

class LoaderStore {
public:
    // Opens (and creates, when missing) the workspace folder and its database.
    static StoreOpen open(const std::string& workspace, const StoreOptions& options = {});
    ~LoaderStore();

    const std::string& workspace() const;

    // Grants a file or folder the user dropped or named, records it in
    // inputs.toml and reopens the database. The root folder cannot be granted.
    Error grant(const std::string& path);
    std::vector<std::string> granted_files() const;
    std::vector<std::string> granted_folders() const;

    // Validates the loader and its paths, writes loaders/<id>.loader.toml and
    // its caliper.loaders row. Its sources stay as the last run left them.
    Error save(const Loader& loader);
    struct LoaderGet {
        Error error;
        Loader loader;
        LoaderRecord record;
    };
    LoaderGet get(const std::string& id);
    std::vector<LoaderRecord> list();
    // Deletes the loader's file, its row, its sources, their views and tables
    // and their derived tensors.
    Error remove(const std::string& id);
    // Runs a saved loader: replaces its sources atomically, or leaves the
    // previous ones whole when it fails. The status is recorded either way.
    RunResult run(const std::string& id);

    // For files the user dropped: grants each one outside the workspace (when
    // `grant_outside`), infers its reader and saves a default loader for it.
    // Each item succeeds or fails on its own. The loaders are not run.
    struct ImportItem {
        std::string path;
        Error error;
        std::string loader_id;
    };
    std::vector<ImportItem> import_files(const std::vector<std::string>& paths, bool grant_outside = true);

    // The catalog side: sources filtered by kind ("table", "tensor" or empty)
    // and a name prefix, at most `limit` (0 for all), ordered by id.
    std::vector<LoadedSource> sources(const std::string& kind = "", const std::string& prefix = "",
                                      std::size_t limit = 0);
    struct SourceGet {
        Error error;
        LoadedSource source;
    };
    SourceGet describe(const std::string& source_id);
    // A table: {"id", "columns", "rows", "truncated"} with at most `limit` rows
    // (1 to 1000). A tensor: {"id", "dtype", "shape", "stats", "slice"}.
    struct Sample {
        Error error;
        nlohmann::json value;
    };
    Sample sample(const std::string& source_id, uint64_t limit = 20);
    // A tensor source's data, mapped from its file.
    TensorResult load_tensor(const std::string& source_id);

    // Read-only SQL over the workspace database: exactly one SELECT, rolled
    // back, at most row_limit rows (clamped to 1..100,000), interrupted after
    // timeout_ms (clamped to 1..60,000).
    QueryResult query(const std::string& sql, uint64_t row_limit = 1000, uint32_t timeout_ms = 5000);

    // Recreates the database rows from the workspace files: drops the
    // relations loaders made, empties both tables, reads every loader file and
    // runs each loader. A loader that fails keeps its row with the error.
    Error rebuild();

    struct Impl;

private:
    LoaderStore();
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_builder
