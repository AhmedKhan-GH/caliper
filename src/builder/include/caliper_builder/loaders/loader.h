#pragma once
// The loader format v0: a declarative recipe that turns files into catalog
// sources. docs/plans/loaders.md ("The loader format v0") is the reference.
//
//   caliper_loader = 0
//   [loader]   id, title, description
//   [input]    path (a file or a glob), reader (optional; inferred)
//   [reader]   the reader's options
//   [[output]] name, title, mode, sql (tabular readers) or tensor and
//              [[output.transform]] records (tensor readers)
//
// Parsing is strict: an unknown key, a wrong type or a missing key is an error
// whose `where` is the line and column ("12:7") when the text came from TOML,
// or the key path ("output[0].mode") when it came from JSON; the message names
// the key path either way. The canonical writer emits the keys in a fixed
// order and keeps no comments. The JSON form is the TOML tables as JSON
// objects. No function here throws.
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/readers.h>
#include <caliper_builder/loaders/tensor_ops.h>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

constexpr std::size_t kMaxLoaderText = 1u << 20;
constexpr std::size_t kMaxLoaderSql = 64u << 10;
constexpr std::size_t kMaxLoaderOutputs = 256;
constexpr std::size_t kMaxLoaderTransforms = 64;

struct LoaderOutput {
    std::string name;
    std::string title;                 // empty: the name
    std::optional<std::string> mode;   // "view" (the default) or "table"; tabular readers only
    std::optional<std::string> sql;    // one SELECT over `input`; tabular readers only
    std::optional<std::string> tensor; // a tensor name or "*"; tensor readers only
    std::vector<TensorTransform> transforms;
};

struct Loader {
    std::string id;
    std::string title;        // empty: the id
    std::string description;
    std::string path;         // input.path
    std::optional<ReaderKind> reader;
    nlohmann::json reader_options = nlohmann::json::object();
    std::vector<LoaderOutput> outputs;

    nlohmann::json to_json() const;
};

struct LoaderParse {
    Error error;
    Loader loader;
};

// Whether a string is a valid id: [a-z][a-z0-9_-]{0,63}.
bool valid_id(const std::string& id);

// Parses and validates the TOML text of a loader.
LoaderParse parse_loader(const std::string& text);
// Validates the JSON form of a loader (for example the arguments of loader.create).
LoaderParse loader_from_json(const nlohmann::json& value);
// The canonical TOML text.
std::string write_loader(const Loader& loader);

// The reader a loader uses before any file is read: the one it names, or the
// one its path's extension implies. Empty when neither says.
std::optional<ReaderKind> declared_reader(const Loader& loader);

// Checks the parts that depend on the reader: tabular readers take mode and
// sql, tensor readers take tensor and transforms, multi-tensor readers need
// tensor, single-tensor readers refuse it, and the reader options must suit the
// reader. Run again once the reader is known.
Error check_loader_for_reader(const Loader& loader, ReaderKind reader);

// Makes a valid id or source name from any text: lower case, every character
// outside [a-z0-9_-] replaced by `_`, a leading "f_" when it does not start
// with a letter, and at most 64 characters (a longer name keeps 55 and ends
// with "-" and 8 hexadecimal digits of a hash of the whole text).
std::string make_id(const std::string& text, bool underscores_only = false);

}  // namespace caliper_builder
