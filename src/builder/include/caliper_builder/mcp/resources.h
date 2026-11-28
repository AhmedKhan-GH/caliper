#pragma once
// The MCP resources of caliper-mcp (design/mcp-server.md, "Resources"):
// read-only text that helps an agent use Caliper, under the URI scheme
// caliper://.
//
//   caliper://docs/<path>      a Markdown page (.md) under the --docs folder
//   caliper://sdk/<path>       a header (.h, .hpp) under the --sdk folder
//   caliper://schema/caliper-spec-v2.schema.json
//                              the JSON Schema of a spec's canonical JSON,
//                              format version 2, compiled into caliper-mcp
//   caliper://commands/command_table.json
//                              the command table the tools come from; the
//                              server adds it, since it depends on the app
//   caliper://library/<path>   a file of the built-in library of pieces
//                              (design/programmable-graphics.md, Library
//                              Pieces): lib/<module>/<version>.wgsl and
//                              pieces/<name>/<version>/piece.toml and its WGSL
//                              files, compiled into caliper-mcp, so that an
//                              agent reads a piece before it pins, copies or
//                              imitates it
//
// The two folders are confined. A path in a URI is percent-decoded once and
// must then be segments of A-Z a-z 0-9 . _ - that are not empty and do not
// start with a dot, so "..", hidden files and separators of another kind are
// refused before the file system is asked. The file's real path, with every
// symbolic link resolved, must lie inside the folder's real path, must be a
// regular file with an allowed extension, and must be at most 4 MiB. Listing
// applies the same rules, does not descend into hidden folders or follow
// linked folders, goes at most 16 levels deep and stops at 5000 files. Files
// are only read, never written.
//
// Error codes: mcp.bad_resource_dir, mcp.resource_not_found,
// mcp.resource_unreadable. No function here throws.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::mcp {

constexpr const char* kDocsUriPrefix = "caliper://docs/";
constexpr const char* kSdkUriPrefix = "caliper://sdk/";
constexpr const char* kSpecSchemaUri = "caliper://schema/caliper-spec-v2.schema.json";
constexpr const char* kCommandTableUri = "caliper://commands/command_table.json";
constexpr const char* kLibraryUriPrefix = "caliper://library/";

struct ResourceEntry {
    std::string uri;
    std::string name;
    std::string title;         // empty: none
    std::string description;   // empty: none
    std::string mime_type;
    std::uint64_t size = 0;    // in bytes; 0 when not known

    nlohmann::json to_json() const;   // an MCP Resource object
};

struct ResourceContent {
    Error error;
    std::string uri;
    std::string mime_type;
    std::string text;
};

struct ResourceOptions {
    std::string docs_dir;           // empty: no docs resources
    std::string sdk_dir;            // empty: no SDK resources
    std::string spec_schema_text;   // empty: no schema resource
    bool library = false;           // serve the built-in library of pieces; caliper-mcp's server does
    std::size_t max_files = 5000;   // per folder
    int max_depth = 16;
    std::size_t max_file_bytes = 4u << 20;
};

// The decoded relative path of a URI's path part, checked as above; empty
// when it is refused.
std::string checked_resource_path(std::string_view encoded);

class ResourceStore {
public:
    // Resolves the folders; each that is given must exist and be a folder
    // (mcp.bad_resource_dir).
    Error open(const ResourceOptions& options);

    // Every resource this store serves, sorted by URI. The folders are
    // scanned at each call. `truncated` is set when a folder had more files
    // than max_files.
    std::vector<ResourceEntry> list(bool* truncated = nullptr) const;

    // The resource templates: one per folder that is given.
    nlohmann::json templates() const;

    // Reads one resource of this store; mcp.resource_not_found for a URI it
    // does not serve or a path that is refused.
    ResourceContent read(const std::string& uri) const;

    bool has_docs() const { return !docs_root_.empty(); }
    bool has_sdk() const { return !sdk_root_.empty(); }

private:
    void scan(const std::string& root, const char* prefix, const std::vector<std::string>& extensions,
              std::vector<ResourceEntry>* out, bool* truncated) const;
    ResourceContent read_file(const std::string& root, const std::string& uri, std::string_view encoded,
                              const std::vector<std::string>& extensions) const;

    ResourceOptions options_;
    std::string docs_root_;   // real paths
    std::string sdk_root_;
};

}  // namespace caliper_builder::mcp
