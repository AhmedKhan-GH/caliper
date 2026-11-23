// The MCP resources of caliper-mcp (resources.h).
#include <caliper_builder/mcp/resources.h>
#include <caliper_builder/programs/pieces.h>

#include "codec.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace caliper_builder::mcp {

namespace fs = std::filesystem;

namespace detail {
// The schema of spec version 1, compiled in (spec_schema_v1_embed.cpp, which
// src/builder/mcp/CMakeLists.txt generates).
extern const char* const kSpecSchemaV1Text;
}  // namespace detail
using nlohmann::json;

namespace {

const std::vector<std::string> kDocsExtensions = {".md"};
const std::vector<std::string> kSdkExtensions = {".h", ".hpp"};

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool segment_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
           c == '-';
}

bool has_extension(const std::string& name, const std::vector<std::string>& extensions) {
    for (const auto& e : extensions)
        if (name.size() > e.size() && name.compare(name.size() - e.size(), e.size(), e) == 0) return true;
    return false;
}

std::string mime_for(const std::string& name) {
    if (has_extension(name, {".md"})) return "text/markdown";
    if (has_extension(name, {".hpp"})) return "text/x-c++hdr";
    if (has_extension(name, {".h"})) return "text/x-c";
    return "text/plain";
}

// True when `inner` is `root` or lies below it, compared by path components.
bool is_inside(const fs::path& root, const fs::path& inner) {
    auto r = root.begin();
    auto i = inner.begin();
    for (; r != root.end(); ++r, ++i) {
        if (i == inner.end() || *r != *i) return false;
    }
    return true;
}

std::string real_folder(const std::string& dir, Error* error, const char* what) {
    std::error_code ec;
    const fs::path real = fs::canonical(dir, ec);
    if (ec || !fs::is_directory(real, ec)) {
        *error = {"mcp.bad_resource_dir", std::string("The ") + what + " folder does not exist or is not a folder.", dir};
        return {};
    }
    return real.string();
}

// The MIME type of a library file: WGSL or TOML.
std::string library_mime(std::string_view path) {
    return path.size() >= 5 && path.substr(path.size() - 5) == ".toml" ? "application/toml" : "text/wgsl";
}

}  // namespace

json ResourceEntry::to_json() const {
    json j = {{"uri", uri}, {"name", name}, {"mimeType", mime_type}};
    if (!title.empty()) j["title"] = title;
    if (!description.empty()) j["description"] = description;
    if (size > 0) j["size"] = size;
    return j;
}

std::string checked_resource_path(std::string_view encoded) {
    if (encoded.empty() || encoded.size() > 3072) return {};
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '%') {
            decoded += encoded[i];
            continue;
        }
        if (i + 2 >= encoded.size()) return {};
        const int hi = hex_value(encoded[i + 1]), lo = hex_value(encoded[i + 2]);
        if (hi < 0 || lo < 0) return {};
        decoded += char(hi * 16 + lo);
        i += 2;
    }
    if (decoded.size() > 1024) return {};
    std::size_t start = 0;
    int segments = 0;
    for (;;) {
        const std::size_t slash = decoded.find('/', start);
        const std::string_view seg =
            std::string_view(decoded).substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (seg.empty() || seg.size() > 255 || seg[0] == '.' || ++segments > 32) return {};
        for (char c : seg)
            if (!segment_char(c)) return {};
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return decoded;
}

Error ResourceStore::open(const ResourceOptions& options) {
    options_ = options;
    if (options_.spec_schema_text.empty()) {
        options_.spec_schema_v1_text.clear();
    } else if (options_.spec_schema_v1_text.empty() && detail::kSpecSchemaV1Text) {
        options_.spec_schema_v1_text = detail::kSpecSchemaV1Text;
    }
    docs_root_.clear();
    sdk_root_.clear();
    Error error;
    if (!options.docs_dir.empty()) {
        docs_root_ = real_folder(options.docs_dir, &error, "docs");
        if (error) return error;
    }
    if (!options.sdk_dir.empty()) {
        sdk_root_ = real_folder(options.sdk_dir, &error, "SDK");
        if (error) return error;
    }
    return {};
}

void ResourceStore::scan(const std::string& root, const char* prefix, const std::vector<std::string>& extensions,
                         std::vector<ResourceEntry>* out, bool* truncated) const {
    std::error_code ec;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    std::size_t count = 0;
    const fs::path root_path(root);
    for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        const fs::directory_entry& entry = *it;
        const std::string leaf = entry.path().filename().string();
        std::error_code e2;
        if (leaf.empty() || leaf[0] == '.') {
            if (entry.is_directory(e2)) it.disable_recursion_pending();
            continue;
        }
        if (entry.is_directory(e2)) {
            if (it.depth() + 1 >= options_.max_depth) it.disable_recursion_pending();
            continue;
        }
        if (!has_extension(leaf, extensions)) continue;
        const std::string rel = entry.path().lexically_relative(root_path).generic_string();
        if (checked_resource_path(rel) != rel) continue;
        const fs::path real = fs::canonical(entry.path(), e2);
        if (e2 || !is_inside(root_path, real) || !fs::is_regular_file(real, e2)) continue;
        if (!has_extension(real.filename().string(), extensions)) continue;
        const auto size = fs::file_size(real, e2);
        if (e2 || size > options_.max_file_bytes) continue;
        if (count >= options_.max_files) {
            if (truncated) *truncated = true;
            break;
        }
        ++count;
        ResourceEntry r;
        r.uri = prefix + rel;
        r.name = rel;
        r.mime_type = mime_for(rel);
        r.size = size;
        out->push_back(std::move(r));
    }
}

std::vector<ResourceEntry> ResourceStore::list(bool* truncated) const {
    if (truncated) *truncated = false;
    std::vector<ResourceEntry> out;
    if (!docs_root_.empty()) scan(docs_root_, kDocsUriPrefix, kDocsExtensions, &out, truncated);
    if (!sdk_root_.empty()) scan(sdk_root_, kSdkUriPrefix, kSdkExtensions, &out, truncated);
    if (!options_.spec_schema_text.empty()) {
        ResourceEntry r;
        r.uri = kSpecSchemaUri;
        r.name = "caliper-spec-v0.schema.json";
        r.title = "Spec format v0 JSON Schema";
        r.description = "The JSON Schema of the canonical JSON form of a spec, format version 0.";
        r.mime_type = "application/schema+json";
        r.size = options_.spec_schema_text.size();
        out.push_back(std::move(r));
    }
    if (!options_.spec_schema_v1_text.empty()) {
        ResourceEntry r;
        r.uri = kSpecSchemaV1Uri;
        r.name = "caliper-spec-v1.schema.json";
        r.title = "Spec format v1 JSON Schema";
        r.description = "The JSON Schema of the canonical JSON form of a spec, format version 1: version 0 with the "
                        "scene primitive, the control table, the GPU programs, and the provenance table.";
        r.mime_type = "application/schema+json";
        r.size = options_.spec_schema_v1_text.size();
        out.push_back(std::move(r));
    }
    if (options_.library) {
        for (const programs::LibraryFile& f : programs::library_files()) {
            ResourceEntry r;
            r.uri = kLibraryUriPrefix + std::string(f.path);
            r.name = std::string(f.path);
            r.description = f.path.rfind("lib/", 0) == 0 ? "A lib module of the built-in library of pieces."
                                                         : "A file of a piece of the built-in library.";
            r.mime_type = library_mime(f.path);
            r.size = f.text.size();
            out.push_back(std::move(r));
        }
    }
    std::sort(out.begin(), out.end(), [](const ResourceEntry& a, const ResourceEntry& b) { return a.uri < b.uri; });
    return out;
}

json ResourceStore::templates() const {
    json out = json::array();
    if (!docs_root_.empty())
        out.push_back({{"uriTemplate", std::string(kDocsUriPrefix) + "{+path}"},
                       {"name", "docs"},
                       {"title", "Caliper documentation pages"},
                       {"description", "A Markdown page of the Caliper documentation, by its path under docs/, such as "
                                       "agents/http-api.md."},
                       {"mimeType", "text/markdown"}});
    if (!sdk_root_.empty())
        out.push_back({{"uriTemplate", std::string(kSdkUriPrefix) + "{+path}"},
                       {"name", "sdk"},
                       {"title", "Caliper SDK headers"},
                       {"description", "A header of the Caliper SDK, by its path under the SDK folder."},
                       {"mimeType", "text/x-c"}});
    return out;
}

ResourceContent ResourceStore::read_file(const std::string& root, const std::string& uri, std::string_view encoded,
                                         const std::vector<std::string>& extensions) const {
    ResourceContent out;
    out.uri = uri;
    auto not_found = [&](const std::string& why) {
        out.error = {"mcp.resource_not_found", "No such resource: " + why + ".", uri};
        return out;
    };
    const std::string rel = checked_resource_path(encoded);
    if (rel.empty()) return not_found("the path is not one this server serves");
    if (!has_extension(rel, extensions)) return not_found("this folder serves only files with another extension");
    std::error_code ec;
    const fs::path root_path(root);
    const fs::path real = fs::canonical(root_path / fs::path(rel), ec);
    if (ec) return not_found("the file does not exist");
    if (!is_inside(root_path, real)) return not_found("the path leads outside its folder");
    if (!fs::is_regular_file(real, ec)) return not_found("it is not a regular file");
    if (!has_extension(real.filename().string(), extensions)) return not_found("the file it links to has another extension");
    std::string why;
    if (!detail::read_small_file(real.string(), options_.max_file_bytes, &out.text, &why)) {
        out.error = {"mcp.resource_unreadable", "The resource cannot be read: " + why + ".", uri};
        return out;
    }
    out.mime_type = mime_for(rel);
    return out;
}

ResourceContent ResourceStore::read(const std::string& uri) const {
    const std::string docs = kDocsUriPrefix, sdk = kSdkUriPrefix;
    if (uri.compare(0, docs.size(), docs) == 0 && !docs_root_.empty())
        return read_file(docs_root_, uri, std::string_view(uri).substr(docs.size()), kDocsExtensions);
    if (uri.compare(0, sdk.size(), sdk) == 0 && !sdk_root_.empty())
        return read_file(sdk_root_, uri, std::string_view(uri).substr(sdk.size()), kSdkExtensions);
    ResourceContent out;
    out.uri = uri;
    if (uri == kSpecSchemaV1Uri && !options_.spec_schema_v1_text.empty()) {
        out.mime_type = "application/schema+json";
        out.text = options_.spec_schema_v1_text;
        return out;
    }
    const std::string library = kLibraryUriPrefix;
    if (options_.library && uri.compare(0, library.size(), library) == 0) {
        const std::string_view path = std::string_view(uri).substr(library.size());
        for (const programs::LibraryFile& f : programs::library_files()) {
            if (f.path != path) continue;
            out.mime_type = library_mime(f.path);
            out.text = std::string(f.text);
            return out;
        }
    }
    if (uri == kSpecSchemaUri && !options_.spec_schema_text.empty()) {
        out.mime_type = "application/schema+json";
        out.text = options_.spec_schema_text;
        return out;
    }
    out.error = {"mcp.resource_not_found", "No such resource: this server does not serve that URI.",
                 uri.size() <= 1024 ? uri : uri.substr(0, 1024)};
    return out;
}

}  // namespace caliper_builder::mcp
