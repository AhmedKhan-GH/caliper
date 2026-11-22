// The sources of pieces and the piece commands (piece_commands.h,
// design/programmable-graphics.md: Piece Files, Piece Sources, Versions and
// Pins, Piece Commands).
//
// Every path into the workspace is built from a checked name and version, the
// pieces folder and every folder and file below it is refused when it is a
// link, and a file is read only when its real path is inside
// <workspace>/pieces/ and its size is inside its limit. Writes go to a fresh
// hidden folder beside the version's folder and become the version with one
// rename, so that a reader never sees half a piece; a mutex orders the
// writes of this process.
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <system_error>
#include <utility>

namespace caliper_builder {

using nlohmann::json;
namespace fs = std::filesystem;
namespace cp = programs;

namespace {

constexpr const char* kPiecesFolder = "pieces";
constexpr std::size_t kManifestBytes = 256 * 1024;  // a piece.toml
constexpr std::size_t kFolderEntries = 64;          // files and folders of one version's folder
constexpr std::size_t kVersionsListed = 4096;       // versions of one name that a listing reads

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

std::string quote(std::string_view s) { return "\"" + std::string(s.substr(0, 64)) + "\""; }

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::optional<std::uint32_t> version_arg(const json& args, const char* key = "version") {
    const auto it = args.find(key);
    if (it == args.end() || !it->is_number_integer()) return std::nullopt;
    const auto v = it->get<std::int64_t>();
    if (v < 1 || v > spec::piece_limits::kVersionMax) return std::nullopt;
    return static_cast<std::uint32_t>(v);
}

std::optional<spec::PieceFrom> from_arg(const json& args) {
    const std::string f = str_arg(args, "from");
    if (f.empty()) return std::nullopt;
    return spec::piece_from_from_name(f);
}


// The first error of a decode, as an error of the command.
Error first_error(const std::vector<spec::Diagnostic>& ds, const char* code, const std::string& base) {
    for (const spec::Diagnostic& d : ds) {
        if (!d.is_error()) continue;
        std::string message = d.message;
        if (d.line > 0) message += " (line " + std::to_string(d.line) + ", column " + std::to_string(d.column) + ")";
        return fail(code ? code : d.code, message, d.path.empty() ? base : base + "." + d.path);
    }
    return {};
}

// ---- base64 of a sample's data (RFC 4648, with padding)

bool base64_decode(std::string_view in, std::string* out) {
    auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (in.size() % 4 != 0) return false;
    out->clear();
    out->reserve(in.size() / 4 * 3);
    for (std::size_t i = 0; i < in.size(); i += 4) {
        int v[4];
        int pad = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = in[i + static_cast<std::size_t>(k)];
            if (c == '=' && i + 4 == in.size() && k >= 2) {
                v[k] = 0;
                ++pad;
            } else if (pad > 0 || (v[k] = value(c)) < 0) {
                return false;
            }
        }
        const std::uint32_t n = (std::uint32_t(v[0]) << 18) | (std::uint32_t(v[1]) << 12) | (std::uint32_t(v[2]) << 6) |
                                std::uint32_t(v[3]);
        out->push_back(static_cast<char>((n >> 16) & 0xFF));
        if (pad < 2) out->push_back(static_cast<char>((n >> 8) & 0xFF));
        if (pad < 1) out->push_back(static_cast<char>(n & 0xFF));
    }
    return true;
}

// ---- the workspace's pieces folder

bool is_link(const fs::path& p) {
    std::error_code ec;
    return fs::is_symlink(fs::symlink_status(p, ec));
}

// Whether `real` lies strictly inside `base`; both are canonical.
bool inside(const fs::path& real, const fs::path& base) {
    auto r = real.begin();
    for (auto b = base.begin(); b != base.end(); ++b, ++r)
        if (r == real.end() || *r != *b) return false;
    return r != real.end();
}

// The workspace's pieces folder: <canonical workspace>/pieces. It is refused
// when it is a link or not a folder; `create` makes it when it is absent.
Error pieces_root(const std::string& workspace, bool create, fs::path* root) {
    std::error_code ec;
    if (workspace.empty()) return fail(cp::kPieceBadPath, "The session has no workspace folder for pieces.");
    const fs::path w = fs::canonical(fs::u8path(workspace), ec);
    if (ec || !fs::is_directory(w, ec)) return fail(cp::kPieceBadPath, "The workspace folder cannot be resolved.");
    const fs::path p = w / kPiecesFolder;
    fs::file_status st = fs::symlink_status(p, ec);
    if (fs::is_symlink(st))
        return fail(cp::kPieceBadPath, "The workspace's pieces folder is a link; the host refuses links there.");
    if (!fs::exists(st) && create) {
        fs::create_directory(p, ec);
        if (ec) return fail(cp::kPieceBadPath, "The workspace's pieces folder cannot be made: " + ec.message() + ".");
        st = fs::symlink_status(p, ec);
    }
    if (fs::exists(st) && (fs::is_symlink(st) || !fs::is_directory(st)))
        return fail(cp::kPieceBadPath, "The workspace's pieces entry is not a folder.");
    *root = p;
    return {};
}

// Reads a regular file of at most `limit` bytes whose real path lies inside
// `root`. A file that does not exist is piece.not_found; a link, another kind
// of entry, a real path outside or a larger file is piece.bad_path.
Error read_inside(const fs::path& root, const fs::path& file, std::size_t limit, std::string* out) {
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(file, ec);
    const std::string shown = file.filename().u8string();
    if (ec || !fs::exists(st)) return fail(cp::kPieceNotFound, "The piece has no file " + shown + ".");
    if (fs::is_symlink(st) || !fs::is_regular_file(st))
        return fail(cp::kPieceBadPath, "The piece's " + shown + " is a link or not a regular file.");
    const fs::path base = fs::canonical(root, ec);
    if (ec) return fail(cp::kPieceBadPath, "The pieces folder cannot be resolved.");
    const fs::path real = fs::canonical(file, ec);
    if (ec || !inside(real, base))
        return fail(cp::kPieceBadPath, "The piece's " + shown + " lies outside the workspace's pieces folder.");
    const std::uintmax_t size = fs::file_size(real, ec);
    if (ec || size > limit)
        return fail(cp::kPieceBadPath,
                    "The piece's " + shown + " holds more than " + std::to_string(limit) + " bytes, its limit.");
    std::ifstream in(real, std::ios::binary);
    out->assign(static_cast<std::size_t>(size), '\0');
    if (!in.read(out->data(), static_cast<std::streamsize>(size)))
        return fail(cp::kPieceBadPath, "The piece's " + shown + " cannot be read.");
    return {};
}

// A version's folder holds no link anywhere, and at most kFolderEntries entries.
Error no_links(const fs::path& folder) {
    std::error_code ec;
    std::size_t n = 0;
    for (fs::recursive_directory_iterator it(folder, fs::directory_options::none, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (++n > kFolderEntries)
            return fail(cp::kPieceBadPath, "The piece's folder holds more entries than a piece has.");
        if (it->is_symlink(ec))
            return fail(cp::kPieceBadPath, "The piece's folder holds the link " + it->path().filename().u8string() +
                                               "; the host refuses links in a piece.");
        if (it.depth() >= 1) it.disable_recursion_pending();
    }
    if (ec) return fail(cp::kPieceBadPath, "The piece's folder cannot be listed.");
    return {};
}

// A version folder's name: digits without a leading zero, 1 to kVersionMax.
std::optional<std::uint32_t> version_name(const std::string& s) {
    if (s.empty() || s.size() > 7 || s[0] == '0') return std::nullopt;
    std::uint32_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return std::nullopt;
        v = v * 10 + static_cast<std::uint32_t>(c - '0');
    }
    if (v < 1 || v > spec::piece_limits::kVersionMax) return std::nullopt;
    return v;
}

// The saved versions of a name in the workspace, ascending.
std::vector<std::uint32_t> workspace_versions(const fs::path& root, const std::string& name) {
    std::vector<std::uint32_t> out;
    if (!cp::check_piece_name(name, 1).ok()) return out;
    const fs::path dir = root / name;
    std::error_code ec;
    if (is_link(dir) || !fs::is_directory(fs::symlink_status(dir, ec))) return out;
    std::size_t seen = 0;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end && seen < kVersionsListed; it.increment(ec), ++seen) {
        if (it->is_symlink(ec) || !it->is_directory(ec)) continue;
        if (auto v = version_name(it->path().filename().u8string())) out.push_back(*v);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The names of the workspace's pieces.
std::vector<std::string> workspace_names(const fs::path& root) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(fs::symlink_status(root, ec))) return out;
    std::size_t seen = 0;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end && seen < kVersionsListed; it.increment(ec), ++seen) {
        const std::string n = it->path().filename().u8string();
        if (!it->is_symlink(ec) && it->is_directory(ec) && cp::check_piece_name(n, 1).ok()) out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string workspace_path(const std::string& name, std::uint32_t version) {
    return std::string(kPiecesFolder) + "/" + name + "/" + std::to_string(version);
}

// One saved version of the workspace, with its hash checked against a hash
// that its manifest gives.
FoundPiece read_workspace_piece(const fs::path& root, const std::string& name, std::uint32_t version) {
    FoundPiece f;
    f.from = spec::PieceFrom::workspace;
    if (cp::Verdict v = cp::check_piece_name(name, version); !v.ok()) {
        f.error = fail(v.code, v.message, v.where);
        return f;
    }
    const fs::path folder = root / name / std::to_string(version);
    std::error_code ec;
    if (is_link(root / name) || is_link(folder)) {
        f.error = fail(cp::kPieceBadPath, "The piece's folder is a link; the host refuses links in the pieces folder.");
        return f;
    }
    if (!fs::is_directory(fs::symlink_status(folder, ec))) {
        f.error = fail(cp::kPieceNotFound,
                       "The workspace has no piece " + name + " version " + std::to_string(version) + ".");
        return f;
    }
    if (Error e = no_links(folder)) {
        f.error = e;
        return f;
    }
    std::string manifest;
    if (Error e = read_inside(root, folder / "piece.toml", kManifestBytes, &manifest)) {
        f.error = e.code == cp::kPieceNotFound ? fail(cp::kPieceBadManifest, "The piece's folder has no piece.toml.") : e;
        return f;
    }
    spec::PieceParse parsed = spec::piece_from_toml(manifest);
    if (Error e = first_error(parsed.diagnostics, cp::kPieceBadManifest, "piece.toml")) {
        f.error = e;
        return f;
    }
    f.piece = std::move(parsed.piece);
    if (f.piece.name != name || f.piece.version != version) {
        f.error = fail(cp::kPieceBadManifest, "The manifest names " + f.piece.name + " version " +
                                                  std::to_string(f.piece.version) + ", and its folder is " +
                                                  workspace_path(name, version) + ".");
        return f;
    }
    for (const char* file : spec::kPieceFiles) {
        std::string text;
        Error e = read_inside(root, folder / file, spec::piece_limits::kFileBytes, &text);
        if (e.code == cp::kPieceNotFound) continue;
        if (e) {
            f.error = e;
            return f;
        }
        f.piece.files[file] = std::move(text);
    }
    const std::string hash = cp::piece_hash(f.piece);
    if (!f.piece.hash.empty() && f.piece.hash != hash) {
        f.error = fail(cp::kPieceHashMismatch, "The piece.toml of " + workspace_path(name, version) +
                                                   " gives a hash that differs from its files' hash " + hash +
                                                   "; a saved version never changes.");
        return f;
    }
    f.piece.hash = hash;
    f.path = workspace_path(name, version);
    return f;
}

std::mutex& write_mutex() {
    static std::mutex m;
    return m;
}

// Writes a new version of the workspace: piece.toml with the hash, the WGSL
// files and the samples, into a hidden folder that one rename makes the
// version's folder. A version that exists is piece.exists.
Error write_workspace_piece(const std::string& workspace, const spec::Piece& piece,
                            const std::map<std::string, std::string>& samples, std::string* path) {
    if (cp::Verdict v = cp::check_piece_name(piece.name, piece.version); !v.ok()) return fail(v.code, v.message, v.where);
    std::lock_guard<std::mutex> lock(write_mutex());
    fs::path root;
    if (Error e = pieces_root(workspace, true, &root)) return e;
    std::error_code ec;
    const fs::path dir = root / piece.name;
    if (is_link(dir)) return fail(cp::kPieceBadPath, "The piece's folder is a link; the host refuses links there.");
    if (!fs::exists(fs::symlink_status(dir, ec))) fs::create_directory(dir, ec);
    if (ec || !fs::is_directory(fs::symlink_status(dir, ec)))
        return fail(cp::kPieceBadPath, "The folder of the piece " + piece.name + " cannot be made.");
    const fs::path final_folder = dir / std::to_string(piece.version);
    if (fs::exists(fs::symlink_status(final_folder, ec)))
        return fail(cp::kPieceExists, "The workspace already has " + piece.name + " version " +
                                          std::to_string(piece.version) + "; a saved version never changes.");
    static std::atomic<std::uint64_t> counter{0};
    const fs::path tmp = dir / (".tmp-" + std::to_string(piece.version) + "-" + std::to_string(++counter));
    fs::remove_all(tmp, ec);
    if (!fs::create_directory(tmp, ec) || ec)
        return fail(cp::kPieceBadPath, "A folder for the new version cannot be made: " + ec.message() + ".");
    auto write = [&](const fs::path& file, const std::string& bytes) {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(out);
    };
    bool ok = write(tmp / "piece.toml", spec::piece_to_toml(piece));
    for (const auto& [file, text] : piece.files) ok = ok && write(tmp / file, text);
    if (ok && !samples.empty()) {
        ok = fs::create_directory(tmp / "sample", ec) && !ec;
        for (const auto& [input, bytes] : samples) ok = ok && write(tmp / "sample" / (input + ".bin"), bytes);
    }
    if (ok) fs::rename(tmp, final_folder, ec);
    if (!ok || ec) {
        std::error_code ignored;
        fs::remove_all(tmp, ignored);
        if (fs::exists(fs::symlink_status(final_folder, ignored)))
            return fail(cp::kPieceExists, "The workspace already has " + piece.name + " version " +
                                              std::to_string(piece.version) + ".");
        return fail(cp::kPieceBadPath, "The new version could not be written to the workspace.");
    }
    *path = workspace_path(piece.name, piece.version);
    return {};
}

Error remove_workspace_piece(const std::string& workspace, const std::string& name, std::uint32_t version) {
    if (cp::Verdict v = cp::check_piece_name(name, version); !v.ok()) return fail(v.code, v.message, v.where);
    std::lock_guard<std::mutex> lock(write_mutex());
    fs::path root;
    if (Error e = pieces_root(workspace, false, &root)) return e;
    const fs::path dir = root / name;
    const fs::path folder = dir / std::to_string(version);
    std::error_code ec;
    if (is_link(dir) || is_link(folder))
        return fail(cp::kPieceBadPath, "The piece's folder is a link; the host refuses links in the pieces folder.");
    if (!fs::is_directory(fs::symlink_status(folder, ec)))
        return fail(cp::kPieceNotFound, "The workspace has no piece " + name + " version " + std::to_string(version) + ".");
    fs::remove_all(folder, ec);  // removes a link inside as a link, never what it names
    if (ec) return fail(cp::kPieceBadPath, "The version's folder could not be removed: " + ec.message() + ".");
    if (fs::is_empty(dir, ec) && !ec) fs::remove(dir, ec);
    return {};
}

// ---- messages

json message(const char* severity, const std::string& code, const std::string& text, const std::string& file,
             std::uint32_t line, std::uint32_t column, const std::string& where = {}) {
    json m = {{"severity", severity}, {"code", code}, {"message", text}, {"line", line}, {"column", column}};
    if (!file.empty()) m["file"] = file;
    if (!where.empty()) m["where"] = where;
    return m;
}

json verdict_message(const cp::Verdict& v, const std::string& file) {
    return message("error", v.code, v.message, file, v.line, v.column, v.where);
}

bool has_errors(const json& messages) {
    for (const json& m : messages)
        if (m.value("severity", "") == "error") return true;
    return false;
}

// The lines of a composed module's generated header.
std::string header_of(const cp::Composed& c) {
    for (const cp::Segment& s : c.segments) {
        if (s.part != "header") continue;
        std::size_t begin = 0;
        for (std::uint32_t l = 1; l < s.first_line && begin != std::string::npos; ++l) {
            begin = c.text.find('\n', begin);
            if (begin != std::string::npos) ++begin;
        }
        if (begin == std::string::npos) return std::string();
        std::size_t end = begin;
        for (std::uint32_t l = 0; l < s.lines && end != std::string::npos; ++l) {
            end = c.text.find('\n', end);
            if (end != std::string::npos) ++end;
        }
        return c.text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    }
    return std::string();
}

// Composes and compiles each pass of one half, and adds its messages.
void check_half(const spec::Piece& piece, const cp::Resolved& resolved, cp::Half half, json& messages, json& passes,
                std::string* header) {
    const bool derive = half == cp::Half::derive;
    const std::size_t n = derive ? piece.derive->passes.size() : piece.draw->passes.size();
    const char* half_name = derive ? "derive" : "draw";
    for (std::size_t i = 0; i < n; ++i) {
        const cp::Composed c = derive ? cp::compose_derive(piece, resolved) : cp::compose_draw(piece, resolved, i);
        if (c.verdict.code == cp::kProgramNotBuilt) {
            messages.push_back(message("warning", cp::kProgramNotBuilt,
                                       std::string("The composer of the ") + half_name +
                                           " half is not in this build, so its passes were not compiled here.",
                                       "", 0, 0));
            return;
        }
        if (!c.verdict.ok()) {
            messages.push_back(verdict_message(c.verdict, ""));
            passes.push_back({{"half", half_name}, {"index", i}, {"ok", false}});
            continue;
        }
        if (header->empty()) *header = header_of(c);
        if (!cp::compiler_available()) {
            if (i == 0)
                messages.push_back(message("warning", cp::kProgramCompilerUnavailable,
                                           "This build has no WGSL compiler, so the passes were not compiled here.", "",
                                           0, 0));
            if (derive) return;
            continue;
        }
        const cp::Compiled compiled = cp::compile_and_reflect(c.text);
        bool ok = compiled.verdict.ok();
        if (!ok) {
            const cp::SourcePlace place = cp::source_place(c, compiled.verdict.line, compiled.verdict.column);
            messages.push_back(message("error", compiled.verdict.code, compiled.verdict.message, place.part, place.line,
                                       place.column, std::string(half_name) + ".passes[" + std::to_string(i) + "]"));
        } else {
            const cp::Verdict r = cp::check_piece_reflection(piece, half, i, compiled.reflection);
            if (r.code == cp::kProgramNotBuilt) {
                if (i == 0)
                    messages.push_back(message("warning", cp::kProgramNotBuilt,
                                               "The reflection check of pieces is not in this build.", "", 0, 0));
            } else if (!r.ok()) {
                ok = false;
                messages.push_back(verdict_message(r, ""));
            }
        }
        if (derive) {  // the derive half is one module for all its passes
            for (std::size_t k = 0; k < n; ++k) passes.push_back({{"half", half_name}, {"index", k}, {"ok", ok}});
            return;
        }
        passes.push_back({{"half", half_name}, {"index", i}, {"ok", ok}});
    }
}

// A piece's summary in piece.list.
json summary(const spec::Piece& p, spec::PieceFrom from, const spec::Spec& spec) {
    json inputs = json::array(), params = json::array(), outputs = json::array(), pinned = json::array();
    for (const auto& in : p.inputs) inputs.push_back(in.name);
    for (const auto& pa : p.params) params.push_back(pa.name);
    for (const auto& r : p.resources)
        if (r.publish) outputs.push_back(r.id);
    for (const spec::Program& prog : spec.programs)
        if (prog.piece && prog.piece->name == p.name && prog.piece->version == p.version && prog.piece->from == from)
            pinned.push_back(prog.id);
    return {{"name", p.name},          {"version", p.version}, {"from", spec::piece_from_name(from)},
            {"description", p.description}, {"hash", p.hash}, {"draws", p.draw.has_value()},
            {"derives", p.derive.has_value()}, {"inputs", inputs}, {"params", params},
            {"outputs", outputs},       {"pinned_by", pinned}};
}

json pinned_by(const spec::Spec& spec, const std::string& name, std::uint32_t version, spec::PieceFrom from) {
    json out = json::array();
    for (const spec::Program& p : spec.programs)
        if (p.piece && p.piece->name == name && p.piece->version == version && p.piece->from == from)
            out.push_back(p.id);
    return out;
}

// The piece of a command's `piece` argument: the samples' data set aside,
// the rest decoded and checked as piece.check checks it.
struct Given {
    Error error;
    spec::Piece piece;
    json check;
    std::map<std::string, std::string> samples;  // input to bytes
};

Given given_piece(const json& args) {
    Given g;
    json piece_json = args.contains("piece") ? args["piece"] : json::object();
    if (piece_json.is_object() && piece_json.contains("sample") && piece_json["sample"].is_array()) {
        for (json& s : piece_json["sample"]) {
            if (!s.is_object() || !s.contains("data")) continue;
            std::string bytes;
            if (!s["data"].is_string() || !base64_decode(s["data"].get<std::string>(), &bytes)) {
                g.error = fail(cp::kPieceBadManifest, "A sample's data is base64 of its raw little-endian elements.",
                               "args.piece.sample");
                return g;
            }
            g.samples[s.value("input", "")] = std::move(bytes);
            s.erase("data");
        }
    }
    g.check = check_piece_json(piece_json, false, &g.piece);
    for (const json& m : g.check["messages"]) {
        if (m.value("severity", "") != "error") continue;
        std::string text = m.value("message", "");
        const std::string file = m.value("file", "");
        if (!file.empty() && m.value("line", 0) > 0)
            text += " (" + file + ", line " + std::to_string(m.value("line", 0)) + ", column " +
                    std::to_string(m.value("column", 0)) + ")";
        g.error = fail(m.value("code", "piece.bad_manifest"), text, "args.piece" + (m.contains("where") ? "." + m["where"].get<std::string>() : std::string()));
        return g;
    }
    // Each sample's bytes match its shape: 4 bytes an element of a tensor
    // input, the layout's stride of a buffer input.
    for (const auto& [input, bytes] : g.samples) {
        auto it = g.piece.sample.find(input);
        if (it == g.piece.sample.end()) {
            g.error = fail(cp::kPieceBadManifest, "A sample names " + quote(input) + ", which is no input.",
                           "args.piece.sample");
            return g;
        }
        std::uint64_t element = 4;
        for (const auto& in : g.piece.inputs) {
            cp::ElementLayout l;
            if (in.name == input && in.kind == spec::InputKind::buffer && cp::element_layout(in.layout, &l).ok())
                element = l.stride;
        }
        std::uint64_t want = element;
        for (std::uint32_t n : it->second.shape) want = std::min<std::uint64_t>(want * n, std::uint64_t(1) << 40);
        if (want != bytes.size()) {
            g.error = fail(cp::kPieceBadManifest,
                           "The sample of " + quote(input) + " holds " + std::to_string(bytes.size()) +
                               " bytes, and its shape needs " + std::to_string(want) + ".",
                           "args.piece.sample");
            return g;
        }
    }
    return g;
}

json warnings_of(const json& check) {
    json out = json::array();
    for (const json& m : check["messages"])
        if (m.value("severity", "") == "warning") out.push_back(m);
    return out;
}

// ---- the commands

CommandOutput piece_check(const json& args) {
    spec::Piece piece;
    json result = check_piece_json(args.contains("piece") ? args["piece"] : json::object(),
                                   args.value("headers", false), &piece);
    return CommandOutput::ok(std::move(result));
}

CommandOutput piece_list(const json& args, const spec::Spec& spec, const std::string& workspace,
                         const PieceLibrary& library) {
    const std::optional<spec::PieceFrom> from = from_arg(args);
    const std::string match = str_arg(args, "match");
    auto wanted = [&](spec::PieceFrom f, const std::string& name) {
        return (!from || *from == f) && (match.empty() || name.find(match) != std::string::npos);
    };
    json list = json::array();
    for (const spec::Piece& p : library.pieces())
        if (wanted(spec::PieceFrom::library, p.name)) list.push_back(summary(p, spec::PieceFrom::library, spec));
    fs::path root;
    if (!from || *from == spec::PieceFrom::workspace) {
        if (!pieces_root(workspace, false, &root)) {
            for (const std::string& name : workspace_names(root)) {
                if (!wanted(spec::PieceFrom::workspace, name)) continue;
                for (std::uint32_t v : workspace_versions(root, name)) {
                    FoundPiece f = read_workspace_piece(root, name, v);
                    if (!f.error) list.push_back(summary(f.piece, spec::PieceFrom::workspace, spec));
                }
            }
        }
    }
    for (const spec::Piece& p : spec.pieces) {
        if (!wanted(spec::PieceFrom::inline_piece, p.name)) continue;
        spec::Piece q = p;
        q.hash = cp::piece_hash(p);
        list.push_back(summary(q, spec::PieceFrom::inline_piece, spec));
    }
    json modules = json::array();
    for (std::string_view m : cp::lib_modules()) modules.push_back(std::string(m));
    return CommandOutput::ok({{"pieces", std::move(list)}, {"modules", std::move(modules)}});
}

CommandOutput piece_show(const json& args, const spec::Spec& spec, const std::string& workspace,
                         const PieceLibrary& library) {
    const std::string name = str_arg(args, "name");
    const std::optional<std::uint32_t> version = version_arg(args);
    std::vector<spec::PieceFrom> order = {spec::PieceFrom::library, spec::PieceFrom::workspace,
                                          spec::PieceFrom::inline_piece};
    if (auto f = from_arg(args)) order = {*f};
    FoundPiece found;
    for (spec::PieceFrom f : order) {
        found = find_piece(library, workspace, spec, f, name, version);
        if (found.error.code != cp::kPieceNotFound) break;
    }
    if (found.error) return CommandOutput::fail(found.error);
    return CommandOutput::ok({{"piece", spec::to_json(found.piece)},
                              {"from", spec::piece_from_name(found.from)},
                              {"path", found.path.empty() ? json(nullptr) : json(found.path)},
                              {"pinned_by", pinned_by(spec, found.piece.name, found.piece.version, found.from)}});
}

CommandOutput piece_copy(const json& args, const std::string& workspace, const PieceLibrary& library) {
    const std::string name = str_arg(args, "name");
    const std::string as = args.contains("as") ? str_arg(args, "as") : name;
    if (cp::Verdict v = cp::check_piece_name(name, 1); !v.ok()) return CommandOutput::fail(fail(v.code, v.message, "args.name"));
    if (cp::Verdict v = cp::check_piece_name(as, 1); !v.ok()) return CommandOutput::fail(fail(v.code, v.message, "args.as"));
    const spec::Piece* source = library.find(name, version_arg(args));
    if (!source)
        return CommandOutput::fail(fail(cp::kPieceNotFound, "The library has no piece " + name +
                                                                (version_arg(args) ? " version " + std::to_string(*version_arg(args)) : "") +
                                                                "; piece.list lists the library.",
                                        "args.name"));
    fs::path root;
    if (Error e = pieces_root(workspace, false, &root)) return CommandOutput::fail(e);
    if (!workspace_versions(root, as).empty())
        return CommandOutput::fail(fail(cp::kPieceExists,
                                        "The workspace already has the piece " + as +
                                            "; copy it as another name, or write its next version with piece.update.",
                                        "args.as"));
    spec::Piece copy = *source;
    copy.name = as;
    copy.version = 1;
    copy.sample.clear();  // the library's samples are not embedded
    copy.hash = cp::piece_hash(copy);
    std::string path;
    if (Error e = write_workspace_piece(workspace, copy, {}, &path)) return CommandOutput::fail(e);
    return CommandOutput::ok({{"name", copy.name}, {"version", copy.version}, {"hash", copy.hash}, {"path", path}});
}

CommandOutput piece_update(const json& args, const std::string& workspace) {
    Given g = given_piece(args);
    if (g.error) return CommandOutput::fail(g.error);
    spec::Piece piece = std::move(g.piece);
    fs::path root;
    if (Error e = pieces_root(workspace, false, &root)) return CommandOutput::fail(e);
    const std::vector<std::uint32_t> saved = workspace_versions(root, piece.name);
    const bool explicit_version = args.contains("piece") && args["piece"].is_object() && args["piece"].contains("version");
    if (!explicit_version) {
        piece.version = saved.empty() ? 1 : saved.back() + 1;
        if (piece.version > spec::piece_limits::kVersionMax)
            return CommandOutput::fail(fail(cp::kPieceBadName, "The piece has no version left.", "args.piece.version"));
    } else if (!saved.empty() && piece.version <= saved.back()) {
        return CommandOutput::fail(fail(cp::kPieceExists,
                                        "The workspace has " + piece.name + " up to version " +
                                            std::to_string(saved.back()) +
                                            "; a new version is above every saved one, and saved versions never change.",
                                        "args.piece.version"));
    }
    const std::string hash = cp::piece_hash(piece);
    if (!piece.hash.empty() && explicit_version && piece.hash != hash)
        return CommandOutput::fail(fail(cp::kPieceHashMismatch,
                                        "The given hash differs from the piece's hash " + hash + ".", "args.piece.hash"));
    piece.hash = hash;
    std::string path;
    if (Error e = write_workspace_piece(workspace, piece, g.samples, &path)) return CommandOutput::fail(e);
    return CommandOutput::ok({{"name", piece.name},
                              {"version", piece.version},
                              {"hash", piece.hash},
                              {"path", path},
                              {"messages", warnings_of(g.check)}});
}

CommandOutput piece_remove(const json& args, const spec::Spec& spec, const std::string& workspace,
                           const PieceLibrary& library) {
    const std::string name = str_arg(args, "name");
    const std::optional<std::uint32_t> version = version_arg(args);
    if (!version) return CommandOutput::fail(fail(cp::kPieceBadName, "piece.remove needs a version.", "args.version"));
    if (cp::Verdict v = cp::check_piece_name(name, *version); !v.ok())
        return CommandOutput::fail(fail(v.code, v.message, "args." + v.where));
    const json users = pinned_by(spec, name, *version, spec::PieceFrom::workspace);
    if (!users.empty())
        return CommandOutput::fail(fail(cp::kPieceInUse,
                                        "The program " + users[0].get<std::string>() + " of the open spec pins " + name +
                                            " version " + std::to_string(*version) +
                                            "; pin another version with program.patch first.",
                                        "args.name"));
    Error e = remove_workspace_piece(workspace, name, *version);
    if (e.code == cp::kPieceNotFound && library.find(name, *version))
        return CommandOutput::fail(fail(cp::kPieceReadOnly,
                                        "The piece " + name + " version " + std::to_string(*version) +
                                            " is a library piece, which never changes; only workspace pieces are removed.",
                                        "args.name"));
    if (e) return CommandOutput::fail(e);
    return CommandOutput::ok({{"name", name}, {"version", *version}, {"removed", true}});
}

}  // namespace

// ---- the library

PieceLibrary::PieceLibrary(const std::vector<programs::LibraryFile>& files) : files_(files) {
    // Each pieces/<name>/<version>/piece.toml with the WGSL files beside it.
    std::map<std::string, std::map<std::string, std::string>> folders;
    for (const programs::LibraryFile& f : files_) {
        const std::string path(f.path);
        if (path.rfind("pieces/", 0) != 0) continue;
        const std::size_t cut = path.rfind('/');
        folders[path.substr(0, cut)][path.substr(cut + 1)] = std::string(f.text);
    }
    for (auto& [folder, texts] : folders) {
        auto manifest = texts.find("piece.toml");
        if (manifest == texts.end()) continue;
        spec::PieceParse parsed = spec::piece_from_toml(manifest->second);
        if (!parsed.ok()) continue;
        spec::Piece p = std::move(parsed.piece);
        if ("pieces/" + p.name + "/" + std::to_string(p.version) != folder) continue;
        for (const char* file : spec::kPieceFiles)
            if (auto it = texts.find(file); it != texts.end()) p.files[file] = it->second;
        const std::string hash = cp::piece_hash(p);
        if (!p.hash.empty() && p.hash != hash) continue;
        p.hash = hash;
        pieces_.push_back(std::move(p));
    }
    std::sort(pieces_.begin(), pieces_.end(), [](const spec::Piece& a, const spec::Piece& b) {
        return a.name != b.name ? a.name < b.name : a.version < b.version;
    });
}

const PieceLibrary& PieceLibrary::builtin() {
    static const PieceLibrary library(programs::library_files());
    return library;
}

const spec::Piece* PieceLibrary::find(std::string_view name, std::optional<std::uint32_t> version) const {
    const spec::Piece* best = nullptr;
    for (const spec::Piece& p : pieces_)
        if (p.name == name && (!version || p.version == *version)) best = &p;
    return best;
}

// ---- finding and resolving

FoundPiece find_piece(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                      spec::PieceFrom from, std::string_view name_view, std::optional<std::uint32_t> version) {
    FoundPiece f;
    f.from = from;
    const std::string name(name_view);
    if (cp::Verdict v = cp::check_piece_name(name, version.value_or(1)); !v.ok()) {
        f.error = fail(v.code, v.message, v.where);
        return f;
    }
    const std::string which = name + (version ? " version " + std::to_string(*version) : std::string());
    switch (from) {
        case spec::PieceFrom::library: {
            const spec::Piece* p = library.find(name, version);
            if (!p) {
                f.error = fail(cp::kPieceNotFound, "The library has no piece " + which + ".");
                return f;
            }
            f.piece = *p;
            f.path = "caliper://library/pieces/" + p->name + "/" + std::to_string(p->version) + "/";
            return f;
        }
        case spec::PieceFrom::workspace: {
            fs::path root;
            if (Error e = pieces_root(workspace, false, &root)) {
                f.error = e;
                return f;
            }
            std::uint32_t v = 0;
            if (version) {
                v = *version;
            } else {
                const std::vector<std::uint32_t> saved = workspace_versions(root, name);
                if (saved.empty()) {
                    f.error = fail(cp::kPieceNotFound, "The workspace has no piece " + which + ".");
                    return f;
                }
                v = saved.back();
            }
            return read_workspace_piece(root, name, v);
        }
        case spec::PieceFrom::inline_piece: {
            const spec::Piece* best = nullptr;
            for (const spec::Piece& p : spec.pieces)
                if (p.name == name && (!version || p.version == *version) && (!best || p.version > best->version))
                    best = &p;
            if (!best) {
                f.error = fail(cp::kPieceNotFound, "The open spec has no inline piece " + which + ".");
                return f;
            }
            f.piece = *best;
            const std::string hash = cp::piece_hash(*best);
            if (!best->hash.empty() && best->hash != hash)
                f.error = fail(cp::kPieceHashMismatch, "The inline piece " + name + " gives the hash " + best->hash +
                                                           ", and its text has the hash " + hash + ".");
            f.piece.hash = hash;
            return f;
        }
    }
    return f;
}

FoundPiece resolve_pin(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                       const spec::PieceRef& pin) {
    FoundPiece f = find_piece(library, workspace, spec, pin.from, pin.name, pin.version);
    const std::string which = pin.name + " version " + std::to_string(pin.version);
    if (f.error.code == cp::kPieceNotFound) {
        f.error = fail(cp::kPieceMissing,
                       "The program pins the piece " + which + " from " + spec::piece_from_name(pin.from) +
                           ", which that source lacks; restore it, or pin an existing version with program.patch.",
                       "piece");
        return f;
    }
    if (f.error) return f;
    if (!pin.hash.empty() && pin.hash != f.piece.hash)
        f.error = fail(cp::kPieceHashMismatch,
                       "The program pins " + which + " with the hash " + pin.hash + ", and the piece has the hash " +
                           f.piece.hash + "; restore the folder, or pin the piece again with program.patch.",
                       "piece.hash");
    return f;
}

Error check_instance(const PieceLibrary& library, const std::string& workspace, const spec::Spec& spec,
                     const spec::Program& program, const spec::Piece& piece, const spec::SourceResolver& resolver) {
    for (const auto& [name, target] : program.bind) {
        const spec::PieceInput* input = nullptr;
        for (const spec::PieceInput& in : piece.inputs)
            if (in.name == name) input = &in;
        if (!input)
            return fail(cp::kPieceBadBinding, "The piece " + piece.name + " has no input " + quote(name) + ".",
                        "bind." + name);
    }
    for (const spec::PieceInput& in : piece.inputs) {
        const std::string where = "bind." + in.name;
        auto it = program.bind.find(in.name);
        if (it == program.bind.end())
            return fail(cp::kPieceBadBinding,
                        "The input " + quote(in.name) + " of the piece " + piece.name + " is unbound; bind it to " +
                            (in.kind == spec::InputKind::tensor ? "live:<tensor>." : "program:<id>/<resource>."),
                        where);
        const std::string& target = it->second;
        if (in.kind == spec::InputKind::tensor) {
            if (target.rfind("live:", 0) != 0)
                return fail(cp::kPieceBadBinding, "The tensor input " + quote(in.name) + " binds live:<tensor>.", where);
            if (!resolver) continue;
            const std::optional<spec::ResolvedSource> r = resolver(target);
            if (!r || r->dtype.empty() || !r->shape) continue;
            const cp::Verdict v =
                cp::check_tensor_binding(in, r->dtype, static_cast<std::uint32_t>(r->shape->size()));
            if (!v.ok()) return fail(v.code, v.message, where);
            continue;
        }
        const std::size_t slash = target.find('/');
        if (target.rfind("program:", 0) != 0 || slash == std::string::npos)
            return fail(cp::kPieceBadBinding, "The buffer input " + quote(in.name) + " binds program:<id>/<resource>.",
                        where);
        const std::string other_id = target.substr(8, slash - 8);
        const std::string resource = target.substr(slash + 1);
        const spec::Program* other = spec.find_program(other_id);
        if (!other || !other->piece)
            return fail(cp::kPieceBadBinding, "The binding names " + quote(other_id) +
                                                  ", which is no program of a piece in the open spec.",
                        where);
        const FoundPiece op = resolve_pin(library, workspace, spec, *other->piece);
        if (op.error)
            return fail(cp::kPieceBadBinding, "The program " + other_id + " cannot be resolved: " + op.error.message,
                        where);
        const spec::Resource* published = nullptr;
        for (const spec::Resource& r : op.piece.resources)
            if (r.id == resource && r.publish && r.kind == spec::ResourceKind::buffer) published = &r;
        if (!published)
            return fail(cp::kPieceBadBinding,
                        "The piece of " + other_id + " publishes no buffer " + quote(resource) + ".", where);
        if (cp::Verdict v = cp::check_buffer_binding(in, published->layout); !v.ok())
            return fail(v.code, v.message, where);
    }
    for (const auto& kv : program.params) {
        bool known = false;
        for (const spec::PieceParam& p : piece.params) known = known || p.name == kv.first;
        if (!known)
            return fail(cp::kProgramParamsMismatch,
                        "The piece " + piece.name + " has no parameter " + quote(kv.first) + ".", "params." + kv.first);
    }
    if (piece.draw && !program.panel)
        return fail("spec.missing_key", "The piece " + piece.name + " draws, so its program names a scene panel.",
                    "panel");
    return {};
}

Error complete_instance(const PieceLibrary& library, const std::string& workspace, spec::Spec& after,
                        const std::string& id, const spec::SourceResolver& resolver, json* args) {
    spec::Program* program = nullptr;
    for (spec::Program& p : after.programs)
        if (p.id == id) program = &p;
    if (!program || !program->piece) return {};
    const FoundPiece found = resolve_pin(library, workspace, after, *program->piece);
    if (found.error) return fail(found.error.code, found.error.message, "args." + found.error.where);
    if (program->piece->hash.empty()) {
        program->piece->hash = found.piece.hash;
        if (args && args->is_object()) {
            if (args->contains("patch") && (*args)["patch"].is_object()) {
                (*args)["patch"]["piece"] = spec::to_json(*program->piece);
            } else if (args->contains("piece") && (*args)["piece"].is_object()) {
                (*args)["piece"]["hash"] = found.piece.hash;
            } else if (args->contains("inline")) {
                (*args)["piece"] = spec::to_json(*program->piece);
            }
        }
    }
    if (Error e = check_instance(library, workspace, after, *program, found.piece, resolver))
        return fail(e.code, e.message, "args." + e.where);
    return {};
}

json check_piece_json(const json& piece_json, bool headers, spec::Piece* out) {
    json messages = json::array();
    json passes = json::array();
    json result = {{"ok", false}, {"hash", nullptr}, {"messages", json::array()}, {"passes", json::array()},
                   {"headers", {{"derive", ""}, {"draw", ""}}}};
    spec::PieceParse parsed = spec::piece_from_json_value(piece_json, "");
    for (const spec::Diagnostic& d : parsed.diagnostics)
        messages.push_back(message(d.is_error() ? "error" : "warning", d.code, d.message, "piece.toml", 0, 0, d.path));
    if (!parsed.ok()) {
        result["messages"] = std::move(messages);
        return result;
    }
    spec::Piece piece = std::move(parsed.piece);
    // piece.check and piece.update take the name alone: a missing version is 1.
    const cp::Verdict manifest = cp::check_piece_manifest(piece);
    if (!manifest.ok()) messages.push_back(verdict_message(manifest, "piece.toml"));
    for (const auto& [file, text] : piece.files) {
        const cp::Verdict v = cp::check_piece_source(text, spec::piece_limits::kFileBytes);
        if (!v.ok()) messages.push_back(verdict_message(v, file));
    }
    const std::string hash = cp::piece_hash(piece);
    result["hash"] = hash;
    if (!piece.hash.empty() && piece.hash != hash)
        messages.push_back(message("error", cp::kPieceHashMismatch,
                                   "The given hash differs from the piece's hash " + hash + ".", "piece.toml", 0, 0,
                                   "hash"));
    if (!has_errors(messages)) {
        const cp::Resolved resolved = cp::resolve_imports(piece);
        if (!resolved.verdict.ok()) {
            messages.push_back(verdict_message(resolved.verdict, resolved.verdict.line ? resolved.verdict.where : "piece.toml"));
        } else {
            std::string derive_header, draw_header;
            if (piece.derive) check_half(piece, resolved, cp::Half::derive, messages, passes, &derive_header);
            if (piece.draw) check_half(piece, resolved, cp::Half::draw, messages, passes, &draw_header);
            if (headers) result["headers"] = {{"derive", derive_header}, {"draw", draw_header}};
        }
    }
    result["ok"] = !has_errors(messages);
    result["messages"] = std::move(messages);
    result["passes"] = std::move(passes);
    if (out) {
        piece.hash = hash;
        *out = std::move(piece);
    }
    return result;
}

PieceSave save_inline_piece(const std::string& workspace, const spec::Spec& spec, const json& args, bool dry) {
    PieceSave out;
    const std::string name = str_arg(args, "name");
    const std::optional<std::uint32_t> version = version_arg(args);
    const FoundPiece found =
        find_piece(PieceLibrary::builtin(), workspace, spec, spec::PieceFrom::inline_piece, name, version);
    if (found.error) {
        out.error = fail(found.error.code, found.error.message, "args.name");
        return out;
    }
    spec::Piece piece = found.piece;
    const json check = check_piece_json(spec::to_json(piece), false, nullptr);
    for (const json& m : check["messages"]) {
        if (m.value("severity", "") != "error") continue;
        out.error = fail(m.value("code", "piece.bad_manifest"), m.value("message", ""), "args.name");
        return out;
    }
    fs::path root;
    if (Error e = pieces_root(workspace, false, &root)) {
        out.error = e;
        return out;
    }
    const std::vector<std::uint32_t> saved = workspace_versions(root, piece.name);
    if (std::find(saved.begin(), saved.end(), piece.version) != saved.end()) {
        out.error = fail(cp::kPieceExists,
                         "The workspace already has " + piece.name + " version " + std::to_string(piece.version) +
                             "; give the inline piece another version first.",
                         "args.name");
        return out;
    }
    std::string path = workspace_path(piece.name, piece.version);
    if (!dry) {
        if (Error e = write_workspace_piece(workspace, piece, {}, &path)) {
            out.error = e;
            return out;
        }
    }
    out.after = spec;
    auto& pieces = out.after.pieces;
    pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                                [&](const spec::Piece& p) { return p.name == piece.name && p.version == piece.version; }),
                 pieces.end());
    json programs = json::array();
    for (spec::Program& p : out.after.programs) {
        if (!p.piece || p.piece->from != spec::PieceFrom::inline_piece || p.piece->name != piece.name ||
            p.piece->version != piece.version)
            continue;
        p.piece->from = spec::PieceFrom::workspace;
        p.piece->hash = piece.hash;
        programs.push_back(p.id);
    }
    out.result = {{"name", piece.name},   {"version", piece.version}, {"hash", piece.hash},
                  {"path", path},         {"programs", programs},     {"messages", warnings_of(check)},
                  {"applied", !dry}};
    return out;
}

CommandOutput piece_command(const CommandContext& context, const spec::Spec& spec, const std::string& workspace) {
    return piece_command(context, spec, workspace, PieceLibrary::builtin());
}

CommandOutput piece_command(const CommandContext& context, const spec::Spec& spec, const std::string& workspace,
                            const PieceLibrary& library) {
    const std::string& name = context.invocation.name;
    const json& args = context.invocation.args;
    try {
        if (name == "piece.check") return piece_check(args);
        if (name == "piece.list") return piece_list(args, spec, workspace, library);
        if (name == "piece.show") return piece_show(args, spec, workspace, library);
        if (name == "piece.copy") return piece_copy(args, workspace, library);
        if (name == "piece.update") return piece_update(args, workspace);
        if (name == "piece.remove") return piece_remove(args, spec, workspace, library);
    } catch (const std::exception& e) {
        return CommandOutput::fail(fail("piece.internal", std::string("The piece command failed: ") + e.what()));
    }
    return CommandOutput::fail(fail("command.unknown", name + " is not a piece command."));
}

}  // namespace caliper_builder
