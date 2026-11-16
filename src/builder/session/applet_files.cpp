#include "applet_files.h"

#include <cstdio>
#include <exception>

#include <toml++/toml.hpp>

#include "util.h"

namespace caliper_builder::session_detail {

ManifestInfo read_manifest(const std::string& path) {
    ManifestInfo out;
    std::string text;
    if (Error e = read_file_bounded(path, kMaxManifestBytes, &text, "applet.not_spec")) {
        out.error = e.code == "session.io" ? Error("applet.not_found", "The applet's manifest could not be read.", path) : e;
        return out;
    }
    toml::table table;
#if TOML_EXCEPTIONS
    try {
        table = toml::parse(text);
    } catch (const toml::parse_error& e) {
        out.error = Error("applet.not_spec", "The manifest is not valid TOML: " + std::string(e.description()) + ".", path);
        return out;
    }
#else
    toml::parse_result parsed = toml::parse(text);
    if (!parsed) {
        out.error = Error("applet.not_spec", "The manifest is not valid TOML.", path);
        return out;
    }
    table = std::move(parsed).table();
#endif
    const toml::table* applet = table["applet"].as_table();
    if (!applet) {
        out.error = Error("applet.not_spec", "The manifest has no [applet] table.", path);
        return out;
    }
    auto text_of = [&](const char* key) { return (*applet)[key].value<std::string>().value_or(std::string()); };
    out.id = text_of("id");
    out.name = text_of("name");
    out.version = text_of("version");
    out.kind = (*applet)["kind"] ? text_of("kind") : std::string("library");
    out.spec = text_of("spec");
    return out;
}

std::string toml_quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c == 0x7F) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04X", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

std::string manifest_text(const std::string& manifest_id, const std::string& name, const std::string& version,
                          const std::optional<std::string>& summary, const std::string& spec_file) {
    std::string out = "[applet]\n";
    out += "id = " + toml_quote(manifest_id) + "\n";
    out += "name = " + toml_quote(name) + "\n";
    out += "version = " + toml_quote(version) + "\n";
    if (summary) out += "summary = " + toml_quote(*summary) + "\n";
    out += "kind = \"spec\"\n";
    out += "spec = " + toml_quote(spec_file) + "\n";
    return out;
}

std::string manifest_id_for(const std::string& workspace_id) {
    std::string out = "local.";
    for (char c : workspace_id) out += c == '_' ? '-' : c;
    return out;
}

bool plain_file_name(const std::string& name) {
    if (name.empty() || name == "." || name == ".." || name.size() > 255) return false;
    for (unsigned char c : name)
        if (c == '/' || c == '\\' || c < 0x20 || c == 0x7F || c == ':') return false;
    return true;
}

}  // namespace caliper_builder::session_detail
