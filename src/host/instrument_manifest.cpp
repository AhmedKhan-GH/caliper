#include "instrument_manifest.h"
#include <toml++/toml.hpp>
#include <cctype>
#include <cstdint>
#include <sstream>

namespace caliper_host {

bool is_valid_semver(const std::string& v) {
    int part = 0, digits = 0;
    for (char c : v) {
        if (std::isdigit((unsigned char)c)) { digits++; continue; }
        if (c == '.') {
            if (digits == 0) return false;
            part++; digits = 0; continue;
        }
        return false;
    }
    return part == 2 && digits > 0;
}

bool is_valid_instrument_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    int seg = 0;   // chars in the current dot-separated segment
    for (char c : id) {
        if (c == '.') {
            if (seg == 0) return false;   // leading dot, "..", or "a..b"
            seg = 0;
            continue;
        }
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '-';
        if (!ok) return false;            // uppercase, '_', '/', '\\', ...
        seg++;
    }
    return seg > 0;                       // no trailing dot
}

bool is_plain_file_name(const std::string& name) {
    if (name.empty() || name.size() > 255 || name == "." || name == "..") return false;
    for (unsigned char c : name)
        if (c == '/' || c == '\\' || c == ':' || c < 0x20 || c == 0x7F) return false;
    return true;
}

const char* instrument_kind_word(InstrumentKind kind) {
    return kind == InstrumentKind::Spec ? "spec" : "library";
}

namespace {
ManifestResult fail(std::string msg) {
    ManifestResult r; r.error = std::move(msg); return r;
}
// A present key whose value must be a non-empty string.
bool non_empty_string(toml::node_view<toml::node> v, std::string& dst) {
    if (!v.is_string()) return false;
    dst = *v.value<std::string>();
    return !dst.empty();
}
std::vector<std::string> read_array(const toml::table& t, const char* key) {
    std::vector<std::string> out;
    if (auto* arr = t[key].as_array())
        for (auto& e : *arr)
            if (auto s = e.value<std::string>()) out.push_back(*s);
    return out;
}
// The rest of a kind = "spec" instrument manifest, after [instrument]'s id, name,
// version, summary and tag (docs/instruments/manifest.md): instrument.spec
// is required and is a plain file name; [compat].min_host keeps its meaning;
// the keys that only a library can honour are errors rather than ignored, so
// that a manifest meant for a library is never taken for a kind = "spec" instrument.
template <class Fail>
ManifestResult parse_spec_kind(const toml::table& root, const toml::table& instrument,
                               ManifestResult& r, const Fail& fail) {
    auto spec = instrument["spec"];
    if (!spec) return fail("manifest missing instrument.spec, the spec file of a kind = \"spec\" instrument");
    const auto file = spec.value<std::string>();
    if (!spec.is_string() || !file || !is_plain_file_name(*file))
        return fail("instrument.spec is not a plain file name in the instrument's folder: " +
                    (file ? *file : std::string("(not a string)")));
    r.manifest.spec = *file;

    if (auto compat_node = root["compat"]) {
        const toml::table* compat = compat_node.as_table();
        if (!compat) return fail("[compat] is not a table");
        if (compat->contains("abi_epoch"))
            return fail("compat.abi_epoch does not apply to a kind = \"spec\" instrument, which has no library");
        if (compat->contains("torch"))
            return fail("compat.torch does not apply to a kind = \"spec\" instrument, which has no library");
        if (auto mh = (*compat)["min_host"].value<std::string>()) {   // as for a library
            if (!is_valid_semver(*mh))
                return fail("compat.min_host is not strict semver x.y.z: " + *mh);
            r.manifest.min_host = *mh;
        }
    }
    if (root.contains("build"))
        return fail("[build] does not apply to a kind = \"spec\" instrument, which has no library");
    if (root.contains("services"))
        return fail("[services] does not apply to a kind = \"spec\" instrument, which calls no service");
    if (root.contains("data"))
        return fail("[data] does not apply to a kind = \"spec\" instrument, which keeps no data of its own");
    r.ok = true;
    return r;
}
} // namespace

ManifestResult parse_manifest_text(const std::string& toml_text) {
    toml::table root;
    try {
        root = toml::parse(toml_text);
    } catch (const toml::parse_error& e) {
        return fail(std::string("manifest parse error: ") + e.what());
    }

    ManifestResult r;
    auto* instrument = root["instrument"].as_table();
    if (!instrument) return fail("manifest missing [instrument] table");

    // The kind first: every refusal below keeps it, so that the loader lists
    // a broken kind = "spec" instrument with its declared kind (ManifestResult).
    std::string kind_error;
    if (auto kind = (*instrument)["kind"]) {
        const auto word = kind.value<std::string>();
        if (!kind.is_string() || !word)
            kind_error = "instrument.kind is not a string";
        else if (*word == "spec")
            r.manifest.kind = InstrumentKind::Spec;
        else if (*word != "library")
            kind_error = "instrument.kind is not \"library\" or \"spec\": " + *word;
    }
    // From here on `fail` is this lambda, which keeps the declared kind.
    const InstrumentKind declared = r.manifest.kind;
    auto fail = [declared](std::string msg) {
        ManifestResult f;
        f.error = std::move(msg);
        f.manifest.kind = declared;
        return f;
    };

    auto req_str = [&](const char* key, std::string& dst) -> bool {
        if (auto v = (*instrument)[key].value<std::string>()) { dst = *v; return true; }
        return false;
    };
    if (!req_str("id", r.manifest.id) || r.manifest.id.empty())
        return fail("manifest missing instrument.id");
    if (!is_valid_instrument_id(r.manifest.id))
        return fail("instrument.id is not a valid instrument id (lowercase reverse-DNS, "
                    "[a-z0-9.-]): " + r.manifest.id);
    if (!req_str("name", r.manifest.name) || r.manifest.name.empty())
        return fail("manifest missing instrument.name");
    if (!req_str("version", r.manifest.version))
        return fail("manifest missing instrument.version");
    if (!is_valid_semver(r.manifest.version))
        return fail("instrument.version is not strict semver x.y.z: " + r.manifest.version);
    req_str("summary", r.manifest.summary);
    req_str("tag", r.manifest.tag);
    if (!kind_error.empty()) return fail(kind_error);
    if (r.manifest.kind == InstrumentKind::Spec) return parse_spec_kind(root, *instrument, r, fail);

    auto* compat = root["compat"].as_table();
    if (!compat) return fail("manifest missing [compat].abi_epoch");
    if (auto e = (*compat)["abi_epoch"].value<int64_t>(); e && *e >= 1)
        r.manifest.abi_epoch = (uint32_t)*e;
    else
        return fail("manifest missing or invalid [compat].abi_epoch (integer >= 1)");
    if (auto mh = (*compat)["min_host"].value<std::string>()) {
        if (!is_valid_semver(*mh))
            return fail("compat.min_host is not strict semver x.y.z: " + *mh);
        r.manifest.min_host = *mh;
    }
    if (auto torch = (*compat)["torch"]) {
        if (!non_empty_string(torch, r.manifest.torch))
            return fail("compat.torch is not a non-empty string");
    }

    if (auto* services = root["services"].as_table()) {
        r.manifest.required_services = read_array(*services, "required");
        r.manifest.optional_services = read_array(*services, "optional");
    }

    // [build] is written by the build, never by hand. sdk is a record for
    // diagnosis, so a value of the wrong type is ignored rather than refused;
    // ui_stack gates loading, so it must be well formed when present.
    if (auto* build = root["build"].as_table()) {
        if (auto sdk = (*build)["sdk"]; sdk.is_string())
            r.manifest.sdk = *sdk.value<std::string>();
        if (auto ui = (*build)["ui_stack"]) {
            if (!non_empty_string(ui, r.manifest.ui_stack))
                return fail("build.ui_stack is not a non-empty string");
        }
    }

    if (auto* data = root["data"].as_table()) {
        if (auto schema = (*data)["schema"]) {
            const auto* n = schema.as_integer();
            if (!n || n->get() < 1 || n->get() > INT32_MAX)
                return fail("data.schema is not an integer >= 1");
            r.manifest.data_schema = (uint32_t)n->get();
        }
    }

    r.ok = true;
    return r;
}

ManifestResult parse_manifest_file(const std::string& path) {
    toml::table root;
    try {
        root = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        return fail(std::string("manifest parse error: ") + e.what());
    } catch (...) {
        return fail("manifest unreadable: " + path);
    }
    std::ostringstream oss; oss << toml::toml_formatter(root);
    return parse_manifest_text(oss.str());
}

} // namespace caliper_host
