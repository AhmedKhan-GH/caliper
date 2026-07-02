#include "applet_manifest.h"
#include <toml++/toml.hpp>
#include <cctype>
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

namespace {
ManifestResult fail(std::string msg) {
    ManifestResult r; r.error = std::move(msg); return r;
}
std::vector<std::string> read_array(const toml::table& t, const char* key) {
    std::vector<std::string> out;
    if (auto* arr = t[key].as_array())
        for (auto& e : *arr)
            if (auto s = e.value<std::string>()) out.push_back(*s);
    return out;
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
    auto* applet = root["applet"].as_table();
    if (!applet) return fail("manifest missing [applet] table");

    auto req_str = [&](const char* key, std::string& dst) -> bool {
        if (auto v = (*applet)[key].value<std::string>()) { dst = *v; return true; }
        return false;
    };
    if (!req_str("id", r.manifest.id) || r.manifest.id.empty())
        return fail("manifest missing applet.id");
    if (!req_str("name", r.manifest.name) || r.manifest.name.empty())
        return fail("manifest missing applet.name");
    if (!req_str("version", r.manifest.version))
        return fail("manifest missing applet.version");
    if (!is_valid_semver(r.manifest.version))
        return fail("applet.version is not strict semver x.y.z: " + r.manifest.version);
    req_str("summary", r.manifest.summary);
    req_str("tag", r.manifest.tag);

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

    if (auto* services = root["services"].as_table()) {
        r.manifest.required_services = read_array(*services, "required");
        r.manifest.optional_services = read_array(*services, "optional");
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
