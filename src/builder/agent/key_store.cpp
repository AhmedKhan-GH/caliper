// The KeyStore front (validation), the presets, the in-memory store, the
// placement of a key in a child's environment, and the system store where no
// native one exists yet.
#include <caliper_builder/agent/key_store.h>
#include <caliper_builder/commands/sha256.h>
#include <caliper_builder/presets_json.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <vector>

namespace caliper_builder::agent {

namespace {

// Variables that could make the sidecar's SDK use another credential, send its
// requests elsewhere, add headers or log requests, or replace the Claude API
// with a script. The sidecar removes the SDK's own too; removing them at spawn
// means they never reach it.
const char* const kOverrideVariables[] = {
    "ANTHROPIC_AUTH_TOKEN",          "ANTHROPIC_BASE_URL",          "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_PROFILE",             "ANTHROPIC_CONFIG_DIR",        "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",  "ANTHROPIC_WORKSPACE_ID",      "ANTHROPIC_LOG",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY",
    // The OpenAI SDK, which the sidecar uses for OpenAI-compatible servers, and
    // Pydantic AI's gateway and telemetry; the sidecar removes them too.
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_WEBHOOK_SECRET", "OPENAI_LOG",
    "PYDANTIC_AI_GATEWAY_API_KEY", "LOGFIRE_TOKEN",
    // The sidecar's test switch, which replaces the Claude API with a scripted
    // file; the app never passes it on.
    "CALIPER_AGENT_TEST_SCRIPT",
};

// The variables a key reaches the sidecar in; each spawn keeps only the one of
// the provider chosen.
const char* const kKeyVariables[] = {"ANTHROPIC_API_KEY", "CALIPER_MODEL_API_KEY"};
constexpr const char* kCustomPrefix = "custom-";
constexpr const char* kCustomKeyVariable = "CALIPER_MODEL_API_KEY";

bool is_custom_account(const std::string& provider) {
    static const std::regex kCustom("^custom-[0-9a-f]{12}$");
    return std::regex_match(provider, kCustom);
}

Error check_provider(const std::string& provider) {
    static const std::regex kProvider("^[a-z][a-z0-9_-]{0,31}$");
    if (provider.size() > 32 || !std::regex_match(provider, kProvider))
        return {"keystore.bad_provider", "A provider id is 1 to 32 characters of a-z, 0-9, _ and -, starting with a letter."};
    if (!is_provider(provider))
        return {"keystore.bad_provider", "Caliper knows no provider called " + provider + "."};
    return {};
}

KeyNeed parse_need(const std::string& s) {
    if (s == "none") return KeyNeed::none;
    if (s == "optional") return KeyNeed::optional;
    return KeyNeed::required;
}

Error check_key(const std::string& key) {
    if (key.empty()) return {"keystore.bad_key", "The key is empty."};
    if (key.size() > kMaxKeyBytes) return {"keystore.bad_key", "The key is longer than 1024 bytes."};
    for (unsigned char c : key)
        if (c < 0x21 || c > 0x7E)
            return {"keystore.bad_key", "The key holds a space, a control character or a non-ASCII character."};
    return {};
}

#if !defined(__APPLE__)
class UnsupportedKeyStore final : public KeyStore {
protected:
    static Error unsupported() {
#ifdef _WIN32
        return {"keystore.unsupported", "Keeping a key in the Windows Credential Manager is planned but not built yet."};
#else
        return {"keystore.unsupported", "This system has no credential store that Caliper supports yet."};
#endif
    }
    Error do_set(const std::string&, const std::string&) override { return unsupported(); }
    KeyPresence do_has(const std::string&) override { return {unsupported(), false}; }
    Error do_remove(const std::string&) override { return unsupported(); }
    Error do_read(const std::string&, std::string*) override { return unsupported(); }
};
#endif

}  // namespace

void wipe(std::string* s) {
    if (!s) return;
    volatile char* p = s->empty() ? nullptr : &(*s)[0];
    for (std::size_t i = 0; i < s->size(); ++i) p[i] = 0;
    s->clear();
    s->shrink_to_fit();
}

const std::vector<ProviderPreset>& provider_presets() {
    static const std::vector<ProviderPreset> presets = [] {
        std::vector<ProviderPreset> out;
        const nlohmann::json doc = nlohmann::json::parse(detail::kPresetsJson, nullptr, false);
        if (!doc.is_object() || !doc.contains("presets") || !doc["presets"].is_array()) return out;
        for (const auto& p : doc["presets"]) {
            if (!p.is_object()) continue;
            ProviderPreset preset;
            preset.id = p.value("id", "");
            preset.name = p.value("name", "");
            preset.api = p.value("api", "");
            if (p.contains("base_url") && p["base_url"].is_string()) preset.base_url = p["base_url"].get<std::string>();
            preset.key = parse_need(p.value("key", "required"));
            if (p.contains("key_env") && p["key_env"].is_string()) preset.key_variable = p["key_env"].get<std::string>();
            preset.local = p.value("local", false);
            for (const auto& m : p.value("models", nlohmann::json::array()))
                if (m.is_object() && m.contains("id") && m["id"].is_string())
                    preset.models.emplace_back(m["id"].get<std::string>(), m.value("name", m["id"].get<std::string>()));
            for (const auto& e : p.value("efforts", nlohmann::json::array()))
                if (e.is_string()) preset.efforts.push_back(e.get<std::string>());
            if (!preset.id.empty()) out.push_back(std::move(preset));
        }
        return out;
    }();
    return presets;
}

const ProviderPreset* find_preset(const std::string& id) {
    for (const auto& p : provider_presets())
        if (p.id == id) return &p;
    return nullptr;
}

std::string custom_key_account(const std::string& base_url) {
    std::string url = base_url;
    while (!url.empty() && url.back() == '/') url.pop_back();
    return kCustomPrefix + command_sha256_hex(url).substr(0, 12);
}

bool is_provider(const std::string& provider) {
    if (is_custom_account(provider)) return true;
    // "custom" itself is the preset; its keys are named by custom_key_account().
    return provider != "custom" && find_preset(provider) != nullptr;
}

KeyNeed key_need(const std::string& provider) {
    if (is_custom_account(provider)) return KeyNeed::optional;
    const ProviderPreset* p = find_preset(provider);
    return p ? p->key : KeyNeed::none;
}

const char* key_variable(const std::string& provider) {
    if (is_custom_account(provider)) return kCustomKeyVariable;
    if (provider == "custom") return nullptr;
    const ProviderPreset* p = find_preset(provider);
    if (!p || p->key == KeyNeed::none || p->key_variable.empty()) return nullptr;
    return p->key_variable.c_str();
}

Error KeyStore::set(const std::string& provider, const std::string& key) {
    if (Error e = check_provider(provider)) return e;
    if (key_need(provider) == KeyNeed::none) {
        const ProviderPreset* p = find_preset(provider);
        return {"keystore.no_key_needed", "The " + (p ? p->name : provider) + " preset is a server on this machine and needs no key."};
    }
    if (Error e = check_key(key)) return e;
    return do_set(provider, key);
}

KeyPresence KeyStore::has(const std::string& provider) {
    if (Error e = check_provider(provider)) return {e, false};
    if (key_need(provider) == KeyNeed::none) return {{}, false};
    return do_has(provider);
}

Error KeyStore::remove(const std::string& provider) {
    if (Error e = check_provider(provider)) return e;
    if (key_need(provider) == KeyNeed::none) return {};
    return do_remove(provider);
}

Error place_key_in_environment(KeyStore& store, const std::string& provider, acp::SpawnOptions* spawn) {
    if (!spawn) return {"keystore.failed", "No spawn options were given."};
    if (Error e = check_provider(provider)) return e;
    const KeyNeed need = key_need(provider);
    std::string key;
    if (need != KeyNeed::none) {
        Error e = store.do_read(provider, &key);
        if (e) {
            wipe(&key);
            // A Custom entry may have no key: then none is placed.
            if (!(need == KeyNeed::optional && e.code == "keystore.not_found")) return e;
        } else if (Error bad = check_key(key)) {
            wipe(&key);
            return {"keystore.bad_key", "The stored key is not well formed; enter it again."};
        }
    }
    const std::string variable = key.empty() ? std::string() : std::string(key_variable(provider));

    auto stripped = [&](const std::string& name) {
        for (const char* o : kOverrideVariables)
            if (name == o) return true;
        for (const char* k : kKeyVariables)
            if (name == k) return true;
        return false;
    };
    auto& env = spawn->env;
    env.erase(std::remove_if(env.begin(), env.end(), [&](const acp::EnvVar& v) { return stripped(v.name); }),
              env.end());
    auto add_removed = [&](const char* name) {
        if (name == variable) return;
        if (std::find(spawn->env_remove.begin(), spawn->env_remove.end(), name) == spawn->env_remove.end())
            spawn->env_remove.emplace_back(name);
    };
    for (const char* o : kOverrideVariables) add_removed(o);
    for (const char* k : kKeyVariables) add_removed(k);
    if (!key.empty()) env.push_back(acp::EnvVar{variable, key});
    wipe(&key);
    pass_offline_switch(spawn);
    return {};
}

void pass_offline_switch(acp::SpawnOptions* spawn) {
    if (!spawn) return;
    const char* value = std::getenv(kOfflineVariable);
    if (!value) return;
    auto& removed = spawn->env_remove;
    removed.erase(std::remove(removed.begin(), removed.end(), std::string(kOfflineVariable)), removed.end());
    auto& env = spawn->env;
    env.erase(std::remove_if(env.begin(), env.end(), [](const acp::EnvVar& v) { return v.name == kOfflineVariable; }),
              env.end());
    env.push_back(acp::EnvVar{kOfflineVariable, value});
}

MemoryKeyStore::~MemoryKeyStore() {
    for (auto& kv : keys_) wipe(&kv.second);
}

Error MemoryKeyStore::do_set(const std::string& provider, const std::string& key) {
    auto it = keys_.find(provider);
    if (it != keys_.end()) wipe(&it->second);
    keys_[provider] = key;
    return {};
}

KeyPresence MemoryKeyStore::do_has(const std::string& provider) {
    return {{}, keys_.count(provider) != 0};
}

Error MemoryKeyStore::do_remove(const std::string& provider) {
    auto it = keys_.find(provider);
    if (it != keys_.end()) {
        wipe(&it->second);
        keys_.erase(it);
    }
    return {};
}

Error MemoryKeyStore::do_read(const std::string& provider, std::string* out) {
    auto it = keys_.find(provider);
    if (it == keys_.end()) return {"keystore.not_found", "No key is set for " + provider + "."};
    *out = it->second;
    return {};
}

#if !defined(__APPLE__)
std::unique_ptr<KeyStore> make_system_key_store() {
    return std::make_unique<UnsupportedKeyStore>();
}
#endif

}  // namespace caliper_builder::agent
