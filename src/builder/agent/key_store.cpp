// The KeyStore front (validation), the in-memory store, the placement of a key
// in a child's environment, and the system store where no native one exists yet.
#include <caliper_builder/agent/key_store.h>

#include <algorithm>
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
    // The sidecar's test switch, which replaces the Claude API with a scripted
    // file; the app never passes it on.
    "CALIPER_AGENT_TEST_SCRIPT",
};

Error check_provider(const std::string& provider) {
    static const std::regex kProvider("^[a-z][a-z0-9_-]{0,31}$");
    if (provider.size() > 32 || !std::regex_match(provider, kProvider))
        return {"keystore.bad_provider", "A provider id is 1 to 32 characters of a-z, 0-9, _ and -, starting with a letter."};
    if (key_variable(provider) == nullptr)
        return {"keystore.bad_provider", "Caliper knows no provider called " + provider + "."};
    return {};
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

const char* key_variable(const std::string& provider) {
    if (provider == kAnthropicProvider) return "ANTHROPIC_API_KEY";
    return nullptr;
}

Error KeyStore::set(const std::string& provider, const std::string& key) {
    if (Error e = check_provider(provider)) return e;
    if (Error e = check_key(key)) return e;
    return do_set(provider, key);
}

KeyPresence KeyStore::has(const std::string& provider) {
    if (Error e = check_provider(provider)) return {e, false};
    return do_has(provider);
}

Error KeyStore::remove(const std::string& provider) {
    if (Error e = check_provider(provider)) return e;
    return do_remove(provider);
}

Error place_key_in_environment(KeyStore& store, const std::string& provider, acp::SpawnOptions* spawn) {
    if (!spawn) return {"keystore.failed", "No spawn options were given."};
    if (Error e = check_provider(provider)) return e;
    std::string key;
    Error e = store.do_read(provider, &key);
    if (e) {
        wipe(&key);
        return e;
    }
    if (Error bad = check_key(key)) {
        wipe(&key);
        return {"keystore.bad_key", "The stored key is not well formed; enter it again."};
    }
    const std::string variable = key_variable(provider);

    auto& env = spawn->env;
    env.erase(std::remove_if(env.begin(), env.end(),
                             [&](const acp::EnvVar& v) {
                                 if (v.name == variable) return true;
                                 for (const char* o : kOverrideVariables)
                                     if (v.name == o) return true;
                                 return false;
                             }),
              env.end());
    for (const char* o : kOverrideVariables)
        if (std::find(spawn->env_remove.begin(), spawn->env_remove.end(), o) == spawn->env_remove.end())
            spawn->env_remove.emplace_back(o);
    env.push_back(acp::EnvVar{variable, key});
    wipe(&key);
    return {};
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
