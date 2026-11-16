#pragma once
// caliper_builder::agent::KeyStore: where Caliper keeps the API keys of the
// widget agent's providers (bring your own key), one key per provider. A key
// lives in the operating system's credential store and nowhere else: never in
// a file, a spec, the journal, a log, an Error or a command line.
//
// The interface sets a key, says whether one is set, and removes it. It has no
// function that returns a key, for display or otherwise. The one way a key
// leaves a store is place_key_in_environment(), which copies it into the
// environment of a child process about to be spawned (the caliper-agent
// sidecar) under the variable the provider's SDK reads, and removes from that
// environment the variables that could make the child use another credential,
// another endpoint or extra headers.
//
// A provider is named by an id from the presets (caliper_agent/presets.json,
// embedded here at build time and read by the sidecar too): "anthropic" (the
// Claude preset), "openai", "gemini" and "openrouter" take a key; "ollama" and
// "lmstudio" are local servers that take none, so set() refuses them and
// place_key_in_environment() places nothing for them; a Custom entry is named
// by custom_key_account() of its base URL, and its key is optional. The id is
// the account of the Keychain item, so each provider has an item of its own. A
// key must be 1 to 1024 bytes of printable ASCII without spaces, the rule the
// sidecar applies too. Errors never quote a key.
//
// Implementations:
//   MemoryKeyStore           in memory, for tests; it overwrites what it held
//                            when a key is replaced or removed and when it is
//                            destroyed;
//   make_system_key_store()  the macOS Keychain: a generic password item with
//                            the service "caliper.agent-keys" and the provider
//                            id as the account, readable only while the device
//                            is unlocked and never synchronised to other
//                            devices. On Windows (Credential Manager is
//                            planned) and other systems, a store whose every
//                            call fails with keystore.unsupported.
//
// The offline switch of the tests, CALIPER_AGENT_OFFLINE: with it the sidecar
// reaches no model server off loopback (caliper_agent/offline.py). Every test
// and tool of src/builder/ that starts the sidecar sets it. When this process
// has it, place_key_in_environment() passes it to the child with this
// process's value, even if `spawn` removes or replaces it, so code under test
// cannot strip it; the sidecar's test switch, CALIPER_AGENT_TEST_SCRIPT, is
// removed as before.
//
// No automated test calls the system store, because touching the login
// keychain can show a system prompt; it is compiled on macOS and checked by the
// manual check in design/byok-agent.md.
//
// Error codes: keystore.bad_provider, keystore.bad_key, keystore.not_found,
// keystore.no_key_needed, keystore.unsupported, keystore.failed.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/error.h>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_builder::agent {

constexpr const char* kAnthropicProvider = "anthropic";
constexpr const char* kKeychainService = "caliper.agent-keys";
constexpr std::size_t kMaxKeyBytes = 1024;
constexpr const char* kOfflineVariable = "CALIPER_AGENT_OFFLINE";

// Whether a provider takes a key.
enum class KeyNeed { required, optional, none };

// One preset of caliper_agent/presets.json.
struct ProviderPreset {
    std::string id;            // "anthropic", "openai", ...; also the key's account
    std::string name;          // "Claude", "OpenAI", ...
    std::string api;           // "claude" or "openai-compatible"
    std::string base_url;      // as the design page gives it; empty for Custom
    KeyNeed key = KeyNeed::required;
    std::string key_variable;  // where the key reaches the sidecar; empty when it takes none
    bool local = false;        // a server on this machine
    // The models the preset offers without asking its server (the Claude
    // preset's four), as {id, name}; empty for a preset whose server lists
    // its own. And the efforts it takes, empty when it takes none.
    std::vector<std::pair<std::string, std::string>> models;
    std::vector<std::string> efforts;
    // The context length the preset loads the model with for the widget agent
    // (Ollama's num_ctx; presets.json's context_length.widget), unless the
    // person sets another; 0 leaves it to the server.
    long long context_length = 0;
};

// The presets, in the table's order (Claude, OpenAI, Gemini, OpenRouter,
// Ollama, LM Studio, Custom).
const std::vector<ProviderPreset>& provider_presets();
const ProviderPreset* find_preset(const std::string& id);

// The id of a Custom entry's key: "custom-" and the first 12 hexadecimal
// digits of the SHA-256 of its base URL without trailing slashes, as the
// sidecar computes it (caliper_agent/presets.py).
std::string custom_key_account(const std::string& base_url);

// Whether a provider id is one the store knows, and whether it takes a key.
bool is_provider(const std::string& provider);
KeyNeed key_need(const std::string& provider);

// The variable a provider's key is placed in: ANTHROPIC_API_KEY for Claude,
// CALIPER_MODEL_API_KEY for an OpenAI-compatible provider that takes a key;
// nullptr for a local server or an unknown provider.
const char* key_variable(const std::string& provider);

// Whether a key is set. `error` is set when the store could not tell.
struct KeyPresence {
    Error error;
    bool present = false;
};

class KeyStore;

// Copies the provider's key into `spawn->env` (replacing an entry of the same
// name), removes from `spawn->env` and adds to `spawn->env_remove` every
// variable that could redirect the child and every other provider's key
// variable, and leaves `spawn` unchanged on any error. For a local server, and
// for a Custom entry without a key, it places no key and removes all of them.
// An AcpClient takes the result as `agent_env` and `agent_env_remove`.
Error place_key_in_environment(KeyStore& store, const std::string& provider, acp::SpawnOptions* spawn);

// Places no key: removes from `spawn->env` and adds to `spawn->env_remove`
// every variable that could redirect the child and every provider's key
// variable, as place_key_in_environment() does for a local server, and passes
// the offline switch. For a child that must never see a key the app's own
// environment holds, when no store is asked for one.
void strip_key_environment(acp::SpawnOptions* spawn);

// When this process has CALIPER_AGENT_OFFLINE, sets it in `spawn->env` with
// the same value and takes it out of `spawn->env_remove`; otherwise does
// nothing. place_key_in_environment() calls it.
void pass_offline_switch(acp::SpawnOptions* spawn);

class KeyStore {
public:
    virtual ~KeyStore() = default;

    Error set(const std::string& provider, const std::string& key);
    KeyPresence has(const std::string& provider);
    Error remove(const std::string& provider);   // removing an absent key succeeds

protected:
    virtual Error do_set(const std::string& provider, const std::string& key) = 0;
    virtual KeyPresence do_has(const std::string& provider) = 0;
    virtual Error do_remove(const std::string& provider) = 0;
    // Reads the key into `out`; keystore.not_found when there is none. Only
    // place_key_in_environment() calls it.
    virtual Error do_read(const std::string& provider, std::string* out) = 0;

    friend Error place_key_in_environment(KeyStore& store, const std::string& provider, acp::SpawnOptions* spawn);
};

class MemoryKeyStore final : public KeyStore {
public:
    MemoryKeyStore() = default;
    ~MemoryKeyStore() override;
    MemoryKeyStore(const MemoryKeyStore&) = delete;
    MemoryKeyStore& operator=(const MemoryKeyStore&) = delete;

protected:
    Error do_set(const std::string& provider, const std::string& key) override;
    KeyPresence do_has(const std::string& provider) override;
    Error do_remove(const std::string& provider) override;
    Error do_read(const std::string& provider, std::string* out) override;

private:
    std::map<std::string, std::string> keys_;
};

// The operating system's credential store (see above), one item per provider.
// Never null.
std::unique_ptr<KeyStore> make_system_key_store();

// Overwrites a string's bytes before clearing it. Best effort: copies the
// standard library made earlier are out of its reach.
void wipe(std::string* s);

}  // namespace caliper_builder::agent
