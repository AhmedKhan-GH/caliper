#pragma once
// caliper_builder::agent::KeyStore: where Caliper keeps the API key of the agent
// pane's BYOK backend (bring your own key). The key lives in the operating
// system's credential store and nowhere else: never in a file, a spec, the
// journal, a log, an Error or a command line.
//
// The interface sets a key, says whether one is set, and removes it. It has no
// function that returns a key, for display or otherwise. The one way a key
// leaves a store is place_key_in_environment(), which copies it into the
// environment of a child process about to be spawned (the caliper-agent
// sidecar) under the variable the provider's SDK reads, and removes from that
// environment the variables that could make the child use another credential,
// another endpoint or extra headers.
//
// A provider is named by a short id; "anthropic" (the Claude API) is the only
// one now. A key must be 1 to 1024 bytes of printable ASCII without spaces, the
// rule the sidecar applies too. Errors never quote a key.
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
// No automated test calls the system store, because touching the login
// keychain can show a system prompt; it is compiled on macOS and checked by the
// manual check in design/byok-agent.md.
//
// Error codes: keystore.bad_provider, keystore.bad_key, keystore.not_found,
// keystore.unsupported, keystore.failed.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/error.h>

#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace caliper_builder::agent {

constexpr const char* kAnthropicProvider = "anthropic";
constexpr const char* kKeychainService = "caliper.agent-keys";
constexpr std::size_t kMaxKeyBytes = 1024;

// The variable a provider's key is placed in, or nullptr for an unknown provider.
const char* key_variable(const std::string& provider);

// Whether a key is set. `error` is set when the store could not tell.
struct KeyPresence {
    Error error;
    bool present = false;
};

class KeyStore;

// Copies the provider's key into `spawn->env` (replacing an entry of the same
// name), removes from `spawn->env` and adds to `spawn->env_remove` every
// variable that could redirect the child, and leaves `spawn` unchanged on any
// error. An AcpClient takes the result as `agent_env` and `agent_env_remove`.
Error place_key_in_environment(KeyStore& store, const std::string& provider, acp::SpawnOptions* spawn);

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

// The operating system's credential store (see above). Never null.
std::unique_ptr<KeyStore> make_system_key_store();

// Overwrites a string's bytes before clearing it. Best effort: copies the
// standard library made earlier are out of its reach.
void wipe(std::string* s);

}  // namespace caliper_builder::agent
