// Tests of the key store: the in-memory store, the checks of the front, and
// the placement of a key in a child's environment. The system store is never
// called here (touching the login keychain could show a system prompt).
#include <caliper_builder/agent/key_store.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <string>

using namespace caliper_builder;
using namespace caliper_builder::agent;

namespace {

const std::string kKey = "sk-ant-api03-TESTONLY-key-0123456789abcdefABCDEF_-";

bool mentions(const Error& e, const std::string& secret) {
    return e.message.find(secret) != std::string::npos || e.where.find(secret) != std::string::npos ||
           e.code.find(secret) != std::string::npos;
}

const acp::EnvVar* find_var(const acp::SpawnOptions& s, const std::string& name) {
    for (const auto& v : s.env)
        if (v.name == name) return &v;
    return nullptr;
}

}  // namespace

TEST_CASE("the memory store sets, reports and removes a key") {
    MemoryKeyStore store;
    KeyPresence p = store.has(kAnthropicProvider);
    CHECK(p.error.ok());
    CHECK_FALSE(p.present);
    CHECK(store.set(kAnthropicProvider, kKey).ok());
    CHECK(store.has(kAnthropicProvider).present);
    CHECK(store.set(kAnthropicProvider, kKey + "2").ok());  // replacing is allowed
    CHECK(store.has(kAnthropicProvider).present);
    CHECK(store.remove(kAnthropicProvider).ok());
    CHECK_FALSE(store.has(kAnthropicProvider).present);
    CHECK(store.remove(kAnthropicProvider).ok());  // removing an absent key succeeds
}

TEST_CASE("provider ids are checked") {
    MemoryKeyStore store;
    const std::string bad_ids[] = {"", "Anthropic", "../x", "mistral", "custom", "custom-0123", "custom-ABCDEF012345",
                                   "a b", std::string(40, 'a'), std::string("an\0x", 4)};
    for (const auto& bad : bad_ids) {
        CAPTURE(bad);
        CHECK(store.set(bad, kKey).code == "keystore.bad_provider");
        CHECK(store.has(bad).error.code == "keystore.bad_provider");
        CHECK(store.remove(bad).code == "keystore.bad_provider");
    }
    CHECK(key_variable(kAnthropicProvider) == std::string("ANTHROPIC_API_KEY"));
    CHECK(key_variable("openai") == std::string("CALIPER_MODEL_API_KEY"));
    CHECK(key_variable("ollama") == nullptr);
    CHECK(key_variable("mistral") == nullptr);
}

TEST_CASE("the presets are the table of presets.json") {
    const auto& presets = provider_presets();
    REQUIRE(presets.size() == 7);
    struct Row { const char* id; const char* name; const char* api; const char* base_url; KeyNeed key; const char* var; bool local; };
    const Row rows[] = {
        {"anthropic", "Claude", "claude", "https://api.anthropic.com", KeyNeed::required, "ANTHROPIC_API_KEY", false},
        {"openai", "OpenAI", "openai-compatible", "https://api.openai.com/v1", KeyNeed::required, "CALIPER_MODEL_API_KEY", false},
        {"gemini", "Gemini", "openai-compatible", "https://generativelanguage.googleapis.com/v1beta/openai/",
         KeyNeed::required, "CALIPER_MODEL_API_KEY", false},
        {"openrouter", "OpenRouter", "openai-compatible", "https://openrouter.ai/api/v1", KeyNeed::required,
         "CALIPER_MODEL_API_KEY", false},
        {"ollama", "Ollama", "openai-compatible", "http://localhost:11434/v1", KeyNeed::none, "", true},
        {"lmstudio", "LM Studio", "openai-compatible", "http://localhost:1234/v1", KeyNeed::none, "", true},
        {"custom", "Custom", "openai-compatible", "", KeyNeed::optional, "CALIPER_MODEL_API_KEY", false},
    };
    for (std::size_t i = 0; i < 7; ++i) {
        CAPTURE(rows[i].id);
        CHECK(presets[i].id == rows[i].id);
        CHECK(presets[i].name == rows[i].name);
        CHECK(presets[i].api == rows[i].api);
        CHECK(presets[i].base_url == rows[i].base_url);
        CHECK(presets[i].key == rows[i].key);
        CHECK(presets[i].key_variable == rows[i].var);
        CHECK(presets[i].local == rows[i].local);
        CHECK(find_preset(rows[i].id) == &presets[i]);
    }
    CHECK(find_preset("mistral") == nullptr);
    // The Claude preset's models and efforts, which the app's settings offer.
    REQUIRE(presets[0].models.size() == 4);
    CHECK(presets[0].models[2] == std::make_pair(std::string("claude-haiku-5-5"), std::string("Claude Haiku 5.5")));
    CHECK(presets[0].efforts == std::vector<std::string>{"low", "medium", "high", "xhigh", "max"});
    for (std::size_t i = 1; i < 7; ++i) {
        CHECK(presets[i].models.empty());
        CHECK(presets[i].efforts.empty());
    }
    CHECK(key_need("ollama") == KeyNeed::none);
    CHECK(key_need("gemini") == KeyNeed::required);
}

TEST_CASE("a Custom entry's key is named by its base URL as the sidecar names it") {
    // The same value as caliper_agent.presets.custom_key_account() (tests/test_presets.py).
    CHECK(custom_key_account("https://api.mistral.ai/v1") == "custom-7c6535712768");
    CHECK(custom_key_account("https://api.mistral.ai/v1/") == "custom-7c6535712768");
    const std::string id = custom_key_account("https://api.groq.com/openai/v1");
    CHECK(is_provider(id));
    CHECK(key_need(id) == KeyNeed::optional);
    CHECK(key_variable(id) == std::string("CALIPER_MODEL_API_KEY"));
    CHECK_FALSE(is_provider("custom"));
}

TEST_CASE("the store keeps one key per provider") {
    MemoryKeyStore store;
    const std::string custom = custom_key_account("https://api.mistral.ai/v1");
    REQUIRE(store.set("anthropic", kKey).ok());
    REQUIRE(store.set("openai", kKey + "-openai").ok());
    REQUIRE(store.set(custom, kKey + "-custom").ok());
    for (const char* id : {"anthropic", "openai"}) CHECK(store.has(id).present);
    CHECK(store.has(custom).present);
    CHECK_FALSE(store.has("gemini").present);
    CHECK(store.remove("openai").ok());
    CHECK_FALSE(store.has("openai").present);
    CHECK(store.has("anthropic").present);
    CHECK(store.has(custom).present);
    // A local server takes no key.
    Error e = store.set("ollama", kKey);
    CHECK(e.code == "keystore.no_key_needed");
    CHECK_FALSE(mentions(e, kKey));
    KeyPresence p = store.has("lmstudio");
    CHECK(p.error.ok());
    CHECK_FALSE(p.present);
    CHECK(store.remove("ollama").ok());
}

TEST_CASE("each provider's key goes in its own variable, and nothing for a local server") {
    MemoryKeyStore store;
    REQUIRE(store.set("anthropic", kKey).ok());
    REQUIRE(store.set("openai", kKey + "-openai").ok());
    const std::string inherited_names[] = {"ANTHROPIC_API_KEY", "CALIPER_MODEL_API_KEY", "OPENAI_API_KEY",
                                           "OPENAI_BASE_URL", "LOGFIRE_TOKEN"};
    auto fresh = [&] {
        acp::SpawnOptions spawn;
        spawn.argv = {"caliper-agent"};
        for (const auto& name : inherited_names) spawn.env.push_back({name, "inherited-value-0123456789"});
        spawn.env.push_back({"KEEP", "1"});
        return spawn;
    };
    auto removed = [](const acp::SpawnOptions& s, const char* name) {
        return std::count(s.env_remove.begin(), s.env_remove.end(), std::string(name)) == 1;
    };

    SUBCASE("an OpenAI-compatible preset") {
        acp::SpawnOptions spawn = fresh();
        REQUIRE(place_key_in_environment(store, "openai", &spawn).ok());
        REQUIRE(find_var(spawn, "CALIPER_MODEL_API_KEY") != nullptr);
        CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY")->value == kKey + "-openai");
        CHECK(find_var(spawn, "ANTHROPIC_API_KEY") == nullptr);
        CHECK(find_var(spawn, "OPENAI_API_KEY") == nullptr);
        CHECK(find_var(spawn, "KEEP") != nullptr);
        CHECK(removed(spawn, "ANTHROPIC_API_KEY"));
        CHECK(removed(spawn, "OPENAI_API_KEY"));
        CHECK_FALSE(removed(spawn, "CALIPER_MODEL_API_KEY"));
        CHECK(spawn.argv == std::vector<std::string>{"caliper-agent"});
    }
    SUBCASE("the Claude preset") {
        acp::SpawnOptions spawn = fresh();
        REQUIRE(place_key_in_environment(store, "anthropic", &spawn).ok());
        CHECK(find_var(spawn, "ANTHROPIC_API_KEY")->value == kKey);
        CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY") == nullptr);
        CHECK(removed(spawn, "CALIPER_MODEL_API_KEY"));
    }
    SUBCASE("a local server") {
        for (const char* local : {"ollama", "lmstudio"}) {
            CAPTURE(local);
            acp::SpawnOptions spawn = fresh();
            REQUIRE(place_key_in_environment(store, local, &spawn).ok());
            CHECK(find_var(spawn, "ANTHROPIC_API_KEY") == nullptr);
            CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY") == nullptr);
            CHECK(removed(spawn, "ANTHROPIC_API_KEY"));
            CHECK(removed(spawn, "CALIPER_MODEL_API_KEY"));
            CHECK(find_var(spawn, "KEEP") != nullptr);
        }
    }
    SUBCASE("a Custom entry, with and without a key") {
        const std::string custom = custom_key_account("https://api.mistral.ai/v1");
        acp::SpawnOptions spawn = fresh();
        REQUIRE(place_key_in_environment(store, custom, &spawn).ok());
        CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY") == nullptr);
        CHECK(removed(spawn, "CALIPER_MODEL_API_KEY"));
        REQUIRE(store.set(custom, kKey + "-custom").ok());
        spawn = fresh();
        REQUIRE(place_key_in_environment(store, custom, &spawn).ok());
        CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY")->value == kKey + "-custom");
    }
    SUBCASE("a preset whose key is not set leaves the spawn unchanged") {
        acp::SpawnOptions spawn = fresh();
        const std::size_t before = spawn.env.size();
        CHECK(place_key_in_environment(store, "gemini", &spawn).code == "keystore.not_found");
        CHECK(spawn.env.size() == before);
        CHECK(spawn.env_remove.empty());
    }
}

TEST_CASE("malformed keys are refused without being quoted") {
    MemoryKeyStore store;
    const std::string bad_keys[] = {
        "",
        std::string(kMaxKeyBytes + 1, 'k'),
        "sk-ant with-space-0123456789",
        "sk-ant-line\nbreak-0123456789",
        std::string("sk-ant-nul\0-0123456789", 22),
        "sk-ant-\xc3\xa9-0123456789",
        "sk-ant-tab\t0123456789",
    };
    for (const auto& bad : bad_keys) {
        Error e = store.set(kAnthropicProvider, bad);
        CHECK(e.code == "keystore.bad_key");
        if (bad.size() > 8) {
            CHECK_FALSE(mentions(e, bad));
        }
    }
    CHECK_FALSE(store.has(kAnthropicProvider).present);
    CHECK(store.set(kAnthropicProvider, std::string(kMaxKeyBytes, 'k')).ok());
}

TEST_CASE("a key is placed in a child's environment and nothing that could redirect it") {
    MemoryKeyStore store;
    acp::SpawnOptions spawn;
    spawn.argv = {"caliper-agent"};
    spawn.env = {{"KEEP", "1"},
                 {"ANTHROPIC_API_KEY", "an-older-value-0123456789"},
                 {"ANTHROPIC_BASE_URL", "http://elsewhere.example"},
                 {"ANTHROPIC_AUTH_TOKEN", "token-0123456789"}};
    spawn.env_remove = {"ANTHROPIC_LOG"};

    SUBCASE("without a key nothing changes") {
        const acp::SpawnOptions before = spawn;
        Error e = place_key_in_environment(store, kAnthropicProvider, &spawn);
        CHECK(e.code == "keystore.not_found");
        CHECK(spawn.env.size() == before.env.size());
        CHECK(spawn.env_remove == before.env_remove);
    }
    SUBCASE("with a key") {
        REQUIRE(store.set(kAnthropicProvider, kKey).ok());
        REQUIRE(place_key_in_environment(store, kAnthropicProvider, &spawn).ok());
        int count = 0;
        for (const auto& v : spawn.env) count += v.name == "ANTHROPIC_API_KEY";
        CHECK(count == 1);
        REQUIRE(find_var(spawn, "ANTHROPIC_API_KEY") != nullptr);
        CHECK(find_var(spawn, "ANTHROPIC_API_KEY")->value == kKey);
        CHECK(find_var(spawn, "KEEP") != nullptr);
        CHECK(find_var(spawn, "ANTHROPIC_BASE_URL") == nullptr);
        CHECK(find_var(spawn, "ANTHROPIC_AUTH_TOKEN") == nullptr);
        for (const char* name : {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS",
                                 "ANTHROPIC_PROFILE", "ANTHROPIC_LOG", "CALIPER_AGENT_TEST_SCRIPT"}) {
            CAPTURE(name);
            CHECK(std::count(spawn.env_remove.begin(), spawn.env_remove.end(), std::string(name)) == 1);
        }
        CHECK(std::count(spawn.env_remove.begin(), spawn.env_remove.end(), std::string("ANTHROPIC_API_KEY")) == 0);
        CHECK(spawn.argv == std::vector<std::string>{"caliper-agent"});  // never in argv
        // Placing it again keeps one entry.
        REQUIRE(place_key_in_environment(store, kAnthropicProvider, &spawn).ok());
        count = 0;
        for (const auto& v : spawn.env) count += v.name == "ANTHROPIC_API_KEY";
        CHECK(count == 1);
    }
    SUBCASE("bad arguments") {
        REQUIRE(store.set(kAnthropicProvider, kKey).ok());
        Error e = place_key_in_environment(store, "mistral", &spawn);
        CHECK(e.code == "keystore.bad_provider");
        CHECK_FALSE(mentions(e, kKey));
        CHECK(place_key_in_environment(store, kAnthropicProvider, nullptr).code == "keystore.failed");
    }
}

namespace {
void set_parent_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
}  // namespace

TEST_CASE("the offline switch passes through to the child and cannot be stripped") {
    MemoryKeyStore store;
    REQUIRE(store.set(kAnthropicProvider, kKey).ok());
    acp::SpawnOptions spawn;
    spawn.env = {{"CALIPER_AGENT_OFFLINE", "0"}, {"CALIPER_AGENT_TEST_SCRIPT", "/x/script.json"}};
    spawn.env_remove = {"CALIPER_AGENT_OFFLINE"};

    SUBCASE("with the switch in this process") {
        set_parent_env("CALIPER_AGENT_OFFLINE", "1");
        REQUIRE(place_key_in_environment(store, kAnthropicProvider, &spawn).ok());
        set_parent_env("CALIPER_AGENT_OFFLINE", nullptr);
        REQUIRE(find_var(spawn, "CALIPER_AGENT_OFFLINE") != nullptr);
        CHECK(find_var(spawn, "CALIPER_AGENT_OFFLINE")->value == "1");
        int count = 0;
        for (const auto& v : spawn.env) count += v.name == "CALIPER_AGENT_OFFLINE";
        CHECK(count == 1);
        CHECK(std::count(spawn.env_remove.begin(), spawn.env_remove.end(), std::string("CALIPER_AGENT_OFFLINE")) == 0);
        // The test switch is still removed.
        CHECK(find_var(spawn, "CALIPER_AGENT_TEST_SCRIPT") == nullptr);
        CHECK(std::count(spawn.env_remove.begin(), spawn.env_remove.end(),
                         std::string("CALIPER_AGENT_TEST_SCRIPT")) == 1);
    }
    SUBCASE("without it nothing about it changes") {
        set_parent_env("CALIPER_AGENT_OFFLINE", nullptr);
        REQUIRE(place_key_in_environment(store, kAnthropicProvider, &spawn).ok());
        REQUIRE(find_var(spawn, "CALIPER_AGENT_OFFLINE") != nullptr);
        CHECK(find_var(spawn, "CALIPER_AGENT_OFFLINE")->value == "0");
        CHECK(std::count(spawn.env_remove.begin(), spawn.env_remove.end(), std::string("CALIPER_AGENT_OFFLINE")) == 1);
    }
}

TEST_CASE("wipe clears a string") {
    std::string s = kKey;
    wipe(&s);
    CHECK(s.empty());
    std::string empty;
    wipe(&empty);
    wipe(nullptr);
}

TEST_CASE("the system store exists on every platform") {
    // Construction only: no function of the system store is called, since the
    // Keychain could show a prompt. Elsewhere it reports keystore.unsupported.
    std::unique_ptr<KeyStore> store = make_system_key_store();
    REQUIRE(store != nullptr);
#if !defined(__APPLE__)
    CHECK(store->has(kAnthropicProvider).error.code == "keystore.unsupported");
    CHECK(store->set(kAnthropicProvider, kKey).code == "keystore.unsupported");
#endif
}
