#pragma once
// ============================================================================
// agent_settings.h: the widget agent's settings in the app
// (design/widget-agent.md, Presets, Capabilities and degradations, GPU
// contention). One surface chooses where ⌘K's model requests go:
//
//   - Provider: the presets of caliper_agent/presets.json (Claude, OpenAI,
//     Gemini, OpenRouter, Ollama, LM Studio, Custom), one chosen.
//   - Base URL: fixed for the hosted presets and shown as text; a field for
//     Custom and for the local servers, with the tags "Remote address" (a
//     local preset or Custom on a host that is not this machine) and
//     "Unencrypted" (plain http to such a host).
//   - Model: a field, and the provider's models as choices where it offers
//     them (the Claude preset's four, or the server's list after List).
//   - Effort: where the provider takes one (Claude).
//   - Key: per provider, write only. The row shows Set or Not set and never
//     the key; Save writes the field's text into the key store and wipes it,
//     Replace opens the field again, Remove deletes the key. A local server
//     has no key row. Every key goes through the KeyStore interface, which
//     has no function that returns a key.
//   - Capabilities, beside the model: tool calling, vision and the context
//     length as caliper-agent probe found them, each with the person's
//     override (kept in the probe's cache, model-capabilities.json in the
//     app-data folder, the one place the sidecar reads them), and Probe, which
//     runs the probe again. A model without tool calling shows the plain state
//     "No tool calling", and ⌘K then offers only Ask in chat.
//   - The GPU note: "Shared GPU" while a live run is attached and the
//     provider is a server on this machine (a loopback base URL).
//   - The cache note, under Probe: "Context cache about 18 GB" when a server
//     on this machine would hold a context cache of 4 GB or more for the
//     model, from the context length and the bytes of cache a token takes,
//     which the probe reads from an Ollama server's model metadata. The
//     Ollama preset loads the model with a context length of its own (16,384
//     tokens, about 1 GB for a 27B model), so the note covers a person who
//     raises it, and an Ollama server reached as Custom or LM Studio's
//     address, which loads the model at its full context.
//
// With them, Permissions (design/command-layer.md, Agent permissions): the
// mode of the side chat's agent and of caliper-mcp's clients (Ask, Accept
// edits, Read only, Allow all; each saved and held across launches), and the
// saved rules, each "<agent> · <command>" with Revoke. The
// session keeps them in agent-permissions.json in the app-data folder; this
// surface reads and changes them only through PermissionHooks, as the
// person's own UI.
//
// Where it opens: a popup over the page, a share of the window, from the ⌘K
// prompt's Setup and Model actions and from the landing screen's ⌘, Agent.
// What it stores: widget-agent.json in the app-data folder, beside the side
// chat's side-chat.json: the chosen preset and, per preset, the base URL, the
// model and the effort. Never a key, never in the workspace.
//
// Three parts, as the other components: the state (AgentSettings, with its
// actions and what it shows), the layout (layout_agent_settings(), plain
// rectangles from measured text) and the ImGui side (draw()). The probe runs
// caliper-agent as a child on a thread of its own, with the provider's key
// placed only in its environment by the key store, which also removes every
// key and endpoint variable of the app's own environment from it; when the
// store cannot place the key the provider requires, the probe does not run.
// shutdown() and the destructor end a probe's child at once, without waiting
// for it (its thread reaps it), and an answer for a choice that is no longer
// the current one is dropped.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/widget_surface.h"   // WidgetRect, WidgetMeasure, WidgetFont

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/agent/key_store.h>

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caliper_host {

struct SpecLook;

// ---- what is stored: widget-agent.json, never a key -------------------------------------

struct AgentProviderChoice {
    std::string base_url;   // a local preset's or Custom's; "" takes the preset's own
    std::string model;
    std::string effort;     // for a preset that takes one (Claude)
};

struct AgentSettingsData {
    std::string preset;                                     // "" until one is chosen: no provider
    std::map<std::string, AgentProviderChoice> providers;   // by preset id
    AgentProviderChoice current() const;                    // the chosen preset's, or empty
};

// read_agent_settings() leaves the defaults for a missing file, and for a file
// that does not parse (false, with the reason); unknown presets are dropped.
bool read_agent_settings(const std::string& path, AgentSettingsData* out, std::string* why = nullptr);
bool write_agent_settings(const std::string& path, const AgentSettingsData& data, std::string* why = nullptr);
nlohmann::json agent_settings_to_json(const AgentSettingsData& data);

// ---- what the choice means --------------------------------------------------------------

struct AgentChoice {
    const caliper_builder::agent::ProviderPreset* preset = nullptr;   // null: no provider chosen
    std::string base_url;            // without a trailing slash; "" for Claude
    bool base_url_editable = false;  // Custom and the local servers
    std::string model, effort;
    std::string account;             // the key store's id: the preset's, Custom's per base URL, "" for a local server
    caliper_builder::agent::KeyNeed key = caliper_builder::agent::KeyNeed::none;
    bool loopback = false;           // the base URL's host is this machine
};
AgentChoice agent_choice(const AgentSettingsData& data);

// An http or https address with a host and without a user name, a password,
// a query or a fragment, as the sidecar checks it.
bool agent_base_url_valid(const std::string& url);
bool agent_url_is_loopback(const std::string& url);
// The tags beside the base URL: "Invalid address", "Remote address", "Unencrypted".
std::vector<std::string> agent_base_url_warnings(const AgentChoice& choice);
// "Shared GPU" while a live run is attached and the provider is a server on
// this machine; "" otherwise.
std::string agent_gpu_note(bool live_attached, const AgentChoice& choice);

// What the sidecar will assume of a model, read from the probe's cache
// (model-capabilities.json in `cache_dir`) as the sidecar reads it: the
// person's override, else a probe answer younger than 30 days.
struct AgentCapabilities {
    std::optional<bool> tools, vision;
    std::optional<long long> context;
    std::optional<long long> kv_bytes_per_token;   // the context cache one token takes, where the probe read it
    std::string tools_source = "unknown", vision_source = "unknown", context_source = "unknown";
    nlohmann::json overrides = nlohmann::json::object();
};
AgentCapabilities read_cached_capabilities(const std::string& cache_dir, const std::string& base_url,
                                           const std::string& model, double now_seconds);
// "Context cache about <N> GB" when the provider is a server on this machine
// and the model's context length times the cache a token takes reaches
// kLargeCacheBytes; "" otherwise, and when either is unknown.
constexpr long long kLargeCacheBytes = 4000000000LL;
std::string agent_cache_note(const AgentChoice& choice, const AgentCapabilities& caps);

// ---- the sidecar's start ----------------------------------------------------------------

struct AgentLaunch {
    bool usable = false;
    std::string why;                 // when not usable: "No provider", "Invalid address", "No model", "No key", "No tool calling"
    std::vector<std::string> args;   // --preset, --base-url, --model, --effort, --cache-dir
    std::string account;             // the key store's id, "" for a local server
    bool key_required = false;
    bool local = false;              // a loopback base URL
};
// `keys` null: the key's presence is not checked (a hidden app's sidecar on a
// test's script). `cache_dir` "" leaves --cache-dir out.
AgentLaunch agent_launch(const AgentSettingsData& data, caliper_builder::agent::KeyStore* keys,
                         const std::string& cache_dir, double now_seconds);

// The app's key store: the system's (the macOS Keychain) for a normal run, an
// in-memory one for a hidden app, which never touches the system's. One per
// process; the settings and the widget agent share it.
caliper_builder::agent::KeyStore* app_key_store(bool hidden);

// ---- the probe's answer -----------------------------------------------------------------

struct AgentProbeAnswer {
    bool ran = false;
    std::string failure;                  // the probe did not answer: why
    std::vector<std::string> models;      // the server's list, when it gave one
    AgentCapabilities caps;
    std::vector<std::string> problems;    // noun phrases of the probe's errors: "Server unreachable", ...
    std::vector<std::string> warnings;
    bool can_drive = true;
};
// The JSON that `caliper-agent probe` prints.
AgentProbeAnswer parse_probe_answer(const std::string& text);
// A probe error's code as a noun phrase ("probe.unreachable": "Server unreachable").
std::string probe_problem_name(const std::string& code);

// ---- the layout -------------------------------------------------------------------------

struct AgentSettingsMetrics {
    float row = 28;        // a row of fields or buttons
    float line = 20;       // a line of text
    float pad = 16;        // inside the window
    float gap = 8;         // between rows and buttons
    float button_pad = 10; // inside a button, left and right
    float label = 120;     // the label column
    float field = 280;     // the narrowest field
    float column = 400;    // the narrowest capabilities column
    float min_w = 560;
    float share_w = 0.62f;   // of the work area's width; the height is the content's
    float top_share = 0.12f; // the window's top, as a share of the work area's height from its top
    float margin = 24;     // between the window and the work area's edges
    static AgentSettingsMetrics for_text_scale(float s);
};

struct AgentSettingsLayoutInput {
    WidgetRect area;                       // the work area the popup sits in
    std::vector<std::string> presets;      // the presets' names
    bool base_url_field = false;           // a field rather than fixed text
    int warnings = 0;                      // tags under the base URL
    std::vector<std::string> models;       // choices under the model field
    std::vector<std::string> efforts;      // empty: no effort row
    bool key_row = true;
    bool key_field = false;                // the field and Save are shown
    bool key_set = false;                  // Replace and Remove are shown
    bool note = false;                     // the GPU note
    int problems = 0;                      // the probe's tags
    bool state = false;                    // the plain state tag ("No tool calling")
    std::string cache_note;                // "Context cache about 18 GB", or ""
    bool permissions = false;              // the Permissions section
    int rules = 0;                         // its saved rules (at most kMaxRulesShown are laid out)
};

struct AgentSettingsLayout {
    WidgetRect window, title, close;
    std::vector<WidgetRect> presets;       // in the presets' order
    WidgetRect note;
    WidgetRect base_url, base_url_label;
    std::vector<WidgetRect> warnings;
    WidgetRect model, model_label, list;
    std::vector<WidgetRect> models;        // the choices shown; the rest fold
    int models_folded = 0;
    WidgetRect effort_label;
    std::vector<WidgetRect> efforts;
    WidgetRect key_label, key_status, key_field;
    std::vector<WidgetRect> key_buttons;   // Save (and Cancel), or Replace and Remove
    WidgetRect caps;                       // the capabilities column, or the block under the model
    WidgetRect caps_title;
    WidgetRect tools, vision, context;     // their rows
    std::vector<WidgetRect> tools_buttons, vision_buttons;   // Yes, No, Auto
    WidgetRect context_field, context_auto;
    WidgetRect probe;
    std::vector<WidgetRect> problems;
    WidgetRect state;
    WidgetRect cache_note;
    bool two_columns = false;
    // Permissions: in the left column under the key with two columns, else
    // across the window under the rest; empty without them.
    WidgetRect perm_title;
    WidgetRect chat_label, mcp_label;
    std::vector<WidgetRect> chat_modes, mcp_modes;   // Ask, Accept edits, Read only, Allow all
    WidgetRect rules_label;
    std::vector<WidgetRect> rules, revokes;          // a rule's text and its Revoke
    WidgetRect rules_more;                           // "+N" past kMaxRulesShown, or the "None" line
};

// The most saved rules the section lists; the rest fold into "+N".
constexpr int kMaxRulesShown = 8;

AgentSettingsLayout layout_agent_settings(const AgentSettingsLayoutInput& in, const AgentSettingsMetrics& m,
                                          const WidgetMeasure& measure);

// ---- the component ----------------------------------------------------------------------

// What the Permissions section reads and changes, through the session
// (Session::agent_mode() and its neighbours). "chat" and "mcp" name whose
// mode; a mode is "ask", "accept_edits", "read_only" or "allow_all". Unset
// hooks hide the section.
struct AgentPermissionHooks {
    std::function<std::string(const std::string& who)> mode;
    std::function<void(const std::string& who, const std::string& mode)> set_mode;
    std::function<std::vector<std::pair<std::string, std::string>>()> rules;   // agent, command
    std::function<void(const std::string& agent, const std::string& command)> revoke;
};

class AgentSettings {
public:
    struct Options {
        std::string settings_path;   // widget-agent.json in the app-data folder
        std::string cache_dir;       // the app-data folder, where the probe's cache lives
        std::string python, source;  // as the widget agent's (widget_app.h); "" leaves Probe and List out
        caliper_builder::agent::KeyStore* keys = nullptr;   // app_key_store(); null: no key row works
        std::function<bool()> live_attached;                // a live run is attached (main.cpp's live_)
        std::function<void(const std::string& event, const std::string& object, const std::string& detail)> log;
        std::function<double()> now;                        // seconds since the epoch, for the cache's expiry
        // Added to the probe's environment, and removed from it; the probe
        // also inherits CALIPER_AGENT_OFFLINE through the key store.
        std::vector<caliper_builder::acp::EnvVar> probe_env;
        std::vector<std::string> probe_env_remove = {"CALIPER_AGENT_TEST_SCRIPT"};
        int probe_timeout_ms = 200000;
    };

    AgentSettings();
    ~AgentSettings();
    AgentSettings(const AgentSettings&) = delete;
    AgentSettings& operator=(const AgentSettings&) = delete;

    // Reads widget-agent.json; the surface stays closed.
    void start(Options options);
    const Options& options() const { return options_; }

    bool is_open() const { return open_; }
    void set_open(bool open);
    // Grows with every change of the choice or of a key, and with a probe's
    // answer for the current choice, so that the widget agent is made again
    // with it. `on_change` runs on the frame thread after each change (the
    // spec page drops its widget agent then); one listener, null for none.
    int revision() const { return revision_; }
    void set_on_change(std::function<void()> on_change) { on_change_ = std::move(on_change); }
    // The Permissions section's hooks; the app sets them once its session is open.
    void set_permissions(AgentPermissionHooks hooks) { permissions_ = std::move(hooks); }
    bool has_permissions() const { return permissions_.mode && permissions_.set_mode && permissions_.rules; }
    // The section's state and actions, as its buttons read and do them.
    std::string permission_mode(const std::string& who) const {
        return permissions_.mode ? permissions_.mode(who) : std::string();
    }
    void set_permission_mode(const std::string& who, const std::string& mode) {
        if (permissions_.set_mode) permissions_.set_mode(who, mode);
    }
    std::vector<std::pair<std::string, std::string>> saved_rules() const {
        return permissions_.rules ? permissions_.rules() : std::vector<std::pair<std::string, std::string>>();
    }
    void revoke_rule(const std::string& agent, const std::string& command) {
        if (permissions_.revoke) permissions_.revoke(agent, command);
    }

    // ---- the choice
    const AgentSettingsData& data() const { return data_; }
    AgentChoice choice() const { return agent_choice(data_); }
    void choose_preset(const std::string& id);
    void set_base_url(std::string url);
    void set_model(std::string model);
    void set_effort(std::string effort);
    std::vector<std::string> warnings() const { return agent_base_url_warnings(choice()); }
    std::string gpu_note() const;
    std::string cache_note() const;   // agent_cache_note() of the current choice

    // ---- the key: write only
    enum class KeyStatus { none, set, not_set, unknown };   // none: no key row (a local server, no preset)
    KeyStatus key_status() const;
    bool key_editing() const { return key_editing_; }
    const std::string& key_problem() const { return key_problem_; }
    void begin_key_entry();
    void set_key_text(std::string text);   // what the field holds, until Save or Cancel
    bool key_text_empty() const { return key_text_.empty(); }
    bool save_key();
    void cancel_key();
    bool remove_key();

    // ---- models and capabilities
    // The choices under the model field: the preset's own, else the server's
    // list after List.
    std::vector<std::string> model_choices() const;
    void list_models();
    void probe(bool refresh = true);
    // "yes", "no" or "auto" for tools and vision; a number of tokens or "auto"
    // for context_length. Kept in the probe's cache.
    void set_override(const std::string& name, const std::string& value);
    bool busy() const;
    const AgentProbeAnswer& probe_answer() const { return answer_; }
    // The probe's answer, else the cache's for the chosen model.
    AgentCapabilities capabilities() const;
    // The plain state: "No provider", "No model", "No key", "Invalid address",
    // "No tool calling", or "" when ⌘K can run.
    std::string state() const;
    // Every text the surface shows, for tests: never a key.
    std::vector<std::string> visible_text() const;

    // ---- per frame (frame thread)
    void frame();   // takes the probe's answer when it has come, if it is still the current choice's
    // The popup in the work area `x, y, w, h`, while open; Escape closes it.
    void draw(float x, float y, float w, float h, const SpecLook& look);
    // ⌘, (Ctrl+, elsewhere) opens it; inside a frame.
    void read_keys();
    // Ends a probe's child at once and forgets the probe; never waits for it.
    void shutdown();

    // The layout input of the surface as it is.
    AgentSettingsLayoutInput layout_input(const WidgetRect& area) const;

private:
    void changed();
    void bump();   // the revision grows and the listener runs
    void save();
    void run_probe(std::vector<std::string> extra, bool list_only);

    Options options_;
    AgentSettingsData data_;
    bool open_ = false, focus_ = false;
    int revision_ = 0;
    std::function<void()> on_change_;
    AgentPermissionHooks permissions_;
    bool key_editing_ = false;
    std::string key_text_;
    std::string key_problem_;
    mutable int key_revision_seen_ = -1;
    mutable KeyStatus key_status_ = KeyStatus::unknown;
    std::map<std::string, std::vector<std::string>> listed_;   // the server's models, by preset and base URL
    AgentProbeAnswer answer_;
    std::string answer_for_;                                   // the base URL and model of answer_
    std::string context_text_;                                 // the context override's field
    // The fields' text while they are edited; applied when the field is left.
    std::string base_url_edit_, model_edit_;
    bool base_url_active_ = false, model_active_ = false;
    mutable int caps_revision_seen_ = -1;
    mutable AgentCapabilities caps_cached_;
    struct Runner;
    std::unique_ptr<Runner> runner_;
};

}  // namespace caliper_host
