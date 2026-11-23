// agent_settings.cpp: the widget agent's settings in the app (agent_settings.h).
#include "builder/agent_settings.h"

#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/agent/key_store.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#ifndef _WIN32
#include <csignal>
#include <sys/types.h>
#endif

namespace caliper_host {

using nlohmann::json;
namespace agent = caliper_builder::agent;
namespace acp = caliper_builder::acp;

namespace {

constexpr double kProbeTtlSeconds = 30.0 * 24 * 3600;   // as the sidecar's cache (capabilities.py)
constexpr const char* kClaudeWidgetModel = "claude-haiku-5-5";   // the sidecar's widget default
constexpr const char* kCacheFile = "model-capabilities.json";

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string strip_slashes(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

struct UrlParts {
    bool ok = false;
    std::string scheme, host;
};

UrlParts parse_url(const std::string& u) {
    UrlParts p;
    if (u.empty() || u.size() > 2048) return p;
    for (unsigned char c : u)
        if (c <= 0x20 || c == 0x7F) return p;
    const size_t sep = u.find("://");
    if (sep == std::string::npos) return p;
    p.scheme = lower(u.substr(0, sep));
    const std::string after = u.substr(sep + 3);
    const size_t end = after.find_first_of("/?#");
    const std::string authority = after.substr(0, end);
    const std::string rest = end == std::string::npos ? std::string() : after.substr(end);
    if (authority.find('@') != std::string::npos) return p;                               // a user name or password
    if (rest.find('?') != std::string::npos || rest.find('#') != std::string::npos) return p;   // a query or a fragment
    std::string host, port;
    if (!authority.empty() && authority[0] == '[') {
        const size_t close = authority.find(']');
        if (close == std::string::npos) return p;
        host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') return p;
            port = authority.substr(close + 2);
        }
    } else {
        const size_t colon = authority.find(':');
        host = authority.substr(0, colon);
        if (colon != std::string::npos) port = authority.substr(colon + 1);
    }
    if (!port.empty()) {
        if (port.size() > 5 || !std::all_of(port.begin(), port.end(), [](char c) { return std::isdigit((unsigned char)c); }) ||
            std::stol(port) > 65535)
            return p;
    }
    p.host = lower(host);
    p.ok = (p.scheme == "http" || p.scheme == "https") && !p.host.empty();
    return p;
}

bool loopback_host(const std::string& host) {
    if (host == "localhost" || host == "::1") return true;
    if (host.rfind("127.", 0) != 0) return false;
    int dots = 0;
    for (char c : host) {
        if (c == '.') ++dots;
        else if (!std::isdigit((unsigned char)c)) return false;
    }
    return dots == 3;
}

std::string read_file(const std::string& path, size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string s(limit + 1, '\0');
    in.read(&s[0], (std::streamsize)s.size());
    s.resize((size_t)in.gcount());
    return s;
}

bool write_atomic(const std::string& path, const std::string& text, std::string* why) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!(out << text)) {
            if (why) *why = "cannot write " + fs::path(path).filename().string();
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        if (why) *why = ec.message();
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

std::string model_for_cache(const AgentChoice& c) {
    if (c.preset && c.preset->api == "claude" && c.model.empty()) return kClaudeWidgetModel;
    return c.model;
}

std::string base_for_cache(const AgentChoice& c) {
    if (c.preset && c.preset->api == "claude") return strip_slashes(c.preset->base_url);
    return c.base_url;
}

// What a probe's answer belongs to: the base URL and the model it checked,
// or, for a list, the preset and the base URL.
std::string probe_key(const AgentChoice& c) { return base_for_cache(c) + " " + model_for_cache(c); }
std::string list_key(const AgentChoice& c) { return c.preset ? c.preset->id + " " + c.base_url : std::string(); }

// Why the key store could not place a key, as the probe's tag.
std::string key_failure(const caliper_builder::Error& e) {
    if (e.code == "keystore.not_found") return "No key";
    if (e.code == "keystore.bad_key") return "Malformed key";
    return "Unreadable key";
}

// The id the key store places the key under: the choice's account, or the
// preset's own for a server that takes none (place_key_in_environment() then
// places nothing and removes every key variable).
std::string placement_id(const AgentChoice& c) {
    if (!c.account.empty()) return c.account;
    return c.preset && c.preset->id != "custom" ? c.preset->id : std::string();
}

}  // namespace

// ---- storage ------------------------------------------------------------------------------

AgentProviderChoice AgentSettingsData::current() const {
    const auto it = providers.find(preset);
    return it == providers.end() ? AgentProviderChoice{} : it->second;
}

json agent_settings_to_json(const AgentSettingsData& d) {
    json providers = json::object();
    for (const auto& [id, c] : d.providers) {
        json p = json::object();
        if (!c.base_url.empty()) p["base_url"] = c.base_url;
        if (!c.model.empty()) p["model"] = c.model;
        if (!c.effort.empty()) p["effort"] = c.effort;
        providers[id] = p;
    }
    return {{"version", 1}, {"preset", d.preset}, {"providers", providers}};
}

bool read_agent_settings(const std::string& path, AgentSettingsData* out, std::string* why) {
    *out = AgentSettingsData{};
    const std::string text = read_file(path, 64 << 10);
    if (text.empty()) return true;
    const json j = json::parse(text, nullptr, false);
    if (!j.is_object() || j.value("version", 0) != 1) {
        if (why) *why = "widget-agent.json does not parse";
        return false;
    }
    auto bounded = [](const json& o, const char* key) {
        return o.contains(key) && o[key].is_string() && o[key].get<std::string>().size() <= 2048 ? o[key].get<std::string>()
                                                                                                  : std::string();
    };
    const std::string preset = bounded(j, "preset");
    if (agent::find_preset(preset)) out->preset = preset;
    if (j.contains("providers") && j["providers"].is_object()) {
        for (auto it = j["providers"].begin(); it != j["providers"].end(); ++it) {
            if (!agent::find_preset(it.key()) || !it.value().is_object()) continue;
            AgentProviderChoice c;
            c.base_url = bounded(it.value(), "base_url");
            c.model = bounded(it.value(), "model");
            c.effort = bounded(it.value(), "effort");
            out->providers[it.key()] = c;
        }
    }
    return true;
}

bool write_agent_settings(const std::string& path, const AgentSettingsData& d, std::string* why) {
    return write_atomic(path, agent_settings_to_json(d).dump(1) + "\n", why);
}

// ---- the choice ---------------------------------------------------------------------------

bool agent_base_url_valid(const std::string& url) { return parse_url(url).ok; }

bool agent_url_is_loopback(const std::string& url) {
    const UrlParts p = parse_url(url);
    return p.ok && loopback_host(p.host);
}

AgentChoice agent_choice(const AgentSettingsData& d) {
    AgentChoice c;
    c.preset = agent::find_preset(d.preset);
    if (!c.preset) return c;
    const AgentProviderChoice pc = d.current();
    const agent::ProviderPreset& p = *c.preset;
    if (p.api == "claude") {
        c.base_url.clear();
    } else if (p.id == "custom") {
        c.base_url = strip_slashes(pc.base_url);
        c.base_url_editable = true;
    } else if (p.local) {
        c.base_url = strip_slashes(pc.base_url.empty() ? p.base_url : pc.base_url);
        c.base_url_editable = true;
    } else {
        c.base_url = strip_slashes(p.base_url);
    }
    c.model = pc.model;
    if (!p.efforts.empty()) c.effort = pc.effort;
    if (p.id == "custom") {
        c.key = agent::KeyNeed::optional;
        if (agent_base_url_valid(c.base_url)) c.account = agent::custom_key_account(c.base_url);
    } else {
        c.key = p.key;
        if (p.key != agent::KeyNeed::none) c.account = p.id;
    }
    c.loopback = p.api != "claude" && agent_url_is_loopback(c.base_url);
    return c;
}

std::vector<std::string> agent_base_url_warnings(const AgentChoice& c) {
    std::vector<std::string> out;
    if (!c.preset || c.preset->api == "claude" || !c.base_url_editable) return out;
    if (c.base_url.empty()) return out;
    const UrlParts u = parse_url(c.base_url);
    if (!u.ok) return {"Invalid address"};
    if (!loopback_host(u.host)) {
        out.push_back("Remote address");
        if (u.scheme == "http") out.push_back("Unencrypted");
    }
    return out;
}

std::string agent_gpu_note(bool live_attached, const AgentChoice& c) {
    return live_attached && c.preset && c.loopback ? "Shared GPU" : std::string();
}

std::string agent_cache_note(const AgentChoice& c, const AgentCapabilities& caps) {
    if (!c.preset || !c.loopback || !caps.context || !caps.kv_bytes_per_token) return {};
    if (*caps.context <= 0 || *caps.kv_bytes_per_token <= 0) return {};
    const double bytes = (double)*caps.context * (double)*caps.kv_bytes_per_token;
    if (bytes < (double)kLargeCacheBytes) return {};
    const double gb = bytes / 1e9;
    char text[64];
    std::snprintf(text, sizeof text, "Context cache about %.0f GB", std::round(gb));
    return text;
}

AgentCapabilities read_cached_capabilities(const std::string& cache_dir, const std::string& base_url,
                                           const std::string& model, double now) {
    AgentCapabilities caps;
    if (cache_dir.empty() || model.empty()) return caps;
    const std::string text = read_file((std::filesystem::path(cache_dir) / kCacheFile).string(), 1 << 20);
    const json j = json::parse(text, nullptr, false);
    if (!j.is_object() || j.value("version", 0) != 1 || !j.contains("entries") || !j["entries"].is_array()) return caps;
    for (const json& e : j["entries"]) {
        if (!e.is_object() || e.value("base_url", "") != strip_slashes(base_url) || e.value("model", "") != model) continue;
        const json result = e.value("result", json::object());
        const json over = e.value("overrides", json::object());
        const bool fresh = e.contains("probed_at") && e["probed_at"].is_number() &&
                           now - e["probed_at"].get<double>() <= kProbeTtlSeconds;
        caps.overrides = over.is_object() ? over : json::object();
        auto pick_bool = [&](const char* name, std::optional<bool>* value, std::string* source) {
            if (over.is_object() && over.contains(name) && over[name].is_boolean()) {
                *value = over[name].get<bool>();
                *source = "override";
            } else if (fresh && result.contains(name) && result[name].is_object() && result[name].contains("value") &&
                       result[name]["value"].is_boolean()) {
                *value = result[name]["value"].get<bool>();
                *source = result[name].value("source", "probe");
            }
        };
        if (fresh && result.contains("quirks") && result["quirks"].is_object() &&
            result["quirks"].contains("kv_bytes_per_token") && result["quirks"]["kv_bytes_per_token"].is_number_integer())
            caps.kv_bytes_per_token = result["quirks"]["kv_bytes_per_token"].get<long long>();
        pick_bool("tools", &caps.tools, &caps.tools_source);
        pick_bool("vision", &caps.vision, &caps.vision_source);
        if (over.is_object() && over.contains("context_length") && over["context_length"].is_number_integer()) {
            caps.context = over["context_length"].get<long long>();
            caps.context_source = "override";
        } else if (fresh && result.contains("context_length") && result["context_length"].is_object() &&
                   result["context_length"].contains("value") && result["context_length"]["value"].is_number_integer()) {
            caps.context = result["context_length"]["value"].get<long long>();
            caps.context_source = result["context_length"].value("source", "server");
        }
        break;
    }
    return caps;
}

AgentLaunch agent_launch(const AgentSettingsData& d, agent::KeyStore* keys, const std::string& cache_dir, double now) {
    AgentLaunch L;
    const AgentChoice c = agent_choice(d);
    if (!c.preset) return L.why = "No provider", L;
    L.local = c.loopback;
    L.args = {"--preset", c.preset->id};
    if (c.preset->api != "claude") {
        if (!agent_base_url_valid(c.base_url)) return L.why = "Invalid address", L;
        if (c.base_url_editable) L.args.insert(L.args.end(), {"--base-url", c.base_url});
        if (c.model.empty()) return L.why = "No model", L;
    }
    if (!c.model.empty()) L.args.insert(L.args.end(), {"--model", c.model});
    if (!c.effort.empty()) L.args.insert(L.args.end(), {"--effort", c.effort});
    if (!cache_dir.empty()) L.args.insert(L.args.end(), {"--cache-dir", cache_dir});
    L.account = placement_id(c);
    L.key_required = c.key == agent::KeyNeed::required;
    if (keys && L.key_required && !keys->has(c.account).present) return L.why = "No key", L;
    const AgentCapabilities caps = read_cached_capabilities(cache_dir, base_for_cache(c), model_for_cache(c), now);
    if (caps.tools && !*caps.tools) return L.why = "No tool calling", L;
    L.usable = true;
    return L;
}

agent::KeyStore* app_key_store(bool hidden) {
    if (hidden) {
        static agent::MemoryKeyStore memory;   // a hidden app never touches the system's store
        return &memory;
    }
    static std::unique_ptr<agent::KeyStore> system = agent::make_system_key_store();
    return system.get();
}

// ---- the probe's answer -------------------------------------------------------------------

std::string probe_problem_name(const std::string& code) {
    if (code == "probe.unreachable") return "Server unreachable";
    if (code == "probe.timeout") return "Timeout";
    if (code == "probe.key_rejected") return "Key rejected";
    if (code == "probe.forbidden") return "Access refused";
    if (code == "probe.model_unavailable") return "Model unavailable";
    if (code == "probe.server_error") return "Server error";
    if (code == "probe.cut_off") return "Reply cut off";
    if (code == "probe.no_key") return "No key";
    if (code == "agent.offline") return "Offline";
    return "Probe failed";
}

AgentProbeAnswer parse_probe_answer(const std::string& text) {
    AgentProbeAnswer a;
    const json j = json::parse(text, nullptr, false);
    if (!j.is_object()) {
        a.failure = "No answer";
        return a;
    }
    a.ran = true;
    if (j.contains("models") && j["models"].is_array())
        for (const json& m : j["models"])
            if (m.is_string()) a.models.push_back(m.get<std::string>());
    const json caps = j.value("capabilities", json::object());
    auto cap = [&](const char* name) { return caps.is_object() && caps.contains(name) ? caps[name] : json::object(); };
    auto as_bool = [](const json& c, std::optional<bool>* v, std::string* src) {
        if (c.contains("value") && c["value"].is_boolean()) *v = c["value"].get<bool>();
        *src = c.value("source", std::string("unknown"));
    };
    as_bool(cap("tools"), &a.caps.tools, &a.caps.tools_source);
    as_bool(cap("vision"), &a.caps.vision, &a.caps.vision_source);
    const json ctx = cap("context_length");
    if (ctx.contains("value") && ctx["value"].is_number_integer()) a.caps.context = ctx["value"].get<long long>();
    a.caps.context_source = ctx.value("source", std::string("unknown"));
    if (j.contains("overrides") && j["overrides"].is_object()) a.caps.overrides = j["overrides"];
    const json quirks = j.value("quirks", json::object());
    if (quirks.is_object() && quirks.contains("kv_bytes_per_token") && quirks["kv_bytes_per_token"].is_number_integer())
        a.caps.kv_bytes_per_token = quirks["kv_bytes_per_token"].get<long long>();
    for (const json& e : j.value("errors", json::array())) {
        if (!e.is_object()) continue;
        const std::string name = probe_problem_name(e.value("code", ""));
        if (std::find(a.problems.begin(), a.problems.end(), name) == a.problems.end()) a.problems.push_back(name);
    }
    for (const json& w : j.value("warnings", json::array()))
        if (w.is_string()) a.warnings.push_back(w.get<std::string>());
    a.can_drive = j.value("can_drive_caliper", true);
    return a;
}

// ---- the layout ---------------------------------------------------------------------------

AgentSettingsMetrics AgentSettingsMetrics::for_text_scale(float s) {
    if (!(s > 0.0f)) s = 1.0f;
    AgentSettingsMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.row = r(m.row);
    m.line = r(m.line);
    m.pad = r(m.pad);
    m.gap = r(m.gap);
    m.button_pad = r(m.button_pad);
    m.label = r(m.label);
    m.field = r(m.field);
    m.column = r(m.column);
    m.min_w = r(m.min_w);
    m.margin = r(m.margin);
    return m;
}

namespace {

// The rows inside a window whose x, y and width are given; `bottom` gets the
// content's lowest edge.
AgentSettingsLayout layout_in_window(const AgentSettingsLayoutInput& in, const AgentSettingsMetrics& m,
                                     const WidgetMeasure& measure, const WidgetRect& window, float* bottom) {
    AgentSettingsLayout L;
    L.window = window;
    const WidgetRect& W = L.window;
    auto bw = [&](const std::string& label) { return std::round(measure(label, WidgetFont::label) + 2 * m.button_pad); };

    L.title = {W.x + m.pad, W.y + m.pad, W.w - 2 * m.pad, m.row};
    const float close_w = bw("esc Close");
    L.close = {L.title.right() - close_w, L.title.y, close_w, m.row};

    // Two columns when the window holds the left rows and the capabilities.
    const float inner = W.w - 2 * m.pad;
    L.two_columns = inner >= m.label + m.field + m.gap * 3 + m.column;
    const float left_w = L.two_columns ? inner - m.column - m.gap * 3 : inner;
    const float x0 = W.x + m.pad;
    const float fx = x0 + m.label;            // fields start after the label column
    const float fw = left_w - m.label;        // and take the rest of the left column
    float y = L.title.bottom() + m.gap * 2;

    // Buttons laid out in rows that wrap inside [fx, fx + fw).
    auto flow = [&](const std::vector<std::string>& labels, std::vector<WidgetRect>* out, int max_rows, int* folded) {
        float x = fx, row_y = y;
        int rows = 1;
        for (size_t i = 0; i < labels.size(); ++i) {
            const float b = std::min(bw(labels[i]), fw);
            if (x > fx && x + b > fx + fw) {
                if (rows == max_rows) {
                    if (folded) *folded = (int)(labels.size() - i);
                    break;
                }
                ++rows;
                x = fx;
                row_y += m.row + m.gap;
            }
            out->push_back({x, row_y, b, m.row});
            x += b + m.gap;
        }
        y = row_y + m.row + m.gap;
    };

    flow(in.presets, &L.presets, 3, nullptr);
    if (in.note) {
        L.note = {fx, y, std::min(fw, bw("Shared GPU")), m.line};
        y += m.line + m.gap;
    }
    y += m.gap;

    L.base_url_label = {x0, y, m.label - m.gap, m.row};
    L.base_url = {fx, y, fw, m.row};
    y += m.row + m.gap;
    if (in.warnings > 0) {
        float x = fx;
        for (int i = 0; i < in.warnings; ++i) {
            const float b = bw(i == 0 ? "Remote address" : "Unencrypted");
            L.warnings.push_back({x, y, b, m.line});
            x += b + m.gap;
        }
        y += m.line + m.gap;
    }
    y += m.gap;

    L.model_label = {x0, y, m.label - m.gap, m.row};
    const float list_w = bw("List");
    L.model = {fx, y, std::max(0.0f, fw - list_w - m.gap), m.row};
    L.list = {fx + fw - list_w, y, list_w, m.row};
    y += m.row + m.gap;
    if (!in.models.empty()) flow(in.models, &L.models, 3, &L.models_folded);
    y += m.gap;

    if (!in.efforts.empty()) {
        L.effort_label = {x0, y, m.label - m.gap, m.row};
        flow(in.efforts, &L.efforts, 1, nullptr);
        y += m.gap;
    }

    if (in.key_row) {
        L.key_label = {x0, y, m.label - m.gap, m.row};
        const float status_w = std::max(bw("Not set"), bw("Set"));
        L.key_status = {fx, y, status_w, m.row};
        float x = L.key_status.right() + m.gap;
        if (in.key_field) {
            const float save = bw("Save"), cancel = in.key_set ? bw("Cancel") : 0.0f;
            const float field = std::max(0.0f, fx + fw - x - save - m.gap - (cancel > 0 ? cancel + m.gap : 0.0f));
            L.key_field = {x, y, field, m.row};
            x += field + m.gap;
            L.key_buttons.push_back({x, y, save, m.row});
            x += save + m.gap;
            if (cancel > 0) L.key_buttons.push_back({x, y, cancel, m.row});
        } else if (in.key_set) {
            for (const char* b : {"Replace", "Remove"}) {
                L.key_buttons.push_back({x, y, bw(b), m.row});
                x += bw(b) + m.gap;
            }
        }
        y += m.row + m.gap * 2;
    }

    // The capabilities: beside the model in two columns, else below.
    const float cx = L.two_columns ? x0 + left_w + m.gap * 3 : x0;
    const float cw = L.two_columns ? inner - left_w - m.gap * 3 : inner;
    float cy = L.two_columns ? L.base_url.y : y;
    L.caps = {cx, cy, cw, 0};
    L.caps_title = {cx, cy, cw, m.line};
    cy += m.line + m.gap;
    const float yes = bw("Yes"), no = bw("No"), autob = bw("Auto");
    auto cap_row = [&](WidgetRect* row, std::vector<WidgetRect>* buttons) {
        *row = {cx, cy, cw, m.row};
        float x = cx + cw - (yes + no + autob + 2 * m.gap);
        for (float b : {yes, no, autob}) {
            buttons->push_back({x, cy, b, m.row});
            x += b + m.gap;
        }
        cy += m.row + m.gap;
    };
    cap_row(&L.tools, &L.tools_buttons);
    cap_row(&L.vision, &L.vision_buttons);
    L.context = {cx, cy, cw, m.row};
    const float digits = std::round(measure("10000000", WidgetFont::mono) + 2 * m.button_pad);
    L.context_auto = {cx + cw - autob, cy, autob, m.row};
    L.context_field = {L.context_auto.x - m.gap - digits, cy, digits, m.row};
    cy += m.row + m.gap * 2;
    L.probe = {cx, cy, bw("Probe"), m.row};
    cy += m.row + m.gap;
    float px = cx;
    for (int i = 0; i < in.problems; ++i) {
        const float b = std::min(bw("Server unreachable"), cw);
        if (px > cx && px + b > cx + cw) {
            px = cx;
            cy += m.line + m.gap;
        }
        L.problems.push_back({px, cy, b, m.line});
        px += b + m.gap;
    }
    if (in.problems > 0) cy += m.line + m.gap;
    if (in.state) {
        L.state = {cx, cy, std::min(cw, bw("No tool calling")), m.line};
        cy += m.line + m.gap;
    }
    if (!in.cache_note.empty()) {
        L.cache_note = {cx, cy, std::min(cw, bw(in.cache_note)), m.line};
        cy += m.line + m.gap;
    }
    L.caps.h = cy - L.caps.y;

    // Permissions: in the left column under the key when there are two
    // columns, where the capabilities leave room; else under everything.
    if (in.permissions) {
        const float pw = L.two_columns ? left_w : inner;
        if (!L.two_columns) y = std::max(y, cy);
        y += m.gap * 2;
        L.perm_title = {x0, y, pw, m.line};
        y += m.line + m.gap;
        if (in.perm_note) {
            L.perm_note = {x0, y, pw, 2 * m.line};
            y += 2 * m.line + m.gap;
        }
        const std::vector<std::string> modes = {"Ask", "Accept edits", "Read only", "Allow all"};
        auto mode_row = [&](WidgetRect* label, std::vector<WidgetRect>* buttons) {
            *label = {x0, y, m.label - m.gap, m.row};
            float x = fx;
            for (const auto& b : modes) {
                // A button that would pass the column's edge starts a line of its own.
                if (x > fx && x + bw(b) > x0 + pw) {
                    x = fx;
                    y += m.row + m.gap;
                }
                buttons->push_back({x, y, bw(b), m.row});
                x += bw(b) + m.gap;
            }
            y += m.row + m.gap;
        };
        mode_row(&L.chat_label, &L.chat_modes);
        mode_row(&L.mcp_label, &L.mcp_modes);
        L.rules_label = {x0, y, m.label - m.gap, m.row};
        const float revoke = bw("Revoke");
        const int shown = std::min(in.rules, kMaxRulesShown);
        for (int i = 0; i < shown; ++i) {
            L.rules.push_back({fx, y, std::max(0.0f, x0 + pw - fx - revoke - m.gap), m.row});
            L.revokes.push_back({x0 + pw - revoke, y, revoke, m.row});
            y += m.row + m.gap;
        }
        if (in.rules == 0 || in.rules > shown) {
            L.rules_more = {fx, y, std::max(0.0f, x0 + pw - fx), m.row};
            y += m.row + m.gap;
        }
    }
    *bottom = std::max(y, cy);
    return L;
}

}  // namespace

AgentSettingsLayout layout_agent_settings(const AgentSettingsLayoutInput& in, const AgentSettingsMetrics& m,
                                          const WidgetMeasure& measure) {
    // One placement rule at every size: a share of the work area's width,
    // centred across it; the top at a fixed share of its height from its top
    // edge; as tall as the content, moving up only when the content would
    // pass the area's bottom margin.
    const WidgetRect& A = in.area;
    const float w = std::round(std::max(std::min(m.min_w, A.w - 2 * m.margin), std::min(A.w * m.share_w, A.w - 2 * m.margin)));
    const float avail = std::max(0.0f, A.h - 2 * m.margin);
    const float x = std::round(A.x + (A.w - w) * 0.5f);
    float bottom = 0;
    layout_in_window(in, m, measure, {x, A.y + m.margin, w, avail}, &bottom);
    const float h = std::round(std::min(avail, bottom - (A.y + m.margin) + m.pad));
    float top = A.y + std::max(m.margin, std::round(A.h * m.top_share));
    if (top + h > A.bottom() - m.margin) top = std::max(A.y + m.margin, A.bottom() - m.margin - h);
    return layout_in_window(in, m, measure, {x, std::round(top), w, h}, &bottom);
}

// ---- the component ------------------------------------------------------------------------

namespace {

// What the frame thread and a probe's thread share; the thread may outlive
// the settings.
struct ProbeShared {
    std::mutex m;
    bool done = false;
    acp::RunResult result;
    bool cancel = false;     // the settings went or the app quits: end the child
    std::int64_t pid = 0;    // the child's, while it runs and is not reaped
};

// Ends a probe's child and the processes it started, at once and without
// waiting; its thread reaps it.
void end_child(std::int64_t pid) {
#ifndef _WIN32
    if (pid <= 0) return;
    ::kill(-(pid_t)pid, SIGKILL);   // its own process group
    ::kill((pid_t)pid, SIGKILL);
#else
    (void)pid;
#endif
}

// The probe's child, on the probe's own thread: run_process()'s reading loop,
// which also ends the child when the settings ask it to. While it reads, the
// child is never reaped, so its pid, published in `shared`, names it alone
// and shutdown() may signal it from the frame thread; the pid is withdrawn
// before this thread reaps it.
void run_probe_child(const std::shared_ptr<ProbeShared>& shared, const acp::SpawnOptions& spawn,
                     int timeout_ms) {
    using clock = std::chrono::steady_clock;
    constexpr std::size_t kMaxOutput = 4u << 20;
    acp::RunResult r;
    std::unique_ptr<acp::Process> p = acp::Process::spawn(spawn, &r.error);
    if (p) {
        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(shared->m);
            shared->pid = p->pid();
            cancelled = shared->cancel;
        }
        p->close_stdin();
        const auto deadline = clock::now() + std::chrono::milliseconds(timeout_ms);
        bool out_open = true, err_open = true;
        std::vector<char> buf(16 * 1024);
        auto take = [&](acp::IoStatus st, std::size_t got, std::string* sink, bool* truncated, bool* open) {
            if (st == acp::IoStatus::ok) {
                const std::size_t room = kMaxOutput > sink->size() ? kMaxOutput - sink->size() : 0;
                sink->append(buf.data(), std::min(got, room));
                if (got > room) *truncated = true;
            } else if (st == acp::IoStatus::eof || st == acp::IoStatus::error) {
                *open = false;
            }
        };
        while (!cancelled && (out_open || err_open)) {
            if (clock::now() >= deadline) {
                r.timed_out = true;
                break;
            }
            std::size_t got = 0;
            if (out_open) {
                const acp::IoStatus st = p->read_stdout(buf.data(), buf.size(), &got, 50);
                take(st, got, &r.out, &r.out_truncated, &out_open);
            }
            if (err_open) {
                got = 0;
                const acp::IoStatus st = p->read_stderr(buf.data(), buf.size(), &got, out_open ? 0 : 50);
                take(st, got, &r.err, &r.err_truncated, &err_open);
            }
            std::lock_guard<std::mutex> lock(shared->m);
            cancelled = shared->cancel;
        }
        {
            std::lock_guard<std::mutex> lock(shared->m);
            shared->pid = 0;
            cancelled = cancelled || shared->cancel;
        }
        if (cancelled || r.timed_out) {
            r.status = p->terminate(2000);
        } else {
            // Both streams closed: the child is ending. Wait for it within the
            // deadline, and end it if the settings go meanwhile.
            while (!p->wait_for(50, &r.status)) {
                bool c = false;
                {
                    std::lock_guard<std::mutex> lock(shared->m);
                    c = shared->cancel;
                }
                if (c || clock::now() >= deadline) {
                    r.timed_out = !c;
                    r.status = p->terminate(2000);
                    break;
                }
            }
        }
    }
    std::lock_guard<std::mutex> lock(shared->m);
    shared->result = std::move(r);
    shared->done = true;
}

}  // namespace

struct AgentSettings::Runner {
    std::shared_ptr<ProbeShared> shared;
    bool list_only = false;
    std::string list_key;   // the preset and base URL a list belongs to
    std::string for_key;    // the base URL and model a probe belongs to
};

AgentSettings::AgentSettings() = default;

AgentSettings::~AgentSettings() { shutdown(); }

void AgentSettings::start(Options o) {
    options_ = std::move(o);
    std::string why;
    if (!read_agent_settings(options_.settings_path, &data_, &why) && options_.log) options_.log("settings", "widget agent", why);
}

void AgentSettings::set_open(bool open) {
    if (open && !open_) focus_ = true;
    open_ = open;
    if (!open) cancel_key();
}

void AgentSettings::changed() {
    save();
    bump();
}

void AgentSettings::bump() {
    ++revision_;
    if (on_change_) on_change_();
}

void AgentSettings::save() {
    if (options_.settings_path.empty()) return;
    std::string why;
    if (!write_agent_settings(options_.settings_path, data_, &why) && options_.log) options_.log("settings", "widget agent", why);
}

void AgentSettings::choose_preset(const std::string& id) {
    if (!agent::find_preset(id) || data_.preset == id) return;
    cancel_key();
    data_.preset = id;
    data_.providers[id];   // remembered from now on
    answer_ = {};
    changed();
    if (options_.log) options_.log("provider", agent::find_preset(id)->name, "widget agent");
}

void AgentSettings::set_base_url(std::string url) {
    const AgentChoice c = choice();
    if (!c.preset || !c.base_url_editable) return;
    if (url.size() > 2048) url.resize(2048);
    if (data_.providers[data_.preset].base_url == url) return;
    data_.providers[data_.preset].base_url = std::move(url);
    answer_ = {};
    changed();
}

void AgentSettings::set_model(std::string model) {
    if (data_.preset.empty()) return;
    if (model.size() > 256) model.resize(256);
    if (data_.providers[data_.preset].model == model) return;
    data_.providers[data_.preset].model = std::move(model);
    answer_ = {};
    changed();
}

void AgentSettings::set_effort(std::string effort) {
    const AgentChoice c = choice();
    if (!c.preset || std::find(c.preset->efforts.begin(), c.preset->efforts.end(), effort) == c.preset->efforts.end())
        return;
    if (data_.providers[data_.preset].effort == effort) return;
    data_.providers[data_.preset].effort = std::move(effort);
    changed();
}

std::string AgentSettings::cache_note() const { return agent_cache_note(choice(), capabilities()); }

std::string AgentSettings::gpu_note() const {
    return agent_gpu_note(options_.live_attached && options_.live_attached(), choice());
}

AgentSettings::KeyStatus AgentSettings::key_status() const {
    const AgentChoice c = choice();
    // No row for a local server, nor for Custom before its address is valid
    // (its key is named by the address).
    if (!c.preset || c.key == agent::KeyNeed::none || c.account.empty()) return KeyStatus::none;
    if (!options_.keys) return KeyStatus::unknown;
    if (key_revision_seen_ != revision_) {   // the store is asked again after a change only
        const agent::KeyPresence p = options_.keys->has(c.account);
        key_status_ = p.error ? KeyStatus::unknown : p.present ? KeyStatus::set : KeyStatus::not_set;
        key_revision_seen_ = revision_;
    }
    return key_status_;
}

void AgentSettings::begin_key_entry() {
    if (key_status() == KeyStatus::none || key_status() == KeyStatus::unknown) return;
    agent::wipe(&key_text_);
    key_problem_.clear();
    key_editing_ = true;
}

void AgentSettings::set_key_text(std::string text) {
    agent::wipe(&key_text_);
    key_text_ = std::move(text);
    agent::wipe(&text);
}

bool AgentSettings::save_key() {
    const AgentChoice c = choice();
    if (!options_.keys || c.account.empty() || key_text_.empty()) return false;
    const caliper_builder::Error e = options_.keys->set(c.account, key_text_);
    agent::wipe(&key_text_);
    if (e) {
        key_problem_ = e.code == "keystore.bad_key" ? "Malformed key" : "Not saved";
        return false;
    }
    key_problem_.clear();
    key_editing_ = false;
    if (options_.log) options_.log("key", c.preset ? c.preset->name : c.account, "set");
    bump();   // the key's row and the widget agent; nothing is written to the settings file
    return true;
}

void AgentSettings::cancel_key() {
    agent::wipe(&key_text_);
    key_editing_ = false;
}

bool AgentSettings::remove_key() {
    const AgentChoice c = choice();
    if (!options_.keys || c.account.empty()) return false;
    cancel_key();
    const caliper_builder::Error e = options_.keys->remove(c.account);
    if (e) {
        key_problem_ = "Not removed";
        return false;
    }
    if (options_.log) options_.log("key", c.preset ? c.preset->name : c.account, "removed");
    bump();   // the widget agent's sidecar, which holds the key, ends now
    return true;
}

std::vector<std::string> AgentSettings::model_choices() const {
    const AgentChoice c = choice();
    std::vector<std::string> out;
    if (!c.preset) return out;
    if (!c.preset->models.empty()) {
        for (const auto& m : c.preset->models) out.push_back(m.first);
        return out;
    }
    const auto it = listed_.find(c.preset->id + " " + c.base_url);
    return it == listed_.end() ? out : it->second;
}

bool AgentSettings::busy() const {
    if (!runner_ || !runner_->shared) return false;
    std::lock_guard<std::mutex> lock(runner_->shared->m);
    return !runner_->shared->done;
}

void AgentSettings::run_probe(std::vector<std::string> extra, bool list_only) {
    if (busy()) return;
    const AgentChoice c = choice();
    if (!c.preset) return;
    if (options_.python.empty() || options_.source.empty()) {
        answer_ = {};
        answer_.failure = "No caliper-agent";
        return;
    }
    if (c.preset->api != "claude" && !agent_base_url_valid(c.base_url)) return;
    acp::SpawnOptions spawn;
    namespace fs = std::filesystem;
    const fs::path source(options_.source);
    spawn.argv = {options_.python, "-P", "-m", "caliper_agent", "probe", "--preset", c.preset->id};
    if (c.base_url_editable) spawn.argv.insert(spawn.argv.end(), {"--base-url", c.base_url});
    const std::string model = model_for_cache(c);
    spawn.argv.insert(spawn.argv.end(), {"--model", model.empty() ? std::string("any") : model});
    if (!options_.cache_dir.empty()) spawn.argv.insert(spawn.argv.end(), {"--cache-dir", options_.cache_dir});
    for (auto& a : extra) spawn.argv.push_back(std::move(a));
    spawn.env = {{"PYTHONPATH", (source / "agent").string()}, {"PYTHONDONTWRITEBYTECODE", "1"}};
    for (const auto& v : options_.probe_env) spawn.env.push_back(v);
    spawn.env_remove = options_.probe_env_remove;
    // The child's environment is the sidecar's: the provider's key, placed by
    // the store, and none of the key or endpoint variables of the app's own
    // environment; the offline switch passes through. When the store cannot
    // place a key, or a key is required and there is no store, the probe
    // does not run, as the sidecar's start does not (AcpWidgetAgent::start()).
    const std::string id = placement_id(c);
    if (options_.keys && !id.empty()) {
        if (const caliper_builder::Error e = agent::place_key_in_environment(*options_.keys, id, &spawn)) {
            answer_ = {};
            answer_.failure = key_failure(e);
            return;
        }
    } else if (c.key == agent::KeyNeed::required) {
        answer_ = {};
        answer_.failure = "No key";
        return;
    } else {
        agent::strip_key_environment(&spawn);
    }
    runner_ = std::make_unique<Runner>();
    runner_->shared = std::make_shared<ProbeShared>();
    runner_->list_only = list_only;
    runner_->list_key = list_key(c);
    runner_->for_key = probe_key(c);
    const int timeout = options_.probe_timeout_ms;
    std::thread([shared = runner_->shared, spawn, timeout] { run_probe_child(shared, spawn, timeout); }).detach();
}

void AgentSettings::list_models() {
    const AgentChoice c = choice();
    if (!c.preset || !c.preset->models.empty()) return;   // the preset's own list
    run_probe({"--list-models"}, true);
}

void AgentSettings::probe(bool refresh) {
    const AgentChoice c = choice();
    if (!c.preset || model_for_cache(c).empty()) return;
    run_probe(refresh ? std::vector<std::string>{"--refresh"} : std::vector<std::string>{}, false);
}

void AgentSettings::set_override(const std::string& name, const std::string& value) {
    const AgentChoice c = choice();
    if (!c.preset || model_for_cache(c).empty()) return;
    if (name != "tools" && name != "vision" && name != "context_length") return;
    run_probe({"--set", name + "=" + value}, false);
}

void AgentSettings::frame() {
    if (!runner_ || !runner_->shared) return;
    acp::RunResult r;
    {
        std::lock_guard<std::mutex> lock(runner_->shared->m);
        if (!runner_->shared->done) return;
        r = std::move(runner_->shared->result);
    }
    std::unique_ptr<Runner> done = std::move(runner_);
    AgentProbeAnswer a = parse_probe_answer(r.out);
    if (!a.ran) a.failure = r.timed_out ? "Timeout" : "No answer";
    // An answer for a choice that is no longer the current one (the model or
    // the address changed while it ran) leaves the tags as they are; a list
    // is still kept under its own preset and address.
    const AgentChoice c = choice();
    if (done->list_only) {
        if (a.ran) listed_[done->list_key] = a.models;
        if (done->list_key == list_key(c) && (!a.problems.empty() || !a.failure.empty())) {
            answer_.problems = a.problems;
            answer_.failure = a.failure;
        }
    } else if (done->for_key == probe_key(c)) {
        answer_ = std::move(a);
        answer_for_ = done->for_key;
        bump();   // the widget agent reads the cache again
    }
}

AgentCapabilities AgentSettings::capabilities() const {
    const AgentChoice c = choice();
    if (answer_.ran && answer_for_ == probe_key(c)) return answer_.caps;
    if (caps_revision_seen_ != revision_) {   // the cache file is read again after a change only
        const double now = options_.now ? options_.now()
                                        : (double)std::chrono::duration_cast<std::chrono::seconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count();
        caps_cached_ = read_cached_capabilities(options_.cache_dir, base_for_cache(c), model_for_cache(c), now);
        // Every model the Claude preset offers calls tools and reads images,
        // as the sidecar assumes without a probe.
        if (c.preset && c.preset->api == "claude") {
            if (!caps_cached_.tools) caps_cached_.tools = true, caps_cached_.tools_source = "known";
            if (!caps_cached_.vision) caps_cached_.vision = true, caps_cached_.vision_source = "known";
        }
        // A preset that loads the model with a context length of its own
        // (Ollama) has that one until the person sets another, as the sidecar.
        if (c.preset && c.preset->context_length > 0 && caps_cached_.context_source != "override")
            caps_cached_.context = c.preset->context_length, caps_cached_.context_source = "preset";
        caps_revision_seen_ = revision_;
    }
    return caps_cached_;
}

std::string AgentSettings::state() const {
    const AgentChoice c = choice();
    if (!c.preset) return "No provider";
    if (c.preset->api != "claude") {
        if (!agent_base_url_valid(c.base_url)) return "Invalid address";
        if (c.model.empty()) return "No model";
    }
    if (c.key == agent::KeyNeed::required && key_status() != KeyStatus::set) return "No key";
    const AgentCapabilities caps = capabilities();
    if (caps.tools && !*caps.tools) return "No tool calling";
    return {};
}

namespace {

std::string yes_no(const std::optional<bool>& v) { return !v ? "Unknown" : *v ? "Yes" : "No"; }

std::string tokens_text(const std::optional<long long>& v) {
    if (!v) return "Unknown";
    std::string digits = std::to_string(*v), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out + " tokens";
}

std::string source_text(const std::string& s) {
    if (s == "probe") return "probe";
    if (s == "server") return "server";
    if (s == "known") return "known";
    if (s == "override") return "override";
    if (s == "default") return "assumed";
    if (s == "preset") return "preset";
    return "unknown";
}

const char* key_status_text(AgentSettings::KeyStatus s) {
    switch (s) {
        case AgentSettings::KeyStatus::set: return "Set";
        case AgentSettings::KeyStatus::not_set: return "Not set";
        case AgentSettings::KeyStatus::unknown: return "Unknown";
        case AgentSettings::KeyStatus::none: return "";
    }
    return "";
}

std::string model_name(const AgentChoice& c, const std::string& id) {
    if (c.preset)
        for (const auto& m : c.preset->models)
            if (m.first == id) return m.second;
    return id;
}

}  // namespace

std::vector<std::string> AgentSettings::visible_text() const {
    const AgentChoice c = choice();
    std::vector<std::string> out = {"Widget agent", "esc Close", "Provider", "Base URL", "Model", "List"};
    for (const auto& p : agent::provider_presets()) out.push_back(p.name);
    const std::string note = gpu_note();
    if (!note.empty()) out.push_back(note);
    if (c.preset) out.push_back(c.preset->api == "claude" ? c.preset->base_url : c.base_url);
    for (const auto& w : warnings()) out.push_back(w);
    out.push_back(c.model.empty() && c.preset && c.preset->api == "claude" ? model_name(c, kClaudeWidgetModel)
                                                                            : model_name(c, c.model));
    for (const auto& m : model_choices()) out.push_back(model_name(c, m));
    if (c.preset && !c.preset->efforts.empty()) {
        out.push_back("Effort");
        for (const auto& e : c.preset->efforts) out.push_back(e);
    }
    const KeyStatus ks = key_status();
    if (ks != KeyStatus::none) {
        out.push_back("Key");
        out.push_back(key_status_text(ks));
        if (key_editing_ || ks == KeyStatus::not_set) out.push_back("Save");
        if (key_editing_ && ks == KeyStatus::set) out.push_back("Cancel");
        if (!key_editing_ && ks == KeyStatus::set) {
            out.push_back("Replace");
            out.push_back("Remove");
        }
        if (!key_problem_.empty()) out.push_back(key_problem_);
    }
    const AgentCapabilities caps = capabilities();
    for (const char* s : {"Capabilities", "Tool calling", "Vision", "Context length", "Yes", "No", "Auto", "Probe"})
        out.push_back(s);
    if (has_permissions()) {
        for (const char* s :
             {"Permissions", "Side chat", "MCP clients", "Ask", "Accept edits", "Read only", "Allow all", "Saved rules"})
            out.push_back(s);
        const auto rules = permissions_.rules();
        for (const auto& r : rules) out.push_back((r.first == "mcp" ? std::string("MCP clients") : r.first) + " " + r.second);
        if (rules.empty()) out.push_back("None");
        if (permissions_.note)
            if (const std::string note = permissions_.note(); !note.empty()) out.push_back(note);
    }
    out.push_back(yes_no(caps.tools) + " " + source_text(caps.tools_source));
    out.push_back(yes_no(caps.vision) + " " + source_text(caps.vision_source));
    out.push_back(tokens_text(caps.context) + " " + source_text(caps.context_source));
    for (const auto& p : answer_.problems) out.push_back(p);
    if (!answer_.failure.empty()) out.push_back(answer_.failure);
    const std::string st = state();
    if (!st.empty()) out.push_back(st);
    const std::string cache = cache_note();
    if (!cache.empty()) out.push_back(cache);
    return out;
}

AgentSettingsLayoutInput AgentSettings::layout_input(const WidgetRect& area) const {
    const AgentChoice c = choice();
    AgentSettingsLayoutInput in;
    in.area = area;
    for (const auto& p : agent::provider_presets()) in.presets.push_back(p.name);
    in.base_url_field = c.base_url_editable;
    in.warnings = (int)warnings().size();
    for (const auto& m : model_choices()) in.models.push_back(model_name(c, m));
    if (c.preset) in.efforts = c.preset->efforts;
    const KeyStatus ks = key_status();
    in.key_row = ks != KeyStatus::none;
    in.key_set = ks == KeyStatus::set;
    in.key_field = in.key_row && (key_editing_ || ks == KeyStatus::not_set);
    in.note = !gpu_note().empty();
    in.problems = (int)answer_.problems.size() + (answer_.failure.empty() ? 0 : 1);
    in.state = !state().empty();
    in.cache_note = cache_note();
    if (has_permissions()) {
        in.permissions = true;
        in.rules = (int)permissions_.rules().size();
        in.perm_note = permissions_.note && !permissions_.note().empty();
    }
    return in;
}

void AgentSettings::shutdown() {
    cancel_key();
    if (runner_ && runner_->shared) {
        // A probe still running: its child, which may hold the key in its
        // environment, ends now; its thread reaps it. Nothing here waits.
        std::lock_guard<std::mutex> lock(runner_->shared->m);
        runner_->shared->cancel = true;
        end_child(runner_->shared->pid);
    }
    runner_.reset();
}

// ---- the ImGui side -----------------------------------------------------------------------

namespace {

int resize_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = &(*s)[0];
    }
    return 0;
}

ImVec2 extent(ImFont* font, float size, const std::string& t) {
    if (!font) font = ImGui::GetFont();
    return font->CalcTextSizeA(size, FLT_MAX, 0.0f, t.c_str());
}

void text_at(ImDrawList* dl, ImFont* font, float size, ImVec2 p, ImU32 col, const std::string& t) {
    if (!font) font = ImGui::GetFont();
    dl->AddText(font, size, ImVec2(std::round(p.x), std::round(p.y)), col, t.c_str());
}

// A button in the house's look: a 1 px outline, bronze when lit.
bool house_button(const SpecLook& L, ImDrawList* dl, const WidgetRect& r, const char* id, const std::string& label,
                  bool enabled = true, bool lit = false) {
    if (r.w < 1.0f) return false;
    ImGui::SetCursorScreenPos(ImVec2(r.x, r.y));
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(r.w, r.h)) && enabled;
    const bool hovered = enabled && ImGui::IsItemHovered();
    if (hovered) dl->AddRectFilled(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom()), L.plate_hover);
    dl->AddRect(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom()), lit ? L.accent : enabled ? L.rule : L.rule_faint, 0.0f, 0,
                1.0f);
    const ImVec2 e = extent(L.ui, L.ui_size, label);
    text_at(dl, L.ui, L.ui_size, ImVec2(r.x + (r.w - e.x) * 0.5f, r.y + (r.h - e.y) * 0.5f),
            !enabled ? L.faint : lit ? L.text : L.secondary, label);
    return clicked;
}

void tag(const SpecLook& L, ImDrawList* dl, const WidgetRect& r, const std::string& label, ImU32 col) {
    if (r.w < 1.0f) return;
    dl->AddRect(ImVec2(r.x, r.y), ImVec2(r.right(), r.bottom()), col, 0.0f, 0, 1.0f);
    const ImVec2 e = extent(L.ui, L.small_size, label);
    text_at(dl, L.ui, L.small_size, ImVec2(r.x + (r.w - e.x) * 0.5f, r.y + (r.h - e.y) * 0.5f), col, label);
}

bool field(const SpecLook& L, const WidgetRect& r, const char* id, std::string* value, ImGuiInputTextFlags flags = 0) {
    if (r.w < 1.0f) return false;
    ImGui::SetCursorScreenPos(ImVec2(r.x, r.y));
    ImGui::PushItemWidth(r.w);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::ColorConvertU32ToFloat4(L.deep));
    ImGui::PushStyleColor(ImGuiCol_Border, ImGui::ColorConvertU32ToFloat4(L.rule));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(std::round(8.0f * L.text_scale),
                                                           std::max(0.0f, (r.h - L.ui_size) * 0.5f)));
    if (L.ui) ImGui::PushFont(L.ui, L.ui_size);
    if (value->capacity() < 64) value->reserve(64);
    const bool done = ImGui::InputText(id, &(*value)[0], value->capacity() + 1,
                                       flags | ImGuiInputTextFlags_CallbackResize, resize_callback, value);
    if (L.ui) ImGui::PopFont();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    ImGui::PopItemWidth();
    return done;
}

}  // namespace

void AgentSettings::read_keys() {
    // ImGuiMod_Ctrl is the Command key on macOS, as the page's ⌘K reads it.
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Comma)) set_open(!open_);
}

void AgentSettings::draw(float x, float y, float w, float h, const SpecLook& L) {
    if (!open_) return;
    const AgentSettingsMetrics m = AgentSettingsMetrics::for_text_scale(L.text_scale);
    const WidgetMeasure measure = [&L](const std::string& t, WidgetFont f) {
        return f == WidgetFont::mono ? extent(L.mono, L.mono_size, t).x : extent(L.ui, L.ui_size, t).x;
    };
    const AgentSettingsLayout lay = layout_agent_settings(layout_input({x, y, w, h}), m, measure);
    const WidgetRect& W = lay.window;
    ImGui::SetNextWindowPos(ImVec2(W.x, W.y));
    ImGui::SetNextWindowSize(ImVec2(W.w, W.h));
    if (focus_) ImGui::SetNextWindowFocus();
    focus_ = false;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::ColorConvertU32ToFloat4(L.ground));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                   ImGuiWindowFlags_NoScrollWithMouse;
    const bool shown = ImGui::Begin("##widget.agent.settings", nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    if (!shown) {
        ImGui::End();
        return;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRect(ImVec2(W.x, W.y), ImVec2(W.right(), W.bottom()), L.rule, 0.0f, 0, 1.0f);
    const AgentChoice c = choice();
    const float ty = (m.row - extent(L.ui, L.ui_size, "Ag").y) * 0.5f;
    auto label = [&](const WidgetRect& r, const char* t) { text_at(dl, L.ui, L.ui_size, ImVec2(r.x, r.y + ty), L.faint, t); };

    // The title strip.
    text_at(dl, L.ui, L.ui_size, ImVec2(lay.title.x, lay.title.y + ty), L.text, "Widget agent");
    if (house_button(L, dl, lay.close, "##settings.close", "esc Close")) set_open(false);
    dl->AddLine(ImVec2(W.x + m.pad, lay.title.bottom() + m.gap), ImVec2(W.right() - m.pad, lay.title.bottom() + m.gap),
                L.rule_faint);

    // Provider.
    label({W.x + m.pad, lay.presets.empty() ? 0 : lay.presets[0].y, m.label, m.row}, "Provider");
    const auto& presets = agent::provider_presets();
    for (size_t i = 0; i < presets.size() && i < lay.presets.size(); ++i) {
        const bool lit = c.preset && c.preset->id == presets[i].id;
        if (house_button(L, dl, lay.presets[i], ("##preset." + presets[i].id).c_str(), presets[i].name, true, lit))
            choose_preset(presets[i].id);
    }
    if (!lay.note.empty()) tag(L, dl, lay.note, gpu_note(), L.accent);

    // Base URL.
    label(lay.base_url_label, "Base URL");
    if (c.preset) {
        if (c.base_url_editable) {
            if (!base_url_active_) base_url_edit_ = c.base_url;
            const bool entered = field(L, lay.base_url, "##settings.base_url", &base_url_edit_,
                                       ImGuiInputTextFlags_EnterReturnsTrue);
            base_url_active_ = ImGui::IsItemActive();
            if (entered || ImGui::IsItemDeactivatedAfterEdit()) set_base_url(base_url_edit_);
        } else {
            text_at(dl, L.mono, L.mono_size, ImVec2(lay.base_url.x, lay.base_url.y + ty), L.secondary,
                    c.preset->api == "claude" ? c.preset->base_url : c.base_url);
        }
    }
    const std::vector<std::string> warns = warnings();
    for (size_t i = 0; i < warns.size() && i < lay.warnings.size(); ++i) tag(L, dl, lay.warnings[i], warns[i], L.accent);

    // Model.
    label(lay.model_label, "Model");
    if (c.preset) {
        if (!model_active_) model_edit_ = c.model;
        const bool entered = field(L, lay.model, "##settings.model", &model_edit_, ImGuiInputTextFlags_EnterReturnsTrue);
        model_active_ = ImGui::IsItemActive();
        if (entered || ImGui::IsItemDeactivatedAfterEdit()) set_model(model_edit_);
        const bool can_list = c.preset->models.empty() && !busy() && !options_.python.empty();
        if (house_button(L, dl, lay.list, "##settings.list", "List", can_list)) list_models();
        const std::vector<std::string> choices = model_choices();
        for (size_t i = 0; i < choices.size() && i < lay.models.size(); ++i) {
            const bool lit = choices[i] == c.model || (c.model.empty() && choices[i] == kClaudeWidgetModel &&
                                                       c.preset->api == "claude");
            if (house_button(L, dl, lay.models[i], ("##model." + std::to_string(i)).c_str(), model_name(c, choices[i]),
                             true, lit))
                set_model(choices[i]);
        }
        if (lay.models_folded > 0 && !lay.models.empty()) {
            const WidgetRect& last = lay.models.back();
            text_at(dl, L.ui, L.small_size, ImVec2(last.right() + m.gap, last.y + ty), L.faint,
                    "+" + std::to_string(lay.models_folded));
        }
    }

    // Effort.
    if (c.preset && !c.preset->efforts.empty()) {
        label(lay.effort_label, "Effort");
        for (size_t i = 0; i < c.preset->efforts.size() && i < lay.efforts.size(); ++i) {
            const std::string& e = c.preset->efforts[i];
            const bool lit = c.effort == e || (c.effort.empty() && e == "low");
            if (house_button(L, dl, lay.efforts[i], ("##effort." + e).c_str(), e, true, lit)) set_effort(e);
        }
    }

    // Key: write only.
    const KeyStatus ks = key_status();
    if (ks != KeyStatus::none) {
        label(lay.key_label, "Key");
        text_at(dl, L.ui, L.ui_size, ImVec2(lay.key_status.x, lay.key_status.y + ty),
                ks == KeyStatus::set ? L.text : L.faint, key_status_text(ks));
        if (!lay.key_field.empty()) {
            const bool entered = field(L, lay.key_field, "##settings.key", &key_text_,
                                       ImGuiInputTextFlags_Password | ImGuiInputTextFlags_EnterReturnsTrue);
            if (!key_editing_ && ImGui::IsItemActive()) key_editing_ = true;
            if ((house_button(L, dl, lay.key_buttons.empty() ? WidgetRect{} : lay.key_buttons[0], "##key.save", "Save",
                              !key_text_.empty()) ||
                 entered) &&
                !key_text_.empty())
                save_key();
            if (lay.key_buttons.size() > 1 && house_button(L, dl, lay.key_buttons[1], "##key.cancel", "Cancel"))
                cancel_key();
        } else if (lay.key_buttons.size() == 2) {
            if (house_button(L, dl, lay.key_buttons[0], "##key.replace", "Replace")) begin_key_entry();
            if (house_button(L, dl, lay.key_buttons[1], "##key.remove", "Remove")) remove_key();
        }
        if (!key_problem_.empty())
            text_at(dl, L.ui, L.small_size, ImVec2(lay.key_status.x, lay.key_status.bottom() + 2), L.accent, key_problem_);
    }

    // Capabilities, beside the model.
    const AgentCapabilities caps = capabilities();
    text_at(dl, L.ui, L.small_size, ImVec2(lay.caps_title.x, lay.caps_title.y), L.faint, "Capabilities");
    const bool can_probe = c.preset && !busy() && !options_.python.empty() && !model_for_cache(c).empty();
    // A row's name, then its value and, where it fits before the row's first
    // control, where the value came from.
    const float name_w = extent(L.ui, L.ui_size, "Context length").x + m.gap * 2;
    auto value_text = [&](const WidgetRect& row, float end, const std::string& value, const std::string& source) {
        const float vx = row.x + name_w;
        const float vw = extent(L.ui, L.ui_size, value).x;
        if (vx + vw > end) return;
        text_at(dl, L.ui, L.ui_size, ImVec2(vx, row.y + ty), L.text, value);
        const float sx = vx + vw + m.gap;
        if (source != "unknown" && sx + extent(L.ui, L.small_size, source).x <= end)
            text_at(dl, L.ui, L.small_size, ImVec2(sx, row.y + ty), L.faint, source);
    };
    auto cap_row = [&](const WidgetRect& row, const std::vector<WidgetRect>& buttons, const char* name,
                       const std::string& value, const std::string& source, const char* key, const json& over) {
        text_at(dl, L.ui, L.ui_size, ImVec2(row.x, row.y + ty), L.secondary, name);
        value_text(row, (buttons.empty() ? row.right() : buttons[0].x) - m.gap, value, source);
        const std::string current = over.is_boolean() ? (over.get<bool>() ? "yes" : "no") : "auto";
        const char* values[] = {"yes", "no", "auto"};
        const char* labels[] = {"Yes", "No", "Auto"};
        for (size_t i = 0; i < 3 && i < buttons.size(); ++i)
            if (house_button(L, dl, buttons[i], (std::string("##") + key + "." + values[i]).c_str(), labels[i], can_probe,
                             current == values[i]))
                set_override(key, values[i]);
    };
    cap_row(lay.tools, lay.tools_buttons, "Tool calling", yes_no(caps.tools), source_text(caps.tools_source), "tools",
            caps.overrides.value("tools", json()));
    cap_row(lay.vision, lay.vision_buttons, "Vision", yes_no(caps.vision), source_text(caps.vision_source), "vision",
            caps.overrides.value("vision", json()));
    {
        const WidgetRect& row = lay.context;
        text_at(dl, L.ui, L.ui_size, ImVec2(row.x, row.y + ty), L.secondary, "Context length");
        value_text(row, lay.context_field.x - m.gap, tokens_text(caps.context), source_text(caps.context_source));
        if (field(L, lay.context_field, "##settings.context", &context_text_,
                  ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_EnterReturnsTrue) &&
            !context_text_.empty() && can_probe)
            set_override("context_length", context_text_);
        if (house_button(L, dl, lay.context_auto, "##context.auto", "Auto", can_probe,
                         !caps.overrides.contains("context_length")))
            set_override("context_length", "auto");
    }
    if (house_button(L, dl, lay.probe, "##settings.probe", busy() ? "Probing" : "Probe", can_probe)) probe(true);
    {
        std::vector<std::string> problems = answer_.problems;
        if (!answer_.failure.empty()) problems.push_back(answer_.failure);
        for (size_t i = 0; i < problems.size() && i < lay.problems.size(); ++i) tag(L, dl, lay.problems[i], problems[i], L.faint);
    }
    const std::string st = state();
    if (!lay.state.empty() && !st.empty()) tag(L, dl, lay.state, st, L.accent);
    const std::string cache = cache_note();
    if (!lay.cache_note.empty() && !cache.empty()) tag(L, dl, lay.cache_note, cache, L.accent);

    // Permissions: the two modes and the saved rules, each with Revoke.
    // Allow all is the last mode of each row.
    if (!lay.perm_title.empty() && has_permissions()) {
        dl->AddLine(ImVec2(lay.perm_title.x, lay.perm_title.y - m.gap),
                    ImVec2(lay.perm_title.right(), lay.perm_title.y - m.gap), L.rule_faint);
        text_at(dl, L.ui, L.small_size, ImVec2(lay.perm_title.x, lay.perm_title.y), L.faint, "Permissions");
        if (!lay.perm_note.empty() && permissions_.note) {
            const std::string note = permissions_.note();
            dl->AddText(L.ui ? L.ui : ImGui::GetFont(), L.ui_size, ImVec2(lay.perm_note.x, lay.perm_note.y), L.accent,
                        note.c_str(), nullptr, lay.perm_note.w);
        }
        const char* names[] = {"ask", "accept_edits", "read_only", "allow_all"};
        const char* words[] = {"Ask", "Accept edits", "Read only", "Allow all"};
        auto modes = [&](const WidgetRect& lr, const char* text, const std::vector<WidgetRect>& buttons, const char* who) {
            label(lr, text);
            const std::string now = permissions_.mode(who);
            for (size_t i = 0; i < 4 && i < buttons.size(); ++i)
                if (house_button(L, dl, buttons[i], (std::string("##perm.") + who + "." + names[i]).c_str(), words[i], true,
                                 now == names[i]))
                    permissions_.set_mode(who, names[i]);
        };
        modes(lay.chat_label, "Side chat", lay.chat_modes, "chat");
        modes(lay.mcp_label, "MCP clients", lay.mcp_modes, "mcp");
        label(lay.rules_label, "Saved rules");
        const auto rules = permissions_.rules();
        for (size_t i = 0; i < rules.size() && i < lay.rules.size(); ++i) {
            const std::string agent = rules[i].first == "mcp" ? "MCP clients" : rules[i].first;
            text_at(dl, L.ui, L.ui_size, ImVec2(lay.rules[i].x, lay.rules[i].y + ty), L.secondary, agent);
            const float ax = lay.rules[i].x + extent(L.ui, L.ui_size, agent).x + m.gap;
            text_at(dl, L.mono, L.mono_size, ImVec2(ax, lay.rules[i].y + ty), L.text, rules[i].second);
            if (i < lay.revokes.size() && permissions_.revoke &&
                house_button(L, dl, lay.revokes[i], ("##perm.revoke." + std::to_string(i)).c_str(), "Revoke"))
                permissions_.revoke(rules[i].first, rules[i].second);
        }
        if (!lay.rules_more.empty())
            text_at(dl, L.ui, L.ui_size, ImVec2(lay.rules_more.x, lay.rules_more.y + ty), L.faint,
                    rules.empty() ? std::string("None") : "+" + std::to_string(rules.size() - lay.rules.size()));
    }

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        !ImGui::GetIO().WantTextInput)
        set_open(false);
    ImGui::End();
}

}  // namespace caliper_host
