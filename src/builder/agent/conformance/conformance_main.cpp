// caliper-agent-conformance: the widget agent's provider conformance suite
// (design/widget-agent.md, The provider conformance suite).
//
// It builds the fixture workspace (a Parquet table run42_metrics, a
// safetensors tensor attn_weights and the spec dashboard.spec.toml open),
// opens a builder session on it behind the stand-in's command server, and runs
// each task of tasks.json (embedded at build time) as the app would: it begins
// a stacked preview of the target, builds the context bundle with
// build_context_bundle(), starts caliper-agent through the real AcpClient,
// opens a widget session with the bundle's _meta, sends the bundle and the
// request in one prompt, and then checks the preview structurally against the
// open spec (the spec as JSON, and the resolved data of a new panel) with the
// task's check, before it discards the preview.
//
// Modes:
//   --scripted pass|fail   the sidecar answers from each task's scripted
//                          answer (Pydantic AI's FunctionModel behind its test
//                          switch), to test the suite itself; no model.
//   --preset ID / --provider ... --base-url URL --model ID
//                          a provider and model. A server on a loopback
//                          address runs with the sidecar's offline switch, so
//                          nothing else is reached; a hosted provider needs
//                          --real, which is the one path without the switch,
//                          and its key from this process's environment
//                          (ANTHROPIC_API_KEY or CALIPER_MODEL_API_KEY),
//                          handed to the sidecar by place_key_in_environment().
//                          --real also needs CALIPER_CONFORMANCE_REAL=1, set
//                          by the person who runs it, and is refused in a test
//                          run: when this process has the offline switch
//                          (every ctest entry does) or runs under ctest or
//                          pytest (real_refusal()).
// It first runs `caliper-agent probe` for the capabilities (not in the
// scripted mode); a model without tool calling is recorded as unable to drive
// Caliper, and the vision task is not applicable to a model without vision.
//
// It writes conformance-<date>-<model>.json and .md into --out: the provider,
// the model, the date, the capabilities, each task's runs and outcome, and the
// verdict, and nothing secret (the key's text is checked to be absent).
// Exit codes: 0 when the run finished, whatever its verdict; 1 when the
// workspace or the sidecar could not be set up; 2 for bad usage.
#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/agent/key_store.h>
#include <caliper_builder/conformance_tasks_json.h>
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/context_bundle.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#include <duckdb.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_builder;

namespace {

constexpr const char* kSuiteName = "Caliper widget agent conformance";
constexpr int kSuiteVersion = 1;
constexpr const char* kFakeKey = "sk-ant-api03-TESTONLY-conformance-0123456789abcdef-fakeAA";
constexpr int kRows = 200;
const std::vector<int64_t> kTensorShape = {4, 8, 16, 16};

// The colours of the vision task, as the probe's (caliper_agent/probe.py).
struct Colour {
    const char* name;
    unsigned char rgb[3];
};
const Colour kColours[] = {{"red", {220, 30, 30}}, {"green", {30, 170, 40}}, {"blue", {30, 60, 220}},
                           {"yellow", {240, 215, 20}}};

// ---- small helpers ----------------------------------------------------------------------------------

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool write_file(const std::string& path, const std::string& text) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return static_cast<bool>(out);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

std::string slug(const std::string& s) {
    std::string out;
    for (char c : lower(s)) out += (std::isalnum(static_cast<unsigned char>(c)) ? c : '-');
    while (out.find("--") != std::string::npos) out.replace(out.find("--"), 2, "-");
    return out.substr(0, 64);
}

std::string base64(const std::string& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8) |
                           static_cast<unsigned char>(in[i + 2]);
        out += {t[(v >> 18) & 63], t[(v >> 12) & 63], t[(v >> 6) & 63], t[v & 63]};
    }
    if (i + 1 == in.size()) {
        const unsigned v = static_cast<unsigned char>(in[i]) << 16;
        out += {t[(v >> 18) & 63], t[(v >> 12) & 63], '=', '='};
    } else if (i + 2 == in.size()) {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8);
        out += {t[(v >> 18) & 63], t[(v >> 12) & 63], t[(v >> 6) & 63], '='};
    }
    return out;
}

std::uint32_t crc32(const std::string& data) {
    std::uint32_t c = 0xFFFFFFFFu;
    for (unsigned char b : data) {
        c ^= b;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

std::string be32(std::uint32_t v) {
    return {static_cast<char>(v >> 24), static_cast<char>(v >> 16), static_cast<char>(v >> 8), static_cast<char>(v)};
}

// A PNG of one solid colour, with an uncompressed (stored) zlib stream.
std::string solid_png(const unsigned char rgb[3], int width, int height) {
    std::string raw;
    for (int y = 0; y < height; ++y) {
        raw += '\0';
        for (int x = 0; x < width; ++x) raw.append(reinterpret_cast<const char*>(rgb), 3);
    }
    std::string z = "\x78\x01";
    std::uint32_t a = 1, b = 0;
    for (unsigned char c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    for (std::size_t off = 0; off < raw.size() || off == 0; off += 65535) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - off);
        const bool last = off + n >= raw.size();
        z += static_cast<char>(last ? 1 : 0);
        z += {static_cast<char>(n & 0xFF), static_cast<char>(n >> 8), static_cast<char>(~n & 0xFF),
              static_cast<char>((~n >> 8) & 0xFF)};
        z += raw.substr(off, n);
        if (last) break;
    }
    z += be32((b << 16) | a);
    auto chunk = [](const std::string& kind, const std::string& data) {
        return be32(static_cast<std::uint32_t>(data.size())) + kind + data + be32(crc32(kind + data));
    };
    std::string ihdr = be32(width) + be32(height) + std::string("\x08\x02\x00\x00\x00", 5);
    return std::string("\x89PNG\r\n\x1a\n", 8) + chunk("IHDR", ihdr) + chunk("IDAT", z) + chunk("IEND", "");
}

float tensor_value(std::size_t i) { return static_cast<float>((i * 37) % 101) / 100.0f; }

bool is_loopback_url(const std::string& url) {
    const std::size_t p = url.find("://");
    if (p == std::string::npos) return false;
    std::string rest = url.substr(p + 3);
    rest = rest.substr(0, rest.find('/'));
    std::string host;
    if (!rest.empty() && rest[0] == '[') host = rest.substr(1, rest.find(']') - 1);
    else host = rest.substr(0, rest.find(':'));
    host = lower(host);
    return host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
}

std::string url_host(const std::string& url) {
    const std::size_t p = url.find("://");
    std::string rest = p == std::string::npos ? url : url.substr(p + 3);
    rest = rest.substr(0, rest.find('/'));
    if (!rest.empty() && rest[0] == '[') return rest.substr(1, rest.find(']') - 1);
    return rest.substr(0, rest.find(':'));
}

std::string hardware() {
#ifdef __APPLE__
    char brand[256] = {0};
    std::size_t n = sizeof brand;
    std::uint64_t mem = 0;
    std::size_t m = sizeof mem;
    sysctlbyname("machdep.cpu.brand_string", brand, &n, nullptr, 0);
    sysctlbyname("hw.memsize", &mem, &m, nullptr, 0);
    return std::string(brand) + ", " + std::to_string(mem >> 30) + " GB";
#else
    return "unknown";
#endif
}

// ---- the spec as JSON, by id ----------------------------------------------------------------------

json normal(const json& spec) {
    json out = json::object();
    out["applet"] = spec.value("applet", json::object());
    json panels = json::object(), sources = json::object();
    for (const auto& p : spec.value("panel", json::array())) panels[p.value("id", "")] = p;
    for (const auto& s : spec.value("source", json::array())) sources[s.value("id", "")] = s;
    out["panel"] = panels;
    out["source"] = sources;
    out["layout"] = spec.contains("layout") ? spec["layout"] : json(nullptr);
    return out;
}

void diff(const json& a, const json& b, const std::string& path, std::vector<std::string>* out) {
    if (a == b) return;
    if (a.is_object() && b.is_object()) {
        std::set<std::string> keys;
        for (auto it = a.begin(); it != a.end(); ++it) keys.insert(it.key());
        for (auto it = b.begin(); it != b.end(); ++it) keys.insert(it.key());
        for (const auto& k : keys)
            diff(a.contains(k) ? a[k] : json(nullptr), b.contains(k) ? b[k] : json(nullptr),
                 path.empty() ? k : path + "." + k, out);
        return;
    }
    out->push_back(path);
}

json at_path(const json& j, const std::string& path) {
    const json* cur = &j;
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t dot = path.find('.', start);
        const std::string key = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (!cur->is_object() || !cur->contains(key)) return nullptr;
        cur = &(*cur)[key];
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return *cur;
}

void layout_ids(const json& node, std::set<std::string>* out) {
    if (node.is_string()) out->insert(node.get<std::string>());
    else if (node.is_object() && node.contains("children"))
        for (const auto& c : node["children"]) layout_ids(c, out);
}

json substitute(const json& j, const std::string& colour) {
    if (j.is_string() && j.get<std::string>() == "$colour") return colour;
    if (j.is_object()) {
        json out = json::object();
        for (auto it = j.begin(); it != j.end(); ++it) out[it.key()] = substitute(it.value(), colour);
        return out;
    }
    if (j.is_array()) {
        json out = json::array();
        for (const auto& v : j) out.push_back(substitute(v, colour));
        return out;
    }
    return j;
}

// ---- options ------------------------------------------------------------------------------------

struct Options {
    std::string python;
    std::string scripted;   // "pass", "fail" or empty
    std::string preset, provider, base_url, model;
    bool real = false;
    int runs = 0;
    std::set<int> tasks;
    std::string out = ".";
    std::string work;
    int timeout_s = 300;
    bool assume_vision = true;
    std::string server, quantization;
};

const char* kUsage =
    "usage: caliper-agent-conformance --python PATH\n"
    "         (--scripted pass|fail | --preset ID | --provider claude|openai-compatible)\n"
    "         [--base-url URL] [--model ID] [--real] [--runs N] [--tasks 1,2,...] [--out DIR] [--work DIR]\n"
    "         [--timeout SECONDS] [--assume-vision yes|no] [--server TEXT] [--quantization TEXT]\n";

bool parse_args(int argc, char** argv, Options* o, std::string* problem) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string* v) {
            if (i + 1 >= argc) return false;
            *v = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--python" && next(&o->python)) continue;
        if (a == "--scripted" && next(&o->scripted)) continue;
        if (a == "--preset" && next(&o->preset)) continue;
        if (a == "--provider" && next(&o->provider)) continue;
        if (a == "--base-url" && next(&o->base_url)) continue;
        if (a == "--model" && next(&o->model)) continue;
        if (a == "--out" && next(&o->out)) continue;
        if (a == "--work" && next(&o->work)) continue;
        if (a == "--server" && next(&o->server)) continue;
        if (a == "--quantization" && next(&o->quantization)) continue;
        if (a == "--real") {
            o->real = true;
            continue;
        }
        if (a == "--runs" && next(&v)) {
            o->runs = std::atoi(v.c_str());
            continue;
        }
        if (a == "--timeout" && next(&v)) {
            o->timeout_s = std::atoi(v.c_str());
            continue;
        }
        if (a == "--assume-vision" && next(&v)) {
            o->assume_vision = v == "yes";
            continue;
        }
        if (a == "--tasks" && next(&v)) {
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) o->tasks.insert(std::atoi(item.c_str()));
            continue;
        }
        *problem = "unknown or incomplete option " + a;
        return false;
    }
    if (o->python.empty()) return *problem = "--python is required", false;
    if (!o->scripted.empty() && o->scripted != "pass" && o->scripted != "fail")
        return *problem = "--scripted is pass or fail", false;
    if (o->scripted.empty() && o->preset.empty() && o->provider.empty())
        return *problem = "give --scripted, --preset or --provider", false;
    if (o->runs <= 0) o->runs = o->scripted.empty() ? 3 : 1;
    if (o->runs > 10) return *problem = "--runs is at most 10", false;
    if (o->timeout_s <= 0 || o->timeout_s > 3600) return *problem = "--timeout is 1 to 3600 seconds", false;
    for (std::string* s : {&o->server, &o->quantization}) {
        if (s->size() > 80) return *problem = "--server and --quantization are at most 80 characters", false;
        for (char c : *s)
            if (static_cast<unsigned char>(c) < 0x20) return *problem = "--server and --quantization are printable", false;
    }
    if (o->work.empty()) o->work = (fs::path(o->out) / "conformance-work").string();
    return true;
}

// ---- the provider -------------------------------------------------------------------------------

struct Provider {
    std::string kind;      // scripted, claude, openai-compatible
    std::string preset;    // the preset id, or empty
    std::string base_url;  // empty for Claude
    std::string account;   // the key store's id for the key, or empty
    bool offline = true;   // the sidecar runs with CALIPER_AGENT_OFFLINE=1
};

// Why --real may not run here, or empty: it needs the person's opt-in, and it
// never runs from a test, which has the offline switch (every ctest entry
// inherits it, src/builder/cmake/CaliperOfflineTests.cmake) or ctest's or
// pytest's own variable.
std::string real_refusal() {
    auto set = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && *v != '\0';
    };
    const char* offline = std::getenv(agent::kOfflineVariable);
    if (offline && *offline && std::string(offline) != "0")
        return std::string("--real is refused while ") + agent::kOfflineVariable +
               " is set: a test run never reaches a hosted provider";
    if (set("CTEST_INTERACTIVE_DEBUG_MODE") || set("PYTEST_CURRENT_TEST"))
        return "--real is refused under ctest or pytest: a test run never reaches a hosted provider";
    const char* opt_in = std::getenv("CALIPER_CONFORMANCE_REAL");
    if (!opt_in || std::string(opt_in) != "1")
        return "--real also needs CALIPER_CONFORMANCE_REAL=1 in the environment, set by the person who runs it on "
               "purpose";
    return {};
}

bool choose_provider(const Options& o, Provider* p, std::string* problem) {
    if (o.real) {
        if (std::string why = real_refusal(); !why.empty()) return *problem = why, false;
    }
    if (!o.scripted.empty()) {
        p->kind = "scripted";
        return true;
    }
    if (!o.preset.empty()) {
        const agent::ProviderPreset* preset = agent::find_preset(o.preset);
        if (!preset) return *problem = "no preset is called " + o.preset, false;
        p->preset = preset->id;
        p->kind = preset->api;
        if (preset->id == "custom") {
            if (o.base_url.empty()) return *problem = "the Custom preset needs --base-url", false;
            p->base_url = o.base_url;
        } else if (preset->api != "claude") {
            p->base_url = (preset->local && !o.base_url.empty()) ? o.base_url : preset->base_url;
        }
    } else {
        p->kind = o.provider;
        if (p->kind != "claude" && p->kind != "openai-compatible")
            return *problem = "--provider is claude or openai-compatible", false;
        if (p->kind == "openai-compatible") {
            if (o.base_url.empty()) return *problem = "an OpenAI-compatible server needs --base-url", false;
            p->base_url = o.base_url;
        }
    }
    if (p->kind == "openai-compatible" && o.model.empty()) return *problem = "an OpenAI-compatible server needs --model", false;
    const bool hosted = p->kind == "claude" || !is_loopback_url(p->base_url);
    if (hosted && !o.real)
        return *problem = "a hosted provider or a server off this machine costs money or sends data away; pass --real "
                          "to run against it on purpose",
               false;
    p->offline = !o.real;
    if (p->kind == "claude") p->account = agent::kAnthropicProvider;
    else if (!p->preset.empty() && p->preset != "custom") p->account = agent::key_need(p->preset) == agent::KeyNeed::none ? "" : p->preset;
    else p->account = agent::custom_key_account(p->base_url);
    return true;
}

std::vector<std::string> provider_args(const Options& o, const Provider& p) {
    std::vector<std::string> args;
    if (p.kind == "scripted") return args;
    if (!p.preset.empty()) args = {"--preset", p.preset};
    else args = {"--provider", p.kind};
    if (!p.base_url.empty() && (p.preset.empty() || p.preset == "custom" || agent::find_preset(p.preset)->local))
        args.insert(args.end(), {"--base-url", p.base_url});
    if (!o.model.empty()) args.insert(args.end(), {"--model", o.model});
    return args;
}

// The sidecar's environment: the key through the key store's placement, the
// offline switch unless --real, and this repository's sidecar on the path.
bool sidecar_env(const Provider& p, acp::SpawnOptions* spawn, std::string* secret, std::string* problem) {
    const std::string agent_dir = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/agent";
    spawn->env.push_back({"PYTHONPATH", agent_dir});
    spawn->env.push_back({"PYTHONDONTWRITEBYTECODE", "1"});
    if (p.kind == "scripted") {
        spawn->env.push_back({"ANTHROPIC_API_KEY", kFakeKey});
    } else if (!p.account.empty()) {
        agent::MemoryKeyStore store;
        const char* var = agent::key_variable(p.account);
        const char* value = var ? std::getenv(var) : nullptr;
        if (value && *value) {
            if (Error e = store.set(p.account, value)) return *problem = e.message, false;
            *secret = value;
        } else if (agent::key_need(p.account) == agent::KeyNeed::required) {
            return *problem = std::string("this provider needs its key in ") + (var ? var : "the environment"), false;
        }
        if (Error e = agent::place_key_in_environment(store, p.account, spawn)) return *problem = e.message, false;
    }
    if (p.offline) {
        spawn->env.push_back({agent::kOfflineVariable, "1"});
    } else {
        spawn->env_remove.push_back(agent::kOfflineVariable);
    }
    return true;
}

// ---- the fixture workspace and its session ---------------------------------------------------------

class RenderHost final : public SessionHost {
public:
    std::atomic<int> colour{0};
    CommandOutput app_info() override { return CommandOutput::fail({"command.not_bound", "No app here."}); }
    CommandOutput app_applets() override { return CommandOutput::fail({"command.not_bound", "No app here."}); }
    CommandOutput app_launch(const std::string&) override { return CommandOutput::fail({"command.not_bound", "No app here."}); }
    CommandOutput capture_frame() override { return CommandOutput::fail({"command.not_bound", "No app here."}); }
    // The suite's stand-in for drawing a panel: a small picture of one colour,
    // which the vision task asks the model to name.
    CommandOutput render_panel(const spec::Spec&, const PanelData&, int, int) override {
        const Colour& c = kColours[colour.load() % 4];
        return CommandOutput::ok({{"png_base64", base64(solid_png(c.rgb, 64, 40))}, {"width", 64}, {"height", 40}});
    }
};

struct Workspace {
    std::string root, ws, api_file;
    std::unique_ptr<Session> session;
    std::shared_ptr<RenderHost> host = std::make_shared<RenderHost>();
    std::unique_ptr<CommandServer> server;
    std::mutex mutex;  // one invocation at a time: the drainer or this thread
    std::atomic<bool> stop{false};
    std::thread drainer;
    std::map<std::string, json> loaders;  // file stem -> {id, outputs}

    ~Workspace() {
        if (server) {
            session->queue().shutdown();
            server->stop();
        }
        stop.store(true);
        if (drainer.joinable()) drainer.join();
    }

    CommandResult invoke(const std::string& name, json args, const std::string& preview = {}) {
        std::lock_guard<std::mutex> lock(mutex);
        CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = CommandOrigin::ui;
        inv.preview = preview;
        return session->invoke(std::move(inv));
    }
};

bool build_workspace(const std::string& root, Workspace* w, std::string* problem) {
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root + "/workspace/data", ec);
    w->root = fs::canonical(root).string();
    w->ws = w->root + "/workspace";
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);
        auto r = con.Query("COPY (SELECT range AS step, 1.0 / CAST(range + 1 AS DOUBLE) AS loss, "
                           "CAST(range AS DOUBLE) / 200 AS accuracy FROM range(" + std::to_string(kRows) + ")) TO '" +
                           w->ws + "/data/run42_metrics.parquet' (FORMAT parquet)");
        if (r->HasError()) return *problem = "the Parquet fixture: " + r->GetError(), false;
    }
    {
        std::size_t n = 1;
        for (int64_t d : kTensorShape) n *= static_cast<std::size_t>(d);
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = tensor_value(i);
        json header;
        header["weights"] = {{"dtype", "F32"}, {"shape", kTensorShape}, {"data_offsets", {0, n * 4}}};
        std::string h = header.dump();
        while (h.size() % 8) h += ' ';
        std::string bytes(8, '\0');
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((static_cast<std::uint64_t>(h.size()) >> (8 * i)) & 0xFF);
        bytes += h;
        bytes.append(reinterpret_cast<const char*>(v.data()), n * 4);
        if (!write_file(w->ws + "/data/attn.safetensors", bytes)) return *problem = "the safetensors fixture", false;
    }
    const std::string fixture = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/spec/tests/fixtures/dashboard.spec.toml";
    if (!write_file(w->ws + "/dashboard.spec.toml", read_file(fixture))) return *problem = "the spec fixture", false;
    auto opened = Session::open(w->ws);
    if (!opened.error.ok()) return *problem = opened.error.message, false;
    w->session = std::move(opened.session);
    w->session->set_host(w->host);
    CommandResult imported = w->invoke("data.import", {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}});
    if (!imported.ok) return *problem = "data.import: " + imported.error.message, false;
    for (const auto& l : imported.result.value("loaders", json::array())) {
        const std::string path = l.value("path", "");
        w->loaders[path.find("run42") != std::string::npos ? "run42_metrics" : "attn_weights"] = l;
    }
    CommandResult open = w->invoke("spec.open", {{"path", "dashboard.spec.toml"}});
    if (!open.ok) return *problem = "spec.open: " + open.error.message, false;
    CommandServerOptions options;
    options.token = generate_session_token();
    options.origin = CommandOrigin::agent;  // the token is the sidecar's
    w->server = std::make_unique<CommandServer>(w->session->queue(), options);
    if (Error e = w->server->start()) return *problem = e.message, false;
    w->drainer = std::thread([w] {
        while (!w->stop.load()) {
            {
                std::lock_guard<std::mutex> lock(w->mutex);
                w->session->queue().drain();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    if (Error e = write_api_file(w->root + "/api", w->server->port(), options.token, &w->api_file))
        return *problem = e.message, false;
    return true;
}

// ---- one run of one task -------------------------------------------------------------------------

struct ModelCaps {
    bool tools = true, vision = true;
    std::uint64_t context = 0;
    bool context_known = false;  // reported by the server or set by the person, not assumed
    json probe = nullptr;  // what the probe printed
};

struct TaskRun {
    bool passed = false;
    std::string reason;
    double seconds = 0;
    std::string stop_reason;
    int edits = 0;
    json handoff = nullptr;
    std::vector<std::string> tools;
};

json script_steps(const json& answer, const std::string& colour) {
    json steps = json::array();
    const json usage = {{"input_tokens", 40}, {"output_tokens", 12}};
    const json calls = answer.value("calls", json::array());
    if (!calls.empty()) {
        json content = json::array();
        int n = 0;
        for (const auto& c : calls)
            content.push_back({{"type", "tool_use"}, {"id", "toolu_c" + std::to_string(++n)}, {"name", c["name"]},
                               {"input", substitute(c["input"], colour)}});
        steps.push_back({{"message", {{"content", content}, {"stop_reason", "tool_use"}, {"usage", usage}}}});
    }
    const std::string text = answer.value("text", "");
    steps.push_back({{"message", {{"content", json::array({{{"type", "text"}, {"text", text.empty() ? "Done." : text}}})},
                                  {"stop_reason", "end_turn"}, {"usage", usage}}}});
    return steps;
}

std::string tool_name(const std::string& command) {
    std::string s = command;
    const std::size_t dot = s.find('.');
    if (dot != std::string::npos) s[dot] = '_';
    return s;
}

// The task's check of the preview against the open spec.
std::string check(Workspace& w, const json& task, const std::string& preview, const json& before, const json& after,
                  const std::string& reply, const json& handoff, const std::string& colour) {
    const json c = task["check"];
    const std::string kind = c.value("kind", "");
    std::vector<std::string> changed;
    diff(before, after, "", &changed);
    auto joined = [&] {
        std::string s;
        for (const auto& p : changed) s += (s.empty() ? "" : ", ") + p;
        return s.empty() ? std::string("nothing") : s;
    };
    auto existing_unchanged = [&](std::string* why) {
        for (const char* group : {"panel", "source"})
            for (auto it = before[group].begin(); it != before[group].end(); ++it)
                if (!after[group].contains(it.key()) || after[group][it.key()] != it.value())
                    return *why = std::string("the existing ") + group + " " + it.key() + " changed", false;
        return true;
    };
    auto new_panels = [&](const std::string& primitive) {
        std::vector<std::string> ids;
        for (auto it = after["panel"].begin(); it != after["panel"].end(); ++it)
            if (!before["panel"].contains(it.key()) && it.value().value("primitive", "") == primitive)
                ids.push_back(it.key());
        return ids;
    };
    auto resolve = [&](const std::string& panel) {
        std::lock_guard<std::mutex> lock(w.mutex);
        return w.session->resolve(panel, preview);
    };

    if (kind == "only_changes" || kind == "linear_y") {
        std::set<std::string> want;
        if (kind == "linear_y") want.insert("panel." + c.value("panel", "") + ".options.y_log");
        for (const auto& p : c.value("paths", json::array())) want.insert(p.get<std::string>());
        if (std::set<std::string>(changed.begin(), changed.end()) != want) return "the preview changed " + joined();
        if (kind == "linear_y") {
            const json v = at_path(after, "panel." + c.value("panel", "") + ".options.y_log");
            if (!(v.is_null() || v == false)) return "the y axis is still logarithmic";
            return "";
        }
        for (auto it = c["equals"].begin(); it != c["equals"].end(); ++it) {
            const json expected = substitute(it.value(), colour);
            const json got = at_path(after, it.key());
            if (got != expected) return it.key() + " is " + got.dump().substr(0, 80) + ", not " + expected.dump();
        }
        return "";
    }
    if (kind == "no_edit_reply") {
        if (!changed.empty()) return "the preview changed " + joined();
        for (const auto& m : c["mentions"])
            if (lower(reply).find(lower(m.get<std::string>())) == std::string::npos)
                return "the reply does not mention " + m.get<std::string>();
        return "";
    }
    if (kind == "handoff") {
        if (!changed.empty()) return "the preview changed " + joined();
        if (handoff.is_null()) return "the request was not handed to the side chat";
        return "";
    }
    std::string why;
    if (kind == "removed_panel") {
        const std::string gone = c.value("panel", "");
        if (after["panel"].contains(gone)) return "the panel " + gone + " is still there";
        for (const auto& keep : c["remaining"]) {
            const std::string id = keep.get<std::string>();
            if (!after["panel"].contains(id) || after["panel"][id] != before["panel"][id]) return "the panel " + id + " changed";
        }
        if (after["source"] != before["source"]) return "a source changed";
        std::set<std::string> ids;
        layout_ids(after["layout"], &ids);
        for (const auto& keep : c["remaining"])
            if (!ids.count(keep.get<std::string>())) return "the layout lost " + keep.get<std::string>();
        if (ids.count(gone)) return "the layout still holds " + gone;
        return "";
    }
    if (!existing_unchanged(&why)) return why;
    if (kind == "new_table_rows") {
        const auto ids = new_panels("table");
        if (ids.empty()) return "no table panel was added";
        const PanelData d = resolve(ids[0]);
        if (!d.error.ok()) return "the new table does not resolve: " + d.error.message;
        for (const auto& s : d.slots) {
            if (s.kind != SlotData::Kind::table) continue;
            std::size_t col = s.columns.size();
            for (std::size_t i = 0; i < s.columns.size(); ++i)
                if (s.columns[i].name == c.value("column", "")) col = i;
            if (col == s.columns.size()) return "the new table has no column " + c.value("column", "");
            json got = json::array();
            for (const auto& row : s.values) got.push_back(row[col]);
            if (got != c["values"]) return "the new table holds the steps " + got.dump().substr(0, 120);
            return "";
        }
        return "the new table has no rows";
    }
    if (kind == "new_series") {
        const auto ids = new_panels("series");
        if (ids.empty()) return "no series panel was added";
        const PanelData d = resolve(ids[0]);
        if (!d.error.ok()) return "the new panel does not resolve: " + d.error.message;
        std::string x;
        std::set<std::string> ys;
        for (const auto& s : d.slots) {
            if (s.slot == "x") x = s.column;
            if (s.slot == "y") ys.insert(s.column);
        }
        if (x != c.value("x", "") || ys != std::set<std::string>{c.value("y", "")})
            return "the new panel plots " + (ys.empty() ? std::string("nothing") : *ys.begin()) + " against " +
                   (x.empty() ? std::string("its index") : x);
        return "";
    }
    if (kind == "new_heatmap_reference") {
        const auto ids = new_panels("heatmap");
        if (ids.empty()) return "no heatmap panel was added";
        const PanelData d = resolve(ids[0]);
        if (!d.error.ok()) return "the new heatmap does not resolve: " + d.error.message;
        for (const auto& s : d.slots) {
            if (s.slot != "values") continue;
            const auto& t = s.tensor;
            if (t.shape() != std::vector<int64_t>{16, 16}) return "the heatmap's data has the wrong shape";
            const int layer = c.value("layer", 0);
            for (int q = 0; q < 16; ++q)
                for (int k = 0; k < 16; ++k) {
                    double ref = 0;
                    for (int h = 0; h < 8; ++h)
                        ref += tensor_value(static_cast<std::size_t>(((layer * 8 + h) * 16 + q) * 16 + k));
                    ref /= 8;
                    const std::size_t i = static_cast<std::size_t>(q * 16 + k);
                    double got;
                    if (t.dtype() == DType::f32) got = reinterpret_cast<const float*>(t.data())[i];
                    else if (t.dtype() == DType::f64) got = reinterpret_cast<const double*>(t.data())[i];
                    else return "the heatmap's data is not floating point";
                    if (std::fabs(got - ref) > 1e-6) return "the heatmap's data differs from the reference";
                }
            return "";
        }
        return "the new heatmap has no values";
    }
    return "the suite has no check called " + kind;
}

TaskRun run_once(const Options& o, const Provider& p, const ModelCaps& caps, Workspace& w, const json& task,
                   int run_index, std::string* secret) {
    TaskRun r;
    const int id = task.value("id", 0);
    const std::string preview = "widget-t" + std::to_string(id) + "-r" + std::to_string(run_index);
    const json target_spec = task["target"];
    w.host->colour.store(id + run_index);
    const std::string colour = kColours[(id + run_index) % 4].name;

    if (task.contains("injection")) {
        const json inj = task["injection"];
        CommandResult res = w.invoke("panel.patch", {{"id", inj["panel"]}, {"patch", {{"title", inj["title"]}}}});
        if (!res.ok) return r.reason = "the suite could not plant its data: " + res.error.message, r;
    }
    struct Restore {
        Workspace& w;
        bool on;
        ~Restore() {
            if (on) w.invoke("history.undo", json::object());
        }
    } restore{w, task.contains("injection")};

    WidgetTarget target;
    const std::string kind = target_spec.value("kind", "panel");
    json begin = {{"id", preview}};
    if (kind == "panel" || kind == "selection") {
        target.kind = kind == "panel" ? WidgetTargetKind::panel : WidgetTargetKind::selection;
        target.panel = target_spec.value("panel", "");
        begin["panel"] = target.panel;
    } else if (kind == "property") {
        target.kind = WidgetTargetKind::property;
        target.panel = target_spec.value("panel", "");
        target.path = target_spec.value("path", "");
        begin["panel"] = target.panel;
    } else {
        target.kind = WidgetTargetKind::dropped_file;
        const json loader = w.loaders[target_spec.value("file", "")];
        target.loader = loader.value("id", "");
        for (const auto& out : loader.value("outputs", json::array())) target.catalog.push_back(out);
    }
    CommandResult begun = w.invoke("preview.begin", begin);
    if (!begun.ok) return r.reason = "preview.begin: " + begun.error.message, r;

    ContextBundleOptions options;
    options.preview = preview;
    options.context_tokens = caps.context;
    options.vision = caps.vision;
    if (task.contains("selection")) {
        WidgetSelection sel;
        sel.columns = {"step", "loss", "accuracy"};
        const int from = task["selection"].value("from_step", 0), count = task["selection"].value("count", 0);
        for (int s = from; s < from + count; ++s)
            sel.rows.push_back({s, 1.0 / (s + 1), s / 200.0});
        sel.range = {{"x", {from, from + count - 1}}};
        options.selection = sel;
    }
    ContextBundle bundle;
    json before;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        bundle = build_context_bundle(*w.session, target, options);
        before = normal(spec::to_json(w.session->open_spec()->spec));
    }
    if (!bundle.error.ok()) {
        w.invoke("preview.discard", {{"id", preview}});
        return r.reason = "the context bundle: " + bundle.error.message, r;
    }
    std::set<std::string> allowed = {"handoff"};
    for (const auto& s : bundle.scope) allowed.insert(tool_name(s));

    acp::SpawnOptions spawn;
    std::string problem;
    if (!sidecar_env(p, &spawn, secret, &problem)) {
        w.invoke("preview.discard", {{"id", preview}});
        return r.reason = problem, r;
    }
    if (p.kind == "scripted") {
        const std::string script = w.root + "/scripts/t" + std::to_string(id) + "-r" + std::to_string(run_index) + ".json";
        write_file(script, json{{"steps", script_steps(task[o.scripted], colour)},
                                {"capabilities", {{"vision", caps.vision}}},
                                {"backoff_scale", 0}}.dump());
        spawn.env.push_back({"CALIPER_AGENT_TEST_SCRIPT", script});
        spawn.env_remove.erase(std::remove(spawn.env_remove.begin(), spawn.env_remove.end(), "CALIPER_AGENT_TEST_SCRIPT"),
                               spawn.env_remove.end());
    } else {
        spawn.env_remove.push_back("CALIPER_AGENT_TEST_SCRIPT");
    }
    acp::AcpClientOptions co;
    co.agent_argv = {o.python, "-P", "-m", "caliper_agent", "--command-table",
                     std::string(CALIPER_BUILDER_SOURCE_DIR) + "/commands/command_table.json", "--api-file", w.api_file};
    for (const auto& a : provider_args(o, p)) co.agent_argv.push_back(a);
    if (p.kind == "openai-compatible") {
        co.agent_argv.insert(co.agent_argv.end(), {"--model-vision", caps.vision ? "yes" : "no"});
        if (caps.context_known)
            co.agent_argv.insert(co.agent_argv.end(), {"--context-length", std::to_string(caps.context)});
    }
    co.agent_env = spawn.env;
    co.agent_env_remove = spawn.env_remove;
    co.agent_cwd = w.ws;
    co.request_timeout_ms = 60000;
    co.stall_timeout_ms = o.timeout_s * 1000;
    acp::AcpClient client(co);
    acp::AgentInfo info;
    acp::SessionInfo si;
    std::string reply;
    json widget_meta = nullptr;
    Error failure;
    if (Error e = client.start(&info)) {
        failure = e;
    } else if (Error e = client.new_session(w.ws, {}, &si, bundle.meta)) {
        failure = e;
    } else {
        json blocks = bundle.blocks;
        blocks.push_back({{"type", "text"}, {"text", task.value("request", "")}});
        const auto t0 = std::chrono::steady_clock::now();
        if (Error e = client.prompt_blocks(blocks)) failure = e;
        bool done = !failure.ok();
        while (!done) {
            acp::AcpEvent ev;
            if (!client.wait(&ev, 500)) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(o.timeout_s)) {
                    client.cancel();
                    failure = {"conformance.timeout", "no answer within the time limit"};
                    done = true;
                }
                continue;
            }
            switch (ev.kind) {
                case acp::AcpEventKind::agent_message_chunk: reply += ev.text; break;
                case acp::AcpEventKind::tool_call:
                    r.tools.push_back(ev.raw.is_object() ? ev.raw.value("name", ev.tool_call.title) : ev.tool_call.title);
                    break;
                case acp::AcpEventKind::permission_request:
                    for (const auto& opt : ev.options)
                        if (opt.kind == "reject_once") client.respond_permission(ev.permission_id, opt.id);
                    break;
                case acp::AcpEventKind::turn_ended:
                    r.stop_reason = ev.stop_reason;
                    failure = ev.error;
                    if (ev.raw.is_object() && ev.raw.contains("_meta"))
                        widget_meta = ev.raw["_meta"].value("caliper", json::object()).value("widget", json(nullptr));
                    done = true;
                    break;
                case acp::AcpEventKind::agent_exited:
                case acp::AcpEventKind::stalled:
                    failure = ev.error.ok() ? Error{"conformance.agent_ended", "the agent ended or stalled"} : ev.error;
                    done = true;
                    break;
                default: break;
            }
        }
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    client.shutdown();

    json after;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        const PreviewState st = w.session->preview(preview);
        after = st.error.ok() ? normal(spec::to_json(st.spec.spec)) : before;
    }
    if (widget_meta.is_object()) {
        r.edits = widget_meta.value("edits", 0);
        r.handoff = widget_meta.value("handoff", json(nullptr));
    }
    if (!failure.ok()) r.reason = failure.code + ": " + failure.message.substr(0, 200);
    else if (r.stop_reason != "end_turn") r.reason = "the turn ended with " + r.stop_reason;
    else if (widget_meta.is_object() && widget_meta.value("limit", false)) r.reason = "the turn reached its limit";
    else {
        for (const auto& t : r.tools)
            if (!allowed.count(t)) r.reason = "the model called " + t + ", which is outside the scope";
        if (r.reason.empty()) r.reason = check(w, task, preview, before, after, reply, r.handoff, colour);
    }
    r.passed = r.reason.empty();
    w.invoke("preview.discard", {{"id", preview}});
    return r;
}

// ---- the probe ----------------------------------------------------------------------------------

bool probe(const Options& o, const Provider& p, ModelCaps* caps, std::string* secret, std::string* problem) {
    if (p.kind == "scripted") {
        caps->vision = o.assume_vision;
        caps->probe = {{"tools", {{"value", true}, {"source", "script"}}},
                       {"vision", {{"value", o.assume_vision}, {"source", "script"}}},
                       {"context_length", {{"value", nullptr}, {"source", "unknown"}}}};
        return true;
    }
    acp::SpawnOptions spawn;
    if (!sidecar_env(p, &spawn, secret, problem)) return false;
    spawn.env_remove.push_back("CALIPER_AGENT_TEST_SCRIPT");
    spawn.argv = {o.python, "-P", "-m", "caliper_agent", "probe"};
    for (const auto& a : provider_args(o, p)) spawn.argv.push_back(a);
    if (o.model.empty()) spawn.argv.insert(spawn.argv.end(), {"--model", "claude-haiku-5-5"});
    acp::RunLimits limits;
    limits.timeout_ms = 200000;
    const acp::RunResult run = acp::run_process(spawn, limits);
    json doc = json::parse(run.out, nullptr, false);
    if (!doc.is_object() || !doc.contains("capabilities"))
        return *problem = "caliper-agent probe gave no answer: " + run.err.substr(0, 300), false;
    const json c = doc["capabilities"];
    caps->tools = c["tools"].value("value", json(true)) != false;
    caps->vision = c["vision"].value("value", json(true)) == true;
    const json ctx = c["context_length"].value("value", json(nullptr));
    caps->context = ctx.is_number_unsigned() || ctx.is_number_integer() ? ctx.get<std::uint64_t>() : 0;
    const std::string source = c["context_length"].value("source", std::string("unknown"));
    caps->context_known = caps->context > 0 && source != "default" && source != "unknown";
    caps->probe = c;
    return true;
}

std::string sidecar_version(const Options& o) {
    acp::SpawnOptions spawn;
    spawn.argv = {o.python, "-P", "-m", "caliper_agent", "--version"};
    spawn.env = {{"PYTHONPATH", std::string(CALIPER_BUILDER_SOURCE_DIR) + "/agent"}, {agent::kOfflineVariable, "1"}};
    acp::RunLimits limits;
    limits.timeout_ms = 60000;
    std::string out = acp::run_process(spawn, limits).out;
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.substr(0, 64);
}

std::string verdict(bool can_drive, int passed, int applicable) {
    if (!can_drive) return "cannot drive Caliper";
    if (applicable - passed <= 1) return "passes";
    if (passed * 10 >= applicable * 6) return "partial";
    return "fails";
}

std::string seconds_text(double s) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f s", s);
    return buf;
}

std::string markdown(const json& doc) {
    std::ostringstream md;
    md << "# Conformance run\n\n";
    md << "| Preset | Model | Server | Hardware | Sidecar | Date | Verdict | Tasks passed | Median time |\n";
    md << "|---|---|---|---|---|---|---|---|---|\n";
    md << "| " << doc.value("preset", "") << " | " << doc.value("model", "") << " | " << doc.value("server", "unknown")
       << " | " << doc.value("hardware", "") << " | " << doc.value("sidecar", "") << " | " << doc.value("date", "")
       << " | " << doc.value("verdict", "") << " | " << doc.value("passed", 0) << " of " << doc.value("applicable", 0)
       << " | " << seconds_text(doc.value("median_seconds", 0.0)) << " |\n\n";
    const json caps = doc.value("capabilities", json::object());
    auto cap = [&](const char* name) {
        const json c = caps.value(name, json::object());
        const json v = c.value("value", json(nullptr));
        const std::string value = v.is_boolean() ? (v.get<bool>() ? "yes" : "no")
                                  : v.is_number() ? v.dump() + " tokens" : "unknown";
        return value + " (" + c.value("source", std::string("unknown")) + ")";
    };
    const int runs = doc.value("runs_per_task", 0);
    md << "Mode " << doc.value("mode", "") << ", " << runs << (runs == 1 ? " run" : " runs")
       << " of each task. Capabilities: tool calling " << cap("tools") << ", vision " << cap("vision")
       << ", context length " << cap("context_length") << ".\n\n";
    md << "| # | Task | Outcome | Runs passed | Median time | Why a run failed |\n|---|---|---|---|---|---|\n";
    for (const auto& t : doc["tasks"]) {
        int passed = 0;
        std::string why;
        std::vector<double> times;
        for (const auto& r : t["runs"]) {
            passed += r.value("passed", false) ? 1 : 0;
            times.push_back(r.value("seconds", 0.0));
            if (why.empty() && !r.value("passed", false)) why = r.value("reason", "");
        }
        std::sort(times.begin(), times.end());
        std::string reason;
        for (char ch : why) reason += ch == '|' ? '/' : (ch == '\n' ? ' ' : ch);
        md << "| " << t.value("id", 0) << " | " << t.value("title", "") << " | " << t.value("outcome", "") << " | "
           << passed << " of " << t["runs"].size() << " | "
           << (times.empty() ? std::string("") : seconds_text(times[times.size() / 2])) << " | " << reason << " |\n";
    }
    return md.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string problem;
    if (!parse_args(argc, argv, &o, &problem)) {
        std::cerr << "caliper-agent-conformance: " << problem << "\n" << kUsage;
        return 2;
    }
    Provider p;
    if (!choose_provider(o, &p, &problem)) {
        std::cerr << "caliper-agent-conformance: " << problem << "\n";
        return 2;
    }
    const json tasks = json::parse(agent::detail::kConformanceTasksJson, nullptr, false);
    if (!tasks.is_object() || !tasks.contains("tasks")) {
        std::cerr << "caliper-agent-conformance: the embedded tasks do not parse\n";
        return 1;
    }
    std::string secret;
    ModelCaps caps;
    if (!probe(o, p, &caps, &secret, &problem)) {
        std::cerr << "caliper-agent-conformance: " << problem << "\n";
        return 1;
    }
    Workspace w;
    if (caps.tools && !build_workspace(o.work, &w, &problem)) {
        std::cerr << "caliper-agent-conformance: the fixture workspace: " << problem << "\n";
        return 1;
    }

    json doc = {{"suite", kSuiteName},
                {"suite_version", kSuiteVersion},
                {"date", today()},
                {"mode", p.kind == "scripted" ? "scripted-" + o.scripted : (p.offline ? "offline" : "real")},
                {"provider", p.kind},
                {"preset", p.preset.empty() ? (p.kind == "scripted" ? "scripted" : "none") : p.preset},
                {"base_url_host", p.base_url.empty() ? (p.kind == "claude" ? "api.anthropic.com" : "") : url_host(p.base_url)},
                {"model", !o.model.empty() ? o.model : (p.kind == "openai-compatible" ? "" : "claude-haiku-5-5")},
                {"server", o.server.empty() ? "unknown" : o.server},
                {"quantization", o.quantization.empty() ? json(nullptr) : json(o.quantization)},
                {"hardware", p.kind == "scripted" ? std::string("none")
                             : is_loopback_url(p.base_url) ? hardware() : std::string("hosted")},
                {"sidecar", sidecar_version(o)},
                {"capabilities", caps.probe},
                {"runs_per_task", o.runs},
                {"tasks", json::array()}};
    int passed = 0, applicable = 0;
    std::vector<double> times;
    for (const auto& task : tasks["tasks"]) {
        const int id = task.value("id", 0);
        if (!o.tasks.empty() && !o.tasks.count(id)) continue;
        json t = {{"id", id}, {"title", task.value("title", "")}, {"runs", json::array()}};
        if (!caps.tools) {
            t["outcome"] = "not run";
        } else if (task.value("needs_vision", false) && !caps.vision) {
            t["outcome"] = "not applicable";
        } else {
            int ok = 0;
            for (int run = 0; run < o.runs; ++run) {
                const TaskRun r = run_once(o, p, caps, w, task, run, &secret);
                ok += r.passed ? 1 : 0;
                times.push_back(r.seconds);
                t["runs"].push_back({{"passed", r.passed}, {"reason", r.reason}, {"seconds", std::round(r.seconds * 10) / 10},
                                     {"stop_reason", r.stop_reason}, {"edits", r.edits}, {"handoff", r.handoff},
                                     {"tool_calls", r.tools}});
                std::cerr << "task " << id << " run " << run + 1 << ": " << (r.passed ? "pass" : "fail: " + r.reason)
                          << "\n";
            }
            ++applicable;
            const bool task_passed = ok * 2 > o.runs;
            passed += task_passed ? 1 : 0;
            t["outcome"] = task_passed ? "pass" : "fail";
        }
        doc["tasks"].push_back(t);
    }
    std::sort(times.begin(), times.end());
    doc["passed"] = passed;
    doc["applicable"] = applicable;
    doc["verdict"] = verdict(caps.tools, passed, applicable);
    doc["median_seconds"] = times.empty() ? 0.0 : std::round(times[times.size() / 2] * 10) / 10;

    const std::string text = doc.dump(1);
    const std::string md = markdown(doc);
    if (!secret.empty() && (text.find(secret) != std::string::npos || md.find(secret) != std::string::npos)) {
        std::cerr << "caliper-agent-conformance: the result would hold the key; nothing was written\n";
        return 1;
    }
    const std::string stem = (fs::path(o.out) / ("conformance-" + doc.value("date", std::string()) + "-" +
                                                 slug(doc.value("model", std::string("model"))))).string();
    if (!write_file(stem + ".json", text + "\n") || !write_file(stem + ".md", md)) {
        std::cerr << "caliper-agent-conformance: the result files cannot be written\n";
        return 1;
    }
    std::cout << stem << ".json\n" << stem << ".md\n";
    std::cerr << "verdict: " << doc.value("verdict", "") << ", " << passed << " of " << applicable << " tasks\n";
    return 0;
}
