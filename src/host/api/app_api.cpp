// Caliper API v0 for the caliper executable (app_api.h).
#include "api/app_api.h"

#include "api/spec_page.h"
#include "../app_paths.h"   // executable_dir, for the spec page's views
#include "open_url.h"

#include "export_service.h"
#include "negotiation.h"
#include "net/json.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

namespace caliper_host {

bool api_wanted(const CommandLine& cl) {
    if (cl.api) return true;
#ifdef _WIN32
    return false;   // CALIPER_API is ignored on Windows (host_cli.h, windows_unavailable)
#else
    const char* v = std::getenv("CALIPER_API");
    return v && std::strcmp(v, "1") == 0;
#endif
}

namespace {

// The folder of the WebAssembly views that the spec page loads for the
// programs of its scene panels (api_server.h, kSpecAssets): CALIPER_WEB_DIR
// when set, else web/ in the app bundle's Resources or beside the executable,
// else, for a build folder inside the source tree (as the CMake presets make
// it), the Python package's _web folder of that tree, where
// scripts/build_wasm.py puts them. The first folder that holds the loader
// counts; "" when none does.
std::string find_spec_assets_dir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> tries;
    if (const char* env = std::getenv("CALIPER_WEB_DIR"); env && *env) tries.emplace_back(fs::path(env));
    const std::string exe = caliper::executable_dir();
    if (!exe.empty()) {
        tries.push_back(fs::path(exe) / ".." / "Resources" / "web");
        tries.push_back(fs::path(exe) / "web");
        tries.push_back(fs::path(exe) / ".." / "python" / "caliper" / "_web");
    }
    for (const fs::path& d : tries) {
        std::error_code ec;
        if (fs::is_regular_file(d / caliper_api::kSpecAssets[0], ec)) return d.lexically_normal().string();
    }
    return {};
}

}  // namespace

bool AppApi::start(const CommandLine& cl, const std::string& capture_unavailable,
                   const std::string& data_folder, std::ostream& out, std::string* err) {
    if (server_) return true;
    CliContext ctx;
    ctx.caps = host_caps();
    ctx.sdk_version = host_sdk_version();

    caliper_api::Options o;
    o.port = (uint16_t)(cl.api_port >= 0 ? cl.api_port : caliper_api::kDefaultPort);
    o.token = caliper_net::random_token();
    // The spec page's link carries a token of its own, which opens only the
    // page and its read-only commands (api_server.h, THE SPEC PAGE).
    do o.viewer_token = caliper_net::random_token();
    while (o.viewer_token == o.token);
    // The API file's token, for outside agents: origin mcp, reviewed in the app.
    do o.mcp_token = caliper_net::random_token();
    while (o.mcp_token == o.token || o.mcp_token == o.viewer_token);
    o.spec_page_html = caliper_api::spec_page_html();
    o.spec_assets_dir = find_spec_assets_dir();
    o.info_json = info_json(ctx);
    o.software_version = ctx.caps.version;
    o.allow_control = cl.allow_control;
    o.capture_unavailable = capture_unavailable;
    o.encode_png = [](const uint8_t* rgba, uint32_t w, uint32_t h, std::string* png) {
        return export_encode_png(rgba, w, h, png);
    };
    o.log = [](const std::string& line) { std::cerr << "[api] " << line << std::endl; };
    auto server = std::make_unique<caliper_api::Server>(std::move(o));
    server->set_command_gateway(gateway_);
    if (!server->start(err)) {
        if (cl.api_port < 0)
            *err += "; is another Caliper running with --api? Choose a port with --api-port";
        return false;
    }

    data_folder_ = data_folder;
    server_ = std::move(server);
    file_ = (std::filesystem::path(data_folder) / "api.json").string();
    file_text_ = file_json(server_->mcp_token());
    std::string why;
    if (!caliper_net::write_hub_file(file_, file_text_, &why)) {
        out << "[api] " << why << "; outside agents cannot find the API" << std::endl;
        file_.clear();
        file_text_.clear();
    }
    // The person's token is on this line only; the API file holds the token
    // for outside agents, whose commands are reviewed in the app.
    out << "[api] Caliper API on " << server_->base_url() << " with the session token " << server_->token();
    if (!file_.empty()) out << " (agents read " << file_ << ", whose commands are reviewed)";
    out << std::endl;
    if (!caliper_api::spec_page_html().empty())
        out << "[api] Instruments in a browser at " << server_->spec_url() << "&instrument=<id>" << std::endl;
    return true;
}

std::string AppApi::file_json(const std::string& token) const {
#ifdef _WIN32
    const long pid = (long)_getpid();
#else
    const long pid = (long)getpid();
#endif
    caliper_net::Json j = caliper_net::Json::object();
    j.set("version", 1);
    j.set("pid", (int64_t)pid);
    j.set("port", (uint64_t)server_->port());
    j.set("token", token);
    j.set("url", server_->base_url());
    j.set("stream_url", server_->stream_url());
    if (!caliper_api::spec_page_html().empty()) j.set("spec_url", server_->spec_url());
    return j.dump() + "\n";
}

std::string AppApi::issue_agent_file(std::string* why, const std::string& client) {
    if (!server_) {
        if (why) *why = "Caliper API is not running";
        return {};
    }
#ifdef _WIN32
    const long pid = (long)_getpid();
#else
    const long pid = (long)getpid();
#endif
    AgentFile f;
    f.token = server_->issue_token("agent", client);
    if (f.token.empty()) {
        if (why) *why = "the agent's token could not be made";
        return {};
    }
    f.path = (std::filesystem::path(data_folder_) /
              ("api-agent-" + std::to_string(pid) + "-" + std::to_string(++agent_files_made_) + ".json"))
                 .string();
    f.text = file_json(f.token);
    std::string err;
    if (!caliper_net::write_hub_file(f.path, f.text, &err)) {
        server_->revoke_token(f.token);
        if (why) *why = err;
        return {};
    }
    agent_files_.push_back(f);
    return f.path;
}

void AppApi::revoke_agent_file(const std::string& path) {
    for (auto it = agent_files_.begin(); it != agent_files_.end(); ++it) {
        if (it->path != path) continue;
        if (server_) server_->revoke_token(it->token);
        caliper_net::remove_hub_file(it->path, it->text);
        agent_files_.erase(it);
        return;
    }
}

bool AppApi::viewer_link_possible(const std::string& instrument, std::string* why) {
    auto refuse = [&](const char* reason) {
        if (why) *why = reason;
        return false;
    };
    if (caliper_api::spec_page_html().empty()) return refuse("this build has no browser viewer");
    // A workspace id, as instrument.save names the folder: nothing to escape,
    // and never "." or "..".
    if (!viewer_instrument_id(instrument)) return refuse("the instrument's id cannot be a link");
    return true;
}

std::string AppApi::viewer_link(const std::string& instrument, std::string* why) const {
    auto refuse = [&](const char* reason) {
        if (why) *why = reason;
        return std::string();
    };
    if (!viewer_link_possible(instrument, why)) return {};
    if (!server_) return refuse("Caliper API is not running");
    const std::string base = server_->spec_url();
    const std::string mark = std::string("?") + CALIPER_NET_TOKEN_PARAM + "=";
    const size_t at = base.find(mark);
    if (base.empty() || at == std::string::npos) return refuse("Caliper API has no viewer token");
    const std::string token = base.substr(at + mark.size());
    // The link is easily copied and kept in a browser's history: it carries
    // the viewer token alone.
    if (token.empty() || token == server_->token() || token == server_->mcp_token())
        return refuse("the link would carry a token that opens more than the viewer");
    return base + "&instrument=" + instrument;
}

void AppApi::stop() {
    if (!server_) return;
    // Every agent's file and token go with the API.
    while (!agent_files_.empty()) revoke_agent_file(agent_files_.front().path);
    server_->stop();
    server_.reset();
    if (!file_text_.empty()) caliper_net::remove_hub_file(file_, file_text_);
    file_text_.clear();
}

std::string launch_pending_json(const std::string& id, const std::string& version, const std::string& status,
                                const std::string& note) {
    caliper_net::Json j = caliper_net::Json::object();
    j.set("id", id);
    j.set("version", version);
    j.set("status", status);
    j.set("already_open", false);
    j.set("note", note);
    return j.dump();
}

std::string launched_json(const std::string& id, const std::string& version, bool already_open) {
    caliper_net::Json j = caliper_net::Json::object();
    j.set("id", id);
    j.set("version", version);
    j.set("status", "ACTIVE");
    j.set("already_open", already_open);
    return j.dump();
}

}  // namespace caliper_host
