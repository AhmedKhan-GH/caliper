// Caliper API v0 for the caliper executable (app_api.h).
#include "api/app_api.h"

#include "export_service.h"
#include "negotiation.h"
#include "net/json.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#ifdef _WIN32
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

namespace caliper_host {

bool api_wanted(const CommandLine& cl) {
    if (cl.api) return true;
    const char* v = std::getenv("CALIPER_API");
    return v && std::strcmp(v, "1") == 0;
}

bool AppApi::start(const CommandLine& cl, const std::string& capture_unavailable,
                   const std::string& data_folder, std::ostream& out, std::string* err) {
    if (server_) return true;
    CliContext ctx;
    ctx.caps = host_caps();
    ctx.sdk_version = host_sdk_version();

    caliper_api::Options o;
    o.port = (uint16_t)(cl.api_port >= 0 ? cl.api_port : caliper_api::kDefaultPort);
    o.token = caliper_net::random_token();
    o.info_json = info_json(ctx);
    o.software_version = ctx.caps.version;
    o.capture_unavailable = capture_unavailable;
    o.encode_png = [](const uint8_t* rgba, uint32_t w, uint32_t h, std::string* png) {
        return export_encode_png(rgba, w, h, png);
    };
    o.log = [](const std::string& line) { std::cerr << "[api] " << line << std::endl; };
    auto server = std::make_unique<caliper_api::Server>(std::move(o));
    if (!server->start(err)) {
        if (cl.api_port < 0)
            *err += "; is another Caliper running with --api? Choose a port with --api-port";
        return false;
    }

    file_ = (std::filesystem::path(data_folder) / "api.json").string();
#ifdef _WIN32
    const long pid = (long)_getpid();
#else
    const long pid = (long)getpid();
#endif
    caliper_net::Json j = caliper_net::Json::object();
    j.set("version", 1);
    j.set("pid", (int64_t)pid);
    j.set("port", (uint64_t)server->port());
    j.set("token", server->token());
    j.set("url", server->base_url());
    j.set("stream_url", server->stream_url());
    file_text_ = j.dump() + "\n";
    std::string why;
    if (!caliper_net::write_hub_file(file_, file_text_, &why)) {
        out << "[api] " << why << "; the token is only in this line" << std::endl;
        file_.clear();
        file_text_.clear();
    }
    out << "[api] Caliper API on " << server->base_url() << " with the session token "
        << server->token();
    if (!file_.empty()) out << " (also in " << file_ << ")";
    out << std::endl;
    server_ = std::move(server);
    return true;
}

void AppApi::stop() {
    if (!server_) return;
    server_->stop();
    server_.reset();
    if (!file_text_.empty()) caliper_net::remove_hub_file(file_, file_text_);
    file_text_.clear();
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
