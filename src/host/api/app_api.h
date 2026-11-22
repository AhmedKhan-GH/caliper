#pragma once
// ============================================================================
// app_api.h: Caliper API v0 (api_server.h) as the caliper executable runs it,
// for the app (`caliper --api`, `caliper dev <folder> --api`) and for
// `caliper shot --api`, whose offscreen canvas is the app's windowless mode.
//
// start() makes a random session token for the person (origin api) and a
// second one for outside agents (origin mcp), listens on 127.0.0.1 at
// --api-port (7341 without it, 0 for any free port), writes the API file and
// prints one line with the address, the person's token and the file. The API
// file tells programs on this machine where the API is: api.json in the
// app-data folder, written as the hub file of `caliper serve` is (owner only,
// mode 0600, through a temporary file) and removed by stop() when it still
// holds this run's text. It is one JSON object {"version": 1, "pid", "port",
// "token", "url", "stream_url", "spec_url"}, whose "token" is the mcp token:
// commands sent with it are reviewed in the app, and the person's own token is
// in no file. spec_url is the browser viewer of spec applets with the viewer
// token (api_server.h, THE SPEC PAGE); a saved spec applet opens at spec_url +
// "&applet=<id>", and start() prints that line too.
//
// An agent the app starts (the side chat's agent, the widget agent's
// sidecar) gets a token of its own, origin agent, in an API file of its own
// (issue_agent_file(): the same shape, mode 0600, in the app-data folder),
// revoked and removed when that agent's session ends (revoke_agent_file())
// and when the API stops.
//
// The command layer's endpoints are served once set_command_gateway() gives
// the server a gateway; take() then also hands their requests to it.
//
// The caller answers the calls of the frame thread itself (main.cpp, shot.cpp):
// applets with applets_json() (host_cli.h), launch by its own navigation, and
// capture by reading back a frame it drew: shot reads its offscreen canvas at
// the end of a frame, and the app the frame its window presents
// (window_captures.h).
// ============================================================================
#include "api/api_server.h"
#include "host_cli.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace caliper_host {

// True when the command line asks for the API (--api) or CALIPER_API is 1.
// On Windows, where main() refuses --api, CALIPER_API is ignored.
bool api_wanted(const CommandLine& cl);

class AppApi {
public:
    AppApi() = default;
    ~AppApi() { stop(); }
    AppApi(const AppApi&) = delete;
    AppApi& operator=(const AppApi&) = delete;

    // Starts the API (see above). capture_unavailable is the reason
    // GET /v0/capture.png answers 501 when the app cannot read its frame back,
    // "" when it can. data_folder is the app-data folder of the API file.
    // False with *err when it cannot listen.
    bool start(const CommandLine& cl, const std::string& capture_unavailable,
               const std::string& data_folder, std::ostream& out, std::string* err);
    // Stops the API and removes the API file.
    void stop();
    bool running() const { return server_ != nullptr; }
    std::string url() const { return server_ ? server_->base_url() : std::string(); }

    // The frame thread's side (api_server.h).
    std::vector<caliper_api::Call> take() {
        return server_ ? server_->take() : std::vector<caliper_api::Call>();
    }
    void reply(uint64_t ticket, int status, std::string json) {
        if (server_) server_->reply(ticket, status, std::move(json));
    }
    void reply_error(uint64_t ticket, int status, const std::string& message) {
        reply(ticket, status, caliper_api::error_json(message));
    }
    void reply_capture(uint64_t ticket, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
        if (server_) server_->reply_capture(ticket, w, h, std::move(rgba));
    }

    // The command layer behind POST /v0/commands/{name} (api/command_gateway.h;
    // api/session_gateway.h makes one over the app's builder session). Set it
    // on the frame thread once the session is open, and clear it with nullptr
    // before the session closes. It is kept across stop() and start().
    void set_command_gateway(std::shared_ptr<caliper_api::CommandGateway> gateway) {
        gateway_ = gateway;
        if (server_) server_->set_command_gateway(std::move(gateway));
    }

    const std::string& file() const { return file_; }

    // An API file for one agent session, with a new token of the origin
    // agent for `client` ("chat" for the side chat's agent, "widget" for the
    // widget agent's sidecar; Server::issue_token()), whose commands the
    // command layer governs by that agent's mode; its path, or "" with the
    // reason when the API is not running or the file cannot be written.
    std::string issue_agent_file(std::string* why, const std::string& client);
    // Revokes that file's token and removes the file.
    void revoke_agent_file(const std::string& path);

    // The link of the browser viewer of the saved spec applet `applet` (its
    // workspace id): spec_url + "&applet=<id>", with the viewer token, which
    // opens the spec page and its reads only (api_server.h, THE SPEC PAGE).
    // Never the person's session token nor the API file's: "" with the
    // reason when the link would carry either, when the API is not running,
    // when this build has no spec page, or when `applet` is not an id.
    std::string viewer_link(const std::string& applet, std::string* why) const;
    // Whether viewer_link() can give a link of `applet` once the API runs:
    // false, with the reason, for a build with no browser viewer or an id
    // that viewer_applet_id() (open_url.h) refuses. Needs no running API, so
    // that a caller asks before it starts one.
    static bool viewer_link_possible(const std::string& applet, std::string* why);

private:
    std::string file_json(const std::string& token) const;
    std::unique_ptr<caliper_api::Server> server_;
    std::shared_ptr<caliper_api::CommandGateway> gateway_;
    std::string file_, file_text_;
    std::string data_folder_;
    struct AgentFile {
        std::string path, token, text;
    };
    std::vector<AgentFile> agent_files_;
    unsigned agent_files_made_ = 0;
};

// The body of a launch that opened, or found open, applet `id`.
std::string launched_json(const std::string& id, const std::string& version, bool already_open);
// The body of a launch whose page is not open yet: `status` PENDING_APPROVAL
// when its applet.open waits for the person's approval, PENDING when the app
// cannot tell yet whether it will, and `note` saying so and how to check.
std::string launch_pending_json(const std::string& id, const std::string& version, const std::string& status,
                                const std::string& note);

}  // namespace caliper_host
