#pragma once
// ============================================================================
// app_api.h: Caliper API v0 (api_server.h) as the caliper executable runs it,
// for the app (`caliper --api`, `caliper dev <folder> --api`) and for
// `caliper shot --api`, whose offscreen canvas is the app's windowless mode.
//
// start() makes a random session token, listens on 127.0.0.1 at --api-port
// (7341 without it, 0 for any free port), writes the API file and prints one
// line with the address, the token and the file. The API file tells programs
// on this machine where the API is: api.json in the app-data folder, written
// as the hub file of `caliper serve` is (owner only, mode 0600, through a
// temporary file) and removed by stop() when it still holds this run's text.
// It is one JSON object {"version": 1, "pid", "port", "token", "url",
// "stream_url"}.
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

    const std::string& file() const { return file_; }

private:
    std::unique_ptr<caliper_api::Server> server_;
    std::string file_, file_text_;
};

// The body of a launch that opened, or found open, applet `id`.
std::string launched_json(const std::string& id, const std::string& version, bool already_open);

}  // namespace caliper_host
