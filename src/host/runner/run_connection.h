#pragma once
// The model runner's own viewer connection to one managed run
// (design/hosted-runs.md, Runner Connection): wire protocol 3.1 over
// caliper_net's WebSocket client, to the stream link that the run's script
// wrote into its link file. It is separate from the app's attachment: it
// reads the run state, the step, the parameters and the actions from
// CONTROLS, sends CONTROL requests such as the actions caliper.checkpoint and
// caliper.stop, and subscribes to every series view of the run, whose values
// it keeps for the run sources run:<id>/<view>.
//
// The handshake: HELLO as a viewer with the capability controls; the script's
// server answers WELCOME, SESSIONS and VIEWS, attaches the connection to its
// one session and sends CONTROLS. The connection answers each PING.
//
// Threads: one RunConnection belongs to the runner's thread; only wake() may
// be called from another thread. Nothing here throws.
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_net {
class Client;
}

namespace caliper_runner {

// The stream URL (ws://127.0.0.1:<port>/v3/stream?token=<token>) of a page
// link (http://127.0.0.1:<port>/?token=<token>) or of a stream link; "" with
// *error for a link that is neither, or names a host other than this
// computer.
std::string stream_url_of(const std::string& link, std::string* error);

class RunConnection {
public:
    enum class State { idle, connecting, open, closed };

    // The answer to one CONTROL request: the first, and for an action that
    // waited for the script's checkpoint, the later one with the same id.
    struct Reply {
        std::uint32_t request_id = 0;
        std::string op;
        std::string name;
        std::string status;          // ok, queued or error
        nlohmann::json result;
        std::string message;         // an action's text
        std::string error_code;      // status error
        std::string error_message;
    };

    // The values of one series view, from the absolute index `first`.
    struct Series {
        std::vector<float> values;
        std::uint64_t first = 0;     // the absolute index of values[0]
        std::uint64_t total = 0;     // the values the script appended so far
    };

    // `max_values`: the newest values kept for each series.
    explicit RunConnection(std::size_t max_values = 262144);
    ~RunConnection();
    RunConnection(const RunConnection&) = delete;
    RunConnection& operator=(const RunConnection&) = delete;

    // Starts the connection to the link; false with *error for a bad link or
    // a socket that fails at once.
    bool open(const std::string& link, std::string* error);
    // Reads what arrived for at most `timeout_ms`, answers it, subscribes to
    // new series views and sends nothing else. True when the run state, a
    // series, a reply or the connection's state changed.
    bool poll(int timeout_ms);
    // Ends a poll() that waits. Any thread.
    void wake();
    // Closes the connection.
    void close();

    State state() const { return state_; }
    // Why the connection closed, in one sentence; "" while it is open.
    const std::string& close_reason() const { return close_reason_; }
    // Whether WELCOME allowed control.
    bool control_allowed() const { return control_allowed_; }
    // The session id of the script, from WELCOME; "" before.
    const std::string& session() const { return session_; }
    // The latest CONTROLS as {run, params, actions}; null before the first.
    const nlohmann::json& controls() const { return controls_; }
    // Sends one CONTROL: `op` pause, resume, step (with `count`), set (with
    // `name` and `value`) or run (with `name`). Its id, or 0 with *error
    // when the connection is not open or the message breaks a limit.
    std::uint32_t send_control(const std::string& op, const std::string& name, const nlohmann::json& value,
                               std::uint32_t count, std::string* error);
    nlohmann::json interpret(const std::string& method, const nlohmann::json& params, std::uint32_t request = 0);
    // The replies that came since the last call, oldest first.
    std::vector<Reply> take_replies();
    // The series views by name.
    const std::map<std::string, Series>& series() const { return series_; }

private:
    void on_message(const std::vector<std::uint8_t>& bytes);
    void send_bytes(const std::vector<std::uint8_t>& bytes);

    std::unique_ptr<caliper_net::Client> client_;
    State state_ = State::idle;
    std::string close_reason_;
    bool control_allowed_ = false;
    bool interpretation_ = false;
    std::uint64_t interpretation_revision_ = 0;
    std::map<std::uint32_t, std::pair<std::string, Reply>> interpretation_replies_;
    std::string session_;
    nlohmann::json controls_;
    std::vector<Reply> replies_;
    std::map<std::string, Series> series_;
    std::map<std::string, std::uint32_t> subscribed_;   // view name -> sub id
    std::map<std::uint32_t, std::string> sub_views_;    // sub id -> view name
    std::uint32_t next_sub_ = 1;
    std::uint32_t next_request_ = 1;
    std::size_t max_values_;
};

}  // namespace caliper_runner
