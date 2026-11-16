#pragma once
// caliper_builder::acp::JsonRpcPeer: one end of a JSON-RPC 2.0 connection over
// newline-delimited JSON, as the stdio transport of ACP and MCP uses it. Each
// message is one line of UTF-8 JSON; a message never holds a raw newline.
//
// The peer owns two threads once started:
// - the reader reads lines, answers requests through the handlers, completes
//   pending calls and expires the calls whose timeout has passed;
// - the writer drains an outgoing queue, so no caller ever blocks on a peer
//   that has stopped reading (the app's frame thread answers permission
//   requests and must never wait on a pipe).
//
// Handlers and call callbacks run on the reader thread. A request handler may
// answer at once or later from any thread with respond() or respond_error();
// a call_sync() from inside a handler would deadlock and is refused.
//
// Every message from the other side is hostile input. A line longer than
// `max_message_bytes` is dropped without being kept in memory; a line nested
// deeper than `max_depth` is refused before it is parsed; malformed JSON, a
// batch, a message that is not JSON-RPC 2.0 and a response to an unknown id
// are dropped. Each of these is reported to the fault handler and the peer
// carries on. A request for a method that has no handler is answered with
// -32601 (method not found).
//
// Fault codes: rpc.parse_error, rpc.message_too_large, rpc.too_deep,
// rpc.invalid_message, rpc.unknown_response. Call errors: rpc.closed,
// rpc.timeout, rpc.remote_error (the remote code is in RpcResult::rpc_code),
// rpc.queue_full, rpc.bad_argument.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace caliper_builder::acp {

// A two-way byte stream: the stdio pipes of a child, or an in-process pipe.
// read() and write() follow the timeout rules of Process.
class ByteChannel {
public:
    virtual ~ByteChannel() = default;
    virtual IoStatus read(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) = 0;
    virtual IoStatus write(const char* data, std::size_t size, std::size_t* written, int timeout_ms) = 0;
    // Closes the writing side, so the other end reads end of file.
    virtual void close_write() = 0;
};

// Reads the child's stdout and writes its stdin. `process` must outlive it.
std::unique_ptr<ByteChannel> make_process_channel(Process& process);

// Two channels joined by two in-process pipes: what one writes, the other
// reads. On Windows `error` is process.unsupported.
struct ChannelPair {
    std::unique_ptr<ByteChannel> a;
    std::unique_ptr<ByteChannel> b;
    Error error;
};
ChannelPair make_pipe_channel_pair();

// The outcome of a call. `error.code` is empty on success, and `result` holds
// the result member of the response.
struct RpcResult {
    Error error;
    int rpc_code = 0;              // the remote error code, for rpc.remote_error
    nlohmann::json result;
    nlohmann::json error_data;     // the remote error's data member, if any

    bool ok() const { return error.ok(); }
};

// Standard JSON-RPC error codes, and the two ACP adds.
namespace rpc_code {
constexpr int parse_error = -32700;
constexpr int invalid_request = -32600;
constexpr int method_not_found = -32601;
constexpr int invalid_params = -32602;
constexpr int internal_error = -32603;
constexpr int request_cancelled = -32800;
constexpr int auth_required = -32000;
constexpr int resource_not_found = -32002;
}  // namespace rpc_code

struct JsonRpcOptions {
    std::size_t max_message_bytes = 8u << 20;     // one line, without its newline
    int max_depth = 128;                          // nesting of arrays and objects
    std::size_t max_queued_bytes = 64u << 20;     // outgoing bytes not yet written
    std::size_t max_pending_calls = 1024;
    int poll_ms = 25;                             // how often the threads check for a stop
};

class JsonRpcPeer {
public:
    using RequestHandler = std::function<void(const nlohmann::json& id, const nlohmann::json& params)>;
    using NotificationHandler = std::function<void(const nlohmann::json& params)>;
    using CallCallback = std::function<void(RpcResult)>;
    using FaultHandler = std::function<void(const Error& fault)>;
    using CloseHandler = std::function<void(const Error& reason)>;

    JsonRpcPeer(std::unique_ptr<ByteChannel> channel, JsonRpcOptions options = {});
    ~JsonRpcPeer();  // close()
    JsonRpcPeer(const JsonRpcPeer&) = delete;
    JsonRpcPeer& operator=(const JsonRpcPeer&) = delete;

    // Handlers are set before start().
    void on_request(const std::string& method, RequestHandler handler);
    void on_notification(const std::string& method, NotificationHandler handler);
    void on_fault(FaultHandler handler);
    // Runs once on the reader thread when the connection ends: end of file,
    // a read error or close(). Pending calls have failed with rpc.closed.
    void on_close(CloseHandler handler);

    void start();

    // Sends a request. `timeout_ms` <= 0 waits without limit. The callback
    // runs exactly once, on the reader thread (or on the calling thread when
    // the request cannot be sent). Returns the request id, or -1.
    std::int64_t call_async(const std::string& method, const nlohmann::json& params, int timeout_ms,
                            CallCallback callback);
    // Sends a request and waits for its outcome.
    RpcResult call_sync(const std::string& method, const nlohmann::json& params, int timeout_ms);
    // Moves the deadline of a pending call to `timeout_ms` from now (<= 0:
    // no limit). Returns false when the call is no longer pending.
    bool set_call_timeout(std::int64_t id, int timeout_ms);

    Error notify(const std::string& method, const nlohmann::json& params);
    Error respond(const nlohmann::json& id, const nlohmann::json& result);
    Error respond_error(const nlohmann::json& id, int code, const std::string& message,
                        const nlohmann::json& data = nullptr);

    // Waits up to `timeout_ms` for the outgoing queue to drain.
    bool flush(int timeout_ms);
    // Stops both threads, closes the writing side of the channel and fails
    // pending calls with rpc.closed. Safe to call more than once.
    void close();

    bool closed() const { return closed_.load(); }
    // When the last complete line arrived (or start() ran).
    std::chrono::steady_clock::time_point last_receive() const;

private:
    struct Pending {
        std::string method;
        std::chrono::steady_clock::time_point deadline;
        bool has_deadline = false;
        CallCallback callback;
    };

    void reader_loop();
    void writer_loop();
    void handle_line(std::string& line);
    void handle_message(nlohmann::json& msg);
    void expire_calls();
    void fault(const std::string& code, const std::string& message, const std::string& where = {});
    Error enqueue(const nlohmann::json& msg);
    void finish(const Error& reason);

    std::unique_ptr<ByteChannel> channel_;
    JsonRpcOptions options_;
    std::map<std::string, RequestHandler> request_handlers_;
    std::map<std::string, NotificationHandler> notification_handlers_;
    FaultHandler fault_handler_;
    CloseHandler close_handler_;

    std::mutex pending_mutex_;
    std::map<std::int64_t, Pending> pending_;
    std::int64_t next_id_ = 1;

    std::mutex out_mutex_;
    std::condition_variable out_cv_;
    std::deque<std::string> out_queue_;
    std::size_t out_bytes_ = 0;
    bool writing_ = false;

    std::atomic<bool> stop_{false};
    std::atomic<bool> closed_{false};
    std::atomic<bool> started_{false};
    std::atomic<std::int64_t> last_receive_ns_{0};
    std::thread reader_;
    std::thread writer_;
    std::thread::id reader_id_;
    std::mutex close_mutex_;
    bool finished_ = false;
};

// Text of an IoStatus, for messages.
const char* io_status_name(IoStatus status);

// The deepest nesting of arrays and objects in `text`, outside strings. Stops
// counting once `limit` is passed.
int json_nesting_depth(const std::string& text, int limit);

}  // namespace caliper_builder::acp
