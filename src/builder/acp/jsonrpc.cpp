#include <caliper_builder/acp/jsonrpc.h>

#include <cstring>
#include <future>
#include <vector>

#ifndef _WIN32
#include "fd_io.h"

#include <unistd.h>
#endif

namespace caliper_builder::acp {

using nlohmann::json;
using clock_type = std::chrono::steady_clock;

const char* io_status_name(IoStatus status) {
    switch (status) {
        case IoStatus::ok: return "ok";
        case IoStatus::timeout: return "timeout";
        case IoStatus::eof: return "end of file";
        case IoStatus::error: return "I/O error";
    }
    return "unknown";
}

int json_nesting_depth(const std::string& text, int limit) {
    int depth = 0;
    int deepest = 0;
    bool in_string = false;
    bool escaped = false;
    for (char c : text) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '[' || c == '{') {
            if (++depth > deepest) {
                deepest = depth;
                if (deepest > limit) return deepest;
            }
        } else if (c == ']' || c == '}') {
            if (depth > 0) --depth;
        }
    }
    return deepest;
}

// ---- channels ---------------------------------------------------------------

namespace {

class ProcessChannel final : public ByteChannel {
public:
    explicit ProcessChannel(Process& p) : p_(p) {}
    IoStatus read(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) override {
        return p_.read_stdout(buffer, capacity, got, timeout_ms);
    }
    IoStatus write(const char* data, std::size_t size, std::size_t* written, int timeout_ms) override {
        return p_.write_stdin(data, size, written, timeout_ms);
    }
    void close_write() override { p_.close_stdin(); }

private:
    Process& p_;
};

#ifndef _WIN32
class FdChannel final : public ByteChannel {
public:
    FdChannel(int read_fd, int write_fd) : r_(read_fd), w_(write_fd) {}
    ~FdChannel() override {
        if (r_ >= 0) ::close(r_);
        if (w_ >= 0) ::close(w_);
    }
    IoStatus read(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) override {
        return detail::read_fd(r_, buffer, capacity, got, timeout_ms);
    }
    IoStatus write(const char* data, std::size_t size, std::size_t* written, int timeout_ms) override {
        return detail::write_fd(w_, data, size, written, timeout_ms);
    }
    void close_write() override {
        if (w_ >= 0) ::close(w_);
        w_ = -1;
    }

private:
    int r_;
    int w_;
};
#endif

}  // namespace

std::unique_ptr<ByteChannel> make_process_channel(Process& process) {
    return std::make_unique<ProcessChannel>(process);
}

ChannelPair make_pipe_channel_pair() {
    ChannelPair pair;
#ifndef _WIN32
    int ab[2] = {-1, -1};
    int ba[2] = {-1, -1};
    if (!detail::make_pipe(ab) || !detail::make_pipe(ba)) {
        for (int fd : {ab[0], ab[1], ba[0], ba[1]})
            if (fd >= 0) ::close(fd);
        pair.error = {"process.spawn_failed", "Could not create a pipe."};
        return pair;
    }
    for (int fd : {ab[0], ab[1], ba[0], ba[1]}) detail::set_nonblocking(fd);
    detail::set_nosigpipe(ab[1]);
    detail::set_nosigpipe(ba[1]);
    pair.a = std::make_unique<FdChannel>(ba[0], ab[1]);
    pair.b = std::make_unique<FdChannel>(ab[0], ba[1]);
#else
    pair.error = {"process.unsupported", "In-process pipes are not implemented on Windows yet."};
#endif
    return pair;
}

// ---- the peer ---------------------------------------------------------------

namespace {

std::string dump_line(const json& msg) {
    // Strings are escaped, so the text holds no raw newline; invalid UTF-8 is
    // replaced instead of throwing.
    return msg.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count();
}

}  // namespace

JsonRpcPeer::JsonRpcPeer(std::unique_ptr<ByteChannel> channel, JsonRpcOptions options)
    : channel_(std::move(channel)), options_(options) {
    last_receive_ns_ = now_ns();
}

JsonRpcPeer::~JsonRpcPeer() { close(); }

void JsonRpcPeer::on_request(const std::string& method, RequestHandler handler) {
    request_handlers_[method] = std::move(handler);
}

void JsonRpcPeer::on_notification(const std::string& method, NotificationHandler handler) {
    notification_handlers_[method] = std::move(handler);
}

void JsonRpcPeer::on_fault(FaultHandler handler) { fault_handler_ = std::move(handler); }

void JsonRpcPeer::on_close(CloseHandler handler) { close_handler_ = std::move(handler); }

void JsonRpcPeer::start() {
    if (started_.exchange(true) || !channel_) return;
    last_receive_ns_ = now_ns();
    writer_ = std::thread([this] { writer_loop(); });
    reader_ = std::thread([this] { reader_loop(); });
    reader_id_ = reader_.get_id();
}

std::chrono::steady_clock::time_point JsonRpcPeer::last_receive() const {
    return clock_type::time_point(std::chrono::nanoseconds(last_receive_ns_.load()));
}

void JsonRpcPeer::fault(const std::string& code, const std::string& message, const std::string& where) {
    if (!fault_handler_) return;
    try {
        fault_handler_(Error(code, message, where));
    } catch (...) {
    }
}

Error JsonRpcPeer::enqueue(const json& msg) {
    std::string line;
    try {
        line = dump_line(msg);
    } catch (...) {
        return {"rpc.bad_argument", "The message could not be written as JSON."};
    }
    line += '\n';
    {
        std::lock_guard<std::mutex> lock(out_mutex_);
        if (closed_ || stop_) return {"rpc.closed", "The connection is closed."};
        if (out_bytes_ + line.size() > options_.max_queued_bytes)
            return {"rpc.queue_full", "The other side is not reading; the outgoing queue is full."};
        out_bytes_ += line.size();
        out_queue_.push_back(std::move(line));
    }
    out_cv_.notify_all();
    return {};
}

std::int64_t JsonRpcPeer::call_async(const std::string& method, const json& params, int timeout_ms,
                                     CallCallback callback) {
    auto fail = [&](Error e) {
        RpcResult r;
        r.error = std::move(e);
        r.error.where = method;
        if (callback) callback(std::move(r));
    };
    if (closed_ || stop_) {
        fail({"rpc.closed", "The connection is closed."});
        return -1;
    }
    std::int64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_.size() >= options_.max_pending_calls) {
            id = -1;
        } else {
            id = next_id_++;
            Pending p;
            p.method = method;
            p.callback = std::move(callback);
            if (timeout_ms > 0) {
                p.has_deadline = true;
                p.deadline = clock_type::now() + std::chrono::milliseconds(timeout_ms);
            }
            pending_.emplace(id, std::move(p));
        }
    }
    if (id < 0) {
        fail({"rpc.queue_full", "Too many calls are waiting for an answer."});
        return -1;
    }
    json msg = {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}};
    if (!params.is_null()) msg["params"] = params;
    Error e = enqueue(msg);
    if (e) {
        CallCallback cb;
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            auto it = pending_.find(id);
            if (it != pending_.end()) {
                cb = std::move(it->second.callback);
                pending_.erase(it);
            }
        }
        if (cb) {
            RpcResult r;
            r.error = e;
            r.error.where = method;
            cb(std::move(r));
        }
        return -1;
    }
    return id;
}

RpcResult JsonRpcPeer::call_sync(const std::string& method, const json& params, int timeout_ms) {
    if (started_ && std::this_thread::get_id() == reader_id_) {
        RpcResult r;
        r.error = {"rpc.bad_argument", "A handler cannot wait for a call on the reader thread.", method};
        return r;
    }
    auto promise = std::make_shared<std::promise<RpcResult>>();
    std::future<RpcResult> future = promise->get_future();
    call_async(method, params, timeout_ms, [promise](RpcResult r) { promise->set_value(std::move(r)); });
    return future.get();
}

bool JsonRpcPeer::set_call_timeout(std::int64_t id, int timeout_ms) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return false;
    it->second.has_deadline = timeout_ms > 0;
    it->second.deadline = clock_type::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);
    return true;
}

Error JsonRpcPeer::notify(const std::string& method, const json& params) {
    json msg = {{"jsonrpc", "2.0"}, {"method", method}};
    if (!params.is_null()) msg["params"] = params;
    return enqueue(msg);
}

Error JsonRpcPeer::respond(const json& id, const json& result) {
    return enqueue(json{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}});
}

Error JsonRpcPeer::respond_error(const json& id, int code, const std::string& message, const json& data) {
    json err = {{"code", code}, {"message", message}};
    if (!data.is_null()) err["data"] = data;
    return enqueue(json{{"jsonrpc", "2.0"}, {"id", id}, {"error", err}});
}

bool JsonRpcPeer::flush(int timeout_ms) {
    std::unique_lock<std::mutex> lock(out_mutex_);
    return out_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms),
                            [&] { return (out_queue_.empty() && !writing_) || stop_; }) &&
           out_queue_.empty() && !writing_;
}

void JsonRpcPeer::close() {
    stop_ = true;
    out_cv_.notify_all();
    const bool on_reader = started_ && std::this_thread::get_id() == reader_id_;
    if (writer_.joinable() && writer_.get_id() != std::this_thread::get_id()) writer_.join();
    if (on_reader) return;  // the reader finishes by itself; the destructor joins it
    if (channel_) channel_->close_write();
    if (reader_.joinable()) reader_.join();
    finish(Error("rpc.closed", "The connection was closed."));
}

void JsonRpcPeer::finish(const Error& reason) {
    {
        std::lock_guard<std::mutex> lock(close_mutex_);
        if (finished_) return;
        finished_ = true;
    }
    stop_ = true;
    closed_ = true;
    out_cv_.notify_all();
    std::map<std::int64_t, Pending> pending;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending.swap(pending_);
    }
    for (auto& [id, p] : pending) {
        (void)id;
        if (!p.callback) continue;
        RpcResult r;
        r.error = {"rpc.closed", "The connection closed before the answer came: " + reason.message, p.method};
        try {
            p.callback(std::move(r));
        } catch (...) {
        }
    }
    if (close_handler_) {
        try {
            close_handler_(reason);
        } catch (...) {
        }
    }
}

void JsonRpcPeer::expire_calls() {
    const auto now = clock_type::now();
    std::vector<Pending> expired;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.has_deadline && it->second.deadline <= now) {
                expired.push_back(std::move(it->second));
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& p : expired) {
        if (!p.callback) continue;
        RpcResult r;
        r.error = {"rpc.timeout", "No answer came in time.", p.method};
        try {
            p.callback(std::move(r));
        } catch (...) {
        }
    }
}

void JsonRpcPeer::writer_loop() {
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lock(out_mutex_);
            out_cv_.wait_for(lock, std::chrono::milliseconds(options_.poll_ms),
                             [&] { return stop_.load() || !out_queue_.empty(); });
            if (stop_) break;
            if (out_queue_.empty()) continue;
            line = std::move(out_queue_.front());
            out_queue_.pop_front();
            writing_ = true;
        }
        std::size_t done = 0;
        bool failed = false;
        while (done < line.size() && !stop_) {
            std::size_t n = 0;
            const IoStatus st = channel_->write(line.data() + done, line.size() - done, &n, options_.poll_ms);
            if (st == IoStatus::ok) done += n;
            else if (st != IoStatus::timeout) {
                failed = true;
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(out_mutex_);
            out_bytes_ -= line.size();
            writing_ = false;
            if (failed) {
                out_bytes_ = 0;
                out_queue_.clear();
            }
        }
        out_cv_.notify_all();
        if (failed) {
            fault("rpc.closed", "The other side stopped reading its input.");
            stop_ = true;  // the reader sees it and finishes the connection
            break;
        }
    }
    std::lock_guard<std::mutex> lock(out_mutex_);
    writing_ = false;
    out_cv_.notify_all();
}

void JsonRpcPeer::reader_loop() {
    std::string line;
    bool discarding = false;
    std::size_t discarded = 0;
    std::vector<char> buf(64 * 1024);
    Error reason("rpc.closed", "The connection was closed.");
    auto last_expiry = clock_type::now();

    while (!stop_) {
        std::size_t got = 0;
        const IoStatus st = channel_->read(buf.data(), buf.size(), &got, options_.poll_ms);
        const auto now = clock_type::now();
        if (now - last_expiry >= std::chrono::milliseconds(options_.poll_ms)) {
            expire_calls();
            last_expiry = now;
        }
        if (st == IoStatus::timeout) continue;
        if (st == IoStatus::eof) {
            reason = {"rpc.closed", "The other side closed its output."};
            break;
        }
        if (st == IoStatus::error) {
            reason = {"rpc.closed", "Reading from the other side failed."};
            break;
        }
        std::size_t start = 0;
        while (start < got) {
            const char* nl = static_cast<const char*>(std::memchr(buf.data() + start, '\n', got - start));
            const std::size_t end = nl ? static_cast<std::size_t>(nl - buf.data()) : got;
            const std::size_t len = end - start;
            if (discarding) {
                discarded += len;
            } else if (line.size() + len > options_.max_message_bytes) {
                discarding = true;
                discarded = line.size() + len;
                std::string().swap(line);
            } else {
                line.append(buf.data() + start, len);
            }
            if (!nl) break;
            if (discarding) {
                fault("rpc.message_too_large",
                      "A message of " + std::to_string(discarded) + " bytes is larger than the limit of " +
                          std::to_string(options_.max_message_bytes) + " bytes and was dropped.");
                discarding = false;
                discarded = 0;
                last_receive_ns_ = now_ns();
            } else {
                handle_line(line);
                line.clear();
            }
            start = end + 1;
        }
    }
    if (!line.empty() && !discarding) handle_line(line);  // a last line without its newline
    finish(reason);
}

void JsonRpcPeer::handle_line(std::string& line) {
    last_receive_ns_ = now_ns();
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find_first_not_of(" \t\r") == std::string::npos) return;
    const int depth = json_nesting_depth(line, options_.max_depth);
    if (depth > options_.max_depth) {
        fault("rpc.too_deep", "A message nested deeper than " + std::to_string(options_.max_depth) +
                                  " levels was dropped.");
        return;
    }
    json msg = json::parse(line, nullptr, /*allow_exceptions=*/false);
    if (msg.is_discarded()) {
        fault("rpc.parse_error", "A line of " + std::to_string(line.size()) + " bytes is not valid JSON.");
        return;
    }
    try {
        handle_message(msg);
    } catch (const std::exception& e) {
        fault("rpc.invalid_message", std::string("A message could not be handled: ") + e.what());
    } catch (...) {
        fault("rpc.invalid_message", "A message could not be handled.");
    }
}

void JsonRpcPeer::handle_message(json& msg) {
    if (msg.is_array()) {
        fault("rpc.invalid_message", "A batch of messages was dropped; batches are not used.");
        return;
    }
    if (!msg.is_object()) {
        fault("rpc.invalid_message", "A message that is not a JSON object was dropped.");
        return;
    }
    auto jr = msg.find("jsonrpc");
    if (jr == msg.end() || !jr->is_string() || jr->get<std::string>() != "2.0") {
        fault("rpc.invalid_message", "A message without \"jsonrpc\": \"2.0\" was dropped.");
        return;
    }
    auto id_it = msg.find("id");
    const bool has_id = id_it != msg.end() && !id_it->is_null();
    const bool good_id = has_id && (id_it->is_string() || id_it->is_number_integer());
    auto method_it = msg.find("method");

    if (method_it != msg.end()) {
        if (!method_it->is_string()) {
            fault("rpc.invalid_message", "A message whose method is not a string was dropped.");
            if (good_id) respond_error(*id_it, rpc_code::invalid_request, "The method must be a string.");
            return;
        }
        const std::string method = method_it->get<std::string>();
        json params = msg.contains("params") ? msg["params"] : json();
        if (!params.is_null() && !params.is_object() && !params.is_array()) {
            fault("rpc.invalid_message", "A message whose params are not an object was dropped.", method);
            if (good_id) respond_error(*id_it, rpc_code::invalid_params, "The params must be an object.");
            return;
        }
        if (has_id) {
            if (!good_id) {
                fault("rpc.invalid_message", "A request whose id is neither a string nor an integer was dropped.",
                      method);
                return;
            }
            auto h = request_handlers_.find(method);
            if (h == request_handlers_.end()) {
                respond_error(*id_it, rpc_code::method_not_found, "Method not found: " + method);
                return;
            }
            try {
                h->second(*id_it, params);
            } catch (...) {
                respond_error(*id_it, rpc_code::internal_error, "The request could not be handled.");
            }
            return;
        }
        auto h = notification_handlers_.find(method);
        if (h != notification_handlers_.end()) {
            try {
                h->second(params);
            } catch (...) {
                fault("rpc.invalid_message", "A notification could not be handled.", method);
            }
        }
        return;  // unknown notifications are ignored, as JSON-RPC allows
    }

    const bool has_result = msg.contains("result");
    const bool has_error = msg.contains("error");
    if (!has_id || (has_result == has_error)) {
        fault("rpc.invalid_message", "A message that is neither a request, a notification nor a response was dropped.");
        return;
    }
    if (!id_it->is_number_integer()) {
        fault("rpc.unknown_response", "A response to an id this side never sent was dropped.");
        return;
    }
    const std::int64_t id = id_it->get<std::int64_t>();
    Pending p;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto it = pending_.find(id);
        if (it != pending_.end()) {
            p = std::move(it->second);
            pending_.erase(it);
            found = true;
        }
    }
    if (!found) {
        fault("rpc.unknown_response", "A response to id " + std::to_string(id) + ", which is not pending, was dropped.");
        return;
    }
    RpcResult r;
    if (has_error) {
        const json& e = msg["error"];
        r.error.code = "rpc.remote_error";
        r.error.where = p.method;
        if (e.is_object()) {
            if (e.contains("code") && e["code"].is_number_integer()) r.rpc_code = e["code"].get<int>();
            if (e.contains("message") && e["message"].is_string()) {
                std::string m = e["message"].get<std::string>();
                if (m.size() > 2000) m.resize(2000);
                r.error.message = m;
            }
            if (e.contains("data")) r.error_data = e["data"];
        }
        if (r.error.message.empty()) r.error.message = "The other side answered with an error.";
    } else {
        r.result = std::move(msg["result"]);
    }
    if (p.callback) p.callback(std::move(r));
}

}  // namespace caliper_builder::acp
