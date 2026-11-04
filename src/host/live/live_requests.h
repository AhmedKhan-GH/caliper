#pragma once
#include <string>
#include <vector>

namespace caliper_live {
// Frame-thread, nonblocking, bounded NDJSON channel. No automatic replay.
class RequestChannel {
public:
    ~RequestChannel() { close(); }
    RequestChannel() = default;
    RequestChannel(const RequestChannel&) = delete;
    RequestChannel& operator=(const RequestChannel&) = delete;
    bool connect(const std::string& session);
    void close();
    bool send(const std::string& json);
    std::vector<std::string> poll();
    bool connected() const { return fd_ >= 0; }
    const std::string& error() const { return error_; }
private:
    int fd_ = -1;
    bool connecting_ = false;
    std::string incoming_, outgoing_, error_;
    void fail(const char* reason);
};
} // namespace caliper_live
