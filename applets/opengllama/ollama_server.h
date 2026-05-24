#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>

namespace httplib { class Server; }

class OpenGllamaApplet;

class OllamaServer {
public:
    using TokenCallback = std::function<bool(const std::string& piece)>;

    explicit OllamaServer(OpenGllamaApplet* applet);
    ~OllamaServer();

    void start(int port = 11435);
    void stop();
    bool is_running() const { return running_; }
    int port() const { return port_; }

private:
    void setup_routes();

    OpenGllamaApplet* applet_;
    httplib::Server* server_ = nullptr;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
    int port_ = 11435;
    std::atomic<uint64_t> request_seq_{0};
    std::mutex inference_mutex_;
};
