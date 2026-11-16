#pragma once
// A test helper: caliper-mcp as a child process, with its standard output read
// message by message and its standard error collected on a thread, for the
// process tests and the acceptance test of the MCP route. POSIX only.
#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caliper_builder::mcp::test {

using acp::EnvVar;
using acp::ExitStatus;
using acp::IoStatus;
using acp::Process;
using acp::SpawnOptions;
using nlohmann::json;
using McpClock = std::chrono::steady_clock;

inline json modern_meta() {
    return {{"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
            {"io.modelcontextprotocol/clientCapabilities", json::object()},
            {"io.modelcontextprotocol/clientInfo", {{"name", "process-test"}, {"version", "1"}}}};
}

// caliper-mcp as a child, with its stdout read line by line and its stderr
// collected on a thread.
class McpChild {
public:
    explicit McpChild(const std::vector<std::string>& args, const std::vector<EnvVar>& env = {}) {
        SpawnOptions so;
        so.argv = {CALIPER_MCP_EXE};
        so.argv.insert(so.argv.end(), args.begin(), args.end());
        so.env_remove = {"CALIPER_DATA_DIR"};
        so.env = env;
        Error e;
        proc_ = Process::spawn(so, &e);
        REQUIRE_MESSAGE(proc_, e.message);
        err_thread_ = std::thread([this] {
            char b[4096];
            for (;;) {
                std::size_t got = 0;
                const IoStatus st = proc_->read_stderr(b, sizeof b, &got, 100);
                if (st == IoStatus::ok) {
                    std::lock_guard<std::mutex> g(mu_);
                    err_.append(b, got);
                } else if (st != IoStatus::timeout || stop_) {
                    break;
                }
            }
        });
    }
    ~McpChild() {
        if (proc_) proc_->terminate(2000);
        stop_ = true;
        if (err_thread_.joinable()) err_thread_.join();
    }

    void send_raw(const std::string& bytes) {
        REQUIRE(proc_->write_stdin_all(bytes.data(), bytes.size(), 10000) == IoStatus::ok);
    }
    void send(const json& j) { send_raw(j.dump() + "\n"); }

    // The next message, or null when none comes in time or stdout ends.
    json next(int timeout_ms) {
        const auto end = McpClock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const auto nl = buf_.find('\n');
            if (nl != std::string::npos) {
                const std::string line = buf_.substr(0, nl);
                buf_.erase(0, nl + 1);
                out_ += line + "\n";
                json j = json::parse(line, nullptr, false);
                REQUIRE_MESSAGE(!j.is_discarded(), "stdout carried a line that is not JSON: " << line);
                REQUIRE(j.is_object());
                CHECK(j["jsonrpc"] == "2.0");
                return j;
            }
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - McpClock::now()).count();
            if (left <= 0) return nullptr;
            char b[65536];
            std::size_t got = 0;
            const IoStatus st = proc_->read_stdout(b, sizeof b, &got, int(std::min<long long>(left, 100)));
            if (st == IoStatus::ok) buf_.append(b, got);
            else if (st == IoStatus::eof || st == IoStatus::error) return nullptr;
        }
    }

    // The response with this id; other messages are kept in `notes`.
    json await(const json& id, int timeout_ms = 20000) {
        const auto end = McpClock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - McpClock::now()).count();
            REQUIRE_MESSAGE(left > 0, "no response with id " << id.dump());
            json m = next(int(left));
            REQUIRE_MESSAGE(!m.is_null(), "stdout ended before the response with id " << id.dump());
            if (m.contains("id") && m["id"] == id && !m.contains("method")) return m;
            notes.push_back(std::move(m));
        }
    }

    json request(const json& id, const std::string& method, json params, bool modern = true) {
        if (modern) params["_meta"] = modern_meta();
        send({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}});
        return await(id);
    }

    json call(const json& id, const std::string& tool, const json& args, bool modern = true) {
        const json r = request(id, "tools/call", {{"name", tool}, {"arguments", args}}, modern);
        REQUIRE_MESSAGE(r.contains("result"), r.dump());
        return r["result"];
    }

    // Closes stdin, reads what is left on stdout and waits for the exit.
    ExitStatus finish(int timeout_ms = 10000) {
        proc_->close_stdin();
        const auto end = McpClock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - McpClock::now()).count();
            if (left <= 0) break;
            json m = next(int(left));
            if (m.is_null()) break;
            notes.push_back(std::move(m));
        }
        ExitStatus st;
        REQUIRE(proc_->wait_for(timeout_ms, &st));
        stop_ = true;
        if (err_thread_.joinable()) err_thread_.join();
        return st;
    }

    std::string all_out() const { return out_ + buf_; }
    std::string all_err() const {
        std::lock_guard<std::mutex> g(mu_);
        return err_;
    }

    std::vector<json> notes;

private:
    std::unique_ptr<Process> proc_;
    std::thread err_thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mu_;
    std::string err_;
    std::string buf_;
    std::string out_;
};

}  // namespace caliper_builder::mcp::test
