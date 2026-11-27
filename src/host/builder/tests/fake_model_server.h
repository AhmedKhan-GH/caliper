#pragma once
// A fake OpenAI-compatible model server on loopback for the widget agent's
// settings tests and caliper-agent-settings-shot: the script
// src/builder/agent/tests/fake_openai_serve.py run with the test interpreter,
// which prints its base URL and serves until its input closes. Nothing leaves
// the machine and no key is used.
#include <caliper_builder/acp/subprocess.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host::test {

class FakeModelServer {
public:
    // `agent_dir`: src/builder/agent. Null, with the reason, when it does not start.
    static std::unique_ptr<FakeModelServer> start(const std::string& python, const std::string& agent_dir,
                                                  const std::vector<std::string>& args, std::string* why) {
        namespace acp = caliper_builder::acp;
        acp::SpawnOptions o;
        o.argv = {python, "-I", agent_dir + "/tests/fake_openai_serve.py"};
        o.argv.insert(o.argv.end(), args.begin(), args.end());
        o.env = {{"CALIPER_AGENT_OFFLINE", "1"}, {"PYTHONDONTWRITEBYTECODE", "1"}};
        caliper_builder::Error error;
        std::unique_ptr<acp::Process> p = acp::Process::spawn(o, &error);
        if (!p) {
            if (why) *why = error.message;
            return nullptr;
        }
        std::string line;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (line.find('\n') == std::string::npos && std::chrono::steady_clock::now() < deadline) {
            char buf[256];
            std::size_t got = 0;
            const acp::IoStatus s = p->read_stdout(buf, sizeof buf, &got, 500);
            if (s == acp::IoStatus::ok) line.append(buf, got);
            else if (s == acp::IoStatus::eof || s == acp::IoStatus::error) break;
        }
        const std::size_t nl = line.find('\n');
        if (nl == std::string::npos || line.rfind("http://127.0.0.1:", 0) != 0) {
            std::string err;
            char ebuf[512];
            std::size_t egot = 0;
            while (p->read_stderr(ebuf, sizeof ebuf, &egot, 0) == acp::IoStatus::ok && egot > 0)
                err.append(ebuf, egot);
            if (why) *why = err.empty() ? "the fake server printed no address" : "the fake server printed no address: " + err;
            p->terminate(2000);
            return nullptr;
        }
        std::unique_ptr<FakeModelServer> s(new FakeModelServer);
        s->url_ = line.substr(0, nl);
        s->process_ = std::move(p);
        return s;
    }

    ~FakeModelServer() {
        if (process_) process_->terminate(3000);
    }

    const std::string& base_url() const { return url_; }

private:
    FakeModelServer() = default;
    std::string url_;
    std::unique_ptr<caliper_builder::acp::Process> process_;
};

}  // namespace caliper_host::test
