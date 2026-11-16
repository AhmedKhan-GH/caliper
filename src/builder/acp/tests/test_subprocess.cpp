// The subprocess helper, against caliper_acp_test_child. POSIX only.
#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <string>
#include <thread>

#include <unistd.h>

using namespace caliper_builder::acp;
using caliper_builder::Error;

namespace {

const std::string kChild = CALIPER_ACP_TEST_CHILD;

SpawnOptions child(std::vector<std::string> args) {
    SpawnOptions so;
    so.argv = {kChild};
    so.argv.insert(so.argv.end(), args.begin(), args.end());
    return so;
}

std::filesystem::path tmp_dir(const std::string& name) {
    std::filesystem::path p = std::filesystem::path(CALIPER_BUILDER_TEST_TMP) / name;
    std::filesystem::create_directories(p);
    return p;
}

std::string read_first_line(Process& p, int timeout_ms = 3000) {
    std::string out;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    char c = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        std::size_t got = 0;
        const IoStatus st = p.read_stdout(&c, 1, &got, 50);
        if (st == IoStatus::ok) {
            if (c == '\n') return out;
            out += c;
        } else if (st != IoStatus::timeout) {
            break;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("exit codes and signals are reported") {
    for (int code : {0, 3, 255}) {
        RunResult r = run_process(child({"exit", std::to_string(code)}), {});
        REQUIRE(r.error.ok());
        CHECK(r.status.exited);
        CHECK(r.status.code == code);
        CHECK(r.status.success() == (code == 0));
        CHECK_FALSE(r.timed_out);
    }
    RunResult r = run_process(child({"raise-term"}), {});
    CHECK_FALSE(r.status.exited);
    CHECK(r.status.signal == SIGTERM);
    CHECK(r.status.describe() == "signal " + std::to_string(SIGTERM));
}

TEST_CASE("a missing program or a bad argument is refused before anything runs") {
    Error e;
    SpawnOptions so;
    so.argv = {"caliper-no-such-program-anywhere"};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.not_found");

    so.argv = {"/nonexistent/folder/program"};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.not_found");

    so.argv = {};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.bad_argument");

    so = child({"exit", "0"});
    so.env = {{"BAD=NAME", "x"}};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.bad_argument");

    so.env = {{"SECRET", std::string("top\0secret", 10)}};
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.bad_argument");
    CHECK(e.message.find("top") == std::string::npos);
    CHECK(e.where == "SECRET");

    so = child({"exit", "0"});
    so.argv.push_back(std::string("a\0b", 3));
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK(e.code == "process.bad_argument");

    RunResult r = run_process(SpawnOptions{}, {});
    CHECK(r.error.code == "process.bad_argument");
}

TEST_CASE("large output on both streams does not deadlock") {
    const std::size_t n = 4u << 20;
    const auto t0 = std::chrono::steady_clock::now();
    RunResult r = run_process(child({"spew", std::to_string(n)}), {});
    REQUIRE(r.error.ok());
    CHECK(r.status.success());
    CHECK(r.out.size() == n);
    CHECK(r.err.size() == n);
    CHECK(r.out.find_first_not_of('o') == std::string::npos);
    CHECK(r.err.find_first_not_of('e') == std::string::npos);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20));
}

TEST_CASE("output past the limit is read and dropped") {
    RunLimits limits;
    limits.max_output_bytes = 1000;
    RunResult r = run_process(child({"spew", "200000"}), limits);
    CHECK(r.status.success());
    CHECK(r.out.size() == 1000);
    CHECK(r.err.size() == 1000);
    CHECK(r.out_truncated);
    CHECK(r.err_truncated);
}

TEST_CASE("input is fed while output is read") {
    RunLimits limits;
    limits.stdin_data.assign(1u << 20, 'i');
    for (std::size_t i = 0; i < limits.stdin_data.size(); i += 97) limits.stdin_data[i] = '\n';
    RunResult r = run_process(child({"echo"}), limits);
    CHECK(r.status.success());
    CHECK(r.out == limits.stdin_data);
}

TEST_CASE("a child past its time limit is terminated") {
    RunLimits limits;
    limits.timeout_ms = 200;
    const auto t0 = std::chrono::steady_clock::now();
    RunResult r = run_process(child({"hang"}), limits);
    CHECK(r.timed_out);
    CHECK(r.status.signal == SIGTERM);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
}

TEST_CASE("terminate escalates to SIGKILL after the grace period") {
    Error e;
    auto p = Process::spawn(child({"ignore-term"}), &e);
    REQUIRE(p);
    REQUIRE(read_first_line(*p) == "ready");
    ExitStatus st;
    CHECK_FALSE(p->try_wait(&st));
    const auto t0 = std::chrono::steady_clock::now();
    st = p->terminate(300);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(st.signal == SIGKILL);
    CHECK(ms >= 290);
    CHECK(ms < 5000);
    // Terminating again is harmless.
    CHECK(p->terminate(300).signal == SIGKILL);
}

TEST_CASE("a hanging child stops at SIGTERM and wait_for reports it") {
    Error e;
    auto p = Process::spawn(child({"hang"}), &e);
    REQUIRE(p);
    CHECK(p->pid() > 0);
    ExitStatus st;
    CHECK_FALSE(p->wait_for(100, &st));
    st = p->terminate(2000);
    CHECK(st.signal == SIGTERM);
    CHECK(p->try_wait(&st));
}

TEST_CASE("terminate reaches the child's own children through its process group") {
    Error e;
    auto p = Process::spawn(child({"grandchild", kChild}), &e);
    REQUIRE(p);
    const std::string line = read_first_line(*p);
    REQUIRE_FALSE(line.empty());
    const pid_t grandchild = static_cast<pid_t>(std::stol(line));
    REQUIRE(grandchild > 0);
    CHECK(kill(grandchild, 0) == 0);
    p->terminate(1000);
    bool gone = false;
    for (int i = 0; i < 200 && !gone; ++i) {
        gone = kill(grandchild, 0) != 0 && errno == ESRCH;
        if (!gone) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(gone);
}

TEST_CASE("the environment is inherited, extended and trimmed") {
    ::setenv("CALIPER_ACP_TEST_INHERITED", "from-parent", 1);
    RunResult r = run_process(child({"env", "CALIPER_ACP_TEST_INHERITED"}), {});
    CHECK(r.out == "from-parent");

    SpawnOptions so = child({"env", "CALIPER_ACP_TEST_SET"});
    so.env = {{"CALIPER_ACP_TEST_SET", "value with spaces = and equals"}};
    CHECK(run_process(so, {}).out == "value with spaces = and equals");

    so = child({"env", "CALIPER_ACP_TEST_INHERITED"});
    so.env_remove = {"CALIPER_ACP_TEST_INHERITED"};
    CHECK(run_process(so, {}).out == "<unset>");

    so = child({"env", "CALIPER_ACP_TEST_INHERITED"});
    so.env = {{"CALIPER_ACP_TEST_INHERITED", "replaced"}};
    CHECK(run_process(so, {}).out == "replaced");

    so = child({"env", "PATH"});
    so.inherit_environment = false;
    CHECK(run_process(so, {}).out == "<unset>");
    ::unsetenv("CALIPER_ACP_TEST_INHERITED");
}

TEST_CASE("the child starts in the working folder it was given") {
    const auto dir = tmp_dir("subprocess-cwd");
    SpawnOptions so = child({"pwd"});
    so.cwd = dir.string();
    RunResult r = run_process(so, {});
    REQUIRE(r.status.success());
    CHECK(std::filesystem::equivalent(std::filesystem::path(r.out), dir));

    so.cwd = (dir / "missing").string();
    Error e;
    CHECK(Process::spawn(so, &e) == nullptr);
    CHECK_FALSE(e.ok());
}

TEST_CASE("writing to a child that closed its input reports eof, not SIGPIPE") {
    Error e;
    auto p = Process::spawn(child({"close-stdin"}), &e);
    REQUIRE(p);
    REQUIRE(read_first_line(*p) == "closed");
    const std::string data(256 * 1024, 'w');
    IoStatus st = IoStatus::ok;
    for (int i = 0; i < 20 && st == IoStatus::ok; ++i) st = p->write_stdin_all(data.data(), data.size(), 500);
    CHECK(st == IoStatus::eof);
    p->terminate(500);
    // Writing after close_stdin() is an error, not a crash.
    std::size_t n = 0;
    CHECK(p->write_stdin("x", 1, &n, 10) == IoStatus::error);
}

TEST_CASE("reads report timeout and then end of file") {
    Error e;
    auto p = Process::spawn(child({"echo"}), &e);
    REQUIRE(p);
    char buf[16];
    std::size_t got = 0;
    CHECK(p->read_stdout(buf, sizeof buf, &got, 30) == IoStatus::timeout);
    CHECK(p->write_stdin_all("hi", 2, 1000) == IoStatus::ok);
    p->close_stdin();
    std::string out;
    IoStatus st = IoStatus::timeout;
    for (int i = 0; i < 100 && st != IoStatus::eof; ++i) {
        st = p->read_stdout(buf, sizeof buf, &got, 50);
        if (st == IoStatus::ok) out.append(buf, got);
    }
    CHECK(out == "hi");
    CHECK(st == IoStatus::eof);
    ExitStatus status;
    CHECK(p->wait_for(2000, &status));
    CHECK(status.success());
}

TEST_CASE("find_executable searches a PATH list") {
    CHECK(find_executable("sh", "/nonexistent:/bin:/usr/bin").size() > 0);
    CHECK(find_executable("caliper-no-such-program", "/bin:/usr/bin").empty());
    CHECK(find_executable(kChild, "") == kChild);
    CHECK(find_executable("/nonexistent/x", "/bin").empty());
    CHECK(find_executable("", "/bin").empty());
    // A folder is not a program.
    CHECK(find_executable("bin", "/").empty());
}
