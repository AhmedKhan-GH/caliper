// caliper_acp_test_child: a small program the subprocess tests start. Its
// first argument picks what it does:
//   exit <n>              exits with code n
//   echo                  copies its input to its output until end of file
//   spew <n>              writes n bytes to stdout and n bytes to stderr, interleaved
//   env <name>            prints the variable's value, or <unset>
//   pwd                   prints its working folder
//   hang                  sleeps until it is killed
//   ignore-term           ignores SIGTERM, prints "ready" and sleeps until killed
//   close-stdin           closes its input, prints "closed" and sleeps for 2 s
//   grandchild <self>     starts `<self> hang` as a child of its own, prints its
//                         pid and sleeps until killed
//   raise-term            ends itself with SIGTERM
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <csignal>
#include <spawn.h>
#include <unistd.h>
#endif

namespace {

void sleep_forever() {
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 64;
    const std::string mode = argv[1];
    if (mode == "exit" && argc > 2) return std::atoi(argv[2]);
    if (mode == "echo") {
        char buf[4096];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof buf, stdin)) > 0) std::fwrite(buf, 1, n, stdout);
        return 0;
    }
    if (mode == "spew" && argc > 2) {
        const std::size_t total = std::strtoull(argv[2], nullptr, 10);
        std::string block(8192, 'o');
        std::string eblock(8192, 'e');
        for (std::size_t done = 0; done < total;) {
            const std::size_t n = std::min(block.size(), total - done);
            std::fwrite(block.data(), 1, n, stdout);
            std::fwrite(eblock.data(), 1, n, stderr);
            done += n;
        }
        return 0;
    }
    if (mode == "env" && argc > 2) {
        const char* v = std::getenv(argv[2]);
        std::cout << (v ? v : "<unset>");
        return 0;
    }
    if (mode == "pwd") {
        std::cout << std::filesystem::current_path().string();
        return 0;
    }
    if (mode == "hang") sleep_forever();
#ifndef _WIN32
    if (mode == "ignore-term") {
        std::signal(SIGTERM, SIG_IGN);
        std::cout << "ready" << std::endl;
        sleep_forever();
    }
    if (mode == "close-stdin") {
        close(0);
        std::cout << "closed" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 0;
    }
    if (mode == "grandchild" && argc > 2) {
        pid_t pid = 0;
        char hang[] = "hang";
        char* child_argv[] = {argv[2], hang, nullptr};
        extern char** environ;
        if (posix_spawn(&pid, argv[2], nullptr, nullptr, child_argv, environ) != 0) return 70;
        std::cout << pid << std::endl;
        sleep_forever();
    }
    if (mode == "raise-term") {
        std::raise(SIGTERM);
        return 0;
    }
#endif
    return 64;
}
