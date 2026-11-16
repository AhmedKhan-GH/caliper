// caliper build <folder> (cli_build.h): CMake against the installed SDK, then
// the checks of caliper check.
#include "cli_build.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <thread>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace caliper_host {

namespace fs = std::filesystem;

namespace {

bool holds_sdk(const fs::path& prefix) {
    std::error_code ec;
    return !prefix.empty() && fs::is_regular_file(prefix / sdk_config_file(), ec);
}

bool is_program(const fs::path& p) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return false;
#ifdef _WIN32
    return true;
#else
    return ::access(p.c_str(), X_OK) == 0;
#endif
}

bool contains(const std::string& text, const char* what) { return text.find(what) != std::string::npos; }

#ifndef _WIN32
// Runs argv (argv[0] is a path), with its standard output and error copied to
// `out` as they come and kept in *text. The exit code, or -1 when it could
// not start or did not exit normally.
int run_process(const std::vector<std::string>& args, std::ostream& out, std::string* text) {
    text->clear();
    int fds[2];
    if (::pipe(fds) != 0) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    std::vector<std::string> copy = args;
    std::vector<char*> argv;
    for (std::string& a : copy) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(fds[1]);
    if (rc != 0) {
        ::close(fds[0]);
        return -1;
    }
    char buf[8192];
    for (;;) {
        const ssize_t n = ::read(fds[0], buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        text->append(buf, (size_t)n);
        out.write(buf, n);
        out.flush();
    }
    ::close(fds[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

}  // namespace

SdkSearch find_sdk_prefix(const char* sdk_dir_env, const std::string& exe_dir, const char* home) {
    SdkSearch s;
    if (sdk_dir_env && *sdk_dir_env) {
        s.tried.push_back(sdk_dir_env);
        std::error_code ec;
        if (holds_sdk(sdk_dir_env))
            s.prefix = fs::absolute(sdk_dir_env, ec).lexically_normal().string();
        else
            s.error = std::string("CALIPER_SDK_DIR is ") + sdk_dir_env + ", which holds no Caliper SDK (no " +
                      sdk_config_file() + " in it)";
        return s;
    }
    std::vector<fs::path> candidates;
    if (!exe_dir.empty()) {
        const fs::path exe = fs::path(exe_dir).lexically_normal();
        candidates.push_back(exe.parent_path());
        candidates.push_back(exe);
    }
    if (home && *home) candidates.push_back(fs::path(home) / "caliper-sdk");
    for (const fs::path& p : candidates) {
        if (std::find(s.tried.begin(), s.tried.end(), p.string()) != s.tried.end()) continue;
        s.tried.push_back(p.string());
        if (holds_sdk(p)) {
            s.prefix = p.string();
            break;
        }
    }
    return s;
}

std::vector<std::string> default_cmake_fallbacks() {
#ifdef __APPLE__
    return {"/Applications/CMake.app/Contents/bin/cmake"};
#else
    return {};
#endif
}

std::string find_cmake(const char* cmake_env, const char* path_env, const std::vector<std::string>& fallbacks,
                       std::string* error) {
    if (cmake_env && *cmake_env) {
        if (is_program(cmake_env)) return cmake_env;
        *error = std::string("CALIPER_CMAKE is ") + cmake_env + ", which is not a program";
        return {};
    }
#ifdef _WIN32
    const char sep = ';';
    const char* name = "cmake.exe";
#else
    const char sep = ':';
    const char* name = "cmake";
#endif
    const std::string path = path_env ? path_env : "";
    size_t at = 0;
    while (at <= path.size()) {
        size_t end = path.find(sep, at);
        if (end == std::string::npos) end = path.size();
        const std::string dir = path.substr(at, end - at);
        if (!dir.empty() && is_program(fs::path(dir) / name)) return (fs::path(dir) / name).string();
        at = end + 1;
    }
    for (const std::string& f : fallbacks)
        if (is_program(f)) return f;
    *error = "cmake was not found: install CMake 3.18 or later and put it on PATH, or set CALIPER_CMAKE to its path";
    return {};
}

BuildLack configure_lack(const std::string& t) {
    if (contains(t, "No CMAKE_CXX_COMPILER could be found") || contains(t, "CMAKE_CXX_COMPILER not set") ||
        contains(t, "is not a full path to an existing compiler tool") ||
        contains(t, "The CXX compiler identification is unknown") ||
        contains(t, "is not able to compile a simple test program"))
        return BuildLack::Compiler;
    if (contains(t, "unable to find a build program corresponding to") ||
        contains(t, "CMAKE_MAKE_PROGRAM is not set"))
        return BuildLack::BuildTool;
    if (contains(t, "for package \"caliper-sdk\" that is compatible")) return BuildLack::SdkVersion;
    if (contains(t, "provided by \"caliper-sdk\"") || contains(t, "Findcaliper-sdk.cmake"))
        return BuildLack::Sdk;
    return BuildLack::Unknown;
}

int run_build(const CommandLine& cl, const CliContext& ctx, std::ostream& out, std::ostream& err) {
    std::error_code ec;
    const fs::path folder = fs::absolute(cl.operands[0], ec).lexically_normal();
    if (ec || !fs::is_directory(folder, ec)) {
        err << "caliper: not a folder: " << cl.operands[0] << "\n";
        return 1;
    }
    if (!fs::is_regular_file(folder / "CMakeLists.txt", ec)) {
        err << "caliper: " << folder.string()
            << " has no CMakeLists.txt; build takes an applet's folder, such as one that caliper new made\n";
        return 1;
    }
#ifdef _WIN32
    (void)ctx;
    (void)out;
    err << "caliper: " << windows_unavailable_message("caliper build")
        << "; configure and build the folder with CMake and the SDK's prefix on CMAKE_PREFIX_PATH, "
           "then run caliper check on its build folder\n";
    return 1;
#else
    std::string why;
    const std::string cmake =
        find_cmake(std::getenv("CALIPER_CMAKE"), std::getenv("PATH"), default_cmake_fallbacks(), &why);
    if (cmake.empty()) {
        err << "caliper: " << why << "\n";
        return 1;
    }
    const SdkSearch sdk = find_sdk_prefix(std::getenv("CALIPER_SDK_DIR"), ctx.exe_dir, std::getenv("HOME"));
    if (!sdk.error.empty()) {
        err << "caliper: " << sdk.error << "\n";
        return 1;
    }
    if (!sdk.prefix.empty()) {
        out << "using the Caliper SDK in " << sdk.prefix << "\n";
    } else {
        out << "no Caliper SDK in";
        for (size_t i = 0; i < sdk.tried.size(); ++i) out << (i ? ", " : " ") << sdk.tried[i];
        out << "; CMake searches its own places\n";
    }

    const fs::path build = folder / "build";
    std::vector<std::string> configure = {cmake, "-S", folder.string(), "-B", build.string(),
                                          "-DCMAKE_BUILD_TYPE=Release"};
    if (!sdk.prefix.empty()) configure.push_back("-DCMAKE_PREFIX_PATH=" + sdk.prefix);
    out << "configuring " << build.string() << " with " << cmake << std::endl;
    std::string text;
    int rc = run_process(configure, out, &text);
    if (rc != 0) {
        switch (configure_lack(text)) {
            case BuildLack::Compiler:
                err << "caliper: CMake found no working C++ compiler; install the Xcode command line "
                       "tools (xcode-select --install), or name a compiler in CXX\n";
                break;
            case BuildLack::BuildTool:
                err << "caliper: CMake found no build tool for its generator; install the Xcode "
                       "command line tools for make, or Ninja\n";
                break;
            case BuildLack::Sdk:
                err << "caliper: no Caliper SDK was found. Install it from a Caliper build with "
                       "cmake --install <build folder> --prefix <prefix> --component sdk, then set "
                       "CALIPER_SDK_DIR=<prefix> or use the prefix ~/caliper-sdk\n";
                break;
            case BuildLack::SdkVersion:
                err << "caliper: the Caliper SDK found is older than the version the project asks for; "
                       "install the SDK of this Caliper (" << (ctx.sdk_version.empty() ? "unknown" : ctx.sdk_version)
                    << ")\n";
                break;
            case BuildLack::Unknown:
                if (rc < 0)
                    err << "caliper: could not run " << cmake << "\n";
                else
                    err << "caliper: configuring " << folder.string() << " failed (see the CMake output above)\n";
                break;
        }
        return 1;
    }

    const unsigned jobs = std::max(1u, std::thread::hardware_concurrency());
    out << "building " << build.string() << std::endl;
    rc = run_process({cmake, "--build", build.string(), "--config", "Release", "--parallel", std::to_string(jobs)},
                     out, &text);
    if (rc != 0) {
        err << "caliper: building " << folder.string() << " failed (see the output above)\n";
        return 1;
    }

    // The checks of `caliper check`, on the build folder (its applets/).
    CommandLine check = cl;
    check.command = CommandLine::Command::Check;
    check.operands = {build.string()};
    return run_check(check, ctx, out, err);
#endif
}

}  // namespace caliper_host
