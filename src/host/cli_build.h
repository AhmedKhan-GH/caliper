#pragma once
// ============================================================================
// caliper build <folder>: configures and builds an applet folder against the
// installed Caliper SDK with CMake, then runs the checks of `caliper check`
// on the result (host_cli.h). It opens no window and starts no app.
//
//   cmake -S <folder> -B <folder>/build -DCMAKE_BUILD_TYPE=Release
//         [-DCMAKE_PREFIX_PATH=<sdk prefix>]
//   cmake --build <folder>/build --config Release --parallel
//   caliper check <folder>/build
//
// THE SDK. The prefix is found without being named: the first of
//   1. CALIPER_SDK_DIR, when it is set and not empty (it must hold the SDK);
//   2. the prefix the executable is installed in: the folder above it, then
//      its own folder;
//   3. caliper-sdk in the home folder, where the tutorials install it;
// that holds lib/cmake/caliper-sdk/caliper-sdk-config.cmake. When none does,
// CMake searches its own places (CMAKE_PREFIX_PATH in the environment and the
// system prefixes), and a project that still finds no SDK gets a message that
// says how to install one.
//
// CMAKE. CALIPER_CMAKE when it is set and not empty, else `cmake` on PATH,
// else, on macOS, the command of CMake.app in /Applications. Its generator
// and compiler are CMake's defaults, or what CMAKE_GENERATOR and CXX say.
// What it prints goes to standard output as it comes; when it fails for want
// of a compiler, a build tool or the SDK, a line says which.
//
// POSIX only for now (tested on macOS): on Windows `build` says so and exits
// with 1.
// ============================================================================
#include "host_cli.h"

#include <ostream>
#include <string>
#include <vector>

namespace caliper_host {

// The package file whose presence marks an SDK prefix.
inline const char* sdk_config_file() { return "lib/cmake/caliper-sdk/caliper-sdk-config.cmake"; }

struct SdkSearch {
    std::string prefix;               // "" when no candidate holds the SDK
    std::vector<std::string> tried;   // the candidates, in order
    std::string error;                // CALIPER_SDK_DIR names a folder without the SDK
};
// The SDK prefix by the order above. `sdk_dir_env` is CALIPER_SDK_DIR, `home`
// the home folder (both may be null).
SdkSearch find_sdk_prefix(const char* sdk_dir_env, const std::string& exe_dir, const char* home);

// The cmake to run by the order above, or "" with *error. `path_env` is PATH.
// `fallbacks` are tried after PATH (CMake.app on macOS).
std::string find_cmake(const char* cmake_env, const char* path_env,
                       const std::vector<std::string>& fallbacks, std::string* error);
std::vector<std::string> default_cmake_fallbacks();

// What a failed configure says it lacked, from CMake's output.
enum class BuildLack { Unknown, Compiler, BuildTool, Sdk, SdkVersion };
BuildLack configure_lack(const std::string& cmake_output);

int run_build(const CommandLine& cl, const CliContext& ctx, std::ostream& out, std::ostream& err);

}  // namespace caliper_host
