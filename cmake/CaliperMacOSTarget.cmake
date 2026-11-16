# The minimum macOS of every Apple binary Caliper builds: macOS 14 (Sonoma).
# Include this before project(), where CMake reads CMAKE_OSX_DEPLOYMENT_TARGET
# to choose the compiler's -mmacosx-version-min. The installed SDK and embed
# packages carry the same value to the projects built against them
# (sdk/cmake/caliper-sdk-info.cmake.in, cmake/caliper-embed-config.cmake).
#
# Why 14.0: the JSON writer of wire protocol v3 (src/host/net/json.cpp) uses
# std::to_chars for double, which Apple's libc++ provides from macOS 13.3; and
# pip recognises only macosx_<major>_0 platform tags for macOS 11 and later,
# so a wheel tagged 13_3 would install nowhere. 14.0 covers the first and gives
# the clean tag macosx_14_0_arm64.
#
# It is a default: -DCMAKE_OSX_DEPLOYMENT_TARGET=<version>, or the environment
# variable MACOSX_DEPLOYMENT_TARGET, wins. An empty value counts as unset,
# since CMake creates the cache entry empty on Apple platforms.
set(CALIPER_MACOS_MIN "14.0")
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${CALIPER_MACOS_MIN}" CACHE STRING
        "The minimum macOS version of the binaries; Caliper needs 14.0 or later" FORCE)
endif()
