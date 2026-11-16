# caliper-embed CMake package. find_package(caliper-embed CONFIG) provides:
#
#   caliper::embed  The embedding library: the core (applet loader, services,
#                   renderer, tensor bridge) in one shared library that exports
#                   only the C functions of caliper/embed.h, caliper/live.h and
#                   caliper/live_views.h. Linking it adds the include root
#                   include/caliper-embed, which also holds the SDK's C headers
#                   those headers include and the service tables that
#                   caliper_core_get_service returns.
#
# caliper-embed_VERSION is the embed interface version (major.minor.patch, the
# parts of CALIPER_EMBED_VERSION); a request for 1.x accepts any 1.y with y >= x.
#
# On macOS the library's install name is @rpath/libcaliper_embed.dylib: a
# program built in its build folder finds it through the build rpath, and an
# installed program needs an rpath to <prefix>/lib. On Windows the program
# needs bin/caliper_embed.dll beside it or on PATH.
#
# The library's minimum macOS, 14.0 by default, is CALIPER_MACOS_MIN. On macOS
# it becomes CMAKE_OSX_DEPLOYMENT_TARGET unless the project,
# -DCMAKE_OSX_DEPLOYMENT_TARGET or MACOSX_DEPLOYMENT_TARGET chose one.
set(CALIPER_MACOS_MIN "@CMAKE_OSX_DEPLOYMENT_TARGET@")
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${CALIPER_MACOS_MIN}")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/CaliperEmbedTargets.cmake")
