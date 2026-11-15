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
include("${CMAKE_CURRENT_LIST_DIR}/CaliperEmbedTargets.cmake")
