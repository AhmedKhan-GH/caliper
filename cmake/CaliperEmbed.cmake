# The caliper-embed package and the embedding library it carries.
#
# `cmake --install <build> --prefix <prefix> --component embed` installs what a
# program outside this repository needs to embed the core with
# find_package(caliper-embed CONFIG):
#
#   lib/libcaliper_embed.dylib            The embedding library (macOS).
#   bin/caliper_embed.dll,
#   lib/caliper_embed.lib                 The same, and its import library (Windows).
#   include/caliper-embed/caliper/        embed.h, live.h and live_views.h, plus
#                                         the SDK's C headers they include and the
#                                         service tables caliper_core_get_service
#                                         returns (tensor.h, arrow_c.h, services/).
#   lib/cmake/caliper-embed/              The config, its version file and the
#                                         exported target caliper::embed.
#   share/caliper-embed/                  LICENSE and NOTICE.
#
# libcaliper is a static library over a dozen static libraries of its own
# (caliper_host_lib, the Metal or Vulkan backend, DuckDB, ImGui, ImPlot,
# ImPlot3D, ImGuiFileDialog, GLFW, GLEW). The package does not ship those
# archives. caliper_embed links all of them into one shared library that
# exports only the C functions the three headers declare, so an embedder links
# one file, needs none of the libraries or frameworks behind it, and keeps its
# own copy of ImGui, GLFW or DuckDB apart from the core's.
#
# The include root is include/caliper-embed, not the SDK's include/, so an
# applet that links caliper::sdk cannot reach caliper/embed.h even when both
# packages are installed into one prefix.
#
# The package version is the embed interface version of caliper/embed.h
# (CALIPER_EMBED_API_VERSION.CALIPER_EMBED_VERSION_MINOR.CALIPER_EMBED_VERSION_PATCH),
# with SameMajorVersion compatibility.
#
# Inputs: the libcaliper target. Outputs: the caliper_embed target, its alias
# caliper::embed, CALIPER_EMBED_PACKAGE_VERSION, and the install rules of the
# embed component. On platforms other than macOS and Windows nothing is defined.

if(NOT (APPLE OR WIN32))
    message(STATUS "caliper-embed: the embedding library is built on macOS and Windows only")
    return()
endif()

set(_embed_include "${CMAKE_SOURCE_DIR}/include/caliper")

function(_caliper_embed_define header name out)
    file(STRINGS "${header}" line REGEX "^#define ${name}[ \t]+[0-9]+")
    string(REGEX REPLACE "^#define ${name}[ \t]+([0-9]+).*" "\\1" value "${line}")
    if(NOT value MATCHES "^[0-9]+$")
        message(FATAL_ERROR "caliper-embed: ${header} does not define ${name}")
    endif()
    set(${out} "${value}" PARENT_SCOPE)
endfunction()
_caliper_embed_define("${_embed_include}/embed.h" CALIPER_EMBED_API_VERSION _embed_major)
_caliper_embed_define("${_embed_include}/embed.h" CALIPER_EMBED_VERSION_MINOR _embed_minor)
_caliper_embed_define("${_embed_include}/embed.h" CALIPER_EMBED_VERSION_PATCH _embed_patch)
set(CALIPER_EMBED_PACKAGE_VERSION "${_embed_major}.${_embed_minor}.${_embed_patch}")

# The exported functions are the ones the headers declare. A declaration starts
# its line with its return type; comment lines start with "/*", " *" or "//"
# and never match. The caliper_embed_install test compares the result with the
# C functions that libcaliper.a defines, so a declaration this misses fails it.
set(_embed_symbols "")
foreach(_header embed.h live.h live_views.h)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_embed_include}/${_header}")
    file(STRINGS "${_embed_include}/${_header}" _decls
        REGEX "^[A-Za-z_][A-Za-z0-9_ \t*]*[ \t*]caliper_[a-z0-9_]+[ \t]*\\(")
    foreach(_decl IN LISTS _decls)
        string(REGEX MATCH "caliper_[a-z0-9_]+[ \t]*\\(" _name "${_decl}")
        string(REGEX REPLACE "[ \t]*\\($" "" _name "${_name}")
        if(_name)
            list(APPEND _embed_symbols "${_name}")
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES _embed_symbols)
list(SORT _embed_symbols)
if(WIN32)
    # The live-session client and its listing calls (caliper_live_*) are macOS
    # only; live_unsupported.cpp stands in for the caliper_core_live_* calls.
    list(FILTER _embed_symbols EXCLUDE REGEX "^caliper_live_")
endif()

# A library target needs one source of its own; every object of the library
# comes from libcaliper and the archives behind it.
set(_embed_generated "${CMAKE_BINARY_DIR}/generated")
file(CONFIGURE OUTPUT "${_embed_generated}/caliper_embed_library.cpp"
    CONTENT "// caliper_embed: libcaliper linked into one shared library (cmake/CaliperEmbed.cmake).\n")
add_library(caliper_embed SHARED "${_embed_generated}/caliper_embed_library.cpp")
add_library(caliper::embed ALIAS caliper_embed)
target_link_libraries(caliper_embed PRIVATE libcaliper)
target_include_directories(caliper_embed INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/include>
    $<INSTALL_INTERFACE:include/caliper-embed>)
set_target_properties(caliper_embed PROPERTIES
    OUTPUT_NAME caliper_embed
    EXPORT_NAME embed
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

if(APPLE)
    # -u makes the linker load each function's object from libcaliper.a, which
    # nothing in the library itself refers to, and fails the link when the list
    # names a function that does not exist. The exported-symbols list keeps
    # every other symbol (the C++ of the core, DuckDB, ImGui, GLFW) inside the
    # library. The install name is @rpath/libcaliper_embed.dylib.
    list(TRANSFORM _embed_symbols PREPEND "_" OUTPUT_VARIABLE _embed_macho)
    list(JOIN _embed_macho "\n" _embed_list)
    set(_embed_exports "${_embed_generated}/caliper_embed.exports")
    file(CONFIGURE OUTPUT "${_embed_exports}" CONTENT "${_embed_list}\n")
    list(TRANSFORM _embed_macho PREPEND "LINKER:-u," OUTPUT_VARIABLE _embed_u)
    target_link_options(caliper_embed PRIVATE
        ${_embed_u} "LINKER:-exported_symbols_list,${_embed_exports}")
    set_target_properties(caliper_embed PROPERTIES LINK_DEPENDS "${_embed_exports}")
elseif(WIN32)
    # A module-definition file lists the exports; each entry also makes the
    # linker load that function's object from libcaliper.lib.
    list(JOIN _embed_symbols "\n    " _embed_list)
    set(_embed_def "${_embed_generated}/caliper_embed.def")
    file(CONFIGURE OUTPUT "${_embed_def}" CONTENT "EXPORTS\n    ${_embed_list}\n")
    target_sources(caliper_embed PRIVATE "${_embed_def}")
endif()

# ---- Install rules (component: embed) ----
include(CMakePackageConfigHelpers)
install(TARGETS caliper_embed EXPORT CaliperEmbedTargets
    LIBRARY DESTINATION lib COMPONENT embed
    ARCHIVE DESTINATION lib COMPONENT embed
    RUNTIME DESTINATION bin COMPONENT embed)
install(DIRECTORY include/caliper/
    DESTINATION include/caliper-embed/caliper COMPONENT embed)
install(FILES sdk/include/caliper/tensor.h sdk/include/caliper/arrow_c.h
    DESTINATION include/caliper-embed/caliper COMPONENT embed)
install(DIRECTORY sdk/include/caliper/services/
    DESTINATION include/caliper-embed/caliper/services COMPONENT embed)
install(EXPORT CaliperEmbedTargets NAMESPACE caliper::
    DESTINATION lib/cmake/caliper-embed COMPONENT embed)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/caliper-embed-config-version.cmake"
    VERSION ${CALIPER_EMBED_PACKAGE_VERSION} COMPATIBILITY SameMajorVersion)
# The config records the library's minimum macOS (CMAKE_OSX_DEPLOYMENT_TARGET).
configure_file("${CMAKE_SOURCE_DIR}/cmake/caliper-embed-config.cmake"
    "${CMAKE_BINARY_DIR}/caliper-embed-config.cmake" @ONLY)
install(FILES
    "${CMAKE_BINARY_DIR}/caliper-embed-config.cmake"
    "${CMAKE_BINARY_DIR}/caliper-embed-config-version.cmake"
    DESTINATION lib/cmake/caliper-embed COMPONENT embed)
install(FILES LICENSE NOTICE DESTINATION share/caliper-embed COMPONENT embed)
