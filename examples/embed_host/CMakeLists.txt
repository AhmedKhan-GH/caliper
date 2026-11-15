# examples/embed_host — an embedder of the core on a bare native window
# (AppKit on macOS, Win32 on Windows), with an offscreen mode that opens no
# window (CALIPER_EMBED_OFFSCREEN=1, embed_host.h). It links caliper::embed and
# nothing of the caliper executable; it is not an applet and needs no ImGui,
# torch or renderer type of its own.
#
# On its own it builds against an installed caliper-embed package:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<embed prefix>
#   cmake --build build
# Inside the core build it links the in-tree caliper::embed.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(embed_host LANGUAGES CXX)
    if(NOT (APPLE OR WIN32))
        message(STATUS "embed_host has a host for macOS and Windows only")
        return()
    endif()
    if(APPLE)
        enable_language(OBJCXX)
    endif()
    find_package(caliper-embed 1.4 CONFIG REQUIRED)
endif()

if(APPLE)
    add_executable(embed_host main.mm)
    # ARC: the ObjC++ host owns NSWindow/NSView/NSTimer lifetimes.
    set_source_files_properties(main.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(embed_host PRIVATE
        caliper::embed
        "-framework Cocoa" "-framework QuartzCore" "-framework Metal")
elseif(WIN32)
    add_executable(embed_host main_win32.cpp)
    target_link_libraries(embed_host PRIVATE caliper::embed)
    # Windows has no rpath: the embedding DLL goes beside the executable.
    add_custom_command(TARGET embed_host POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:caliper::embed> $<TARGET_FILE_DIR:embed_host>
        COMMENT "Copying the embedding library next to embed_host")
else()
    return()  # Linux has no embed host yet.
endif()

# Binary lands at <build>/examples/embed_host/embed_host in the core build (the
# target's own folder: the output name equals the subdirectory, so an override
# to <build>/examples/ would collide with this generated folder).
set_target_properties(embed_host PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# In the core build, the same libtorch rpath the caliper exe uses, for the
# torch applets the host loads. A standalone build and a torch-free core build
# load no torch applet and carry no libtorch rpath.
if(NOT CALIPER_WITH_TORCH)
    # no libtorch rpath
elseif(APPLE)
    set_target_properties(embed_host PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
elseif(UNIX)
    set_target_properties(embed_host PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

# Windows has no rpath: embed_host lands in its own dir (build/examples/
# embed_host/), which the caliper-exe DLL-copy step does not cover — without
# this the host dies 0xc0000135 before main. Mirror of the exe's copy
# (root CMakeLists "Copying PyTorch DLLs"); spec §5.2 of the Windows pass.
if(MSVC AND CALIPER_WITH_TORCH)
    file(GLOB TORCH_DLLS "${LIBTORCH_DIR}/lib/*.dll")
    add_custom_command(TARGET embed_host POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:embed_host>
        COMMENT "Copying PyTorch DLLs to embed_host directory")
endif()
