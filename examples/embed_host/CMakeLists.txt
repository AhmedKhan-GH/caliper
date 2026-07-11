# examples/embed_host — the SECOND embedder of libcaliper (R4/Compass L2b).
# A bare native-window host (AppKit on macOS, Win32 on Windows) that links the
# platform core and NOTHING of the caliper exe. It is NOT an applet and is NOT
# linked into `caliper`; it proves the embed C ABI (caliper/embed.h) stands on
# its own. libcaliper PUBLIC-propagates the renderer backend + embed include
# root, so the host needs no ImGui/torch/renderer type of its own.
if(APPLE)
    add_executable(embed_host main.mm)
    # ARC: the ObjC++ host owns NSWindow/NSView/NSTimer lifetimes.
    set_source_files_properties(main.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(embed_host PRIVATE
        libcaliper caliper::sdk ${CALIPER_DEPENDENCY_LIBS}
        "-framework Cocoa" "-framework QuartzCore" "-framework Metal")
elseif(WIN32)
    add_executable(embed_host main_win32.cpp)
    target_link_libraries(embed_host PRIVATE
        libcaliper caliper::sdk ${CALIPER_DEPENDENCY_LIBS})
else()
    return()  # Linux triple is a later item (design §8); no host here yet.
endif()

target_include_directories(embed_host PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
# Binary lands at build/examples/embed_host/embed_host (target's own dir — the
# output name equals the subdir, so an override to build/examples/ would collide
# with this generated directory).
set_target_properties(embed_host PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Same libtorch rpath the caliper exe uses (deps arrive transitively via
# libcaliper -> CALIPER_DEPENDENCY_LIBS).
if(APPLE)
    set_target_properties(embed_host PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
elseif(UNIX)
    set_target_properties(embed_host PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
