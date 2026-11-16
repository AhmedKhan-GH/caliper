# doctest v2.4.11 declares cmake_minimum_required 2.8, which CMake >= 4.0
# rejects; this floor lets FetchContent configure it without a manual flag.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(caliper_tests
    test_main.cpp
    test_abi.cpp
    abi_c_check.c
    test_manifest.cpp
    test_negotiation.cpp
    test_crash_guard.cpp
    test_watchdog.cpp
    test_sugar.cpp
    test_jobs.cpp
    test_device.cpp
    test_sugar_services.cpp
    test_mnist_idx.cpp
    test_metrics_store.cpp
    test_feed_store.cpp
    test_feed_provider.cpp
    test_artifact_store.cpp
    test_data_store.cpp
    test_tensor_bridge.cpp
    test_alloc_registry.cpp
    test_obj.cpp
    test_instance_field.cpp
    test_pulse_ring.cpp
    test_export_affordance.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp
)
# On Windows the host build provides ZLIB::ZLIB in-tree (Dependencies.cmake).
if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()
target_link_libraries(caliper_tests PRIVATE doctest::doctest caliper::sdk caliper_host_lib
    caliper::sdk_testing caliper::ui_stack ZLIB::ZLIB)
# Wire protocol v3 networking (src/host/net): the codec against the golden
# frames in tests/fixtures/wire_v3, everywhere; the WebSocket layer against its
# own server on 127.0.0.1 and the hub (src/host/net/hub.h) in-process with fake
# producers and viewers, not on Windows yet (caliper_net_tests below).
target_sources(caliper_tests PRIVATE test_net_wire.cpp)
target_sources(caliper_tests PRIVATE test_net_wire_32.cpp)
if(NOT WIN32)
    target_sources(caliper_tests PRIVATE test_net_ws.cpp test_net_hub.cpp)
    target_sources(caliper_tests PRIVATE test_net_hub_32.cpp)   # the scenes of the 3.2 draft
endif()
target_link_libraries(caliper_tests PRIVATE caliper_net)
# mnist_idx.h now lives with EmbedScope (the sole exemplar is self-contained).
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/embed_scope)
# InstanceScope's pure pose/tint/mesh reference (instance_field.h) is torch-free,
# so its unit test rides the fast caliper_tests suite.
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/instance_scope)
# PulseScope's pure ring-copy/gap logic (pulse_ring.h) is torch-free + UI-free,
# so its unit test rides the fast caliper_tests suite (feed spec §5).
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/pulse_scope)
# Export E2 affordance logic (export_affordance.h: frame budget, pacing, path
# assembly) is torch-free + UI-free, so its unit test rides caliper_tests too.
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/common)
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# The graphics layer's core on its recording backend, with no GPU (src/host/gfx).
target_sources(caliper_tests PRIVATE test_gfx_layer.cpp)
target_link_libraries(caliper_tests PRIVATE caliper_gfx)
# The colormap by keys on the CPU (src/host/gfx/colormap_keys.h) on the boundary sets of
# tests/shaders/colormap_cases.h, against the reference compiled into this binary.
target_sources(caliper_tests PRIVATE test_colormap_keys.cpp)
target_sources(caliper_tests PRIVATE test_loader.cpp)
# UI recovery after an applet fault (src/host/ui/ui_recovery.h, header-only):
# headless ImGui/ImPlot/ImPlot3D contexts handed to the throwing fixture.
target_sources(caliper_tests PRIVATE test_ui_recovery.cpp)
# Docking by role on the applet page (src/host/ui/dock_layout.h, header-only)
# on a headless ImGui context.
target_sources(caliper_tests PRIVATE test_dock_layout.cpp)
# The work area a page's bars leave, and when the jobs tray takes its strip
# (src/host/ui/work_area.h, header-only), on a headless ImGui context.
target_sources(caliper_tests PRIVATE test_work_area.cpp)
# The applet search order, copies of one id, and the command line
# (src/host/host_cli.h); the acceptance check for applets built outside the
# core runs only when CALIPER_TEST_APPLET_DIR is set.
target_sources(caliper_tests PRIVATE test_applet_search.cpp test_applet_acceptance.cpp)
# The command line's subcommands (src/host/host_cli.h) on temporary folders,
# with caliper build's own source and the header's out-of-line part, which the
# executable compiles too.
target_sources(caliper_tests PRIVATE test_cli.cpp ${CMAKE_SOURCE_DIR}/src/host/cli_build.cpp
    ${CMAKE_SOURCE_DIR}/src/host/host_cli.cpp)
# Caliper API v0 (src/host/api/api_server.h): the routes, the token, the Host
# and Origin refusals and the limits, with the server in this process; the
# ctest entry caliper_cli_api runs the same binary against `caliper shot --api`.
# Not on Windows yet (caliper_net_tests below).
if(NOT WIN32)
    target_sources(caliper_tests PRIVATE test_api.cpp)
    target_link_libraries(caliper_tests PRIVATE caliper_api)
    target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party/stb)
endif()
# On Windows the socket cases of the WebSocket layer, the hub and Caliper API
# are not built or run: every feature built on them is refused there
# (windows_unavailable in src/host/host_cli.h), and their socket and timer
# behaviour has not run on Winsock yet. This entry says so in the ctest log.
if(WIN32)
    add_test(NAME caliper_net_tests COMMAND "${CMAKE_COMMAND}" -E echo
        "skipped on Windows: the socket tests of the WebSocket layer, the hub and Caliper API do not run there yet")
    set_tests_properties(caliper_net_tests PROPERTIES SKIP_REGULAR_EXPRESSION "skipped on Windows")
endif()
# The throwing fixture (a raw descriptor that throws across the ABI) backs the
# crash guard's exception case; it builds into test-applets, not applets.
add_subdirectory(fixtures/throwing_applet)
add_dependencies(caliper_tests hello_applet throwing_applet)
target_compile_definitions(caliper_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets"
    CALIPER_TEST_FIXTURES_DIR="${CMAKE_BINARY_DIR}/test-applets"
    CALIPER_TEST_DATA_ROOT="${CMAKE_BINARY_DIR}/test-data"
    CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

add_test(NAME caliper_tests COMMAND caliper_tests)
# The stores this binary opens go to a work folder, never the user's app-data.
set_tests_properties(caliper_tests PROPERTIES ENVIRONMENT
    "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_tests")

# The scene core (src/host/scene): the store, matrices, cameras, fits, compiled draw lists
# and picks, headless and with no GPU, and the fixture draw lists through the graphics
# layer's validation on its recording backend (caliper_gfx, which caliper_scene links).
add_executable(caliper_scene_tests
    scene/scene_main.cpp
    scene/test_scene_store.cpp
    scene/test_scene_matrices.cpp
    scene/test_scene_cameras.cpp
    scene/test_scene_compile.cpp
    scene/test_scene_pick.cpp
    scene/test_scene_row.cpp
    scene/test_scene_layer.cpp
)
target_link_libraries(caliper_scene_tests PRIVATE doctest::doctest caliper_scene)
set_target_properties(caliper_scene_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME caliper_scene_tests COMMAND caliper_scene_tests)

# The colormap by keys against map_f32_to_rgba8 for all 2^32 inputs on each boundary range,
# label slow: it exits 77 (skipped) unless CALIPER_SLOW_TESTS=1, so the default run passes over
# it. Run it as a heavy job: CALIPER_SLOW_TESTS=1 ctest -L slow (6 threads at most).
find_package(Threads REQUIRED)
add_executable(caliper_colormap_exhaustive
    test_colormap_exhaustive.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp)
target_include_directories(caliper_colormap_exhaustive PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_colormap_exhaustive PRIVATE caliper::sdk Threads::Threads)
set_target_properties(caliper_colormap_exhaustive PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME caliper_colormap_exhaustive COMMAND caliper_colormap_exhaustive)
set_tests_properties(caliper_colormap_exhaustive PROPERTIES LABELS "slow" SKIP_RETURN_CODE 77 TIMEOUT 1800)
# The generated colormap_texture on Metal, through the layer as it compiles it, against the
# reference for all 2^32 inputs on the same ranges, also under the label slow (Apple only).
if(APPLE)
    add_executable(caliper_colormap_gpu_exhaustive
        test_colormap_gpu_exhaustive.mm
        ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp)
    set_source_files_properties(test_colormap_gpu_exhaustive.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(caliper_colormap_gpu_exhaustive PRIVATE caliper_gfx caliper::sdk Threads::Threads
        "-framework Metal" "-framework Foundation")
    set_target_properties(caliper_colormap_gpu_exhaustive PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_colormap_gpu_exhaustive COMMAND caliper_colormap_gpu_exhaustive)
    set_tests_properties(caliper_colormap_gpu_exhaustive PROPERTIES LABELS "slow" SKIP_RETURN_CODE 77 TIMEOUT 3600)
endif()

# caliper_colormap_cases <file> writes the colormap's boundary cases, with the C++ reference's
# RGBA for each input, for the browser rows (web/test/shaders.spec.ts, colormap.spec.ts).
add_executable(caliper_colormap_cases
    shaders/colormap_cases_tool.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp)
target_include_directories(caliper_colormap_cases PRIVATE ${CMAKE_SOURCE_DIR}/src/host ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(caliper_colormap_cases PRIVATE caliper::sdk)
set_target_properties(caliper_colormap_cases PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# The caliper executable's subcommands, run as a user runs them. Each runs and
# exits without a window (host_cli.h); `caliper dev` and plain `caliper` start
# the app, so no test runs them. The applets folder of this build stands in
# for an outside build folder. Every entry runs with a controlled environment:
# CALIPER_APPLET_PATH unset, CALIPER_USER_APPLETS_DIR in the build tree, and
# CALIPER_DATA_DIR on a folder of its own that must still not exist after the
# run (only `install` without CALIPER_USER_APPLETS_DIR may create applets/ in
# it), which proves that the subcommands open no store and create nothing in
# the app-data folder.
# ---------------------------------------------------------------------------
if(TARGET caliper)
    set(_cli_work "${CMAKE_CURRENT_BINARY_DIR}/cli-work")
    function(caliper_cli_test name)
        cmake_parse_arguments(PARSE_ARGV 1 t "DEFAULT_USER_APPLETS;EMPTY_DATA_DIR"
            "ARGS;EXIT;MATCH;ERROR;CLEAN;STAGE;ALLOW;ENV" "")
        set(data "${_cli_work}/app-data-${name}")
        set(env "--unset=CALIPER_APPLET_PATH|CALIPER_DATA_DIR=${data}")
        if(NOT t_DEFAULT_USER_APPLETS)
            string(APPEND env "|CALIPER_USER_APPLETS_DIR=${_cli_work}/user-applets")
        endif()
        if(DEFINED t_ENV)   # more of the run's environment, "|"-separated
            string(APPEND env "|${t_ENV}")
        endif()
        set(defs -DEXE=$<TARGET_FILE:caliper> "-DARGS=${t_ARGS}" -DEXIT=${t_EXIT}
                 "-DENV=${env}" "-DDATA_DIR=${data}")
        foreach(opt MATCH ERROR CLEAN STAGE ALLOW)
            if(DEFINED t_${opt})
                list(APPEND defs "-D${opt}=${t_${opt}}")
            endif()
        endforeach()
        if(t_EMPTY_DATA_DIR)
            list(APPEND defs "-DEMPTY_DATA_DIR=1")
        endif()
        add_test(NAME caliper_cli_${name}
                 COMMAND ${CMAKE_COMMAND} ${defs} -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_run.cmake)
        set_tests_properties(caliper_cli_${name} PROPERTIES LABELS cli)
    endfunction()

    caliper_cli_test(help ARGS "--help" EXIT 0 MATCH "usage: caliper")
    caliper_cli_test(bad_option ARGS "--frobnicate" EXIT 2
        ERROR "unknown option: --frobnicate")
    caliper_cli_test(applets_needs_folder ARGS "--applets|--help" EXIT 2
        ERROR "--applets needs a folder")
    caliper_cli_test(info ARGS "info" EXIT 0
        MATCH "abi_epoch 2.*service caliper\\.ui\\.v1_1")
    caliper_cli_test(list ARGS "list|--applets|${CMAKE_BINARY_DIR}/applets" EXIT 0
        MATCH "READY +dev\\.caliper\\.hello +0\\.1\\.0 ")
    caliper_cli_test(check ARGS "check|${CMAKE_BINARY_DIR}/test-applets" EXIT 0
        MATCH "1 of 1 applets pass the checks")
    # An applet's build folder, its manifests only in build/applets as
    # caliper_add_applet() leaves them: check uses that subfolder and says so.
    # Printed folders use the platform's separator, '\' on Windows, so the
    # patterns accept either.
    caliper_cli_test(check_build_folder ARGS "check|${_cli_work}/applet-build" EXIT 0
        MATCH "^using the applets in .*applet-build[/\\]applets.*1 of 1 applets pass the checks"
        CLEAN "${_cli_work}/applet-build"
        STAGE "${CMAKE_BINARY_DIR}/test-applets|${_cli_work}/applet-build/applets")
    caliper_cli_test(check_missing ARGS "check|${_cli_work}/nowhere" EXIT 1
        ERROR "not a folder")
    caliper_cli_test(install ARGS "install|${CMAKE_BINARY_DIR}/test-applets" EXIT 0
        MATCH "installed dev\\.caliper\\.throwing 0\\.1\\.0 in .*user-applets"
        CLEAN "${_cli_work}/user-applets")
    # Without CALIPER_USER_APPLETS_DIR install writes applets/ under the
    # app-data folder, and nothing else there.
    caliper_cli_test(install_default ARGS "install|${CMAKE_BINARY_DIR}/test-applets" EXIT 0
        MATCH "installed dev\\.caliper\\.throwing 0\\.1\\.0 in .*app-data-install_default[/\\]applets"
        DEFAULT_USER_APPLETS ALLOW "applets")
    caliper_cli_test(new ARGS "new|com.example.cli-made|${_cli_work}/cli-made" EXIT 0
        MATCH "created .*cli-made for applet com\\.example\\.cli-made"
        CLEAN "${_cli_work}/cli-made")
    caliper_cli_test(new_reserved ARGS "new|test|${_cli_work}/reserved" EXIT 2
        ERROR "the CMake target \"test\" is reserved")

    # The template of `caliper new`, configured, built and tested against
    # this build's installed SDK, then checked with `caliper check`.
    add_test(NAME caliper_cli_new_builds COMMAND "${CMAKE_COMMAND}"
        "-DEXE=$<TARGET_FILE:caliper>"
        "-DCALIPER_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-new-build"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DBUILD_TYPE=$<CONFIG>"
        "-DMACOS_MIN=$<$<PLATFORM_ID:Darwin>:${CMAKE_OSX_DEPLOYMENT_TARGET}>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_new_build.cmake")
    set_tests_properties(caliper_cli_new_builds PROPERTIES LABELS "cli;sdk" TIMEOUT 900)

    # `caliper build`: the template configured, built and checked against
    # this build's installed SDK, then its clear errors for a missing SDK,
    # CMake or compiler (tests/cli_build.cmake). POSIX only, as build is.
    caliper_cli_test(build_operand ARGS "build" EXIT 2 ERROR "build takes one folder")
    caliper_cli_test(build_applets ARGS "build|${_cli_work}|--applets|${_cli_work}" EXIT 2
        ERROR "--applets applies only to starting the app, dev, list and shot")
    if(NOT WIN32)
        add_test(NAME caliper_cli_build COMMAND "${CMAKE_COMMAND}"
            "-DEXE=$<TARGET_FILE:caliper>"
            "-DCALIPER_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-build"
            "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DMACOS_MIN=$<$<PLATFORM_ID:Darwin>:${CMAKE_OSX_DEPLOYMENT_TARGET}>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_build.cmake")
        set_tests_properties(caliper_cli_build PROPERTIES LABELS "cli;sdk" TIMEOUT 900)
    endif()

    # `caliper shot`: the command-line errors first, which need no GPU and
    # create nothing; an unknown id is found before any folder is made.
    caliper_cli_test(shot_needs_launch ARGS "shot|--out|${_cli_work}/x.png" EXIT 2
        ERROR "shot needs --launch <id>")
    caliper_cli_test(shot_bad_size
        ARGS "shot|--launch|dev.caliper.hello|--size|0x10|--out|${_cli_work}/x.png" EXIT 2
        ERROR "--size takes <W>x<H>")
    caliper_cli_test(shot_option_elsewhere ARGS "list|--frames|3" EXIT 2
        ERROR "--frames applies only to shot")
    caliper_cli_test(shot_unknown_applet
        ARGS "shot|--applets|${CMAKE_BINARY_DIR}/applets|--launch|dev.example.nowhere|--out|${CMAKE_CURRENT_BINARY_DIR}/x.png"
        EXIT 1 ERROR "no applet with the id dev\\.example\\.nowhere")

    # No fallback renderer (renderer core, step 5). Where the platform's
    # renderer cannot start, the app and shot refuse with one message that
    # names it and why, and exit with 1. The test variable
    # CALIPER_TEST_RENDERER_FAIL=1 makes the renderer's start fail, so these run
    # on any machine, with or without a GPU. CALIPER_RENDERER=gl, which asks for
    # the removed OpenGL renderer, is refused before any renderer is made.
    # CALIPER_EXIT_AFTER ends an app that starts after all, which then fails on
    # its exit code instead of running until the timeout. A refused run leaves
    # at most the empty data folder, which the app and shot resolve their paths
    # under before the renderer starts.
    set(_no_renderer "the (Metal|Vulkan) renderer could not start \\(CALIPER_TEST_RENDERER_FAIL=1 made its start fail\\), and there is no fallback renderer until the WebGPU fallback through Dawn lands")
    # (The pattern's "removed." stands for "removed;": a ";" would split the argument.)
    set(_gl_removed "CALIPER_RENDERER=gl asks for the OpenGL renderer, which was removed. unset it, and Caliper draws with Metal on macOS and Vulkan on Windows")
    caliper_cli_test(app_no_renderer ARGS "--hidden" EXIT 1 EMPTY_DATA_DIR
        ENV "--unset=CALIPER_RENDERER|CALIPER_TEST_RENDERER_FAIL=1|CALIPER_EXIT_AFTER=2"
        ERROR "caliper: ${_no_renderer}")
    caliper_cli_test(app_renderer_gl ARGS "--hidden" EXIT 1 EMPTY_DATA_DIR
        ENV "CALIPER_RENDERER=gl|CALIPER_EXIT_AFTER=2" ERROR "caliper: ${_gl_removed}")
    caliper_cli_test(shot_no_renderer
        ARGS "shot|--applets|${CMAKE_BINARY_DIR}/applets|--launch|dev.caliper.hello|--out|${CMAKE_CURRENT_BINARY_DIR}/x.png"
        EXIT 1 EMPTY_DATA_DIR ENV "--unset=CALIPER_RENDERER|CALIPER_TEST_RENDERER_FAIL=1"
        ERROR "caliper: shot cannot draw offscreen: ${_no_renderer}")
    caliper_cli_test(shot_renderer_gl
        ARGS "shot|--applets|${CMAKE_BINARY_DIR}/applets|--launch|dev.caliper.hello|--out|${CMAKE_CURRENT_BINARY_DIR}/x.png"
        EXIT 1 EMPTY_DATA_DIR ENV "CALIPER_RENDERER=gl"
        ERROR "caliper: shot cannot draw offscreen: ${_gl_removed}")

    # Then real shots of the hello fixture: the image has the requested size
    # and is not one flat colour (caliper_png_check), and the run leaves
    # nothing behind (tests/shot_run.cmake). Without a GPU the run is
    # reported as skipped, as the GPU suites do.
    add_executable(caliper_png_check png_check.cpp)
    target_include_directories(caliper_png_check PRIVATE ${CMAKE_SOURCE_DIR}/third_party/stb)
    set_target_properties(caliper_png_check PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_png_check hello_applet)
    foreach(mode temp data_dir)
        set(name caliper_cli_shot_${mode})
        add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}"
            "-DEXE=$<TARGET_FILE:caliper>"
            "-DCHECK=$<TARGET_FILE:caliper_png_check>"
            "-DAPPLETS=${CMAKE_BINARY_DIR}/applets"
            "-DID=dev.caliper.hello" "-DSIZE=640x400" "-DFRAMES=10"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/shot-work-${mode}" "-DMODE=${mode}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/shot_run.cmake")
        set_tests_properties(${name} PROPERTIES LABELS "cli;shot" TIMEOUT 180
            SKIP_REGULAR_EXPRESSION "run skipped: no GPU")
    endforeach()

    caliper_cli_test(serve_operand ARGS "serve|extra" EXIT 2 ERROR "serve takes no operands")
    caliper_cli_test(serve_option_elsewhere ARGS "list|--port|1" EXIT 2
        ERROR "--port applies only to serve")
    # caliper serve as a process (tests/test_cli.cpp): it prints its link,
    # serves a viewer and a producer, stops on SIGTERM with BYE to both and
    # leaves neither its hub file nor a listening socket. POSIX only.
    if(NOT WIN32)
        add_test(NAME caliper_cli_serve
                 COMMAND caliper_tests "--test-case=cli: caliper serve runs as a process*")
        set_tests_properties(caliper_cli_serve PROPERTIES LABELS cli
            ENVIRONMENT "CALIPER_TEST_CALIPER_EXE=$<TARGET_FILE:caliper>")
    endif()

    # Caliper API as a process (tests/test_api.cpp): `caliper shot --api` on the
    # hello fixture, every endpoint, the capture as a PNG of the shot's size,
    # the stream's handshake on /v3/stream and /v0/stream, the native viewer
    # attached through /v3/stream, and a clean stop on SIGTERM. No window: shot
    # draws offscreen. Without a GPU the run is reported as skipped, as the
    # shot runs do. POSIX only.
    caliper_cli_test(api_elsewhere ARGS "list|--api" EXIT 2
        ERROR "--api applies only to starting the app, dev and shot")
    caliper_cli_test(hidden_elsewhere ARGS "shot|--hidden|--launch|x|--out|${_cli_work}/x.png" EXIT 2
        ERROR "--hidden applies only to starting the app and dev")
    # On Windows the pieces that have not run there as a process yet are
    # refused before anything starts, with exit code 1 and one message
    # (host_cli.h, windows_unavailable); no window opens and nothing is written.
    if(WIN32)
        caliper_cli_test(windows_serve ARGS "serve|--port|0" EXIT 1
            ERROR "caliper serve is not available on Windows yet")
        caliper_cli_test(windows_build ARGS "build|${_cli_work}" EXIT 1
            ERROR "caliper build is not available on Windows yet")
        caliper_cli_test(windows_api ARGS "--api|--api-port|0" EXIT 1
            ERROR "Caliper API \\(--api\\) is not available on Windows yet")
        caliper_cli_test(windows_shot_api
            ARGS "shot|--launch|dev.caliper.hello|--out|${_cli_work}/x.png|--api" EXIT 1
            ERROR "Caliper API \\(--api\\) is not available on Windows yet")
        caliper_cli_test(windows_url ARGS "--url|ws://127.0.0.1:9/v3/stream?token=x" EXIT 1
            ERROR "attaching to a stream \\(--url\\) is not available on Windows yet")
    endif()
    if(NOT WIN32)
        add_test(NAME caliper_cli_api
                 COMMAND caliper_tests "--test-case=api: caliper shot --api serves*")
        # With caliper_viewer built, the native viewer also attaches through
        # /v3/stream on the API port (headless, no window).
        set(_api_env "CALIPER_TEST_CALIPER_EXE=$<TARGET_FILE:caliper>"
                     "CALIPER_TEST_APPLETS=${CMAKE_BINARY_DIR}/applets")
        if(TARGET caliper_viewer)
            list(APPEND _api_env "CALIPER_TEST_VIEWER_EXE=$<TARGET_FILE:caliper_viewer>")
        endif()
        set_tests_properties(caliper_cli_api PROPERTIES LABELS "cli;shot;api" TIMEOUT 180
            ENVIRONMENT "${_api_env}"
            SKIP_REGULAR_EXPRESSION "run skipped: no GPU")
        # The real app with its window never shown (--hidden, the mechanism of
        # tests/glfw_background.h) and Caliper API: every endpoint, the capture
        # of the hidden window's frame, the stream on /v3/stream, and a normal
        # close on SIGTERM. Skipped without a Metal device.
        add_test(NAME caliper_cli_app_hidden
                 COMMAND caliper_tests "--test-case=api: the hidden app serves*")
        set_tests_properties(caliper_cli_app_hidden PROPERTIES LABELS "cli;api" TIMEOUT 180
            ENVIRONMENT "${_api_env}"
            SKIP_REGULAR_EXPRESSION "run skipped: no GPU")
        # The app's live page as a 3.1 viewer (tests/test_net_hub.cpp): the
        # hidden app attached with --url through an in-process hub, its run bar
        # clicked by scripted ImGui input, with and without --allow-control.
        if(APPLE)
            add_test(NAME caliper_app_live_controls
                     COMMAND caliper_tests "--test-case=app: the hidden app's run bar*")
            set_tests_properties(caliper_app_live_controls PROPERTIES LABELS "cli;live" TIMEOUT 240
                ENVIRONMENT "${_api_env}"
                SKIP_REGULAR_EXPRESSION "run skipped: no GPU")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# caliper_gfx_tests — windowed, pixel-exact tensor->texture proof (§16).
# Separate binary because it pulls in the platform's renderer (NOT
# caliper_host_lib): it links the renderer + bridge sources directly into the
# exe/gfx-test scope. Registered under the "gfx" label so `ctest -L gfx` selects
# it. Without a GUI session or a device every case skips (gpu_skip.h): counted,
# and the binary exits 77, which SKIP_RETURN_CODE turns into ctest "Skipped"
# rather than "Passed".
# CALIPER_REQUIRE_GPU=1 makes those skips failures on a lane that must have a
# GPU.
add_executable(caliper_gfx_tests
    gfx/gfx_main.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp
)
target_include_directories(caliper_gfx_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_gfx_tests PRIVATE
    doctest::doctest caliper::sdk caliper::ui_stack ${CALIPER_DEPENDENCY_LIBS})
set_target_properties(caliper_gfx_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# On Apple, the §16 matrix runs on the Metal backend (C5). Link the
# Metal HostRenderer backend and compile the harness itself as Objective-C++
# with ARC so it can allocate the raw MTLBuffers the device paths consume and
# blit textures back for verification.
if(APPLE)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_metal_backend)
    target_compile_definitions(caliper_gfx_tests PRIVATE CALIPER_HAVE_METAL=1)
    set_source_files_properties(gfx/gfx_main.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    # The graphics layer's rows on Metal (src/host/gfx), on the same renderer's device.
    target_sources(caliper_gfx_tests PRIVATE gfx/gfx_layer.cpp)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_gfx)
    target_include_directories(caliper_gfx_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party/stb)
    set_source_files_properties(gfx/gfx_layer.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    # The scene core's rows: scenes compiled by caliper_scene and drawn through the layer on Metal.
    target_sources(caliper_gfx_tests PRIVATE gfx/gfx_scene.cpp)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_scene)
    set_source_files_properties(gfx/gfx_scene.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    # The scene view's rows: scenes from wire protocol 3.2 messages, applied to the live views'
    # scene view (src/host/live/live_scene.cpp, compiled in here as libcaliper compiles it).
    target_sources(caliper_gfx_tests PRIVATE gfx/gfx_scene_view.cpp
        ${CMAKE_SOURCE_DIR}/src/host/live/live_scene.cpp)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_net)
    target_compile_definitions(caliper_gfx_tests PRIVATE CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
    set_source_files_properties(gfx/gfx_scene_view.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    # The colormap by keys through the layer: the texture path's compute form and the points'
    # vertex stage, from layer and imported buffers, against map_f32_to_rgba8.
    target_sources(caliper_gfx_tests PRIVATE gfx/gfx_colormap.cpp)
    set_source_files_properties(gfx/gfx_colormap.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
endif()

# On Windows the §16 matrix runs on the Vulkan backend (CPU-staged path),
# plus hardware-gated CUDA device-path + alloc_shared byte-exact tests. Link the
# Vulkan HostRenderer backend and compile cuda_driver.cpp in (the backend refs
# it, and the CUDA tests build device tensors through it).
if(WIN32)
    target_sources(caliper_gfx_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host/cuda_driver.cpp)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_vulkan_backend)
    target_compile_definitions(caliper_gfx_tests PRIVATE CALIPER_HAVE_VULKAN=1)
endif()

add_test(NAME caliper_gfx_tests COMMAND caliper_gfx_tests)
set_tests_properties(caliper_gfx_tests PROPERTIES LABELS "gfx" SKIP_RETURN_CODE 77)

# caliper_shader_tests: the generated shader programs (src/host/gfx/shaders/generated) on
# Metal, headless. Every MSL program compiles and every entry and checked pair becomes a
# pipeline, and the colormap by keys matches map_f32_to_rgba8 byte for byte on its boundary
# cases. Skips (77) without a Metal device, as the gfx suite does.
if(APPLE)
    add_executable(caliper_shader_tests
        shaders/shaders_metal.mm
        ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp)
    target_include_directories(caliper_shader_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
    target_link_libraries(caliper_shader_tests PRIVATE doctest::doctest caliper::sdk
        "-framework Metal" "-framework Foundation")
    set_source_files_properties(shaders/shaders_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_shader_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_shader_tests COMMAND caliper_shader_tests)
    set_tests_properties(caliper_shader_tests PROPERTIES LABELS "gfx" SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------
# caliper_export_tests — caliper.export.v1 battery (Rung E1). Links libcaliper
# (host_services vends the export service + the composition reaches its g_bridge/
# g_renderer) and the Metal backend, and drives the vended service as an embedder
# would: bind a renderer -> import a quad -> view_png/sequence -> decode the PNG
# back (stb_image, TEST-ONLY) and byte-compare. The live cases skip with no GPU
# (counted; exit 77 -> ctest "Skipped", gpu_skip.h); the sidecar-golden /
# atomic-write / sugar cases are GPU-free. Own "export" label so `ctest -L gfx`
# (49/49) is untouched.
# ---------------------------------------------------------------------------
add_executable(caliper_export_tests test_export.cpp)
target_include_directories(caliper_export_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/host
    ${CMAKE_SOURCE_DIR}/third_party/stb)
target_link_libraries(caliper_export_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk caliper::sdk_testing
    caliper::ui_stack ${CALIPER_DEPENDENCY_LIBS})
set_target_properties(caliper_export_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    target_compile_definitions(caliper_export_tests PRIVATE CALIPER_HAVE_METAL=1)
    # ObjC++ + ARC so the harness can allocate the MTLBuffers the imported-
    # geometry path consumes (mirrors caliper_gfx_tests).
    set_source_files_properties(test_export.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    set_target_properties(caliper_export_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
if(WIN32)
    target_compile_definitions(caliper_export_tests PRIVATE CALIPER_HAVE_VULKAN=1)
endif()
add_test(NAME caliper_export_tests COMMAND caliper_export_tests)
# The stores this binary opens go to a work folder, never the user's app-data.
set_tests_properties(caliper_export_tests PROPERTIES ENVIRONMENT
    "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_export_tests")
set_tests_properties(caliper_export_tests PROPERTIES LABELS "export"
    SKIP_RETURN_CODE 77)

# ---------------------------------------------------------------------------
# caliper_torch_tests — the applet-side torch adapter (adapters/torch.hpp).
# A SEPARATE binary because it links libtorch: keeping torch out of
# caliper_tests spares the fast unit suite the torch link cost. Registered
# under the "torch" label so `ctest -L torch` selects it; the MPS cases skip
# when no MPS device is present (counted; exit 77 -> ctest "Skipped").
# ---------------------------------------------------------------------------
if(CALIPER_WITH_TORCH)   # torch-linking test binaries; closed below
# Windows has no rpath: a test binary that links libtorch finds c10.dll /
# torch_cpu.dll only beside itself or on PATH. Without them it dies 0xc0000135
# before main, and on a desktop session (a CI runner included) the loader's
# "DLL was not found" dialog holds the process open until ctest's timeout.
# One copy into the tests' output dir serves every torch-linked test binary
# (and the torch applets caliper_embed_tests loads), mirroring the caliper
# exe's "Copying PyTorch DLLs" step in the root CMakeLists.
if(MSVC)
    file(GLOB _caliper_test_torch_dlls "${LIBTORCH_DIR}/lib/*.dll")
    set(_caliper_test_torch_dlls_stamp "${CMAKE_CURRENT_BINARY_DIR}/torch_dlls.stamp")
    add_custom_command(OUTPUT "${_caliper_test_torch_dlls_stamp}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_caliper_test_torch_dlls} "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_caliper_test_torch_dlls_stamp}"
        DEPENDS ${_caliper_test_torch_dlls}
        COMMENT "Copying PyTorch DLLs next to the test executables")
    add_custom_target(caliper_test_torch_dlls DEPENDS "${_caliper_test_torch_dlls_stamp}")
endif()
# Each torch-linked test target calls this, so the DLLs land before it runs.
function(caliper_test_needs_torch_dlls target)
    if(TARGET caliper_test_torch_dlls)
        add_dependencies(${target} caliper_test_torch_dlls)
    endif()
endfunction()

add_executable(caliper_torch_tests test_torch_adapter.cpp)
caliper_test_needs_torch_dlls(caliper_torch_tests)
target_link_libraries(caliper_torch_tests PRIVATE
    doctest::doctest caliper::sdk "${TORCH_LIBRARIES}")
set_target_properties(caliper_torch_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# adapters/exportable_pool.hpp includes <caliper/caliper.hpp> for caliper::Bridge;
# the SDK sugar header pulls in the imgui/implot UI-stack headers, so this target
# needs their include dirs (caliper::ui_stack — the same interface caliper_tests
# links). ImTextureID etc. are header-only typedefs; the libs come along harmlessly.
target_link_libraries(caliper_torch_tests PRIVATE caliper::ui_stack)
if(APPLE)
    set_target_properties(caliper_torch_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
    # adapters/torch.hpp reaches the producer MTLCommandQueue through the C
    # ObjC runtime (objc_msgSend/sel_registerName — M2), so any macOS TU
    # including it must link libobjc.
    target_link_libraries(caliper_torch_tests PRIVATE objc)
endif()

add_test(NAME caliper_torch_tests COMMAND caliper_torch_tests)
set_tests_properties(caliper_torch_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()  # CALIPER_WITH_TORCH (caliper_torch_tests)

# ---------------------------------------------------------------------------
# caliper_embed_tests — the embed C ABI (caliper/embed.h) battery (R4 L2a).
# Links libcaliper (the embeddable core) and drives it as an embedder would:
# offscreen core -> load the hello fixture -> pump -> read_pixels -> shutdown,
# plus the refusal gates. Needs a live GPU for the pixel cases (skips like the
# gfx suite when none is present: counted, exit 77 -> ctest "Skipped"); the
# gate cases are GPU-free.
# ---------------------------------------------------------------------------
# embed_c_check.c compiles the embed headers as C and checks the version macros.
add_executable(caliper_embed_tests test_embed.cpp embed_c_check.c)
# libcaliper ONLY (no CALIPER_DEPENDENCY_LIBS): this binary links exactly as a
# third-party embedder (the Python extension) would, so it proves libcaliper's
# own link interface is self-sufficient.
target_link_libraries(caliper_embed_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk)
# src/host on the include path for the §7 host-axis case: it reaches the process
# bridge + renderer (white-box) to byte-compare the tensor->texture upload the
# embed core wired against the shared CPU reference (host_services.h,
# tensor_bridge.h, renderer/host_renderer.h).
target_include_directories(caliper_embed_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
target_compile_definitions(caliper_embed_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets")
set_target_properties(caliper_embed_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# The offscreen case composites + reads back the hello fixture, so its dylib
# and manifest must exist in build/applets before the test runs. sine_scope is
# the SECOND distinct applet the load-different-id-after-active case swaps to
# (and, being jobs.v1/device.v1-backed, it exercises the "new applet's init jobs
# survive the old one's teardown" reorder).
add_dependencies(caliper_embed_tests hello_applet)
if(CALIPER_WITH_TORCH)
    add_dependencies(caliper_embed_tests sine_scope)
    caliper_test_needs_torch_dlls(caliper_embed_tests)   # sine_scope links libtorch
else()
    # Torch-free build: sine_scope (links libtorch) is not built, so the cases
    # that swap to dev.example.sine-scope compile out (test_embed.cpp guards).
    target_compile_definitions(caliper_embed_tests PRIVATE CALIPER_TESTS_NO_TORCH_APPLETS=1)
endif()
# libtorch rpath only for the torch applets it loads (sine_scope); a torch-free
# binary carries no reference to the vendored libtorch at all.
if(APPLE AND CALIPER_WITH_TORCH)
    set_target_properties(caliper_embed_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_test(NAME caliper_embed_tests COMMAND caliper_embed_tests)
# The stores this binary opens go to a work folder, never the user's app-data.
set_tests_properties(caliper_embed_tests PROPERTIES ENVIRONMENT
    "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_embed_tests")
set_tests_properties(caliper_embed_tests PROPERTIES SKIP_RETURN_CODE 77)

# ---------------------------------------------------------------------------
# caliper_live_tests — the live-streaming attach (src/host/live, include/caliper/
# live.h): client segments + viewer-side attach driven in ONE process over an
# offscreen core. Links exactly like caliper_embed_tests. GPU cases skip
# without a device (counted; exit 77 -> ctest "Skipped", gpu_skip.h).
# ---------------------------------------------------------------------------
if(APPLE)   # the live session code is macOS-only so far
add_executable(caliper_live_tests test_live.cpp)
target_link_libraries(caliper_live_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk)
target_include_directories(caliper_live_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
set_target_properties(caliper_live_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    set_source_files_properties(test_live.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
endif()
# The network part (test_live_net.cpp): a fake wire protocol v3 producer on
# loopback (net/ws.h server, net/wire.h encoder) streams every kind to a core
# attached by URL (caliper_core_live_attach_url, live_net_source.h).
target_sources(caliper_live_tests PRIVATE test_live_net.cpp)
# The 3.1 golden frames (tests/fixtures/wire_v3/v3_1) that the native source
# must tolerate.
target_compile_definitions(caliper_live_tests PRIVATE CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME caliper_live_tests COMMAND caliper_live_tests)
# The stores this binary opens go to a work folder, never the user's app-data.
set_tests_properties(caliper_live_tests PROPERTIES ENVIRONMENT
    "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_live_tests")
set_tests_properties(caliper_live_tests PROPERTIES SKIP_RETURN_CODE 77)

# caliper_live_views_tests — the typed live views (caliper/live_views.h): apply
# heatmap/image/points/series/histogram snapshots to an offscreen core and read
# textures/values back. Same link set as caliper_live_tests. GPU cases skip
# like there (exit 77 -> ctest "Skipped").
add_executable(caliper_live_views_tests test_live_views.cpp)
target_link_libraries(caliper_live_views_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk)
target_include_directories(caliper_live_views_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
set_target_properties(caliper_live_views_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    set_source_files_properties(test_live_views.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
endif()
add_test(NAME caliper_live_views_tests COMMAND caliper_live_views_tests)
# The stores this binary opens go to a work folder, never the user's app-data.
set_tests_properties(caliper_live_views_tests PROPERTIES ENVIRONMENT
    "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_live_views_tests")
set_tests_properties(caliper_live_views_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()  # APPLE (live tests)

# ---------------------------------------------------------------------------
# Applet learning-core / engine tests. Each is a SEPARATE binary that links
# only doctest + libtorch and puts one applet's header directory on the include
# path — keeping torch out of the fast caliper_tests unit suite. The four blocks
# below were near-verbatim, so they share one helper (§8.f cleanup class).
# ---------------------------------------------------------------------------
if(CALIPER_WITH_TORCH)   # applet learning-core tests all link libtorch
function(add_caliper_torch_model_test name src applet_dir)
    add_executable(${name} ${src})
    caliper_test_needs_torch_dlls(${name})
    target_link_libraries(${name} PRIVATE doctest::doctest "${TORCH_LIBRARIES}")
    target_include_directories(${name} PRIVATE "${applet_dir}")
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    if(APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# SculptScope learning-core (energy loss, target samplers, that the net learns,
# and that the addmm_out display fusion == the true forward).
add_caliper_torch_model_test(caliper_sculpt_tests test_sculpt.cpp
    "${CMAKE_SOURCE_DIR}/applets/sculpt_scope")

# MeshScope learning-core (that the tanh MLP learns the target surface, the
# per-vertex error map is honest, finite-difference normals match the analytic
# normals of a known heightfield, and reset(seed) is reproducible).
add_caliper_torch_model_test(caliper_mesh_tests test_mesh_surface.cpp
    "${CMAKE_SOURCE_DIR}/applets/mesh_scope")

# FieldScope PIC physics (FFT Poisson solve, charge-conserving CIC deposition,
# Boris gyration + E×B drift, plasma momentum conservation).
add_caliper_torch_model_test(caliper_em_pic_tests test_em_pic.cpp
    "${CMAKE_SOURCE_DIR}/applets/field_scope")

# TwinScope v2 surface engine (midpoint subdivision, cotan Laplacian, Voronoi
# masses, stable dt, UV bake). The asset-coverage row loads the committed
# housing.obj via the T5 loader, so this target also needs the SDK include dir
# and the source-root macro.
add_caliper_torch_model_test(caliper_surface_tests test_twin_surface.cpp
    "${CMAKE_SOURCE_DIR}/applets/twin_scope")
target_include_directories(caliper_surface_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/sdk/include")
target_compile_definitions(caliper_surface_tests PRIVATE
    CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

# TwinScope v2 thermal model + chasing learner (batched surface heat step,
# seeded duty schedule, MLP learner). Pure torch, small subdivided-tetra sims.
add_caliper_torch_model_test(caliper_twin_tests test_twin_thermal.cpp
    "${CMAKE_SOURCE_DIR}/applets/twin_scope")

set_tests_properties(caliper_torch_tests PROPERTIES LABELS "torch")
endif()  # CALIPER_WITH_TORCH (applet model tests)
