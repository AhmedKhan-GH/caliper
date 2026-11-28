# The pixel comparison between the app and the browser module
# (design/programmable-graphics.md, Pixel Comparison).
#
# caliper_draw_parity_tests checks what needs no GPU. caliper_draw_parity_app writes the fixture
# (two instances of pieces as PROGRAM frames and one version of each resource as RESOURCE frames)
# into <build>/tests/draw_parity/out and draws it
# with the app's draw runtime on Metal, offscreen. web/test/draw_parity.spec.ts draws the same
# frames with the browser module in headless Chromium and compares the color and ID targets; it
# reads the folder that CALIPER_DRAW_PARITY_DIR names. Without a Metal device the tool writes the
# fixture only and exits 77, which ctest reports as skipped.
#
# The draw runtime (caliper_draw) is defined with the builder after this folder, thus the target
# is made by a call deferred to the end of the top folder.
function(caliper_draw_parity_targets)
    if(NOT TARGET caliper_draw OR NOT TARGET caliper_net)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/draw_parity")
    # What needs no GPU: the fixture's frames and plan, the decode of PROGRAM with the pin's hash,
    # the params, the module's refusals, and an install of each program on the recording backend.
    # The host's colormaps come from the tensor bridge's tables (colormap_lut), compiled in as the
    # graphics suite compiles them.
    add_executable(caliper_draw_parity_tests "${_dir}/test_draw_parity.cpp")
    target_include_directories(caliper_draw_parity_tests PRIVATE "${CMAKE_SOURCE_DIR}/src/host")
    target_link_libraries(caliper_draw_parity_tests PRIVATE caliper_draw caliper_net caliper_tensor_bridge caliper::sdk doctest::doctest
        $<TARGET_NAME_IF_EXISTS:caliper_builder_spec>)
    set_target_properties(caliper_draw_parity_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_draw_parity_tests COMMAND caliper_draw_parity_tests)
    set_tests_properties(caliper_draw_parity_tests PROPERTIES LABELS "gfx;draw_parity")
    if(NOT APPLE)
        return()
    endif()
    add_executable(caliper_draw_parity_app "${_dir}/draw_parity_app.mm")
    target_include_directories(caliper_draw_parity_app PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_draw_parity_app PRIVATE caliper_draw caliper_net caliper_tensor_bridge caliper::sdk $<TARGET_NAME_IF_EXISTS:caliper_builder_spec>
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_dir}/draw_parity_app.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_draw_parity_app PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_draw_parity_app
        COMMAND caliper_draw_parity_app "${CMAKE_BINARY_DIR}/tests/draw_parity/out")
    set_tests_properties(caliper_draw_parity_app PROPERTIES LABELS "gfx;draw_parity" SKIP_RETURN_CODE 77 TIMEOUT 120)
    # The scene comparison (scene_parity.h): the scene fixtures drawn by the app's scene view with the
    # library pieces into <out>/scene, which draw_parity.spec.ts compares with the browser module's.
    # The scene view (src/host/live/live_scene.cpp, with the piece form reader scene_piece_form.cpp) is
    # compiled in with the draw layer, as the graphics suite compiles it.
    if(TARGET caliper_draw_host AND TARGET caliper_scene)
        add_executable(caliper_scene_parity_app "${_dir}/scene_parity_app.mm"
            "${CMAKE_SOURCE_DIR}/src/host/live/live_scene.cpp" "${CMAKE_SOURCE_DIR}/src/host/live/scene_legacy.cpp" "${CMAKE_SOURCE_DIR}/src/host/live/scene_piece_form.cpp")
        target_include_directories(caliper_scene_parity_app PRIVATE
            "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
        target_link_libraries(caliper_scene_parity_app PRIVATE caliper_draw_host caliper_scene caliper_gfx caliper_net
            caliper_tensor_bridge caliper::sdk "-framework Metal" "-framework Foundation")
        set_source_files_properties("${_dir}/scene_parity_app.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        set_target_properties(caliper_scene_parity_app PROPERTIES
            CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        add_test(NAME caliper_scene_parity_app
            COMMAND caliper_scene_parity_app "${CMAKE_BINARY_DIR}/tests/draw_parity/out")
        set_tests_properties(caliper_scene_parity_app PROPERTIES LABELS "gfx;draw_parity" SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
    # The whole dashboard: the recorded run of the Attention Arc Tower in tests/draw_parity/dashboard,
    # drawn by the app's spec runtime offscreen on Metal into <build>/tests/draw_parity/dashboard_out,
    # which web/test/draw_parity_dashboard.spec.ts compares with the browser's spec page
    # (CALIPER_DASHBOARD_PARITY_DIR). Without a Metal device it exits 77, a skip.
    if(TARGET caliper_spec_runtime)
        add_executable(caliper_dashboard_parity_app "${_dir}/dashboard_parity_app.cpp")
        target_include_directories(caliper_dashboard_parity_app PRIVATE "${CMAKE_SOURCE_DIR}/src/host")
        target_link_libraries(caliper_dashboard_parity_app PRIVATE caliper_spec_runtime)
        set_target_properties(caliper_dashboard_parity_app PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
        caliper_spec_runtime_fonts(caliper_dashboard_parity_app)
        add_test(NAME caliper_draw_parity_dashboard_app
            COMMAND caliper_dashboard_parity_app "${_dir}/dashboard" "${CMAKE_SOURCE_DIR}/tests/acceptance/arc_tower"
                    "${CMAKE_BINARY_DIR}/tests/draw_parity/dashboard_out")
        set_tests_properties(caliper_draw_parity_dashboard_app PROPERTIES
            LABELS "gfx;draw_parity" SKIP_RETURN_CODE 77 TIMEOUT 180)
    endif()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_draw_parity_targets)
