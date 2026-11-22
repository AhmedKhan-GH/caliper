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
    add_executable(caliper_draw_parity_tests "${_dir}/test_draw_parity.cpp" "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp")
    target_include_directories(caliper_draw_parity_tests PRIVATE "${CMAKE_SOURCE_DIR}/src/host")
    target_link_libraries(caliper_draw_parity_tests PRIVATE caliper_draw caliper_net caliper::sdk doctest::doctest)
    set_target_properties(caliper_draw_parity_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_draw_parity_tests COMMAND caliper_draw_parity_tests)
    set_tests_properties(caliper_draw_parity_tests PROPERTIES LABELS "gfx;draw_parity")
    if(NOT APPLE)
        return()
    endif()
    add_executable(caliper_draw_parity_app "${_dir}/draw_parity_app.mm" "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp")
    target_include_directories(caliper_draw_parity_app PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_draw_parity_app PRIVATE caliper_draw caliper_net caliper::sdk
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_dir}/draw_parity_app.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_draw_parity_app PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_draw_parity_app
        COMMAND caliper_draw_parity_app "${CMAKE_BINARY_DIR}/tests/draw_parity/out")
    set_tests_properties(caliper_draw_parity_app PROPERTIES LABELS "gfx;draw_parity" SKIP_RETURN_CODE 77 TIMEOUT 120)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_draw_parity_targets)
