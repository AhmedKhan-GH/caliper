# The match of the library piece points@1 with the built-in points primitive
# (design/programmable-graphics.md, Library Pieces). caliper_points_match_tests draws one cloud of
# points with the scene core's built-in draw and with the piece in the draw runtime, both on
# Metal and offscreen, and compares the color images and the picks. Without a Metal device the
# test skips and the binary exits 77, which ctest reports as skipped. With the environment
# variable CALIPER_POINTS_MATCH_CAPTURE_DIR set, the test writes the two images and their
# difference there as PNG files.
#
# The draw runtime (caliper_draw) is defined with the builder after this folder, thus the target
# is made by a call deferred to the end of the top folder.
function(caliper_points_match_targets)
    if(NOT APPLE OR NOT TARGET caliper_draw OR NOT TARGET caliper_draw_host OR NOT TARGET caliper_net OR NOT TARGET caliper_scene)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/points_match")
    add_executable(caliper_points_match_tests "${_dir}/test_points_match.mm"
        "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp")
    target_include_directories(caliper_points_match_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_points_match_tests PRIVATE caliper_draw caliper_draw_host caliper_scene caliper_net caliper::sdk
        doctest::doctest "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_dir}/test_points_match.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_points_match_tests PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_points_match COMMAND caliper_points_match_tests)
    set_tests_properties(caliper_points_match PROPERTIES LABELS "gfx;points_match" SKIP_RETURN_CODE 77 TIMEOUT 120)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_points_match_targets)
