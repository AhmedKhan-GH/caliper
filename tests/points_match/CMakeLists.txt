# Library piece checks use saved original images and pick identities.
# All draws use the piece runtime on offscreen Metal targets.
# A missing Metal device returns 77.
function(caliper_points_match_targets)
    if(NOT APPLE OR NOT TARGET caliper_draw OR NOT TARGET caliper_draw_host OR NOT TARGET caliper_net OR NOT TARGET caliper_scene OR NOT TARGET caliper_producer)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/points_match")
    add_executable(caliper_points_match_tests "${_dir}/test_points_match.mm"
        "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp")
    target_include_directories(caliper_points_match_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_points_match_tests PRIVATE caliper_draw caliper_draw_host caliper_scene caliper_net caliper_producer caliper::sdk
        doctest::doctest "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_dir}/test_points_match.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_points_match_tests PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    # Saved reference images cover the depth cues of the points view.
    target_sources(caliper_points_match_tests PRIVATE "${_dir}/test_points_cues.mm")
    set_source_files_properties("${_dir}/test_points_cues.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    add_test(NAME caliper_points_match COMMAND caliper_points_match_tests)
    # The original renderer supplied these fixed reference images.
    target_compile_definitions(caliper_points_match_tests PRIVATE CALIPER_MATCH_SUITE="points")
    set_property(TEST caliper_points_match APPEND PROPERTY ENVIRONMENT
        "CALIPER_MATCH_REFERENCES=${CMAKE_SOURCE_DIR}/tests/match_references")
    set_tests_properties(caliper_points_match PROPERTIES LABELS "gfx;points_match" SKIP_RETURN_CODE 77 TIMEOUT 120)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_points_match_targets)
