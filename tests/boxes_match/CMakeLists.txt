# Library piece checks use saved original images and pick identities.
# All draws use the piece runtime on offscreen Metal targets.
# A missing Metal device returns 77.
function(caliper_boxes_match_targets)
    if(NOT APPLE OR NOT TARGET caliper_draw OR NOT TARGET caliper_draw_host OR NOT TARGET caliper_net
       OR NOT TARGET caliper_scene OR NOT TARGET caliper_producer)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/boxes_match")
    add_executable(caliper_boxes_match_tests "${_dir}/test_boxes_match.mm")
    target_include_directories(caliper_boxes_match_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_boxes_match_tests PRIVATE caliper_draw caliper_draw_host caliper_scene caliper_net
        caliper_tensor_bridge caliper_producer caliper::sdk doctest::doctest "-framework Metal" "-framework MetalPerformanceShaders"
        "-framework Foundation")
    set_source_files_properties("${_dir}/test_boxes_match.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_boxes_match_tests PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_boxes_match COMMAND caliper_boxes_match_tests)
    # The original renderer supplied these fixed reference images.
    target_compile_definitions(caliper_boxes_match_tests PRIVATE CALIPER_MATCH_SUITE="boxes")
    set_property(TEST caliper_boxes_match APPEND PROPERTY ENVIRONMENT
        "CALIPER_MATCH_REFERENCES=${CMAKE_SOURCE_DIR}/tests/match_references")
    set_tests_properties(caliper_boxes_match PROPERTIES LABELS "gfx;boxes_match" SKIP_RETURN_CODE 77 TIMEOUT 180)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_boxes_match_targets)
