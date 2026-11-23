# The match of the library piece surface@1 with the built-in surface draw kind
# (design/programmable-graphics.md, Library Pieces). caliper_surface_match_tests checks the piece
# as piece.check does, runs its derive half on the producer runtime against the CPU reference of
# the surface operator, and draws grids of heights twice on Metal, offscreen: once as the app
# draws a surface entity (the spec runtime's mesh of the grid through the scene core) and once by
# the piece in the draw runtime, then compares the color images, the depth where a depth offset
# applies, and the picks with their fields. Without a Metal device the test skips and the binary
# exits 77, which ctest reports as skipped. With the environment variable
# CALIPER_SURFACE_MATCH_CAPTURE_DIR set, the test writes the two images of each case and their
# difference there as PNG files.
#
# The draw runtime (caliper_draw) and the spec runtime are defined after this folder, thus the
# target is made by a call deferred to the end of the top folder.
function(caliper_surface_match_targets)
    if(NOT APPLE OR NOT TARGET caliper_draw OR NOT TARGET caliper_draw_host OR NOT TARGET caliper_net
       OR NOT TARGET caliper_scene OR NOT TARGET caliper_spec_runtime OR NOT TARGET caliper_producer
       OR NOT TARGET caliper_builder_operators)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/surface_match")
    add_executable(caliper_surface_match_tests "${_dir}/test_surface_match.mm")
    target_include_directories(caliper_surface_match_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_surface_match_tests PRIVATE caliper_spec_runtime caliper_producer
        caliper_builder_operators caliper_draw caliper_draw_host caliper_scene caliper_net caliper::sdk
        doctest::doctest "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_dir}/test_surface_match.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_surface_match_tests PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_surface_match COMMAND caliper_surface_match_tests)
    set_tests_properties(caliper_surface_match PROPERTIES LABELS "gfx;surface_match" SKIP_RETURN_CODE 77 TIMEOUT 180)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_surface_match_targets)
