# The match of the library piece image@1 with the built-in image of the scene view
# (design/programmable-graphics.md, Library Pieces). caliper_image_match_tests runs the piece's
# derive half in the producer runtime over fixed tensors, draws its versions in the draw runtime,
# and compares the picture, the picks and the item map with the app's scene view after the same
# image as protocol 3.2 messages, all on Metal and offscreen. Without a Metal device the test skips
# and the binary exits 77, which ctest reports as skipped. With the environment variable
# CALIPER_IMAGE_MATCH_CAPTURE_DIR set, each case writes its two pictures and their difference there
# as PNG files. tests/depth_match and tests/segmentation_match use image_match_target() with the
# rig of this folder.
#
# The draw runtime (caliper_draw) is defined with the builder after this folder, thus the target
# is made by a call deferred to the end of the top folder.
function(caliper_image_match_target name)
    if(NOT APPLE OR NOT TARGET caliper_draw OR NOT TARGET caliper_draw_host OR NOT TARGET caliper_net
       OR NOT TARGET caliper_scene OR NOT TARGET caliper_producer OR NOT TARGET caliper_builder_session)
        return()
    endif()
    set(_src "${CMAKE_SOURCE_DIR}/tests/${name}_match/test_${name}_match.mm")
    add_executable(caliper_${name}_match_tests "${_src}"
        "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp"
        "${CMAKE_SOURCE_DIR}/src/host/live/live_scene.cpp" "${CMAKE_SOURCE_DIR}/src/host/live/scene_legacy.cpp"
        "${CMAKE_SOURCE_DIR}/src/host/live/scene_piece_form.cpp")
    target_include_directories(caliper_${name}_match_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
    target_link_libraries(caliper_${name}_match_tests PRIVATE caliper_draw caliper_draw_host caliper_scene caliper_net
        caliper_producer caliper_builder_session caliper::sdk doctest::doctest
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    set_source_files_properties("${_src}" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_${name}_match_tests PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_${name}_match COMMAND caliper_${name}_match_tests)
    set_tests_properties(caliper_${name}_match PROPERTIES LABELS "gfx;${name}_match" SKIP_RETURN_CODE 77 TIMEOUT 240)
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_image_match_target image)
