# The match of the library piece frustum@1 with the built-in frustum of a camera entity in the scene
# view (design/programmable-graphics.md, Library Pieces): caliper_frustum_match_tests, made with the
# rig and the function of tests/image_match (CMakeLists.txt there). Without a Metal device the test
# skips and the binary exits 77, which ctest reports as skipped.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_image_match_target frustum)
