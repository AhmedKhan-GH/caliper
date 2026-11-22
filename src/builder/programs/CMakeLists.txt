# caliper_builder_programs: the sandbox of GPU programs and stages
# (design/gpu-programs.md, Sandbox; include/caliper_builder/programs/sandbox.h).
# The editor's program commands, the producer runtime in the script and the
# viewer's stages run the same checks. It needs only the standard library and
# the header-only value types of spec/program.h (with nlohmann_json), and it
# links the runtime WGSL compiler caliper::wgsl when the build has it.
# graphics.cpp holds the composer and the checks of the graphics form
# (design/programmable-graphics.md), which the browser module also compiles,
# and draw_prelude.cmake embeds draw_prelude.wgsl for it.
include("${CMAKE_CURRENT_SOURCE_DIR}/draw_prelude.cmake")
caliper_draw_prelude_source(_caliper_draw_prelude_cpp)
add_library(caliper_builder_programs STATIC sandbox.cpp graphics.cpp "${_caliper_draw_prelude_cpp}")
target_include_directories(caliper_builder_programs PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_programs PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_programs PUBLIC cxx_std_17)
set_target_properties(caliper_builder_programs PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET caliper::wgsl)
    target_link_libraries(caliper_builder_programs PRIVATE caliper::wgsl)
    target_compile_definitions(caliper_builder_programs PRIVATE CALIPER_BUILDER_PROGRAMS_HAVE_WGSL=1)
endif()
if(NOT MSVC)
    target_compile_options(caliper_builder_programs PRIVATE -Wall -Wextra)
endif()

caliper_builder_add_tests(programs
    SOURCES
        tests/test_sandbox.cpp
        tests/test_graphics.cpp
    LIBS caliper_builder_programs)
