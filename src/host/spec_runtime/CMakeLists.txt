# The spec runtime (src/host/spec_runtime, design/spec-format.md): it draws the
# panels of a spec natively in the app's look from the data the builder
# session resolves. Built only with CALIPER_WITH_BUILDER, since it reads specs
# and resolved data through the builder's libraries.
#
#   caliper_spec_runtime     the layout, the prepared data, the drawing, the
#                            offscreen canvas and panel.render
#   caliper-spec-shot        a development tool, not installed: a spec drawn
#                            into an offscreen canvas and written as a PNG
#   caliper_spec_runtime_tests   ctest caliper_spec_runtime (headless) and
#                            caliper_spec_runtime_gpu (offscreen canvas)
if(NOT TARGET caliper_builder_session)
    message(STATUS "caliper_spec_runtime: the builder session is not built; the spec runtime is skipped")
    return()
endif()

# caliper_sternbild_ui, the house style and fonts (src/host/ui) that the
# runtime draws with, is defined in the root CMakeLists.txt beside the caliper
# executable, which links the same library.
add_library(caliper_spec_runtime STATIC
    spec_controls.cpp
    spec_layout.cpp
    spec_prepare.cpp
    spec_runtime.cpp
    spec_offscreen.cpp
    spec_stages.cpp
    spec_entities.cpp
    spec_shortcuts.cpp
    spec_transport.cpp
    spec_annotations.cpp
    label_layout.cpp)
target_include_directories(caliper_spec_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_spec_runtime PUBLIC libcaliper caliper_builder_session caliper_sternbild_ui)
set_target_properties(caliper_spec_runtime PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
# The runtime WGSL compiler translates the viewer's stages (spec_stages.cpp,
# design/gpu-programs.md); without it the stage host refuses each stage with
# program.compiler_unavailable.
if(TARGET caliper::wgsl)
    target_link_libraries(caliper_spec_runtime PRIVATE caliper::wgsl)
endif()
if(NOT MSVC)
    target_compile_options(caliper_spec_runtime PRIVATE -Wall -Wextra)
endif()

# The house fonts beside a binary that draws text, as the caliper executable
# has them (sternbild_fonts.cpp looks in <exe dir>/assets/fonts). Several
# binaries share one folder, and two copies into one folder at once fail in a
# parallel build, so each folder is copied into by one step only: a binary
# beside the caliper executable depends on it, whose own post-build step
# copies the fonts there; a binary in another folder depends on one copy
# target for that folder.
function(caliper_spec_runtime_fonts target)
    get_target_property(_dir ${target} BINARY_DIR)
    if(_dir STREQUAL CMAKE_BINARY_DIR)
        add_dependencies(${target} caliper)
        return()
    endif()
    string(MD5 _hash "${_dir}")
    set(_copy caliper_house_fonts_${_hash})
    if(NOT TARGET ${_copy})
        add_custom_target(${_copy}
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/assets/fonts" "${_dir}/assets/fonts"
            COMMENT "Copying house fonts to ${_dir}")
    endif()
    add_dependencies(${target} ${_copy})
endfunction()

add_executable(caliper-spec-shot spec_shot_main.cpp)
target_link_libraries(caliper-spec-shot PRIVATE caliper_spec_runtime tomlplusplus::tomlplusplus)
set_target_properties(caliper-spec-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
caliper_spec_runtime_fonts(caliper-spec-shot)

if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_spec_runtime_tests
        tests/test_main.cpp
        tests/test_spec_controls.cpp
        tests/test_spec_layout.cpp
        tests/test_spec_prepare.cpp
        tests/test_spec_scene.cpp
        tests/test_spec_generation.cpp
        tests/test_spec_draw.cpp
        tests/test_panel_render.cpp
        tests/test_spec_shot.cpp
        tests/test_spec_stages.cpp
        tests/test_spec_pixels.cpp
        tests/test_spec_entities.cpp
        tests/test_spec_transport.cpp
        tests/test_spec_annotations.cpp
        tests/test_label_layout.cpp)
    target_link_libraries(caliper_spec_runtime_tests PRIVATE
        caliper_spec_runtime doctest::doctest caliper_builder_duckdb caliper_builder_acp
        caliper_catalog_providers)
    target_include_directories(caliper_spec_runtime_tests PRIVATE ${THIRD_PARTY_DIR}/stb)
    target_compile_definitions(caliper_spec_runtime_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/test-tmp"
        CALIPER_SPEC_SHOT_EXE="$<TARGET_FILE:caliper-spec-shot>")
    set_target_properties(caliper_spec_runtime_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_spec_runtime_tests caliper-spec-shot)
    caliper_spec_runtime_fonts(caliper_spec_runtime_tests)
    # Headless: the layout and the prepared data, no GPU.
    add_test(NAME caliper_spec_runtime COMMAND caliper_spec_runtime_tests --test-suite-exclude=gpu)
    # Offscreen: each primitive drawn and read back, the states, panel.render
    # through a fake host, and caliper-spec-shot as a process over a workspace.
    add_test(NAME caliper_spec_runtime_gpu COMMAND caliper_spec_runtime_tests --test-suite=gpu)
    set_tests_properties(caliper_spec_runtime caliper_spec_runtime_gpu PROPERTIES LABELS "builder;spec_runtime")
    set_tests_properties(caliper_spec_runtime_gpu PROPERTIES TIMEOUT 300)
endif()
