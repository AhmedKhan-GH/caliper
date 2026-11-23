# caliper_builder_spec: the spec format, versions 0 and 1
# (design/spec-format.md). It parses a spec from TOML or its canonical JSON
# form, validates it, writes it back and applies merge patches to one panel,
# source or control.
add_library(caliper_builder_spec STATIC
    annotation.cpp
    colormaps.cpp
    common.cpp
    entity.cpp
    instrument.cpp
    json_form.cpp
    patch.cpp
    reference.cpp
    spec.cpp
    state.cpp
    toml_form.cpp
    validate.cpp
    validate_additions.cpp
    validate_annotations.cpp
    validate_bound_text.cpp
    validate_controls.cpp
    validate_entities.cpp
    validate_history.cpp
    validate_layout.cpp
    validate_operators.cpp
    validate_references.cpp
    validate_states.cpp)
target_include_directories(caliper_builder_spec PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
# The validator holds programs to the sandbox's declaration checks
# (caliper_builder_programs), so that the editor and the hosts share one set
# of limits.
target_link_libraries(caliper_builder_spec
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE tomlplusplus::tomlplusplus caliper_builder_programs caliper_builder_text)
target_compile_features(caliper_builder_spec PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(caliper_builder_spec PRIVATE -Wall -Wextra)
endif()

caliper_builder_add_tests(spec
    SOURCES
        tests/test_codec.cpp
        tests/test_codec_instrument.cpp
        tests/test_entities_spec.cpp
        tests/test_operators_spec.cpp
        tests/test_references_spec.cpp
        tests/test_history_spec.cpp
        tests/test_annotations_spec.cpp
        tests/test_patch.cpp
        tests/test_program_spec.cpp
        tests/test_schema.cpp
        tests/test_validate.cpp
    LIBS caliper_builder_spec)
if(TARGET caliper_builder_spec_tests)
    # The tests read their fixtures from tests/fixtures and the schema from a
    # copy in the build folder.
    configure_file(schema/caliper-spec-v0.schema.json
        "${CMAKE_CURRENT_BINARY_DIR}/caliper-spec-v0.schema.json" COPYONLY)
    target_compile_definitions(caliper_builder_spec_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
        CALIPER_SPEC_SCHEMA="${CMAKE_CURRENT_BINARY_DIR}/caliper-spec-v0.schema.json")
endif()
