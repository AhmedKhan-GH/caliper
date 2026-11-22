# caliper_builder_text: the format language of bound texts
# (design/instrument-interface.md, Bound Text; include/caliper_builder/text/bound_text.h).
# It needs only the standard library and nlohmann_json, so that the browser's
# spec runtime can build it to WebAssembly.
add_library(caliper_builder_text STATIC bound_text.cpp)
target_include_directories(caliper_builder_text PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_text PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_text PUBLIC cxx_std_17)
set_target_properties(caliper_builder_text PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(caliper_builder_text PRIVATE -Wall -Wextra)
endif()

caliper_builder_add_tests(text
    SOURCES
        tests/test_bound_text.cpp
    LIBS caliper_builder_text)
