add_library(sine_scope SHARED sine_scope.cpp)
target_link_libraries(sine_scope PRIVATE
    caliper::sdk caliper::ui_stack
    "${TORCH_LIBRARIES}")                  # vendored third_party/libtorch
target_compile_definitions(sine_scope PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(sine_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)   # let the dylib find libtorch at runtime
    set_target_properties(sine_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_custom_command(TARGET sine_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/sine_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/sine_scope.caliper.toml)
