# The ML exemplar (PLATFORM.md §17 Phase 2). Links the in-tree libtorch the
# same way repnet_demo does — runtime packs replace this at Phase 4.
add_library(ml_scope_applet SHARED ml_scope.cpp)
target_link_libraries(ml_scope_applet PRIVATE
    caliper::sdk caliper::ui_stack "${TORCH_LIBRARIES}")
target_compile_definitions(ml_scope_applet PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(ml_scope_applet PROPERTIES
    OUTPUT_NAME ml_scope
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    set_target_properties(ml_scope_applet PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_custom_command(TARGET ml_scope_applet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ml_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/ml_scope.caliper.toml)
