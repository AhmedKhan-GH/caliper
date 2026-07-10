add_library(twin_scope SHARED
    plugin.cpp
    twin_scope.cpp
)

target_include_directories(twin_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(twin_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
)
# TWIN_SCOPE_ASSET_PATH is the source-tree DEV fallback (§8.c): the applet reads
# the DLL-relative staged copy first (module_dir()/assets/twin_scope/housing.obj,
# staged by the POST_BUILD step below) and only falls back to this macro when the
# staged copy is missing. The staging destination MUST match what the code reads.
target_compile_definitions(twin_scope PRIVATE
    CALIPER_APPLET_EXPORT
    TWIN_SCOPE_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/housing.obj")

set_target_properties(twin_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(twin_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

# Manifest + asset staging. The asset destination is DLL-relative
# (<applets>/assets/twin_scope/housing.obj) so module_dir() resolution in the
# applet finds it next to the loaded library (§8.c).
add_custom_command(TARGET twin_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/twin_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/twin_scope.caliper.toml
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/applets/assets/twin_scope
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/assets/housing.obj
        ${CMAKE_BINARY_DIR}/applets/assets/twin_scope/housing.obj)
