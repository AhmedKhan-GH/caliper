add_library(circuitnet SHARED
    plugin.cpp
    circuitnet.cpp
    verilog_parser.cpp
    circuit_db.cpp
    circuit_viz.cpp
)

target_include_directories(circuitnet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(circuitnet PRIVATE
    caliper::sdk
    caliper::ui_stack
    caliper_app_paths
    imgui-node-editor
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
)

target_compile_definitions(circuitnet PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(circuitnet PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_custom_command(TARGET circuitnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/circuitnet.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/circuitnet.caliper.toml)
