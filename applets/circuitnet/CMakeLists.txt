add_library(circuitnet SHARED
    plugin.cpp
    circuitnet.cpp
    verilog_parser.cpp
    circuit_db.cpp
    circuit_viz.cpp
    ${CMAKE_SOURCE_DIR}/src/app_paths.cpp
)

target_include_directories(circuitnet PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(circuitnet PRIVATE
    caliper_applet_sdk
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
