add_library(ucdh_workbench SHARED
    plugin.cpp
    ucdh_workbench.cpp
    dataset.cpp
    app_paths.cpp
    model_viz.cpp
)

target_include_directories(ucdh_workbench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ucdh_workbench PRIVATE
    caliper_applet_sdk
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(ucdh_workbench PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(ucdh_workbench PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(ucdh_workbench PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
endif()
