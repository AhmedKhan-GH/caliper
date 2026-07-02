add_library(repnet_demo SHARED
    plugin.cpp
    repnet_demo.cpp
    dataset.cpp
    app_paths.cpp
    model_viz.cpp
    # Training Lab: live PerLeadCNN training + visualization
    train/dsp.cpp
    train/sgkf.cpp
    train/perlead_cnn.cpp
    train/ecg_augment.cpp
    train/train_dataset.cpp
    train/train_engine.cpp
    train/training_lab_tab.cpp
)

target_include_directories(repnet_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/train
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor  # nlohmann/json.hpp
)

target_link_libraries(repnet_demo PRIVATE
    caliper::sdk
    caliper::ui_stack
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(repnet_demo PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(repnet_demo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(repnet_demo PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
endif()
