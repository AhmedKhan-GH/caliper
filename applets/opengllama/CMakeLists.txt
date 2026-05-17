add_library(opengllama SHARED
    plugin.cpp
    opengllama.cpp
    ollama_models.cpp
    ${CMAKE_SOURCE_DIR}/src/app_paths.cpp
)

target_include_directories(opengllama PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
)

target_link_libraries(opengllama PRIVATE
    caliper_applet_sdk
    llama
    libglew_static
    imterm
)

target_compile_definitions(opengllama PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(opengllama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(opengllama PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    )
endif()
