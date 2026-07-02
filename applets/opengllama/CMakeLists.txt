if(NOT TARGET cpp-httplib)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib
                     ${CMAKE_BINARY_DIR}/_deps/cpp-httplib EXCLUDE_FROM_ALL)
endif()

add_library(opengllama SHARED
    plugin.cpp
    opengllama.cpp
    ollama_models.cpp
    ollama_server.cpp
    model_profiles.cpp
)

target_include_directories(opengllama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib
)

target_link_libraries(opengllama PRIVATE
    caliper::sdk
    caliper::ui_stack
    caliper_app_paths
    llama
    cpp-httplib
)

target_compile_definitions(opengllama PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(opengllama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(opengllama PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    )
endif()

add_custom_command(TARGET opengllama POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/opengllama.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/opengllama.caliper.toml)
