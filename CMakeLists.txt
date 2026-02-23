cmake_minimum_required(VERSION 3.18)
project(caliper)

set(CMAKE_CXX_STANDARD 17)

# Set prefix paths for all third-party libraries
list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/libtorch"
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/glfw"
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/glm"
)

# Find packages
find_package(Torch REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenGL REQUIRED)

# ImGui
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/imgui")
include_directories("${IMGUI_DIR}/include")
link_directories("${IMGUI_DIR}/lib")

# ImGui backend sources
set(IMGUI_BACKENDS
    "${IMGUI_DIR}/include/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/include/backends/imgui_impl_opengl3.cpp"
)

add_executable(caliper
    main.cpp
    ${IMGUI_BACKENDS}
)

target_link_libraries(caliper
    "${TORCH_LIBRARIES}"
    glfw
    glm::glm
    imgui
    OpenGL::GL
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
)

# Copy dynamic libraries to build directory
set_target_properties(caliper PROPERTIES
    INSTALL_RPATH "@executable_path"
)
