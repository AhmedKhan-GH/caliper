# Dear ImGui wrapper — lives OUTSIDE the imgui submodule to keep it pristine.
# Parent build sets IMGUI_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(imgui CXX)

if(NOT DEFINED IMGUI_SRC_DIR)
    message(FATAL_ERROR "IMGUI_SRC_DIR must be defined (path to imgui submodule)")
endif()

add_library(imgui STATIC
    ${IMGUI_SRC_DIR}/imgui.cpp
    ${IMGUI_SRC_DIR}/imgui_demo.cpp
    ${IMGUI_SRC_DIR}/imgui_draw.cpp
    ${IMGUI_SRC_DIR}/imgui_tables.cpp
    ${IMGUI_SRC_DIR}/imgui_widgets.cpp
    ${IMGUI_SRC_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_SRC_DIR}/backends/imgui_impl_opengl3.cpp
)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

target_include_directories(imgui PUBLIC
    ${IMGUI_SRC_DIR}
    ${IMGUI_SRC_DIR}/backends
)

find_package(OpenGL REQUIRED)
target_link_libraries(imgui PUBLIC libglew_static glfw OpenGL::GL)

set_target_properties(imgui PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
