# ImGuiColorTextEdit wrapper — lives OUTSIDE the submodule to keep it pristine.
# Upstream ships TextEditor.{h,cpp} at the repo root with no CMakeLists.
# Parent build sets IMGUI_COLOR_TEXT_EDIT_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(imgui-color-text-edit CXX)

if(NOT DEFINED IMGUI_COLOR_TEXT_EDIT_SRC_DIR)
    message(FATAL_ERROR "IMGUI_COLOR_TEXT_EDIT_SRC_DIR must be defined (path to ImGuiColorTextEdit submodule)")
endif()

add_library(imgui-color-text-edit STATIC
    ${IMGUI_COLOR_TEXT_EDIT_SRC_DIR}/TextEditor.cpp
)

target_include_directories(imgui-color-text-edit PUBLIC
    ${IMGUI_COLOR_TEXT_EDIT_SRC_DIR}
)

target_link_libraries(imgui-color-text-edit PUBLIC imgui)

set_target_properties(imgui-color-text-edit PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
