# imgui-node-editor wrapper — lives OUTSIDE the submodule to keep it pristine.
# Parent build sets IMGUI_NODE_EDITOR_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(imgui-node-editor CXX)

if(NOT DEFINED IMGUI_NODE_EDITOR_SRC_DIR)
    message(FATAL_ERROR "IMGUI_NODE_EDITOR_SRC_DIR must be defined (path to imgui-node-editor submodule)")
endif()

add_library(imgui-node-editor STATIC
    ${IMGUI_NODE_EDITOR_SRC_DIR}/imgui_node_editor.cpp
    ${IMGUI_NODE_EDITOR_SRC_DIR}/imgui_node_editor_api.cpp
    ${IMGUI_NODE_EDITOR_SRC_DIR}/imgui_canvas.cpp
    ${IMGUI_NODE_EDITOR_SRC_DIR}/crude_json.cpp
)

target_include_directories(imgui-node-editor PUBLIC
    ${IMGUI_NODE_EDITOR_SRC_DIR}
)

target_link_libraries(imgui-node-editor PUBLIC imgui)

set_target_properties(imgui-node-editor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
