# Dear ImGui wrapper — lives OUTSIDE the imgui submodule to keep it pristine.
# Parent build sets IMGUI_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(imgui CXX)

if(NOT DEFINED IMGUI_SRC_DIR)
    message(FATAL_ERROR "IMGUI_SRC_DIR must be defined (path to imgui submodule)")
endif()

# imgui: the core library, the part of the UI stack that applets also compile
# (caliper::ui_stack). It carries the UI-stack definitions, which the SDK hands
# to applets so both sides compile ImGui the same way.
add_library(imgui STATIC
    ${IMGUI_SRC_DIR}/imgui.cpp
    ${IMGUI_SRC_DIR}/imgui_demo.cpp
    ${IMGUI_SRC_DIR}/imgui_draw.cpp
    ${IMGUI_SRC_DIR}/imgui_tables.cpp
    ${IMGUI_SRC_DIR}/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_SRC_DIR})
if(CALIPER_UI_STACK_DEFINITIONS)
    target_compile_definitions(imgui PUBLIC ${CALIPER_UI_STACK_DEFINITIONS})
endif()
set_target_properties(imgui PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# imgui_backends: the GLFW platform backend. Only the host links it; applets
# never see a window or a graphics API. The renderer backends (imgui_impl_metal,
# imgui_impl_vulkan) compile with the host's Metal and Vulkan backends.
add_library(imgui_backends STATIC
    ${IMGUI_SRC_DIR}/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui_backends PUBLIC ${IMGUI_SRC_DIR}/backends)
target_link_libraries(imgui_backends PUBLIC imgui glfw)
set_target_properties(imgui_backends PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
