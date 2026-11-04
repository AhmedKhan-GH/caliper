# ImGuizmo wrapper: lives OUTSIDE the ImGuizmo submodule to keep it pristine.
# Parent build sets IMGUIZMO_SRC_DIR to the submodule path. Only ImGuizmo itself
# is built: the inspector's 3D view uses its view cube (ViewManipulate).
cmake_minimum_required(VERSION 3.18)
project(imguizmo CXX)

if(NOT DEFINED IMGUIZMO_SRC_DIR)
    message(FATAL_ERROR "IMGUIZMO_SRC_DIR must be defined (path to ImGuizmo submodule)")
endif()

add_library(imguizmo STATIC ${IMGUIZMO_SRC_DIR}/src/ImGuizmo.cpp)
target_include_directories(imguizmo PUBLIC ${IMGUIZMO_SRC_DIR}/src)
target_link_libraries(imguizmo PUBLIC imgui)

set_target_properties(imguizmo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
