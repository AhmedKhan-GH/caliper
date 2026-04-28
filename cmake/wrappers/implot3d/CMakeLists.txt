# ImPlot3D wrapper — lives OUTSIDE the implot3d submodule to keep it pristine.
# Parent build sets IMPLOT3D_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(implot3d CXX)

if(NOT DEFINED IMPLOT3D_SRC_DIR)
    message(FATAL_ERROR "IMPLOT3D_SRC_DIR must be defined (path to implot3d submodule)")
endif()

add_library(implot3d STATIC
    ${IMPLOT3D_SRC_DIR}/implot3d.cpp
    ${IMPLOT3D_SRC_DIR}/implot3d_items.cpp
    ${IMPLOT3D_SRC_DIR}/implot3d_meshes.cpp
    ${IMPLOT3D_SRC_DIR}/implot3d_demo.cpp
)

target_include_directories(implot3d PUBLIC ${IMPLOT3D_SRC_DIR})
target_link_libraries(implot3d PUBLIC imgui)

set_target_properties(implot3d PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
