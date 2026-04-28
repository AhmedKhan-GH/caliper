# ImPlot wrapper — lives OUTSIDE the implot submodule to keep it pristine.
# Parent build sets IMPLOT_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(implot CXX)

if(NOT DEFINED IMPLOT_SRC_DIR)
    message(FATAL_ERROR "IMPLOT_SRC_DIR must be defined (path to implot submodule)")
endif()

add_library(implot STATIC
    ${IMPLOT_SRC_DIR}/implot.cpp
    ${IMPLOT_SRC_DIR}/implot_items.cpp
    ${IMPLOT_SRC_DIR}/implot_demo.cpp
)

target_include_directories(implot PUBLIC ${IMPLOT_SRC_DIR})
target_link_libraries(implot PUBLIC imgui)

set_target_properties(implot PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
