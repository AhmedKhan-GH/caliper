# GLEW wrapper — lives OUTSIDE the glew submodule to keep it pristine.
# The parent build sets GLEW_SRC_DIR to the submodule path and adds this
# directory via add_subdirectory().
cmake_minimum_required(VERSION 3.18)
project(glew C)

if(NOT DEFINED GLEW_SRC_DIR)
    message(FATAL_ERROR "GLEW_SRC_DIR must be defined (path to glew submodule)")
endif()

add_library(libglew_static STATIC ${GLEW_SRC_DIR}/src/glew.c)

target_include_directories(libglew_static PUBLIC ${GLEW_SRC_DIR}/include)
target_compile_definitions(libglew_static PUBLIC GLEW_STATIC)

if(UNIX AND NOT APPLE)
    target_link_libraries(libglew_static PUBLIC ${CMAKE_DL_LIBS})
endif()

find_package(OpenGL REQUIRED)
target_link_libraries(libglew_static PUBLIC OpenGL::GL)
