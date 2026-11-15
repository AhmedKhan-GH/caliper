# The web viewer spike (docs/plans/web-viewer-spike.md): Caliper's live view
# layer compiled to WebAssembly with Emscripten and drawn by a WebGPU
# HostRenderer. A separate entry point, not part of the main build; configure
# it with emcmake (see build.sh):
#
#   emcmake cmake -S spikes/web-viewer -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build <build>
#
# The output is caliper_web_spike.js + .wasm next to a copy of web/index.html.
cmake_minimum_required(VERSION 3.22)
project(caliper_web_spike C CXX)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "The web viewer spike builds with Emscripten only (emcmake cmake ...)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(CALIPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TP "${CALIPER_ROOT}/third_party")

option(SPIKE_CLOSURE "Run the Closure compiler over the JavaScript glue" ON)

add_executable(caliper_web_spike
    # The page host, the WebGPU backend and the link stand-ins (this spike).
    src/web_main.cpp
    src/webgpu_renderer.cpp
    src/web_stubs.cpp
    # Caliper's view layer, compiled unchanged.
    ${CALIPER_ROOT}/src/host/live/live_views.cpp
    ${CALIPER_ROOT}/src/host/live/live_views_draw.cpp
    ${CALIPER_ROOT}/src/host/live/live_unsupported.cpp
    ${CALIPER_ROOT}/src/host/embed/embed_core.cpp
    ${CALIPER_ROOT}/src/host/tensor_bridge.cpp
    # Dear ImGui, its GLFW and WebGPU backends, and ImPlot.
    ${TP}/imgui/imgui.cpp
    ${TP}/imgui/imgui_draw.cpp
    ${TP}/imgui/imgui_tables.cpp
    ${TP}/imgui/imgui_widgets.cpp
    ${TP}/imgui/backends/imgui_impl_glfw.cpp
    ${TP}/imgui/backends/imgui_impl_wgpu.cpp
    ${TP}/implot/implot.cpp
    ${TP}/implot/implot_items.cpp
)

target_include_directories(caliper_web_spike PRIVATE
    src
    ${CALIPER_ROOT}/src/host
    ${CALIPER_ROOT}/src
    ${CALIPER_ROOT}/include
    ${CALIPER_ROOT}/sdk/include
    ${TP}/imgui
    ${TP}/imgui/backends
    ${TP}/implot)

target_compile_definitions(caliper_web_spike PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_DAWN)

# The WebGPU port must be named when compiling too (it provides webgpu.h);
# GLFW's header is always in the Emscripten sysroot.
target_compile_options(caliper_web_spike PRIVATE --use-port=emdawnwebgpu)
target_link_options(caliper_web_spike PRIVATE
    --use-port=emdawnwebgpu
    -sUSE_GLFW=3
    -sENVIRONMENT=web
    -sALLOW_MEMORY_GROWTH=1
    -sERROR_ON_UNDEFINED_SYMBOLS=1
    "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$UTF8ToString"
)
if(SPIKE_CLOSURE AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(caliper_web_spike PRIVATE --closure=1)
endif()

add_custom_command(TARGET caliper_web_spike POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html
            $<TARGET_FILE_DIR:caliper_web_spike>/index.html)
