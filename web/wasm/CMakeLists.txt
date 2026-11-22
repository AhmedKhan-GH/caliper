# web/wasm: Caliper's view layer compiled to WebAssembly and drawn by the
# WebGPU HostRenderer (src/host/renderer/webgpu_renderer.*). It is its own
# Emscripten project, not part of the native build, built with the Emscripten
# release in emscripten-version.txt; web/wasm/README.md has the build command:
#
#   python scripts/build_wasm.py
#
# The output is caliper-views.wasm and its JavaScript glue caliper-views.js,
# an ES module whose default export is createCaliperViews(moduleArgs).
cmake_minimum_required(VERSION 3.22)
project(caliper_views_web C CXX)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "web/wasm builds with Emscripten only (emcmake cmake ...)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(CALIPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TP "${CALIPER_ROOT}/third_party")

option(CALIPER_WEB_CLOSURE "Run the Closure compiler over the JavaScript glue (Release)" ON)

# ---- the view layer, without the native embed core ---------------------------
# The live views, the embed v1.2 views and the hosted core (embed_views.cpp,
# split out of embed_core.cpp so the applet loader, the service registry, the
# job system and the app paths stay out), the tensor bridge, the WebGPU
# renderer with the graphics layer's core and WebGPU backend, Dear ImGui and
# ImPlot. live_unsupported.cpp is the existing
# stand-in for the shared-memory attach, which a browser does not have.
add_library(caliper_view_layer STATIC
    ${CALIPER_ROOT}/src/host/live/live_views.cpp
    ${CALIPER_ROOT}/src/host/live/live_views_draw.cpp
    ${CALIPER_ROOT}/src/host/live/live_unsupported.cpp
    ${CALIPER_ROOT}/src/host/embed/embed_views.cpp
    ${CALIPER_ROOT}/src/host/tensor_bridge.cpp
    ${CALIPER_ROOT}/src/host/renderer/webgpu_renderer.cpp
    ${CALIPER_ROOT}/src/host/renderer/webgpu_backend.cpp
    ${CALIPER_ROOT}/src/host/gfx/core.cpp
    ${CALIPER_ROOT}/src/host/gfx/ops.cpp
    ${CALIPER_ROOT}/src/host/ui/sternbild_fonts.cpp
    ${CALIPER_ROOT}/src/host/ui/sternbild_theme.cpp
    ${TP}/imgui/imgui.cpp
    ${TP}/imgui/imgui_draw.cpp
    ${TP}/imgui/imgui_tables.cpp
    ${TP}/imgui/imgui_widgets.cpp
    ${TP}/imgui/backends/imgui_impl_wgpu.cpp
    ${TP}/implot/implot.cpp
    ${TP}/implot/implot_items.cpp
)
target_include_directories(caliper_view_layer PUBLIC
    ${CALIPER_ROOT}/src/host
    ${CALIPER_ROOT}/src
    ${CALIPER_ROOT}/include
    ${CALIPER_ROOT}/sdk/include
    ${TP}/imgui
    ${TP}/imgui/backends
    ${TP}/implot)
# The scene view and the points view of the live views (src/host/live/live_scene.h, live_points.h)
# and what they draw through: the scene core over the graphics layer's WebGPU backend, which the
# renderer owns.
#
# stages.cpp holds the scene view's stage cache (scene/stage_cache.h) and the stage entry points as
# text. It needs no WGSL translator: the app's stage host (src/host/spec_runtime) translates the
# stages, and the browser module does not have that host. Thus no browser view gets a stage source,
# the cache stays empty, and an entity with a stage draws with its built-in program in the browser.
target_sources(caliper_view_layer PRIVATE
    ${CALIPER_ROOT}/src/host/live/live_views_scene.cpp
    ${CALIPER_ROOT}/src/host/live/scene_overlays.cpp
    ${CALIPER_ROOT}/src/host/spec_runtime/label_layout.cpp
    ${CALIPER_ROOT}/src/host/live/live_scene.cpp
    ${CALIPER_ROOT}/src/host/live/live_points.cpp
    ${CALIPER_ROOT}/src/host/scene/stages.cpp
    ${CALIPER_ROOT}/src/host/scene/path.cpp
    ${CALIPER_ROOT}/src/host/scene/scene_tree.cpp
    ${CALIPER_ROOT}/src/host/scene/camera.cpp
    ${CALIPER_ROOT}/src/host/scene/compile.cpp
    ${CALIPER_ROOT}/src/host/scene/pick.cpp
    ${CALIPER_ROOT}/src/host/scene/scale_bar.cpp
    ${CALIPER_ROOT}/src/host/scene/backend_resources.cpp)
# The draw half of pieces (design/programmable-graphics.md): the same render
# graph, composer, prelude and lib modules as the app, from one source each.
include("${CALIPER_ROOT}/src/builder/programs/programs_text.cmake")
caliper_programs_text_source(_caliper_programs_text_cpp)
target_sources(caliper_view_layer PRIVATE
    ${CALIPER_ROOT}/src/host/draw/draw_runtime.cpp
    ${CALIPER_ROOT}/src/builder/programs/pieces.cpp
    ${_caliper_programs_text_cpp})
target_include_directories(caliper_view_layer PUBLIC ${CALIPER_ROOT}/src/builder/include)
target_compile_definitions(caliper_view_layer PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
# The scene core's arithmetic without contraction, as natively (the root CMakeLists.txt's caliper_scene),
# so that both compute the same uniform bytes; WebAssembly has no fused multiply-add, so this only
# states it.
set_source_files_properties(${CALIPER_ROOT}/src/host/scene/scene_tree.cpp ${CALIPER_ROOT}/src/host/scene/camera.cpp
    ${CALIPER_ROOT}/src/host/scene/compile.cpp ${CALIPER_ROOT}/src/host/scene/pick.cpp
    ${CALIPER_ROOT}/src/host/scene/backend_resources.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The WebGPU port must be named when compiling too: it provides webgpu.h.
target_compile_options(caliper_view_layer PUBLIC --use-port=emdawnwebgpu)

# ---- the wire protocol v3 codec (shared with the native NetSource) ---------------
add_library(caliper_wire STATIC
    ${CALIPER_ROOT}/src/host/net/wire.cpp
    ${CALIPER_ROOT}/src/host/net/json.cpp)
target_include_directories(caliper_wire PUBLIC ${CALIPER_ROOT}/src/host ${CALIPER_ROOT}/include)

# ---- the module ----------------------------------------------------------------
add_executable(caliper-views
    src/main.cpp
    src/net_source.cpp
    src/source_poll.cpp
    src/web_services.cpp
    src/web_input.cpp
    src/page_style.cpp
    src/fonts.cpp
    src/gfx_selftest.cpp
    src/scene_selftest.cpp
)
# fonts.cpp embeds the native app's faces from assets/fonts (C23 #embed).
set_source_files_properties(src/fonts.cpp PROPERTIES
    COMPILE_OPTIONS "--embed-dir=${CALIPER_ROOT}/assets/fonts;-Wno-c23-extensions"
    OBJECT_DEPENDS "${CALIPER_ROOT}/assets/fonts/IBMPlexSansCondensed-Regular.ttf;${CALIPER_ROOT}/assets/fonts/SourceCodePro-Regular.ttf")
target_link_libraries(caliper-views PRIVATE caliper_view_layer caliper_wire websocket.js)
target_link_options(caliper-views PRIVATE
    --use-port=emdawnwebgpu
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createCaliperViews
    -sENVIRONMENT=web
    -sFILESYSTEM=0
    # A capped heap: 16 MB to start (one view uses about 2 MB, a page of four
    # about 12 MB), growing to at most 512 MB. One snapshot costs about four
    # times its payload while it is applied (the socket's copy, the source's
    # two buffers, the hover copy), so the cap holds a 64 MB frame, the
    # protocol's largest.
    -sINITIAL_MEMORY=16MB
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=512MB
    -sERROR_ON_UNDEFINED_SYMBOLS=1
    "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$UTF8ToString,$stringToNewUTF8,$stringToUTF8OnStack,$stackSave,$stackRestore,$specialHTMLTargets"
    "-sEXPORTED_FUNCTIONS=_main,_malloc,_free,_caliper_views_set_data_view,_caliper_views_set_toggle,_caliper_views_set_link,_caliper_views_gfx_selftest,_caliper_views_scene_selftest,_caliper_views_colormap_keys,_caliper_views_destroy"
    "-sEXPORTED_RUNTIME_METHODS=ccall,UTF8ToString,stringToNewUTF8"
)
if(CALIPER_WEB_CLOSURE AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(caliper-views PRIVATE --closure=1)
endif()

# The loader and the demo page sit beside the module in the build directory,
# which is the set of files a server hands out (README.md, "Serving").
add_custom_command(TARGET caliper-views POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/caliper-views-loader.js
            ${CMAKE_CURRENT_SOURCE_DIR}/demo.html
            $<TARGET_FILE_DIR:caliper-views>/)
