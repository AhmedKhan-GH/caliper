# doctest v2.4.11 declares cmake_minimum_required 2.8, which CMake >= 4.0
# rejects; this floor lets FetchContent configure it without a manual flag.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(caliper_tests
    test_main.cpp
    test_abi.cpp
    abi_c_check.c
    test_manifest.cpp
    test_negotiation.cpp
    test_crash_guard.cpp
    test_watchdog.cpp
    test_sugar.cpp
    test_jobs.cpp
    test_device.cpp
    test_sugar_services.cpp
    test_mnist_idx.cpp
    test_metrics_store.cpp
    test_feed_store.cpp
    test_feed_provider.cpp
    test_artifact_store.cpp
    test_data_store.cpp
    test_tensor_bridge.cpp
    test_alloc_registry.cpp
    test_obj.cpp
    test_instance_field.cpp
    test_pulse_ring.cpp
    test_export_affordance.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp
)
# On Windows the host build provides ZLIB::ZLIB in-tree (Dependencies.cmake).
if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()
target_link_libraries(caliper_tests PRIVATE doctest::doctest caliper::sdk caliper_host_lib
    caliper::sdk_testing caliper::ui_stack ZLIB::ZLIB)
# mnist_idx.h now lives with EmbedScope (the sole exemplar is self-contained).
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/embed_scope)
# InstanceScope's pure pose/tint/mesh reference (instance_field.h) is torch-free,
# so its unit test rides the fast caliper_tests suite.
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/instance_scope)
# PulseScope's pure ring-copy/gap logic (pulse_ring.h) is torch-free + UI-free,
# so its unit test rides the fast caliper_tests suite (feed spec §5).
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/pulse_scope)
# Export E2 affordance logic (export_affordance.h: frame budget, pacing, path
# assembly) is torch-free + UI-free, so its unit test rides caliper_tests too.
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/applets/common)
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_sources(caliper_tests PRIVATE test_loader.cpp)
add_dependencies(caliper_tests hello_applet)
target_compile_definitions(caliper_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets"
    CALIPER_TEST_DATA_ROOT="${CMAKE_BINARY_DIR}/test-data"
    CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

add_test(NAME caliper_tests COMMAND caliper_tests)

# ---------------------------------------------------------------------------
# caliper_gfx_tests — windowed, pixel-exact tensor->texture proof (§16).
# Separate binary because it pulls in the GL renderer (NOT caliper_host_lib):
# it links the renderer + bridge sources directly into the exe/gfx-test scope.
# Registered under the "gfx" label so `ctest -L gfx` selects it; it self-skips
# with a MESSAGE when glfwInit() fails, so a headless CI run stays green.
add_executable(caliper_gfx_tests
    gfx/gfx_main.cpp
    ${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/host/renderer/gl_renderer.cpp
)
target_include_directories(caliper_gfx_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_gfx_tests PRIVATE
    doctest::doctest caliper::sdk caliper::ui_stack ${CALIPER_DEPENDENCY_LIBS})
set_target_properties(caliper_gfx_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# On Apple, the same §16 matrix runs on the Metal backend too (C5). Link the
# Metal HostRenderer backend and compile the harness itself as Objective-C++
# with ARC so it can allocate the raw MTLBuffers the device paths consume and
# blit textures back for verification.
if(APPLE)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_metal_backend)
    target_compile_definitions(caliper_gfx_tests PRIVATE CALIPER_HAVE_METAL=1)
    set_source_files_properties(gfx/gfx_main.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
endif()

# On Windows the same §16 matrix runs on the Vulkan backend (CPU-staged path),
# plus hardware-gated CUDA device-path + alloc_shared byte-exact tests. Link the
# Vulkan HostRenderer backend and compile cuda_driver.cpp in (the backend refs
# it, and the CUDA tests build device tensors through it).
if(WIN32)
    target_sources(caliper_gfx_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host/cuda_driver.cpp)
    target_link_libraries(caliper_gfx_tests PRIVATE caliper_vulkan_backend)
    target_compile_definitions(caliper_gfx_tests PRIVATE CALIPER_HAVE_VULKAN=1)
endif()

add_test(NAME caliper_gfx_tests COMMAND caliper_gfx_tests)
set_tests_properties(caliper_gfx_tests PROPERTIES LABELS "gfx")

# ---------------------------------------------------------------------------
# caliper_export_tests — caliper.export.v1 battery (Rung E1). Links libcaliper
# (host_services vends the export service + the composition reaches its g_bridge/
# g_renderer) and the Metal backend, and drives the vended service as an embedder
# would: bind a renderer -> import a quad -> view_png/sequence -> decode the PNG
# back (stb_image, TEST-ONLY) and byte-compare. The live cases self-skip with no
# GPU; the sidecar-golden / atomic-write / sugar cases are GPU-free. Own "export"
# label so `ctest -L gfx` (49/49) is untouched.
# ---------------------------------------------------------------------------
add_executable(caliper_export_tests test_export.cpp)
target_include_directories(caliper_export_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/host
    ${CMAKE_SOURCE_DIR}/third_party/stb)
target_link_libraries(caliper_export_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk caliper::sdk_testing
    caliper::ui_stack ${CALIPER_DEPENDENCY_LIBS})
set_target_properties(caliper_export_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    target_compile_definitions(caliper_export_tests PRIVATE CALIPER_HAVE_METAL=1)
    # ObjC++ + ARC so the harness can allocate the MTLBuffers the imported-
    # geometry path consumes (mirrors caliper_gfx_tests).
    set_source_files_properties(test_export.cpp PROPERTIES
        COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    set_target_properties(caliper_export_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
if(WIN32)
    target_compile_definitions(caliper_export_tests PRIVATE CALIPER_HAVE_VULKAN=1)
endif()
add_test(NAME caliper_export_tests COMMAND caliper_export_tests)
set_tests_properties(caliper_export_tests PROPERTIES LABELS "export")

# ---------------------------------------------------------------------------
# caliper_torch_tests — the applet-side torch adapter (adapters/torch.hpp).
# A SEPARATE binary because it links libtorch: keeping torch out of
# caliper_tests spares the fast unit suite the torch link cost. Registered
# under the "torch" label so `ctest -L torch` selects it; the MPS cases
# self-skip (MESSAGE + pass) when no MPS device is present, staying green.
# ---------------------------------------------------------------------------
add_executable(caliper_torch_tests test_torch_adapter.cpp)
target_link_libraries(caliper_torch_tests PRIVATE
    doctest::doctest caliper::sdk "${TORCH_LIBRARIES}")
set_target_properties(caliper_torch_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# adapters/exportable_pool.hpp includes <caliper/caliper.hpp> for caliper::Bridge;
# the SDK sugar header pulls in the imgui/implot UI-stack headers, so this target
# needs their include dirs (caliper::ui_stack — the same interface caliper_tests
# links). ImTextureID etc. are header-only typedefs; the libs come along harmlessly.
target_link_libraries(caliper_torch_tests PRIVATE caliper::ui_stack)
if(APPLE)
    set_target_properties(caliper_torch_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
    # adapters/torch.hpp reaches the producer MTLCommandQueue through the C
    # ObjC runtime (objc_msgSend/sel_registerName — M2/D24), so any macOS TU
    # including it must link libobjc.
    target_link_libraries(caliper_torch_tests PRIVATE objc)
endif()

add_test(NAME caliper_torch_tests COMMAND caliper_torch_tests)

# ---------------------------------------------------------------------------
# caliper_embed_tests — the embed C ABI (caliper/embed.h) battery (R4 L2a).
# Links libcaliper (the embeddable core) and drives it as an embedder would:
# offscreen core -> load the hello fixture -> pump -> read_pixels -> shutdown,
# plus the refusal gates. Needs a live GPU for the pixel case (self-skips like
# the gfx suite when none is present); the gate cases are GPU-free.
# ---------------------------------------------------------------------------
add_executable(caliper_embed_tests test_embed.cpp)
target_link_libraries(caliper_embed_tests PRIVATE
    doctest::doctest libcaliper caliper::sdk ${CALIPER_DEPENDENCY_LIBS})
# src/host on the include path for the §7 host-axis case: it reaches the process
# bridge + renderer (white-box) to byte-compare the tensor->texture upload the
# embed core wired against the shared CPU reference (host_services.h,
# tensor_bridge.h, renderer/host_renderer.h).
target_include_directories(caliper_embed_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host)
target_compile_definitions(caliper_embed_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets")
set_target_properties(caliper_embed_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# The offscreen case composites + reads back the hello fixture, so its dylib
# and manifest must exist in build/applets before the test runs. sine_scope is
# the SECOND distinct applet the load-different-id-after-active case swaps to
# (and, being jobs.v1/device.v1-backed, it exercises the "new applet's init jobs
# survive the old one's teardown" reorder).
add_dependencies(caliper_embed_tests hello_applet sine_scope)
if(APPLE)
    set_target_properties(caliper_embed_tests PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_test(NAME caliper_embed_tests COMMAND caliper_embed_tests)

# ---------------------------------------------------------------------------
# Applet learning-core / engine tests. Each is a SEPARATE binary that links
# only doctest + libtorch and puts one applet's header directory on the include
# path — keeping torch out of the fast caliper_tests unit suite. The four blocks
# below were near-verbatim, so they share one helper (§8.f cleanup class).
# ---------------------------------------------------------------------------
function(add_caliper_torch_model_test name src applet_dir)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE doctest::doctest "${TORCH_LIBRARIES}")
    target_include_directories(${name} PRIVATE "${applet_dir}")
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    if(APPLE)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# SculptScope learning-core (energy loss, target samplers, that the net learns,
# and that the addmm_out display fusion == the true forward).
add_caliper_torch_model_test(caliper_sculpt_tests test_sculpt.cpp
    "${CMAKE_SOURCE_DIR}/applets/sculpt_scope")

# MeshScope learning-core (that the tanh MLP learns the target surface, the
# per-vertex error map is honest, finite-difference normals match the analytic
# normals of a known heightfield, and reset(seed) is reproducible).
add_caliper_torch_model_test(caliper_mesh_tests test_mesh_surface.cpp
    "${CMAKE_SOURCE_DIR}/applets/mesh_scope")

# FieldScope PIC physics (FFT Poisson solve, charge-conserving CIC deposition,
# Boris gyration + E×B drift, plasma momentum conservation).
add_caliper_torch_model_test(caliper_em_pic_tests test_em_pic.cpp
    "${CMAKE_SOURCE_DIR}/applets/field_scope")

# TwinScope v2 surface engine (midpoint subdivision, cotan Laplacian, Voronoi
# masses, stable dt, UV bake). The asset-coverage row loads the committed
# housing.obj via the T5 loader, so this target also needs the SDK include dir
# and the source-root macro.
add_caliper_torch_model_test(caliper_surface_tests test_twin_surface.cpp
    "${CMAKE_SOURCE_DIR}/applets/twin_scope")
target_include_directories(caliper_surface_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/sdk/include")
target_compile_definitions(caliper_surface_tests PRIVATE
    CALIPER_TEST_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

# TwinScope v2 thermal model + chasing learner (batched surface heat step,
# seeded duty schedule, MLP learner). Pure torch, small subdivided-tetra sims.
add_caliper_torch_model_test(caliper_twin_tests test_twin_thermal.cpp
    "${CMAKE_SOURCE_DIR}/applets/twin_scope")

set_tests_properties(caliper_torch_tests PROPERTIES LABELS "torch")
