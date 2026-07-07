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
    test_artifact_store.cpp
    test_data_store.cpp
    test_tensor_bridge.cpp
    test_alloc_registry.cpp
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
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_sources(caliper_tests PRIVATE test_loader.cpp)
add_dependencies(caliper_tests hello_applet)
target_compile_definitions(caliper_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets"
    CALIPER_TEST_DATA_ROOT="${CMAKE_BINARY_DIR}/test-data")

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
set_tests_properties(caliper_torch_tests PROPERTIES LABELS "torch")
