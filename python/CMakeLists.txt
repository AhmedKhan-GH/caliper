# caliper._native — nanobind extension over the embed C ABI (caliper/embed.h).
#
# Built by scikit-build-core from the repo root with CALIPER_WITH_TORCH=OFF, so
# libcaliper (core + Metal renderer) carries no libtorch and lives next to
# Python's own torch in one process. Only the "python" install component goes
# into the wheel.
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native NB_STATIC src/_native.mm)

target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Escape hatch while the embed v1.2 views are header-only in a tree: OFF builds
# the module without them (HAVE_VIEWS=False) even when embed.h declares them.
option(CALIPER_PY_VIEWS "Bind the embed v1.2 view API when embed.h declares it" ON)
if(NOT CALIPER_PY_VIEWS)
    target_compile_definitions(_native PRIVATE CALIPER_PY_DISABLE_VIEWS=1)
endif()
target_compile_options(_native PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
set_target_properties(_native PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)

target_link_libraries(_native PRIVATE
    libcaliper
    caliper::sdk
    glfw
    "-framework Cocoa"
    "-framework QuartzCore"
    "-framework Metal"
    "-framework IOKit")

# Export ONLY the module init. libcaliper's static deps (DuckDB, ImGui, GLFW, ...)
# are built with default visibility; left exported, dyld would coalesce their
# weak C++ symbols with another extension's copy (e.g. the duckdb wheel) in the
# same process.
target_link_options(_native PRIVATE "LINKER:-exported_symbol,_PyInit__native")

install(TARGETS _native LIBRARY DESTINATION caliper COMPONENT python)

# caliper._live — the live-streaming client (shared-memory snapshot segments the
# viewer process maps; python/src/_live.mm). Same link set as _native.
nanobind_add_module(_live NB_STATIC src/_live.mm)
target_include_directories(_live PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(_live PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
set_target_properties(_live PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
target_link_libraries(_live PRIVATE
    libcaliper
    caliper::sdk
    "-framework Cocoa"
    "-framework QuartzCore"
    "-framework Metal"
    "-framework IOKit")
target_link_options(_live PRIVATE "LINKER:-exported_symbol,_PyInit__live")
install(TARGETS _live LIBRARY DESTINATION caliper COMPONENT python)

# The viewer process ships inside the wheel as caliper/bin/caliper-viewer.
if(TARGET caliper_viewer)
    set_target_properties(caliper_viewer PROPERTIES OUTPUT_NAME caliper-viewer)
    install(TARGETS caliper_viewer RUNTIME DESTINATION caliper/bin COMPONENT python)
endif()
