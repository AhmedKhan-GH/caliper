# caliper_producer: the producer runtime's core (caliper_producer.h,
# design/gpu-programs.md and design/producer-runtime.md), which the native
# module caliper._producer links: the graphics layer without a window, the
# runtime WGSL compiler and the sandbox of caliper_builder_programs.
#
# On Apple it is the Metal runtime (producer.mm) on the graphics layer. Other
# platforms build producer.cpp, which answers CALIPER_PRODUCER_UNSUPPORTED
# until a Vulkan runtime exists. The WGSL compiler (caliper::wgsl) and the
# sandbox (caliper_builder_programs, built with the builder) are linked when
# the build has them; without the compiler a program cannot run, and without
# the sandbox the runtime's own binder and the layer's read_only flag still
# hold the read-only rule.
if(APPLE)
    add_library(caliper_producer STATIC producer.mm)
    set_source_files_properties(producer.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(caliper_producer PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
    target_link_libraries(caliper_producer PUBLIC caliper_gfx "-framework Metal" "-framework Foundation")
else()
    add_library(caliper_producer STATIC producer.cpp)
endif()
# The derive half of a piece (caliper_producer_derive.h): producer.mm on Apple, and
# derive.cpp, which answers CALIPER_PRODUCER_UNSUPPORTED, elsewhere.
if(NOT APPLE)
    target_sources(caliper_producer PRIVATE derive.cpp)
endif()
target_include_directories(caliper_producer PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
set_target_properties(caliper_producer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
if(TARGET caliper::wgsl)
    target_link_libraries(caliper_producer PRIVATE caliper::wgsl)
    target_compile_definitions(caliper_producer PRIVATE CALIPER_PRODUCER_HAVE_WGSL=1)
endif()
if(TARGET caliper_builder_programs)
    target_link_libraries(caliper_producer PRIVATE caliper_builder_programs)
    target_compile_definitions(caliper_producer PRIVATE CALIPER_PRODUCER_HAVE_SANDBOX=1)
endif()
if(NOT MSVC)
    target_compile_options(caliper_producer PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_producer_tests tests/test_producer.cpp tests/test_producer_operators.cpp
                                          tests/test_producer_derive.cpp)
    target_link_libraries(caliper_producer_tests PRIVATE caliper_producer doctest::doctest)
    set_target_properties(caliper_producer_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    if(APPLE)
        # The runtime's cases on the Metal device: imports, programs, passes, outputs and refusals.
        target_sources(caliper_producer_tests PRIVATE tests/test_producer_metal.mm tests/test_producer_derive_metal.mm)
        set_source_files_properties(tests/test_producer_metal.mm tests/test_producer_derive_metal.mm
                                    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        set_target_properties(caliper_producer_tests PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        if(TARGET caliper::wgsl)
            target_compile_definitions(caliper_producer_tests PRIVATE CALIPER_PRODUCER_HAVE_WGSL=1)
        endif()
        # The built-in lib modules, for the case of a derive half whose common carries them.
        if(TARGET caliper_builder_programs)
            target_link_libraries(caliper_producer_tests PRIVATE caliper_builder_programs)
            target_compile_definitions(caliper_producer_tests PRIVATE CALIPER_PRODUCER_HAVE_LIB_MODULES=1)
        endif()
    endif()
    add_test(NAME caliper_producer_tests COMMAND caliper_producer_tests)
endif()
