function(caliper_interp_piece_targets)
    if(NOT TARGET caliper_builder_session)
        return()
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/tests/interp_pieces")
    add_executable(caliper_interp_piece_contracts "${_dir}/test_contracts.cpp")
    target_link_libraries(caliper_interp_piece_contracts PRIVATE caliper_builder_session doctest::doctest)
    target_compile_features(caliper_interp_piece_contracts PRIVATE cxx_std_17)
    add_test(NAME caliper_interp_piece_contracts COMMAND caliper_interp_piece_contracts)
    set_tests_properties(caliper_interp_piece_contracts PROPERTIES LABELS "pieces;interp")
    if(APPLE AND TARGET caliper_draw AND TARGET caliper_producer)
        add_executable(caliper_interp_piece_gpu "${_dir}/test_gpu.mm" "${CMAKE_SOURCE_DIR}/src/host/tensor_bridge.cpp")
        target_include_directories(caliper_interp_piece_gpu PRIVATE
            "${CMAKE_SOURCE_DIR}/src/host" "${CMAKE_SOURCE_DIR}/third_party/stb")
        target_link_libraries(caliper_interp_piece_gpu PRIVATE caliper_draw caliper_scene caliper_net
            caliper_producer caliper::sdk caliper_builder_spec doctest::doctest
            "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
        set_source_files_properties("${_dir}/test_gpu.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        set_target_properties(caliper_interp_piece_gpu PROPERTIES
            CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        target_compile_definitions(caliper_interp_piece_gpu PRIVATE
            CALIPER_MATCH_SUITE="interp_pieces"
            CALIPER_INTERP_PIECES_DEFAULT_OUT="${CMAKE_BINARY_DIR}/tests/interp_pieces/out")
        add_test(NAME caliper_interp_piece_gpu COMMAND caliper_interp_piece_gpu)
        set_tests_properties(caliper_interp_piece_gpu PROPERTIES LABELS "gfx;pieces;interp" SKIP_RETURN_CODE 77 TIMEOUT 180)
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL caliper_interp_piece_targets)
