# Build-time copy of PyTorch runtime DLLs on MSVC. The DLL set depends on the
# LibTorch variant (CPU vs CUDA, Debug vs Release), so it is discovered at
# build time rather than with configure-time file(GLOB) in CMakeLists.txt.
if(NOT DEFINED SRC_DIR OR NOT DEFINED DST_DIR)
    message(FATAL_ERROR "copy_torch_dlls.cmake requires -DSRC_DIR=... and -DDST_DIR=...")
endif()
file(GLOB _torch_dlls "${SRC_DIR}/*.dll")
if(_torch_dlls)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_torch_dlls} "${DST_DIR}"
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Failed to copy PyTorch DLLs from ${SRC_DIR} to ${DST_DIR}")
    endif()
endif()
