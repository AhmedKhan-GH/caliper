# Regenerates the git-commit stamp header at BUILD time (export.v1 sidecar —
# PUBLISHING.md §3: provenance must track the bytes being built, not the last
# configure). Writes only when the content changes, so incremental builds do
# not recompile host_services.cpp gratuitously.
execute_process(
    COMMAND git -C "${SRC_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE _hash OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR _hash STREQUAL "")
    set(_hash "unknown")
endif()
# A dirty working tree stamps honestly: hash + "-dirty".
execute_process(
    COMMAND git -C "${SRC_DIR}" status --porcelain --untracked-files=no
    OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _drc)
if(_drc EQUAL 0 AND NOT _dirty STREQUAL "" AND NOT _hash STREQUAL "unknown")
    set(_hash "${_hash}-dirty")
endif()
set(_content "#pragma once\n#define CALIPER_GIT_COMMIT \"${_hash}\"\n")
if(EXISTS "${OUT_FILE}")
    file(READ "${OUT_FILE}" _old)
else()
    set(_old "")
endif()
if(NOT _content STREQUAL _old)
    file(WRITE "${OUT_FILE}" "${_content}")
endif()
