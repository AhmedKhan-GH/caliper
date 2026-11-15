# Runs the caliper executable once for a ctest entry and checks its exit code
# and what it printed. Arguments (-D before -P):
#   EXE     the caliper executable
#   ARGS    its arguments, separated by "|"
#   EXIT    the exit code it must return
#   MATCH   a regular expression its stdout must match (optional)
#   ERROR   a regular expression its stderr must match (optional)
#   ENV     NAME=value pairs for its environment, separated by "|" (optional)
#   CLEAN   a folder removed before the run (optional)
#   DATA_DIR  the CALIPER_DATA_DIR of the run: removed before it, and it must
#           not exist after it unless ALLOW names what it may hold (optional)
#   ALLOW   the names allowed at the top of DATA_DIR, separated by "|"
#   STAGE   <from>|<to>: a folder copied to <to> before the run, after CLEAN
#           (optional)
if(CLEAN)
    file(REMOVE_RECURSE "${CLEAN}")
endif()
if(DATA_DIR)
    file(REMOVE_RECURSE "${DATA_DIR}")
endif()
if(STAGE)
    string(REPLACE "|" ";" stage "${STAGE}")
    list(GET stage 0 stage_from)
    list(GET stage 1 stage_to)
    file(COPY "${stage_from}/" DESTINATION "${stage_to}")
endif()
string(REPLACE "|" ";" args "${ARGS}")
string(REPLACE "|" ";" env "${ENV}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${env} "${EXE}" ${args}
    RESULT_VARIABLE code
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
message(STATUS "caliper ${args}\n--- stdout\n${out}--- stderr\n${err}--- exit ${code}")
if(NOT "${code}" STREQUAL "${EXIT}")
    message(FATAL_ERROR "exit code ${code}, expected ${EXIT}")
endif()
if(DEFINED MATCH AND NOT out MATCHES "${MATCH}")
    message(FATAL_ERROR "stdout does not match: ${MATCH}")
endif()
if(DEFINED ERROR AND NOT err MATCHES "${ERROR}")
    message(FATAL_ERROR "stderr does not match: ${ERROR}")
endif()
if(DATA_DIR AND EXISTS "${DATA_DIR}")
    if(NOT ALLOW)
        message(FATAL_ERROR "the run created ${DATA_DIR}")
    endif()
    string(REPLACE "|" ";" allow "${ALLOW}")
    file(GLOB children RELATIVE "${DATA_DIR}" "${DATA_DIR}/*" "${DATA_DIR}/.*")
    foreach(child IN LISTS children)
        if(NOT child IN_LIST allow)
            message(FATAL_ERROR "the run created ${DATA_DIR}/${child}")
        endif()
    endforeach()
endif()
