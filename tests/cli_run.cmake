# Runs the caliper executable once for a ctest entry and checks its exit code
# and what it printed. Arguments (-D before -P):
#   EXE     the caliper executable
#   ARGS    its arguments, separated by "|"
#   EXIT    the exit code it must return
#   MATCH   a regular expression its stdout must match (optional)
#   ERROR   a regular expression its stderr must match (optional)
#   ENV     NAME=value pairs for its environment, separated by "|" (optional)
#   CLEAN   a folder removed before the run (optional)
if(CLEAN)
    file(REMOVE_RECURSE "${CLEAN}")
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
