# Runs `caliper shot` once for a ctest entry and checks the image and what the
# run left behind. Arguments (-D before -P):
#   EXE       the caliper executable
#   CHECK     caliper_png_check, which checks a PNG's size and that it is not
#             one flat colour
#   INSTRUMENTS the --instruments folder that holds the instrument
#   ID          the instrument id to launch
#   SIZE      <W>x<H>
#   FRAMES    the number of frames
#   WORK_DIR  a work folder; each run makes a folder of its own in it,
#             run-<random>, removed after a run that passes and kept after one
#             that fails, so two runs at once never share one
#   MODE      temp: CALIPER_DATA_DIR unset, so shot uses a temporary data
#             folder, which must be gone after the run, and creates nothing
#             in the app-data folder;
#             data_dir: CALIPER_DATA_DIR on data/ in the run's folder, which must then
#             hold the instrument's data folder and no imgui.ini
# The run's HOME, TMPDIR (TMP and TEMP on Windows) and user instruments folder are
# inside the run's folder, so it touches nothing of the user's own. When the
# renderer cannot draw offscreen (no GPU) the run is reported as skipped,
# unless CALIPER_REQUIRE_GPU=1 makes that a failure.
foreach(var EXE CHECK INSTRUMENTS ID SIZE FRAMES WORK_DIR MODE)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "shot_run.cmake needs -D${var}=...")
    endif()
endforeach()

string(RANDOM LENGTH 12 ALPHABET "0123456789abcdef" run_id)
set(WORK_DIR "${WORK_DIR}/run-${run_id}")
foreach(dir tmp home user-instruments)
    file(MAKE_DIRECTORY "${WORK_DIR}/${dir}")
endforeach()
set(out "${WORK_DIR}/shot.png")
set(env --unset=CALIPER_INSTRUMENT_PATH --unset=CALIPER_CHROMELESS
    "CALIPER_USER_INSTRUMENTS_DIR=${WORK_DIR}/user-instruments"
    "HOME=${WORK_DIR}/home" "TMPDIR=${WORK_DIR}/tmp"
    "TMP=${WORK_DIR}/tmp" "TEMP=${WORK_DIR}/tmp")
if(MODE STREQUAL "temp")
    list(APPEND env --unset=CALIPER_DATA_DIR)
elseif(MODE STREQUAL "data_dir")
    list(APPEND env "CALIPER_DATA_DIR=${WORK_DIR}/data")
else()
    message(FATAL_ERROR "MODE is temp or data_dir, not ${MODE}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${env}
        "${EXE}" shot --instruments "${INSTRUMENTS}" --launch "${ID}" --size "${SIZE}"
        --frames "${FRAMES}" --out "${out}"
    RESULT_VARIABLE code OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 120)
message(STATUS "caliper shot ${ID}\n--- stdout\n${stdout}--- stderr\n${stderr}--- exit ${code}")
if(NOT code EQUAL 0)
    if(stderr MATCHES "shot cannot draw offscreen" AND NOT "$ENV{CALIPER_REQUIRE_GPU}" STREQUAL "1")
        message(STATUS "caliper shot: run skipped: no GPU")
        file(REMOVE_RECURSE "${WORK_DIR}")
        return()
    endif()
    message(FATAL_ERROR "caliper shot failed (exit ${code})")
endif()
string(REPLACE "." "\\." id_re "${ID}")
if(NOT stdout MATCHES "wrote .*shot\\.png: ${id_re} at ${SIZE} after ${FRAMES} frames")
    message(FATAL_ERROR "caliper shot did not report the image it wrote")
endif()

string(REPLACE "x" ";" wh "${SIZE}")
list(GET wh 0 width)
list(GET wh 1 height)
execute_process(COMMAND "${CHECK}" "${out}" ${width} ${height}
    RESULT_VARIABLE code OUTPUT_VARIABLE check_out ERROR_VARIABLE check_err)
message(STATUS "${check_out}${check_err}")
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the shot is not a ${SIZE} image with content")
endif()

# What the run left behind. Only shot's own temporary data folder is looked
# for: a GPU driver may keep caches of its own in the temporary folder.
file(GLOB left RELATIVE "${WORK_DIR}/tmp" "${WORK_DIR}/tmp/caliper-shot-*")
if(left)
    message(FATAL_ERROR "the run left ${left} in its temporary folder")
endif()
if(MODE STREQUAL "temp")
    # macOS and Linux find the app-data folder from HOME; Windows asks the
    # shell for it, which HOME does not move.
    file(GLOB_RECURSE made LIST_DIRECTORIES true RELATIVE "${WORK_DIR}/home" "${WORK_DIR}/home/*")
    if(made)
        message(FATAL_ERROR "the run created ${made} under HOME")
    endif()
else()
    if(EXISTS "${WORK_DIR}/data/imgui.ini")
        message(FATAL_ERROR "the run wrote imgui.ini into CALIPER_DATA_DIR")
    endif()
    if(NOT IS_DIRECTORY "${WORK_DIR}/data/data/${ID}")
        message(FATAL_ERROR "the run made no data folder for ${ID} under CALIPER_DATA_DIR")
    endif()
endif()
message(STATUS "caliper shot: OK (${out})")
file(REMOVE_RECURSE "${WORK_DIR}")
