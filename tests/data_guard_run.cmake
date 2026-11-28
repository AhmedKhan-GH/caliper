# The app-data guard (src/app_paths.h, THE GUARD) and the tests' own data
# folders, checked as the suite's launches run, with HOME on a stand-in for
# your home folder, so that your own app-data folder is never touched. Run by
# the ctest entry caliper_test_data_guard. Arguments (-D before -P):
#   EXE          the caliper executable
#   EMBED_TESTS  caliper_embed_tests, a test binary that opens the stores
#   APPLETS      the build's applets folder (dev.caliper.hello-world)
#   PYTHON       an interpreter for the Python package, "" for none
#   PYTHON_DIR   the package's folder (python/ in the source tree)
#   WORK_DIR     a work folder; each run makes run-<random> in it, removed after
#                a run that passes and kept after one that fails
# Each step runs with HOME on <run>/home, TMPDIR in the run's folder, and
# CALIPER_DATA_DIR and CALIPER_TEST_RUN (which ctest sets for every entry)
# removed, then set only where the step says. After every step nothing may
# exist in the stand-in's app-data folder. The steps:
#   1. caliper --hidden without a data folder is refused (exit 1, the guard's
#      line) before it creates anything.
#      A data folder named inside the app-data folder is refused the same way.
#   2. caliper --hidden with CALIPER_DATA_DIR runs and opens its stores there.
#   3. caliper_embed_tests run directly, as from a shell or an IDE, without a
#      data folder is refused: the run that once wrote its metrics into the
#      app-data folder.
#   4. caliper shot in a test run (CALIPER_TEST_RUN=1) without a data folder
#      is refused.
#   5. caliper shot --api as you would run it, without CALIPER_DATA_DIR: it
#      keeps its stores and its API file in a temporary folder.
#   6. caliper.serve() of the Python package in a test run without a data
#      folder raises the guard's error rather than writing its stream file.
# Steps 2 and 5 need a GPU; without one they are reported as skipped, and the
# folder is still checked.
foreach(var EXE EMBED_TESTS APPLETS PYTHON_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "data_guard_run.cmake needs -D${var}=...")
    endif()
endforeach()

string(RANDOM LENGTH 12 ALPHABET "0123456789abcdef" run_id)
set(WORK_DIR "${WORK_DIR}/run-${run_id}")
foreach(dir tmp home data)
    file(MAKE_DIRECTORY "${WORK_DIR}/${dir}")
endforeach()
set(home "${WORK_DIR}/home")
# The app-data folder the stand-in home would give: macOS, then Linux.
set(real_folders "${home}/Library/Application Support/Caliper" "${home}/.local/share/caliper")
set(base_env --unset=CALIPER_DATA_DIR --unset=CALIPER_TEST_RUN --unset=CALIPER_USER_APPLETS_DIR
    --unset=CALIPER_WORKSPACE --unset=CALIPER_APPLET_PATH --unset=CALIPER_HIDDEN --unset=XDG_DATA_HOME
    "HOME=${home}" "TMPDIR=${WORK_DIR}/tmp" "TMP=${WORK_DIR}/tmp" "TEMP=${WORK_DIR}/tmp"
    CALIPER_AGENT_OFFLINE=1)

# Runs one step: ARGS is the command, ENV more environment, EXIT the code it
# must end with ("" for any). Leaves its output in step_out and step_code.
function(step name)
    cmake_parse_arguments(PARSE_ARGV 1 s "" "EXIT" "ENV;ARGS")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${base_env} ${s_ENV} ${s_ARGS}
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120
        WORKING_DIRECTORY "${WORK_DIR}")
    message(STATUS "step ${name}: exit ${code}\n--- stdout\n${out}--- stderr\n${err}---")
    foreach(folder ${real_folders})
        if(EXISTS "${folder}")
            file(GLOB_RECURSE made LIST_DIRECTORIES true RELATIVE "${folder}" "${folder}/*")
            message(FATAL_ERROR "step ${name} created the stand-in's app-data folder ${folder} (${made})")
        endif()
    endforeach()
    if(NOT "${s_EXIT}" STREQUAL "" AND NOT code EQUAL s_EXIT)
        message(FATAL_ERROR "step ${name} ended with ${code}, not ${s_EXIT}")
    endif()
    set(step_out "${out}${err}" PARENT_SCOPE)
    set(step_code "${code}" PARENT_SCOPE)
endfunction()

function(expect_refusal name what)
    if(NOT step_out MATCHES "caliper: ${what} has no data folder of its own and would use your app-data folder")
        message(FATAL_ERROR "step ${name} did not print the guard's refusal for ${what}")
    endif()
endfunction()

# 1. The hidden app without a data folder. CALIPER_EXIT_AFTER only bounds a
#    run that the guard should never let start.
step(hidden_refused EXIT 1 ENV CALIPER_EXIT_AFTER=5 ARGS "${EXE}" --hidden --applets "${APPLETS}")
expect_refusal(hidden_refused "the hidden app \\(--hidden\\)")

# 1b. caliper --hidden with a data folder named inside the app-data folder is
# refused too, before it creates anything.
step(hidden_inside_refused EXIT 1
     ENV "CALIPER_DATA_DIR=${WORK_DIR}/home/Library/Application Support/Caliper/inside" CALIPER_EXIT_AFTER=5
     ARGS "${EXE}" --hidden --applets "${APPLETS}")
if(NOT step_out MATCHES "was given a data folder inside your app-data folder")
    message(FATAL_ERROR "step hidden_inside_refused did not print the guard's refusal")
endif()

# 2. The hidden app with one: it runs for a second and closes normally.
step(hidden_data_dir ENV "CALIPER_DATA_DIR=${WORK_DIR}/data" CALIPER_EXIT_AFTER=1
     ARGS "${EXE}" --hidden --applets "${APPLETS}")
if(step_code EQUAL 0)
    if(NOT EXISTS "${WORK_DIR}/data/metrics.duckdb")
        message(FATAL_ERROR "the hidden app with CALIPER_DATA_DIR opened no metrics store there")
    endif()
elseif(NOT step_out MATCHES "\\[renderer\\] metal" AND NOT "$ENV{CALIPER_REQUIRE_GPU}" STREQUAL "1")
    message(STATUS "step hidden_data_dir: run skipped: no GPU")
else()
    message(FATAL_ERROR "the hidden app with CALIPER_DATA_DIR failed (exit ${step_code})")
endif()

# 3. A test binary run directly. The case opens a core and writes a run with
#    a "loss" metric.
step(embed_tests_refused EXIT 1
     ARGS "${EMBED_TESTS}" "--test-case=embed/get_service: the Compass case*")
expect_refusal(embed_tests_refused "a test binary")

# 4. caliper shot in a test run.
step(shot_test_run_refused EXIT 1 ENV CALIPER_TEST_RUN=1
     ARGS "${EXE}" shot --applets "${APPLETS}" --launch dev.caliper.hello-world --size 320x200 --frames 3
          --out "${WORK_DIR}/refused.png")
expect_refusal(shot_test_run_refused "a test run \\(CALIPER_TEST_RUN is set\\)")

# 5. caliper shot --api as a person runs it.
step(shot_api
     ARGS "${EXE}" shot --applets "${APPLETS}" --launch dev.caliper.hello-world --size 320x200 --frames 3
          --out "${WORK_DIR}/shot.png" --api --api-port 0)
if(step_code EQUAL 0)
    if(NOT step_out MATCHES "agents read .*caliper-shot-[^/]*/api\\.json")
        message(FATAL_ERROR "caliper shot --api did not write its API file in its temporary folder")
    endif()
elseif(NOT step_out MATCHES "shot cannot draw offscreen" OR "$ENV{CALIPER_REQUIRE_GPU}" STREQUAL "1")
    message(FATAL_ERROR "caliper shot --api failed (exit ${step_code})")
else()
    message(STATUS "step shot_api: run skipped: no GPU")
endif()

# 6. The Python package, when the interpreter can import it.
if(PYTHON)
    step(python_probe ENV "PYTHONPATH=${PYTHON_DIR}" ARGS "${PYTHON}" -c "import caliper._net")
    if(step_code EQUAL 0)
        step(python_serve_refused EXIT 1 ENV "PYTHONPATH=${PYTHON_DIR}" CALIPER_TEST_RUN=1
             ARGS "${PYTHON}" -c "import caliper; caliper.serve()")
        expect_refusal(python_serve_refused "a test run \\(CALIPER_TEST_RUN is set\\)")
    else()
        message(STATUS "step python_serve_refused: skipped: ${PYTHON} cannot import the package")
    endif()
else()
    message(STATUS "step python_serve_refused: skipped: no Python interpreter")
endif()

message(STATUS "caliper_test_data_guard: OK")
file(REMOVE_RECURSE "${WORK_DIR}")
