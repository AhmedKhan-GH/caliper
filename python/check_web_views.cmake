# The browser's WebAssembly views in a wheel. scikit-build-core packs
# python/caliper/ with the files that are in _web/ when the wheel is built,
# and the three files of Caliper's views compiled to WebAssembly are build
# outputs that git ignores: scripts/build_wasm.py builds them,
# scripts/fetch_wasm.py downloads CI's build, and CI puts its own build there
# before it packages. A wheel without them works, and browsers then draw with
# the TypeScript views, so a local build only warns. When CI is true, as on
# GitHub Actions, or CALIPER_REQUIRE_WEB_VIEWS is 1, a missing file fails the
# build; CALIPER_REQUIRE_WEB_VIEWS=0 turns that back into a warning.
#
# python/CMakeLists.txt includes this for wheel builds (SKBUILD). It also runs
# on its own: cmake [-DCALIPER_WEB_DIR=<dir>] -P python/check_web_views.cmake
if(NOT DEFINED CALIPER_WEB_DIR)
    set(CALIPER_WEB_DIR "${CMAKE_CURRENT_LIST_DIR}/caliper/_web")
endif()
set(_caliper_missing_views "")
foreach(_caliper_view IN ITEMS caliper-views-loader.js caliper-views.js caliper-views.wasm)
    if(NOT EXISTS "${CALIPER_WEB_DIR}/${_caliper_view}")
        list(APPEND _caliper_missing_views "${_caliper_view}")
    endif()
endforeach()
if(_caliper_missing_views)
    list(JOIN _caliper_missing_views ", " _caliper_missing_list)
    set(_caliper_msg
        "The wheel will not ship the browser's WebAssembly views: ${_caliper_missing_list} "
        "not in ${CALIPER_WEB_DIR}, so browsers will draw with the TypeScript views. Put them "
        "there first with `python scripts/build_wasm.py` (Emscripten) or "
        "`python scripts/fetch_wasm.py` (CI's build).")
    string(JOIN "" _caliper_msg ${_caliper_msg})
    set(_caliper_require "$ENV{CALIPER_REQUIRE_WEB_VIEWS}")
    if(_caliper_require STREQUAL "" AND "$ENV{CI}" STREQUAL "true")
        set(_caliper_require 1)
    endif()
    if(_caliper_require STREQUAL "1")
        message(FATAL_ERROR "${_caliper_msg} They are required here (CI=true or CALIPER_REQUIRE_WEB_VIEWS=1).")
    else()
        message(WARNING "${_caliper_msg}")
    endif()
else()
    message(STATUS "The wheel ships the browser's WebAssembly views from ${CALIPER_WEB_DIR}")
endif()
