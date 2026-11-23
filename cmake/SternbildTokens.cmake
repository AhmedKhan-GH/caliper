# ============================================================================
# Sternbild design tokens
# ============================================================================
#
# The colours, type sizes, spacing, metrics and house fonts of the app's theme
# come from the organization's shared design package. The package generates
# sternbild_tokens.h and ships the fonts and icons. This file is the one place
# in the repository that names the release: change STERNBILD_TOKENS_TAG here to
# take a new one. Nothing from the package is copied into the repository.
#
# The first configure of a build folder downloads the package with git; later
# configures use that copy and need no network. A configure without network can
# use a local checkout of the organization repository at the same tag instead:
#   -DFETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS=<checkout>
#
# After include(), the INTERFACE target sternbild::tokens carries the header,
# and its properties hold the font paths (STERNBILD_FONTS_DIR, STERNBILD_FONT_UI,
# STERNBILD_FONT_MONO, STERNBILD_FONT_DISPLAY_OTF).

include_guard(GLOBAL)
include(FetchContent)

set(STERNBILD_TOKENS_REPOSITORY https://github.com/AhmedKhan-GH/organization.git)
set(STERNBILD_TOKENS_TAG design-v2.0.0)

FetchContent_Declare(sternbild_tokens
    GIT_REPOSITORY ${STERNBILD_TOKENS_REPOSITORY}
    GIT_TAG ${STERNBILD_TOKENS_TAG}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR packages/sternbild-tokens-c)

# When the download cannot happen, FetchContent stops in a sub-build with git's
# retries and a call stack that do not name the design tokens. Thus, before a
# download, the repository is asked for the tag, and a failure stops the
# configure with a message that says what is missing and how to go on. The
# check runs only when the build folder has no copy of the tag yet.
function(_sternbild_tokens_check_reachable)
    if(FETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS OR FETCHCONTENT_FULLY_DISCONNECTED)
        return()
    endif()
    find_package(Git QUIET)
    set(_what "The app's theme reads the Sternbild design tokens, release ${STERNBILD_TOKENS_TAG} of ${STERNBILD_TOKENS_REPOSITORY} (pinned in cmake/SternbildTokens.cmake)")
    set(_way_out "or configure with -DFETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS=<path> set to a local checkout of that repository at ${STERNBILD_TOKENS_TAG}")
    if(NOT GIT_FOUND)
        message(FATAL_ERROR "${_what}. The first configure of a build folder downloads them with git, and git was not found. Install git and configure again, ${_way_out}.")
    endif()
    set(_base "${FETCHCONTENT_BASE_DIR}")
    if(NOT _base)
        set(_base "${CMAKE_BINARY_DIR}/_deps")
    endif()
    set(_src "${_base}/sternbild_tokens-src")
    if(EXISTS "${_src}/.git")
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_src}" rev-parse -q --verify HEAD
                        OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc_head ERROR_QUIET)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_src}" rev-parse -q --verify "refs/tags/${STERNBILD_TOKENS_TAG}^{commit}"
                        OUTPUT_VARIABLE _tag OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc_tag ERROR_QUIET)
        if(_rc_head EQUAL 0 AND _rc_tag EQUAL 0 AND _head STREQUAL _tag)
            return()   # the build folder holds the tag: FetchContent needs no network
        endif()
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env GIT_TERMINAL_PROMPT=0
                "${GIT_EXECUTABLE}" ls-remote --exit-code --tags "${STERNBILD_TOKENS_REPOSITORY}" "refs/tags/${STERNBILD_TOKENS_TAG}"
        RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_VARIABLE _err ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 120)
    if(_rc EQUAL 2)
        message(FATAL_ERROR "${_what}. The repository has no tag ${STERNBILD_TOKENS_TAG}. Correct STERNBILD_TOKENS_TAG in cmake/SternbildTokens.cmake.")
    elseif(NOT _rc EQUAL 0)
        message(FATAL_ERROR "${_what}. The first configure of a build folder downloads them, and git could not reach the repository:\n  ${_err}\nConnect to the network and configure again, ${_way_out}.")
    endif()
endfunction()

_sternbild_tokens_check_reachable()
FetchContent_MakeAvailable(sternbild_tokens)
