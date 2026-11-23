# ============================================================================
# Sternbild design tokens
# ============================================================================
#
# The colours, type sizes, spacing, metrics and house fonts of the app's theme
# come from the organization's shared design package. The package generates
# sternbild_tokens.h and ships the fonts and icons. This file is the one place
# in the repository that names the release: change STERNBILD_TOKENS_TAG and
# STERNBILD_TOKENS_COMMIT here to take a new one. Nothing from the package is
# copied into the repository.
#
# STERNBILD_TOKENS_TAG names the release and is the line that a bump tool
# rewrites, and STERNBILD_TOKENS_COMMIT is the commit that the download checks
# out. A tag can be moved and the package runs CMake code at configure time, so
# the build takes the commit, and the check below stops a configure when the tag
# on the repository no longer points to that commit.
#
# The first configure of a build folder downloads the package with git; later
# configures use that copy and need no network. A configure without network can
# use a local checkout of the organization repository at the same commit instead:
#   -DFETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS=<checkout>
#
# After include(), the INTERFACE target sternbild::tokens carries the header,
# and its properties hold the font paths (STERNBILD_FONTS_DIR, STERNBILD_FONT_UI,
# STERNBILD_FONT_MONO, STERNBILD_FONT_DISPLAY_OTF).

include_guard(GLOBAL)
include(FetchContent)

set(STERNBILD_TOKENS_REPOSITORY https://github.com/AhmedKhan-GH/organization.git)
set(STERNBILD_TOKENS_TAG design-v2.0.0)
set(STERNBILD_TOKENS_COMMIT f88dc88a2223cdeb029625639d23be1ab5755f95)

# By the commit, thus without GIT_SHALLOW, which CMake allows only for a branch or a tag.
FetchContent_Declare(sternbild_tokens
    GIT_REPOSITORY ${STERNBILD_TOKENS_REPOSITORY}
    GIT_TAG ${STERNBILD_TOKENS_COMMIT}
    SOURCE_SUBDIR packages/sternbild-tokens-c)

# When the download cannot happen, FetchContent stops in a sub-build with git's
# retries and a call stack that do not name the design tokens. Thus, before a
# download, the repository is asked for the tag, and a failure stops the
# configure with a message that says what is missing and how to go on. The same
# answer gives the tag's commit, and a commit other than the pinned one also
# stops the configure. The check runs only when the build folder does not hold
# the pinned commit yet.
function(_sternbild_tokens_check_reachable)
    if(FETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS OR FETCHCONTENT_FULLY_DISCONNECTED)
        return()
    endif()
    find_package(Git QUIET)
    set(_what "The app's theme reads the Sternbild design tokens, release ${STERNBILD_TOKENS_TAG} (commit ${STERNBILD_TOKENS_COMMIT}) of ${STERNBILD_TOKENS_REPOSITORY}, pinned in cmake/SternbildTokens.cmake")
    set(_way_out "or configure with -DFETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS=<path> set to a local checkout of that repository at commit ${STERNBILD_TOKENS_COMMIT}")
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
        if(_rc_head EQUAL 0 AND _head STREQUAL STERNBILD_TOKENS_COMMIT)
            return()   # the build folder holds the pinned commit: FetchContent needs no network
        endif()
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env GIT_TERMINAL_PROMPT=0
                "${GIT_EXECUTABLE}" ls-remote --exit-code --tags "${STERNBILD_TOKENS_REPOSITORY}"
                "refs/tags/${STERNBILD_TOKENS_TAG}" "refs/tags/${STERNBILD_TOKENS_TAG}^{}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 120)
    if(_rc EQUAL 2)
        message(FATAL_ERROR "${_what}. The repository has no tag ${STERNBILD_TOKENS_TAG}. Correct STERNBILD_TOKENS_TAG in cmake/SternbildTokens.cmake.")
    elseif(NOT _rc EQUAL 0)
        message(FATAL_ERROR "${_what}. The first configure of a build folder downloads them, and git could not reach the repository:\n  ${_err}\nConnect to the network and configure again, ${_way_out}.")
    endif()
    # The tag's commit: the peeled line of an annotated tag, else the tag's own line.
    set(_remote "")
    string(REGEX MATCH "([0-9a-f]+)[ \t]+refs/tags/${STERNBILD_TOKENS_TAG}\\^{}" _peeled "${_out}")
    if(_peeled)
        set(_remote "${CMAKE_MATCH_1}")
    elseif(_out MATCHES "([0-9a-f]+)[ \t]+refs/tags/${STERNBILD_TOKENS_TAG}")
        set(_remote "${CMAKE_MATCH_1}")
    endif()
    if(NOT _remote STREQUAL STERNBILD_TOKENS_COMMIT)
        message(FATAL_ERROR "${_what}. The tag ${STERNBILD_TOKENS_TAG} on the repository points to commit ${_remote}, which is not the pinned commit ${STERNBILD_TOKENS_COMMIT}. A tag that moved can carry other code. Check the release, then set STERNBILD_TOKENS_TAG and STERNBILD_TOKENS_COMMIT together: a bump changes both lines.")
    endif()
endfunction()

_sternbild_tokens_check_reachable()
FetchContent_MakeAvailable(sternbild_tokens)
