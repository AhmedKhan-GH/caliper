# ============================================================================
# Sternbild design tokens
# ============================================================================
#
# The colours, type sizes, spacing, metrics and house fonts of the app's theme
# come from the organization's shared design package. The package generates
# sternbild_tokens.h and ships the fonts and icons. The FetchContent declaration
# lives in cmake/Dependencies.cmake; STERNBILD_TOKENS_TAG and
# STERNBILD_TOKENS_COMMIT are kept here as well for scripts/shared_colors.py and
# web/build.mjs. Update both files together when bumping the release.
#
# STERNBILD_TOKENS_TAG names the release and is the line that a bump tool
# rewrites, and STERNBILD_TOKENS_COMMIT is the commit that the download checks
# out. A tag can be moved and the package runs CMake code at configure time, so
# the build takes the commit, and caliper_fetch_sternbild_tokens() stops a
# configure when the tag on the repository no longer points to that commit.
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

set(STERNBILD_TOKENS_REPOSITORY https://github.com/AhmedKhan-GH/organization.git)
set(STERNBILD_TOKENS_TAG design-v2.0.0)
set(STERNBILD_TOKENS_COMMIT f88dc88a2223cdeb029625639d23be1ab5755f95)

if(NOT COMMAND caliper_fetch_sternbild_tokens)
    set(_caliper_saved_decl_only "${CALIPER_DEPENDENCIES_DECLARATIONS_ONLY}")
    set(_caliper_had_decl_only FALSE)
    if(DEFINED CALIPER_DEPENDENCIES_DECLARATIONS_ONLY)
        set(_caliper_had_decl_only TRUE)
    endif()
    set(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY ON)
    include("${CMAKE_CURRENT_LIST_DIR}/Dependencies.cmake")
    if(_caliper_had_decl_only)
        set(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY "${_caliper_saved_decl_only}")
    else()
        unset(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY)
    endif()
endif()

caliper_fetch_sternbild_tokens()
