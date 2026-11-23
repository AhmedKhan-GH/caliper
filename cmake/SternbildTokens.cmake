# ============================================================================
# Sternbild design tokens
# ============================================================================
#
# The colours, type sizes, spacing, metrics and house fonts of the app's theme
# come from the organization's shared design package. The package generates
# sternbild_tokens.h and ships the fonts and icons. This file is the one place
# in the repository that names the release: change GIT_TAG here to take a new
# one. Nothing from the package is copied into the repository.
#
# After include(), the INTERFACE target sternbild::tokens carries the header,
# and its properties hold the font paths (STERNBILD_FONTS_DIR, STERNBILD_FONT_UI,
# STERNBILD_FONT_MONO, STERNBILD_FONT_DISPLAY_OTF).

include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(sternbild_tokens
    GIT_REPOSITORY https://github.com/AhmedKhan-GH/organization.git
    GIT_TAG design-v1.1.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR packages/sternbild-tokens-c)
FetchContent_MakeAvailable(sternbild_tokens)
