# caliper-sdk CMake package entry point.
# Phase 0 scope: ABI headers only. The UI stack joins the package at Phase 3
# (PLATFORM.md §17); until then applets in this repo use caliper::ui_stack.
include("${CMAKE_CURRENT_LIST_DIR}/CaliperSDKTargets.cmake")
