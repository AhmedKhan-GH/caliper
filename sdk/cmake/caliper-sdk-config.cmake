# caliper-sdk CMake package entry point.
# Scope: ABI headers only. The UI stack is not part of the package; applets in this repo use caliper::ui_stack.
include("${CMAKE_CURRENT_LIST_DIR}/CaliperSDKTargets.cmake")
