# caliper-sdk CMake package. find_package(caliper-sdk CONFIG) provides:
#
#   caliper::sdk          The ABI headers, the C service tables, the C++ sugar
#                         (caliper/caliper.hpp) and the adapters.
#   caliper::ui_stack     The pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog
#                         that the host was built with, compiled from the sources
#                         in this package inside the consumer's build
#                         (caliper-ui-stack.cmake explains why).
#   caliper::sdk_testing  The fixture host (caliper/fixture_host.h) and doctest
#                         (doctest/doctest.h) for an applet's own tests.
#   caliper_add_applet()  Builds an applet library and its stamped manifest
#                         (CaliperAddApplet.cmake).
#
# and these variables:
#
#   CALIPER_SDK_VERSION    The SDK version.
#   CALIPER_SDK_ABI_EPOCH  The ABI epoch the SDK's headers speak.
#   CALIPER_UI_STACK_HASH  The hash of the pinned UI stack (see
#                          share/caliper-sdk/ui_stack/ui_stack.txt for its inputs).
include("${CMAKE_CURRENT_LIST_DIR}/caliper-sdk-info.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CaliperSDKTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/caliper-ui-stack.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CaliperAddApplet.cmake")
