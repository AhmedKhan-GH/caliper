# caliper-sdk CMake package. find_package(caliper-sdk CONFIG) provides:
#
#   caliper::sdk          The ABI headers, the C service tables, the C++ sugar
#                         (caliper/caliper.hpp) and the adapters.
#   caliper::ui_stack     The pinned ImGui, ImPlot, ImPlot3D and ImGuiFileDialog
#                         that the host was built with, compiled from the sources
#                         in this package inside the consumer's build
#                         (caliper-ui-stack.cmake explains why).
#   caliper::sdk_testing      The fixture host (caliper/fixture_host.h) and doctest
#                             (doctest/doctest.h) for an instrument's own tests.
#   caliper_add_instrument()  Builds an instrument library and its stamped manifest
#                             (CaliperAddInstrument.cmake).
#
# and these variables:
#
#   CALIPER_SDK_VERSION    The SDK version.
#   CALIPER_SDK_ABI_EPOCH  The ABI epoch the SDK's headers speak.
#   CALIPER_UI_STACK_HASH  The hash of the pinned UI stack (see
#                          share/caliper-sdk/ui_stack/ui_stack.txt for its inputs).
#   CALIPER_TORCH_VERSION  The host's libtorch version, empty when the SDK was
#                          built without libtorch.
#   CALIPER_TORCH_DIR      Where the host's libtorch is. It defaults to the
#                          libtorch of the build that installed the SDK; pass
#                          -DCALIPER_TORCH_DIR=<libtorch root> to use another
#                          copy of the same version. Only instruments that call
#                          caliper_add_instrument(... TORCH) need it.
#   CALIPER_MACOS_MIN      The host's minimum macOS, 14.0 by default. On macOS
#                          it becomes CMAKE_OSX_DEPLOYMENT_TARGET unless the
#                          project or the environment chose one.
include("${CMAKE_CURRENT_LIST_DIR}/caliper-sdk-info.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CaliperSDKTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/caliper-ui-stack.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CaliperAddInstrument.cmake")
