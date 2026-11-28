# throwing_instrument — a hand-written descriptor that throws a C++
# exception INTO the host from a chosen entry point (CALIPER_THROWING_WHERE),
# or from (or faults in) a background job it submits through jobs.v1, or
# fails in frame() with windows and plots left open on the host's UI contexts
# (hence the UI stack, for caliper.ui.v1).
# The CALIPER_INSTRUMENT sugar (hello_world) catches at the ABI boundary, so it cannot
# exercise the crash guard's catch(...); this fixture can. It lands in
# ${CMAKE_BINARY_DIR}/test-instruments — beside the tests, never in the host's
# instruments dir — so the app never shows a card for it.
add_library(throwing_instrument SHARED throwing_instrument.cpp)
target_link_libraries(throwing_instrument PRIVATE caliper::sdk caliper::ui_stack)
set_target_properties(throwing_instrument PROPERTIES
    OUTPUT_NAME throwing
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test-instruments"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test-instruments"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET throwing_instrument POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/throwing.caliper.toml
        ${CMAKE_BINARY_DIR}/test-instruments/throwing.caliper.toml)
