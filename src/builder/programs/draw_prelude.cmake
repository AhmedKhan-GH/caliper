# caliper_draw_prelude_source(<out_var>): writes a C++ source into the
# current binary folder that defines draw_prelude() of
# caliper_builder/programs/graphics.h as the text of draw_prelude.wgsl, and
# sets <out_var> to its path. The builder's programs library and the browser
# module (web/wasm/CMakeLists.txt) both call it, so that the app, the script
# and the browser read one prelude file (design/programmable-graphics.md,
# Prelude). A change of the file runs the configure step again.
set(CALIPER_DRAW_PRELUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(caliper_draw_prelude_source out_var)
    set(_wgsl "${CALIPER_DRAW_PRELUDE_DIR}/draw_prelude.wgsl")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_wgsl}")
    file(READ "${_wgsl}" CALIPER_DRAW_PRELUDE)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/caliper_draw_prelude.cpp")
    configure_file("${CALIPER_DRAW_PRELUDE_DIR}/draw_prelude.cpp.in" "${_out}" @ONLY)
    set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()
