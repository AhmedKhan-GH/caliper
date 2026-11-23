# The acceptance test of programmable graphics: the hidden app with no script, and caliper-mcp as a child.
# tests/programmable_graphics/driver.py builds the Attention Arc Tower from the pieces of demos/arc_tower through
# the MCP tools alone, on a managed run of examples/python/arc_tower_train.py: the arcs and the grid draw in two
# scene panels with no built-in primitive, a pick returns the layer, the head and the token pair, the ablation of
# the head raises the in-context loss, a fork from a past step with the head ablated never forms it, a draw over
# its time is stopped and named while training goes on, and a piece of a foreign spec waits in review. It compares
# the SHA-256 of the app, caliper-mcp and the Python package before and after, and it writes its captures into the
# build folder. It exits 77 where the app, Metal, torch, the native network sender or the producer module is
# missing, and prints the reason. It uses the Python of the instrument acceptance test.
if(NOT WIN32 AND CALIPER_WITH_BUILDER AND _acceptance_python)
    add_test(NAME caliper_programmable_graphics_acceptance
        COMMAND "${_acceptance_python}" "${CMAKE_CURRENT_SOURCE_DIR}/driver.py"
        --app "$<TARGET_FILE:caliper>" --mcp "$<TARGET_FILE:caliper-mcp>" --python "${_acceptance_python}"
        --work "${CMAKE_BINARY_DIR}/tests/programmable_graphics-work"
        --capture-dir "${CMAKE_BINARY_DIR}/tests/programmable_graphics-captures")
    set_tests_properties(caliper_programmable_graphics_acceptance PROPERTIES
        LABELS "programmable_graphics;acceptance" TIMEOUT 900 SKIP_RETURN_CODE 77)
endif()

# The browser's dashboard viewer end to end (tests/programmable_graphics/viewer_driver.py): the real spec page of the
# hidden app, opened by its viewer link in headless Chromium with WebGPU and with SwiftShader, draws a workspace piece
# and an inline piece of a live run of examples/python/arc_tower_train.py, counted in the page in the same task as the
# module's frame, and says so when the app does not have the dashboard open. It exits 77 where the app, Metal, torch,
# Node with the web package or the WebAssembly views are missing, and prints the reason.
if(NOT WIN32 AND CALIPER_WITH_BUILDER AND _acceptance_python)
    add_test(NAME caliper_dashboard_viewer_acceptance
        COMMAND "${_acceptance_python}" "${CMAKE_CURRENT_SOURCE_DIR}/viewer_driver.py"
        --app "$<TARGET_FILE:caliper>" --mcp "$<TARGET_FILE:caliper-mcp>" --python "${_acceptance_python}"
        --work "${CMAKE_BINARY_DIR}/tests/dashboard_viewer-work")
    set_tests_properties(caliper_dashboard_viewer_acceptance PROPERTIES
        LABELS "programmable_graphics;acceptance" TIMEOUT 900 SKIP_RETURN_CODE 77)
endif()
