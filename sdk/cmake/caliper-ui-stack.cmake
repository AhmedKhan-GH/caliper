# caliper::ui_stack for an applet built against the installed SDK.
#
# The package ships the pinned sources of ImGui, ImPlot, ImPlot3D and
# ImGuiFileDialog (share/caliper-sdk/ui_stack) and compiles them here, inside
# the consumer's build, into one static library with position-independent code
# and the host's UI-stack definitions (CALIPER_UI_STACK_DEFINITIONS). Prebuilt
# static libraries would tie every applet to the compiler, standard library,
# build type and CRT the SDK was built with: on Windows a Release /MD library
# fails to link into a Debug applet, and ImGuiFileDialog uses the STL. Sources
# compiled with the applet's own toolchain avoid that, and the UI-stack hash
# still names exactly what was compiled. The cost is a few seconds of compile
# time per applet build folder.
#
# No GLFW, GLEW or ImGui backend is part of it: an applet draws into the
# host's ImGui context and never talks to a window or a graphics API.
if(NOT TARGET caliper_sdk_ui_stack)
    get_filename_component(_caliper_ui_root
        "${CMAKE_CURRENT_LIST_DIR}/../../../share/caliper-sdk/ui_stack" ABSOLUTE)
    add_library(caliper_sdk_ui_stack STATIC EXCLUDE_FROM_ALL
        "${_caliper_ui_root}/imgui/imgui.cpp"
        "${_caliper_ui_root}/imgui/imgui_demo.cpp"
        "${_caliper_ui_root}/imgui/imgui_draw.cpp"
        "${_caliper_ui_root}/imgui/imgui_tables.cpp"
        "${_caliper_ui_root}/imgui/imgui_widgets.cpp"
        "${_caliper_ui_root}/implot/implot.cpp"
        "${_caliper_ui_root}/implot/implot_items.cpp"
        "${_caliper_ui_root}/implot/implot_demo.cpp"
        "${_caliper_ui_root}/implot3d/implot3d.cpp"
        "${_caliper_ui_root}/implot3d/implot3d_items.cpp"
        "${_caliper_ui_root}/implot3d/implot3d_meshes.cpp"
        "${_caliper_ui_root}/implot3d/implot3d_demo.cpp"
        "${_caliper_ui_root}/ImGuiFileDialog/ImGuiFileDialog.cpp")
    target_include_directories(caliper_sdk_ui_stack PUBLIC
        "${_caliper_ui_root}/imgui"
        "${_caliper_ui_root}/implot"
        "${_caliper_ui_root}/implot3d"
        "${_caliper_ui_root}/ImGuiFileDialog")
    if(CALIPER_UI_STACK_DEFINITIONS)
        target_compile_definitions(caliper_sdk_ui_stack PUBLIC ${CALIPER_UI_STACK_DEFINITIONS})
    endif()
    if(UNIX)
        # ImGuiFileDialog's own CMake adds the same flag.
        target_compile_options(caliper_sdk_ui_stack PRIVATE -Wno-unknown-pragmas)
    endif()
    set_target_properties(caliper_sdk_ui_stack PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
    add_library(caliper::ui_stack ALIAS caliper_sdk_ui_stack)
    unset(_caliper_ui_root)
endif()
