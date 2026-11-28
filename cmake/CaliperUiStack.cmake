# The UI stack and its hash.
#
# The host hands its ImGui, ImPlot and ImPlot3D contexts to every instrument, and
# the instrument draws into them with its own compiled copy of the same libraries.
# That only works when both copies come from the same sources with the same
# imconfig and the same definitions, so the SDK ships the host's pinned sources
# (installed below) and names the set with the UI-stack hash. The loader
# compares the hash the host was built with (CALIPER_HOST_UI_STACK_HASH) with
# the one caliper_add_instrument() stamps into each instrument's manifest.
#
# The hash is the first 16 hex digits of the SHA-256 of these lines, each ending
# in a newline:
#
#   imgui_version=<IMGUI_VERSION_NUM from imgui.h>
#   imgui=<commit>
#   implot=<commit>
#   implot3d=<commit>
#   ImGuiFileDialog=<commit>
#   imconfig=<SHA-256 of imgui/imconfig.h>
#   definitions=<CALIPER_UI_STACK_DEFINITIONS, sorted, joined with ",">
#
# A commit is the checked-out HEAD of the library's own git checkout. Without
# one, it is the commit the superproject pins, and without that,
# "sha256:<SHA-256 of the library's main source file>". The recipe is written
# to <build>/caliper-ui-stack.txt and installed as
# share/caliper-sdk/ui_stack/ui_stack.txt.
#
# Inputs: THIRD_PARTY_DIR and CALIPER_UI_STACK_DEFINITIONS.
# Output: CALIPER_UI_STACK_HASH.

find_package(Git QUIET)

function(_caliper_ui_stack_pin name dir main_file out)
    set(pin "")
    if(GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse --show-toplevel
            OUTPUT_VARIABLE top RESULT_VARIABLE rc
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        get_filename_component(dir_real "${dir}" REALPATH)
        if(rc EQUAL 0)
            get_filename_component(top_real "${top}" REALPATH)
        endif()
        if(rc EQUAL 0 AND top_real STREQUAL dir_real)
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse HEAD
                OUTPUT_VARIABLE pin RESULT_VARIABLE rc
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(NOT rc EQUAL 0)
                set(pin "")
            endif()
        endif()
        if(pin STREQUAL "")
            file(RELATIVE_PATH rel "${CMAKE_SOURCE_DIR}" "${dir}")
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" ls-tree HEAD "${rel}"
                OUTPUT_VARIABLE tree RESULT_VARIABLE rc
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(rc EQUAL 0 AND tree MATCHES "^160000 commit ([0-9a-f]+)")
                set(pin "${CMAKE_MATCH_1}")
            endif()
        endif()
    endif()
    if(pin STREQUAL "")
        file(SHA256 "${dir}/${main_file}" sha)
        set(pin "sha256:${sha}")
        message(STATUS "UI stack: no git commit for ${name}; using the content of ${main_file}")
    endif()
    set(${out} "${pin}" PARENT_SCOPE)
endfunction()

set(_ui_imgui    "${THIRD_PARTY_DIR}/imgui")
set(_ui_implot   "${THIRD_PARTY_DIR}/implot")
set(_ui_implot3d "${THIRD_PARTY_DIR}/implot3d")
set(_ui_igfd     "${THIRD_PARTY_DIR}/ImGuiFileDialog")

file(STRINGS "${_ui_imgui}/imgui.h" _ui_version_line REGEX "^#define IMGUI_VERSION_NUM[ \t]+[0-9]+")
string(REGEX REPLACE "^#define IMGUI_VERSION_NUM[ \t]+([0-9]+).*" "\\1" _ui_version "${_ui_version_line}")
_caliper_ui_stack_pin(imgui "${_ui_imgui}" imgui.cpp _ui_pin_imgui)
_caliper_ui_stack_pin(implot "${_ui_implot}" implot.cpp _ui_pin_implot)
_caliper_ui_stack_pin(implot3d "${_ui_implot3d}" implot3d.cpp _ui_pin_implot3d)
_caliper_ui_stack_pin(ImGuiFileDialog "${_ui_igfd}" ImGuiFileDialog.cpp _ui_pin_igfd)
file(SHA256 "${_ui_imgui}/imconfig.h" _ui_imconfig)
set(_ui_definitions ${CALIPER_UI_STACK_DEFINITIONS})
list(SORT _ui_definitions)
list(JOIN _ui_definitions "," _ui_definitions)

set(CALIPER_UI_STACK_RECIPE "")
string(APPEND CALIPER_UI_STACK_RECIPE
    "imgui_version=${_ui_version}\n"
    "imgui=${_ui_pin_imgui}\n"
    "implot=${_ui_pin_implot}\n"
    "implot3d=${_ui_pin_implot3d}\n"
    "ImGuiFileDialog=${_ui_pin_igfd}\n"
    "imconfig=${_ui_imconfig}\n"
    "definitions=${_ui_definitions}\n")
string(SHA256 _ui_sha "${CALIPER_UI_STACK_RECIPE}")
string(SUBSTRING "${_ui_sha}" 0 16 CALIPER_UI_STACK_HASH)
message(STATUS "UI-stack hash: ${CALIPER_UI_STACK_HASH} (ImGui ${_ui_version})")

file(WRITE "${CMAKE_BINARY_DIR}/caliper-ui-stack.txt.new"
    "# The inputs of the UI-stack hash ${CALIPER_UI_STACK_HASH}: the first 16 hex\n"
    "# digits of the SHA-256 of the lines below.\n"
    "${CALIPER_UI_STACK_RECIPE}")
configure_file("${CMAKE_BINARY_DIR}/caliper-ui-stack.txt.new"
    "${CMAKE_BINARY_DIR}/caliper-ui-stack.txt" COPYONLY)

# A changed header or imconfig reconfigures, which recomputes the hash. Of these
# files only IMGUI_VERSION_NUM in imgui.h and the content of imconfig.h are
# inputs; an uncommitted edit to the other sources leaves the hash unchanged.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_ui_imgui}/imgui.h" "${_ui_imgui}/imconfig.h" "${_ui_implot}/implot.h"
    "${_ui_implot3d}/implot3d.h" "${_ui_igfd}/ImGuiFileDialog.h"
    "${_ui_igfd}/ImGuiFileDialogConfig.h")

# ---- The pinned sources in the SDK package (component sdk) ----
# caliper-ui-stack.cmake in the package compiles exactly these files.
set(_ui_dest share/caliper-sdk/ui_stack)
install(FILES
    "${_ui_imgui}/imgui.h" "${_ui_imgui}/imgui_internal.h" "${_ui_imgui}/imconfig.h"
    "${_ui_imgui}/imstb_rectpack.h" "${_ui_imgui}/imstb_textedit.h" "${_ui_imgui}/imstb_truetype.h"
    "${_ui_imgui}/imgui.cpp" "${_ui_imgui}/imgui_demo.cpp" "${_ui_imgui}/imgui_draw.cpp"
    "${_ui_imgui}/imgui_tables.cpp" "${_ui_imgui}/imgui_widgets.cpp" "${_ui_imgui}/LICENSE.txt"
    DESTINATION ${_ui_dest}/imgui COMPONENT sdk)
install(FILES
    "${_ui_implot}/implot.h" "${_ui_implot}/implot_internal.h" "${_ui_implot}/implot.cpp"
    "${_ui_implot}/implot_items.cpp" "${_ui_implot}/implot_demo.cpp" "${_ui_implot}/LICENSE"
    DESTINATION ${_ui_dest}/implot COMPONENT sdk)
install(FILES
    "${_ui_implot3d}/implot3d.h" "${_ui_implot3d}/implot3d_internal.h"
    "${_ui_implot3d}/implot3d.cpp" "${_ui_implot3d}/implot3d_items.cpp"
    "${_ui_implot3d}/implot3d_meshes.cpp" "${_ui_implot3d}/implot3d_demo.cpp"
    "${_ui_implot3d}/LICENSE"
    DESTINATION ${_ui_dest}/implot3d COMPONENT sdk)
install(FILES
    "${_ui_igfd}/ImGuiFileDialog.h" "${_ui_igfd}/ImGuiFileDialog.cpp"
    "${_ui_igfd}/ImGuiFileDialogConfig.h" "${_ui_igfd}/LICENSE"
    DESTINATION ${_ui_dest}/ImGuiFileDialog COMPONENT sdk)
# dirent.h stands in for the POSIX header on Windows.
install(DIRECTORY "${_ui_igfd}/dirent" DESTINATION ${_ui_dest}/ImGuiFileDialog COMPONENT sdk)
install(FILES "${CMAKE_BINARY_DIR}/caliper-ui-stack.txt"
    DESTINATION ${_ui_dest} RENAME ui_stack.txt COMPONENT sdk)
