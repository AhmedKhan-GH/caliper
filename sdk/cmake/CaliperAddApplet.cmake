# caliper_add_applet() builds one applet. The same file serves the core build
# (root CMakeLists.txt includes it) and the installed caliper-sdk package
# (caliper-sdk-config.cmake includes it).
#
#   caliper_add_applet(<target>
#       SOURCES <source>...
#       MANIFEST <stem>.caliper.toml
#       [OUTPUT_NAME <stem>]          # default: <target>
#       [OUTPUT_DIRECTORY <dir>])     # default: ${CMAKE_BINARY_DIR}/applets
#
# It builds the shared library lib<stem>.dylib, <stem>.dll or lib<stem>.so as
# C++20 with CALIPER_APPLET_EXPORT, linked to caliper::sdk and caliper::ui_stack,
# and writes a stamped copy of the manifest, <stem>.caliper.toml, beside it.
# The loader pairs a manifest with the library of the same stem, so the source
# manifest must be named <stem>.caliper.toml.
#
# The manifest is validated at configure time: [applet] id, name and version
# and [compat] abi_epoch are required, the id must be a valid applet id, the
# versions must be strict semver (x.y.z), and abi_epoch must equal the SDK's
# ABI epoch. The stamped copy appends a [build] table that records the SDK
# version and the UI-stack hash the applet was compiled against, so the source
# manifest must not contain a [build] table of its own.
#
# The manifest reader handles the subset of TOML that manifests use: [table]
# headers and single-line key = value pairs with string or integer values.
#
# It needs these variables, which the package (or the core build) sets:
# CALIPER_SDK_VERSION, CALIPER_SDK_ABI_EPOCH and CALIPER_UI_STACK_HASH.

include_guard(GLOBAL)

# Reads the scalar keys of the [applet] and [compat] tables of a manifest into
# <prefix>_<table>_<key> (the value) and <prefix>_<table>_<key>_type (string or
# integer) in the caller's scope. Also sets <prefix>_has_build (TRUE when the
# manifest has a [build] table).
function(_caliper_read_manifest manifest prefix)
    file(READ "${manifest}" text)
    string(REPLACE "\r" "" text "${text}")
    # Protect the characters that CMake treats specially in lists, then split
    # the text into lines.
    string(REPLACE ";" "<caliper-semicolon>" text "${text}")
    string(REPLACE "[" "<caliper-lbracket>" text "${text}")
    string(REPLACE "]" "<caliper-rbracket>" text "${text}")
    string(REPLACE "\n" ";" lines "${text}")

    set(table "")
    set(has_build FALSE)
    foreach(line IN LISTS lines)
        string(REPLACE "<caliper-semicolon>" ";" line "${line}")
        string(REPLACE "<caliper-lbracket>" "[" line "${line}")
        string(REPLACE "<caliper-rbracket>" "]" line "${line}")
        string(STRIP "${line}" line)

        if(line MATCHES "^\\[\\[")
            # An array of tables. Manifests use none; skip its keys.
            set(table "<array>")
        elseif(line MATCHES "^\\[[ \t]*([^]]*[^] \t])[ \t]*\\]")
            set(table "${CMAKE_MATCH_1}")
            if(table STREQUAL "build" OR table MATCHES "^build[ \t]*\\.")
                set(has_build TRUE)
            endif()
        elseif(table STREQUAL "" AND line MATCHES "^build[ \t]*[.=]")
            set(has_build TRUE)
        elseif((table STREQUAL "applet" OR table STREQUAL "compat")
               AND line MATCHES "^([A-Za-z0-9_-]+)[ \t]*=[ \t]*(.*)$")
            set(key "${CMAKE_MATCH_1}")
            set(raw "${CMAKE_MATCH_2}")
            if(raw MATCHES "^\"([^\"]*)\"")
                set(value "${CMAKE_MATCH_1}")
                set(type string)
            elseif(raw MATCHES "^'([^']*)'")
                set(value "${CMAKE_MATCH_1}")
                set(type string)
            elseif(raw MATCHES "^([+-]?[0-9]+)([ \t]|#|$)")
                set(value "${CMAKE_MATCH_1}")
                set(type integer)
            else()
                set(value "${raw}")
                set(type other)
            endif()
            set(${prefix}_${table}_${key} "${value}" PARENT_SCOPE)
            set(${prefix}_${table}_${key}_type "${type}" PARENT_SCOPE)
        endif()
    endforeach()
    set(${prefix}_has_build ${has_build} PARENT_SCOPE)
endfunction()

function(caliper_add_applet target)
    cmake_parse_arguments(PARSE_ARGV 1 arg
        "" "MANIFEST;OUTPUT_NAME;OUTPUT_DIRECTORY" "SOURCES")
    set(where "caliper_add_applet(${target})")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${where}: unknown arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT arg_SOURCES)
        message(FATAL_ERROR "${where}: SOURCES is required.")
    endif()
    if(NOT arg_MANIFEST)
        message(FATAL_ERROR "${where}: MANIFEST is required.")
    endif()
    foreach(var CALIPER_SDK_VERSION CALIPER_SDK_ABI_EPOCH CALIPER_UI_STACK_HASH)
        if(NOT ${var})
            message(FATAL_ERROR "${where}: ${var} is not set. Call find_package(caliper-sdk CONFIG) first.")
        endif()
    endforeach()

    set(stem "${target}")
    if(arg_OUTPUT_NAME)
        set(stem "${arg_OUTPUT_NAME}")
    endif()
    set(out_dir "${CMAKE_BINARY_DIR}/applets")
    if(arg_OUTPUT_DIRECTORY)
        get_filename_component(out_dir "${arg_OUTPUT_DIRECTORY}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    # ---- The manifest ----
    get_filename_component(manifest "${arg_MANIFEST}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${manifest}")
        message(FATAL_ERROR "${where}: manifest not found: ${manifest}")
    endif()
    get_filename_component(manifest_name "${manifest}" NAME)
    if(NOT manifest_name STREQUAL "${stem}.caliper.toml")
        message(FATAL_ERROR
            "${where}: the manifest must be named ${stem}.caliper.toml, after the "
            "library's stem \"${stem}\", because the loader pairs <stem>.caliper.toml "
            "with the library of the same stem. Got ${manifest_name}; rename it or "
            "set OUTPUT_NAME.")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest}")

    _caliper_read_manifest("${manifest}" m)
    set(bad "${manifest}:")
    foreach(key id name version)
        if(NOT DEFINED m_applet_${key})
            message(FATAL_ERROR "${bad} [applet] ${key} is required.")
        endif()
        if(NOT m_applet_${key}_type STREQUAL "string")
            message(FATAL_ERROR "${bad} [applet] ${key} must be a string.")
        endif()
    endforeach()
    string(LENGTH "${m_applet_id}" id_length)
    if(NOT m_applet_id MATCHES "^[a-z0-9-]+(\\.[a-z0-9-]+)*$" OR id_length GREATER 128)
        message(FATAL_ERROR
            "${bad} [applet] id \"${m_applet_id}\" is not a valid applet id: lowercase "
            "reverse-DNS, only [a-z0-9.-], dot-separated segments that are not empty, "
            "at most 128 characters.")
    endif()
    string(STRIP "${m_applet_name}" name_stripped)
    if(name_stripped STREQUAL "")
        message(FATAL_ERROR "${bad} [applet] name must not be empty.")
    endif()
    if(NOT m_applet_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        message(FATAL_ERROR
            "${bad} [applet] version \"${m_applet_version}\" is not strict semver (x.y.z).")
    endif()
    if(NOT DEFINED m_compat_abi_epoch)
        message(FATAL_ERROR "${bad} [compat] abi_epoch is required.")
    endif()
    if(NOT m_compat_abi_epoch_type STREQUAL "integer" OR m_compat_abi_epoch LESS 1)
        message(FATAL_ERROR "${bad} [compat] abi_epoch must be an integer of at least 1.")
    endif()
    if(NOT m_compat_abi_epoch EQUAL CALIPER_SDK_ABI_EPOCH)
        message(FATAL_ERROR
            "${bad} [compat] abi_epoch is ${m_compat_abi_epoch}, but this SDK "
            "(${CALIPER_SDK_VERSION}) builds for ABI epoch ${CALIPER_SDK_ABI_EPOCH}.")
    endif()
    if(DEFINED m_compat_min_host)
        if(NOT m_compat_min_host_type STREQUAL "string"
           OR NOT m_compat_min_host MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
            message(FATAL_ERROR
                "${bad} [compat] min_host \"${m_compat_min_host}\" is not strict semver (x.y.z).")
        endif()
    endif()
    if(m_has_build)
        message(FATAL_ERROR
            "${bad} the source manifest has a [build] table. caliper_add_applet() "
            "writes that table into the built copy; remove it from the source.")
    endif()

    # ---- The library ----
    add_library(${target} SHARED ${arg_SOURCES})
    target_link_libraries(${target} PRIVATE caliper::sdk caliper::ui_stack)
    target_compile_definitions(${target} PRIVATE CALIPER_APPLET_EXPORT)
    # A generator expression in the output directory keeps multi-config
    # generators from appending a per-configuration folder, so the library and
    # its manifest always share one folder.
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${stem}"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${out_dir}>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${out_dir}>"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)

    # ---- The stamped manifest ----
    file(READ "${manifest}" text)
    string(REPLACE "\r" "" text "${text}")
    if(NOT text MATCHES "\n$")
        string(APPEND text "\n")
    endif()
    string(APPEND text
        "\n"
        "# Written by caliper_add_applet() when the applet was built.\n"
        "[build]\n"
        "sdk = \"${CALIPER_SDK_VERSION}\"\n"
        "ui_stack = \"${CALIPER_UI_STACK_HASH}\"\n")
    set(stamped "${CMAKE_CURRENT_BINARY_DIR}/caliper_applet/${stem}.caliper.toml")
    file(WRITE "${stamped}.new" "${text}")
    configure_file("${stamped}.new" "${stamped}" COPYONLY)   # rewrites only on change

    # The copy lands beside the library after each link, and a changed stamp
    # relinks the library, so the two never disagree.
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${stamped}")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${stamped}" "${out_dir}/${stem}.caliper.toml"
        VERBATIM)
endfunction()
