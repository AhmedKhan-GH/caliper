# Writes the bytes of INPUT to OUTPUT as a list of C++ hexadecimal literals, for
# an #include inside an array initializer:
#   cmake -DINPUT=<file> -DOUTPUT=<file> -P embed_file.cmake
# The output is replaced only when it changes, so an unchanged input does not
# rebuild the library.
if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "embed_file.cmake needs -DINPUT=<file> and -DOUTPUT=<file>")
endif()
file(READ "${INPUT}" _hex HEX)
if(_hex STREQUAL "")
    message(FATAL_ERROR "embed_file.cmake: ${INPUT} is empty")
endif()
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){24})" "\\1\n" _bytes "${_bytes}")
file(WRITE "${OUTPUT}.tmp" "${_bytes}\n")
configure_file("${OUTPUT}.tmp" "${OUTPUT}" COPYONLY)
file(REMOVE "${OUTPUT}.tmp")
