# ImTerm wrapper — lives OUTSIDE the submodule to keep it pristine.
# Upstream's CMakeLists only exposes the install-time include path
# ($<INSTALL_INTERFACE:include/>), which doesn't help consumers using
# add_subdirectory. We declare a minimal INTERFACE target here that
# points at the submodule's include/ at build time.
#
# Parent build sets IMTERM_SRC_DIR to the submodule path.
cmake_minimum_required(VERSION 3.18)
project(imterm CXX)

if(NOT DEFINED IMTERM_SRC_DIR)
    message(FATAL_ERROR "IMTERM_SRC_DIR must be defined (path to ImTerm submodule)")
endif()

add_library(imterm INTERFACE)

target_include_directories(imterm INTERFACE
    ${IMTERM_SRC_DIR}/include
)

# ImTerm uses C++17 features; enforce on consumers.
target_compile_features(imterm INTERFACE cxx_std_17)

# It builds against ImGui — link our existing imgui target.
target_link_libraries(imterm INTERFACE imgui)

# ImTerm uses the deprecated ImGui::IsKeyPressedMap removed in newer ImGui.
# Provide a compatibility macro so the header-only library compiles.
target_compile_definitions(imterm INTERFACE
    "IsKeyPressedMap=IsKeyPressed"
)
