# The prebuilt caliper_wgsl libraries that cmake/CaliperWgsl.cmake may download, pinned.
#
# The workflow .github/workflows/wgsl.yml builds one archive per platform with
# build_library.py --archive and publishes the archives under the release tag
# caliper-wgsl-<version>. Each archive holds lib/libcaliper_wgsl.a (lib/caliper_wgsl.lib on
# Windows). After a publish run, copy each archive's SHA-256 from the job summary into the
# matching line below in the same commit as any change of the crate. An empty hash means that
# no archive is published for that platform: a download then fails with that reason, and a
# build from source or CALIPER_WGSL_LIBRARY still works.

set(CALIPER_WGSL_PREBUILT_VERSION "0.1.0")
set(CALIPER_WGSL_NAGA_VERSION "30.0.2")

set(CALIPER_WGSL_SHA256_macos-arm64 "")
set(CALIPER_WGSL_SHA256_windows-x64 "")
set(CALIPER_WGSL_SHA256_linux-x64 "")
