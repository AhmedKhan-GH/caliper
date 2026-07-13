# stb (vendored single-file headers)

Vendored — NOT a git submodule (stb ships loose single-file headers, no build).

- Upstream: https://github.com/nothings/stb
- Pinned commit: `31c1ad37456438565541f4919958214b6e762fb4` (master, 2026-07-12)
- `stb_image_write.h` — **v1.16**. Used by `caliper.export.v1` to encode the
  offscreen RGBA8 readback to PNG (`STB_IMAGE_WRITE_IMPLEMENTATION` is defined
  in exactly one TU: `src/host/export_service.cpp`). PNG encode is deterministic
  (fixed filter/zlib), which the export determinism contract relies on.
- `stb_image.h` — **v2.30**. **TEST-ONLY** decode: `tests/test_export.cpp`
  decodes the exported PNG back to pixels to verify the byte-exact round-trip
  and top-down row order. Not linked into libcaliper or the caliper exe.

Both are public domain (MIT/Unlicense dual). To update: bump the pinned commit
above, re-fetch both files at that SHA, and re-run `ctest -L gfx` + the export
battery.
