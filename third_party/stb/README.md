# stb (vendored single-file headers)

Vendored — NOT a git submodule (stb ships loose single-file headers, no build).

- Upstream: https://github.com/nothings/stb
- Pinned commit: `31c1ad37456438565541f4919958214b6e762fb4` (master, 2026-07-12)
- `stb_image_write.h` — **v1.16**. Used by `caliper.export.v1` to encode the
  offscreen RGBA8 readback to PNG (`STB_IMAGE_WRITE_IMPLEMENTATION` is defined
  in exactly one TU: `src/host/export_service.cpp`). PNG encode is deterministic
  (fixed filter/zlib), which the export determinism contract relies on.
- `stb_image.h` — **v2.30**. Decodes PNG and JPEG files in the builder's image
  reader (`src/builder/loaders/image.cpp`, the one TU that defines
  `STB_IMAGE_IMPLEMENTATION` there), with bounded dimensions, and decodes the
  exported PNG back to pixels in `tests/test_export.cpp` to verify the
  byte-exact round-trip and top-down row order.

Both are public domain (MIT/Unlicense dual). To update: bump the pinned commit
above, re-fetch both files at that SHA, and re-run `ctest -L gfx` + the export
battery.
