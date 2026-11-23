# Library Piece References

These PNGs preserve the built-in renderer output before its removal. The match tests compare library pieces with these fixed images at their original tolerances. Each PNG stores four channels. A depth reference stores the four bytes of each `Depth32Float` value in those channels.

The image baseline comes from commit `dd90f140`. The original renderer at `55bb4f7a` supplied the adjacent `metadata.json` files. Each metadata key names one PNG relative to this folder. A pick record holds its pixel coordinates, primitive value, and item identity. Point and surface records also hold the original camera matrices. Tests check piece IDs at those saved pixels, then check the item fields.

## Fixture Order

Each suite assigns image numbers in fixture order. Run the complete test binary through CTest. A filter that omits earlier fixtures changes those numbers and cannot select the correct references. The opt-in capacity tests check derive counts only and require no extra images.

## Baseline Reproduction

Use a separate checkout of `55bb4f7a` to reproduce the original images. Build the seven match targets, then set `CALIPER_MATCH_REFERENCE_WRITE` to an empty output folder. Run the complete match binaries with their default case order. Compare their PNG bytes with this folder before any replacement.

To reproduce metadata, record the decoded result of each original pick with the pixel coordinates that the fixture supplied. Associate these records with the last reference image of that case. For points and surfaces, also record the camera matrices that the fixture gives to the piece. The point radius row also records its radius scale and projection scales. Do not use a library piece to supply a reference identity or image.

A deliberate fixture or graphics change requires separate review of its images, metadata, and tolerances. The current tests read references and never replace them.
