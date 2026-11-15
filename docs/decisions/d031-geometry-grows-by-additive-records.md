# D31. Geometry grows by prefix-identical records

| Date | Status |
|---|---|
| 2026-07-09 | Accepted |

## Decision

Geometry grows by appending fields to draw records, with a `draw_stride` parameter so that older, smaller records still work. Each increase ships only against a demonstrated applet need, is gated on its capability bit, and must be byte-exact against one CPU reference on both backends. Drawing is one atomic call that takes an array of draws, and all checks run before any encoding.

## Reason

If any check fails the whole frame is refused and the pixels stay untouched, which a begin and end style of API could not promise. Growing by records keeps one entry point and lets v1_1, v1_2 and v1_3 coexist. Building only what an applet needs avoids freezing designs that nothing has tested.

## Notes

Index values are read from buffers by the shader, which clamps them, so a bad index cannot cause undefined behaviour. Depth on a view that was not created for it is refused, not ignored.
