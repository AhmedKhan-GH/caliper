# D4. The UI-stack pin defines the ABI epoch

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

Applets write raw ImGui and ImPlot. The exact versions of ImGui, ImPlot and the other UI libraries are pinned, and the pin defines the ABI epoch. Changing the pin starts a new epoch, and applets rebuild against it.

## Reason

Applet authors keep the full power of the immediate-mode libraries instead of a wrapper that hides them. The cost is a rebuild of every applet when the epoch changes; that should be rare, and CI flags it.

## Notes

The code implements this decision: the pin lives with the vendored libraries in `third_party/`, and the epoch rule is described in [Compatibility](../explanations/compatibility.md). [D17](d017-imgui-docking-branch.md) amended the pin to ImGui's docking branch without starting a new epoch. The pin moves into the SDK package only when the SDK is split from the host.
