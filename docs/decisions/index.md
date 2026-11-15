# Decisions

This log records each design decision of the platform in its own file. It is the source of truth: a decision changes only through a new entry that supersedes it.

| Number | Decision | Status | Date |
|---|---|---|---|
| [D1](d001-in-process-c-abi.md) | In-process C ABI plus C++ sugar | Accepted | 2026-06-10 |
| [D2](d002-service-registry.md) | Host context is a service registry | Accepted | 2026-06-10 |
| [D3](d003-no-framework-types-across-the-abi.md) | Torch and DuckDB types never cross the ABI | Accepted | 2026-06-10 |
| [D4](d004-ui-stack-pin-defines-the-epoch.md) | The UI-stack pin defines the ABI epoch | Accepted | 2026-06-10 |
| [D5](d005-runtime-packs.md) | Host-managed runtime packs, one libtorch per process | Proposed | 2026-06-10 |
| [D6](d006-registry-as-a-git-repository.md) | Registry as a git repository | Superseded | 2026-06-10 |
| [D7](d007-dynamic-crt-on-windows.md) | Dynamic CRT on Windows, ImGui allocators handed over | Accepted | 2026-06-10 |
| [D8](d008-applets-keep-their-history.md) | Applet repositories keep their history | Proposed | 2026-06-10 |
| [D9](d009-cli-as-host-subcommands.md) | A CLI as subcommands of the host binary | Proposed | 2026-06-10 |
| [D10](d010-sdk-licence.md) | The SDK licence | Proposed | 2026-06-10 |
| [D11](d011-host-without-libtorch.md) | The host ships without libtorch | Accepted | 2026-06-10 |
| [D12](d012-audience.md) | Audience is collaborators who build from source | Proposed | 2026-06-10 |
| [D13](d013-renderer-agnostic-abi.md) | Renderer-agnostic ABI, Metal and Vulkan primary, OpenGL fallback | Accepted | 2026-06-10 |
| [D14](d014-bridge-allocates-shared-tensors.md) | The bridge allocates shared tensors | Accepted | 2026-06-10 |
| [D15](d015-docs-as-code.md) | Documentation as code | Accepted | 2026-07-01 |
| [D16](d016-gpt-scope-is-the-flagship.md) | GPTScope is the flagship applet | Accepted | 2026-07-02 |
| [D17](d017-imgui-docking-branch.md) | ImGui docking branch, epoch stays 2 | Accepted | 2026-07-02 |
| [D18](d018-data-and-artifacts-built-for-embedscope.md) | data.v1 and artifacts.v1 built for EmbedScope | Accepted | 2026-07-02 |
| [D19](d019-host-names-no-applet.md) | The host names no applet | Proposed | 2026-11-09 |
| [D20](d020-applet-search-order.md) | Applet locations and search order | Proposed | 2026-11-09 |
| [D21](d021-shipped-applets-in-the-app-bundle.md) | Shipped applets live in Contents/PlugIns | Proposed | 2026-11-09 |
| [D22](d022-third-party-applet-signing.md) | Signing policy for third-party applets | Proposed | 2026-11-09 |
| [D23](d023-metal-orders-by-commit-order.md) | Metal orders work by single-queue commit order | Accepted | 2026-07-05 |
| [D24](d024-stream-ordered-handoff-through-caps.md) | CaliperTensor.stream gains meaning through bridge v1_1 caps | Accepted | 2026-07-05 |
| [D25](d025-cuda-writes-the-vulkan-image.md) | CUDA writes the Vulkan image | Accepted | 2026-11-05 |
| [D26](d026-vulkan-cuda-interop-at-buffer-level.md) | Interop at buffer level | Superseded | 2026-07-05 |
| [D27](d027-uuid-matched-device-pairing.md) | UUID-matched device pairing | Accepted | 2026-07-05 |
| [D28](d028-synchronous-interop-first.md) | Synchronous interop first, semaphores second | Accepted | 2026-07-05 |
| [D29](d029-build-time-spir-v.md) | Build-time embedded SPIR-V | Accepted | 2026-07-05 |
| [D30](d030-additive-service-revisions.md) | Additive revisions are new ids with prefix-identical tables | Accepted | 2026-07-05 |
| [D31](d031-geometry-grows-by-additive-records.md) | Geometry grows by prefix-identical records | Accepted | 2026-07-09 |
| [D32](d032-data-flows-one-way.md) | Data flows one way, tensors to pixels | Accepted | 2026-07-09 |
| [D33](d033-honest-fallbacks.md) | A missing capability falls back to a working slower path and says so | Accepted | 2026-07-05 |
| [D34](d034-opt-in-imported-allocations.md) | Imported allocations are opt-in per allocation | Accepted | 2026-07-07 |
| [D35](d035-telemetry-is-a-host-service.md) | Telemetry is a host service with host-owned providers | Accepted | 2026-07-12 |
| [D36](d036-libcaliper-owns-the-canvas.md) | libcaliper owns the applet canvas | Accepted | 2026-07-11 |
| [D37](d037-metrics-reads-through-query.md) | Metrics reads go through metrics.v1_1 query | Accepted | 2026-07-12 |
| [D38](d038-export-is-a-terminal-sink.md) | Export is a terminal sink | Accepted | 2026-07-12 |
| [D39](d039-no-io-on-the-frame-thread.md) | No I/O on the frame thread, live tables are TEMP | Accepted | 2026-07-05 |
| [D40](d040-separate-process-viewer.md) | The separate-process viewer is the primary live mode | Accepted | 2026-10-24 |
| [D41](d041-registry-at-caliperapp.md) | The applet registry at caliperapp.ai | Proposed | 2026-11-15 |

## Open questions

- The licence of the SDK is undecided, and the repository has no LICENSE file ([D10](d010-sdk-licence.md)).
- The signing policy for third-party applets on macOS is undecided ([D22](d022-third-party-applet-signing.md)).
- Whether `caliper.artifacts.v1` stays a minimal file store or the host builds a checkpoint browser.
- Whether to revive the archived OpenGLlama applet as an applet in its own repository or delete it.
- Whether an applet built outside the core may link its own heavy dependencies such as DuckDB or llama.cpp, and what happens when two applets each carry a copy.
- Whether applet-supplied shaders stay excluded once agent-built panels exist ([D32](d032-data-flows-one-way.md)).
- For the OpenGL fallback, whether to replace GLEW with a GLAD 3.3-core loader or to remove the fallback ([D13](d013-renderer-agnostic-abi.md)).
- Whether the host must accept applets built for epoch N−1 before the first outside applet ships.
- Whether a second native host that embeds libcaliper is still needed once applets live outside the core ([D36](d036-libcaliper-owns-the-canvas.md)).
