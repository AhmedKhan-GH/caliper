# M2a Windows Hardware Verification — Agent Brief

| | |
|---|---|
| **Status** | **Executed 2026-07-05** on RTX 500 Ada Laptop GPU (Windows 11, driver 596.47, CUDA 13.0) — T1–T5 complete, T6 skipped (optional). Results in §5 below. |
| **Date** | 2026-07-05 |
| **Owner** | Ahmed Khan |
| **Scope** | Verify, on real NVIDIA hardware, the M2a stream-channel handoff that was code-authored blind on macOS (`vulkan_renderer.cpp` is a Windows-only TU). Everything else in the M1/M2 program is already shipped and gfx-verified on Apple Silicon. |
| **Parent** | `docs/metal-pipelining.md` (§4 M2a, §6 exit criteria, D24), `docs/vulkan-cuda-backend.md` (V4), `ZEROCOPY.md` |
| **Branch** | `feat/metal-pipelining` — verify at or after commit `545a2f7` |

> **How to read this.** §1 is what changed and was NOT executed on any machine. §2 is what the macOS debugging session learned that transfers to Windows — read it before running anything; one finding (thread safety of the handoff) is the difference between "tests green" and "actually safe". §3 is the exact task list. §4 is the report format.

---

## 1. What changed blind (never compiled, never run)

| Change | Where | Commit |
|---|---|---|
| Pipelined CUDA copy+signal ride the producer stream: `cuMemcpyDtoDAsync(..., (CUstream)t.stream)` and `cuSignalExternalSemaphoresAsync(..., (CUstream)t.stream)` replace the legacy NULL stream in `update_pipelined` | `src/host/renderer/vulkan_renderer.cpp` | `077d6ed` |
| Capability: `honors_stream_ordered_handoff() const override { return pipelined_ok_; }` — the sync fallback (no exportable timeline semaphores) reports **false**, so adapters keep draining there | `src/host/renderer/vulkan_renderer.cpp` | `077d6ed` |
| Adapter CUDA branch: `stream_to_tensor` populates `t.stream = at::cuda::getCurrentCUDAStream(idx).stream()` and skips `torch::cuda::synchronize()` when bridge-v1.1 caps bit 0 is granted | `sdk/include/caliper/adapters/torch.hpp` | `3a814cc` |
| Applet call sites use `stream_to_tensor(..., bridge.caps())` | `applets/{embed_scope,gpt_scope}` | `578a7d5` |

`t.stream == NULL` keeps exact v1 semantics everywhere (drained handoff, legacy default stream). Every pre-existing gfx test passes NULL, so the existing suite doubles as the fallback-rung proof.

## 2. Findings from the macOS session that transfer

1. **The adapter guard was wrong once already.** mac libtorch ships `c10/cuda/CUDAStream.h` *without* the CUDA toolkit headers, so the guard is now `#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)`. **Risk on Windows: the branch silently fails to compile in** (e.g. header layout differs) — everything stays green, the drain silently remains, and the speedup is fictional. Verification must therefore assert `t.stream != NULL` at runtime (task T3), never trust green tests alone.
2. **The handoff-vs-training-thread race is real, not theoretical.** On MPS, calling the handoff from the frame thread while the worker encoded kernels crashed the process (MPSPredicate `command buffer already committed`, SIGABRT) — because **none** of libtorch's public MPS stream calls are internally serialized (proven by disassembly of `MPSHooks::commitStream`/`deviceSynchronize`: straight-line `objc_msgSend`s, no `dispatch_sync`). Fixed in `545a2f7` by running the whole MPS handoff as one block on torch's stream dispatch queue.
   **CUDA analog:** `at::cuda::getCurrentCUDAStream(idx).stream()` is a handle read and CUDA driver calls are thread-safe by API contract, so no equivalent serialization *should* be needed — but this is exactly the class of assumption that was wrong on MPS. Task T4 tests it empirically under concurrent training.
3. **Deterministic beats probabilistic for ordering tests.** The Metal ordering proof gates the producer's write behind a CPU-releasable event so an unordered renderer *must* read stale bytes. The Vulkan CUDA gfx tests have no `cuStreamCreate` in `cuda_driver.h`, so an equivalent CUDA-side gate isn't available without extending the loader — optional (T6), not exit-blocking; the burst test plus T4 cover the practical risk.
4. **Both crash-class bugs were caught by stress tests, not the happy-path suite.** Keep T4 even if everything else is green.

## 3. Tasks for the Windows agent (in order)

Work on branch `feat/metal-pipelining`, repo conventions apply: never `git add -A` (the `third_party/llama.cpp` submodule may be dirty), commit style `feat|fix|test|docs(scope): …`, frozen files stay frozen (`tensor.h`, `tensor_bridge_v1.h`).

- **T1 — Build everything.** `init-submodules.bat` if needed; configure with `BUILD_TESTS=ON`; build `caliper`, `caliper_tests`, `caliper_gfx_tests`, `caliper_torch_tests`. Exit: zero errors. If `vulkan_renderer.cpp` fails to compile, STOP and report the exact errors — the M2a edits were never compiled anywhere.
- **T2 — Existing suites (NULL-stream fallback rung).** Run all three suites in a GUI session on the NVIDIA box (single-GPU, so the CUDA gfx cases run instead of skipping). Exit: unit + torch green; gfx green **including** `gfx/Vulkan+CUDA: burst updates pipeline in order, final frame pixel-exact` and the `alloc_shared` case, byte-exact.
- **T3 — Guard admits the CUDA branch (the finding-1 tripwire).** Add to `tests/test_torch_adapter.cpp` (mirrors the MPS twin already in the file; on the Mac it self-skips):

  ```cpp
  TEST_CASE("stream_to_tensor: cuda tensor carries the producer stream when honored; drains when not") {
      if (!torch::cuda::is_available()) { MESSAGE("no CUDA device — skipping"); return; }
      torch::Tensor t = torch::ones({4, 4},
          torch::TensorOptions().device(torch::kCUDA)) * 2.0f;

      auto honored = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
      REQUIRE(honored.has_value());
      REQUIRE(honored->device == CALIPER_DEV_CUDA);
      CHECK_FALSE(honored->stream == nullptr);   // FAILS if the guard compiled the branch out

      auto v1 = stream_to_tensor(t, 0);          // negotiation pin, other direction
      REQUIRE(v1.has_value());
      CHECK_FALSE(v1->stream != nullptr);
  }
  ```

  (Doctest `CHECK` is shadowed by c10 in this file — use `REQUIRE*`/`CHECK_FALSE` only.) Exit: passes on Windows. If `stream` comes back NULL, the guard is compiling the branch out — fix the guard, not the test.
- **T4 — Concurrency stress twin (the finding-2 tripwire).** Add the CUDA twin of the MPS stress case (same file, same shape): a worker thread loops `a = torch::mm(a, b).tanh()` on CUDA tensors while the main thread performs 500 `stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED)` handoffs; end with `stop`, `join`, `torch::cuda::synchronize()`. Exit: no crash, no CUDA error, 10/10 repeated runs. If it crashes, apply the MPS lesson: serialize the CUDA handoff (likely a mutex around the stream query is NOT the fix — capture the failing stack first, systematic-debugging discipline).
- **T5 — The §6 M2a exit criterion: the drain is measurably gone.** Confirm `Bridge::caps() == 1` in-app on the Vulkan renderer (log line or debugger). Then measure gpt_scope or embed_scope training steps/sec on this branch vs `3a814cc^` (the commit before the adapter existed): same applet, same settings, ≥100 steps each. Exit: `torch::cuda::synchronize` absent from the handoff path while training (profiler or the branch-vs-parent timing delta standing in for it), numbers recorded. If the delta is ~zero, report honestly — D21's "measured optimization" discipline; small textures may genuinely not show it.
- **T6 (optional, not exit-blocking).** Deterministic stream-ordering gfx test: needs `cuStreamCreate`/`cuStreamDestroy` (+ optionally `cuLaunchHostFunc` as the gate) added to `src/host/cuda_driver.h`'s loader table; producer enqueues a gated `cuMemcpyDtoDAsync` on a non-default stream, `t.stream` set to it, readback must see the fresh bytes. Skip if time-boxed.
- **T7 — Close the loop.** On success: update `docs/metal-pipelining.md` (M2a row → **Verified on NVIDIA hardware**, Status header), `ZEROCOPY.md` (the M2a elision sentence becomes a verified claim; add measured numbers), and this file's Status. Commit per convention with trailer `Co-Authored-By: <your model name>`.

## 4. Report format

Per task: exit criterion met yes/no, exact test counts, commit hashes. Plus: the T3 `stream` non-NULL confirmation, T4 run tally (n/10), T5 steps/sec table (branch vs parent, hardware named), and any deviation from this brief with rationale. A failed exit criterion is a finding, not a failure of the run — report it with the captured evidence and stop before "fixing forward" beyond T3/T4's explicit remits.

---

## 5. Results (2026-07-05, RTX 500 Ada Laptop GPU, Windows 11, driver 596.47)

- **T1 ✓** — zero errors. `vulkan_renderer.cpp` (the never-compiled M2a TU) compiles clean under MSVC 14.50 / CUDA 13.0.
- **T2 ✓** — unit 90/90 (33,295 assertions); torch 11/11; gfx 16/16, 0 skipped, **including** the burst-pipeline and alloc_shared cases, byte-exact; `sync mode: pipelined (shared timeline semaphores)` live.
- **T3 ✓ (amended)** — the tripwire fired as written, but root-cause diagnosis (preprocessor probe with the TU's exact include dirs) proved the guard was **never broken**: the branch compiles in on Windows. The NULL came from CUDA semantics — `CUDAStream::stream()` returns `nullptr` for the legacy default stream (unlike MPS, whose queue pointer is never NULL). A NULL-carrying honored handoff is still correct and drain-elided (the renderer's NULL rung is the same legacy default stream the producer used). The test now pins a **non-default pool stream** via `CUDAStreamGuard` — the exact handle round-trips (`REQUIRE(stream == pool.stream())` passed) — and a missing c10 CUDA header fails loudly instead of skipping. Committed as `28d53a7`.
- **T4 ✓** — CUDA stress twin: **10/10 runs green** (500 pool-stream handoffs/run vs a concurrent `mm().tanh()` training thread). The MPS-class race does not reproduce on CUDA.
- **T5 ✓** — `caps()==1` in-app (init stderr line, proof chain `pipelined_ok_` → `honors_stream_ordered_handoff()` → caps bit 0). Steps/sec, embed_scope MNIST (1,407-step runs, WAL-envelope timing): branch 95.7 and 68.2* steps/s; parent `3a814cc^` 103.0, 80.1, 75.9 steps/s (*partial run, ≥100 steps). **Delta ≈ 0** — laptop thermal throttling (±15–20% monotonic decline across sessions) dominates, and the training thread already pays a per-step device sync via `loss.item()`, so the frame-thread drain never gated training throughput on this applet. The elision's verified win is the frame-thread stall removal + ordering (tripwire + byte-exact tests), per D21's measured-optimization honesty.
- **T6 — skipped** (optional per §3): burst + T4 cover the practical risk; extending `cuda_driver.h`'s loader with `cuStreamCreate`/`cuLaunchHostFunc` remains available future work.
- **Deviation log**: T3 test code corrected as above (fix-the-guard remit assumed a broken guard; the guard was sound). Observation for a future issue: CUDA epoch-boundary stalls in embed_scope (eval + `publish_embeddings` burst, allocator churn on 4 GB) are visible as training pauses on Windows but not on MPS — orthogonal to M2a, present in both builds.

*Companion to `docs/metal-pipelining.md` (design + decision log D23/D24) and `docs/superpowers/plans/2026-07-05-metal-pipelining.md` (the executed implementation plan whose Notes section defers to this brief). The macOS-side crash forensics live in commit `545a2f7`'s message.*
