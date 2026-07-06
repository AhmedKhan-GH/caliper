# Metal / MPS Handoff Pipelining — Specification

| | |
|---|---|
| **Status** | Implemented & verified — M1 + M2b shipped and gfx-verified on Apple Silicon (in-app step-time measurement pending); M2a **verified on NVIDIA hardware** (RTX 500 Ada laptop, Windows 11, 2026-07-05 — `docs/m2a-windows-verification.md`) |
| **Date** | 2026-07-05 |
| **Owner** | Ahmed Khan |
| **Scope** | The macOS counterpart of the Windows V4 work: remove the CPU sync stalls from the Metal backend's device-update path, and — the part that actually matters, on both platforms — retire the adapter's full-device drain at the tensor handoff by giving the dormant `CaliperTensor.stream` channel semantics. |
| **Parent** | `PLATFORM.md` §5.4/§7.2, `ZEROCOPY.md`, `docs/vulkan-cuda-backend.md` (V4 = the design donor) |

> **How to read this document.** §1 is the verified current state. §2 ranks the two distinct stalls by measured impact — they are *not* equally important, and the cheap fix is not the valuable one. §3–§4 are the proposed target in spec present tense. §6 sequences three independently shippable increments; §7 proposes decisions D23–D24 continuing the log from `vulkan-cuda-backend.md` (D19–D22).

---

## 1. Where We Are

| Asset | Location | State |
|---|---|---|
| Metal backend, synchronous device paths | `src/host/renderer/metal_renderer.mm` | `colormap_compute` and `blit_u8` each end in `[cb waitUntilCompleted]` (`:304`, `:334`) — the CPU blocks per texture op. This is the model the Vulkan backend copied as its v1 and then replaced in V4. |
| Adapter-side full-device drain | `sdk/include/caliper/adapters/torch.hpp:110-113` | `synced_to_tensor`: `torch::mps::synchronize()` (MPS) / `torch::cuda::synchronize()` (CUDA) at every device handoff — a full device barrier, training kernels included. The v1 sync-then-update contract. |
| The dormant stream channel | `sdk/include/caliper/tensor.h:26` | `void* stream; /* cudaStream_t / MTLCommandQueue* / NULL */` — **the field already exists in the frozen ABI.** v1 behavior: always NULL. No struct change or epoch bump is needed to use it; what's missing is defined semantics and a capability signal. |
| The measured cost, in-tree | `applets/embed_scope/embed_scope.cpp:455-456` | *"The extra forward is noise next to the per-step MPS sync cost that dominates step time."* The dominant stall on Mac is the **adapter drain**, not the renderer waits. |
| Windows V4 (design donor) | `src/host/renderer/vulkan_renderer.cpp` | Shipped: per-texture shared timeline semaphore GPU-orders copy → colormap → frame-sample; only host wait is `retire()` back-pressure; synchronous fallback per texture. Verified 16/16 gfx, byte-exact burst test. |
| Metal test readback | `tests/gfx/gfx_main.cpp:385-414` | `metal_readback` blits on **its own `MTLCommandQueue`** — safe today only because the renderer waits per op. Becomes a stale-read hazard the moment M1 drops those waits (§3.1). |
| Renderer-owned readback hook | `src/host/renderer/host_renderer.h` | `debug_readback_rgba8` (added for Vulkan) — the mechanism M1's test-readback fix reuses. |

**Why Metal never "needed" V4's machinery:** Windows pipelining required cross-API external semaphores because CUDA and Vulkan are two drivers sharing one allocation. On Metal there is **no interop** — one API, and the renderer submits tensor ops and the frame render to the *same* `MTLCommandQueue` (`metal_renderer.mm:341`). Metal guarantees command buffers on one queue begin execution in commit order, so most of V4's apparatus is unnecessary: draw-after-update correctness comes free with commit ordering.

---

## 2. The Two Stalls, Ranked

| # | Stall | Where | Cost today | Fix |
|---|---|---|---|---|
| **S1** | Adapter full-device drain at handoff (`torch::mps::synchronize` / `torch::cuda::synchronize`) | applet/adapter, both platforms | **Dominant** — documented as the per-step cost that "dominates step time" on MPS; couples visualization cadence to the training queue | **M2** (stream channel) |
| **S2** | Renderer per-op CPU waits (`waitUntilCompleted` ×2 per f32 update on Metal) | `metal_renderer.mm` | Minor at current workloads (small, gen-gated textures); the Metal twin of what V4 removed on Windows | **M1** (trivial) |

The honest ordering: **M1 is cheap hygiene; M2 is the real win** — and M2's CUDA half also completes Windows V4, whose one residual is exactly this adapter drain.

---

## 3. Design — M1: drop the Metal renderer's per-op waits

`colormap_compute` and `blit_u8` stop calling `waitUntilCompleted`. Correctness analysis:

1. **Draw ordering — free.** The tensor-op command buffer and the frame's command buffer are committed to the same queue, in that order (bridge calls happen during the applet's `frame()`, before `render()` commits). Commit order ⇒ execution order on one queue. The frame samples finished texels with zero added sync.
2. **Resource lifetime — free.** `MTLCommandBuffer` retains referenced resources until completion (default, non-`unretained` encoding). The per-dispatch LUT buffer (`metal_renderer.mm:285-288`) and even a texture whose last strong reference is dropped by `tex_release` mid-flight stay alive until the GPU is done. ARC + retention replace Vulkan's explicit `retire()`/queue-idle discipline — Metal needs **no per-texture sync objects, no back-pressure wait, no pending-waits list**.
3. **Same-texture rapid re-update — free.** Two updates to one texture are two command buffers on one queue: serialized by commit order. (Vulkan needed `retire()` because it re-records one command buffer per texture; Metal allocates a fresh `MTLCommandBuffer` per op, so there is nothing to re-record.)
4. **CPU readback — the one real hazard.** The gfx harness's `metal_readback` blits from a *different* queue; cross-queue has no ordering guarantee once the renderer stops waiting. Fix: implement `debug_readback_rgba8` in `MetalRenderer` (the hook Vulkan already uses) — blit on the **renderer's own queue** (commit order ⇒ reads retired state) with `waitUntilCompleted` only there, and switch the Mac gfx harness's `Backend::readback` to it. Waits belong in test readbacks; they leave the hot path.

`last_device_path()` reporting is unaffected (set at encode time). The u8/f32 acceptance rules, extent bounds check, and pixel math do not change.

---

## 4. Design — M2: stream-channel handoff (retires S1; cross-platform)

**The contract change.** The adapter may populate `t.stream` instead of draining the device:

- **CUDA:** `t.stream = (void*)at::cuda::getCurrentCUDAStream().stream()`. The Vulkan renderer enqueues its existing `cuMemcpyDtoDAsync` + timeline-semaphore signal **on that stream** instead of the legacy NULL stream. Stream order guarantees the copy runs after the producer's kernels — `torch::cuda::synchronize()` is elided entirely. (This is the smaller half: V4's machinery already exists; only the stream argument changes.)
- **Metal:** the producer's pending work lives in torch's `MPSStream` command buffer, which may be **uncommitted** — queue-level ordering alone cannot see it. The adapter therefore (a) commits torch's pending command buffer *without waiting* (`torch::mps::commit()` — enqueue, not drain), and (b) sets `t.stream` to torch's `MTLCommandQueue*`. The renderer keeps one `MTLSharedEvent` per texture (the timeline-semaphore analog, uint64 valued): it commits a tiny command buffer on the **producer's** queue that signals `base+1` (ordered after the just-committed producer work), and encodes `encodeWaitForEvent:value:base+1` at the head of its own tensor-op command buffer. GPU-ordered producer → consumer, no CPU block anywhere. Frame ordering after the op is still free (M1, same queue).
- **NULL stream = today's behavior.** A NULL `stream` keeps the v1 contract: the adapter drained, the renderer proceeds as now. Every rung of the degradation ladder survives.

**Capability negotiation — the one genuinely new mechanism.** The adapter must not skip its drain against a host that ignores `stream`. The bridge vtable is frozen, so: register `CALIPER_TENSOR_BRIDGE_V1_1` in the services registry — same table plus one query, `uint32_t caps(void)`, with bit 0 = *stream-ordered handoff honored*. `synced_to_tensor` (or a new `stream_to_tensor`) checks the bit once: present → populate `stream`, skip the drain; absent → drain as today. Old hosts never see a non-NULL stream from a well-behaved adapter; old applets on new hosts pass NULL and get v1 semantics. No epoch bump, no struct change.

**What M2 explicitly does not promise:** the producer's *thread-safety* story is unchanged — the applet still hands over a tensor it owns at a quiescent point in its own logic; M2 only replaces *how completion of already-enqueued GPU work* is awaited (GPU event instead of CPU drain).

---

## 5. What Deliberately Does NOT Change

| Thing | Why |
|---|---|
| `CaliperTensor` struct, ABI epoch | `stream` already exists; M2 defines semantics + a v1.1 caps query, both additive. |
| Pixel-exactness contract (§16) | Same kernels, same bytes; only *when the CPU waits* changes. Byte-equality tests must keep passing unmodified. |
| GL backend / CPU staging ladder | Untouched; NULL-stream fallback preserves every rung. |
| Windows V4 renderer machinery | M2-CUDA changes one stream argument; the semaphore chain is as shipped. |
| `synced_to_tensor` as the safe default | Remains correct and available; stream handoff is opt-in per call site. |

---

## 6. Increments (each independently shippable)

| # | Deliverable | Exit criterion |
|---|---|---|
| **M1** | Metal renderer drops per-op `waitUntilCompleted`; `MetalRenderer::debug_readback_rgba8` on the renderer's queue; Mac gfx harness readback switched to it | Mac gfx suite green and byte-exact; a burst test (the Vulkan one's twin: N back-to-back device updates, readback once) passes; no per-op CPU wait remains in the hot path. **Shipped:** gfx suite + burst test byte-exact; sole remaining wait is the test readback. |
| **M2a** | CUDA stream channel: adapter populates `t.stream`, Vulkan renderer enqueues copy+signal on it, bridge v1.1 `caps()` bit, adapter drain skipped when honored | gpt_scope/embed_scope training on Windows shows no `torch::cuda::synchronize` in the handoff profile; gfx suite green; NULL-stream fallback covered by a test forcing v1 behavior. **Verified on NVIDIA hardware** (RTX 500 Ada, 2026-07-05): gfx 16/16 incl. burst + alloc_shared byte-exact; the adapter carries a non-default producer stream end-to-end (pool-stream tripwire — note: the DEFAULT stream's handle is legitimately NULL, still drain-elided); 10/10 concurrency stress vs a training thread; `caps()==1` in-app. embed_scope steps/sec delta vs the drained parent ≈ 0 (laptop thermal variance ±15% dominates; the training thread already syncs per step via `loss.item()`) — honest per D21: the elision's win is frame-thread stall removal, proven by the tripwire + byte-exact ordering, not throughput. |
| **M2b** | Metal stream channel: `MTLSharedEvent` per texture, producer-queue signal after `torch::mps::commit()`, consumer-side `encodeWaitForEvent` | embed_scope on Mac no longer pays the "per-step MPS sync cost that dominates step time" (`embed_scope.cpp:455` comment retired with measurement); Mac gfx suite green. **Shipped:** gated-producer ordering test byte-exact; adapter commits + hands the queue; embed_scope step-time measurement pending a manual training run. |

M1 has no dependency on M2. M2a before M2b: it reuses shipped V4 machinery and validates the caps-negotiation design on the platform with the stronger test harness, before the Metal half takes on the torch-MPS API risk.

---

## 7. Decisions for Ratification (continuing D19–D22)

| # | Decision | Rationale / trade accepted |
|---|---|---|
| D23 | **Metal pipelines by single-queue commit order, not sync objects** — no per-texture events/back-pressure for draw correctness; `MTLSharedEvent` appears only where cross-queue ordering genuinely exists (M2b producer queue; test readback stays wait-based on the renderer's queue) | Metal's queue semantics + command-buffer resource retention give what Vulkan needed explicit machinery for; adding that machinery anyway would be cargo-culting the Windows design |
| D24 | **`CaliperTensor.stream` gains semantics via bridge v1.1 `caps()`** — non-NULL stream means "order on this producer stream instead of a drained device"; NULL keeps v1; adapters check the caps bit before skipping the drain | Retires the dominant measured stall on both platforms without touching the frozen struct or epoch; the cost is one more registered service id and a compat matrix (old host × new applet and inverse) that the negotiation test suite must pin |

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| torch MPS API surface for M2b (`torch::mps::commit`, stream/queue accessors) shifts across torch versions | Medium | Medium | Isolate behind the adapter header (applet-side, version-checked); NULL-stream fallback is always available |
| Cross-queue test readback reads stale texels after M1 | Certain if unfixed | High (flaky byte-equality) | Fixed *inside* M1 by design (renderer-queue readback); the burst test exists to catch exactly this |
| Old-host/new-applet caps mismatch silently skips a needed drain | Low | High | The caps bit is the only trigger for skipping; negotiation tests pin both directions; adapters default to drain |
| Metal command-buffer retention assumption (non-`unretained` encoding) | Low | Medium | The backend never opts into `unretained`; assert/document at the encode sites |
| M2 benefit overstated for tiny textures | Medium | Low | Exit criteria demand *measured* removal of the drain from the profile, mirroring D21's "measured optimization" discipline |

---

*Companion to `docs/m2a-windows-verification.md` (the Windows verification agent brief for M2a), `docs/vulkan-cuda-backend.md` (V4, the shipped design this mirrors and completes), `ZEROCOPY.md` (data-crossing budgets unchanged), `PLATFORM.md` §7.2 (the v1 sync-then-update contract M2 supersedes when negotiated). Implementation donors: `metal_renderer.mm` (the two wait sites), `vulkan_renderer.cpp` (timeline chain, `debug_readback_rgba8`, burst-test pattern), `adapters/torch.hpp` (the drain to retire).*
