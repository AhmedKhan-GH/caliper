# caliper.tensor_bridge.v1

Service id `caliper.tensor_bridge.v1` — the platform's USP, productized: a
`CaliperTensor` becomes a live texture this frame, GPU-resident on the native
backends (PLATFORM.md §7.4). This page embeds the header verbatim; the docs
build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/tensor_bridge_v1.h"
```

## Semantics

The bridge turns a `CaliperTensor` into a live texture **this frame**. It is the
platform's reason to exist: on the native backends the tensor's device memory
*becomes* the texture with no CPU round-trip; the ABI never names a graphics API,
so the renderer stays swappable forever (see [Rendering](../../explanation/rendering.md)).

### Acceptance rules (the v1 gate)

The host validates every tensor before it becomes a texture. A violation returns
`0`/`false` and emits one `caliper.log.v1` line — the bridge **never**
misinterprets bytes into a wrong texture. Faithful to the shipped gate:

- **2-D `(H,W)` `f32`** → `texture_from_tensor_mapped` (colormapped through a
  built-in LUT, scaling `[vmin,vmax]` → `[0,1]`);
- **3-D `(H,W,C≤4)` `u8`** → `texture_from_tensor` (direct RGBA/…);
- **contiguous** (row-major, no gaps);
- **device CPU or the active backend's device** (e.g. Metal on macOS).

The applet-side [torch adapter](../../reference/adapters.md) enforces the mirror
of these rules before the tensor ever reaches the host — rejecting rather than
silently copying.

### UI-thread-only

Every bridge entry point is **frame-thread-only**. Unlike
[`caliper.metrics.v1`](metrics-v1.md) (callable from a job thread), the bridge
touches renderer state and must be called only from the applet's `frame()`. In
the MLScope exemplar the training worker *snapshots* weights under a mutex and
never calls the bridge; the frame thread reads that snapshot and does every
upload. Textures are therefore frame-thread-owned and released on the frame
thread (after the job wait, before renderer teardown).

### Per-backend behavior

The applet code is **identical** on both backends. Where the staging happens
differs, and the honest device-path string tells you which ran:

| Backend | Path | Staging | Status |
|---------|------|---------|--------|
| **Metal** (`CALIPER_RENDERER=metal`) | device **compute** (f32 + LUT) / **blit** (u8 HWC) | **zero CPU staging** — the MPS `MTLBuffer` is colormapped on-GPU | §16-verified pixel-exact vs a CPU reference (C5) |
| **GL** (default) | CPU-staged upload | the *bridge* stages; the applet never touches a pixel | §16-verified pixel-exact; `tex_update_from_device` always returns `false` (frozen fallback) |

The **§16 contract** is proven per backend: `caliper_gfx_tests` uploads known
tensors, reads the texture back, and compares byte-for-byte. Metal's `compute`
path is exact vs the CPU `map_f32_to_rgba8` reference at ragged sizes (4×4, 5×3,
17×9); the `blit` path is exact vs `expand_u8_to_rgba8`.

!!! note "One tensor cannot be both"
    A single tensor is either zero-copy on Metal *or* accepted on GL, not both:
    the GL bridge's active device is CPU and rejects an MPS tensor as a foreign
    device. The exemplar hands the training-device tensor first; if the create
    returns `0` (non-Metal renderer) it relocates the tensor to CPU and the
    bridge stages it. The applet does no pixel work on either path (§6c) — the
    bridge's own accept/reject drives the choice.

### Lifecycle: create once, update after

- `texture_from_tensor` / `texture_from_tensor_mapped` **create** a texture and
  return a `CaliperTextureId`.
- `update_texture` **re-uploads** into an existing texture of the **same
  shape/dtype**. Create once on the first snapshot, `update_texture` thereafter —
  do not recreate per frame.
- `release_texture` frees it.

!!! warning "Pinned range in v1 (frozen)"
    `update_texture` has **no colormap-range channel** — it is frozen. The
    `[vmin,vmax]` of a colormapped texture is fixed at creation. MLScope pins the
    symmetric RdBu range at the first kernel snapshot; the filters still visibly
    sharpen (structure changes, not just scale), and the UI states this honestly.
    Recreating the texture per snapshot to re-range would violate create-once and
    is deliberately not done.

### `alloc_shared` — v1 honesty

`alloc_shared` allocates tensor memory that **is** the texture's backing store:
the applet wraps `out_tensor->data` (e.g. `torch::from_blob`) and writes into it,
and the texture sees the result after at most a layout transition. In v1 it
returns a **unified-memory CPU-device tensor** — literal zero-copy for **CPU
writers**; **device writers** must still stage through `update_texture`. Free it
with `free_shared`.

### `CaliperTextureId` lifetime

`CaliperTextureId` is a `uint64_t`, opaque to applets (`0` = invalid;
compare-only — never interpret or dereference it, its representation is
backend-internal). Its value is directly castable to `ImTextureID` for
`ImGui::Image`: the host vends the ImGui-compatible handle per backend (the GL
texture name on GL, the `id<MTLTexture>` pointer on Metal), so the cast Just
Works on both — binding an integer table id here is what crashed
`ImGui_ImplMetal` on the first `Image`. The renderer stays swappable because the
value is the host's business, never the applet's (§5.4). Ids from
`texture_from_tensor*` are freed with `release_texture`; ids from `alloc_shared`
with `free_shared`.

### Sync model

The ABI is stream-free in v1 (`stream == NULL`). Correctness on device textures
comes from the applet draining the device **once** at the handoff
(`torch::mps::synchronize()` via the adapter's `synced_to_tensor`) — sync-then-
update — not from a stream channel. See the [adapter reference](../../reference/adapters.md)
for the cost of that barrier and why you pay it once, not per frame.

---

## Demo checklist (human)

The Phase-2C acceptance demo. These are the live-visual checks that automation
**cannot** cover (they require clicking *start* and watching training on a
display); the machine verification — crash-free startup on both backends and
per-backend pixel-exactness of the exact bridge entry points MLScope uses — is
green (C5/C8).

1. **Metal, GPU-resident.** `CALIPER_RENDERER=metal ./build/caliper`, open
   **MLScope**, click *start*. The 4×2 conv1 kernel grid appears; the RdBu tiles
   **sharpen from noise** into structured filters as the loss falls. The status
   line reads **GPU-resident (Metal, zero CPU staging)**.
2. **GL, identical visuals.** Relaunch on the default GL renderer (no env var),
   train again. The kernel grid is **visually identical**; the status line reads
   **CPU-staged (GL fallback)**. Same applet code — only where the staging
   happens differs.
3. **Cancel / relaunch mid-training.** Cancel training partway (or relaunch the
   app mid-run). The grid **persists its last snapshot**, and there is **no
   crash** — textures are released on the frame thread after the bounded job
   wait.
4. **No frame hitching.** With kernel textures updating every eval cadence point,
   the **Runs dashboard keeps streaming** its loss/accuracy curves smoothly — no
   frame hitching from the uploads.

### Phase-2D additions (bridge-native applets, Metal default)

Phase 2D moved the last raw-GL applets onto the bridge and flipped the macOS
default renderer to Metal. The `grep -rn 'glGenTextures\|glTexImage\|glDeleteTextures\|glBindTexture' applets/`
sweep (§6c) is now **empty** — every applet texture crosses as a
`CaliperTextureId`. These checks extend the demo:

5. **OpenGllama heatmaps (bridge-native).** Open **OpenGllama**, load a GGUF
   model, run a generation. Switch the context-heatmap mode through **EMA
   (decay) / Max / Recent / Final Layer / Single Layer** — every mode renders
   the attention overlay over the context text. All modes change the *composed
   RGBA pixels* and reach the **one** bridge upload path (create-once, then
   `update_texture` in place; recreate only when the text reflows and the size
   changes). The applet issues no raw GL — `tensor_bridge.v1` is now a
   **required** service (an unmet requirement leaves the card unavailable rather
   than crashing).
6. **RepNet viz tabs.** Open **repnet_demo**; the **Model** tab's weight/kernel
   heatmaps and the per-lead detail views render through the same bridge upload
   (RdBu/diverging colormap composed to RGBA8, then `texture_from_tensor`). The
   tabs recompose-then-reupload on dirty (a release-then-create path), and switch
   without artifacts.
7. **MLScope real-data panel.** In MLScope, start training and watch the
   real-data panel: a **fixed probe digit** (t10k[0]) rendered VIRIDIS on the
   left with its **conv1 8× (26,26) feature maps** in a 4×2 grid on the right.
   The maps **sharpen live** across the run as conv1 learns, and the caption
   reads **`pred N / true N`** (green when they agree). The probe reuses the same
   worker-snapshot frame the kernel grid does — no extra bridge calls off the
   frame thread.
8. **Default-flip expectations.** A **bare** `./build/caliper` launches on
   **Metal** (macOS default; the startup line prints `[renderer] metal`). One
   honest consequence: the landing-page 3D background (`IntroScreen`, still raw
   GL) is **absent** on Metal — you get the plain app shell, cards and launch
   flow intact. Relaunch with `CALIPER_RENDERER=gl` for the **full landing**
   (animated 3D backdrop) on the frozen GL fallback; every applet above renders
   identically on both backends — only where the bridge stages the pixels
   differs.

### Phase-2E′ additions (GPTScope, the flagship)

Phase 2E′ shipped **GPTScope** — a char-level mini-GPT trained live on
TinyShakespeare, built entirely on the public service stack. These checks are the
flagship's live-visual acceptance demo (the machine verification — build green,
full `ctest` + `caliper_gfx_tests` + torch-label suites, both renderers headless
for 10s — is green):

9. **Sample-evolution arc.** Open **GPTScope**, click *start*. The TinyShakespeare
   corpus **downloads once** into the data dir (cached forever; a second run is
   offline-clean), then training begins. Train loss **falls**, and the live sample
   panel **evolves from gibberish → words → Shakespearean cadence** across the
   ~3-minute run. The **val perplexity** readout beside the loss plot
   (`exp(val_loss)`) drops alongside.
10. **Attention grid, live.** The per-head attention panel shows the selected
    layer's **4 heads** as VIRIDIS heatmaps (per-head vmax) over a fixed val
    excerpt, refreshing every eval cadence — the heads **sharpen** as the net
    learns. Switching **layer L0–L3** repoints the snapshot (the map updates one
    eval tick after the click). **Hovering** a map highlights the excerpt's
    **source char (row, cyan)** and **target char (col, amber)** — the touch that
    makes attention legible. The status line reads **GPU-resident (Metal, zero CPU
    staging)** on Metal, **CPU-staged (GL fallback)** on GL.
11. **Temperature control.** Drag the **temperature slider** (0.2–1.5); the next
    sample's character changes — **lower is greedier/sharper/more repetitive,
    higher looser/more diverse** (the worker reads the live value at each sample
    tick, so the change lands on the following sample).
12. **Cancel / relaunch clean.** Cancel partway (or relaunch mid-run): the last
    loss curve, sample, and attention grid **persist**, with **no crash** —
    textures release on the frame thread after the bounded job wait.
13. **Runs dashboard.** The GPTScope run appears in the **Runs dashboard**
    (train/loss, val/loss) **alongside MLScope history** — the same optional
    `caliper.metrics.v1` path, streaming smoothly with no frame hitching from the
    attention uploads.
14. **Both renderers.** All of the above renders identically on **Metal**
    (`CALIPER_RENDERER=metal`) and the default **GL** fallback — same applet code,
    only where the bridge stages the pixels differs.

!!! note "Deferred by design: checkpoint save"
    GPTScope's *save checkpoint* button is **disabled** with a tooltip saying it
    arrives with `caliper.artifacts.v1`. That is the honest placeholder for the
    D16 demand-driven clause — checkpointing is the first real demand for an
    artifacts service, not something faked here.
