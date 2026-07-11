# Caliper landing page — copy deck

Final copy for the public case-study page. Task 4 pastes each section
verbatim into HTML. Section ids match the page sections one-to-one.

---

## hero

**Hero claim (verbatim, largest type on the page):**

> GPU tensors become pixels without ever touching the CPU — byte-exact on Metal and Vulkan.

**Descriptor (one line under the claim):**

Caliper is a native C++ desktop platform for watching a neural network's
internal state the same frame the GPU computes it.

**Supporting sub-line:**

Every mainstream tool copies the tensor off the GPU, through the CPU, onto
disk, and back onto a GPU to draw it. Caliper deletes that round trip and
proves the result pixel-exact against a CPU reference on both hardware
ecosystems that matter — Apple Silicon and Windows/NVIDIA.

**Primary CTA label:** View on GitHub

**Secondary CTA label:** Read the whitepaper

---

## problem

**Heading:** The round trip everyone else takes

A model's internal state — weights, activations, embeddings — lives in GPU
memory. Every mainstream tool for looking inside a training run copies it to
CPU memory, encodes it to an image, writes that to disk, polls the file from a
server, decodes it in a browser, and uploads it back onto a GPU to display.
Two bus crossings, one filesystem, seconds of latency — for data that started
on the same class of chip that draws your screen. So researchers sample
sparsely, every N minutes or every epoch, and whole categories of fast
dynamics — what happens in the first two hundred optimizer steps, how a
representation reorganizes during a loss spike — are simply never seen.

---

## hard-part

**Heading:** Why nobody had already deleted it

Three walls each independently force the CPU round trip. The **API wall**:
compute frameworks speak CUDA/MPS, renderers speak Vulkan/Metal — the
external-memory escape hatches are recent, platform-specific, and rarely used
outside game engines. The **process wall**: standard tooling puts the viewer
in a separate process, and once you serialize GPU-resident data the round trip
is already lost. The **correctness wall**: sharing memory between a framework
that is still writing and a renderer that is reading is a race by default;
getting it wrong renders yesterday's bytes instead of crashing politely. The
contribution is crossing all three at once, portably, behind a C contract
small enough to freeze. And the bar for "it works" is not a screenshot — it is
**byte-exact**: every GPU rendering path must read back identical to a single
CPU reference implementation, tested per backend on real hardware every CI run.

**Side-by-side platform note — Apple Silicon (Metal / MPS):**

Unified memory means CPU and GPU share one pool of physical RAM, so a PyTorch
MPS tensor's storage *already is* an `MTLBuffer` — the object Metal renders
from. The handoff is a pointer cast plus safety rules: the frozen struct has
no storage-offset channel, so the adapter rejects non-contiguous tensors and
views rather than silently addressing the wrong texels. Zero host copies, one
on-GPU blit.

**Side-by-side platform note — Windows/NVIDIA (Vulkan / CUDA):**

Discrete GPUs separate VRAM from system RAM, so zero-copy means keep the data
in VRAM and make both APIs share the allocation. The renderer exports (Vulkan
external memory, opaque Win32 handle) and CUDA imports — because torch's
caching allocator can't export. Device pairing is UUID-driven; a hybrid-GPU
mismatch disables interop rather than corrupting. CUDA VMM blocks pad up to the
driver's ~2 MiB allocation granularity, so bounds are derived from the padded
size, not the tensor's. Synchronization is GPU-ordered by a shared timeline
semaphore — no CPU thread waits on the hot path.

---

## built

**Heading:** What's actually built

*(5–6 cards. Each: title + one-liner.)*

**Card 1 — The applet contract**
One C descriptor export at ABI epoch 2; new services add without breaking existing applets.

**Card 2 — Eight host-neutral services**
ui, log, device, tensor_bridge, jobs, metrics, data, artifacts — negotiated by id, versioned, never breaking.

**Card 3 — The HostRenderer seam**
One internal interface, three backends: zero-copy Metal, Vulkan/CUDA interop, and a CPU-staged OpenGL fallback.

**Card 4 — The tensor bridge**
A GPU tensor becomes an `ImGui::Image` the same frame — no CPU staging on native backends.

**Card 5 — The geometry ladder (R0–R3)**
Points, meshes, textured meshes, instanced transforms — all vertex-pulled in place from device tensors.

**Card 6 — The verification wall**
Every GPU path is byte-compared to a single CPU reference, per backend, every CI run.

---

## proof

**Heading:** Byte-exact, both backends, on real hardware

The zero-copy geometry service shipped as an additive ladder — points, then
indexed meshes, then textures on meshes, then instanced transforms. Each rung
draws vertex-pulled in place from imported device allocations, and each is
verified byte-identical to the same CPU reference on both platforms' real
silicon (Apple Silicon and an RTX 500 Ada).

| Geometry rung | Metal / MPS (Apple Silicon) | Vulkan / CUDA (Windows/NVIDIA) |
|---|---|---|
| R0 — `geometry.v1` instanced points | Shipped, byte-exact | Shipped, byte-exact |
| R1 — `geometry.v1_1` indexed triangles / lines / strips | Shipped, byte-exact (13-row §9.2 matrix) | Shipped, byte-exact (13-row matrix mirrored) |
| R2 — `geometry.v1_2` textures on meshes | Shipped, byte-exact | Shipped, byte-exact |
| R3 — `geometry.v1_3` instanced transforms | Shipped, byte-exact | Shipped, byte-exact (gfx 48/48 live) |

**Provenance line (quoted from the repository's run log):**

> "first zero-copy instanced frame drawn — 1000 objects, 1 draw call, 0 mesh copies"

---

## gallery

*(3 figures. Each: caption (shown) + alt text (for screen readers).)*

**Figure 1 — mesh_scope**
Caption: A small MLP learns a fixed 2-D target surface live — every optimizer
step writes its 72×72 prediction into imported device tensors, drawn the same
frame as Lambert-lit triangles colored by per-vertex error, with a wireframe
overlay and the training minibatch as points.
Alt text: Screenshot of a lit 3-D mesh whose height and color encode a neural
network's live prediction of a target surface, wireframe edges and scattered
sample points visible over it.

**Figure 2 — instance_scope**
Caption: A field of N procedural gems bobs and spins in a traveling wave,
drawn with one instanced draw call and zero copies of the mesh — per-frame
`(N,16)` poses and a `(N,)` phase tint are recomputed on device and imported
zero-copy.
Alt text: Screenshot of a grid of identical faceted gems rising and falling in
a wave pattern, each tinted along a magma color ramp by its phase, rendered in
a single instanced draw.

**Figure 3 — repnet_demo Training Lab**
Caption: Live native-C++ training of an ECG model, with per-step convolution
kernels and signal-plus-saliency overlays updating on the frame the GPU
computes them — the extracted origin of the platform's metrics, jobs, and
tensor-bridge services.
Alt text: Screenshot of a training dashboard showing an ECG waveform with a
saliency overlay, a grid of learned convolution kernels, and live loss and
metric curves.

---

## links

*(Exact hrefs. Task 4 wires these verbatim.)*

- **View on GitHub** → `https://github.com/AhmedKhan-GH/caliper`
- **Read the whitepaper** → `https://github.com/AhmedKhan-GH/caliper/blob/main/WHITEPAPER.md`
- **Reference docs** → `https://github.com/AhmedKhan-GH/caliper/tree/main/docs/wiki`
- **Contact** → `mailto:emailahmedebadkhan@gmail.com`
