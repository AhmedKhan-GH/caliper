# ML applet cookbook

The composition idioms for **featurized, live ML demonstrations** — the
patterns that make an applet feel alive rather than merely instrumented.
Every one of them is embodied in the exemplar (`applets/embed_scope/`); this
page names them, shows the code shape, and explains *why* each is shaped
that way. Contracts live in the per-service reference pages; this is the
page about putting them together. Snippets are condensed from the exemplar —
open it beside this page.

## 1. The threading spine

One training **worker** (a `jobs.v1` job), one **frame thread** (your
`draw_ui`). Everything they share lives under **one mutex**, published with
**generation counters**:

```cpp
struct State {
    std::mutex mtx;
    // -- cross-thread, guarded by mtx --
    std::vector<float> loss_hist;
    std::vector<float> ex, ey, ez;     // published embedding coords
    uint64_t gen = 0;                  // bumped per publish (0 = none yet)
    // -- frame-thread-only --
    uint64_t seen_gen = 0;             // last gen the UI consumed
};

// WORKER: compute outside the lock, swap inside it, bump the generation.
void publish(State* st, std::vector<float> x, std::vector<float> y,
             std::vector<float> z) {
    std::lock_guard<std::mutex> lk(st->mtx);
    st->ex = std::move(x); st->ey = std::move(y); st->ez = std::move(z);
    st->gen++;
}

// FRAME: copy out under the lock, render the copies.
void draw_ui(State* st) {
    std::vector<float> ex, ey, ez; uint64_t gen;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        gen = st->gen;
        if (gen != st->seen_gen) { ex = st->ex; ey = st->ey; ez = st->ez; }
    }
    // ... plot ex/ey/ez; expensive derived work only when gen moved ...
}
```

- The worker publishes **owned copies** (`std::vector`) for plot data, or
  **tensor handles** for device-resident display tensors (§3) — never raw
  pointers into transient worker state.
- The frame thread is the ONLY thread that touches `tensor_bridge.v1` and
  `data.v1`. All torch *operations* stay on the worker; the frame thread may
  hold and pass tensor handles but never launches kernels.
- Generation counters make every downstream stage cheap: nothing re-uploads,
  re-splits, or re-queries unless its `gen` moved.

## 2. Three cadences — match update rate to what the data is

A rich demo has streams with different natural rates. Name them explicitly
in the step loop:

```cpp
for (int64_t b = 0; b < n; b += kBatch) {
    if (ctl->cancelled(ctl)) return;                    // §7
    /* ... forward, backward, opt.step() ... */
    step++;

    publish_live(st, model, xb);                        // EVERY step: batch
    publish_cloud(st, model, Xte, yte);                 // EVERY step: cloud
                                                        //   (see the note)
    if (step % kEvalEvery == 0) {                       // every 50: metrics
        float acc = evaluate();
        st->metrics.scalar(run, "test/accuracy", step, acc);
        model->train();
    }
}
```

| Stream | Cadence | Why |
|---|---|---|
| Live layer: current batch, weight tensors, cloud | **every optimizer step** | weights only change when the optimizer steps — per-step IS every change |
| Derived aggregates: SQL centroids, misclassified | **throttled (~2 Hz)**, on the frame side (§10) | rebuilt on the *frame thread*; at step rate it would stutter the UI |
| Quality metrics → `metrics.v1` | **every `kEvalEvery` steps** | metrics want statistical stability, not liveness; they persist to the Runs dashboard |

Don't add a config knob for cadence unless a *measured* cost forces one.
(The exemplar once had a "cloud refresh" slider defending against a 6×
slowdown that turned out to be ~nothing: step time was dominated by GPU sync
overhead, not compute. The knob died.)

## 3. The device-resident pull (the USP pattern)

The framework's reason to exist: tensors go from training memory to pixels
**without CPU staging**. Worker side — build *display tensors* on the
training device and hand over **handles** (refcount bumps, no data moves):

```cpp
// WORKER, per step (inside NoGradGuard):
auto wc = model->conv1->weight.detach();            // (8,1,3,3), on-device
const float km = wc.abs().max().item<float>();      // colormap range
torch::Tensor dc[8];
for (int k = 0; k < 8; k++)                         // §4: sharp blocks
    dc[k] = wc[k][0].repeat_interleave(16, 0)
                    .repeat_interleave(16, 1).contiguous();   // (48,48)
{
    std::lock_guard<std::mutex> lk(st->mtx);
    for (int k = 0; k < 8; k++) st->disp_conv[k] = dc[k];    // handle swap
    st->w_km = std::max(km, 1e-6f);
    st->live_gen++;
}
```

Frame side — upload the **latest** state at most once per frame:

```cpp
// FRAME, when live_gen advanced past tex_gen:
for (int k = 0; k < 8; k++) {
    if (st->conv_tex[k]) st->bridge.release_texture(st->conv_tex[k]);
    auto ct = caliper::adapters::to_tensor(disp_conv[k]);    // no torch ops!
    st->conv_tex[k] = ct ? st->bridge.texture_from_tensor_mapped(
                               &*ct, CALIPER_CMAP_MAGMA, -w_km, w_km) : 0;
}
st->tex_gen = lgen;
// draw: ImGui::Image(caliper::Bridge::imtex(st->conv_tex[k]), {48, 48});
```

On Metal the bridge colormaps **on-GPU** — the weights never visit the CPU.
**Decoupling is the point:** if steps outpace frames you render 60 fps of
*latest* states; if frames outpace steps, identical memory renders identical
pixels. Display never throttles training, training never blocks display.

Rules that bite: MPS tensors must be `contiguous()` with
`storage_offset() == 0` (fresh results of GPU ops are; views may not be —
see [adapters](../reference/adapters.md)). And degrade gracefully — if the
bridge rejects device tensors (GL fallback), flip an atomic so the worker
hands over CPU tensors from the next step:

```cpp
if (any_texture_failed && disp.device().type() != torch::kCPU)
    st->disp_force_cpu.store(true);      // worker adds .to(kCPU) next step
```

## 4. Sharp tiny tensors: block-upscale on device

Bridge textures sample linearly (v1 has no filter flag), so a 3×3 kernel
drawn at 40 px is interpolated mush. Fix it where the data lives — on the
GPU — then draw at 1:1 or integer multiples:

```cpp
auto sharp = w.repeat_interleave(16, 0)      // 3x3 -> 48x48 of HARD blocks
              .repeat_interleave(16, 1).contiguous();
// later: ImGui::Image(tex, ImVec2(48, 48));  // 1:1 — no resampling blur
```

Sharp under any sampler, zero ABI involvement, stays device-resident.

## 5. Texture lifecycle

Bridge textures are **frame-thread-owned**: create/update/release only in
`draw_ui` and `cleanup()`. Gate rebuilds on a gen counter; release-then-
create when a mapped texture's value range evolves (the range is baked at
creation). Cleanup order matters:

```cpp
void cleanup() {
    st->jobs.request_cancel(st->job_id);            // 1. stop the worker
    for (int i = 0; i < 1000 && st->jobs.is_running(st->job_id); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (auto& tx : st->conv_tex)                   // 2. THEN release textures
        if (tx) { st->bridge.release_texture(tx); tx = 0; }
    curl_global_cleanup();                          // 3. pairs with on_init
}
```

Miss a release and the leak is silent; release before the join and you race
the worker.

## 6. Viewport policy: who owns the camera

Three plot situations, three correct answers — and every "why is my plot
fighting me" bug is one of these mismatched:

| Data behavior | Correct axes | Exemplar |
|---|---|---|
| Points **move** through space | **Fixed**: fit once, hold still — motion is only visible against a static frame | the 3-D Cloud |
| Series **grows** (loss, accuracy) | **Following**: AutoFit traces the curve — but AutoFit **input-locks** the axis, so make it a toggle | the Training curves |
| Static content (snapshot, matrix) | Default fit; double-click re-fits (native gesture — leave it enabled) | the Tensors heatmaps |

Fixed axes for moving data (3-D):

```cpp
ImPlot3D::SetupAxesLimits(bmin[0], bmax[0], bmin[1], bmax[1],
                          bmin[2], bmax[2],
                          refit ? ImPlot3DCond_Always    // first publish,
                                : ImPlot3DCond_Once);    // Refit button,
refit = false;                                           // or auto-fit ON
```

Following-with-consent for growing data (2-D):

```cpp
ImGui::Checkbox("follow", &st->follow_curves);   // the auto-scroll idiom
const ImPlotAxisFlags f = st->follow_curves ? ImPlotAxisFlags_AutoFit : 0;
if (ImPlot::BeginPlot("train loss", {-1, 150})) {
    ImPlot::SetupAxes("step", "NLL", f, f);      // unchecked: free zoom/pan
    ImPlot::PlotLine("loss", loss.data(), (int)loss.size());
    ImPlot::EndPlot();
}
```

One visible toggle per plot, sensible motion by default, double-click resets.

## 7. The cancel contract, in practice

`jobs.v1` promises cancel is honored **≤ 100 ms**. That means a check in
every loop that can run longer than that:

```cpp
for (int64_t b = 0; b < n; b += kBatch) {
    if (ctl->cancelled(ctl)) return;             // per training batch
    ...
}
for (int64_t b = 0; b < seen; b += 1000) {
    if (ctl->cancelled(ctl)) return std::nullopt;  // per EVAL batch too
    ...
}
// and during downloads, via curl's progress callback:
int xferinfo(void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* x = static_cast<XferCtx*>(p);
    return x->ctl->cancelled(x->ctl) ? 1 : 0;    // nonzero aborts transfer
}
```

Your `cleanup()` then does cancel → bounded wait → release (§5). The host
additionally joins all workers before teardown, so sloppiness degrades
instead of crashing — but honor the contract; the bounded wait is what
keeps app exit fast.

## 8. Data acquisition (the download recipe)

Datasets are fetched **inside the job** (never the frame thread), cached in
`host.data_dir()`, written atomically so a crash mid-download can't poison
the cache:

```cpp
// download to a sibling .tmp, rename into place (same-filesystem = atomic)
std::ofstream out(path + ".tmp", std::ios::binary);
out.write((const char*)bytes.data(), bytes.size());
out.close();
std::filesystem::rename(path + ".tmp", path);

// and on load: parse failure = corrupt cache -> delete + report, next Train
// re-downloads (self-heal, never a permanent wedge)
```

Pair `curl_global_init` (in `on_init`) with `curl_global_cleanup` (in
`cleanup`, after the worker join). Prefer mirrors that tolerate automation —
the exemplar uses the S3 MNIST mirror; the classic host 403s.

## 9. Checkpoints via `artifacts.v1`

Save on the frame thread — serialize to bytes, `put` with the run id for
free lineage:

```cpp
std::ostringstream oss(std::ios::binary);
st->model->to(torch::kCPU);
torch::save(st->model, oss);
std::string bytes = oss.str();
std::string digest = st->artifacts.put("embedscope-model", bytes.data(),
                                       bytes.size(), st->run_id.load());
```

Load: resolve the path on the frame thread (host strings are
valid-until-next-call), hand it to a job that loads and runs **one eval
pass** — restoring the visuals *without training* is the demo magic:

```cpp
const char* p = st->artifacts.path_of("embedscope-model");  // frame thread
if (p) { st->load_path = p;
         st->job_id = st->jobs.submit("load checkpoint", &eval_job, st); }
```

## 10. SQL over live data (`data.v1`)

Rebuild the table from the latest snapshot, ask real questions, drain with
the helper. Frame-thread only, throttled (§2):

```cpp
// DDL/INSERT still hand back an (empty) stream — release it:
auto exec = [&](const std::string& sql) {
    ArrowArrayStream s{};
    if (!st->data.query(sql.c_str(), &s)) return false;
    if (s.release) s.release(&s);
    return true;
};
exec("CREATE OR REPLACE TABLE embed_points(label INT, pred INT, "
     "x REAL, y REAL, z REAL)");
exec(batched_insert_sql);                        // one VALUES list per publish

ArrowArrayStream cs{};
std::vector<std::vector<double>> cols;
if (st->data.query("SELECT label, AVG(x), AVG(y), AVG(z) "
                   "FROM embed_points GROUP BY label", &cs))
    caliper::Data::drain_numeric(&cs, nullptr, &cols);   // releases cs
```

Throttle the rebuild on the frame side (`ImGui::GetTime()` gate, ~2 Hz) so
the plot never waits on SQL. If the service is absent, say so in the panel
and keep running — every optional service degrades to a visible
"absent (ok)" line, never a crash.

---

**The one-sentence summary:** worker computes and publishes generations;
frame pulls the latest and renders; every stream updates at the rate its
nature dictates; the camera has an owner; and everything optional degrades
visibly instead of failing silently. That's a Caliper ML demo.
