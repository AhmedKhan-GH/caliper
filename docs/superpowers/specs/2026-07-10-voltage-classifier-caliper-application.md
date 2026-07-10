# Applying Caliper to a spatial voltage classifier

**Date:** 2026-07-10
**Status:** application note (not an implementation spec). Maps an external
research use case onto Caliper's existing capabilities and exemplars. No ABI
growth implied; every claim below rides on shipped services or the
`geometry.v1_2` branch in flight.
**Audience:** the researcher (Ahmed) deciding *whether and where* Caliper
earns its cost on this problem, before writing an applet.

## 0. The use case, stated plainly

Input data:
- **Timeseries of voltage propagating over a spatial substrate**, per
  recording.
- A **binary class label** per recording.
- **Metadata about sourcing** (site / device / subject / acquisition —
  whatever distinguishes where each recording came from).

Research goals, in the researcher's words:
1. **Featurize** the recordings.
2. **Classify** into the two classes.
3. **Map the data to physical space** (the substrate geometry).
4. Use all of the above to **improve the model** and **improve insight
   around the model**.

## 1. The frame — what Caliper is and is not for this problem

Caliper is an **instrument**, not a modeling framework. Its invariant is
`tensors → pixels → ImGui`, one way: it draws whatever a tensor holds, the
same frame that tensor is computed, zero-copy, with no round trip to the CPU
and back. It has no opinion about where the tensor came from.

Consequences for this use case:

- **Caliper does not classify or featurize.** You still write the torch model
  — the encoder, the feature representation, the classifier head — exactly as
  you would without Caliper.
- **Caliper does not simulate.** It is not a physics engine (and building one
  would be off-mission; the project's own honesty line forbids claiming FEA).
  If a field is displayed, that field was *supplied* — by replaying measured
  data or by a small worker step — not computed by Caliper.
- **What Caliper adds is the live insight layer wrapped around your model,**
  and uniquely: the ability to draw the model's internal state **on your
  physical geometry**, the frame it is computed, sliced by your metadata.

So the operative question is not "can Caliper do my ML" but **"which parts of
my model's behavior become visible if I can watch them live, on the geometry,
during training."** The rest of this doc answers that stage by stage.

## 2. Stage-by-stage mapping

| Your stage | What Caliper draws | Shipped precedent to copy from |
|---|---|---|
| **Map data → physical space** | Raw voltage timeseries draped on the mesh as a texture; scrub / play through time. TwinScope minus the physics — replay instead of simulate. | TwinScope v2 (R2 textures-on-meshes, `geometry.v1_2` branch) |
| **Featurization** | The learned embedding / feature space, live, as it trains — watch the two classes pull apart (or fail to). Encoder filters / activation maps drawn as they change. | EmbedScope (3-D embedding bottleneck), GPTScope (embedding cloud) |
| **Classification** | Live per-class centroids, misclassified examples, confidence, decision structure — via SQL over a published embedding table. Every view **sliceable by sourcing metadata**. | EmbedScope (live SQL centroids + misclassified-count over the embedding table via `data.v1`) |

None of these require new ABI. Each is a re-skin of an applet that already
exists and runs end-to-end.

## 3. The differentiated use — attribution on the geometry

This is the reason Caliper is worth it here rather than a notebook.

You have **both a spatial signal and a classifier.** That combination lets
Caliper draw the normally-invisible thing: **where on the surface the model
looks to make its decision.** Model attribution — input-gradient saliency,
attention weight, or occlusion sensitivity — is itself a per-vertex /
per-texel scalar field. It draws on the geometry the *same way* the raw
voltage does, and it can be watched **sharpening as the model trains**:

> "The classifier calls this recording class-1 because of voltage behavior in
> *this region* of the surface, at *this phase* of propagation."

That is a research instrument, not a plot. Standard tooling
(TensorBoard / W&B / matplotlib) does not map attribution back onto real 3-D
geometry live during training. This is the project's "make the data *be* the
picture" wedge applied directly to this problem.

## 4. The confound hunt (metadata as a first-class axis)

Sourcing metadata is usually a landmine in this kind of data: the model
learns the **site / device / subject artifact** instead of the real signal
(batch effect). Caliper turns the metadata into a live diagnostic on two
fronts:

- **In embedding space:** `data.v1` SQL-slices the live embedding by source.
  If the two classes separate cleanly but *also* separate by source, the
  confound is visible in real time — not discovered after publication.
- **On the geometry:** if attribution parks on a region explained by
  sensor placement / acquisition rather than substrate behavior, the model is
  keying on an artifact. You see the failure spatially.

This is the strongest single argument for using Caliper on this project:
it makes the failure mode that most threatens the science *observable while
training*, in the two representations where it hides.

## 5. Where NOT to use Caliper (honest boundary)

Keep these in pandas / sklearn / W&B — Caliper adds nothing:

- The featurization math itself (filtering, spectral / wavelet features,
  graph construction).
- Final scalar metrics — ROC / AUC, confusion matrix, cross-validation
  scores. A logged number does not need a zero-copy GPU instrument.
- Hyperparameter sweeps and cross-run comparison. That is W&B's job.

Caliper earns its cost only where **(a)** there is spatial structure to put on
geometry, or **(b)** you want to watch high-frequency internal state evolve
live. This use case has both — but only in the featurization / attribution /
spatial-replay parts, not the bookkeeping.

## 6. What Caliper does and does not improve

- **Model accuracy — indirectly, not directly.** Caliper draws no gradients
  you did not compute and trains no weights. It improves accuracy only by
  making problems *visible* sooner: a confound caught in week one, a feature
  space that never separates, attribution on the wrong region. You act on
  what you see; the tool does not act for you.
- **Insight around the model — directly, and this is the point.** Why the
  model works, where on the surface it looks, at what phase of propagation, and
  whether it is cheating on source artifacts. For a spatial classification
  problem with a physical substrate, that insight layer is the gap between
  "the number went up" and "I understand what it learned" — the part standard
  tooling leaves blind.

## 7. Recommended build order (payoff-first)

Write the model in torch as normal. Then adopt Caliper in three steps, each
shippable on its own:

1. **Voltage-on-mesh replay** *(lowest effort, immediate value).* Swap
   TwinScope's heatsink OBJ for the substrate geometry; feed the recorded
   timeseries instead of a heat-sim worker. Spatial data exploration falls out
   of the `geometry.v1_2` branch for free. Nothing modeled yet — this is pure
   data-on-geometry.
2. **Live embedding separation, colored by class *and* by source.** EmbedScope
   is the template; it already does exactly this over a live SQL embedding
   table for MNIST. Point it at your encoder and your metadata columns.
3. **Attribution draped on the geometry, sharpening during training** *(the
   differentiated instrument).* The one part worth a custom applet. Attribution
   is a scalar field; it rides the same R2 texture path as the raw voltage.

## 8. Open questions to resolve before an applet hardens

These are decisions this doc surfaces but does not make:

- **Surface vs. volume.** Caliper's geometry ladder is 2-manifold
  (points → meshes → textures-on-surfaces → instancing). If **surface
  mapping** is scientifically sufficient, you are on a straight path. If
  you need **volumetric** propagation through the substrate interior,
  there is no volumetric primitive on the roadmap — that would be a new rung,
  not a swap. Resolve this first; it decides straight-line vs. fork.
- **Data ingestion path.** Replaying recorded voltage needs a way to get the
  timeseries into a tensor the applet publishes. For static datasets this is a
  loader + `data.v1`. For *live external feeds* (streaming acquisition), the
  telemetry-ingestion decision is explicitly open (ROADMAP §7, STATUS §5:
  feed-applet pattern vs. new service). Static replay does not need it; live
  feed does.
- **Attribution method.** Which attribution (input-gradient / attention /
  occlusion) is both meaningful for your model class and cheap enough to
  compute per-frame without stalling the frame thread. Torch stays off the
  frame thread (frame-thread discipline is load-bearing across the codebase);
  attribution is computed on the jobs worker and published like any field.

## 9. One-line summary

Caliper will not classify your recordings or raise your AUC on its own; it is
the live instrument that draws your model's feature space and its
**attribution on the actual substrate geometry** as it trains — turning the
confound that most threatens this science (source artifact vs. real signal)
into something you watch happen instead of discover afterward.
