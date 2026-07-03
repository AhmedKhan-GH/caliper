# artifacts.v1 — what it is, and what I'd want it to be

> A design note to revisit. Not a spec change. Captures the current state of
> `caliper.artifacts.v1`, an honest read on its value, and the direction I'd
> take it *if* you decide it earns its place. Written 2026-07-03, after the
> Runs dashboard was removed and the project settled on the "self-contained
> workbench, not a platform" direction.

## 1. What it is today

A frozen host service for **content-addressed checkpoint storage**. The ABI
(`sdk/include/caliper/services/artifacts_v1.h`, immutable):

```c
bool        put(const char* name, const void* bytes, uint64_t len,
                uint64_t run, char out_digest[65]);   // -> sha256 hex
const char* path_of(const char* digest_or_name);      // name -> newest
bool        exists(const char* digest_or_name);
```

Backed by `ArtifactStore` (`src/host/artifact_store.{h,cpp}`): blobs written
as files at `<data>/artifacts/<sha256>`, a DuckDB index of
`(digest, name, run, len, ts)`. Content-addressing gives three things for
free — **dedup** (identical bytes stored once), **naming** (save/reload by a
friendly name, newest wins), and **lineage** (the `run` links a checkpoint to
the `metrics.v1` run that produced it).

**Only consumer today:** EmbedScope's and GPTScope's Save/Load buttons.
Save = `torch::save` to a byte buffer → `put`. Load = `path_of` →
`torch::load` → one eval pass, no training. The visible payoff is
*save → quit → relaunch → load → the model is back without retraining*.

## 2. The honest read

- It is **real and reachable** — unlike the Runs dashboard (a dead viewer),
  artifacts is wired to buttons you press and observe. That's the bar a
  feature has to clear.
- But it is **thin**. Right now it's barely more than "save a file with a
  good name." The content-addressing, dedup, and lineage are architecturally
  nice and cost nothing, but none of them *visibly help you yet* — you have a
  handful of checkpoints, not a collection where dedup or lineage matters.
- The lineage link (`run`) now points at run ids with **no viewer** (the Runs
  dashboard is gone). It's honest bookkeeping, but currently unsurfaced.

Verdict: keep it — it's cheap, correct, and the Save/Load round-trip is a
genuinely good demo moment — but it is not yet *pulling its weight as an
idea*. The question for the revisit is whether to **make it earn its keep**
or **leave it minimal**.

## 3. What I'd want it to be

The workbench framing (audience of one, live experimentation) suggests a
**checkpoint shelf**, not an MLOps artifact registry. Concretely, in rough
priority:

1. **A visible artifact browser** (host-side, the good version of what the
   Runs tab should have been): a small panel listing saved checkpoints with
   name, size, timestamp, and — the thing that makes it worth a screen — the
   ability to **load any of them back into the applet that owns them**. This
   is the missing viewer. It's what turns "I saved a thing somewhere" into "I
   can see and manage my trained models."
2. **Thumbnails / provenance at a glance.** A checkpoint is opaque bytes; a
   good shelf shows *what it is*: the final loss, the step count, a tiny
   preview (EmbedScope's cloud, GPTScope's sample text) captured at save time.
   Store a small JSON/preview blob alongside via a second `put`. Answers "which
   of these seven is the good one" — the exact failure the Runs dashboard died
   of.
3. **"Resume training from here."** Load currently runs eval only. The richer
   move: load weights *and optimizer state* and continue training. Turns a
   checkpoint from a snapshot into a branch point — genuinely useful for a
   live workbench ("this run was going well, let me fork it").
4. **Auto-checkpoint on good.** Optional: an applet asks the store to keep the
   best-N by a metric, pruning the rest (dedup already makes this cheap). Now
   lineage matters — "best checkpoint, and the run that made it."

What I would **not** do (platform-scale scope the workbench doesn't need):
remote/sync backends, signing, a registry, cross-machine artifact sharing.
Those are the Ring-2/Phase-5 ideas; they don't belong in a solo tool.

## 4. Open questions for the revisit

- **Do you actually reload models?** If in practice you always retrain from
  scratch (it's seconds on these toy models), the whole service is ceremony.
  The test: after a week of use, have you ever pressed Load and been glad it
  was there? If no → demote artifacts the way Runs was demoted.
- **If yes → build the browser (§3.1).** That's the one addition that would
  make artifacts feel like a feature instead of a file-save. ~a focused
  session; host-side, no ABI change (the frozen `put/path_of/exists` already
  supports everexactly what a browser needs to read).
- **Metrics coupling.** artifacts' lineage points at `metrics.v1` runs. If you
  later strip metrics persistence (offered after the Runs removal), the `run`
  argument becomes vestigial — decide whether lineage is worth keeping metrics
  alive for, or whether artifacts should stand fully alone.

## 5. One-line position

Keep it, because Save/Load is a real and good moment — but it's currently a
file-save wearing a service's clothes; the single change that would justify
the clothes is a **host-side checkpoint browser with previews**, and that's
worth doing only if you find yourself actually reloading models. Until then,
minimal is honest.
