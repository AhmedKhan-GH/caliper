# Training Lab — design spec

Bring the best PerLeadCNN training regimen from the `repnet` project into the
caliper `repnet_demo` applet as a live **Training Lab** tab: train PerLeadCNN
from random init in native C++ libtorch on the reproducible best split (seed
1119 / split 17), and watch the conv kernels adapt to ECG waveforms and grad-cam
saliency build from noise in real time.

## Decisions (locked)
- **Option A**: identical data + live C++ training from noise. Live weights are
  *close, not bit-identical* to Python (C++ torch RNG ≠ Python). The shipped
  `best_model.pt` is the converged ground-truth reference.
- **C++ owns the DSP** preprocessing (baked scipy SOS coefficients).
- **C++ owns the split**: full `StratifiedGroupKFold` + numpy `RandomState`
  (MT19937) port, gated by a golden-file test against seeds 1119/1120.
- New tab inside `repnet_demo` (not a new applet). CPU-only training.
- Keep a ghost overlay of reference kernels.

## Reproduction recipe (from repnet `reproduce.ipynb` + `export/code/train.py`)
- Outer split: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_i*7+1000)`,
  groups = `Pat_Obfus_MRN`. First fold = **test**. Best model = split_i=17 → seed **1119**.
- Inner split on dev set: `StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=1120)`.
  First fold = **val**, rest = **train**.
- Model init seed = **1121** (`split_seed+2`).

### Model — PerLeadCNN (29,490 params)
- filters `(16,32,48)`, kernels `(31,21,11)`, dropout `0.15`, n_classes 2.
- Backbone = 3× `[Conv1d(in,f,k,stride=2,padding=k//2,bias=False), BatchNorm1d(f), Mish]`, in_ch starts at 1.
- `forward`: x `(B,12,T)` → reshape `(B*12,1,T)` → backbone → `AdaptiveAvgPool1d(1)` →
  `(B*12,48)` → reshape `(B, 12*48=576)` → `Dropout` → `Linear(576,2)`.
- **Module names must match Python** (`backbone`, `pool`, `head_drop`, `fc`) so
  `best_model.pt` state_dict loads directly.

### DSP (matches repnet `prepare.py` / notebook `preprocess`)
Per recording, raw leads in order `[I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6]` (read
by **column name** from CSV header), shape `(12, T_raw)`:
1. If `T_raw==5000`: keep. If `T_raw==2500`: `scipy.signal.resample` to 5000 (upsampled). Else drop.
2. 4th-order Butterworth **high-pass 0.5 Hz** @ fs=500, `sosfiltfilt` (zero-phase).
3. **60 Hz notch** (Q=30) @ fs=500 via `iirnotch`→`tf2sos`, `sosfiltfilt`.
4. Per-lead z-norm: `(x - mean)/(std+1e-8)` over time axis.
5. Downsample `[:, ::2]` → 2500 samples @ 250 Hz.
Drop recordings with non-finite values or any lead std `< 1e-4` (flatline), or
missing/blank `Pat_Obfus_MRN`. Label positive iff
`PatLabel == "Preeclampsia or Other Hypertensive Disorders of Pregnancy"`.
SOS coefficients are **baked constants** (computed once in Python, asserted in test).

### Training regimen (from `train.py`)
- AdamW lr `1.2e-3`, weight_decay `5e-3`. CosineAnnealingLR T_max=80, eta_min=1e-6.
- batch 64, grad-accum 2 steps. MAX_EPOCHS 80, early-stop patience 20 on **val AUROC**.
- Loss: FocalLoss γ=1.0, label_smoothing 0.05, class_weight None.
- Mixup α=0.2 applied to 50% of batches: `lam~Beta(α,α)`, mix x and loss.
- Best checkpoint = max val AUROC over epochs.
- 7 on-the-fly augmentations (train only), per `AUG_CFG`:
  noise (p .5, σ∈[.01,.05]), amp-scale (p .5, ±.15 per lead), time-shift (p .3, ±150),
  lead-drop (p .10, drop prob .12), cutout (p .25, len∈[50,200]), baseline wander
  (p .2, amp∈[0,.2], freq∈[.05,.5]/fs), resample (p .10, rate ±.05). fs=500 in wander term.

## Components (new files under `applets/repnet_demo/`)
| Unit | Files | Test |
|---|---|---|
| DSP | `train/dsp.{h,cpp}` | `tests/test_dsp.cpp` vs golden |
| Split | `train/sgkf.{h,cpp}` | `tests/test_sgkf.cpp` vs seeds 1119/1120 |
| Model | `train/perlead_cnn.{h,cpp}` | `tests/test_perlead.cpp` (params, fwd shape, load best_model) |
| Augment | `train/ecg_augment.{h,cpp}` | `tests/test_augment.cpp` (shape/determinism) |
| Data | `train/train_dataset.{h,cpp}` | covered via dsp+sgkf goldens |
| Engine | `train/train_engine.{h,cpp}` | `tests/test_engine.cpp` (loss-decrease smoke) |
| UI/viz | `train/training_lab_tab.{h,cpp}` | manual run-verify |

### Threading
TrainEngine owns a background `std::thread`. After each epoch it publishes a
double-buffered `TrainSnapshot { epoch, train_loss, val_auroc, best_epoch,
stage1_kernels (16×31), pinned-sample stage activations, grad-cam (12×T),
metric history }` under a mutex. GL `draw_ui` copies the latest snapshot and
renders; it never blocks training. Mirrors the existing `DatasetLoader` worker.

### Live visualization (the deliverable)
1. **Kernels adapting** — 16 stage-1 filters (len 31) as waveforms morphing noise→QRS detectors; ghost overlay of reference kernels.
2. **Saliency from noise** — grad-cam over a pinned held-out positive, flat→focused across epochs (reuse `compute_grad_cam`).
3. **Feature maps** — pinned-sample stage activations.
4. **Metrics** — live train-loss + val-AUROC curves, best-epoch marker, early-stop countdown, reference-AUROC target line (split-17 = 0.7793).

Controls: Start / Pause / Step / Reset, seed (default 1119), speed throttle,
load reference weights, snap-kernels-to-reference.

## Reproducibility gate (correctness)
- `sgkf` test: C++ fold membership **bit-identical** to Python for seeds 1119 & 1120 (golden id lists).
- `dsp` test: C++ output matches scipy preprocess to < 1e-4 on sample ECGs (golden `.bin`).
- `perlead` test: loads `best_model.pt`; forward parity to golden logits < 1e-4; reproduces split-17 test AUROC 0.7793 on C++-preprocessed test set (end-to-end check).
- Live-trained weights need not bit-match Python.

## TDD + incremental commits
Per global rule: failing golden test first, then implementation; one commit per green unit.
Order: (0) Python export of golden artifacts + C++ test target → (1) dsp → (2) sgkf →
(3) perlead_cnn → (4) ecg_augment → (5) train_dataset+engine → (6) training_lab_tab → (7) wire tab.
Deterministic units are test-gated; GL/UI verified by running the applet.

## Golden artifacts (exported by `scripts/export_training_lab_goldens.py` in repnet → copied to `applets/repnet_demo/tests/golden/`)
- `sos_coeffs.json` — baked HP + notch SOS arrays.
- `dsp_cases/<id>_raw.bin`, `<id>_pre.bin` — a few ECGs raw (12×T) and preprocessed (12×2500), float32 row-major + shape json.
- `sgkf_case.json` — `y` (int), `groups` (int-encoded), and expected test-fold index lists for 5-fold@1119 and 8-fold@1120.
- `perlead_fwd.bin` — fixed input (1×12×2500) + golden logits (1×2).
- `best_model.pt` — copied from `multisplit_dbb6f49/`.
- `meta.json` — dataset dir path, n_samples, positive count, split-17 AUROC 0.7793.

Data dir at runtime: `/Users/ahmed/PycharmProjects/repnet/data/seniordesign_upload`
(user-selectable; same dir the existing dataset browser opens).
