# Applets

Applets live in `applets/` and build with the `release` preset. Every applet links LibTorch, so the `release-notorch` preset builds none of them. Launch one from Instruments on the landing screen (⌘2, then Enter), or start the app with `CALIPER_AUTOLAUNCH=<id>`.

Each description below is the `summary` from the applet's `.caliper.toml` manifest.

## GPTScope

`dev.caliper.gpt-scope` · 0.2.0 · `applets/gpt_scope/`

Mechanistic interpretability for a live-training char GPT (4L/4H/128d, TinyShakespeare): the logit lens across depth, head roles as a distance/entropy scatter with on-demand attention drill-down, W_E embedding geometry as a glyph cloud, attn-vs-MLP residual write accounting with per-group gradient norms, and confidence-colored live sampling. Trains off the frame thread via caliper.jobs.v1, device-negotiated; metrics.v1 streams the run; tensor_bridge.v1 draws the head heatmap; artifacts.v1 saves/loads the model.

## EmbedScope

`dev.caliper.embed-scope` · 0.1.0 · `applets/embed_scope/`

3D embedding projector: a small MNIST net with a learned 3-D bottleneck whose test-set embeddings are drawn as a live ImPlot3D scatter (10 classes) that splits blob->lobes while training runs off the frame thread via caliper.jobs.v1. Hover a point for its digit via tensor_bridge.v1; Save/Load the model via artifacts.v1 (Load skips training); class centroids and misclassified counts via SQL over caliper.data.v1.

## MeshScope

`dev.caliper.mesh-scope` · 0.1.0 · `applets/mesh_scope/`

A small MLP learns a fixed 2-D target surface live: every optimizer step its 72x72-grid prediction is written into imported device tensors and drawn the same frame as Lambert-lit indexed triangles colored by per-vertex squared error through the MAGMA LUT, with a wireframe overlay and the training minibatch as additive points — the caliper.geometry.v1_1 exemplar. Falls back to an input-locked CPU error heatmap wherever the zero-copy path is absent.

## TwinScope

`dev.caliper.twin-scope` · 0.2.0 · `applets/twin_scope/`

A surface-aware synthetic heat twin on a UV-mapped finned housing: a batched cotangent-Laplacian sim flows heat over the 3-D surface while a small net learns it live. The hero splits physics|belief side by side, drawn zero-copy at texture resolution, with a live textured (v1.2) vs per-vertex (v1.1) toggle exposing the R2 resolution gap, plus honest per-vertex and CPU heatmap fallbacks.

## InstanceScope

`dev.caliper.instance-scope` · 0.1.0 · `applets/instance_scope/`

Instancing, made obvious: a field of N procedural gems bobs and spins in a traveling wave, drawn with ONE instanced draw call and ZERO copies of the mesh. Per-frame rigid (N,16) poses and a (N,) sin-phase tint are recomputed on device and imported zero-copy (caliper.geometry.v1_3); the tint rides a fixed [-1,1] MAGMA window that can never saturate. Falls back to a single non-instanced gem where the instanced path is absent.

## FlowScope

`dev.caliper.flow-scope` · 0.1.0 · `applets/flow_scope/`

A million-particle flow field, drawn with zero copies: the torch CUDA simulation writes exportable-pool tensors and the renderer's instanced point pass (caliper.geometry.v1) reads the SAME memory in place, per frame. Left-drag pushes the field with a cursor-ray impulse; right-drag orbits. Falls back to a subsampled CPU scatter wherever the zero-copy path is absent.

## FieldScope

`dev.caliper.field-scope` · 0.1.0 · `applets/field_scope/`

A charged-particle field simulator, drawn with zero copies: the torch simulation writes exportable-pool tensors and the renderer's instanced point pass (caliper.geometry.v1) reads the SAME memory in place, per frame. A self-consistent electrostatic Particle-In-Cell plasma mode (particles deposit charge to a grid, an FFT Poisson solve gives their own field, a Boris pusher advances them) shows magnetized vortices orbit and merge; or switch to a prescribed field geometry to observe charged-particle motion directly — uniform (gyration), magnetic mirror / solenoid bottle (mirroring), toroidal tokamak (confined orbits), or magnetic dipole (Earth-like trapped belts). Left-drag perturbs, right-drag orbits, wheel zooms. Falls back to a subsampled CPU scatter wherever zero-copy is absent.

## SculptScope

`dev.caliper.sculpt-scope` · 0.1.0 · `applets/sculpt_scope/`

A small libtorch MLP maps N latent codes to N 3-D points and trains live (Adam, real backprop) to match a target shape. Its forward output is written straight into the pool-born tensor the renderer's instanced point pass (caliper.geometry.v1) reads in place — the buffer the final Linear layer writes IS the buffer the GPU draws, with zero copies between the ML and the picture. Watch a formless blob flow into a crisp shape as the network learns; color tracks per-point motion so a converged cloud dims. Right-drag orbits, wheel zooms. Falls back to a subsampled CPU scatter wherever the zero-copy path is absent.

## PulseScope

`dev.caliper.pulse-scope` · 0.1.0 · `applets/pulse_scope/`

This machine's own vitals, live: PulseScope enumerates every caliper.feed.v1 channel the host vends (CPU/GPU utilization, memory, thermal state, fan, temperature, power on this Mac) and draws one input-locked strip chart per channel from a job-thread poller. Current value + units, a climbing "last sample N.N s ago" staleness label, and a cumulative "lost M samples" gap counter make the honest-loss contract visible. No sensors on this host -> one honest line, nothing faked. CPU + ImPlot only: no tensors, no GPU.

## Examples

Tutorial applets in `examples/`, built with the host.

### SineScope

`dev.example.sine-scope` · 0.1.0 · `examples/sine_scope/`

Tutorial applet: a tiny MLP learns sin(x) live — loss curve, prediction vs target, first-layer weights as a heatmap.

### Hello

`dev.caliper.hello` · 0.1.0 · `examples/hello/`

Epoch-2 fixture applet: sugar demo + loader-test substrate.
