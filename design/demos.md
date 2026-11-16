# Demos

This page plans an increment demo, demo 0, and four demos of scenes and recordings, numbered 1 to 4. Demo 0 is the first visible increment of [Sprint 4](sprint-4.md): the loss surface of a small transformer around its current weights, drawn in 3-D in a notebook while it trains. The four are a structure-from-motion reconstruction with its cameras, a robot arm moved through its entity-transform hierarchy, detection boxes over the frames of a film, and a training run whose activations are drawn live in 3-D and scrubbed afterwards on the `step` timeline. Each demo answers one question with one decisive visual of a real phenomenon. For each, the page gives the question, the data and its licence, how the data is prepared, the Caliper LIB calls, what the viewer shows, and the acceptance criteria, both the checks that run headless and the check left to the person who runs it. It ends with the order in which to build them and what each needs from [Scenes](scenes.md), [Recordings](recordings.md), [Web geometry](web-geometry.md), [Renderer core](renderer-core.md) and [Wire protocol 3.2](wire-protocol-3-2.md).

## What is built and what is planned

Nothing on this page is built. The demos rest on these built pieces: Caliper LIB's live views and its controls of protocol 3.1 (`caliper.step`, `caliper.param`, pause, resume and step from a viewer), which are on the `sprint2` branches; `caliper.serve()` and the web viewer with its headless checks in Chromium, also on `sprint2`; the notebook `examples/notebooks/live_training.ipynb` on `sprint2`, which trains a small transformer with live views in a notebook and which demo 0 continues; and on `main`, the Python API (`docs/references/python-api.md`) and the example `examples/python/train_show.py`, which trains a small network with live views. The scene calls, timelines and recorders they use are planned in [Scenes](scenes.md) and [Recordings](recordings.md).

## How every demo is made

- **One question, one visual.** Each demo answers one question with one view that shows the answer on real data. A second view appears only where it indexes the first, as a text log of the frames worth looking at. The user has approved the questions below.
- **Open data, never redistributed in bulk.** Every source is openly licensed and needs no registration. A demo's `prepare.py` downloads its data into the user cache folder, `platformdirs.user_cache_dir("caliper")` followed by `demos/<demo>`, checks each file against a SHA-256 pinned in the script, and prints the licence and the attribution it asks for. Nothing is committed to the repository except small files that the licence allows, such as the reference captures of the tests, with their attribution in `demos/NOTICE`.
- **Plain web requests.** The download helper sends a generic user agent and no personal details, and it resumes an interrupted download.
- **The same layout.** Each demo lives in `demos/<demo>/`, with `prepare.py`, `demo.py` and a test in `tests/demos/`. `demo.py` runs on the person's Mac with MPS, serves its views with `caliper.serve()`, records the session with `caliper.record()`, and works the same with the Caliper app attached.
- **Headless acceptance.** The tests start the demo, open the viewer page in headless Chromium through Playwright, never on screen, and read the views' state through the test hooks the web views already have, or open the demo's recording file in the same page with `?recording=<url>`. Captures are compared with reference images under the parity rules of [Web geometry](web-geometry.md#parity-with-the-native-renderers). A reference is made on the first verified run, reviewed by the user, and committed. The tests of demos 1 to 3 need their downloads, so they run where the cache is warm, locally or in a nightly job; demo 4's data is small enough for every CI run.

## The increment demo, a loss landscape

**The question.** What does the loss surface around the current weights look like, and how does the optimizer move across it during training?

**The visual.** One 3-D scene, in a notebook widget and in the browser viewer: the loss on a 41 × 41 grid around the current weights, over the plane of the two directions in which the weights have moved most, the top two principal components of the path; the optimizer's path in that plane, as a line strip that grows every step; the current point as a marker; a box and a floor grid; and the orbit camera with its fit. The plane and the surface are computed again on the device every N steps while the model trains. A switch replaces the plane with two filter-normalised random directions. The user chose this demo, its question and its visual as the first visible increment of sprint 4, and decided that the path's plane is the default.

Unlike the four demos below, demo 0 is a notebook, downloads nothing and records nothing, because recordings come in a later increment. It lives in `demos/loss_landscape/`, as `loss_landscape.ipynb` and `landscape.py`, with its test in `tests/demos/`.

**The model and the data.** Demo 0 uses the model, data and training loop of sprint 2's notebook `examples/notebooks/live_training.ipynb`, which its tutorial, Watch a training run in a notebook, walks through on the `sprint2` branch. A transformer of two layers with four heads and width 64, about 104,000 parameters, learns to classify synthetic sequences of 24 tokens into six classes, with AdamW at a learning rate of 2 × 10<sup>-3</sup>, batches of 256 and 8,000 steps on MPS. The data is generated on the device at every step, so there is nothing to download and no licence to check. The notebook repeats those cells, so that it runs alone, and adds the landscape. The losses are taken on a fixed landscape batch, the first 128 of the notebook's 512 evaluation sequences. This choice makes the notebook one continuous story: the tutorial's loss stays on a plateau near the chance loss, ln 6 ≈ 1.79, for one to a few thousand steps and falls within a few hundred once the model has learned to find the marker, and that fall is the event the landscape shows. A perceptron on Fashion-MNIST, demo 4's model, learns smoothly from the first steps, so its landscape would show only a bowl that deepens.

**The weights in the plane.** Both modes perturb the same 102,098 weights: every weight matrix and the token and position embeddings. Biases and the LayerNorm gains and offsets, about 1,800 parameters, are left out of the snapshots and of both directions, as the published method leaves them out of its random directions, so the two modes describe the same weights and the surface keeps those parameters at their current values.

**The path's plane, the default.** The plane follows the treatment of optimizer paths in the method of Li, Xu, Taylor, Studer and Goldstein (2018) for visualising loss landscapes, which plots a training trajectory in the plane of its principal components because random directions are nearly orthogonal to it.

- **Snapshots.** Every 25 steps, and at every refit, the script copies the 102,098 weights into a buffer on the device, 0.41 MB each, together with the loss of that step on the landscape batch. The buffer holds at most 256 snapshots, 105 MB. When it is full, every second snapshot is dropped and the interval doubles, so the kept snapshots stay evenly spaced over the whole run: an 8,000-step run thins once, at step 6,400, and ends with 160.
- **The centre.** The components are taken around the current weights, not around the mean of the snapshots. The surface must pass through the current weights, with the current point at the grid's centre, and a plane through the snapshots' mean need not contain them; centring on the current weights is the method's own choice for trajectories, whose reference is the final weights, with "final" read as "current".
- **The fit, on the device.** With M the K × 102,098 matrix whose rows are the kept snapshots minus the current weights, the script forms the Gram matrix G = M Mᵀ, at most 256 × 256, never the 102,098 × 102,098 covariance. Its top two eigenvectors come from orthogonal iteration (matrix products and Gram–Schmidt, 16 iterations, warm-started from the snapshots' coordinates on the last plane), as demo 4 fits its projection, and each maps back to a unit direction u = Mᵀv / ‖Mᵀv‖. The share of the kept path's variance that the plane holds is (λ<sub>1</sub> + λ<sub>2</sub>) / trace G. Nothing is read back to the host.
- **Refits.** The plane is refitted every N steps, together with the surface, with the same parameter `every`. The new components are matched to the old ones by the larger sum of the absolute values of their dot products, which keeps their order, and each is turned so that its dot product with its predecessor is positive, which keeps its sign, so the picture neither swaps its axes nor flips. The order is then no longer the order of the eigenvalues, which the notebook says.
- **Before the first fit.** A plane needs three snapshots, which exist at step 50. The first refit, the first surface and the path therefore wait until step 50, a fraction of a second into the run, and then come at every multiple of N; until then the scene is empty, and the notebook says why.
- **No filter normalisation.** The components are unit directions taken from the path, and their scale is the path's own, in units of weight space. Normalising them row by row, as the random directions are, would tilt the plane so that the path no longer lay in it. Both axes share one scale, so slopes in the picture are true slopes in weight space.

**Random directions, the switch.** With `directions` set to `"random"`, each direction is drawn once from a standard normal distribution, with the shape of every parameter, by a generator seeded with 0, and is filter-normalised against the weights at step 0: each row of each weight matrix, the incoming weights of one neuron, is scaled to the norm of the same row of the weights, as is each row of the token and position embeddings. The model has no convolutions; in a model with them, each filter would be normalised as a whole. The directions are never normalised again, so the same span means the same perturbation for the whole run. In this mode a refit computes only the surface, and the first comes at step 0.

**What each mode shows.**

| | The path's plane, the default | Random directions, the switch |
|---|---|---|
| The surface | The loss over the two directions in which the weights have moved most | The local shape of the loss in two arbitrary directions, scaled to the weights row by row |
| The path | Lies in the plane up to the variance that the two components miss, which the series `plane share` reports at each refit, so the picture shows how the optimizer moved across the surface | Nearly orthogonal to both directions: a displacement as large as the weights projects onto each with about 1/√102,000, or 0.3 %, of its length, so the path reads as an almost vertical line at the centre |
| From one update to the next | The axes change at each refit, so earlier surfaces are not comparable and the path is redrawn | The axes stay fixed for the whole run, so successive surfaces are comparable and the path is never redrawn |

**The grid on the device.** A surface is 41 × 41 = 1,681 losses of the landscape batch, each at θ + a·d<sub>1</sub> + b·d<sub>2</sub>, where θ is the current weights, d<sub>1</sub> and d<sub>2</sub> are the mode's two directions, and a and b take 41 evenly spaced values from −h to h, h being the box's half-width in the mode's units, as the paragraph on the span below says. `torch.func.functional_call` computes the loss of one grid point from the weights, the directions and the batch, and `vmap` evaluates a row of 41 points in one batched forward pass, so a surface is 41 calls under `torch.no_grad()`:

```python
def grid_loss(a, b):                                       # one grid point
    params = {n: t + a * u + b * v for n, t, u, v in zip(names, theta, d1, d2)}
    return F.cross_entropy(functional_call(model, params, (land_x,))[0], land_y)

row = vmap(grid_loss, in_dims=(0, None))                   # 41 points of one row at once
with torch.no_grad():
    losses = torch.stack([row(offsets, b) for b in offsets])   # (41, 41), on the device
```

Looping over perturbed copies instead would make 1,681 small forward passes, each bound by kernel launches on MPS, where `vmap` makes 41 large ones. A grid point costs about 0.64 GFLOP, 128 sequences of 24 tokens at about 0.2 MFLOP a token, so a surface is about 1.1 TFLOP. At a sustained 2 to 5 TFLOPS through MPS, which is an assumption, that is 0.2 to 0.5 s on an Apple laptop GPU, with a peak of about 0.5 GB for a row's 41 perturbed copies (17 MB of weights) and their activations. A refit adds little: the Gram matrix of 256 snapshots of 102,098 weights is about 13 GFLOP, a few milliseconds at the same rate, and the snapshots' projections and the two directions about 0.1 GFLOP more, while the buffer of snapshots holds up to 105 MB and the refit briefly needs as much again for M. Every figure here is an estimate, and the first build step measures them. If an operation of the model has no batching rule on MPS, the fallback is the loop, estimated at 2 to 3 s a surface. The surface is computed at each refit, every N steps, N being the parameter `every` of protocol 3.1, 500 by default, and at once when a person runs the action `surface now`.

**Log loss, height and a stable range.** Height and colour are both the log of the loss: the plateau near 1.79 and the losses after the fall, a thousandth or less, lie three or more decades apart, which a linear height would flatten onto the floor. A vertex stands at 0.5 · log<sub>10</sub> L, so a decade is half a unit and four decades are about as tall as the default grid is wide. The range is fixed for the whole run so that nothing jumps: log<sub>10</sub> L is clipped on the device to the decades from 10<sup>-4</sup> to 10<sup>2</sup>, the colormap's `vmin` and `vmax` are those two ends, and nothing is read back to the host to choose them. A loss below 10<sup>-4</sup> draws on the floor, and the edges of a wide grid where the loss passes 100 draw at the ceiling.

**The path and the frame of reference.** In the path's plane, the coordinates of weights θ are u<sub>1</sub>·(θ − θ<sub>r</sub>) and u<sub>2</sub>·(θ − θ<sub>r</sub>), where θ<sub>r</sub> is the current weights at the latest refit, so the current point is the plane's origin at each refit. At a refit the whole path batch is replaced: one point for each kept snapshot, projected onto the new plane by one product of M with the two directions, at the loss stored with it, and from then on one point per step, projected onto the same plane and centre, which costs a subtraction and two dot products over 102,098 weights and keeps only the two directions and the centre. Between refits the path is therefore exact step by step, and before the latest refit it has the resolution of the snapshots. Each point's height comes from one extra forward pass without gradients at every step, on the landscape batch. In random directions the anchor is fixed, the weights at step 0, θ<sub>a</sub>: a point's coordinates are the least-squares coefficients of θ − θ<sub>a</sub> on d<sub>1</sub> and d<sub>2</sub>, from a 2 × 2 system solved on the device, the surface at step t lies around (α<sub>t</sub>, β<sub>t</sub>), and the path is a record that is redrawn only when `span` changes its scale. In both modes the surface is the slice through the current weights, its centre vertex is the current point at each update, and earlier points of the path need not lie on it, which the notebook says.

**The span, the box and the camera.** Both modes draw the grid from −1 to 1 across and from −2 to 1 in height, so the box, the floor grid and the camera never move. The parameter `span` says what the box's half-width means. In the path's plane it is 1.5 × `span` times the largest coordinate of a kept snapshot on either component, never below 10<sup>-3</sup> of the weights' norm, so at the default of 1 the farthest snapshot sits two thirds of the way from the centre to the box's edge; the scale changes at each refit, with the axes. In random directions it is `span` units of the filter-normalised directions. Between refits a step can leave the box, and the next refit brings the path back inside. The scene view fits once, when the scene first has bounds, and again only when a person asks, and the box is logged before the first surface, so a refit never moves the camera.

**The mesh.** `positions` f32 (1681, 3) hold the grid's x and y and the clipped height; `indices` i32 (3200, 3) hold two triangles for each of the 40 × 40 cells, made once on the device; `values` f32 (1681,) hold the clipped log loss, drawn through `viridis` on the fixed range. Without normals the faces are shaded flat, which also shows the grid's resolution. At each update the heights and values change; x and y are the same at every refit in the path's plane, and move only with the current point's coordinates in random directions. The triangles never change, but a batch is replaced whole, so their 38,400 bytes travel with every update.

**The path, the marker, the box and the grid.** The path is a `lines` strip written into a buffer on the device; the current point is a `points` batch of one point, a red disc. The box and the floor grid are ordinary line draws that the script logs, not overlays of the view: one static batch of segments at the scene's root, the 12 edges of the box and 9 floor lines each way, logged once. The scene's three entities are `surface`, `path` and `current`; the root, the scene's own space, holds the frame.

| Batch | Sent | Bytes per update |
|---|---|---|
| `surface`, the mesh | At each refit | About 66 KB: positions 20,172, indices 38,400, values 6,724, and the metadata |
| `path`, the strip | Every step, at most 10 times a second, and replaced whole at each refit | In the path's plane, 12 for each kept snapshot and each step since the last refit, at most about 9 KB at the defaults; in random directions, 12 for each step so far, 96 KB at step 8,000 |
| `current`, the marker | Every step, at most 10 times a second | About 0.3 KB |
| `plane share`, a series | At each refit, in the path's plane | One value |
| The box and the grid, at the root | Once | About 1 KB |

**Caliper LIB calls.**

```python
scene = caliper.scene("landscape", up="+z", units="")
every = caliper.param("every", 500, min=10, max=5000, help="steps between refits of the plane and surface")
span = caliper.param("span", 1.0, min=0.05, max=4.0, log=True, help="what the box's half-width covers")
directions = caliper.param("directions", "path", choices=["path", "random"],
                           help="the path's principal components, or random filter-normalised directions")
land = Landscape(model, eval_x[:128], eval_y[:128], grid=41, seed=0, snapshot_every=25, keep=256)
scene.lines("", land.frame, static=True)            # the box and the floor grid, as segments, once


@caliper.action("surface now", help="refits the plane and the surface at the current weights")
def surface_now():
    land.request()


for step in range(steps):
    caliper.step()                                  # pauses and parameter values apply here
    train_step()                                    # the tutorial's loop body, unchanged
    p = land.point(model, step)                     # (3,) on the device; a snapshot every 25 steps
    if land.due(step, every.value, directions.value):   # from step 50, then every N steps
        positions, values, share = land.refit(model, span.value, directions.value)
        scene.mesh("surface", positions, land.triangles, values=values, cmap="viridis",
                   vmin=-4.0, vmax=2.0)
        caliper.series(share, name="plane share")   # the path's variance the plane holds
    if land.ready:                                  # once a plane exists
        scene.lines("path", land.path, strip_lengths=[land.path_length], colors=(235, 235, 235),
                    rate=10)
        scene.points("current", p[None], colors=(255, 80, 80), rate=10)
```

**What the viewer shows.** In the notebook, a cell lays out `caliper.controls()` above `caliper.widget("landscape", "scene", height=480)` and a small `caliper.widget("plane share", "series", height=120)`, which says how much of the path the plane holds and so is read with the scene rather than beside it. The link that `caliper.serve()` prints opens the viewer page with the same views and controls panel. Both draw through WebGPU with the same view layer, and a browser without WebGPU shows a notice in place of the scene. The Caliper app draws the same scene from the next increment on.

**Controls.** The run's Pause, Step and Resume of protocol 3.1; `every`, from 10 to 5,000 steps between refits; `span`, from 0.05 to 4, read per mode as above; `directions`, `"path"` or `"random"`; and the action `surface now`, which refits at once, also while the run is paused. A new span or mode applies at the next refit.

**Acceptance, headless.** The main test runs the notebook's code in the path's plane on the CPU, with fixed seeds and deterministic algorithms, for 300 steps with N = 100, so that refits come at steps 50, 100 and 200, and opens the viewer page in headless Chromium. A second short test runs the random mode for 100 steps with N = 50. Together they are estimated to take about a minute and need no download, so they can run in every CI run, as demo 4's test does.

- The scene holds three entities, `surface`, `path` and `current`, and its root holds the box and the grid; nothing is drawn before step 50 in the path's plane.
- The mesh has 1,681 vertices and 3,200 triangles, and every index is below 1,681.
- At every refit, in both modes, the height of the centre vertex, number 840, equals 0.5 · log<sub>10</sub> of the landscape batch's loss at the current weights, computed separately by a plain forward pass, within 10<sup>-4</sup>.
- The path's last point equals the current point exactly.
- In the path's plane, the two directions are orthonormal within 10<sup>-5</sup>; after the refit at step 200, the path's first nine points, one per kept snapshot, equal the snapshots' projections recomputed in float64 on the CPU within 10<sup>-4</sup> in box units, at their stored losses; and `plane share` holds one value per refit, between 0 and 1, whose value at step 200 is at least the one pinned on the first verified run minus 0.05, because the share depends on the run and no published figure fits this model.
- On a fixture of snapshots along a known curve whose second component outgrows its first between two refits, the matched directions keep their order and sign across the refit; a fixture with room for 8 snapshots checks the thinning rule.
- In random directions, each row of each direction has the norm of the same row of the weights at step 0 within 10<sup>-5</sup>, relatively, and the biases and LayerNorm entries of both directions are 0.
- A capture of the scene view after step 299 in the path's plane, 512 × 512 from the default camera, matches its reference under the parity rules: the path's and the marker's colours exact away from edges, and the flat-shaded surface within 2 levels.
- On MPS, at the default N of 500, the notebook's 8,000 steps take at most 1.5 times as long with the landscape as without it. This row runs on the person's Mac, not in CI, as demo 4's cost row does.

**Manual check.** The expected picture, to be confirmed on the first run: on the plateau the path creeps over a broad, shallow surface near the chance loss, and `plane share` may be low while the weights wander; when the loss falls, the path runs downhill across the surface into a valley that each refit shows deeper, and the share rises as one direction dominates; lowering `every` to 50 as the loss starts to fall shows the valley open from refit to refit; and switching `directions` to `"random"` shows the local shape of the loss with the path as a nearly vertical line, which is why the path's plane is the default.

## The first demo, structure from motion

**The question.** Which frames of a moving shot does structure-from-motion place least accurately?

**The visual.** One 3-D scene: the sparse point cloud in the colours of the photographs, a frustum for every registered frame, the camera path through their centres, the centres coloured by each frame's mean reprojection error, and the current frame's image on its own frustum's image plane. Scrubbing the `frame` timeline moves the current camera along the path, so the person sees which stretch of the shot carries the error and what the images there look like.

| Source | Licence | URL | Size |
|---|---|---|---|
| Tears of Steel, the Blender Foundation's open film, live-action footage | Creative Commons Attribution 3.0 (CC BY 3.0); reuse and distribution, also commercial, allowed with proper attribution; logos and sponsor logos excluded | Licence: https://mango.blender.org/sharing/ | To verify: the official host refused automated checks |
| COLMAP, through its Python package pycolmap | COLMAP: "the new BSD license" (3-clause); pycolmap: BSD-3-Clause | https://github.com/colmap/colmap, https://pypi.org/project/pycolmap/ | pycolmap wheels 20 to 38 MB |

**Preparing the data.** `prepare.py` reads one shot of about 10 seconds, a moving camera over a still set, from the film's file over HTTP with range requests, so that only the shot's bytes travel; a host without range support makes it download the whole file once. It decodes every second frame at 1280 pixels wide into the cache with PyAV, then runs pycolmap's feature extraction, sequential matching, which suits video, and incremental mapping, and stores the model in the cache. For each registered image it computes the mean reprojection error of its observations from the model. The first build step chooses the shot, pins its start time, the file's URL, its SHA-256 and the expected number of registered frames.

**Caliper LIB calls.**

```python
scene = caliper.scene("sfm", up="-y", units="arbitrary")  # COLMAP's world starts in the first camera, y down
caliper.timeline("frame")
scene.points("world/cloud", xyz, colors=rgb, static=True)
for i, cam in enumerate(registered):
    scene.transform(f"world/cameras/f{i:04d}", matrix=cam.world_from_cam, static=True)
    scene.camera(f"world/cameras/f{i:04d}", width=w, height=h, focal=(fx, fy), principal=(cx, cy), static=True)
scene.lines("world/path", centres, strip_lengths=[len(registered)], static=True)
scene.points("world/path/errors", centres, values=errors, cmap="magma", vmin=0.0, vmax=2.0, static=True)
scene.camera("world/current", width=w, height=h, focal=(fx, fy), principal=(cx, cy),
             plane_distance=radius / 5, static=True)
for i, cam in enumerate(registered):
    caliper.set_time("frame", i)
    scene.transform("world/current", matrix=cam.world_from_cam)
    scene.image("world/current/frame", jpeg_bytes[i])
```

**What the viewer shows.** The orbit view fits the cloud. Up to about 120 frustums line the path, which runs dark where the error is small and turns bright where it is large; the current frustum carries its frame. Looking through `world/current` shows the frame with the cloud projected over it, so a misplaced camera shows as points that do not sit on the image.

**Acceptance, headless.**

- The scene holds R + 7 entities for R registered frames, the implicit parents `world` and `world/cameras` included, with R at least the pinned count; the cloud shows as many points as the model has, or 100,000 sampled when it has more.
- The recording reopens; at frame 60 the pose of `world/current` equals the model's pose of that frame within 10<sup>-6</sup>, and its image decodes to the bytes the script logged.
- A capture of the orbit view at frame 0, 512 × 512, matches the reference under the parity rules.
- Playing the recording at 1× holds 60 frames per second on the Mac's GPU.

**Manual check.** The cloud looks like the set in the frames, the path follows the camera move the person sees in the shot, and the bright stretch of the path matches frames that are blurred, fast or poor in texture.

## The second demo, a robot arm

**The question.** Where along a straight reach does the arm approach a singularity?

**The visual.** One 3-D scene: a Franka Emika Panda drawn from its own meshes, each link an entity under its parent, moving along a straight end-effector path that stretches the arm towards full extension; the end effector's trail is coloured by the arm's manipulability, √det(J Jᵀ), which falls to zero at a singularity. Scrubbing `step` shows the pose at each step and where the trail darkens.

| Source | Licence | URL | Size |
|---|---|---|---|
| The Panda model of MuJoCo Menagerie, `franka_emika_panda` | Apache-2.0, from the model's own LICENSE file | https://github.com/google-deepmind/mujoco_menagerie | 67 files, about 33 MB |
| MuJoCo's Python package, for kinematics and Jacobians | Apache-2.0, to verify | https://pypi.org/project/mujoco/ | To verify |

**Preparing the data.** `prepare.py` fetches only the model's folder, with a sparse, shallow clone of the repository into the cache, so the repository's other models, about 650 MB, are not downloaded. The script loads the model, reads each body's visual meshes (vertices, faces and normals) and their poses within the body, and solves a damped least-squares inverse kinematics along a straight line of 300 steps from a bent pose outward, recording the joint angles, the end effector's position and the manipulability at each step.

**Caliper LIB calls.** Every link is logged once as static meshes under its body's entity, and each step logs only the moving bodies' transforms relative to their parents, so the hierarchy carries the motion and no geometry travels after the first step.

```python
scene = caliper.scene("arm", up="+z", units="m")
for body in bodies:                                      # world/panda/link0/link1/.../link7/hand
    for k, g in enumerate(body.geoms):
        scene.transform(f"{body.path}/g{k}", matrix=g.local_pose, static=True)
        scene.mesh(f"{body.path}/g{k}", g.vertices, g.faces, normals=g.normals, colors=g.rgba, static=True)
for t in range(steps):
    caliper.step(t)
    for body in moving_bodies:
        scene.transform(body.path, translation=body.rel_pos[t], rotation=body.rel_quat[t])
    scene.lines("world/trail", ee[: t + 1], strip_lengths=[t + 1], width=3.0)
    scene.points("world/trail/w", ee[: t + 1], values=manipulability[: t + 1], cmap="magma",
                 vmin=0.0, vmax=w_max)
```

**What the viewer shows.** The lit arm, its trail, and the trail's points, bright where the arm moves freely and dark where it stretches out. Selecting a link entity in the view's tree highlights its subtree, which shows the hierarchy directly.

**Acceptance, headless.**

- The scene holds one entity per body and per visual mesh, as the model lists them, plus the trail.
- After the static meshes, the recording grows by at most 2 KB a step: transforms and the trail only.
- Reopened at step 150, the world matrix the viewer composes for `hand` equals MuJoCo's pose of that body within 10<sup>-5</sup> m and 10<sup>-5</sup> rad.
- Captures at steps 0 and 299 match their references under the parity rules, the lit links within 2 levels.
- Playback at 1× holds 60 frames per second on the Mac's GPU.

**Manual check.** The arm moves smoothly and looks like a Panda, each joint turns its whole subtree, and the trail darkens as the arm reaches full extension.

## The third demo, detection boxes

**The question.** In which frames does the detector lose a person who is still in view?

**The visual.** One 2-D scene: the current frame of a shot with several actors, and the detector's boxes over it in the class colours of the COCO class table, each labelled with its class and score. Scrubbing the `frame` timeline steps through the shot. A text log beside it lists the frames where the number of person boxes falls below both neighbours, and a click on a line moves the cursor to that frame.

| Source | Licence | URL | Size |
|---|---|---|---|
| Tears of Steel, the same file as demo 1, another shot | CC BY 3.0, as in demo 1 | https://mango.blender.org/sharing/ | Shared with demo 1 |
| RT-DETR, the detector's code | Apache-2.0 | https://github.com/lyuwenyu/RT-DETR | Code only |
| RT-DETR weights `PekingU/rtdetr_r50vd`, trained on COCO | apache-2.0, on the model card | https://huggingface.co/PekingU/rtdetr_r50vd | 172 MB |
| The Transformers library, to run the detector | Apache-2.0, to verify | https://pypi.org/project/transformers/ | To verify |

No AGPL code or weights are used. The weights' card states Apache-2.0, but weights trained on COCO may also carry terms from that dataset, which the demo's NOTICE states.

**Preparing the data.** `prepare.py` reads a shot of about 20 seconds with actors in view, at the film's 24 frames per second, as demo 1 reads its shot, downloads the weights into the cache and pins both. `demo.py` runs the detector on MPS frame by frame with a score threshold of 0.5 and logs each frame as JPEG bytes, so that a frame costs about 150 KB on the wire rather than 2.7 MB.

**Caliper LIB calls.**

```python
scene = caliper.scene("detections", space="2d")         # y down, one unit per pixel
scene.classes({i: name for i, name in enumerate(COCO_CLASSES)})
caliper.timeline("frame")
for i, (jpeg, frame) in enumerate(frames):
    caliper.set_time("frame", i)
    out = detector(frame)                                # boxes (N, 4) as x0, y0, x1, y1 on the device
    scene.image("video/frame", jpeg)
    scene.boxes("video/detections", corners=out.boxes, class_ids=out.labels,
                labels=[f"{COCO_CLASSES[c]} {s:.2f}" for c, s in zip(out.labels.tolist(), out.scores.tolist())])
    if dropped(i):
        caliper.text(f"frame {i}: {n_person[i]} people detected, {n_person[i - 1]} in frame {i - 1}")
```

**What the viewer shows.** The frame at the cursor with its boxes and labels, and the text log of dropped frames. Scrubbing to a listed frame shows the actor who has no box.

**Acceptance, headless.**

- At frame 100 the recording holds as many boxes as the detector returned above the threshold, with corners equal to its output within 10<sup>-3</sup> pixels after the conversion on the device, and the labels the script made.
- The text log holds exactly the lines the script computed, and a click on one moves the cursor to its frame.
- A capture at frame 100 matches the reference under the parity rules.
- Playback at 1× holds the film's 24 frames per second in the browser, decoding included.

**Manual check.** The boxes sit on the actors and objects, and in the listed frames a person is visible without a box.

## The fourth demo, a training run

**The question.** When during training do the ten classes separate in the network's last hidden layer?

**The visual.** One 3-D scene: 2,000 fixed test images, 200 of each class, drawn as points at their activations in the last hidden layer, projected to 3-D on the device and coloured by their true class. Live, the cloud parts into clusters as the network learns; T-shirts, shirts, pullovers and coats stay entangled, which is the dataset's known confusion. Afterwards the recording is scrubbed on `step` to find the moment the clusters part.

| Source | Licence | URL | Size |
|---|---|---|---|
| Fashion-MNIST | MIT License, "Copyright 2017 Zalando SE" | https://github.com/zalandoresearch/fashion-mnist | Training images 26 MB, training labels 29 KB, test images 4.3 MB, test labels 5.1 KB |
| torchvision, to download and read it | BSD-3-Clause | https://github.com/pytorch/vision | Installed with torch |

**Preparing the data.** `prepare.py` downloads the four files with torchvision's `FashionMNIST` loader into the cache, checks their hashes, and chooses the 2,000 probe images with a seeded generator.

**The model and the projection.** A multilayer perceptron, 784 to 256 to 64 to 10 with ReLU, trained with Adam at a learning rate of 10<sup>-3</sup> and batches of 128 for 4 epochs, 1,876 steps, on MPS. Every `probe_every` steps the script runs the probe through the network without gradients and takes its 64 activations of the last hidden layer. The projection is a principal component analysis done on the device with torch: the probe's mean and the 64 × 64 covariance, then the top three directions by orthogonal iteration (matrix products and Gram–Schmidt, eight iterations), warm-started from the previous directions. The directions, the mean and a scale, the probe's root-mean-square distance from its mean, are refitted every 200 steps and held fixed between refits, so that the points move only because the activations move. The warm start keeps each refit close to the last, and an axis whose new direction points against its old one is flipped. Only the 2,000 × 3 result leaves the device, 24 KB per snapshot.

**Controls.** The run controls of protocol 3.1 and one parameter. The person pauses the run when the clusters begin to part, lowers `probe_every` from 10 to 1 so that every step is projected, and steps through the separation one step at a time before resuming. The recording keeps those dense steps.

**Caliper LIB calls.**

```python
scene = caliper.scene("activations")
scene.classes({0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"})
caliper.items("probe", details={"class": [NAMES[c] for c in probe_y.tolist()], "test index": probe_idx.tolist()})
probe_every = caliper.param("probe_every", 10, min=1, max=100, help="steps between projections")
recorder = caliper.record(cache / "activations.mcap")
for t, (x, y) in enumerate(batches):
    caliper.step(t)                      # the checkpoint: pause, step and parameter values apply here
    train_step(x, y)
    if t % probe_every.value == 0:
        h = hidden(probe_x)              # (2000, 64) on MPS
        z = project(h, t)                # (2000, 3) on MPS, the basis refitted every 200 steps
        scene.points("probe", z, class_ids=probe_y, items="probe")
        caliper.series(knn_accuracy(z, probe_y), name="separation")   # for the tests, not in the layout
```

The probe is the item space `probe` of protocol 3.1, so a lasso on a cluster selects those test images, and the person can see which images sit in the wrong cluster.

**What the viewer shows.** In the Caliper app on the same Mac, the probe's positions are drawn in place from the device through import, with no copy; in a browser the 24 KB snapshot is copied twice, as every browser snapshot is. The cloud starts as one blob, spreads within the first epoch, and settles into separate clusters for trousers, bags, footwear and dresses while the tops overlap. The scrubber afterwards replays the run at any step, with the recorded parameter values and run state at the cursor.

**Acceptance, headless.** The test runs the demo on the CPU with fixed seeds and deterministic algorithms for 400 steps, so that its numbers repeat.

- At every projection stamp the scene holds one entity, `probe`, with 2,000 points and 10 classes, and the recording holds a snapshot at every step that is a multiple of 10.
- The test acts as a viewer: it pauses the run at step 100, sets `probe_every` to 1, steps 5 times, sets it back to 10 and resumes. The recording then holds a snapshot at every step from 100 to 104, and at those steps the recorded run state reads stepping or paused and `probe_every` reads 1.
- Reopened and scrubbed to step 200, the probe's positions equal the script's own copy of step 200 within 10<sup>-6</sup>.
- The separation score at step 400 is at least the value pinned on the first verified run minus 0.02, and at step 0 below it by at least 0.2.
- Captures at steps 0 and 400 match their references, with every disc centre's class colour exact.
- On MPS with the Caliper app attached, a training step costs at most 5 % more than the same loop without Caliper, measured as `bench_live.py` measures steps per second.

**Manual check.** Watching live, the clusters form within the first epoch and the tops stay mixed; pause, step and the parameter respond within a second; the scrubbed recording shows the same run.

## The order of building

Demo 0 comes before the four: the user chose it as the first visible increment of [Sprint 4](sprint-4.md), right after the renderer core's minimum cut, and it needs the least, three primitives sent over the network to a notebook and a browser, with no recording. Of the four, demo 4 comes first: it uses the least of the scene model, points and a class table, and exercises the newest and riskiest pieces, recordings, the scrubber and controls in a session with a scene, while it shows Caliper's strengths. Demo 2 adds entity transforms, lit meshes and lines; demo 3 adds a 2-D scene, encoded frames, boxes with labels, a timeline of the script's own and the text log; demo 1 adds cameras, frustums and images on image planes, many entities and the largest cloud, and reuses demo 3's film.

| Order | Demo | Renderer core | Scenes | Web geometry | Recordings | Wire protocol 3.2 | Sprint 2 and 3 |
|---|---|---|---|---|---|---|---|
| 0 | Loss landscape | The minimum cut; the colormap by keys in the browser | `scene`, `mesh`, `lines` as strips and segments, `points`, values through a colormap, `static`; the network route only | Points, segments, meshes, the index clamp, the colormap | None | Kind 6, SCENE_UPDATE for three parts, SCENE_CLEAR | `caliper.serve()`, the widget, controls of 3.1, the tutorial's notebook |
| 1 | Training run | The scene core with points and the class table; import on Metal | `points`, `classes`, `items` | Points, the class table, the colormap | `step`, `record`, the scrubber, recorded controls | Scene messages, recordings | Controls of 3.1, item spaces, `caliper.serve()` |
| 2 | Robot arm | Meshes and per-vertex lighting, the entity tree's world matrices | `transform`, `mesh`, `lines`, `points` with values | Meshes, segments | Keyframes of static meshes | Scene messages | The web viewer |
| 3 | Detection boxes | Textures and boxes | 2-D space, `image` with encoded frames, `boxes` with corners and labels, `classes` | Images, box edges, labels | `timeline`, `set_time`, text | Text kind 7, encoded frames | Kind ids confirmed |
| 4 | Structure from motion | Cameras, looking through a pinhole | `camera`, frustums, images under a camera | Images on image planes, segments for frustums | Large static items | Scenes over several messages | |

Each of the four is about 2 builder days of its own once its dependencies exist: the script, its preparation and its test, with the reference captures and the user's review. Demo 1 needs a third day to choose and pin a shot that reconstructs well. Demo 0 needs 3 days, because besides the notebook and its test it has a numerical module of its own: the snapshots and the plane of the path, the random directions, the grid on the device and the projection. [Sprint 4](sprint-4.md) places each demo in its increment.

## Limits

- **Data still to verify.** The film's download URL and size, the licences of the MuJoCo, Transformers and PyAV packages, and whether a chosen shot of the film reconstructs, are not verified; the first build step of each demo verifies them and pins what it found.
- **Weights trained on COCO.** The detector's weights are Apache-2.0 on their card, but they were trained on COCO, whose images carry varied licences.
- **Large downloads.** Demos 1 and 3 need the film and 172 MB of weights, so their tests do not run in every CI job.
- **Zero-copy.** Only the Caliper app on the same machine draws demo 4's points in place; a browser or a hub receives copies.
- **Determinism.** Training on MPS does not repeat bit for bit, so the exact checks of demos 0 and 4 run on the CPU and their MPS runs are checked for cost only.
- **Demo 0's plane and estimates.** The plane's axes change at each refit, so its surfaces are not comparable from one refit to the next, and early in training two components may hold little of the path's variance, which `plane share` reports. The time and memory of a surface are estimates, and `vmap`'s batching rules on MPS are unverified for the model's operations.

## Decisions

- **Tears of Steel for demos 1 and 3.** It is live-action footage of real places under CC BY 3.0, and one download serves both demos. The multi-view photo sets that were found are non-commercial, unlicensed, or, in one case, 27.5 GB behind links that may need a sign-in.
- **Structure from motion run by the demo with pycolmap.** Both COLMAP and pycolmap are BSD-licensed, and running it gives the per-frame error the question needs.
- **The Panda from MuJoCo Menagerie for demo 2.** Its model and meshes are Apache-2.0, it is a real robot, and its folder is 33 MB.
- **Manipulability as demo 2's measure.** It is the standard measure of nearness to a singularity and falls to zero there.
- **RT-DETR with Apache-2.0 code and weights for demo 3.** It avoids AGPL detectors and runs on MPS through a widely used library.
- **Fashion-MNIST and a small perceptron for demo 4.** The data is MIT-licensed and 30 MB, the model trains in minutes on a laptop, and its four confusable classes give a visible answer.
- **PCA by warm-started orthogonal iteration, refitted every 200 steps.** It uses only matrix products on the device and keeps the axes steady between and across refits.
- **`probe_every` as demo 4's parameter.** It changes what the script computes exactly where the question needs finer time, and the recording keeps the result.
- **Encoded frames in demo 3 and demo 1.** A JPEG frame costs a twentieth of the raw bytes, which matters most in a browser.
- **References made on the first verified run and reviewed by the user.** A capture cannot be specified in advance, and a reviewed one guards against later drift.
- **The four questions, as written above.** Approved by the user.
- **The robot arm for demo 2.** Approved by the user: the Panda, because its singularity gives a single decisive answer. The human skeleton is not built.
- **RT-DETR's COCO-trained weights for demo 3,** with the note about COCO in the demo's NOTICE. Approved by the user.
- **Two shots of Tears of Steel are tried first for demo 1.** Approved by the user. Another source is considered only if both reconstruct poorly.
- **Demo 4's test runs in every CI run,** on the CPU. Approved by the user.
- **A loss landscape in a notebook as demo 0, the first visible increment of sprint 4, with its question and its visual.** Decided by the user, who wants richer 3-D than point clouds in the notebook as soon as possible.
- **Sprint 2's notebook model and data for demo 0.** The notebook continues one story, since the tutorial's plateau and sudden fall are the event the landscape shows; it downloads nothing; and the model trains on MPS in minutes. Fashion-MNIST's perceptron learns smoothly and would show only a bowl that deepens.
- **The plane of the path's two principal components by default, and filter-normalised random directions as the switch.** Decided by the user: the demo's question includes how the optimizer moves across the surface, and in random directions, which are nearly orthogonal to the training trajectory, the path is an almost vertical line, while in the path's plane both the surface and the path mean something.
- **Components around the current weights, from the Gram matrix of at most 256 snapshots taken every 25 steps, refitted every N steps with their order and sign kept.** The surface must pass through the current weights, the Gram matrix keeps the fit on the device at a few milliseconds without a covariance over 102,000 dimensions, thinning by halves keeps the snapshots evenly spaced, and kept order and sign stop the picture from swapping or flipping its axes.
- **The first plane at step 50, when three snapshots exist, and nothing drawn before it.** A plane from fewer snapshots is undefined, and showing random directions first would change the picture's meaning a moment later.
- **One fixed box for both modes, with `span` read per mode.** The box, the floor grid and the camera then never move at a refit; in the path's plane both axes share one scale, so slopes stay true.
- **The same 102,098 weights in both modes, without biases and normalisation parameters.** The two modes then describe the same weights, as the published method's random directions leave those parameters out.
- **A fixed anchor for the random directions.** Their coordinates mean the same thing for the whole run, so the path there is a record that is never redrawn.
- **The log of the loss for height and colour, on a range fixed for the run.** The plateau and the losses after the fall are decades apart, and a fixed, clipped range keeps the surface and its colours from jumping without reading anything back from the device.
- **Rows of 41 grid points batched with `torch.func.functional_call` and `vmap`.** On MPS, 41 large forward passes beat 1,681 small ones bound by kernel launches.
- **The box and the floor grid as line draws at the scene's root.** They need no new renderer code, draw alike natively and in the browser, and show the fixed range, which a box from the data's bounds would not.

## Open decisions

None. The user approved the four demos as written and decided demo 0's default plane, with the choices recorded above.
