# Sprint 5

Sprint 5 is the macOS sprint after sprint 4. It adds run comparison, fields for the physical sciences, robotics, a few viewport features, and the scene additions that sprint 4 left out. It builds on the renderer core, scenes and recordings of sprint 4 and starts once they are merged. The [roadmap](roadmap.md) holds the order and status.

The users are ML engineers in the physical sciences and robotics. They look at data and model internals during training and inference, compare runs, and debug models, without building rendering pipelines. The viewport is therefore an inspection camera for data, not a modelling viewport: a person looks at a tensor in space, measures it and compares it with another run. Every item below is a camera change, a data view or a loader on top of the scene core; none needs a new rendering technique.

## Where things stand

- **Sprint 4 is designed and not built.** Sprint 5 depends on its scene core and primitives, entity transforms, the up-axis switch, looking through a pinhole camera, `view_camera=`, the scale bar from a scene's `units`, index-pair edges, and recordings with timelines and the scrubber ([Scenes](scenes.md), [Recordings](recordings.md)).
- **Sprint 3 is built as libraries and not wired into the app.** Sprint 5's mesh files go through its loaders and catalog ([Applet builder](applet-builder.md)).
- **Run history** in the desktop app's local database is planned in [Data and Cloud](data-and-cloud.md). Run comparison reads it.
- **Recordings are MCAP files.** Importing ROS 2 recordings, which are MCAP too, shares the container.

## How the sizes are given

Each step lists its items with what each depends on and an estimated size in days of one builder. The sizes are estimates made before the detailed design, and they will be refined when the sprint is designed.

## The steps

Each step ends with a check that must pass in the app and, where the step draws anything, in the browser.

### 1. Comparing runs

Comparing runs is the headline of the sprint: it answers the question ML engineers ask most often, which of two runs did better, where and when.

- **Several runs on one page.** A page holds two or more sessions, each live or recorded, on one shared cursor. Runs are aligned by `step` by default, or by `time`, and a run that has no item at the cursor shows its latest earlier item with its distance from the cursor, as the scrubber does.
- **Linked cameras and selections.** Orbiting one run's scene turns the matching view of the other run, and selecting items in one run highlights the items with the same row ids in the other.
- **Difference views.** A heatmap of A minus B for tensors of the same shape, loss curves overlaid on one axis, and histograms on shared bins.
- **Done when** two recordings of the Fashion-MNIST training demo, with different learning rates, scrub together on one page, their embedding scenes turn together, and a difference heatmap of one layer's weights updates with the cursor.

| Item | Depends on | Builder days |
|---|---|---|
| Several sessions on one page with a shared cursor, aligned by step or time | Sprint 4's scrubber and recordings | 6 |
| Linked cameras and linked selections across runs | The item above | 2 |
| Difference views: A minus B heatmaps, overlaid series and histograms on shared bins | The first item | 3 |

### 2. Fields for the physical sciences

- **Slice planes.** A slice plane cuts a 3-D field, such as a flow, a weather volume or a density grid, along an axis. It is placed inside the scene and dragged along its axis. The slice is a data view computed on the device where the field lives, and it is drawn on a textured quad, so only the slice travels and no new renderer is needed.
- **Isosurfaces.** An isosurface is the surface where a field equals a chosen value: a front, a boundary or an orbital. A marching-cubes kernel runs on the device and sends the result as a mesh, which the scene's mesh primitive draws. It comes after the slices, on the same field data.
- **Done when** a 3-D field from a script shows three draggable slice planes and one isosurface at a value set from the controls panel, natively and in the browser.

| Item | Depends on | Builder days |
|---|---|---|
| Slice planes of 3-D fields, placed in the scene and dragged along an axis | Sprint 4's scene core and data views | 5 |
| Isosurfaces computed on the device and sent as meshes | Slice planes, for the field data | 7 |

### 3. Robotics

- **Mesh files.** glTF, OBJ and STL files load through sprint 3's loaders into mesh sources, instead of each script loading them by hand.
- **A URDF helper.** One call in Caliper LIB turns a robot description into entities: a link per entity, with its meshes and joint transforms.
- **Follow an entity.** The orbit's target tracks a moving entity, such as a gripper, while the person orbits around it.
- **Trails over a time window.** A view shows the last N steps of an entity's positions, such as a path, a gripper or particles, in place of only the latest.
- **Uncertainty.** Ellipsoids, for covariances from state estimation, and error bands on series.
- **Importing ROS 2 recordings.** ROS 2 recordings in MCAP open in Caliper, with the common message types (transforms, point clouds, images and markers) decoded into scene entities. Their named coordinate frames are translated into the entity tree.
- **Done when** a robot loads from its URDF, its gripper is followed through a recorded reach with a trail, and a public ROS 2 recording opens with a model's predictions overlaid.

| Item | Depends on | Builder days |
|---|---|---|
| Mesh files (glTF, OBJ, STL) through the loaders | Sprint 3 wired into the app | 4 |
| A URDF helper in Caliper LIB | Mesh files | 3 |
| Follow an entity | Sprint 4's orbit camera | 2 |
| Trails over a time window | Sprint 4's recordings, as a key of the subscription | 3 |
| Ellipsoids and error bands | Sprint 4's scene core | 4 |
| Importing ROS 2 recordings in MCAP, with named frames translated into the entity tree | Sprint 4's recordings | 6 |

### 4. Viewport

- **An orthographic projection and preset views.** A switch between perspective and orthographic projection, and preset views from the top, front, side and three-quarter, for measuring and comparing shapes without perspective distortion.
- **Saved viewpoints.** A named camera position, kept with the view, in recordings and in spec applets, so that a chosen angle can be returned to and shared.
- **Done when** a view switches to orthographic, takes each preset, and reopens a recording on a saved viewpoint.

| Item | Depends on | Builder days |
|---|---|---|
| Orthographic projection and preset views | Sprint 4's orbit camera and up-axis switch | 3 |
| Saved viewpoints in views, recordings and spec applets | `view_camera=` from sprint 4 | 2 |

### 5. Scene additions moved from sprint 4

Sprint 4 left these out to stay a manageable size, and they come here.

- **More shapes.** Capsules, cylinders and planes, beside the ellipsoids of step 3, each drawn from its parameters rather than sent as a mesh.
- **Events from the app back to the script over shared memory.** A local session gets an event ring in the scene's directory segment, which the app writes and Caliper LIB polls, so that picks, selections and control changes reach a script without `caliper.serve()`. The network route of sprint 2 has by then settled what an event carries.
- **Named frames** are covered by step 3's ROS 2 import, which translates them into the entity tree; a second transform system stays out, as below.
- **Done when** a scene draws each new shape natively and in the browser, and a script attached through shared memory receives a pick and a selection without a server.

| Item | Depends on | Builder days |
|---|---|---|
| Capsules, cylinders and planes | Sprint 4's scene core | 3 |
| The event ring from the app to the script over shared memory | Sprint 4's scene layout of the shared-memory session | 3 |

### 6. Demos

Each demo answers one question with one decisive visual, and each question is agreed before the demo is built, as in [Demos](demos.md).

| Demo | Question | Data |
|---|---|---|
| Comparison | Which of two learning rates separates the ten Fashion-MNIST classes first? | Fashion-MNIST (MIT), extending sprint 4's training demo |
| Fields | Where does a neural PDE surrogate's error concentrate inside a 3-D flow? | A public PDE benchmark dataset, with its licence to be checked |
| Robotics | Where along a recorded robot run does a model's prediction drift from what happened? | A public ROS 2 recording in MCAP, with its licence to be checked |

- **Done when** each demo runs from a fresh checkout and answers its question in one view.

| Item | Depends on | Builder days |
|---|---|---|
| The three demos | Steps 1 to 5 | 6 |

## What it adds to Windows parity

Two items join [Windows parity](windows-parity.md):

- **Arrays from JAX and CuPy through DLPack.** Both are built for NVIDIA GPUs, and CuPy does not run on macOS, so they come with Windows parity. Until then a JAX user passes NumPy arrays, at the cost of one copy.
- **Slice planes and isosurfaces on CUDA:** the device-side slice and the marching-cubes kernel.

## Left out

These items are left out because each is either a new rendering technique, which costs weeks and a parity suite on every backend, or editing a scene rather than showing data.

| Item | Why | What would change it |
|---|---|---|
| Ray-marched volume rendering | It needs a transfer function editor, sampling and quality trade-offs on every backend, and dedicated volume tools already do it well. Slice planes and isosurfaces answer most questions an ML engineer asks of a field | Users of slices and isosurfaces asking for it |
| Named coordinate frames as a second transform system | The entity path tree already composes transforms; ROS 2 import translates named frames into it | Data whose frames cannot be mapped to a tree |
| A fly mode | Scenes in ML are object-sized, and orbit, fit and follow are enough | City-scale maps or SLAM environments |
| Object gizmos and scene editing | Caliper shows data as the script sent it, and moving objects in the viewer would make the picture disagree with the data | None expected |
| Lights and materials | They change appearance, not information | None expected |
| Camera paths | They exist to make videos, and Caliper streams data, not video | None expected |

## Open questions

1. Which PDE dataset the fields demo uses, and which public ROS 2 recording the robotics demo uses.
2. Whether difference views need tensors of the same shape, or also align runs whose shapes differ, such as two architectures.
3. Whether ROS 2 import covers only the common message types or more.

## Risks

- **Aligning runs.** Runs that log at different steps or rates need a clear rule for which item pairs with which.
- **Isosurfaces on the device.** The marching-cubes kernel runs on each backend, and its mesh changes size from frame to frame.
- **ROS 2 message types.** Decoding them adds a dependency, and the common types vary between ROS 2 releases.
- **Demo data.** The licences and download sizes of the two new datasets are not checked yet.

## Order

Comparing runs comes first, because it builds directly on sprint 4's recordings. Slice planes and the robotics items follow in parallel, then the viewport items and the scene additions, then isosurfaces after slices, then ROS 2 import, and the demos last. The total is about 62 builder days.
