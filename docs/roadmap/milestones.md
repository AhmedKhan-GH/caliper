# Milestones

This page lists the milestones of Caliper's current trajectory, Agent-Built Playground, and the state of each one.

## Progress

| Milestone | Scope | Status | Pull Request |
|---|---|---|---|
| M0, Reconciled Base | Graphics layer, scenes, protocol 3.2, picking | Done | [#24](https://github.com/AhmedKhan-GH/caliper/pull/24) |
| M1, Agent-Built Instruments | Scene panels, controls, run control and events over MCP | Done | [#25](https://github.com/AhmedKhan-GH/caliper/pull/25) |
| M2, GPU Programs | Agent-written WGSL over live tensors, stages, sandbox, first ablation | Done | [#27](https://github.com/AhmedKhan-GH/caliper/pull/27) |
| M3, Instrument Building | Entities from tensors, operators, shared state, time transport, annotations, all control kinds | Done | [#28](https://github.com/AhmedKhan-GH/caliper/pull/28) |
| M4, Hosted Runs | Start, fork and stop a model's run from Caliper and over MCP | Done | [#26](https://github.com/AhmedKhan-GH/caliper/pull/26) |
| M5, Programmable Graphics | Pieces with a derive half in the run and a draw half in the viewer, written by the agent, in the app and the browser | Done | [#34](https://github.com/AhmedKhan-GH/caliper/pull/34) |
| M6, Graphics Consolidation | Every built-in shape and operator as a library piece, and the old per-shape code removed | In progress | |
| M7, Derivation Catalog | Projections, layouts, overlays and model probes as library pieces | Planned | |
| M8, Steering and Intervention | Steer, scale and patch | Planned | |
| M9, Saved and Editable Instruments | Saved instruments with their pieces, and the node editor | Planned | |
| M10, Browser Parity | The rest of an instrument in the browser | Planned | |
| M11, Training Debugger | First acceptance demo | Planned | |
| M12, Image Sequences and Training Data | Image grids, sequences, augmentation preview | Planned | |
| M13, World-Model Playground | Second acceptance demo | Planned | |
| M14, Vector Drawing and Library Completion | 2-D vector drawing, and the remaining shapes as library pieces | Planned | |
| M15, Time Axis | Recordings and timelines | Deferred | |
| M16, WebAssembly Views | Agent-written interaction logic | Deferred | |

## Order

The milestones up to M4 ran in the order M0, M1, M4, M2, M3. Hosted runs came before GPU programs, so that each later demo starts its own run. From M5 on, the numbers follow the order of work. Programmable graphics comes first, because the agent authors the graphics: it writes its own buffers, pipelines and passes, and the built-in primitives become a library on that layer. The derivation catalog follows as library programs, and steering and intervention can run beside it.

Each milestone ends in a state that works. A milestone is done when the release build is clean, the suites and the browser checks pass, a demo script runs without a person, the docs are written, and a short check shows the result.

## Agent-Built Instruments

M1 lets an agent build a live instrument on a running script through MCP alone. The script needs one line, `caliper.watch(model, optimizer)`. The agent adds a scene panel and controls, and a slider changes the learning rate at the next step after an approval card. A click in the scene returns to the agent as an event.

## Hosted Runs

M4 lets a person or an agent start a model's run from Caliper, with no script of their own. A run can be forked at a checkpoint with a changed parameter, and the two runs show side by side. A review card states exactly what runs before it starts.

## GPU Programs

M2 lets an agent write WGSL programs that read live training tensors on the training GPU with no copy. A program writes geometry for a scene entity or pixels for a panel. A sandbox keeps model tensors read-only, stops a program at its time limit and holds it inside a memory budget. Programs run every N steps, so training keeps its speed. The first intervention, an ablation of a module output through a toggle, came in the same milestone.

## Instrument Building

M3 lets an agent turn any tensor of the running model into a visual with no scene code in the script. The `surface` and `path` operators derive geometry on the GPU. Shared state connects the views: a pick in one panel changes another, and a time transport moves one cursor through the recent steps. Annotations, bound text readouts, segmented groups and every control kind complete the interface.

## Programmable Graphics

M5 gives the agent the graphics layer itself, inside the sandbox. The agent declares buffers, textures and state that persists between frames, and it imports live model tensors read-only with no copy. It writes its own vertex, fragment and compute shaders in WGSL, with its own blending and depth, and it composes compute, render and post passes. The output draws into a panel with the camera given, and an ID buffer keeps picking. Caliper validates each program before it runs and holds it inside its budgets.

The demo is an attention arc tower that the agent writes completely. A small transformer trains, and a compute pass turns each attention weight into an arc between tokens on a ring for each layer. A render pass draws the arcs with additive blending, and a bloom pass adds the glow. The arcs show when induction heads form, and an ablation and a fork test whether they cause the drop in loss.

## Built-In Primitives

The built-in primitives stay as the standard library and as the Python logging API, `caliper.scene`. They give a script one call for a common visual, with picking, labels and cameras included. They get no new features. Over time they move onto the programmable layer, so that one rendering path remains, and an agent can copy the program of a primitive and change it.

## Later Milestones

M6 moves every built-in shape and the `surface` and `path` operators onto the programmable layer as library pieces, moves the earlier demos onto them, and removes the per-shape drawing code, so that one rendering path remains. M7 adds projections and layouts of high-dimensional data, overlays and model probes as library pieces. M8 adds steering, scaling and patching, the core of interpretability work. M9 to M16 complete saved and editable instruments, browser parity, the two acceptance demos, image data, vector drawing, recordings and WebAssembly views.
