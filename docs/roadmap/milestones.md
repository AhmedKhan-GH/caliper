# Milestones

This page lists the milestones of Caliper's current trajectory, Agent-Built Playground, and the state of each one.

## Progress

| Milestone | Scope | Status | Pull Request |
|---|---|---|---|
| M0, Reconciled Base | Graphics layer, scenes, protocol 3.2, picking | Done | [#24](https://github.com/AhmedKhan-GH/caliper/pull/24) |
| M1, Agent-Built Instruments | Scene panels, controls, run control and events over MCP | Done | [#25](https://github.com/AhmedKhan-GH/caliper/pull/25) |
| M2, GPU Programs | Agent-written WGSL over live tensors, stages, sandbox, first ablation | Done | [#27](https://github.com/AhmedKhan-GH/caliper/pull/27) |
| M3, Instrument Building | Entities from tensors, operators, shared state, time transport, annotations, all control kinds | In progress | |
| M4, Hosted Runs | Start, fork and stop a model's run from Caliper and over MCP | Done | [#26](https://github.com/AhmedKhan-GH/caliper/pull/26) |
| M5, Rendering Essentials | Transparency, thick lines, ribbons and tubes, instancing, glow, label placement | Next | |
| M6, GPU Passes | Programs with several passes and state that persists between frames | Planned | |
| M7, Derivation Catalog | Projections and layouts on the GPU, overlays, model probes | Planned | |
| M8, Steering and Intervention | Steer, scale and patch | Planned | |
| M9, Saved and Editable Instruments | Saved instruments and the node editor | Planned | |
| M10, Browser Parity | Saved instruments in the browser | Planned | |
| M11, Training Debugger | First acceptance demo | Planned | |
| M12, Image Sequences and Training Data | Image grids, sequences, augmentation preview | Planned | |
| M13, World-Model Playground | Second acceptance demo | Planned | |
| M14, Vector Drawing and Primitive Completion | 2-D vector drawing and the remaining primitives | Planned | |
| M15, Time Axis | Recordings and timelines | Deferred | |
| M16, WebAssembly Views | Agent-written interaction logic | Deferred | |

## Order

The milestones up to M4 ran in the order M0, M1, M4, M2, M3. Hosted runs came before GPU programs, so that each later demo starts its own run. From M5 on, the numbers follow the order of work. Visual depth comes first: rendering, then GPU passes, then the derivation catalog. Steering and intervention follow, and they can run beside the derivation catalog.

Each milestone ends in a state that works. A milestone is done when the release build is clean, the suites and the browser checks pass, a demo script runs without a person, the docs are written, and a short check shows the result.

## Agent-Built Instruments

M1 lets an agent build a live instrument on a running script through MCP alone. The script needs one line, `caliper.watch(model, optimizer)`. The agent adds a scene panel and controls, and a slider changes the learning rate at the next step after an approval card. A click in the scene returns to the agent as an event.

## Hosted Runs

M4 lets a person or an agent start a model's run from Caliper, with no script of their own. A run can be forked at a checkpoint with a changed parameter, and the two runs show side by side. A review card states exactly what runs before it starts.

## GPU Programs

M2 lets an agent write WGSL programs that read live training tensors on the training GPU with no copy. A program writes geometry for a scene entity or pixels for a panel. A sandbox keeps model tensors read-only, stops a program at its time limit and holds it inside a memory budget. Programs run every N steps, so training keeps its speed. The first intervention, an ablation of a module output through a toggle, came in the same milestone.

## Instrument Building

M3 lets an agent turn any tensor of the running model into a visual with no scene code in the script. The `surface` and `path` operators derive geometry on the GPU. Shared state connects the views: a pick in one panel changes another, and a time transport moves one cursor through the recent steps. Annotations, bound text readouts, segmented groups and every control kind complete the interface.

## Rendering Essentials

M5 makes dense visuals legible. It adds transparency and blending, thick lines with ribbons and tubes, mesh instancing, a glow pass and label placement without overlap, and it applies the scene options for camera, axes and entity filters. Its demo is an attention arc tower. A small transformer trains, and each attention weight becomes an arc between tokens on a ring for each layer. The arcs show when induction heads form, and an ablation and a fork test whether they cause the drop in loss.

## Later Milestones

M6 adds GPU programs with several passes and persistent state, for particles, flows and iterative algorithms. M7 builds on them with projections and layouts of high-dimensional data, overlays and model probes. M8 adds steering, scaling and patching, the core of interpretability work. M9 to M16 complete saved and editable instruments, browser parity, the two acceptance demos, image data, vector drawing, recordings and WebAssembly views.
