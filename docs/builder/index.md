# Builder

The builder is the part of Caliper where you build an instrument by hand as a spec instead of writing C++ code. It is for anyone who wants to turn data into panels, scenes, controls and queries without compiling a library. The result is a spec, a text file of sources, panels, controls and a layout. The app draws a saved spec as an instrument (`kind = "spec"`).

## Builder Session

The app holds one builder session for a workspace. The workspace is a folder that holds loaders, specs, saved spec-based instruments, the command journal and a database. The session opens the workspace when something first needs it, so an app that never builds a spec creates no files. [Workspace](workspace.md) describes the session, the spec page and the command script.

Every change to a spec is a command. The builder checks the command against the spec format, records it in the journal and lets you undo it. The pane, the command script and the agents all use this one command layer. Thus a change looks the same in the journal, whoever made it.

## Spec Construction

You work on the spec page in build mode. You choose a source, which is a loaded file or a live stream, and a primitive, which is a kind of panel. Then you get a panel on the canvas. You arrange the panels, edit their properties in a side list and watch the spec text follow your changes. Each gesture is one undo step. [Builder Pane](builder-pane.md) describes each region of the pane and its keys.

```mermaid
flowchart LR
    files[Dropped files] -->|import| sources[Sources]
    sources -->|choose a primitive| panels[Panels]
    panels -->|arrange and edit| spec[Spec text]
    spec -->|save| instrument[Saved instrument]
    instrument -->|open in browser| viewer[Browser viewer]
```

## Saved Specs

A save writes the instrument into the workspace's `instruments/` folder. The instrument then appears under Projects on the landing screen. An instrument written as a spec has no library, so the host never loads code for it. The spec stays small and the loader checks it before it reads it.

You can also open a saved instrument in a browser. The browser viewer shows the spec without the builder pane, so it suits a quick look or a shared screen. The viewer runs through Caliper API, which listens on the loopback address only.

The builder is also the surface that the [agents](../agents/index.md) work on. An agent edits a spec through the same commands, and each change waits for your review unless you turn review off.
