# Platform candidates

A domain applet developed in a copy of this repository had to build several
pieces the host does not provide. Each piece is a candidate for the host or the
SDK. This page lists them, where each belongs, and the order to move them.

A piece moves only when a second applet needs it. Each move is tested by moving
the original applet and one existing scope onto the new host or SDK version.

Phase 2 starts by settling one more rule, which every promoted piece follows:
each applet splits into a computation that runs natively and an interface that
compiles to WebAssembly, and interface code may use only services that have a
wire form ([Applets in the browser](streaming-and-embedding.md#applets-in-the-browser)).
A piece that draws belongs to the interface, and a piece that computes stays
native. Applets built before the rule stay native-only until they are ported.

## Candidates

| Built in the applet | What it does | Where it belongs |
|---|---|---|
| Python sidecar over pipes | Starts a Python process beside the applet, with a version handshake, a cache version that triggers a rebuild, and a synthetic-data mode | A sidecar service in the catalog, generalising the "Python runs alongside" model |
| Layout grid | Sizes panes as shares of the window with minimums tied to the text scale, as plain rectangles that tests check without a window | `caliper.ui` and the `caliper::viz::` sugar layer |
| Tour overlay | Steps through regions of the applet from a data file, outlining each region | A host tour service: every applet registers its regions and ships a tour file |
| 3-D annotation scene | Arrows, rings, labels, layers and legends on geometry records | `caliper::viz::` |
| Colour by value | Colours points by a chosen value, with a scale and a key | `caliper::viz::` |
| Catalog tree | A taxonomy tree with values inline that opens to the selected item | `caliper::viz::` |
| Scripted runs and capture | Screenshots, a timed demo timeline and screen recording, driven by environment variables | A host automation and capture service (see below) |
| Text scale | One factor applied to fonts, metrics and pane minimums | A host setting in `caliper.ui` |

## Findings about existing services

- **`caliper.jobs.v1` needs a quiet job kind.** The applet moved a long-running
  background computation off `jobs.v1` onto its own thread, because the job's
  finished notification stayed on screen. A background job kind that dismisses
  itself would have let it stay on the service.
- **Screen capture belongs to the host.** Capture started from inside the
  applet failed, while the same capture started from outside it worked. A host
  service that owns capture, scripting and recording gives every applet
  screenshots for documentation and tests, and demo recordings, through one
  interface.
- **ImGui errors went unnoticed.** One pane produced a constant stream of ImGui
  runtime errors, and nothing showed them. Development builds of the host
  should.
- **Applets could not live outside the repository.** The applet needed a full
  copy of the repository and a patch. Applets now build against the installed
  SDK and load from development folders; see [SDK package](../references/sdk-package.md)
  and [Applets outside the core](applets-outside-the-core.md).

## Order

1. The sidecar service
2. The layout grid and the text scale
3. The tour service
4. Automation and capture
5. The `viz` components: annotation scene, colour by value, catalog tree
6. The quiet job kind in `jobs.v1`

Related: [UI principles](../explanations/ui-principles.md).
