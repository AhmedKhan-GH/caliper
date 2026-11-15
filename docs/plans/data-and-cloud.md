# Data and Cloud

This page plans the data that Caliper keeps for the person who runs it: the history of their runs, comparison between runs, and recordings of sessions that can be replayed. It also draws the line between the desktop app, which keeps all of this locally, and Caliper Cloud and Caliper Web, which would later offer the same online. The [roadmap](roadmap.md#data-and-cloud) gives the order and status.

## The line

Caliper is a desktop app with a local database. Run history, comparing runs, and recording and replaying a session all belong to the desktop app, and it stores them on the machine it runs on. The desktop app never requires an account or a server, and it stays local-first.

What moves work to Caliper Cloud and Caliper Web is being online: syncing and sharing the app's history, teams and permissions, hosted storage, and hosted compute. They are separate products that come later:

- **Caliper Cloud** offers the desktop app's history and recordings online and runs hosted compute sessions, for example for models larger than local hardware can hold.
- **Caliper Web** is the browser workspace.

Caliper Platform, the registry and exchange of applets and widgets, is a third product. Phase 4 of the roadmap plans it as the [Applet registry](registry.md). The desktop app can install from it but never needs it, because sideloading keeps working.

## What the desktop app keeps today

Everything the desktop app stores stays on the machine, under the app-data folder:

- The metrics store, behind [`caliper.metrics.v1`](../references/service-metrics-v1.md): scalars, histograms and images per run and step, and SQL queries over them.
- The artifacts store, behind [`caliper.artifacts.v1`](../references/service-artifacts-v1.md): content-addressed files such as checkpoints.
- The data service, [`caliper.data.v1`](../references/service-data-v1.md): SQL in and Arrow out, on an embedded DuckDB.
- Each applet's data folder, `data/<id>/`, which the applet owns and versions with its data schema ([The version of each applet's stored data](../explanations/compatibility.md#the-version-of-each-applets-stored-data)).

## Local history and recording

These are planned for the desktop app, after Phase 1 and after the streaming work. They build on the DuckDB-backed data service and on the metrics and artifacts stores, and they keep everything on the machine.

- **Run history.** The desktop app keeps the runs that the metrics and artifacts stores already record, and shows them as a history that outlives each session.
- **Comparing runs.** The desktop app shows two or more runs from that history side by side.
- **Recording and replay.** The desktop app records a session to a file and replays it later, with no server, so that a session can be reopened or shared as a file. The recording format shares its snapshots with the live protocol of [Streaming and embedding](streaming-and-embedding.md), which is why this work comes after the streaming work.

## Caliper Cloud and Caliper Web

Caliper Cloud and Caliper Web offer the same online, and they are planned in outline only:

- the desktop app's history, synced and shared;
- teams and permissions;
- hosted storage;
- hosted compute sessions that reach a browser through the data streaming of [Streaming and embedding](streaming-and-embedding.md), drawn by its web viewer.

Both are separate, later products, and they live outside this repository. They use the same database schema, the same recording format and the same live protocol as the desktop app, so the desktop app's data moves to them unchanged and the desktop app changes nothing for them.

## Security

The network and compute security of Caliper Cloud and Caliper Web get a dedicated review before anything is hosted. The streaming plan holds the matching rule for sessions that leave the machine ([Security](streaming-and-embedding.md#security)).
