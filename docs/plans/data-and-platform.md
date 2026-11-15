# Data and the platform

This page plans the data that Caliper keeps for the person who runs it: the history of their runs, comparison between runs, and recordings of sessions that can be replayed. It also draws the line between the desktop app, which keeps all of this locally, and a platform that would later offer the same online. The [roadmap](roadmap.md#data-and-the-platform) gives the order and status.

## The line

Caliper is a desktop app with a local database. Run history, comparing runs, and recording and replaying a session all belong to the desktop app, and it stores them on the machine it runs on. The desktop app never requires an account or a server, and it stays local-first.

What makes something platform work is being online: syncing and sharing the app's history, teams and permissions, hosted storage, and hosted compute. That is a separate product that comes later. The site names it in two parts: Caliper Cloud, which runs Caliper on hosted GPU nodes for models larger than local hardware can hold, and Caliper Platform, a place to exchange widgets and applets and to find ready-made tools.

The registry of [Phase 4](roadmap.md#phase-4-the-registry) is planned on its own page, [Applet registry](registry.md). The desktop app can install from it but never needs it, because sideloading keeps working.

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

## The platform

The platform offers the same online, and it is planned in outline only:

- the desktop app's history, synced and shared;
- teams and permissions;
- hosted storage;
- hosted compute sessions that reach a browser through the data streaming and the video streaming of [Streaming and embedding](streaming-and-embedding.md).

The platform is a separate, later product, and it lives outside this repository. It uses the same database schema, the same recording format and the same live protocol as the desktop app, so the desktop app's data moves to it unchanged and the desktop app changes nothing for it.

## Security

The platform's network and compute security get a dedicated review before anything is hosted. The streaming plan holds the matching rule for sessions that leave the machine ([Security](streaming-and-embedding.md#security)).
