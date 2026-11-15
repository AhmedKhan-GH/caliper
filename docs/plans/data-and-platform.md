# Data and the platform

This page draws the line between Caliper, the tool, and a platform that would come later as a separate product. It says what data the tool keeps, which feature sits on the line, and what the platform would add. The [roadmap](roadmap.md#data-and-the-platform) gives the order and status.

## The line

Caliper is a tool for one person on one machine: live views, applets, interventions and streaming. Some features need more than one machine can give them. Run history across many runs, comparing runs, shared and hosted storage, accounts and teams, and hosted compute sessions all need a database, a server and accounts. That makes them a platform, a separate product that comes later. The site names it in two parts: Caliper Cloud, which runs Caliper on hosted GPU nodes for models larger than local hardware can hold, and Caliper Platform, a place to exchange widgets and applets and to find ready-made tools.

The tool works fully without the platform. It never requires an account or a server, and it stays local-first. The platform is additive: it consumes the same live protocol and the same recording format as the tool, so nothing in the tool changes shape for it.

The registry of [Phase 4](roadmap.md#phase-4-the-registry) stays on the tool's side of the line. It publishes by pull request, needs no account, and never gates what the host loads, so sideloading keeps working without it ([Applet registry](registry.md)).

## What the tool keeps today

Everything the tool stores stays on the machine, under the app-data folder:

- The metrics store, behind [`caliper.metrics.v1`](../references/service-metrics-v1.md): scalars, histograms and images per run and step, and SQL queries over them.
- The artifacts store, behind [`caliper.artifacts.v1`](../references/service-artifacts-v1.md): content-addressed files such as checkpoints.
- The data service, [`caliper.data.v1`](../references/service-data-v1.md): SQL in and Arrow out, on an embedded DuckDB.
- Each applet's data folder, `data/<id>/`, which the applet owns and versions with its data schema ([The version of each applet's stored data](../explanations/compatibility.md#the-version-of-each-applets-stored-data)).

## On the line

Recording a session to a file and replaying it later, with no server, would let a session be reopened or shared as a file. It is proposed for the tool, and it is undecided: the tool could build it, or it could wait for the platform. The roadmap lists it under [Undecided](roadmap.md#undecided).

## The platform

The platform is planned in outline only. It would add:

- run history across many runs, and comparison between runs;
- shared storage;
- accounts and permissions;
- hosted sessions that reach a browser through the data streaming and the video streaming of [Streaming and embedding](streaming-and-embedding.md).

The platform lives outside this repository. Its design starts after the streaming work.

## Security

The platform's network and compute security get a dedicated review before anything is hosted. The streaming plan holds the matching rule for sessions that leave the machine ([Security](streaming-and-embedding.md#security)).
