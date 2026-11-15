# D18. data.v1 and artifacts.v1 built for EmbedScope

| Date | Status |
|---|---|
| 2026-07-02 | Accepted |

## Decision

The two services that [D16](d016-gpt-scope-is-the-flagship.md) held back, `artifacts.v1` and `data.v1`, are built for their first real consumer, EmbedScope, a data-driven exemplar with a 3D embedding projector drawn through ImPlot3D. `artifacts.v1` reloads the model; `data.v1` queries the learned embedding table for centroids, nearest neighbours and misclassified points.

## Reason

Demand had appeared (checkpoint reload and tabular embeddings), so freezing the headers no longer broke the rule against inventing designs. ImPlot3D gives a 3D view that works on every renderer. Drawing 3D through raw OpenGL was rejected because it works only on the OpenGL backend, against [D13](d013-renderer-agnostic-abi.md).

## Notes

The code implements this decision: the `embed_scope` applet, `artifacts_v1.h`, `data_v1.h` and the host's artifact and data stores exist.
