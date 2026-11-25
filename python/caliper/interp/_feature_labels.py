"""Feature labels for a pruned graph before identity and publication."""
from ._graph import NODE_CAP, REFERENCE_NODE_CAP, bounded_integer
from ._index import LABEL_FEATURES
from ._reference_graph import PROFILE


def label_graph_nodes(nodes, index=None, *, profile="caliper", check=None):
    """Copy display records and add feature labels without a partial graph update.

    Call this after pruning and before the trace identity, cache, and publication.
    The caller must bind an index with the same model, tokenizer, and dictionary.
    A standalone graph without an index retains explicit layer and feature IDs.
    The profile selects the graph capacity. Index requests keep their own bound.
    """
    if profile not in ("caliper", PROFILE):
        raise ValueError("unsupported graph label profile")
    bounded_integer(len(nodes), "label nodes", 0, REFERENCE_NODE_CAP if profile == PROFILE else NODE_CAP)
    if check is not None:
        check("graph labels admission")
    labels = {}
    if index is not None:
        features = list(dict.fromkeys((node["layer"], node["feature"])
                                      for node in nodes if node["kind"] == "feature"))
        # Preserve index validation for an empty request and retain labels across cache eviction.
        for start in range(0, max(1, len(features)), LABEL_FEATURES):
            if check is not None:
                check("graph label batch admission")
            labels.update(index.labels(features[start:start + LABEL_FEATURES], check=check))
            if check is not None:
                check("graph label batch complete")
    result = []
    for node in nodes:
        record = dict(node)
        if record["kind"] == "feature":
            key = record["layer"], record["feature"]
            record["label"] = labels[key] if index is not None else f"L{key[0]} F{key[1]}"
        result.append(record)
    if check is not None:
        check("graph labels complete")
    return result
