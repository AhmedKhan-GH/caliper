"""Feature labels for a pruned graph before identity and publication."""
from ._graph import NODE_CAP, bounded_integer


def label_graph_nodes(nodes, index=None, *, check=None):
    """Copy display records and add feature labels without a partial graph update.

    Call this after pruning and before the trace identity, cache, and publication.
    The caller must bind an index with the same model, tokenizer, and dictionary.
    A standalone graph without an index retains explicit layer and feature IDs.
    """
    bounded_integer(len(nodes), "label nodes", 0, NODE_CAP)
    features = [(node["layer"], node["feature"]) for node in nodes if node["kind"] == "feature"]
    labels = index.labels(features, check=check) if index is not None else {}
    result = []
    for node in nodes:
        record = dict(node)
        if record["kind"] == "feature":
            key = record["layer"], record["feature"]
            record["label"] = labels[key] if index is not None else f"L{key[0]} F{key[1]}"
        result.append(record)
    return result
