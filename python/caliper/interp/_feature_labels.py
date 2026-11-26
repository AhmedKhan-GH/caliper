"""Feature labels for a pruned graph before identity and publication."""
from ._graph import LARGE_NODE_CAP, bounded_integer

MAX_LABEL_BATCH = 512


def label_graph_nodes(nodes, index=None, *, check=None):
    """Copy display records and add feature labels without a partial graph update."""
    bounded_integer(len(nodes), "label nodes", 0, LARGE_NODE_CAP)
    if check is not None:
        check("graph labels admission")
    labels = {}
    if index is not None:
        ranked_pairs = []
        seen = set()
        feature_nodes = sorted(
            (n for n in nodes if n["kind"] == "feature"),
            key=lambda n: (-float(n.get("influence", 0.0)), n["layer"], n["feature"]),
        )
        for node in feature_nodes:
            key = (node["layer"], node["feature"])
            if key not in seen:
                seen.add(key)
                ranked_pairs.append(key)
                if len(ranked_pairs) == MAX_LABEL_BATCH:
                    break
        if ranked_pairs:
            labels = index.labels(ranked_pairs, check=check)
    result = []
    for node in nodes:
        record = dict(node)
        if record["kind"] == "feature":
            key = (record["layer"], record["feature"])
            record["label"] = labels.get(key, f"L{key[0]} F{key[1]}")
        result.append(record)
    if check is not None:
        check("graph labels complete")
    return result
