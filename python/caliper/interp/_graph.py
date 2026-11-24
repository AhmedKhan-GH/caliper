"""Bounded attribution graph records and fixed device tensor publication."""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import torch


NODE_CAP = 512
EDGE_CAP = 8192
TOKEN_CAP = 32
KINDS = {"token": 0, "feature": 1, "error": 2, "logit": 3}


def bounded_integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


@dataclass
class Graph:
    """Display graph with completeness from all sources before display pruning.

    Node ``id`` values identify semantic nodes across pruning. Edge endpoints
    are display row indices. ``read`` pages records without a device readback.
    ``tensors`` contains exactly the five names of the public trace contract.
    """

    nodes: list[dict]
    edges: list[dict]
    tokens: list[dict]
    layers: int
    completeness: dict
    omitted_influence: float
    error_share: float
    feature_share: float
    trace_id: str
    tensors: dict = field(repr=False)
    status: str = "done"

    @classmethod
    def build(cls, *, nodes, edges, tokens, layers, completeness, omitted_influence,
              error_share, feature_share, trace_id, device):
        bounded_integer(len(nodes), "nodes", 0, NODE_CAP)
        bounded_integer(len(edges), "edges", 0, EDGE_CAP)
        bounded_integer(len(tokens), "tokens", 1, TOKEN_CAP)
        counts = torch.tensor([len(nodes), len(edges), len(tokens), layers], dtype=torch.int32, device=device)
        node_i = torch.zeros((NODE_CAP, 4), dtype=torch.int32, device=device)
        node_f = torch.zeros((NODE_CAP, 4), dtype=torch.float32, device=device)
        edge_i = torch.zeros((EDGE_CAP, 2), dtype=torch.int32, device=device)
        edge_f = torch.zeros((EDGE_CAP, 2), dtype=torch.float32, device=device)
        if nodes:
            node_i[:len(nodes)] = torch.tensor([
                [KINDS[n["kind"]], n["layer"], n["position"], n["feature"] if n["kind"] == "feature"
                 else n.get("token_id", 0)] for n in nodes], dtype=torch.int32, device=device)
            node_f[:len(nodes)] = torch.tensor([
                [n["activation"], n["influence"], n["effect"], 0] for n in nodes], dtype=torch.float32, device=device)
        if edges:
            largest = max(abs(e["weight"]) for e in edges)
            edge_i[:len(edges)] = torch.tensor([[e["source"], e["target"]] for e in edges],
                                             dtype=torch.int32, device=device)
            edge_f[:len(edges)] = torch.tensor([[e["weight"], e["weight"] / largest if largest else 0]
                                              for e in edges], dtype=torch.float32, device=device)
        tensors = dict(zip(("trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f"),
                           (counts, node_i, node_f, edge_i, edge_f)))
        return cls(nodes, edges, tokens, layers, completeness, omitted_influence, error_share,
                   feature_share, trace_id, tensors)

    def publish(self):
        """Replace all five named tensors on their existing device."""
        from . import publish
        for name, tensor in self.tensors.items():
            publish(name, tensor)
        return self

    def read(self, *, node_offset=0, node_limit=128, edge_offset=0, edge_limit=256):
        """Return bounded JSON records, totals, and completeness summary."""
        bounded_integer(node_offset, "node_offset", 0, NODE_CAP)
        bounded_integer(edge_offset, "edge_offset", 0, EDGE_CAP)
        bounded_integer(node_limit, "node_limit", 0, 256)
        bounded_integer(edge_limit, "edge_limit", 0, 256)
        end_n = min(node_offset + node_limit, len(self.nodes))
        end_e = min(edge_offset + edge_limit, len(self.edges))
        summary = {k: v for k, v in self.completeness.items() if k != "targets"}
        return {"trace_id": self.trace_id, "status": self.status,
                "counts": {"nodes": len(self.nodes), "edges": len(self.edges),
                           "tokens": len(self.tokens), "layers": self.layers},
                "nodes": self.nodes[node_offset:end_n], "edges": self.edges[edge_offset:end_e],
                "tokens": self.tokens, "completeness": summary,
                "omitted_influence": self.omitted_influence,
                "error_share": self.error_share, "feature_share": self.feature_share,
                "next_node_offset": end_n if node_limit and end_n < len(self.nodes) else None,
                "next_edge_offset": end_e if edge_limit and end_e < len(self.edges) else None}

    def to_dict(self, **paging):
        return self.read(**paging)

    def to_json(self, **paging):
        return json.dumps(self.read(**paging), allow_nan=False, separators=(",", ":"))
