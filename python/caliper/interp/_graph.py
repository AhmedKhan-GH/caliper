"""Bounded attribution graph records and fixed device tensor publication."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import torch

from ._labels import display_text, pack_labels, token_label


NODE_CAP = 512
EDGE_CAP = 8192
LARGE_NODE_CAP = 8192
LARGE_EDGE_CAP = 262144
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
    logits: list[dict] | None = field(default=None, repr=False)
    settings: dict = field(default_factory=dict)
    _memory: object = field(default=None, repr=False, compare=False)

    @classmethod
    def build(cls, *, nodes, edges, tokens, layers, completeness, omitted_influence,
              error_share, feature_share, trace_id, device, logits=None, settings=None, memory=None):
        settings = {} if settings is None else dict(settings)
        large = len(nodes) > NODE_CAP or len(edges) > EDGE_CAP
        node_cap = LARGE_NODE_CAP if large else NODE_CAP
        edge_cap = LARGE_EDGE_CAP if large else EDGE_CAP
        bounded_integer(len(nodes), "nodes", 0, node_cap)
        bounded_integer(len(edges), "edges", 0, edge_cap)
        bounded_integer(len(tokens), "tokens", 1, TOKEN_CAP)
        counts = torch.tensor([len(nodes), len(edges), len(tokens), layers], dtype=torch.int32, device=device)
        node_i = torch.zeros((node_cap, 4), dtype=torch.int32, device=device)
        node_f = torch.zeros((node_cap, 4), dtype=torch.float32, device=device)
        edge_i = torch.zeros((edge_cap, 2), dtype=torch.int32, device=device)
        edge_f = torch.zeros((edge_cap, 2), dtype=torch.float32, device=device)
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
                   feature_share, trace_id, tensors, logits=logits, settings=settings, _memory=memory)

    @classmethod
    def from_circuit_tracer(
        cls,
        ct_graph,
        pruned,
        *,
        replacement_score: float,
        completeness_score: float,
        node_influence: torch.Tensor,
        tokenizer=None,
        device="cpu",
        trace_id: str = "trace-1",
        max_nodes: int = LARGE_NODE_CAP,
        max_edges: int = LARGE_EDGE_CAP,
        feature_labels: dict[tuple[int, int], str] | None = None,
        settings: dict | None = None,
        memory=None,
    ) -> "Graph":
        """Convert a CPU-resident ``circuit_tracer.Graph`` and ``PruneResult`` into a Caliper ``Graph``."""
        bounded_integer(max_nodes, "max_nodes", 1, LARGE_NODE_CAP)
        bounded_integer(max_edges, "max_edges", 0, LARGE_EDGE_CAP)
        adj = ct_graph.adjacency_matrix.detach().cpu().float()
        input_ids = ct_graph.input_tokens.detach().cpu().tolist()
        n_tokens = len(input_ids)
        n_layers = int(ct_graph.cfg.n_layers)
        n_logits = len(ct_graph.logit_targets)
        selected_coords = ct_graph.active_features[ct_graph.selected_features].detach().cpu().tolist()
        act_vals = ct_graph.activation_values.detach().cpu().float().tolist()
        n_features = len(selected_coords)
        full_nodes = int(adj.shape[0])
        infl = node_influence.detach().cpu().float()
        cum_scores = pruned.cumulative_scores.detach().cpu().float()

        tokens = [
            {
                "position": pos,
                "token_id": int(tok_id),
                "text": token_label(tokenizer, int(tok_id)) if tokenizer is not None else f"#{int(tok_id)}",
            }
            for pos, tok_id in enumerate(input_ids)
        ]

        logit_probs = ct_graph.logit_probabilities.detach().cpu().float().tolist()
        all_nodes: list[dict] = []
        for idx, ((layer, pos, feat), act) in enumerate(zip(selected_coords, act_vals)):
            layer_i, pos_i, feat_i = int(layer), int(pos), int(feat)
            node_id = f"feature:{layer_i}:{feat_i}:{pos_i}"
            lbl = (feature_labels or {}).get((layer_i, feat_i))
            rec = {
                "id": node_id,
                "kind": "feature",
                "layer": layer_i,
                "position": pos_i,
                "feature": feat_i,
                "activation": float(act),
            }
            if lbl is not None:
                rec["label"] = lbl
            all_nodes.append(rec)

        for layer_i in range(n_layers):
            for pos_i in range(n_tokens):
                all_nodes.append({
                    "id": f"error:{layer_i}:{pos_i}",
                    "kind": "error",
                    "layer": layer_i,
                    "position": pos_i,
                    "activation": 0.0,
                })

        for tok in tokens:
            pos_i, tok_id = tok["position"], tok["token_id"]
            all_nodes.append({
                "id": f"token:{pos_i}:{tok_id}",
                "kind": "token",
                "layer": -1,
                "position": pos_i,
                "token_id": tok_id,
                "label": tok["text"],
                "activation": 0.0,
            })

        logit_records: list[dict] = []
        for lt, prob in zip(ct_graph.logit_targets, logit_probs):
            tok_id = int(lt.vocab_idx)
            if tokenizer is not None and 0 <= tok_id < int(ct_graph.vocab_size):
                lbl = token_label(tokenizer, tok_id)
            else:
                lbl = display_text(lt.token_str)
            rec = {
                "id": f"logit:{n_tokens - 1}:{tok_id}",
                "kind": "logit",
                "layer": n_layers,
                "position": n_tokens - 1,
                "token_id": tok_id,
                "probability": max(0.0, min(1.0, float(prob))),
                "label": lbl,
                "activation": float(prob),
            }
            logit_records.append(rec)
            all_nodes.append(rec)

        keep_mask = pruned.node_mask.detach().cpu().bool().clone()
        raw_kept_nodes = int(keep_mask.sum().item())
        raw_kept_features = int(keep_mask[:n_features].sum().item())
        edge_mask = pruned.edge_mask.detach().cpu().bool() & (adj != 0) & keep_mask[:, None] & keep_mask[None, :]
        raw_kept_edges = int(edge_mask.sum().item())

        if int(keep_mask.sum().item()) > max_nodes:
            # Retain the top logit first, then rank remaining kept nodes by influence.
            kept_indices = keep_mask.nonzero().flatten().tolist()
            top_logit_row = full_nodes - n_logits if n_logits > 0 else None
            ranked = sorted(
                kept_indices,
                key=lambda i: (0 if i == top_logit_row else 1, -float(infl[i].item()), i),
            )[:max_nodes]
            keep_mask.zero_()
            keep_mask[ranked] = True
            edge_mask = edge_mask & keep_mask[:, None] & keep_mask[None, :]

        order = keep_mask.nonzero().flatten().tolist()
        row_map = {old: new for new, old in enumerate(order)}
        top_row = (full_nodes - n_logits) if n_logits > 0 else None

        nodes: list[dict] = []
        for old_i in order:
            effect = float(adj[top_row, old_i].item()) if top_row is not None else 0.0
            nodes.append({
                **all_nodes[old_i],
                "index": row_map[old_i],
                "influence": float(infl[old_i].item()),
                "cumulative_influence": float(cum_scores[old_i].item()),
                "effect": effect,
            })

        endpoints = edge_mask.nonzero().tolist()
        weights = adj[edge_mask].tolist()
        if len(endpoints) > max_edges:
            if max_edges == 0:
                endpoints, weights = [], []
            else:
                paired = sorted(
                    zip(endpoints, weights),
                    key=lambda item: (-abs(item[1]), item[0][0], item[0][1]),
                )[:max_edges]
                endpoints = [p[0] for p in paired]
                weights = [p[1] for p in paired]

        edges = [
            {"source": row_map[src], "target": row_map[tgt], "weight": float(w)}
            for (tgt, src), w in zip(endpoints, weights)
        ]

        total_infl = float(infl.sum().item())
        kept_infl = float(infl[keep_mask].sum().item()) if total_infl > 0 else 0.0
        omitted_influence = max(0.0, 1.0 - kept_infl / total_infl) if total_infl > 0 else 0.0
        feat_infl = float(infl[:n_features].sum().item())
        err_infl = float(infl[n_features:n_features + n_layers * n_tokens].sum().item())
        feature_share = feat_infl / total_infl if total_infl > 0 else 0.0
        error_share = err_infl / total_infl if total_infl > 0 else 0.0

        rep_f = float(replacement_score) if math.isfinite(replacement_score) else 0.0
        comp_f = float(completeness_score) if math.isfinite(completeness_score) else 0.0
        completeness = {
            "replacement_score": rep_f,
            "completeness_score": comp_f,
            "explained_fraction": comp_f,
            "full_nodes": full_nodes,
            "kept_nodes": raw_kept_nodes,
            "kept_features": raw_kept_features,
            "kept_nonzero_edges": raw_kept_edges,
        }
        return cls.build(
            nodes=nodes,
            edges=edges,
            tokens=tokens,
            layers=n_layers,
            completeness=completeness,
            omitted_influence=omitted_influence,
            error_share=error_share,
            feature_share=feature_share,
            trace_id=trace_id,
            device=device,
            logits=logit_records,
            settings=settings,
            memory=memory,
        )

    def _panel_tensors(self):
        """Prepare labels and traced probabilities on the structural tensors' device."""
        device = self.tensors["trace.counts"].device
        node_cap = int(self.tensors["trace.node_i"].shape[0])
        labels = [token["text"] for token in self.tokens]
        labels.extend([""] * (TOKEN_CAP - len(labels)))
        labels.extend(node["id"] if node.get("label") is None else node["label"] for node in self.nodes)
        logits = self.logits if self.logits is not None else [n for n in self.nodes if n["kind"] == "logit"]
        logits = logits[:10]
        probabilities = [float(node["probability"]) for node in logits]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Traced probabilities must be finite values from zero to one.")
        probs = torch.zeros((2, 10), dtype=torch.float32)
        tokens = torch.full((10,), -1, dtype=torch.int32)
        if logits:
            probs[:, :len(logits)] = torch.tensor(probabilities, dtype=torch.float32)
            tokens[:len(logits)] = torch.tensor([node["token_id"] for node in logits], dtype=torch.int32)
        values = {
            "trace.label_text": pack_labels(labels, TOKEN_CAP + node_cap),
            "trace.probs": probs,
            "trace.logit_tokens": tokens,
            "trace.logit_label_text": pack_labels([node["label"] for node in logits], 10),
        }
        return {name: value.to(device) for name, value in values.items()}

    def publish(self):
        """Publish structural tensors and all graph and bar inputs on their existing device."""
        from ._publish import _publication_scope, _publish_many
        if self._memory is not None and hasattr(self._memory, "reserve"):
            self._memory.reserve("graph panel publication", self.tensors["trace.node_i"].shape[0] * 1024 + 65536)
        with _publication_scope():
            values = {**self.tensors, **self._panel_tensors()}
            _publish_many(values)
        return self

    def read(self, *, node_offset=0, node_limit=128, edge_offset=0, edge_limit=256):
        """Return bounded JSON records, totals, and completeness summary."""
        node_cap = int(self.tensors["trace.node_i"].shape[0])
        edge_cap = int(self.tensors["trace.edge_i"].shape[0])
        bounded_integer(node_offset, "node_offset", 0, node_cap)
        bounded_integer(edge_offset, "edge_offset", 0, edge_cap)
        bounded_integer(node_limit, "node_limit", 0, 256)
        bounded_integer(edge_limit, "edge_limit", 0, 256)
        end_n = min(node_offset + node_limit, len(self.nodes))
        end_e = min(edge_offset + edge_limit, len(self.edges))
        summary = {k: v for k, v in self.completeness.items() if k != "targets"}
        out = {
            "trace_id": self.trace_id,
            "status": self.status,
            "counts": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "tokens": len(self.tokens),
                "layers": self.layers,
            },
            "nodes": self.nodes[node_offset:end_n],
            "edges": self.edges[edge_offset:end_e],
            "tokens": self.tokens,
            "completeness": summary,
            "omitted_influence": self.omitted_influence,
            "error_share": self.error_share,
            "feature_share": self.feature_share,
            "next_node_offset": end_n if node_limit and end_n < len(self.nodes) else None,
            "next_edge_offset": end_e if edge_limit and end_e < len(self.edges) else None,
        }
        if self.settings:
            out["settings"] = dict(self.settings)
        return out

    def to_dict(self, **paging):
        return self.read(**paging)

    def to_json(self, **paging):
        result = self.read(**paging)
        if not paging:
            result.update(
                nodes=self.nodes,
                edges=self.edges,
                completeness=self.completeness,
                logits=self.logits,
                next_node_offset=None,
                next_edge_offset=None,
            )
        return json.dumps(result, allow_nan=False, separators=(",", ":"))
