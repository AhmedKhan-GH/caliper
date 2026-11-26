"""End-to-end tests for the TransformerLens, circuit-tracer, and SAELens backend adapter."""
from __future__ import annotations

import pytest
import torch
from transformer_lens import patching as tl_patching

from caliper._ablate import InterventionExecutor, points
from caliper.interp._backend import (
    default_device,
    load_backend,
    make_tiny_backend,
    make_tiny_sae_backend,
)


@pytest.mark.parametrize("device", ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])])
def test_tiny_replacement_backend_trace_intervene_generate_and_patch(device):
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    backend = make_tiny_backend(device=device, dtype=dtype)
    assert backend.is_replacement_model
    assert backend.dictionary.metadata["layers"] == 2
    assert backend.dictionary.metadata["latents"] == 32

    pts = {p["point"] for p in points(backend.model)}
    assert {"layers.0.resid_pre", "layers.0.mlp_out", "layers.1.attn_out", "blocks.0.attn#0"} <= pts

    # Dense and sparse activations
    logits, dense = backend.activations("AB", sparse=False)
    _, sparse = backend.activations("AB", sparse=True)
    assert logits.shape == (1, 3, 128)
    assert dense.shape == (2, 3, 32)
    assert sparse.is_sparse and int(sparse._nnz()) > 0

    # Local top-example min-heap windows
    active_feat = int(dense[0, 2].argmax().item())
    assert float(dense[0, 2, active_feat].item()) > 0.0
    examples = backend.top_examples(0, active_feat, prompts=["AB", "CD", "ABE"], limit=2)
    assert 1 <= len(examples) <= 2
    assert len(examples[0]["token_ids"]) == 33
    assert examples[0]["center"] == 16
    assert examples[0]["activation"] > 0

    # Single and group interventions via circuit-tracer ReplacementModel.feature_intervention
    iv_single, _ = backend.intervene_logits("AB", [(0, 2, active_feat, 0.0)])
    iv_group, _ = backend.intervene_logits(
        "AB",
        [
            {"op": "scale", "layer": 0, "positions": [2], "feature": active_feat, "multiplier": -2.0},
            {"op": "clamp", "layer": 1, "positions": [2], "feature": 1, "value": 3.0},
        ],
    )
    assert not torch.equal(iv_single, logits)
    assert not torch.equal(iv_group, iv_single)

    # Generation preserves permanent hooks on ReplacementModel
    perm_before = {
        name: len(hp.fwd_hooks)
        for name, hp in backend.model.hook_dict.items()
        if hp.fwd_hooks
    }
    assert perm_before, "ReplacementModel must have permanent hooks installed"
    gen = backend.generate("AB", [(0, 2, active_feat, 4.0)], max_new_tokens=6)
    perm_after = {
        name: len(hp.fwd_hooks)
        for name, hp in backend.model.hook_dict.items()
        if hp.fwd_hooks
    }
    assert perm_after == perm_before
    assert len(gen["token_ids"][0]) == 6 and len(gen["token_ids"][1]) == 6

    # Batched activation patching matches transformer_lens.patching.get_act_patch_resid_pre
    clean_ids = backend.tokenize("ABCD")
    corrupt_ids = backend.tokenize("EFGH")
    pos_tok, neg_tok = 40, 41
    _, clean_cache = backend.model.run_with_cache(
        clean_ids, prepend_bos=False, names_filter=lambda n: n.endswith("hook_resid_pre")
    )
    corrupt_logits = backend.forward_logits(corrupt_ids)
    corrupt_m = float((corrupt_logits[0, -1, pos_tok].float() - corrupt_logits[0, -1, neg_tok].float()).item())
    clean_logits = backend.forward_logits(clean_ids)
    clean_m = float((clean_logits[0, -1, pos_tok].float() - clean_logits[0, -1, neg_tok].float()).item())

    def norm_metric(lgt: torch.Tensor) -> torch.Tensor:
        raw = (lgt[:, -1, pos_tok].float() - lgt[:, -1, neg_tok].float()).mean()
        return (raw - corrupt_m) / (clean_m - corrupt_m)

    tl_grid = tl_patching.get_act_patch_resid_pre(backend.model, corrupt_ids, clean_cache, norm_metric).cpu()
    sweep = backend.patch_sweep(clean_ids, corrupt_ids, positive_token=pos_tok, negative_token=neg_tok, batch_size=3)
    assert sweep["grid"].shape == (2, 5)
    assert torch.allclose(sweep["grid"].cpu(), tl_grid.float(), atol=5e-3, rtol=5e-3)

    # Attribution + CPU pruning builds a valid Caliper Graph
    graph = backend.attribute_and_prune(
        "AB", max_n_logits=4, max_feature_nodes=32, batch_size=8, node_threshold=0.8, edge_threshold=0.98
    )
    assert graph.layers == 2
    assert len(graph.tokens) == 3
    assert len(graph.nodes) >= 3
    assert set(graph.tensors) == {"trace.counts", "trace.node_i", "trace.node_f", "trace.edge_i", "trace.edge_f"}
    assert 0.0 <= graph.completeness["completeness_score"] <= 1.0


def test_standalone_sae_backend_delta_identity_and_live_executor_hooks():
    backend = make_tiny_sae_backend(device="cpu", dtype=torch.float32, layer=0)
    assert not backend.is_replacement_model
    logits, acts = backend.activations("ABC", sparse=False)
    top_feat = int(acts[0, -1].argmax().item())
    top_val = float(acts[0, -1, top_feat].item())
    assert top_val > 0.0

    # Delta-form clamp to clean activation has exact 0.0 identity error
    id_logits, _ = backend.intervene_logits("ABC", [(0, 3, top_feat, top_val)])
    assert float((id_logits - logits).abs().max().item()) == 0.0

    # Live InterventionExecutor clamp + scale hooks with row selection and last-edit-wins
    executor = InterventionExecutor(backend.model)
    executor.configure(dictionary=backend.dictionary, tokenizer=backend.tokenizer, backend=backend)
    ids = backend.tokenize("ABC")
    base = backend.forward_logits(ids)

    h_id = executor.install({"op": "clamp", "point": "blocks.0.hook_resid_pre", "feature": top_feat,
                             "positions": [3], "value": top_val})
    assert float((backend.forward_logits(ids) - base).abs().max().item()) == 0.0
    h_id()

    h_scale = executor.install({"op": "scale", "point": "blocks.0.hook_resid_pre", "feature": top_feat,
                                "positions": [3], "multiplier": -2.0})
    h_override = executor.install({"op": "clamp", "point": "blocks.0.hook_resid_pre", "feature": top_feat,
                                   "positions": [3], "value": 5.0})
    paired = ids.repeat(2, 1)
    with executor.rows([1]):
        paired_logits = backend.forward_logits(paired)
    assert torch.allclose(paired_logits[0], base[0], atol=1e-6)
    assert not torch.allclose(paired_logits[1], base[0])

    # Last-installed clamp at shared (pos, feature) wins over earlier scale
    expected_clamp, _ = backend.intervene_logits("ABC", [(0, 3, top_feat, 5.0)])
    assert torch.allclose(paired_logits[1], expected_clamp[0], atol=1e-6)
    h_scale()
    h_override()
    executor.close()


def test_load_backend_tiny_lm_and_default_device():
    dev = default_device()
    assert dev.type in {"cpu", "mps", "cuda"}
    backend = load_backend("tiny-lm")
    assert backend.model_id == "tiny-lm"
    assert backend.tokenize("A").shape == (1, 2)


def test_selected_feature_subset_activations_and_str_window_token_ids(monkeypatch):
    from collections import UserDict
    from types import SimpleNamespace
    from caliper import _ablate
    from caliper.interp import Graph

    backend = make_tiny_backend(device="cpu", dtype=torch.float32)

    # 1. Graph.from_circuit_tracer indexes activation_values by selected_features
    active_features = torch.tensor([[0, 1, 2], [0, 1, 5], [1, 1, 7]], dtype=torch.long)
    activation_values = torch.tensor([1.25, 9.5, 4.75], dtype=torch.float32)
    selected_features = torch.tensor([2, 0], dtype=torch.long)
    # 2 selected features + 2*2 error nodes + 2 embed tokens + 1 logit = 9 nodes
    adj = torch.zeros((9, 9), dtype=torch.float32)
    adj[8, 0] = 0.6
    adj[8, 1] = 0.3
    ct_graph = SimpleNamespace(
        adjacency_matrix=adj,
        input_tokens=torch.tensor([1, 2], dtype=torch.long),
        cfg=SimpleNamespace(n_layers=2),
        logit_targets=[SimpleNamespace(vocab_idx=3, token_str="C")],
        logit_probabilities=torch.tensor([0.9], dtype=torch.float32),
        active_features=active_features,
        selected_features=selected_features,
        activation_values=activation_values,
        vocab_size=16,
    )
    pruned = SimpleNamespace(
        node_mask=torch.ones(9, dtype=torch.bool),
        edge_mask=adj != 0,
        cumulative_scores=torch.linspace(0.1, 0.9, 9),
    )
    g = Graph.from_circuit_tracer(
        ct_graph,
        pruned,
        replacement_score=0.8,
        completeness_score=0.9,
        node_influence=torch.ones(9, dtype=torch.float32),
        tokenizer=backend.tokenizer,
    )
    feat_nodes = [n for n in g.nodes if n["kind"] == "feature"]
    assert [(n["layer"], n["feature"], n["activation"]) for n in feat_nodes] == [
        (1, 7, 4.75),
        (0, 2, 1.25),
    ]

    # 2. _pack_str_window maps token strings to vocab IDs (not window positions)
    win = backend._pack_str_window(0, "np-0", ["A", "B", "<unmapped-xyz>"], [0.1, 2.0, 0.0], 1, backend.tokenizer)
    center = win["center"]
    expected_a = backend.tokenizer.encode("A", add_special_tokens=False)[0]
    expected_b = backend.tokenizer.encode("B", add_special_tokens=False)[0]
    assert win["token_ids"][center - 1 : center + 2] == [expected_a, expected_b, None]
    assert win["positions"][center - 1 : center + 2] == [0, 1, 2]

    # 3. _tokenize_prompt accepts UserDict (HF BatchEncoding) & _tl_semantic_map gates mlp_in
    assert "layers.0.mlp_in" not in _ablate._tl_semantic_map(backend.model)
    plain = torch.nn.Linear(4, 4)
    tok_fn = lambda _p, return_tensors="pt": UserDict(
        input_ids=torch.tensor([[3, 4]]), attention_mask=torch.tensor([[1, 1]])
    )
    tok_out = _ablate._tokenize_prompt(plain, tok_fn, "AB")
    assert torch.equal(tok_out["input_ids"], torch.tensor([[3, 4]]))
    assert "attention_mask" in tok_out


def test_published_pruning_default_and_show_more_api():
    from caliper.interp._graph import (
        DEFAULT_EDGE_THRESHOLD,
        DEFAULT_NODE_THRESHOLD,
        LARGE_EDGE_CAP,
        LARGE_NODE_CAP,
        SHOW_MORE_EDGE_THRESHOLD,
        SHOW_MORE_NODE_THRESHOLD,
        Graph,
        Tracer,
        resolve_pruning_thresholds,
    )
    from circuit_tracer import attribute

    assert resolve_pruning_thresholds() == (DEFAULT_NODE_THRESHOLD, DEFAULT_EDGE_THRESHOLD)
    assert resolve_pruning_thresholds(show_more=True) == (
        SHOW_MORE_NODE_THRESHOLD,
        SHOW_MORE_EDGE_THRESHOLD,
    )
    assert resolve_pruning_thresholds(show_more=True, node_threshold=0.85) == (
        0.85,
        SHOW_MORE_EDGE_THRESHOLD,
    )
    with pytest.raises(ValueError, match="show_more"):
        resolve_pruning_thresholds(show_more=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="node_threshold"):
        resolve_pruning_thresholds(node_threshold=1.5)
    with pytest.raises(ValueError, match="edge_threshold"):
        resolve_pruning_thresholds(edge_threshold=float("nan"))

    backend = make_tiny_backend(device="cpu", dtype=torch.float32)
    prompt = "Light bends in a prism"

    default_graph = backend.attribute_and_prune(
        prompt, max_n_logits=4, max_feature_nodes=64, batch_size=8
    )
    more_graph = backend.attribute_and_prune(
        prompt, max_n_logits=4, max_feature_nodes=64, batch_size=8, show_more=True
    )
    unpruned_graph = backend.attribute_and_prune(
        prompt,
        max_n_logits=4,
        max_feature_nodes=64,
        batch_size=8,
        node_threshold=1.0,
        edge_threshold=1.0,
    )

    assert default_graph.settings["node_threshold"] == DEFAULT_NODE_THRESHOLD == 0.7
    assert default_graph.settings["edge_threshold"] == DEFAULT_EDGE_THRESHOLD == 0.9
    assert default_graph.settings["show_more"] is False
    assert default_graph.read()["settings"]["show_more"] is False

    assert more_graph.settings["node_threshold"] == SHOW_MORE_NODE_THRESHOLD
    assert more_graph.settings["edge_threshold"] == SHOW_MORE_EDGE_THRESHOLD
    assert more_graph.settings["show_more"] is True

    # Default published pruning drops nodes/edges; show_more reveals more while still pruning.
    assert len(default_graph.nodes) < len(more_graph.nodes) < len(unpruned_graph.nodes)
    assert len(default_graph.edges) < len(more_graph.edges) < len(unpruned_graph.edges)
    assert len(more_graph.nodes) <= LARGE_NODE_CAP
    assert len(more_graph.edges) <= LARGE_EDGE_CAP

    # Tracer.trace routes show_more cleanly with distinct cache entries.
    tracer = Tracer(backend.model, backend.tokenizer, backend=backend)
    traced_default = tracer.trace(prompt, max_targets=64, max_n_logits=4, batch_size=8)
    traced_more = tracer.trace(prompt, max_targets=64, max_n_logits=4, batch_size=8, show_more=True)
    traced_cached = tracer.trace(prompt, max_targets=64, max_n_logits=4, batch_size=8)
    assert traced_cached is traced_default
    assert traced_more is not traced_default
    assert traced_default.settings["show_more"] is False
    assert traced_more.settings["show_more"] is True
    assert len(traced_default.nodes) < len(traced_more.nodes)
    assert len(traced_default.edges) < len(traced_more.edges)

    # Graph.from_circuit_tracer applies published pruning directly when pruned is omitted.
    ct_graph = attribute(
        prompt,
        backend.model,
        max_n_logits=4,
        desired_logit_prob=0.95,
        batch_size=8,
        max_feature_nodes=64,
        verbose=False,
    )
    direct_default = Graph.from_circuit_tracer(ct_graph, tokenizer=backend.tokenizer)
    direct_more = Graph.from_circuit_tracer(ct_graph, tokenizer=backend.tokenizer, show_more=True)
    assert len(direct_default.nodes) == len(default_graph.nodes)
    assert len(direct_default.edges) == len(default_graph.edges)
    assert len(direct_more.nodes) == len(more_graph.nodes)
    assert len(direct_more.edges) == len(more_graph.edges)
