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
