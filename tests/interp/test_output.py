"""Tests for Output paired comparison and 24-token greedy generation."""
import torch

from caliper import _controls
from caliper.interp import _publish
from caliper.interp._backend import make_tiny_backend
from caliper.interp._output import Output


def test_output_compare_and_generate_with_feature_and_tensor_interventions(monkeypatch):
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    backend = make_tiny_backend()
    ctl = _controls.Controls()
    ivs = ctl.interventions_on(backend.model)
    ivs.configure(dictionary=backend.dictionary, tokenizer=backend.tokenizer, backend=backend)
    out = Output(backend, ivs)
    try:
        base_cmp = out.compare("AB")
        assert len(base_cmp["tokens"]) == 10
        assert base_cmp["probabilities"][0] == base_cmp["probabilities"][1]
        assert set(published) == {"output.probs", "output.tokens", "output.label_text"}
        assert published["output.probs"].shape == (2, 10)

        _, dense = backend.activations("AB")
        feat = int(dense[0, 2].argmax().item())
        ivs.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": feat, "positions": [2], "value": 6.0})
        ctl.checkpoint()
        steered_cmp = out.compare("AB")
        base_by_tok = dict(zip(base_cmp["tokens"], base_cmp["probabilities"][0]))
        steered_base_by_tok = dict(zip(steered_cmp["tokens"], steered_cmp["probabilities"][0]))
        for tok in set(base_by_tok) & set(steered_base_by_tok):
            assert steered_base_by_tok[tok] == base_by_tok[tok]
        assert steered_cmp["baseline_top_probability"] == base_cmp["baseline_top_probability"]
        assert steered_cmp["probabilities"][1] != steered_cmp["probabilities"][0]
        assert all(w in steered_cmp["tokens"] for w in steered_cmp["argmax_token_ids"])

        gen = out.generate("AB")
        assert len(gen["token_ids"][0]) == 24 and len(gen["token_ids"][1]) == 24

        # Tensor ablation path through paired batch row selection
        ivs.clear()
        ivs.set({"kind": "ablate", "point": "layers.1.attn_out"})
        ctl.checkpoint()
        abl_cmp = out.compare("AB")
        assert abl_cmp["baseline_top_probability"] == base_cmp["baseline_top_probability"]
        assert abl_cmp["probabilities"][1] != abl_cmp["probabilities"][0]
    finally:
        ivs.remove()


def test_output_generate_fresh_per_prompt_and_baseline_isolated_from_clamp(monkeypatch):
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    backend = make_tiny_backend()
    ctl = _controls.Controls()
    ivs = ctl.interventions_on(backend.model)
    ivs.configure(dictionary=backend.dictionary, tokenizer=backend.tokenizer, backend=backend)
    out = Output(backend, ivs)
    try:
        clean_carrot = out.generate("carrot")
        assert clean_carrot["baseline"] == clean_carrot["steered"]

        _, dense = backend.activations("carrot")
        feat = int(dense[0, 2].argmax().item())
        ivs.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": feat, "positions": [2], "value": 8.0})
        ctl.checkpoint()

        clamped_carrot = out.generate("carrot")
        assert clamped_carrot["baseline"] == clean_carrot["baseline"]
        assert clamped_carrot["token_ids"][0] == clean_carrot["token_ids"][0]

        lantern_gen = out.generate("lantern")
        assert lantern_gen["token_ids"][0] != clean_carrot["token_ids"][0]
        assert lantern_gen["baseline"] != clean_carrot["baseline"]
    finally:
        ivs.remove()
