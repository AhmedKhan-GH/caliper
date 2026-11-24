"""Index and example activations preserve the live model input precision."""
import json
import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from caliper.interp import capture, load_dictionary
from caliper.interp._index import FeatureIndex, build_index
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import fixture_manifest


@pytest.mark.parametrize("activation", ["topk", "jumprelu"])
def test_half_dictionary_preserves_live_threshold_and_example_values(tmp_path, activation):
    model = TinyLM(layers=1, width=4, heads=1, latents=4, k=1, dtype=torch.float32)
    target = 1 + 2**-12
    with torch.no_grad():
        model.embed_tokens.weight.fill_(1)
        model.layers[0].self_attn.o_proj.weight.zero_()
        norm = model.layers[0].post_attention_layernorm
        norm.weight.fill_(target * math.sqrt(1 + norm.eps))
    metadata = fixture_manifest(model)
    metadata.update(activation=activation, k=1 if activation == "topk" else None)
    weights = {"W_enc": torch.zeros(4, 4, dtype=torch.float16),
               "b_enc": torch.zeros(4, dtype=torch.float16),
               "W_dec": torch.ones(4, 4, dtype=torch.float16),
               "b_dec": torch.zeros(4, dtype=torch.float16)}
    weights["W_enc"][0, 0] = 1
    if activation == "jumprelu":
        weights["threshold"] = torch.ones(4, dtype=torch.float16)
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    save_file(weights, tmp_path / "layer_00.safetensors")
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float16)
    point = metadata["read_points"][0]
    tokens = [65, 66, 67, 68]
    with torch.no_grad(), capture(model, [point]) as saved:
        model(torch.tensor([tokens]))
        # One encoder coefficient is one. The independent scalar oracle is x[0].
        expected = saved[point][0, :, 0].tolist()
        assert all(value > 1 for value in expected)
        assert saved[point][..., 0].half().eq(1).all()
    build_index(tmp_path, [("first", tokens)], model=model, tokenizer=TinyTokenizer(),
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer(), dictionary=dictionary) as index:
        assert int(index.counts[0, 0]) == 4
        assert not index.counts[0, 1:].any()
        np.testing.assert_array_equal(index.table[0, 0]["offset"][:4], list(range(4)))
        np.testing.assert_array_equal(index.table[0, 0]["activation"][:4], expected)
        rows = index.examples(0, 0)
        assert len(rows) == 4
        for row in rows:
            assert row["activation"] == expected[row["position"]]
            assert [value for value in row["activations"] if value is not None] == expected
    for name, value in dictionary.layers[0].state_dict().items():
        assert value.dtype == torch.float16
        assert torch.equal(value, weights[name])
