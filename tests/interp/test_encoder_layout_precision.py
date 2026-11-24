"""Feature identity across encoder blocks and inference input layouts."""
import math

import pytest
import torch

from caliper.interp import _frozen
from caliper.interp._dictionary import TopKTranscoder


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16])
def test_encoder_blocks_preserve_multiterm_rank_across_input_layouts(monkeypatch, model_dtype, storage_dtype):
    width, latents = 32, 11
    # Three rows per block place the two closest candidates in different blocks.
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", width * 3)
    weights = torch.zeros(latents, width, dtype=storage_dtype)
    weights[:, :16] = .125
    weights[:, 16:] = -.0625
    weights[8, 0] += torch.finfo(storage_dtype).eps / 4
    bias = torch.full((latents,), -2., dtype=storage_dtype)
    bias[1] = bias[8] = 0.
    layer = TopKTranscoder(weights, bias, torch.zeros(latents, width, dtype=storage_dtype),
                          torch.zeros(width, dtype=storage_dtype), 1)
    inputs = torch.ones(2, 5, width, dtype=model_dtype)
    inputs[..., 2::3] = .5
    inputs[..., 3::4] = 1.5
    # The reference uses scalar products and a stable sum, without a tensor encoder.
    reference = []
    for row in inputs.reshape(-1, width).tolist():
        values = [max(0., math.fsum(a * b for a, b in zip(row, weight)) + float(offset))
                  for weight, offset in zip(weights.tolist(), bias)]
        winner = max(range(latents), key=values.__getitem__)
        assert winner == 8 and values[8] > values[1]
        reference.append([value if index == winner else 0. for index, value in enumerate(values)])
    expected = torch.tensor(reference, dtype=model_dtype).reshape(2, 5, latents)
    stages = []
    with torch.no_grad():
        full = _frozen._trace_features(layer, inputs, check=stages.append)
        selected = _frozen._trace_features(layer, inputs[:, [1, 3]])
        single = _frozen._trace_features(layer, inputs[0, 1])
    assert stages.count("dictionary encoder block") == 4
    torch.testing.assert_close(full, expected, atol=0, rtol=0)
    torch.testing.assert_close(selected, expected[:, [1, 3]], atol=0, rtol=0)
    torch.testing.assert_close(single, expected[0, 1], atol=0, rtol=0)
    assert not full.requires_grad and not selected.requires_grad and not single.requires_grad
