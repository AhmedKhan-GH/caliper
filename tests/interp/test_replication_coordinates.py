"""Original-coordinate consumer checks with an explicit CPU capture double."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from caliper._controls import Controls
from caliper.interp import _intervene, _replication_actions as actions
from caliper.interp._dictionary import Dictionary, JumpReLUTranscoder, model_identity
from tests.interp.fixture import TinyLM, TinyTokenizer


class ReferenceNormDouble(nn.Module):
    """Expose the actual cast-before-gain coordinate in the consumer fixture."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-.72, 1.31, width).bfloat16(), requires_grad=False)
        self.reference_input = nn.Identity()
        self.eps = 1e-6

    def forward(self, residual):
        value = residual.float()
        normalized = (value / (value.square().mean(-1, keepdim=True) + self.eps).sqrt()).bfloat16()
        return self.reference_input(normalized) * (1 + self.weight.float()).bfloat16()


@pytest.fixture
def original_case(tmp_path, monkeypatch):
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=3, dtype=torch.bfloat16).requires_grad_(False)
    tokenizer = TinyTokenizer()
    for block in model.layers:
        block.post_attention_layernorm = ReferenceNormDouble(8)
    layers = []
    for block in model.layers:
        mlp = block.mlp
        gain = (block.post_attention_layernorm.weight.float() + 1).bfloat16()
        original = mlp.W_enc.detach().clone()
        layer = JumpReLUTranscoder((original.float() / gain.float()).bfloat16(), mlp.b_enc.detach().clone(),
                                  mlp.W_dec.detach().clone(), mlp.b_dec.detach().clone(),
                                  torch.full((8,), .125, dtype=torch.bfloat16))
        layer.register_parameter('W_enc_original', nn.Parameter(original, requires_grad=False))
        layers.append(layer)
    dictionary = Dictionary({'model_hash': model_identity(model), 'width': 8,
                             'read_points': ['layers.0.mlp_in', 'layers.1.mlp_in'],
                             'write_points': ['layers.0.mlp_out', 'layers.1.mlp_out'],
                             'reference_input': deepcopy(_intervene._REFERENCE_INPUT)}, layers)
    captures = []

    @contextmanager
    def capture_double(current):
        values = {}
        captures.append(values)
        with ExitStack() as stack:
            for index, block in enumerate(current.layers):
                def record(module, args, output, index=index):
                    values[f'layers.{index}.mlp_in'] = output.detach()
                stack.callback(block.post_attention_layernorm.reference_input.register_forward_hook(record).remove)
            yield values

    monkeypatch.setattr(_intervene, '_capture_reference_inputs', capture_double)
    monkeypatch.setattr(_intervene, '_reference_read_point',
                        lambda current, layer: current.layers[layer].post_attention_layernorm.reference_input)
    monkeypatch.setattr(actions, 'PROMPTS', {'dallas': 'AB', 'oakland': 'CD', 'shanghai': 'EF', 'vancouver': 'GH'})
    _, active, _, _ = oracle(model, tokenizer('AB')['input_ids'], dictionary)
    _, donor, _, _ = oracle(model, tokenizer('EF')['input_ids'], dictionary)
    monkeypatch.setattr(actions, 'GROUPS', {'Texas': [(0, int(active[0][0, 1].argmax()), 1)],
                                          'China': [(1, int(donor[1][0, 2].argmax()), 2)]})
    monkeypatch.setattr(actions, 'CASES', {
        10: {'label': 'Baseline', 'operations': []},
        26: {'label': 'Replacement', 'operations': [
            {'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'},
            {'group': 'China', 'factor': 2, 'activation_prompt': 'shanghai'}]}})
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    runtime = actions.PublishedCases(model, tokenizer, service, output_directory=tmp_path)
    try:
        runtime.prepare()
        yield model, tokenizer, service, runtime, captures
    finally:
        runtime.close()
        service.remove()


def oracle(model, ids, dictionary, *, clamps=None, patterns=None):
    """Compute original coordinates and grouped deltas without consumer helpers."""
    x = model.embed_tokens(ids)
    values, coordinates, wrong = {}, {}, {}
    for index, block in enumerate(model.layers):
        prior = block.self_attn.frozen_pattern
        block.self_attn.frozen_pattern = None if patterns is None else patterns[f'layers.{index}.attn_pattern']
        try:
            residual = x + block.self_attn(block.input_layernorm(x))
        finally:
            block.self_attn.frozen_pattern = prior
        norm = block.post_attention_layernorm
        residual32 = residual.to(torch.float32)
        denominator = torch.sqrt(torch.mean(residual32 ** 2, dim=-1, keepdim=True) + norm.eps)
        coordinate = (residual32 / denominator).to(torch.bfloat16)
        gain = (norm.weight.to(torch.float32) + 1).to(torch.bfloat16)
        post = coordinate * gain
        layer = dictionary.layers[index]
        pre = F.linear(coordinate, layer.W_enc_original, layer.b_enc)
        active = pre * (pre > layer.threshold)
        values[index], coordinates[index] = active, coordinate
        wrong[index] = (post / gain, F.linear(post, layer.W_enc, layer.b_enc))
        output = block.mlp(post)
        delta = torch.zeros_like(output)
        for (layer_id, feature, position), target in (clamps or {}).items():
            if layer_id == index:
                delta[:, position] += (target - active[:, position, feature, None]) * layer.W_dec[feature]
        x = residual + (output + delta)
    return model.lm_head(model.norm(x)), values, coordinates, wrong


def test_preparation_uses_all_four_original_coordinate_forwards(original_case):
    model, tokenizer, service, runtime, captures = original_case
    assert len(captures) == 4 and all(not values for values in captures)
    inverse_differs = encoder_differs = False
    for source, prompt in actions.PROMPTS.items():
        _, expected, coordinates, wrong = oracle(model, tokenizer(prompt)['input_ids'], service.dictionary)
        for node, actual in runtime.values[source].items():
            layer, feature, position = node
            assert actual == float(expected[layer][0, position, feature])
        for layer, value in coordinates.items():
            inverse_differs |= not torch.equal(value, wrong[layer][0])
            weights = service.dictionary.layers[layer]
            incompatible = wrong[layer][1] * (wrong[layer][1] > weights.threshold)
            encoder_differs |= not torch.equal(expected[layer], incompatible)
    assert inverse_differs and encoder_differs


def test_each_candidate_uses_live_pre_gain_coordinates_and_replay(original_case):
    import numpy as np
    model, tokenizer, service, runtime, captures = original_case
    specs, members = runtime.case(26)
    targets = {(m['layer'], m['feature'], m['position']): m['value'] for m in members}
    ids = tokenizer('AB')['input_ids']
    expected, _, changed, _ = oracle(model, ids, service.dictionary, clamps=targets, patterns=runtime.patterns)
    _, _, baseline, _ = oracle(model, ids, service.dictionary, patterns=runtime.patterns)
    assert not torch.equal(changed[1], baseline[1])
    first = runtime.run(26)
    with np.load(runtime.output_directory / first['evidence']['file'], allow_pickle=False) as archive:
        torch.testing.assert_close(torch.from_numpy(archive['logits'][1]), expected[0, -1].float(), rtol=0, atol=0)
        assert first['softmax_dtype'] == 'torch.bfloat16'
    assert len(captures) == 5 and all(not values for values in captures)
    runtime.run(26)
    assert len(captures) == 6 and all(not values for values in captures)
    state = service.state_dict()
    service.load_state_dict(state, restore_active=True)
    with service.temporary_state(state):
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not values for values in captures) and not _intervene._REFERENCE_INPUTS.get()


def test_missing_live_coordinate_aborts_candidate_and_cleans_scope(original_case, monkeypatch):
    model, _, service, runtime, captures = original_case
    runtime.run(26)
    prior = service.state_dict(), deepcopy(runtime.last_result), set(runtime.directory.iterdir())

    monkeypatch.setattr(_intervene.Intervention, 'read', lambda *args, **kwargs: None)
    with pytest.raises(_intervene.InterventionError, match='shapes differ'):
        runtime.run(26)
    assert service.state_dict() == prior[0] and runtime.last_result == prior[1]
    assert set(runtime.directory.iterdir()) == prior[2]
    assert not _intervene._REFERENCE_INPUTS.get() and not service.executor._published_forward_depth


def test_reference_encoder_refuses_missing_or_wrong_precision(original_case):
    _, _, service, _, _ = original_case
    layer = service.dictionary.layers[0]
    coordinate = torch.ones((1, 3, 8), dtype=torch.bfloat16)
    with pytest.raises(_intervene.InterventionError, match='original bfloat16'):
        _intervene._reference_features(layer, coordinate.float())
    del layer.W_enc_original
    with pytest.raises(_intervene.InterventionError, match='original bfloat16'):
        _intervene._reference_features(layer, coordinate)


def test_actual_arithmetic_capture_dependency():
    """Qualify the real capture only when its owner supplies the module."""
    arithmetic = pytest.importorskip('caliper.interp._gemma_reference',
                                   reason='The independent arithmetic patch is not integrated')
    from tests.interp.test_gemma_runtime import gemma_model
    model = gemma_model().bfloat16()
    arithmetic.install_reference_profile(model, 'circuit_tracer_82996f14')
    residuals = {}
    with ExitStack() as stack:
        for index, block in enumerate(model.model.layers):
            def record(module, args, index=index):
                residuals[index] = args[0].detach().clone()
            stack.callback(block.pre_feedforward_layernorm.register_forward_pre_hook(record).remove)
        with _intervene._reference_input_scope(model) as values:
            with torch.no_grad():
                model(torch.tensor([[1, 5, 9]]), use_cache=False)
            for index, residual in residuals.items():
                value = residual.float()
                eps = model.model.layers[index].pre_feedforward_layernorm.eps
                expected = (value / torch.sqrt(torch.mean(value ** 2, dim=-1, keepdim=True) + eps)).bfloat16()
                torch.testing.assert_close(values[f'model.layers.{index}.mlp_in'], expected, rtol=0, atol=0)
    assert not values and not _intervene._REFERENCE_INPUTS.get()
