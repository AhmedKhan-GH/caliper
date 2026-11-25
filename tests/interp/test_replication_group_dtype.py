"""CPU regressions for the pinned decoder accumulator and mixed output dtype."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._gemma_reference import PROFILE, install_reference_profile
from caliper.interp._intervene import InterventionService
from tests.interp.test_gemma_frozen import original_encoder_dictionary
from tests.interp.test_gemma_reference import reference_fixture
from tests.interp.test_intervention_replay import hook_objects


@pytest.fixture(params=[True, False], ids=['reference', 'ordinary'])
def mixed_group(request):
    model = reference_fixture()
    install_reference_profile(model, PROFILE)
    dictionary = original_encoder_dictionary(model, dtype=torch.bfloat16)
    weights = dictionary.layers[0]
    with torch.no_grad():
        weights.W_enc_original.zero_()
        weights.W_enc_original[1:4, 0] = 1
        weights.W_enc.copy_(weights.W_enc_original)
        weights.b_enc.zero_()
        weights.b_enc[1:4] = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
        weights.threshold.zero_()
        weights.W_dec.zero_()
        weights.W_dec[1, :3] = torch.tensor([1, 2**24, 1], dtype=torch.bfloat16)
        weights.W_dec[2, :3] = torch.tensor([1 / 512, 1, -1], dtype=torch.bfloat16)
        weights.W_dec[3, :3] = torch.tensor([0, -(2**24), 0], dtype=torch.bfloat16)
    if not request.param:
        del dictionary.metadata['reference_input']
    service = InterventionService(model)
    service.configure(dictionary=dictionary)
    original_hooks = hook_objects(model)
    block = model.model.layers[0]
    output_input = torch.ones(2, 3, model.config.hidden_size, dtype=torch.bfloat16)
    baseline = block.post_feedforward_layernorm(output_input)
    assert baseline.dtype == torch.float32
    members = [service.install({'kind': 'clamp', 'point': 'model.layers.0.mlp_out',
                               'feature': feature, 'positions': [position], 'value': value})
               for feature, position, value in [(3, 1, 6), (1, 1, 4), (2, 1, 5), (1, 2, 5)]]
    group = members[0].group
    raw = torch.ones_like(output_input)
    raw[1] = 2
    case = SimpleNamespace(service=service, block=block, group=group, members=members,
                           weights=weights, raw=raw, output_input=output_input,
                           baseline=baseline, reference=request.param)
    try:
        yield case
    finally:
        service.close()
        assert hook_objects(model) == original_hooks
        assert not service._feature_groups


def apply_mixed_group(case):
    if case.reference:
        case.block.pre_feedforward_layernorm.reference_input(case.raw)
    else:
        case.block.mlp(case.raw)
    return case.block.post_feedforward_layernorm(case.output_input)


@pytest.mark.parametrize('rows', [None, [1], []], ids=['all_rows', 'selected_row', 'no_rows'])
def test_group_accumulator_matches_pinned_dtype_and_feature_order(mixed_group, rows):
    case = mixed_group
    assert case.group.reference_input is case.reference
    assert case.weights.W_dec.dtype == case.raw.dtype == torch.bfloat16
    with nullcontext() if rows is None else case.service.rows(rows):
        actual = apply_mixed_group(case)

    # circuit-tracer 82996f14, replacement_model_transformerlens.py:626-694:
    # nonzero orders features, index_add_ sums in cfg.dtype, then output adds delta.
    # Coordinates test lost BF16 precision, order-dependent cancellation, and exact cancellation.
    expected = case.baseline.clone()
    for row in range(2) if rows is None else rows:
        amount = 2 - row
        small = 0 if case.reference else amount / 512
        expected[row, 1, :3] += torch.tensor([amount + small, 0, 0])
        expected[row, 2, :3] += (3 - row) * torch.tensor([1, 2**24, 1])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rows:
        # Specification order 3, 1, 2 would leave 1 in coordinate 1.
        # This also detects order changes in kernels with float32 intermediate sums.
        assert actual[1, 1, 1] == case.baseline[1, 1, 1]
    assert case.group.live_input is None and not case.group.read_stack
    assert all(member.errors == 0 for member in case.members)


@pytest.mark.parametrize('reason', ['cancel', 'budget'])
def test_mixed_group_admission_failure_removes_hooks_before_encoding(mixed_group, monkeypatch, reason):
    from caliper.interp import _intervene
    from caliper.interp._memory import CleanStop

    case = mixed_group
    checks = []

    def check(*args, **kwargs):
        checks.append(1)
        if reason == 'cancel':
            raise CleanStop('group fixture cancellation')
        return 0

    case.service.resource_guard = SimpleNamespace(limit=1, current=0, peak=0,
                                                   device=torch.device('cpu'), check=check)
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=40_000_000_000))
    monkeypatch.setattr(_intervene, '_reference_features', lambda *a: pytest.fail('encoder before admission'))
    monkeypatch.setattr(_intervene, '_trace_features', lambda *a: pytest.fail('encoder before admission'))
    with pytest.raises(CleanStop, match='cancellation|process memory limit'):
        apply_mixed_group(case)
    assert checks
    assert all(member.removed and member.errors == 1 for member in case.members)
    assert not case.group.handles and not case.service._feature_groups
    assert case.group.live_input is None and not case.group.read_stack
