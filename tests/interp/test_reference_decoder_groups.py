"""CPU evidence for whole-layer source geometry and bounded decoder scopes."""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from caliper.interp import _trace
from .test_reference_batch_differential import pinned_context
from .reference_source_oracle import (Geometry, pinned_decoder, source_case, whole_layer_scores,
                                      contraction_evidence, compare_source_products)


def pinned_scores(context_class, gradient, vectors, positions, sizes):
    """Execute source hooks with explicit independent partition boundaries."""
    context = object.__new__(context_class)
    context._batch_buffer = vectors.new_zeros(len(vectors), len(gradient))
    offset = 0
    with Geometry() as geometry:
        for size in sizes:
            rows = slice(offset, offset + size)
            _, hook = context._compute_score_hook(
                "synthetic", vectors[rows], rows, np.s_[:, positions[rows]])
            hook(gradient, None)
            offset += size
    assert offset == len(vectors)
    return context._batch_buffer, geometry.calls


def per_source_products(calls):
    """Expose actual BMM matrix operands without a predicted lowering rule."""
    result = []
    for name, operands in calls:
        if name == "aten.bmm.default":
            left, right, output = operands
            for source in range(left[0][0]):
                result.append((left[3][source], right[3][source], output[3][source]))
    return result


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_singleton_boundary_has_independent_geometry(pinned_context, batch):
    gradient = torch.arange(batch * 3 * 7, dtype=torch.float32).reshape(batch, 3, 7)
    vectors = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7)
    positions = torch.tensor([1, 1, 2])
    _, singles = pinned_scores(pinned_context, gradient, vectors, positions, [1, 1, 1])
    _, joint = pinned_scores(pinned_context, gradient, vectors, positions, [3])
    single_operands = per_source_products(singles)
    joint_operands = per_source_products(joint)
    assert len(single_operands) == len(joint_operands) == 3
    for separate, combined in zip(single_operands, joint_operands):
        if batch == 1:
            assert [tuple(x.shape) for x in separate] == [tuple(x.shape) for x in combined]
        else:
            assert [tuple(x.shape) for x in separate] != [tuple(x.shape) for x in combined]


LAYOUTS = [
    [[2, 2, 0, 1, 2, 3, 1], [1, 2, 2, 0, 3, 2, 1]],
    [[7, 0, 2, 3, 1, 2, 2], [2, 3, 1, 2, 2, 0, 7]],
    [[7, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 1, 2, 3, 1]],
    [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3, 8, 512])
@pytest.mark.parametrize("row_bound", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("counts", LAYOUTS)
def test_groups_match_actual_whole_layer_hooks(
        pinned_context, pinned_decoder, monkeypatch, dtype, batch, row_bound, counts):
    effects = _trace._DirectEffects(*source_case(dtype, counts), profile=_trace._reference_graph.PROFILE)
    bound = row_bound * effects.path.width
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", bound)
    pool = effects.logit_nodes + effects.feature_nodes[:2] + effects.feature_nodes[-2:]
    targets = (pool * batch)[:batch]
    targets.reverse()
    nodes, columns = deepcopy(effects.nodes), dict(effects.feature_columns)
    (actual, accounts), gradients, calls, _ = contraction_evidence(effects, targets, monkeypatch)
    expected, scopes = whole_layer_scores(effects, gradients, pinned_context, pinned_decoder)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype == dtype
    compare_source_products(effects, calls, scopes, batch)
    assert [account["incoming"] for account in accounts] == expected.sum(0, dtype=torch.float32).tolist()
    scaled = {row: vector for scope in scopes for row, vector in zip(scope["rows"], scope["vectors"])}
    cached_rows, refs, storage = [], [], set()
    for layer, blocks in enumerate(effects._decoder_blocks):
        for rows, positions, decoder in blocks:
            refs.append(weakref.ref(decoder))
            storage.add(decoder.untyped_storage().data_ptr())
            assert decoder.numel() <= bound
            if isinstance(positions, int):
                positions = [positions] * len(decoder)
            else:
                refs.append(weakref.ref(positions))
                assert positions.numel() == len(decoder)
                positions = positions.tolist()
            assert all(position == 0 for position in positions) or all(position > 0 for position in positions)
            for row, position, vector in zip(range(rows.start, rows.stop), positions, decoder):
                cached_rows.append(row)
                assert effects.nodes[row]["layer"] == layer
                assert effects.nodes[row]["position"] == position
                torch.testing.assert_close(vector, scaled[row], rtol=0, atol=0)
    assert cached_rows == effects.feature_nodes
    assert len(storage) == sum(map(len, effects._decoder_blocks))
    assert effects.nodes == nodes and effects.feature_columns == columns
    again, again_accounts = effects.backward(targets)
    torch.testing.assert_close(again, actual, rtol=0, atol=0)
    assert again_accounts == accounts
    if cached_rows:
        del decoder, vector
    del blocks
    effects.release_decoders()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("batch", range(1, 513))
def test_all_target_batch_sizes_keep_true_singletons_and_multi_source_tails(
        pinned_context, pinned_decoder, monkeypatch, batch):
    effects = _trace._DirectEffects(*source_case(torch.bfloat16, [[2, 1, 1], [2, 0, 1]], width=3),
                                   profile=_trace._reference_graph.PROFILE)
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", effects.path.width)
    targets = (effects.logit_nodes * batch)[:batch]
    (actual, _), gradients, calls, _ = contraction_evidence(effects, targets, monkeypatch)
    expected, scopes = whole_layer_scores(effects, gradients, pinned_context, pinned_decoder)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compare_source_products(effects, calls, scopes, batch)


def grouped_effects(monkeypatch):
    effects = _trace._DirectEffects(*source_case(torch.bfloat16, LAYOUTS[0]),
                                   profile=_trace._reference_graph.PROFILE)
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 7 * effects.path.width)
    return effects


def test_group_allocation_has_bounded_unique_storage(monkeypatch):
    effects = grouped_effects(monkeypatch)
    path = effects.path
    effects.prepare_decoders()
    blocks = [block for layer in effects._decoder_blocks for block in layer]
    decoder_bytes = sum(decoder.untyped_storage().nbytes() for _, _, decoder in blocks)
    assert decoder_bytes == len(effects.feature_nodes) * path.width * path.embedding.element_size()
    positions = [position for _, position, _ in blocks if isinstance(position, torch.Tensor)]
    assert positions
    assert sum(value.untyped_storage().nbytes() for value in positions) <= 8 * len(effects.feature_nodes)
    assert all(decoder.numel() <= 7 * path.width for _, _, decoder in blocks)
    count, columns, itemsize = len(effects.feature_nodes), 7, path.embedding.element_size()
    retained = count * path.width * itemsize + (count + path.layers * path.tokens) * 256 + count * 8
    scratch = 7 * path.width * (8 + 2 * itemsize) + columns * (8 + itemsize + 16)
    assert dict(path.reserves)["direct decoder cache"] == retained + scratch


def test_single_feature_targets_and_all_bias_categories(monkeypatch, pinned_context, pinned_decoder):
    effects = grouped_effects(monkeypatch)
    actual_accounts = []
    for target in [effects.feature_nodes[0], effects.feature_nodes[-1], effects.logit_nodes[-1]]:
        (actual, accounts), gradients, _, _ = contraction_evidence(effects, [target], monkeypatch)
        expected, _ = whole_layer_scores(effects, gradients, pinned_context, pinned_decoder)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert accounts[0]["incoming"] == float(expected.sum(0, dtype=torch.float32)[0])
        actual_accounts.extend(accounts)
    assert all(any(account["biases"][name] != 0 for account in actual_accounts) for name in
               ("decoder", "normalization", "attention", "intervention", "encoder", "unembedding"))


def test_cache_admission_covers_positions_before_any_allocation(monkeypatch):
    from caliper.interp._memory import CleanStop

    effects = grouped_effects(monkeypatch)
    tensors, tensor = [], torch.tensor

    def allocate(*args, **kwargs):
        tensors.append(args)
        return tensor(*args, **kwargs)

    def refuse(stage, size):
        assert stage == "direct decoder cache"
        count, width = len(effects.feature_nodes), effects.path.width
        # Decoder bytes, positions, row metadata, and bounded gather/cast/scale.
        assert size >= count * (width * 2 + 8) + 7 * width * (8 + 2 * 2)
        raise CleanStop("position reserve")

    effects.path.guard.reserve = refuse
    monkeypatch.setattr(torch, "tensor", allocate)
    with pytest.raises(CleanStop, match="position reserve"):
        effects.prepare_decoders()
    assert tensors == []
    assert effects._decoder_blocks is None


@pytest.mark.parametrize("failure", ["process", "host", "history", "supplied"])
@pytest.mark.parametrize("phase", ["cache", "first", "later"])
def test_groups_keep_real_lower_guards(monkeypatch, failure, phase):
    from caliper.interp import _frozen
    from caliper.interp._memory import CleanStop
    from .test_memory_phases import process_guard

    effects = grouped_effects(monkeypatch)
    if phase != "cache":
        effects.prepare_decoders()
    readings = dict(current=1_000, peak=1_000)
    supplied = process_guard(1_000_000, readings)
    supplied.host_reserve_bytes = 8_000_000_000
    host = SimpleNamespace(available=9_000_000_000)
    monkeypatch.setattr("psutil.virtual_memory", lambda: host)
    monkeypatch.setattr(_frozen, "ProcessMemoryGuard", lambda limit, device, **kwargs:
                        process_guard(int(limit * 1e9), readings))
    guard = _trace._SuppliedTraceGuard("cpu", memory_limit_gb=40, memory=supplied)
    effects.path.guard = guard
    stage = "direct decoder cache" if phase == "cache" else "direct decoder contraction"
    fail_at = 2 if phase == "later" else 1
    reserve, check, calls = guard.reserve, supplied.check, []

    def limited(name, size):
        if name == stage:
            calls.append(size)
            if len(calls) == fail_at:
                if failure == "process":
                    supplied.limit = readings["current"] + size - 1
                elif failure == "host":
                    host.available = supplied.host_reserve_bytes + size - 1
                elif failure == "history":
                    readings["peak"] = supplied.limit + 1
        return reserve(name, size)

    def supplied_check(name, **kwargs):
        if name == stage and len(calls) == fail_at and failure == "supplied":
            raise CleanStop("supplied refusal")
        return check(name, **kwargs)

    monkeypatch.setattr(guard, "reserve", limited)
    monkeypatch.setattr(supplied, "check", supplied_check)
    with pytest.raises(CleanStop) as error:
        effects.backward(effects.logit_nodes)
    assert error.value.__traceback__ is not None
    assert len(calls) == fail_at
    assert guard.limit == supplied.limit < 40_000_000_000
    assert guard.host_reserve_bytes == supplied.host_reserve_bytes
    if failure == "history":
        readings["peak"] = 0
        with pytest.raises(CleanStop, match="memory limit"):
            guard.check("latched breach")
    effects.release_decoders()
    assert effects._decoder_blocks is None


@pytest.mark.parametrize("reason", ["cancel", "deadline"])
@pytest.mark.parametrize("stage", ["direct decoder cache position", "direct decoder cache block",
    "direct decoder cache group", "direct decoder cache complete", "direct decoder block",
    "finite block readback", "direct edge products"])
def test_group_stops_release_storage_with_retained_exception(monkeypatch, reason, stage):
    from contextlib import contextmanager
    from caliper.interp._memory import CleanStop
    from torch.utils._python_dispatch import TorchDispatchMode

    effects = grouped_effects(monkeypatch)
    checks, scopes, refs = [], [], []
    exception = CleanStop if reason == "cancel" else _trace.TraceTimeout

    def check(name):
        checks.append(name)
        if name == stage and ("complete" in name or "readback" in name or "products" in name
                              or checks.count(name) == 2):
            raise exception("group stop")

    @contextmanager
    def scope(device):
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")

    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if (func == torch.ops.aten.mul.Tensor and result.ndim == 2
                    and result.shape[-1] == effects.path.width):
                refs.append(weakref.ref(result))
            return result

    effects.check = check
    monkeypatch.setattr(_trace, "autorelease_scope", scope)
    with Allocations():
        with pytest.raises(exception, match="group stop") as error:
            effects.backward(effects.logit_nodes)
    assert checks[-1] == stage
    assert error.value.__traceback__ is not None
    effects.release_decoders()
    assert effects._decoder_blocks is None
    assert all(ref() is None for ref in refs)
    assert scopes.count("enter") == scopes.count("exit")
    # The failure frame also owns unrelated target and accounting tensors.
    # Inspect decoder cache references separately from those baseline objects.
    frames = []
    trace = error.value.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name in ("prepare_decoders", "backward"):
            frames.append(trace.tb_frame.f_locals)
        trace = trace.tb_next
    for frame in frames:
        assert all(frame.get(name) is None for name in
                   ("decoder", "position", "positions", "selected_columns", "selected", "values", "gathered", "products"))
