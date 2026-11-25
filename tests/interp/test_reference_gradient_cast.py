"""CPU checks for the lifetime and geometry of reference gradient casts."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch

from caliper.interp import _trace
from caliper.interp._memory import CleanStop
from .test_direct_decoder_cache import fixture
from .test_reference_batch_differential import pinned_context
from .reference_source_oracle import (pinned_decoder, whole_layer_scores,
                                      contraction_evidence, compare_source_products)


def position_case(dtype=torch.bfloat16, counts=None):
    effects = fixture(dtype, counts)
    path = effects.path
    path.errors = [value.float() for value in path.errors]
    factors = torch.tensor([1., -.375, 2.125]).reshape(1, path.tokens, 1)

    def run(embedding, writes, **kwargs):
        x, reads = embedding, []
        for layer, write in enumerate(writes):
            reads.append(x.sum(1, keepdim=True).expand(-1, path.tokens, -1))
            x = x + write * (factors + layer)
        return reads, x.sum(1, keepdim=True).expand(-1, path.tokens, -1).to(dtype)

    path.run = run
    _, path.reference_final = run(path.embedding, path.errors)
    return effects


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_elements", [4, 8, 64])
@pytest.mark.parametrize("counts", [None, [2, 1, 1], [2, 1, 2], [4, 0, 1]])
def test_layer_cast_matches_pinned_sources_and_accounts(
        pinned_context, pinned_decoder, monkeypatch, dtype, block_elements, counts):
    effects = position_case(dtype, counts)
    path = effects.path
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", block_elements)
    effects.prepare_decoders()
    nodes = [dict(node) for node in effects.nodes]
    targets = effects.logit_nodes + [effects.feature_nodes[0], effects.feature_nodes[-1]]
    (actual, accounts), gradients, calls, geometry = contraction_evidence(effects, targets, monkeypatch)
    expected, scopes = whole_layer_scores(effects, gradients, pinned_context, pinned_decoder)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compare_source_products(effects, calls, scopes, len(targets))
    gathers = [operands for name, operands in geometry if name == "aten.index.Tensor"]
    blocks = [(layer, rows, position, decoder) for layer, cache in enumerate(effects._decoder_blocks)
              for rows, position, decoder in cache]
    assert len(gathers) == len(blocks)
    for (layer, rows, position, decoder), (source, gathered) in zip(blocks, gathers):
        gradient = gradients[layer + 1]
        assert not torch.equal(gradient[:, 0], gradient[:, 2])
        # Source coordinates come from nodes, independently of cache metadata.
        positions = torch.tensor([effects.nodes[row]["position"] for row in range(rows.start, rows.stop)])
        cached_positions = torch.full_like(positions, position) if isinstance(position, int) else position
        assert torch.equal(positions, cached_positions)
        cast = gradient.to(decoder.dtype)
        assert source[:3] == (tuple(cast.shape), cast.stride(), cast.dtype)
        torch.testing.assert_close(source[3], cast, rtol=0, atol=0)
        reference = cast[:, positions]
        assert gathered[:3] == (tuple(reference.shape), reference.stride(), reference.dtype)
        torch.testing.assert_close(gathered[3], reference, rtol=0, atol=0)
    incoming = expected.sum(0, dtype=torch.float32)
    for index, account in enumerate(accounts):
        target = effects.target(targets[index])
        assert account["expected"] == float(target.expected)
        assert account["incoming"] == float(incoming[index])
        assert account["biases"] == dict(decoder=0., normalization=0., attention=0., intervention=0.,
                                         encoder=float(target.encoder_bias), unembedding=float(target.unembedding_bias))
        residual = float(abs(np.float32(np.float32(float(incoming[index])) + np.float32(float(target.encoder_bias)))
                             - np.float32(float(target.expected))))
        assert account["absolute_residual"] == residual
        assert account["relative_residual"] == residual / max(1., abs(float(target.expected)))
    assert effects.nodes == nodes


def observe_casts(monkeypatch, effects, batch):
    original = torch.Tensor.to
    refs, calls = [], []

    def cast(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if (value.shape == (batch, effects.path.tokens, effects.path.width)
                and args == (effects.path.dtype,) and value.dtype != result.dtype and not value.requires_grad):
            refs.append(weakref.ref(result))
            calls.append((value.shape, value.stride(), result.shape, result.stride()))
        return result

    monkeypatch.setattr(torch.Tensor, "to", cast)
    return refs, calls


@pytest.mark.parametrize("block_elements", [4, 8, 64])
def test_cast_reuse_keeps_each_block_reserve_and_releases_each_layer(monkeypatch, block_elements):
    effects = position_case()
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", block_elements)
    effects.prepare_decoders()
    batch = len(effects.logit_nodes)
    refs, calls = observe_casts(monkeypatch, effects, batch)
    blocks = [block for layer in effects._decoder_blocks for block in layer]
    sizes, live = [], []

    def reserve(stage, size):
        if stage == "direct decoder contraction":
            decoder = blocks[len(sizes)][2]
            count, itemsize = len(decoder), decoder.element_size()
            cast_bytes = batch * effects.path.tokens * effects.path.width * itemsize
            block_bytes = batch * count * (effects.path.width + 1) * itemsize + count * 8
            assert size == cast_bytes + block_bytes
            sizes.append(size)
            live.append(sum(ref() is not None for ref in refs))
        elif stage == "direct decoder bias accounting":
            assert all(ref() is None for ref in refs)

    effects.path.guard.reserve = reserve
    effects.backward(effects.logit_nodes)
    assert len(calls) == effects.path.layers
    assert len(sizes) == len(blocks) > len(calls)
    first_indices = [0, len(effects._decoder_blocks[0])]
    assert live == [0 if index in first_indices else 1 for index in range(len(blocks))]
    assert all(ref() is None for ref in refs)
    sizes.clear()
    live.clear()
    effects.backward(effects.logit_nodes)
    assert len(calls) == 2 * effects.path.layers
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("block", [1, 2])
@pytest.mark.parametrize("reason", [CleanStop, _trace.TraceTimeout])
def test_refusal_precedes_block_allocations_and_releases_cast(monkeypatch, block, reason):
    effects = position_case()
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 4)
    effects.prepare_decoders()
    refs, calls = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    full = torch.full
    allocations, reserves, scopes = [], [], []

    def positions(*args, **kwargs):
        allocations.append(args)
        return full(*args, **kwargs)

    @contextmanager
    def scope(device):
        scopes.append("enter")
        try:
            yield
        finally:
            scopes.append("exit")

    def reserve(stage, size):
        if stage == "direct decoder contraction":
            reserves.append(size)
            if len(reserves) == block:
                raise reason("block refusal")

    effects.path.guard.reserve = reserve
    monkeypatch.setattr(torch, "full", positions)
    monkeypatch.setattr(_trace, "autorelease_scope", scope)
    with pytest.raises(reason, match="block refusal") as failure:
        effects.backward(effects.logit_nodes)
    assert failure.value.__traceback__ is not None
    assert len(allocations) == block - 1
    assert len(calls) == min(1, block - 1)
    assert all(ref() is None for ref in refs)
    assert scopes == [entry for _ in range(block) for entry in ("enter", "exit")]
    effects.release_decoders()
    assert effects._decoder_blocks is None


@pytest.mark.parametrize("stage", ["direct decoder block", "finite block readback", "direct edge products"])
@pytest.mark.parametrize("reason", [CleanStop, _trace.TraceTimeout])
def test_late_failures_keep_checks_and_cleanup(monkeypatch, stage, reason):
    effects = position_case()
    effects.prepare_decoders()
    refs, _ = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    seen = []

    def check(name):
        seen.append(name)
        if name == stage and (stage != "direct decoder block" or seen.count(name) == 2):
            raise reason("late refusal")

    effects.check = check
    with pytest.raises(reason, match="late refusal") as failure:
        effects.backward(effects.logit_nodes)
    assert failure.value.__traceback__ is not None
    assert refs and all(ref() is None for ref in refs)
    assert seen[-1] == stage
    effects.release_decoders()
    assert effects._decoder_blocks is None


@pytest.mark.parametrize("block", [1, 2])
@pytest.mark.parametrize("fault", ["process", "host", "history", "supplied"])
def test_cast_reserves_keep_real_guard_limits(monkeypatch, block, fault):
    from types import SimpleNamespace
    from caliper.interp import _frozen
    from .test_memory_phases import process_guard

    effects = position_case()
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 4)
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
    refs, calls = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    full, reserve, check = torch.full, guard.reserve, supplied.check
    allocations, reserves, stages = [], [], []
    cast_bytes = len(effects.logit_nodes) * effects.path.tokens * effects.path.width * 2

    def positions(*args, **kwargs):
        allocations.append(args)
        return full(*args, **kwargs)

    def supplied_check(stage, **kwargs):
        stages.append(stage)
        if fault == "supplied" and len(reserves) == block:
            raise CleanStop("supplied refusal")
        return check(stage, **kwargs)

    def limited(stage, size):
        if stage == "direct decoder contraction":
            reserves.append(size)
            if len(reserves) == block:
                # Each bound fits the block scratch but cannot also fit the cast.
                if fault == "process":
                    supplied.limit = readings["current"] + size - cast_bytes // 2
                elif fault == "host":
                    host.available = supplied.host_reserve_bytes + size - cast_bytes // 2
                elif fault == "history":
                    readings["peak"] = supplied.limit + 1
        return reserve(stage, size)

    monkeypatch.setattr(torch, "full", positions)
    monkeypatch.setattr(supplied, "check", supplied_check)
    monkeypatch.setattr(guard, "reserve", limited)
    with pytest.raises(CleanStop) as failure:
        effects.backward(effects.logit_nodes)
    assert failure.value.__traceback__ is not None
    assert len(allocations) == block - 1
    assert len(calls) == min(1, block - 1)
    assert all(ref() is None for ref in refs)
    assert stages.count("direct decoder contraction") == block
    assert guard.limit == supplied.limit < 40_000_000_000
    assert guard.host_reserve_bytes == supplied.host_reserve_bytes
    if fault == "history":
        readings["peak"] = 0
        with pytest.raises(CleanStop, match="memory limit"):
            guard.check("latched breach")


@pytest.mark.parametrize("missing", ["gradient", "blocks"])
def test_absent_reference_work_allocates_no_cast(monkeypatch, missing):
    effects = position_case()
    effects.prepare_decoders()
    targets = effects.logit_nodes
    if missing == "gradient":
        targets = [effects.feature_nodes[0]]
    else:
        effects._decoder_blocks = [[] for _ in effects.source_layers]
    refs, calls = observe_casts(monkeypatch, effects, len(targets))
    effects.path.reserves.clear()
    effects.backward(targets)
    assert calls == refs == []
    assert not any(stage == "direct decoder contraction" for stage, _ in effects.path.reserves)


@pytest.mark.parametrize("single_block", [False, True])
def test_cast_lifetime_ends_at_last_gather(monkeypatch, single_block):
    if single_block:
        from .test_direct_reference_edges import LinearPath, effects_for, SOURCE, DIRECTION
        path = LinearPath(SOURCE, SOURCE, DIRECTION, feature_value=1.3046875)
        path.errors = [value.float() for value in path.errors]
        effects = effects_for(path)
    else:
        effects = position_case()
    effects.prepare_decoders()
    refs, _ = observe_casts(monkeypatch, effects, len(effects.logit_nodes))
    final = {id(blocks[-1][2]) for blocks in effects._decoder_blocks if blocks}
    decoders = {id(decoder) for blocks in effects._decoder_blocks for _, _, decoder in blocks}
    products, einsum = [], torch.einsum

    def contract(equation, left, right):
        if id(right) in decoders:
            live = sum(ref() is not None for ref in refs)
            assert live == (0 if id(right) in final else 1)
            products.append(live)
        return einsum(equation, left, right)

    monkeypatch.setattr(torch, "einsum", contract)
    effects.backward(effects.logit_nodes)
    assert len(products) == len(decoders)
    assert refs and all(ref() is None for ref in refs)
