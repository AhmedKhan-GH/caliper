"""Independent CPU source and geometry evidence for bounded decoder groups."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from caliper.interp import _trace
from .test_reference_batch_differential import pinned_context
from .test_reference_gradient_cast import Geometry


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_restricted_groups_preserve_pinned_matrix_operands(pinned_context, dtype, batch):
    generator = torch.Generator().manual_seed(144)
    gradient = torch.randn(batch, 5, 37, generator=generator).to(dtype)
    vectors = torch.randn(13, 37, generator=generator).to(dtype)
    positions = torch.tensor([0, 1, 1, 3, 3, 3, 4, 0, 0, 4, 4, 4, 4])
    baseline, original = pinned_scores(pinned_context, gradient, vectors, positions, [1, 2, 3, 1, 2, 4])
    grouped, candidate = pinned_scores(pinned_context, gradient, vectors, positions, [1, 5, 1, 6])
    original_products = per_source_products(original)
    candidate_products = per_source_products(candidate)
    assert len(original_products) == len(candidate_products) == len(vectors)
    for before, after in zip(original_products, candidate_products):
        for left, right in zip(before, after):
            assert left.shape == right.shape
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(grouped, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_singleton_boundary_has_independent_geometry(pinned_context, batch):
    gradient = torch.arange(batch * 3 * 7, dtype=torch.float32).reshape(batch, 3, 7)
    vectors = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7)
    positions = torch.tensor([0, 1, 2])
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


@pytest.fixture
def baseline_class():
    """Compile the unchanged class from the independently supplied source base."""
    source = os.environ.get("CALIPER_BASE_TRACE_SOURCE")
    if source is None:
        pytest.skip("the unoptimized source base is required for this oracle")
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "9888ae544556e282b5cdfc65584d090f9f435b285190bc08a0893fc5fe2399a4"
    cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name == "_DirectEffects")
    namespace = dict(vars(_trace))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), source, "exec"), namespace)
    return namespace["_DirectEffects"], namespace


@pytest.fixture
def pinned_decoder(pinned_context):
    """Compile the actual pinned sparse decoder and its row selection method."""
    source = Path(os.environ["CALIPER_PINNED_ATTRIBUTION_SOURCE"])
    source = source.parent.parent / "transcoder" / "single_layer_transcoder.py"
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "8fb5719906c96653efed86353bbf83fd7217d646edf9f7c05e6ab157cd1a55e7"
    cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name == "SingleLayerTranscoder")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("decode_sparse", "_get_decoder_vectors")]
    assert len(methods) == 2
    namespace = dict(torch=torch, np=np)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)

    def decode(rows, activations):
        owner = SimpleNamespace(W_dec=rows, dtype=activations.dtype, lazy_decoder=False,
                                W_skip=None, d_model=rows.shape[1], b_dec=rows.new_zeros(rows.shape[1]))
        owner._get_decoder_vectors = lambda ids: namespace["_get_decoder_vectors"](owner, ids)
        return namespace["decode_sparse"](owner, activations)[1]

    return decode


def source_case(dtype, counts, width=37):
    """Use a small linear path with distinct positions and all bias categories."""
    generator = torch.Generator().manual_seed(910)

    def values(*shape, dtype=dtype):
        return (torch.randint(-16, 17, shape, generator=generator).to(dtype) / 16)

    layers, tokens, columns = len(counts), len(counts[0]), 7
    path = SimpleNamespace(device=torch.device("cpu"), dtype=dtype, layers=layers, tokens=tokens,
                           width=width, mixer_width=width, raw_reference_input=True)
    path.embedding = values(1, tokens, width)
    path.errors = [values(1, tokens, width, dtype=torch.float32) for _ in counts]
    path.decoder_biases = [values(1, tokens, width, dtype=torch.float32) for _ in counts]
    path.bias_final = values(4, tokens, width)
    path.bias_reads = [values(4, tokens, width) for _ in counts]
    path.logits = values(1, 3)
    directions = values(3, width)
    path.logit_target = lambda token, **kwargs: (directions[token], path.embedding.new_tensor((token + 1) / 8))
    path.reserves = []
    path.guard = SimpleNamespace(reserve=lambda *args: path.reserves.append(args), check=lambda *args: None)
    factors = torch.arange(1, tokens + 1, dtype=torch.float32).reshape(1, tokens, 1) / 4

    def run(embedding, writes, **kwargs):
        x, reads = embedding, []
        for layer, write in enumerate(writes):
            reads.append(x + x.sum(1, keepdim=True))
            x = x + write * (factors + layer)
        return reads, x.sum(1, keepdim=True).expand(-1, tokens, -1).to(dtype)

    path.run = run
    _, path.reference_final = run(path.embedding, path.errors)
    path.feature_outputs, weights = [], []
    for layer_counts in counts:
        indices = torch.stack([torch.randperm(columns, generator=generator) for _ in layer_counts])
        activations = path.embedding.new_zeros(tokens, columns)
        for position, count in enumerate(layer_counts):
            activations[position, :count] = torch.tensor([.1, -1.3, .5, 2.1, -.25, 1., -.7][:count], dtype=dtype)
        path.feature_outputs.append(SimpleNamespace(indices=indices, activations=activations,
            coefficients=values(tokens, columns), constants=values(tokens, columns)))
        encoder = values(columns, width)
        decoder = values(columns, width, dtype=torch.float64).T.contiguous().T
        weights.append(SimpleNamespace(W_dec=decoder, W_enc=encoder,
                                       W_enc_original=encoder, b_enc=values(columns)))
    dictionary = SimpleNamespace(layers=weights)
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "x")
    return path, dictionary, list(range(tokens)), tokenizer


def whole_layer_scores(effects, gradients, context_class, decode):
    """Build canonical sparse source rows without cache metadata or its planner."""
    path = effects.path
    sparse, vectors, source_rows = [], [], []
    for layer, weights in enumerate(effects.dictionary.layers):
        dense = path.embedding.new_zeros(path.tokens, len(weights.W_dec))
        output = path.feature_outputs[layer]
        dense.scatter_(1, output.indices, output.activations)
        acts = dense.to_sparse().coalesce()
        sparse.append(acts)
        vectors.append(decode(weights.W_dec, acts))
        nodes = {(node["position"], node["feature"]): index for index, node in enumerate(effects.nodes)
                 if node["kind"] == "feature" and node["layer"] == layer}
        source_rows.extend(nodes[tuple(coordinate)] for coordinate in acts.indices().T.tolist())
    activations = torch.stack(sparse).coalesce()
    feature_count = activations._nnz()
    context = context_class(activations, torch.stack([error[0] for error in path.errors]),
        path.embedding[0], torch.cat(vectors), torch.empty(0), torch.arange(feature_count),
        activations.indices()[:2], path.logits)
    context._batch_buffer = path.embedding.new_zeros(context._row_size, len(gradients[0]))
    for name, hook in context._make_attribution_hooks("synthetic"):
        layer = None if name == "hook_embed" else int(name.split(".")[1])
        gradient = gradients[0 if layer is None else layer + 1]
        if gradient is not None:
            hook(gradient, None)
    source_rows.extend(index for index, node in enumerate(effects.nodes) if node["kind"] == "error")
    source_rows.extend(range(path.tokens))
    expected = path.embedding.new_zeros(len(effects.nodes), len(gradients[0]))
    expected[source_rows] = context._batch_buffer
    return expected, dict(zip(source_rows[:feature_count], torch.cat(vectors)))


LAYOUTS = [
    [[2, 2, 0, 1, 2, 3, 1], [1, 2, 2, 0, 3, 2, 1]],
    [[7, 0, 2, 3, 1, 2, 2], [2, 3, 1, 2, 2, 0, 7]],
    [[0, 0, 0, 0, 0, 0, 0], [2, 2, 0, 1, 2, 3, 1]],
    [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("row_bound", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("counts", LAYOUTS)
def test_groups_match_unoptimized_backward_and_whole_layer_sources(
        baseline_class, pinned_context, pinned_decoder, monkeypatch, dtype, batch, row_bound, counts):
    baseline, namespace = baseline_class
    args = source_case(dtype, counts)
    effects = _trace._DirectEffects(*args, profile=_trace._reference_graph.PROFILE)
    original = baseline(*args, profile=_trace._reference_graph.PROFILE)
    bound = row_bound * effects.path.width
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", bound)
    namespace["_ENCODER_BLOCK_ELEMENTS"] = bound
    pool = effects.logit_nodes + effects.feature_nodes[:2] + effects.feature_nodes[-2:]
    targets = (pool * batch)[:batch]
    if batch > 1:
        targets.reverse()
    nodes, columns = deepcopy(effects.nodes), dict(effects.feature_columns)
    gradients, grad = [], torch.autograd.grad

    def record(*args, **kwargs):
        result = grad(*args, **kwargs)
        gradients.append(result)
        return result

    monkeypatch.setattr(torch.autograd, "grad", record)
    expected, accounts = original.backward(targets)
    actual, actual_accounts = effects.backward(targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype == dtype
    assert actual_accounts == accounts
    for observed, reference in zip(gradients[1], gradients[0]):
        if reference is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    pinned, scaled = whole_layer_scores(effects, gradients[0], pinned_context, pinned_decoder)
    torch.testing.assert_close(actual, pinned, rtol=0, atol=0)
    cached_rows, refs, storage = [], [], set()
    for layer, blocks in enumerate(effects._decoder_blocks):
        old_singles = [rows for rows, _, decoder in original._decoder_blocks[layer] if len(decoder) == 1]
        new_singles = [rows for rows, _, decoder in blocks if len(decoder) == 1]
        assert old_singles == new_singles
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
    assert again_accounts == actual_accounts
    if cached_rows:
        del decoder, vector
    del blocks
    effects.release_decoders()
    assert all(ref() is None for ref in refs)


def contraction_evidence(effects, targets, monkeypatch):
    effects.prepare_decoders()
    decoders = {id(decoder): rows for blocks in effects._decoder_blocks for rows, _, decoder in blocks}
    calls, einsum = [], torch.einsum

    def record(equation, left, right):
        if id(right) not in decoders:
            return einsum(equation, left, right)
        assert equation == "bpd,pd->pb"
        with Geometry() as geometry:
            result = einsum(equation, left, right)
        calls.append((decoders[id(right)], left.clone(), right.clone(), geometry.calls))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch, "einsum", record)
        result = effects.backward(targets)
    return result, calls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("row_bound", [3, 7])
def test_actual_group_operands_keep_source_and_target_order(
        baseline_class, pinned_context, monkeypatch, dtype, batch, row_bound):
    baseline, namespace = baseline_class
    args = source_case(dtype, LAYOUTS[0])
    effects = _trace._DirectEffects(*args, profile=_trace._reference_graph.PROFILE)
    original = baseline(*args, profile=_trace._reference_graph.PROFILE)
    bound = row_bound * effects.path.width
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", bound)
    namespace["_ENCODER_BLOCK_ELEMENTS"] = bound
    targets = (effects.logit_nodes[::-1] * batch)[:batch]
    (expected, _), old_calls = contraction_evidence(original, targets, monkeypatch)
    (actual, _), new_calls = contraction_evidence(effects, targets, monkeypatch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if row_bound == 7:
        assert len(new_calls) < len(old_calls)
    old_rows, new_rows = {}, {}
    for calls, rows_by_id in ((old_calls, old_rows), (new_calls, new_rows)):
        for rows, gathered, vectors, geometry in calls:
            # The actual pinned hook sees these same gathered coordinates.
            reference, reference_geometry = pinned_scores(pinned_context, gathered, vectors,
                torch.arange(len(vectors)), [len(vectors)])
            products = per_source_products(geometry)
            reference_products = per_source_products(reference_geometry)
            assert len(products) == len(reference_products) == len(vectors)
            for index, row in enumerate(range(rows.start, rows.stop)):
                for operand, pinned_operand in zip(products[index], reference_products[index]):
                    assert operand.shape == pinned_operand.shape
                    torch.testing.assert_close(operand, pinned_operand, rtol=0, atol=0)
                rows_by_id[row] = (gathered[:, index], vectors[index], products[index])
            torch.testing.assert_close(actual[rows], reference, rtol=0, atol=0)
    assert list(old_rows) == list(new_rows) == effects.feature_nodes
    for row in old_rows:
        before, after = old_rows[row], new_rows[row]
        for left, right in zip((*before[:2], *before[2]), (*after[:2], *after[2])):
            assert left.shape == right.shape
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def grouped_effects(monkeypatch):
    effects = _trace._DirectEffects(*source_case(torch.bfloat16, LAYOUTS[0]),
                                   profile=_trace._reference_graph.PROFILE)
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 7 * effects.path.width)
    return effects


def test_group_allocation_has_bounded_unique_storage(monkeypatch, baseline_class):
    baseline, namespace = baseline_class
    effects = grouped_effects(monkeypatch)
    path = effects.path
    namespace["_ENCODER_BLOCK_ELEMENTS"] = 7 * path.width
    original = baseline(path, effects.dictionary, list(range(path.tokens)),
                        SimpleNamespace(decode=lambda *a, **kw: "x"), profile=_trace._reference_graph.PROFILE)
    original.prepare_decoders()
    old_reserve = dict(path.reserves)["direct decoder cache"]
    effects.prepare_decoders()
    new_reserve = dict(path.reserves)["direct decoder cache"]
    assert new_reserve == old_reserve + 8 * len(effects.feature_nodes)
    blocks = [block for layer in effects._decoder_blocks for block in layer]
    decoder_bytes = sum(decoder.untyped_storage().nbytes() for _, _, decoder in blocks)
    assert decoder_bytes == len(effects.feature_nodes) * path.width * path.embedding.element_size()
    positions = [position for _, position, _ in blocks if isinstance(position, torch.Tensor)]
    assert positions
    assert sum(value.untyped_storage().nbytes() for value in positions) <= 8 * len(effects.feature_nodes)
    assert all(decoder.numel() <= 7 * path.width for _, _, decoder in blocks)


def test_single_feature_targets_and_all_bias_categories(monkeypatch, baseline_class):
    baseline, namespace = baseline_class
    effects = grouped_effects(monkeypatch)
    path = effects.path
    namespace["_ENCODER_BLOCK_ELEMENTS"] = 7 * path.width
    original = baseline(path, effects.dictionary, list(range(path.tokens)),
                        SimpleNamespace(decode=lambda *a, **kw: "x"), profile=_trace._reference_graph.PROFILE)
    targets = [effects.feature_nodes[0], effects.feature_nodes[-1], effects.logit_nodes[-1]]
    actual_accounts = []
    for target in targets:
        expected, accounts = original.backward([target])
        actual, diagnostics = effects.backward([target])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert accounts == diagnostics
        actual_accounts.extend(diagnostics)
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
