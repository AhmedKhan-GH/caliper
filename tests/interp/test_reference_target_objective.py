"""Exact CPU evidence for target seeds against the immutable scalar objective."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from caliper.interp import _trace
from caliper.interp._frozen import FrozenAttention, FrozenInput, FrozenNorm, FrozenPath
from caliper.interp._reference_graph import PROFILE
from .test_reference_batch_differential import pinned_context
from .test_reference_decoder_groups import source_case, whole_layer_scores, pinned_decoder


@pytest.fixture
def original():
    """Load the independent fixed-base class and its exact objective expression."""
    source = os.environ.get("CALIPER_BASE_OBJECTIVE_SOURCE")
    if source is None:
        pytest.skip("the immutable objective source is required")
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "c49d8208c2f3f1221b42448e113d31c05b91c216f935d13894fcfedf2a093986"
    cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name == "_DirectEffects")
    backward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "backward")
    objective = next(node.value for node in ast.walk(backward) if isinstance(node, ast.Assign)
                     and any(isinstance(name, ast.Name) and name.id == "objective" for name in node.targets))
    expression = compile(ast.Expression(objective), source, "eval")
    namespace = dict(vars(_trace))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), source, "exec"), namespace)

    def scalar(path, reads, final, targets, check):
        return eval(expression, dict(torch=torch, path=path, reads=reads, final=final, targets=targets))

    return namespace["_DirectEffects"], scalar


def exact(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert (actual.dtype, actual.shape, actual.stride()) == (expected.dtype, expected.shape, expected.stride())
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def synthetic_effects(features=3, dtype=torch.bfloat16):
    """Use production frozen operators with small deterministic CPU constants."""
    path = FrozenPath.__new__(FrozenPath)
    path.device, path.dtype = torch.device("cpu"), dtype
    path.layers, path.tokens, path.width, path.mixer_width = 3, 3, 8, 8
    path.reference = path.raw_reference_input = True
    path.guard = SimpleNamespace(check=lambda stage: None, reserve=lambda stage, size: None)
    path.embedding = torch.ones(1, path.tokens, path.width, dtype=path.dtype)
    path.errors = [torch.zeros_like(path.embedding, dtype=torch.float32) for _ in range(path.layers)]
    path.decoder_biases = [torch.ones_like(path.embedding) / 8 for _ in range(path.layers)]
    path.output_offsets = [torch.zeros_like(value) for value in path.errors]
    path.norms, path.post_norms, path.attention_post_norms = [], [], []
    path.mixers, path.residual_inputs = [], []

    def norm():
        module = SimpleNamespace(weight=torch.zeros(path.width, dtype=path.dtype), eps=1e-6, bias=None)
        return FrozenNorm(module, path.embedding, qwen=True, reference=True)

    for layer in range(path.layers):
        path.norms.append(norm())
        path.post_norms.append(norm())
        path.attention_post_norms.append(norm())
        # Attention width is twelve, although residual and mixer widths are eight.
        attention = SimpleNamespace(head_dim=6,
            v_proj=SimpleNamespace(weight=torch.full((6, path.width), .03125, dtype=path.dtype), bias=None),
            o_proj=SimpleNamespace(weight=torch.full((path.width, 12), .03125, dtype=path.dtype),
                                   bias=None, in_features=12))
        pattern = torch.eye(path.tokens, dtype=path.dtype)[None, None].repeat(1, 2, 1, 1)
        mixer = FrozenAttention(attention, path.embedding, pattern)
        mixer.output.transforms = [FrozenInput(path.embedding, path.embedding, []) for _ in range(2)]
        path.mixers.append(mixer)
        before = path.embedding if layer == 0 else path.errors[layer]
        path.residual_inputs.append(FrozenInput(before, before, []))
    path.final_norm = norm()
    path.unembedding = torch.stack((torch.ones(path.width), -torch.ones(path.width))).to(path.dtype)
    path.unembedding_bias = None
    path.logits = torch.tensor([[1., 0.]], dtype=path.dtype)
    with torch.no_grad():
        _, path.reference_final = path.run(path.embedding, path.errors)
        zero = torch.zeros_like(path.embedding).expand(4, -1, -1)
        masks = torch.eye(4, dtype=path.dtype).reshape(4, 4, 1, 1)
        path.bias_reads, path.bias_final = path.run(zero, [zero] * path.layers, bias_masks=masks)
    path.feature_outputs, dictionary = [], []
    for _ in range(path.layers):
        values = torch.ones(path.tokens, features, dtype=path.dtype)
        path.feature_outputs.append(SimpleNamespace(
            indices=torch.arange(features).repeat(path.tokens, 1), activations=values,
            coefficients=torch.ones_like(values), constants=torch.zeros_like(values)))
        dictionary.append(SimpleNamespace(
            W_enc_original=torch.full((features, path.width), .125, dtype=path.dtype),
            b_enc=torch.zeros(features, dtype=path.dtype),
            W_dec=torch.ones(path.width, features, dtype=torch.float64).T))
    return _trace._DirectEffects(path, SimpleNamespace(layers=dictionary), list(range(path.tokens)),
                                SimpleNamespace(decode=lambda *args, **kwargs: "x"), profile=PROFILE)


class Products(TorchDispatchMode):
    """Record schemas, values, and strides for each matrix product."""

    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.bmm.default, torch.ops.aten.mm.default):
            self.calls.append((str(func._schema), [
                (value.shape, value.stride(), value.dtype, value.clone()) for value in (*args, output)]))
        return output


def compare_products(actual, expected):
    assert len(actual) == len(expected)
    for (schema, operands), (ref_schema, ref_operands) in zip(actual, expected):
        assert schema == ref_schema
        for operand, ref_operand in zip(operands, ref_operands):
            assert operand[:3] == ref_operand[:3]
            exact(operand[3], ref_operand[3])


def path_order(outputs):
    """Record every executed path node before objective nodes exist."""
    nodes, pending = set(), [value.grad_fn for value in outputs]
    while pending:
        node = pending.pop()
        if node is not None and node not in nodes:
            nodes.add(node)
            pending.extend(edge[0] for edge in node.next_functions)
    nodes = sorted((node for node in nodes if node.name() != "torch::autograd::AccumulateGrad"),
                   key=lambda node: node._sequence_nr())
    events, handles = [], []
    for index, node in enumerate(nodes):
        def record(gradients, index=index, name=node.name()):
            events.append((index, name, [(None if value is None else
                (value.shape, value.stride(), value.dtype, value.clone())) for value in gradients]))
        handles.append(node.register_prehook(record))
    return events, handles


def compare_order(actual, expected):
    assert [entry[:2] for entry in actual] == [entry[:2] for entry in expected]
    for (_, _, gradients), (_, _, ref_gradients) in zip(actual, expected):
        assert len(gradients) == len(ref_gradients)
        for value, reference in zip(gradients, ref_gradients):
            if reference is None:
                assert value is None
            else:
                assert value[:3] == reference[:3]
                exact(value[3], reference[3])


def evaluate(path, targets, objective):
    zeros, leaves = {}, []
    for source in [path.embedding, *path.errors]:
        if source.dtype not in zeros:
            zeros[source.dtype] = source.new_zeros(len(targets), path.tokens, path.width)
        leaves.append(zeros[source.dtype].detach().requires_grad_(True))
    with Products() as products:
        reads, final = path.run(leaves[0], leaves[1:], bias_masks=(0., 0., 0., 0.))
        order, handles = path_order([*reads, final])
        try:
            loss = objective(path, reads, final, targets, lambda stage: None)
            gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
        finally:
            for handle in handles:
                handle.remove()
    return gradients, order, products.calls


@pytest.mark.parametrize("signed_zero", [False, True])
@pytest.mark.parametrize("layers", [(0,), (3,), (0, 0), (3, 3), (2, 0, 3, 2, 0),
                                     (0, 2, 0, 2), (3, 0, 3, 1), (0, 1, 2, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_leaves_path_order_and_products_match_scalar_oracle(original, layers, dtype, signed_zero):
    effects = synthetic_effects(dtype=dtype)
    path = effects.path
    generator = torch.Generator().manual_seed(241)
    targets = [SimpleNamespace(layer=layer, position=(row % 2),
        direction=torch.randn(2 * path.width, generator=generator).to(dtype)[::2])
        for row, layer in enumerate(layers)]
    if signed_zero:
        for target in targets:
            target.direction[::2] = -0.
            target.direction[1::2] = 0.
    expected, order, products = evaluate(path, targets, original[1])
    actual, candidate_order, candidate_products = evaluate(path, targets, _trace._reference_target_objective)
    for value, reference in zip(actual, expected):
        exact(value, reference)
    compare_order(candidate_order, order)
    compare_products(candidate_products, products)
    assert [value is not None for value in actual] == [layer <= max(layers) for layer in range(path.layers + 1)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("count", [1, 2, 4])
def test_signed_zero_and_finite_seed_extremes(original, dtype, count):
    info = torch.finfo(dtype)
    direction = torch.tensor([-0., 0., info.tiny, -info.tiny, info.max, -info.max, info.eps, -info.eps], dtype=dtype)
    # Include the smallest representable subnormal and a strided direction.
    direction[2:4] *= info.eps
    backing = torch.stack((direction, direction), dim=1).flatten()
    targets = [SimpleNamespace(layer=0, position=row % 2, direction=backing[::2]) for row in range(count)]
    path = SimpleNamespace(device=torch.device("cpu"), layers=1, width=8, guard=SimpleNamespace(reserve=lambda *args: None))
    results = []
    for objective in (original[1], _trace._reference_target_objective):
        value = torch.zeros(count, 2, 8, dtype=dtype, requires_grad=True)
        result = torch.autograd.grad(objective(path, [value], value, targets, lambda stage: None), value)[0]
        results.append(result)
    exact(results[1], results[0])
    assert bool(results[0][0, 0, 0].signbit()) == (count == 1)
    assert torch.isfinite(results[0]).all()
    assert results[0][0, 0, 2] != 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("selection", ["mixed", "one", "repeated"])
def test_full_edges_biases_and_pinned_source_hooks(original, pinned_context, pinned_decoder, monkeypatch, dtype, selection):
    args = source_case(dtype, [[2, 1, 2], [1, 2, 2]], width=7)
    expected_effects = original[0](*args, profile=PROFILE)
    effects = _trace._DirectEffects(*args, profile=PROFILE)
    indices = [effects.feature_nodes[-1], effects.feature_nodes[0], effects.logit_nodes[0],
               effects.feature_nodes[-2], effects.logit_nodes[-1]]
    if selection == "one":
        indices = indices[-1:]
    elif selection == "repeated":
        indices = [effects.logit_nodes[0]] * 4
    records, grad = [], torch.autograd.grad

    def record(*args, **kwargs):
        result = grad(*args, **kwargs)
        records.append(result)
        return result

    monkeypatch.setattr(torch.autograd, "grad", record)
    with Products() as baseline_products:
        expected, accounts = expected_effects.backward(indices)
    with Products() as candidate_products:
        actual, diagnostics = effects.backward(indices)
    exact(actual, expected)
    assert json.dumps(diagnostics) == json.dumps(accounts)
    for value, reference in zip(records[1], records[0]):
        exact(value, reference)
    compare_products(candidate_products.calls, baseline_products.calls)
    # These hooks use the immutable objective's gradients, not candidate gradients.
    pinned, _ = whole_layer_scores(expected_effects, records[0], pinned_context, pinned_decoder)
    exact(actual, pinned)
    assert all(any(account["biases"][name] != 0 for account in accounts) for name in
               ("decoder", "normalization", "attention", "intervention", "unembedding"))
    if selection == "mixed":
        assert any(account["biases"]["encoder"] != 0 for account in accounts)


class StorageEvents(TorchDispatchMode):
    """Count unique storage owners and their ATen-visible CPU lifetimes."""

    def __init__(self, external):
        self.external = {value.untyped_storage()._cdata for value in external}
        self.owners, self.calls, self.peak = {}, [], 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from torch.multiprocessing.reductions import StorageWeakRef
        output = func(*args, **(kwargs or {}))
        if isinstance(output, torch.Tensor):
            owner = output.untyped_storage()
            identity, size = owner._cdata, owner.nbytes()
            fresh = identity not in self.external and identity not in self.owners
            if fresh:
                self.owners[identity] = (StorageWeakRef(owner), size)
            self.calls.append((str(func), tuple(output.shape), output.stride(), size, fresh, identity))
            self.peak = max(self.peak, sum(size for ref, size in self.owners.values() if not ref.expired()))
        return output

    def summary(self):
        return dict(allocations=len(self.owners), cumulative=sum(size for _, size in self.owners.values()),
                    peak=self.peak, live=sum(size for ref, size in self.owners.values() if not ref.expired()))


@pytest.mark.parametrize("layers", [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 2)])
def test_scatter_storage_and_lifetimes(original, layers):
    batch, tokens, width = len(layers), 3, 8
    targets = [SimpleNamespace(layer=layer, position=row % 2, direction=torch.ones(width, dtype=torch.bfloat16))
               for row, layer in enumerate(layers)]
    path = SimpleNamespace(device=torch.device("cpu"), layers=2, width=width, guard=SimpleNamespace(reserve=lambda *args: None))
    records = []
    for name, objective in (("original", original[1]), ("grouped", _trace._reference_target_objective)):
        values = [torch.zeros(batch, tokens, width, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        with StorageEvents([*values, *(target.direction for target in targets)]) as events:
            loss = objective(path, values[:2], values[2], targets, lambda stage: None)
            forward = events.summary()
            gradients = torch.autograd.grad(loss, values, allow_unused=True)
        records.append((events, gradients))
        full = [call for call in events.calls if call[1] == (batch, tokens, width) and call[4]]
        if name == "original":
            scatters = [call for call in full if call[0] == "aten.select_backward.default"]
            assert len(scatters) == batch
            assert sum(call[3] for call in scatters) == 2 * batch * batch * tokens * width
        else:
            assert sum(call[0] in ("aten.select_backward.default", "aten.index_put.default") for call in full) == len(set(layers))
            # index_put is out of place. Its zero input has separate storage.
            grouped_layers = sum(layers.count(layer) > 1 for layer in set(layers))
            zeros = [call for call in full if call[0] == "aten.new_zeros.default"]
            assert len(zeros) == grouped_layers
            assert all(call[3] == batch * tokens * width * 2 for call in full)
        after_backward = events.summary()
        del loss, gradients, values
        print("OBJECTIVE_STORAGE", json.dumps(dict(layers=layers, variant=name, forward=forward, backward=after_backward)))
    for value, reference in zip(records[1][1], records[0][1]):
        exact(value, reference)
    for events, gradients in records:
        assert events.summary()["live"] == sum(value.untyped_storage().nbytes() for value in gradients if value is not None)
    del value, reference, gradients
    events_only = [entry[0] for entry in records]
    records.clear()
    assert all(events.summary()["live"] == 0 for events in events_only)


@pytest.mark.parametrize("stage", ["target batch admission", "target forward", "target backward",
                                     "direct decoder block", "finite block readback", "direct edge products"])
@pytest.mark.parametrize("exception", [_trace.CleanStop, _trace.TraceTimeout])
def test_existing_stops_keep_exception_type(original, stage, exception):
    for cls in (original[0], _trace._DirectEffects):
        args = source_case(torch.bfloat16, [[2, 1, 2], [1, 2, 2]], width=7)
        effects = cls(*args, profile=PROFILE)
        effects.prepare_decoders()
        stages = []

        def check(name):
            stages.append(name)
            if name == stage:
                raise exception("objective audit stop")

        effects.check = check
        with pytest.raises(exception, match="objective audit stop"):
            effects.backward(effects.logit_nodes * 2)
        assert stages[-1] == stage


@pytest.mark.parametrize("stage", ["target objective metadata", "target objective tensors"])
@pytest.mark.parametrize("fault", ["process", "host", "history", "supplied"])
def test_new_buffers_require_real_guard_admission(monkeypatch, stage, fault):
    from caliper.interp import _frozen
    from .test_memory_phases import process_guard

    effects = synthetic_effects()
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
    reserve, seen = guard.reserve, []

    def limited(name, size):
        if name == stage:
            seen.append(size)
            assert size > 0
            if fault == "process":
                supplied.limit = readings["current"] + size - 1
            elif fault == "host":
                host.available = supplied.host_reserve_bytes + size - 1
            elif fault == "history":
                readings["peak"] = supplied.limit + 1
            else:
                raise _trace.CleanStop("supplied refusal")
        return reserve(name, size)

    tensors, tensor = [], torch.tensor

    def allocate(*args, **kwargs):
        tensors.append(args)
        return tensor(*args, **kwargs)

    monkeypatch.setattr(guard, "reserve", limited)
    monkeypatch.setattr(torch, "tensor", allocate)
    with pytest.raises(_trace.CleanStop):
        effects.backward(effects.logit_nodes * 2)
    assert len(seen) == 1
    assert tensors == []
    assert guard.limit == supplied.limit < 40_000_000_000
    assert guard.host_reserve_bytes == supplied.host_reserve_bytes
    if fault == "history":
        readings["peak"] = 0
        with pytest.raises(_trace.CleanStop, match="memory limit"):
            guard.check("latched breach")


def test_ordinary_tracing_keeps_original_objective(original, monkeypatch):
    from .test_direct_decoder_cache import fixture
    effects = fixture(torch.float32)
    effects.reference = False
    baseline = original[0](effects.path, effects.dictionary, [0, 1, 2],
                           SimpleNamespace(decode=lambda *args, **kwargs: "x"))
    monkeypatch.setattr(_trace, "_reference_target_objective", lambda *args: pytest.fail("ordinary objective changed"))
    expected, accounts = baseline.backward(baseline.logit_nodes)
    actual, diagnostics = effects.backward(effects.logit_nodes)
    exact(actual, expected)
    assert json.dumps(diagnostics) == json.dumps(accounts)


@pytest.mark.parametrize("exception", [_trace.CleanStop, _trace.TraceTimeout])
def test_partial_objective_releases_new_storage_on_stop(original, exception):
    effects = synthetic_effects()
    path = effects.path
    targets = [SimpleNamespace(layer=layer, position=0, direction=path.embedding[0, 0])
               for layer in (0, 1, 0, 1)]
    values = [torch.zeros(4, path.tokens, path.width, dtype=path.dtype, requires_grad=True) for _ in range(4)]
    calls = []

    def check(stage):
        calls.append(stage)
        if stage == "target objective layer" and calls.count(stage) == 2:
            raise exception("partial objective stop")

    with StorageEvents([*values, *(target.direction for target in targets)]) as events:
        with pytest.raises(exception, match="partial objective stop") as failure:
            _trace._reference_target_objective(path, values[:3], values[3], targets, check)
    assert failure.value.__traceback__ is not None
    assert events.owners
    assert events.summary()["live"] == 0


@pytest.mark.parametrize("field", ["expected", "direction"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_accounts_keep_original_refusal(original, field, value):
    from dataclasses import replace
    for cls in (original[0], _trace._DirectEffects):
        effects = cls(*source_case(torch.bfloat16, [[2, 1, 2], [1, 2, 2]], width=7), profile=PROFILE)
        target = effects.target

        def invalid(index):
            result = target(index)
            return replace(result, **{field: torch.full_like(getattr(result, field), value)})

        effects.target = invalid
        with pytest.raises(ValueError, match="must be finite"):
            effects.backward(effects.logit_nodes * 2)


@pytest.mark.parametrize("write_dtypes", [(torch.bfloat16, torch.float32, torch.bfloat16),
                                          (torch.float32, torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("layers", [(1, 3, 0, 1, 3), (0, 1, 0)])
def test_mixed_write_dtypes_preserve_all_leaf_bits(original, write_dtypes, layers):
    path = synthetic_effects().path
    path.errors = [value.to(dtype) for value, dtype in zip(path.errors, write_dtypes)]
    targets = [SimpleNamespace(layer=layer, position=row % 2,
        direction=torch.tensor([.125, -.375, 0., -0., 2., -3., .0078125, -1.], dtype=path.dtype))
        for row, layer in enumerate(layers)]
    expected, order, products = evaluate(path, targets, original[1])
    actual, candidate_order, candidate_products = evaluate(path, targets, _trace._reference_target_objective)
    for value, reference in zip(actual, expected):
        exact(value, reference)
    compare_order(candidate_order, order)
    compare_products(candidate_products, products)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_noncontiguous_reads_preserve_leaf_strides(original, dtype):
    targets = [SimpleNamespace(layer=0, position=row % 2,
        direction=torch.tensor([-.5, 0., -0., .25, 2., -3., .125, 1.], dtype=dtype)) for row in range(3)]
    path = SimpleNamespace(device=torch.device("cpu"), layers=1, width=8,
                           guard=SimpleNamespace(reserve=lambda *args: None))
    gradients = []
    for objective in (original[1], _trace._reference_target_objective):
        leaf = torch.zeros(3, 8, 2, dtype=dtype, requires_grad=True)
        read = leaf.transpose(1, 2)
        assert not read.is_contiguous()
        loss = objective(path, [read], read, targets, lambda stage: None)
        gradients.append(torch.autograd.grad(loss, leaf)[0])
    exact(gradients[1], gradients[0])
