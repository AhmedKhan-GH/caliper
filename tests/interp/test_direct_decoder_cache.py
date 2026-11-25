"""Independent CPU checks for retained reference decoder rows."""
from math import fsum
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from caliper.interp import _trace
from caliper.interp._memory import CleanStop
from caliper.interp._reference_graph import PROFILE
from .test_direct_reference_edges import DIRECTION, SOURCE, rounded_dot


class SourceGathers(TorchDispatchMode):
    def __init__(self, tensors):
        self.storages = {x.untyped_storage().data_ptr() for x in tensors}
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (func == torch.ops.aten.index.Tensor
                and args[0].untyped_storage().data_ptr() in self.storages):
            self.count += 1
        return func(*args, **(kwargs or {}))


class SumPath:
    """Each read sums all positions of each earlier independent source."""

    def __init__(self, dtype):
        self.device, self.dtype = torch.device("cpu"), dtype
        self.layers, self.tokens, self.width, self.mixer_width = 2, 3, 4, 4
        self.embedding = torch.tensor([[SOURCE, DIRECTION, SOURCE]], dtype=dtype)
        self.errors = [self.embedding / 4, -self.embedding / 8]
        self.decoder_biases = [torch.zeros_like(self.embedding) for _ in range(self.layers)]
        self.logits = torch.tensor([[1., 0.]], dtype=dtype)
        self.direction = torch.tensor(DIRECTION, dtype=dtype)
        self.bias_final = self.embedding.new_zeros(4, self.tokens, self.width)
        self.bias_reads = [self.bias_final.clone() for _ in range(self.layers)]
        self.raw_reference_input = True
        self.reserves = []
        self.guard = SimpleNamespace(reserve=lambda *args: self.reserves.append(args), check=lambda *args: None)
        self.feature_outputs = []
        for layer in range(self.layers):
            activations = torch.tensor([[1.3046875, 0., -2., .5, 3.],
                                        [0., 0., 0., 0., 0.],
                                        [-.25, 1., 0., 2., 0.]], dtype=dtype) * (layer + 1)
            self.feature_outputs.append(SimpleNamespace(
                indices=torch.tensor([[4, 0, 2, 1, 3], [0, 1, 2, 3, 4], [1, 4, 0, 3, 2]]),
                activations=activations, coefficients=torch.ones_like(activations),
                constants=torch.zeros_like(activations)))
        _, self.reference_final = self.run(self.embedding, self.errors)

    def logit_target(self, token, **kwargs):
        return self.direction * (1 if token == 0 else -1), self.embedding.new_zeros(())

    def run(self, embedding, writes, **kwargs):
        reads = []
        sources = [embedding]
        for write in writes:
            reads.append(sum(sources).sum(1, keepdim=True).expand(-1, self.tokens, -1))
            sources.append(write)
        return reads, sum(sources).sum(1, keepdim=True).expand(-1, self.tokens, -1)


def fixture(dtype=torch.bfloat16):
    path = SumPath(dtype)
    layers = []
    for _ in range(path.layers):
        # A wider, strided dictionary exercises the gather and cast scratch.
        decoder = torch.tensor([SOURCE, DIRECTION, SOURCE, DIRECTION, SOURCE], dtype=torch.float64)
        decoder = decoder.T.contiguous().T
        encoder = torch.tensor([DIRECTION, SOURCE, DIRECTION, SOURCE, DIRECTION], dtype=dtype)
        layers.append(SimpleNamespace(W_dec=decoder, W_enc=encoder, W_enc_original=encoder,
                                      b_enc=torch.zeros(5, dtype=dtype)))
    effects = _trace._DirectEffects(path, SimpleNamespace(layers=layers), [0, 1, 2],
                                   SimpleNamespace(decode=lambda *args, **kwargs: "x"), profile=PROFILE)
    return effects


def oracle(effects, targets):
    """Sum binary products independently of tensor contraction and cache code."""
    path = effects.path
    result = path.embedding.new_zeros(len(effects.nodes), len(targets))
    for column, target in enumerate(targets):
        node = effects.nodes[target]
        direction = (path.direction * (1 if node["token_id"] == 0 else -1)
                     if node["kind"] == "logit" else
                     effects.dictionary.layers[node["layer"]].W_enc_original[node["feature"]])
        for row, source in enumerate(effects.nodes):
            if source["kind"] == "logit" or source["layer"] >= node["layer"]:
                continue
            if source["kind"] == "token":
                vector = path.embedding[0, source["position"]]
            elif source["kind"] == "error":
                vector = path.errors[source["layer"]][0, source["position"]]
            else:
                raw = effects.dictionary.layers[source["layer"]].W_dec[source["feature"]].to(path.dtype)
                vector = raw.new_tensor([float(x) * source["activation"] for x in raw])
            result[row, column] = rounded_dot(vector, direction)
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_cache_reuses_exact_sources_across_batches_and_block_tails(monkeypatch, dtype):
    effects = fixture(dtype)
    monkeypatch.setattr(_trace, "_ENCODER_BLOCK_ELEMENTS", 8)
    targets = effects.logit_nodes + effects.feature_nodes
    expected = oracle(effects, targets)
    with SourceGathers([layer.W_dec for layer in effects.dictionary.layers]) as gathers:
        weights, diagnostics = effects.backward(targets)
        first_count = gathers.count
        split = [effects.backward(targets[:3]), effects.backward(targets[3:])]
        assert gathers.count == first_count > 0
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([x[0] for x in split], 1), expected, rtol=0, atol=0)
    assert diagnostics == split[0][1] + split[1][1]
    for column, row in enumerate(diagnostics):
        total = fsum(float(x) for x in expected[:, column])
        account_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        magnitude = fsum(abs(float(x)) for x in expected[:, column]) + abs(row["expected"])
        bound = torch.finfo(account_dtype).eps * (len(effects.nodes) + 1) * magnitude
        assert row["incoming"] == pytest.approx(total, rel=0, abs=bound)
        assert row["absolute_residual"] == pytest.approx(abs(total - row["expected"]), rel=0, abs=bound)
    assert len([x for x in effects.path.reserves if x[0] == "direct decoder cache"]) == 1
    cache_bytes = sum(x[2].numel() * x[2].element_size() for blocks in effects._decoder_blocks for x in blocks)
    assert cache_bytes == len(effects.feature_nodes) * effects.path.width * effects.path.embedding.element_size()
    assert dict(effects.path.reserves)["direct decoder cache"] > cache_bytes
    refs = [weakref.ref(x[2]) for blocks in effects._decoder_blocks for x in blocks]
    effects.release_decoders()
    assert all(ref() is None for ref in refs)


def test_cache_refusal_precedes_dictionary_gathers():
    effects = fixture()
    def refuse(stage, size):
        assert stage == "direct decoder cache" and size > 0
        raise CleanStop("cache refusal")
    effects.path.guard.reserve = refuse
    with SourceGathers([layer.W_dec for layer in effects.dictionary.layers]) as gathers:
        with pytest.raises(CleanStop, match="cache refusal"):
            effects.prepare_decoders()
        assert gathers.count == 0
    assert effects._decoder_blocks is None


def test_incomplete_cache_does_not_survive_a_deadline(monkeypatch):
    effects = fixture()
    count = 0
    def stop(stage):
        nonlocal count
        if stage == "direct decoder cache block":
            count += 1
            if count == 2:
                raise _trace.TraceTimeout("cache deadline")
    effects.check = stop
    with pytest.raises(_trace.TraceTimeout, match="cache deadline"):
        effects.prepare_decoders()
    assert effects._decoder_blocks is None
    effects.check = lambda stage: None
    actual, _ = effects.backward(effects.logit_nodes)
    assert torch.equal(actual, oracle(effects, effects.logit_nodes))


@pytest.mark.parametrize("failure", [None, "direct decoder block"])
def test_trace_releases_cache_on_success_and_failure(monkeypatch, failure):
    from .fixture import TinyLM, TinyTokenizer
    from .test_trace import exact_dictionary
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2).cpu().eval()
    tracer = _trace.Tracer(model, TinyTokenizer(), exact_dictionary(model))
    original = _trace._DirectEffects.prepare_decoders
    refs = []
    def prepare(effects):
        original(effects)
        refs.extend(weakref.ref(x[2]) for blocks in effects._decoder_blocks for x in blocks)
        if failure:
            def stop(stage):
                if stage == failure:
                    raise _trace.TraceTimeout("edge deadline")
            effects.check = stop
    monkeypatch.setattr(_trace._DirectEffects, "prepare_decoders", prepare)
    if failure:
        with pytest.raises(_trace.TraceTimeout, match="edge deadline"):
            tracer.trace("abc", max_targets=0, profile=PROFILE)
        assert not tracer._cache and tracer._forward_cache is None
    else:
        tracer.trace("abc", max_targets=0, profile=PROFILE)
    assert refs and all(ref() is None for ref in refs)
