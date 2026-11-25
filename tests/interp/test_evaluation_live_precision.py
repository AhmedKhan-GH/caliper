"""Mixed precision evaluation with independent CPU references and failure cleanup."""
from contextlib import contextmanager

import pytest
import torch
from torch.nn import functional as F

from caliper.interp import capture, load_dictionary
from caliper.interp import _frozen
from caliper.interp._context import context_policy, prepare_contexts
from caliper.interp._memory import CleanStop
from tools.interp import eval_dictionary as evaluation, evaluation_protocol as protocol
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream
from .fixture import TinyLM
from .test_dictionary import write_fixture
from .test_gemma_runtime import gemma_model, runtime_dictionary
from .test_points import hooks
from .test_research_evaluation import guard, tokenizer
from .test_trace import random_dictionary
from .test_training_stream import documents


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def resources(family, dtype):
    model = (gemma_model() if family == "gemma" else
             TinyLM(layers=2, width=8, heads=2, latents=12, k=3)).to(dtype=dtype)
    dictionary = (runtime_dictionary(model) if family == "gemma" else
                  random_dictionary(model, latents=12, k=3)).float()
    return model, dictionary


def dense(layer, x):
    # Full copies are an independent oracle for these tiny CPU tensors only.
    pre = F.linear(x, layer.W_enc.to(x.dtype), layer.b_enc.to(x.dtype))
    if hasattr(layer, "threshold"):
        features = pre * (pre > layer.threshold.to(x.dtype))
    else:
        values, indices = pre.relu().topk(layer.k, dim=-1)
        features = torch.zeros_like(pre).scatter(-1, indices, values)
    return features, features @ layer.W_dec.to(x.dtype) + layer.b_dec.to(x.dtype)


@contextmanager
def reference_replacement(model, dictionary, active, mask):
    methods, reads = [], {}
    try:
        for index in active:
            block = model.get_submodule(dictionary.metadata["read_points"][index].rsplit(".", 1)[0])
            old = block.mlp.forward
            methods.append((block.mlp, old))
            if model.config.model_type == "gemma2":
                def remember(x, *args, old=old, index=index, **kwargs):
                    reads[index] = x
                    return old(x, *args, **kwargs)
                block.mlp.forward = remember
                norm = block.post_feedforward_layernorm
                original = norm.forward
                methods.append((norm, original))

                def replace(x, *args, original=original, index=index, **kwargs):
                    actual = original(x, *args, **kwargs)
                    decoded = dense(dictionary.layers[index], reads.pop(index))[1]
                    return torch.where(mask[..., None], decoded, actual)
                norm.forward = replace
            else:
                def replace(x, *args, old=old, index=index, **kwargs):
                    decoded = dense(dictionary.layers[index], x)[1]
                    return decoded if mask is None else torch.where(mask[..., None], decoded, old(x, *args, **kwargs))
                block.mlp.forward = replace
        yield
    finally:
        for module, method in reversed(methods):
            module.forward = method


def contexts(model, mode):
    if mode == "research":
        policy = protocol.research_policy(model, 5, tokenizer(model))
        prepared = protocol.research_contexts([[4, 5, 2, 0], [6, 7]], policy, device="cpu", pad_to=5)
        return prepared.input_ids, prepared, 6
    if model.config.model_type == "gemma2":
        prepared = prepare_contexts([[4, 5, 6], [7]], context_policy(model, 5), device="cpu", pad_to=5)
        return prepared.input_ids, prepared, 4
    return torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]]), None, 6


def batch(model, dictionary, mode, ids, prepared, count, memory, **kwargs):
    if mode == "research":
        return evaluation.evaluate_research_batch(model, dictionary, prepared, count, chunk_tokens=2,
                                                  health=lambda: None, check_binding=lambda: None,
                                                  memory=memory, **kwargs)
    return evaluation.evaluate_batch(model, dictionary, ids, count, prepared=prepared,
                                      chunk_tokens=2, memory=memory, **kwargs)


def loss(model, ids, prepared, count):
    options = {} if prepared is None else {"attention_mask": prepared.attention_mask}
    # The protocol applies Gemma softcap after the head cast to float32.
    # The model's own reduced-precision softcap has different boundary values.
    if model.config.model_type == "gemma2":
        hidden = model.model(ids, **options, use_cache=False).last_hidden_state
        head = model.get_output_embeddings()
        logits = F.linear(hidden, head.weight, head.bias).float()
        cap = model.config.final_logit_softcapping
        logits = (logits / cap).tanh() * cap
    else:
        logits = model(ids, **options, use_cache=False).logits.float()
    selected = (prepared.prediction_mask if prepared is not None else
                (torch.arange(ids.numel()).reshape_as(ids)[:, :-1] + 1 < count))
    return F.cross_entropy(logits[:, :-1][selected], ids[:, 1:][selected], reduction="sum").item(), int(selected.sum())


@pytest.mark.parametrize("family", ["gemma", "tiny"])
@pytest.mark.parametrize("mode", ["legacy", "research"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_batches_match_dense_reference_and_preserve_state(family, mode, dtype):
    model, dictionary = resources(family, dtype)
    ids, prepared, count = contexts(model, mode)
    before_hooks = hooks(model)
    weights = {name: (p.data_ptr(), p._version, p.clone()) for name, p in dictionary.named_parameters()}
    mask = None if prepared is None else prepared.content_mask
    before_masks = None if prepared is None else (mask.clone(), prepared.prediction_mask.clone())
    points = dictionary.metadata["read_points"] + dictionary.metadata["write_points"]
    with torch.no_grad():
        baseline, targets = loss(model, ids, prepared, count)
        options = {} if prepared is None else {"attention_mask": prepared.attention_mask}
        baseline_hidden = hidden_forward(model, ids, **options)
        expected_loss = []
        for active in ([0], [1], [0, 1]):
            with reference_replacement(model, dictionary, active, mask):
                value, number = loss(model, ids, prepared, count)
                assert number == targets
                expected_loss.append(value)
        with capture(model, points) as saved:
            hidden_forward(model, ids, **options)
        expected = []
        for i, layer in enumerate(dictionary.layers):
            x, y = (saved[dictionary.metadata[field][i]] for field in ("read_points", "write_points"))
            x, y = ((t.reshape(-1, t.shape[-1])[:count] if mask is None else t[mask]) for t in (x, y))
            features, decoded = dense(layer, x)
            target = y.double() if mode == "research" else y
            mean = target.mean(0)
            error = (target - (decoded.double() if mode == "research" else decoded)).square()
            expected.append((mean.double(), (target - mean).square().sum().double(),
                             sum(error[n:n + 2].sum().double() for n in range(0, len(y), 2)),
                             (features > 0).sum(0)))
        saved.close(clear=True)
        actual = batch(model, dictionary, mode, ids, prepared, count, guard())
        assert torch.equal(hidden_forward(model, ids, **options), baseline_hidden)
    tolerance = {torch.float16: .003, torch.bfloat16: .03, torch.float32: 3e-6, torch.float64: 3e-6}[dtype]
    assert actual.ce_tokens == targets
    assert actual.count == (count if mask is None else int(mask.sum()))
    assert actual.original_ce == pytest.approx(baseline, rel=3e-6)
    assert actual.replaced_ce == pytest.approx(expected_loss[2], rel=tolerance)
    if mode == "research":
        assert actual.source_tokens == count
        assert actual.isolated_ce.tolist() == pytest.approx(expected_loss[:2], rel=tolerance)
    for i, (mean, variance, error, hits) in enumerate(expected):
        torch.testing.assert_close(actual.mean[i], mean, rtol=tolerance, atol=1e-8)
        torch.testing.assert_close(actual.variance_sum[i], variance, rtol=tolerance, atol=1e-8)
        torch.testing.assert_close(actual.error_sum[i], error, rtol=tolerance, atol=1e-8)
        assert torch.equal(actual.hits[i], hits)
    if prepared is not None:
        assert torch.equal(mask, before_masks[0]) and torch.equal(prepared.prediction_mask, before_masks[1])
    assert hooks(model) == before_hooks
    for name, parameter in dictionary.named_parameters():
        pointer, version, value = weights[name]
        assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer and parameter._version == version
        assert torch.equal(parameter, value)


@pytest.mark.parametrize("mode", ["legacy", "research_v1"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_entry_point_accepts_loaded_mixed_dictionary_without_weight_changes(tmp_path, mode, dtype):
    model = TinyLM(layers=2, width=8, heads=2, latents=12, k=3, dtype=dtype)
    write_fixture(tmp_path, model)
    stored = {p.name: p.read_bytes() for p in tmp_path.glob("*.safetensors")}
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float32)
    with TextStream(documents(), tokenizer(model, [4, 5, 6, 7]), "quality") as stream:
        quality = evaluation.evaluate(model, dictionary, stream, tmp_path, protocol=mode, tokens=6,
                                      context=4, batch_tokens=8, chunk_tokens=2, memory=guard())
    assert quality["evaluated_tokens"] == 6 and quality["cross_entropy_tokens"] > 0
    assert quality["stop_reason"] == "complete"
    assert all((tmp_path / name).read_bytes() == value for name, value in stored.items())
    assert all(p.dtype == torch.float32 for p in dictionary.parameters())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("family", ["gemma", "tiny"])
@pytest.mark.parametrize("mode,pass_number", [("legacy", 1), ("legacy", 2),
                                             ("research", 1), ("research", 2), ("research", 4)])
def test_mixed_failure_releases_capture_and_replacement_hooks(monkeypatch, family, mode, pass_number):
    model, dictionary = resources(family, torch.float64)
    ids, prepared, count = contexts(model, mode)
    original = evaluation.hidden_forward
    calls = 0
    memory = guard()
    before = hooks(model)
    modules = tuple(model.modules())
    captures = []

    def observed_capture(*args, **kwargs):
        result = capture(*args, **kwargs)
        captures.append(result)
        return result

    def forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == pass_number:
            memory.limit = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "hidden_forward", forward)
    monkeypatch.setattr(evaluation, "capture", observed_capture)
    with pytest.raises(CleanStop, match="memory|workspace"):
        batch(model, dictionary, mode, ids, prepared, count, memory)
    assert calls == pass_number and hooks(model) == before
    assert tuple(model.modules()) == modules and captures and all(not saved for saved in captures)
    memory.limit = 12_000_000_000
    monkeypatch.setattr(evaluation, "hidden_forward", original)
    assert batch(model, dictionary, mode, ids, prepared, count, memory).ce_tokens > 0


@pytest.mark.parametrize("mode", ["legacy", "research_v1"])
def test_entry_memory_refusal_precedes_output_and_forward(tmp_path, monkeypatch, mode):
    model = TinyLM(layers=2, width=8, heads=2, latents=12, k=3)
    write_fixture(tmp_path, model)
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float32)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    memory = guard()
    memory.limit = 1
    monkeypatch.setattr(evaluation, "hidden_forward", lambda *a, **kw: pytest.fail("forward preceded admission"))
    with TextStream(documents(), tokenizer(model, [4, 5, 6, 7]), "quality") as stream:
        with pytest.raises(CleanStop):
            evaluation.evaluate(model, dictionary, stream, tmp_path, protocol=mode, tokens=6,
                                context=4, batch_tokens=8, chunk_tokens=2, memory=memory)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_conversion_bounds_and_decoder_refusal(monkeypatch):
    model, dictionary = resources("gemma", torch.float64)
    layer = dictionary.layers[0]
    x = torch.ones(2, 16, dtype=torch.float64)
    memory = guard()
    monkeypatch.setattr(_frozen, "_ENCODER_BLOCK_ELEMENTS", 64)
    to = torch.Tensor.to
    copies = []

    def bounded(value, *args, **kwargs):
        if value.ndim == 2 and args == (torch.float64,):
            assert value.numel() <= 64
            copies.append(value.numel())
        return to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", bounded)
    features, decoded = protocol.evaluation_features(layer, x, memory=memory)
    assert features.dtype == decoded.dtype == torch.float64 and len(copies) > 2
    original = _frozen._trace_features

    def lower(*args, **kwargs):
        value = original(*args, **kwargs)
        memory.limit = 1
        return value

    monkeypatch.setattr(_frozen, "_trace_features", lower)
    monkeypatch.setattr(_frozen, "_active_columns", lambda *a: pytest.fail("decoder allocated before refusal"))
    with pytest.raises(CleanStop, match="workspace"):
        protocol.evaluation_features(layer, x, memory=memory)
