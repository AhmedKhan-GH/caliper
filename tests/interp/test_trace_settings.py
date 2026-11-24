"""Trace admission rejects changes to forward settings before cache reuse."""
import pytest
import torch

from caliper.interp._dictionary import model_identity, weights_hash
from caliper.interp._graph import Graph
from caliper.interp._trace import Tracer, _versions
from .fixture import TinyLM, TinyTokenizer
from .test_trace import exact_dictionary, qwen_for_trace, random_dictionary


@pytest.fixture
def tracer(monkeypatch):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = TinyLM(layers=2, width=8, heads=2, latents=8, k=2)
    return Tracer(model, TinyTokenizer(), exact_dictionary(model))


@pytest.mark.parametrize("new_limits", [False, True])
def test_eps_mutation_cannot_reuse_graph_or_forward(tracer, new_limits):
    first = tracer.trace("AB", max_targets=4, max_nodes=32, max_edges=64)
    assert tracer.trace("AB", 4, 32, 64) is first
    baseline_logits = tracer._forward_cache[1].logits.clone()
    model_hash = model_identity(tracer.model)
    dictionary_hash = weights_hash(tracer.dictionary)
    versions = _versions(tracer.model)
    tracer.model.norm.eps = 1.0
    with torch.no_grad():
        changed_logits = tracer.model(tracer.tokenizer("AB")["input_ids"]).logits[:, -1]
    assert not torch.equal(changed_logits, baseline_logits)
    assert model_identity(tracer.model) == model_hash
    assert weights_hash(tracer.dictionary) == dictionary_hash
    assert _versions(tracer.model) == versions
    with pytest.raises(ValueError, match=r"forward setting changed.*norm.eps"):
        tracer.trace("AB", 2 if new_limits else 4, 32, 64)
    assert not tracer._cache and tracer._forward_cache is None


@pytest.mark.parametrize("module", ["layers.0.input_layernorm", "layers.1.post_attention_layernorm"])
def test_block_eps_mutation_is_refused(tracer, module):
    tracer.trace("AB", 0)
    tracer.model.get_submodule(module).eps = 1.0
    with pytest.raises(ValueError, match="forward setting changed"):
        tracer.trace("AB", 0)


def test_eps_mutation_before_first_trace_is_refused(tracer):
    tracer.model.norm.eps = 1.0
    with pytest.raises(ValueError, match="forward setting changed"):
        tracer.trace("AB", 0)


def test_restored_settings_preserve_graph_and_fingerprints(tracer):
    baseline = tracer.trace("AB", 4, 32, 64)
    model_hash = model_identity(tracer.model)
    dictionary_hash = weights_hash(tracer.dictionary)
    metadata = dict(tracer.dictionary.metadata)
    eps = tracer.model.norm.eps
    tracer.model.norm.eps = 1.0
    with pytest.raises(ValueError, match="forward setting changed"):
        tracer.trace("AB", 4, 32, 64)
    tracer.model.norm.eps = eps
    restored = tracer.trace("AB", 4, 32, 64)
    assert restored is not baseline
    assert restored.trace_id == baseline.trace_id
    assert restored.nodes == baseline.nodes and restored.edges == baseline.edges
    assert restored.logits == baseline.logits and restored.completeness == baseline.completeness
    assert model_identity(tracer.model) == model_hash
    assert weights_hash(tracer.dictionary) == dictionary_hash
    assert tracer.dictionary.metadata == metadata
    assert tracer.trace("AB", 4, 32, 64) is restored


@pytest.mark.parametrize(("module", "field", "value"), [
    ("layers.0.self_attn", "num_heads", 1),
    ("layers.0.self_attn", "head_dim", 8),
    ("layers.0.self_attn.o_proj", "in_features", 16),
    ("layers.0.mlp", "k", 1),
    ("embed_tokens", "max_norm", 1.0),
    ("embed_tokens", "padding_idx", 512),
    ("norm", "frozen_scale", torch.ones(1, 3, 1)),
    ("layers.0.self_attn", "frozen_pattern", torch.eye(3)[None, None]),
])
def test_fixture_forward_settings_are_refused(tracer, module, field, value):
    tracer.trace("AB", 0)
    setattr(tracer.model.get_submodule(module), field, value)
    with pytest.raises(ValueError, match="forward setting changed"):
        tracer.trace("AB", 0)


@pytest.mark.parametrize("field", ["frozen_scale", "frozen_pattern"])
def test_fixture_override_alias_write_cannot_reuse_graph(tracer, field):
    with torch.no_grad():
        tracer.model(tracer.tokenizer("AB")["input_ids"])
    module = tracer.model.norm if field == "frozen_scale" else tracer.model.layers[0].self_attn
    value = getattr(module, "last_scale" if field == "frozen_scale" else "last_pattern").clone()
    setattr(module, field, value)
    # Admit the existing override, then change bytes without a version increment.
    tracer = Tracer(tracer.model, tracer.tokenizer, tracer.dictionary)
    baseline = tracer.trace("AB", 0)
    assert tracer.trace("AB", 0) is baseline
    version = value._version
    value.data.mul_(.5)
    assert value._version == version
    with pytest.raises(ValueError, match="forward setting changed"):
        tracer.trace("AB", 0)


@pytest.mark.parametrize("stage", ["forward", "targets"])
def test_mid_trace_setting_mutation_is_not_published(tracer, monkeypatch, stage):
    from caliper.interp import _trace

    def change(*args):
        tracer.model.norm.eps = 1.0

    if stage == "forward":
        # Change a live attention setting after its forward.
        handle = tracer.model.norm.register_forward_pre_hook(
            lambda *args: setattr(tracer.model.layers[0].self_attn, "num_heads", 1))
    else:
        handle = None
        original = _trace._DirectEffects.backward

        def backward(self, indices):
            result = original(self, indices)
            change()
            return result

        monkeypatch.setattr(_trace._DirectEffects, "backward", backward)
    monkeypatch.setattr(Graph, "publish", lambda graph: pytest.fail("changed settings reached publication"))
    try:
        with pytest.raises(ValueError, match="forward setting changed"):
            tracer.trace("AB", 0)
    finally:
        if handle is not None:
            handle.remove()
    assert not tracer._cache and tracer._forward_cache is None


@pytest.mark.parametrize(("module", "field", "value"), [
    ("model.norm", "eps", 1.0),
    ("model.layers.0.input_layernorm", "eps", 1.0),
    ("model.layers.3.post_attention_layernorm", "eps", 1.0),
    ("model.layers.3.self_attn.q_norm", "eps", 1.0),
    ("model.layers.3.self_attn.k_norm", "eps", 1.0),
    ("model.layers.0.linear_attn.norm", "variance_epsilon", 1.0),
    ("model.layers.0.linear_attn.norm", "activation", "relu"),
    ("model.layers.3.self_attn", "scaling", 1.0),
    ("model.layers.3.self_attn", "training", True),
    ("model.layers.0.linear_attn", "activation", "relu"),
    ("model.layers.0.mlp", "act_fn", torch.nn.Tanh()),
    ("model.rotary_emb", "attention_scaling", 2.0),
])
def test_small_qwen_forward_settings_are_refused(monkeypatch, module, field, value):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = qwen_for_trace()
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids):
            return str(ids[0])

    tokenizer = Tokenizer()
    tracer = Tracer(model, tokenizer, random_dictionary(model, latents=4, k=2))
    first = tracer.trace("fixture", 0)
    assert tracer.trace("fixture", 0) is first
    identity = model_identity(model)
    setattr(model.get_submodule(module), field, value)
    assert model_identity(model) != identity
    with pytest.raises(ValueError, match="dictionary model identity mismatch"):
        tracer.trace("fixture", 0)
    assert not tracer._cache and tracer._forward_cache is None


def test_qwen_mutable_config_cannot_change_under_cached_graph(monkeypatch):
    monkeypatch.setattr(Graph, "publish", lambda graph: graph)
    model = qwen_for_trace()
    # Use byte token IDs that fit the small Qwen vocabulary.
    tokenizer = TinyTokenizer()
    tokenizer.bos_token_id = 0
    tracer = Tracer(model, tokenizer, random_dictionary(model, latents=4, k=2))
    baseline = tracer.trace("\x01\x02", 0)
    assert tracer.trace("\x01\x02", 0) is baseline
    # A list mutation must not also mutate the admission snapshot.
    model.config.layer_types[0] = "full_attention"
    with pytest.raises(ValueError, match="dictionary model identity mismatch"):
        tracer.trace("\x01\x02", 0)
    assert not tracer._cache and tracer._forward_cache is None
