"""Offline Gemma runtime checks with small random CPU models."""
from types import SimpleNamespace

import pytest
import torch

from caliper.interp import _admission, _dictionary, _hosted, _index
from caliper.interp._admission import GEMMA_ID, GEMMA_REVISION
from caliper.interp._memory import CleanStop, MemoryGuard
from caliper.interp._output import forward_logits
from tools.interp import model_load


def gemma_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma2Config(
        vocab_size=37, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, sliding_window=4, attn_logit_softcapping=3.0,
        final_logit_softcapping=2.0, query_pre_attn_scalar=8,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = transformers.Gemma2ForCausalLM(config).eval().requires_grad_(False)
    model.config._name_or_path, model.config._commit_hash = GEMMA_ID, GEMMA_REVISION
    return model


def memory(limit=12, **kwargs):
    return MemoryGuard(limit, "cpu", resident=lambda: 0, footprint=lambda: 0,
                       peak=lambda: 0, driver=lambda: 0, pressure=lambda: 80, **kwargs)


def local_gemma(tmp_path, model):
    snapshot = tmp_path / "models--google--gemma-2-2b" / "snapshots" / GEMMA_REVISION
    model.save_pretrained(snapshot, safe_serialization=True)
    return snapshot


def test_gemma_memory_ceiling_and_staged_admission():
    guard = memory(40)
    guard.peak = 30_000_000_000
    _admission._admit_load(guard, 12_000_000_000, "transfer", staged=True)
    with pytest.raises(CleanStop, match="model allocation"):
        _admission._admit_load(guard, 12_000_000_000, "legacy")
    with pytest.raises(ValueError, match="40 GB"):
        memory(40.01)
    over = MemoryGuard(40, "cpu", resident=lambda: 40_000_000_001, footprint=lambda: None,
                       peak=lambda: 0, driver=lambda: 0, pressure=lambda: 80)
    with pytest.raises(CleanStop, match="memory limit"):
        over.check("over budget")


def test_gemma_checkpoint_verifies_tied_weights_and_nonpersistent_buffers(tmp_path):
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert "model.embed_tokens.embed_scale" not in model.state_dict()
    identity = _dictionary.verify_hf_model(model, snapshot)
    assert identity == f"hf:{GEMMA_ID}@{GEMMA_REVISION}"
    model.model.embed_tokens.embed_scale.data.add_(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        _dictionary.model_identity(model)
    with pytest.raises(ValueError, match="nonpersistent buffers"):
        _dictionary.verify_hf_model(model, snapshot)


@pytest.mark.parametrize("change", ["softcap", "sliding", "scale", "tied"])
def test_gemma_checkpoint_refuses_effective_setting_or_tied_weight_changes(tmp_path, change):
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    _dictionary.verify_hf_model(model, snapshot)
    if change == "softcap":
        model.config.final_logit_softcapping *= 2
    elif change == "sliding":
        model.model.layers[0].self_attn.sliding_window = 7
    elif change == "scale":
        model.model.embed_tokens.scalar_embed_scale *= 2
    else:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone() + 1)
    with pytest.raises(ValueError, match="identity mismatch"):
        _dictionary.model_identity(model)
    with pytest.raises(ValueError, match="settings|weights|tensor|checkpoint"):
        _dictionary.verify_hf_model(model, snapshot)


def test_gemma_output_uses_real_softcapped_forward_and_last_token():
    model = gemma_model()
    ids = torch.tensor([[1, 5, 9, 7, 6], [1, 5, 9, 7, 6]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        actual = forward_logits(model, {"input_ids": ids, "attention_mask": mask})
    assert actual.shape == (2, 1, 37)
    torch.testing.assert_close(actual[:, -1], expected[:, -1])
    assert actual.abs().max() <= model.config.final_logit_softcapping


def test_gemma_hidden_forward_preserves_masks_without_head_projection(monkeypatch):
    model = gemma_model()
    ids = torch.tensor([[1, 5, 9, 0], [1, 7, 6, 2]])
    mask = (ids != 0).long()
    with torch.no_grad():
        expected = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    monkeypatch.setattr(model.lm_head, "forward", lambda *a: pytest.fail("vocabulary projection ran"))
    calls = []
    handle = model.model.layers[0].mlp.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        with torch.no_grad():
            actual = model_load.hidden_forward(model, ids, mask)
        torch.testing.assert_close(actual, expected)
        assert calls == [1]
    finally:
        handle.remove()


def mock_gemma_load(tmp_path, monkeypatch, *, info=None):
    import huggingface_hub
    model = gemma_model()
    snapshot = local_gemma(tmp_path, model)
    # Fixture imports can replace the lazy module. Patch its current instance.
    import transformers
    calls = []

    def cached(repo, *, revision, local_files_only):
        assert (repo, revision, local_files_only) == (GEMMA_ID, GEMMA_REVISION, True)
        calls.append("snapshot")
        return str(snapshot)

    def loaded(folder, **kwargs):
        assert folder == snapshot
        assert kwargs["local_files_only"] and kwargs["use_safetensors"]
        assert not kwargs["trust_remote_code"] and kwargs["attn_implementation"] == "eager"
        calls.append("weights")
        return model, info or {}

    def tokenizer(folder, **kwargs):
        assert folder == snapshot and kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        calls.append("tokenizer")
        return object()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    monkeypatch.setattr(transformers, "Gemma2Config", lambda **kw: SimpleNamespace(
        model_type="gemma2", num_hidden_layers=26, hidden_size=2304, vocab_size=256000))
    monkeypatch.setattr(transformers.Gemma2ForCausalLM, "from_pretrained", loaded)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    return model, snapshot, calls


def test_explicit_gemma_loader_uses_only_pinned_local_snapshot(tmp_path, monkeypatch):
    model, _, calls = mock_gemma_load(tmp_path, monkeypatch)
    guard = memory()
    loaded, _ = model_load.load_text_model(model_id=GEMMA_ID, device="cpu", memory=guard)
    assert loaded is model and calls == ["snapshot", "weights", "tokenizer"]
    assert _dictionary.model_identity(model) == f"hf:{GEMMA_ID}@{GEMMA_REVISION}"
    assert guard in _dictionary._VERIFIED_MODELS[model].memory.guards


def test_gemma_incomplete_load_never_records_identity(tmp_path, monkeypatch):
    model, _, calls = mock_gemma_load(tmp_path, monkeypatch, info={"missing_keys": ["model.norm.weight"]})
    with pytest.raises(ValueError, match="weights do not match"):
        model_load.load_text_model(model_id=GEMMA_ID, device="cpu", memory=memory())
    assert model not in _dictionary._VERIFIED_MODELS
    assert calls == ["snapshot", "weights"]


def test_gemma_hosted_load_passes_budget_to_dictionary_and_index(tmp_path, monkeypatch):
    model, snapshot, calls = mock_gemma_load(tmp_path, monkeypatch)
    guard = memory(40)
    from caliper.interp import _memory
    monkeypatch.setattr(_memory, "MemoryGuard", lambda *a: guard)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    verify = _dictionary.verify_hf_model

    def verify_cpu(model, folder, *, memory_limit_gb):
        assert memory_limit_gb == 40
        return verify(model, folder, memory_limit_gb=40)

    def dictionary(*a, **kwargs):
        assert kwargs["memory_limit_gb"] == 40 and kwargs["model"] is model and kwargs["memory"] is guard
        calls.append("dictionary")
        return object()

    def index(*a, **kwargs):
        assert kwargs["memory"] is guard and kwargs["model"] is model
        calls.append("index")
        return object()

    monkeypatch.setattr(_dictionary, "verify_hf_model", verify_cpu)
    monkeypatch.setattr(_dictionary, "load_dictionary", dictionary)
    monkeypatch.setattr(_index, "FeatureIndex", index)
    loaded, _, _, _, actual_guard = _hosted.load_resources(str(snapshot), tmp_path)
    assert loaded is model and actual_guard is guard
    assert calls == ["weights", "tokenizer", "dictionary", "index"]


@pytest.fixture
def gemma_semantics(monkeypatch):
    """Use the real family map after integration, or the agreed probe contract."""
    from caliper.interp import _intervene, _points
    model = gemma_model()
    try:
        _points._architecture(model)
    except NotImplementedError:
        def architecture(model):
            result = {}
            for i in range(model.config.num_hidden_layers):
                base = f"model.layers.{i}"
                for suffix, module, event in (("mlp_in", base + ".mlp", "input"),
                                               ("mlp_out", base + ".post_feedforward_layernorm", "output")):
                    point = base + "." + suffix
                    result[point] = _points.Probe(point, module, event, i, "gemma2")
            return result
        monkeypatch.setattr(_points, "_architecture", architecture)
        monkeypatch.setattr(_intervene, "_architecture", architecture)
    return model


class SyntheticJumpReLU(_dictionary.TopKTranscoder):
    """Test the runtime contract without the dictionary worker's branch."""
    activation = "jumprelu"

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold):
        super().__init__(W_enc, b_enc, W_dec, b_dec, k=1)
        self.k = None
        self.register_parameter("threshold", torch.nn.Parameter(threshold, requires_grad=False))

    def activate(self, value):
        return value * (value > self.threshold)

    def encode(self, x):
        return self.activate(torch.nn.functional.linear(x, self.W_enc, self.b_enc))


def runtime_dictionary(model):
    layers = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        for _ in model.model.layers:
            constructor = getattr(_dictionary, "JumpReLUTranscoder", SyntheticJumpReLU)
            layers.append(constructor(torch.randn(20, 16) * .1, torch.ones(20) * .1,
                                       torch.randn(20, 16) * .1, torch.randn(16) * .03, torch.full((20,), .05)))
    count = len(layers)
    return _dictionary.Dictionary({
        "model_id": GEMMA_ID, "revision": GEMMA_REVISION, "model_hash": _dictionary.model_identity(model),
        "layers": count, "width": 16, "latents": 20, "activation": "jumprelu", "k": None,
        "read_points": [f"model.layers.{i}.mlp_in" for i in range(count)],
        "write_points": [f"model.layers.{i}.mlp_out" for i in range(count)], "training": {},
    }, layers)


def manual_gemma(model, ids, dictionary, *, replace=False, feature=None, input_add=None):
    """Compute residual writes directly without semantic hooks or module swaps."""
    from contextlib import ExitStack
    from unittest.mock import patch
    from torch.nn import functional as F

    def forward(block, layer, hidden_states, *, position_embeddings=None, attention_mask=None, cache=None, **kwargs):
        attention, _ = block.self_attn(block.input_layernorm(hidden_states),
                                      position_embeddings=position_embeddings, attention_mask=attention_mask,
                                      cache=cache, **kwargs)
        residual = hidden_states + block.post_attention_layernorm(attention)
        read = block.pre_feedforward_layernorm(residual)
        if input_add is not None and input_add[0] == layer:
            read[input_add[1], input_add[2]] += read.new_tensor(input_add[3])
        weights = dictionary.layers[layer]
        pre = F.linear(read, weights.W_enc, weights.b_enc)
        activated = pre * (pre > weights.threshold)
        if replace:
            write = activated @ weights.W_dec + weights.b_dec
        else:
            write = block.post_feedforward_layernorm(block.mlp(read))
        if feature is not None and feature[0] == layer:
            _, row, position, index, kind, value = feature
            amount = (value - activated[row, position, index] if kind == "clamp"
                      else (value - 1) * activated[row, position, index])
            write[row, position] += amount * weights.W_dec[index]
        return residual + write

    with ExitStack() as stack:
        for i, block in enumerate(model.model.layers):
            def replacement(hidden_states, block=block, i=i, **kwargs):
                return forward(block, i, hidden_states, **kwargs)
            stack.enter_context(patch.object(block, "forward", replacement))
        return model(input_ids=ids, use_cache=False).logits


@pytest.mark.parametrize("kind,value", [("clamp", 0.), ("scale", 2.)])
def test_gemma_feature_hook_reads_pre_mlp_and_writes_after_norm(gemma_semantics, kind, value):
    from caliper.interp._intervene import InterventionService
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    ids = torch.tensor([[1, 5, 8, 9], [1, 7, 6, 2]])
    with torch.no_grad():
        expected = manual_gemma(model, ids, dictionary, feature=(0, 1, 2, 3, kind, value))
        baseline = model(ids, use_cache=False).logits
    service = InterventionService(model)
    service.configure(dictionary=dictionary)
    try:
        hook = service.install({"op": kind, "point": "model.layers.0.mlp_out", "feature": 3,
                                "value" if kind == "clamp" else "multiplier": value, "positions": [2]})
        with service.rows([1]), torch.no_grad():
            actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], baseline[0])
        assert hook.live_input is None and hook.errors == 0 and len(hook.handles) == 4
    finally:
        service.close()
    assert hook.removed and hook.live_input is None
    assert not model.model.layers[0].mlp._forward_pre_hooks
    assert not model.model.layers[0].mlp._forward_hooks
    assert not model.model.layers[0].post_feedforward_layernorm._forward_hooks


@pytest.mark.parametrize("decoder_only", [False, True])
def test_gemma_feature_hook_clears_live_read_after_forward_failure(gemma_semantics, monkeypatch, decoder_only):
    from caliper.interp._intervene import InterventionService
    model = gemma_semantics
    service = InterventionService(model)
    service.configure(dictionary=runtime_dictionary(model))
    hook = service.install({"op": "clamp", "point": "model.layers.0.mlp_out", "feature": 0, "value": 0})
    def failed(*args):
        raise RuntimeError("synthetic failure after semantic read")
    monkeypatch.setattr(model.model.layers[0].post_feedforward_layernorm, "forward", failed)
    try:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            (model.model if decoder_only else model)(torch.tensor([[1, 4]]), use_cache=False)
        assert hook.live_input is None and hook.read_stack == []
    finally:
        service.close()


def test_gemma_threshold_alias_mutation_invalidates_intervention_identity(gemma_semantics):
    from caliper.interp._intervene import InterventionService, InterventionError
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    service = InterventionService(model)
    service.configure(dictionary=dictionary)
    service._prepare_dictionary_identity()
    assert service._dictionary_snapshot is not None
    dictionary.layers[0].threshold.data[0] += .2
    try:
        with pytest.raises(InterventionError, match="content changed"):
            service.install({"op": "clamp", "point": "model.layers.0.mlp_out", "feature": 0, "value": 0})
    finally:
        service.close()


def test_gemma_index_accepts_no_fixed_k_and_never_projects_logits(gemma_semantics, monkeypatch):
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    assert _index._compatible(model, dictionary)["k"] is None
    ids = torch.tensor([[1, 4, 8, 0], [1, 3, 5, 6]])
    mask = (ids != 0).long()
    with torch.no_grad():
        expected = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    monkeypatch.setattr(model.lm_head, "forward", lambda *a: pytest.fail("vocabulary projection ran"))
    from caliper.interp._points import capture
    with torch.no_grad(), capture(model, dictionary.metadata["read_points"]) as values:
        result = _index._forward(model, ids, mask)
        torch.testing.assert_close(result.last_hidden_state, expected)
        assert all(values[point].shape == (2, 4, 16) for point in dictionary.metadata["read_points"])
    original = _index._dictionary_hash(dictionary)
    dictionary.layers[0].threshold.data[0] += .1
    assert _index._dictionary_hash(dictionary) != original


def test_gemma_clamp_observes_input_edit_installed_after_feature_hook(gemma_semantics):
    from caliper.interp._intervene import InterventionService
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    ids = torch.tensor([[1, 5, 8, 9], [1, 7, 6, 2]])
    vector = [.3] * 16
    with torch.no_grad():
        expected = manual_gemma(model, ids, dictionary, feature=(0, 1, 2, 3, "clamp", 0),
                                input_add=(0, 1, 2, vector))
    service = InterventionService(model)
    service.configure(dictionary=dictionary)
    try:
        feature = service.install({"op": "clamp", "point": "model.layers.0.mlp_out", "feature": 3,
                                   "value": 0, "positions": [2]})
        service.install({"op": "add", "point": "model.layers.0.mlp_in", "vector": vector,
                         "coefficient": 1, "positions": [2]})
        with service.rows([1]), torch.no_grad():
            actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)
        assert feature.live_input is None and feature.read_stack == [] and feature.errors == 0
    finally:
        service.close()


def test_gemma_feature_read_survives_nested_suspended_forward(gemma_semantics):
    from caliper.interp._intervene import InterventionService
    model = gemma_semantics
    dictionary = runtime_dictionary(model)
    ids = torch.tensor([[1, 5, 8, 9], [1, 7, 6, 2]])
    with torch.no_grad():
        expected = manual_gemma(model, ids, dictionary, feature=(0, 1, 2, 3, "clamp", 0))
    service = InterventionService(model)
    service.configure(dictionary=dictionary)
    active = False
    nested = []
    def source(module, args, output):
        nonlocal active
        if not active:
            active = True
            try:
                with service.suspend():
                    nested.append(model(torch.tensor([[1, 2]]), use_cache=False).logits)
            finally:
                active = False
    try:
        feature = service.install({"op": "clamp", "point": "model.layers.0.mlp_out", "feature": 3,
                                   "value": 0, "positions": [2]})
        handle = model.model.layers[0].mlp.register_forward_hook(source)
        try:
            with service.rows([1]), torch.no_grad():
                actual = model(ids, use_cache=False).logits
        finally:
            handle.remove()
        torch.testing.assert_close(actual, expected)
        assert len(nested) == 1 and feature.errors == 0
        assert feature.live_input is None and feature.read_stack == []
    finally:
        service.close()


def test_gemma_small_index_build_and_examples_use_decoder_only(tmp_path, gemma_semantics, monkeypatch):
    import json
    from safetensors.torch import save_file
    from caliper.interp._points import capture
    model = gemma_semantics
    _dictionary.verify_hf_model(model, local_gemma(tmp_path, model), memory_limit_gb=40)
    dictionary = runtime_dictionary(model)
    (tmp_path / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    for i, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), tmp_path / f"layer_{i:02d}.safetensors")
    tokenizer = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                convert_ids_to_tokens=lambda i: f"token-{i}",
                                decode=lambda ids, **kw: " ".join(f"token-{i}" for i in ids))
    documents = [("one", [1, 4, 6, 8, 3]), ("two", [1, 3, 2])]
    expected = []
    for _, tokens in documents:
        with torch.no_grad(), capture(model, dictionary.metadata["read_points"]) as values:
            ids = torch.tensor([[model.config.bos_token_id, *tokens]])
            _index._forward(model, ids, torch.ones_like(ids))
            expected.append(dictionary.layers[0].encode(values[dictionary.metadata["read_points"][0]])[0, 1:])
    all_features = torch.cat(expected)
    feature = int(all_features.sum(0).argmax())
    monkeypatch.setattr(model.lm_head, "forward", lambda *a: pytest.fail("vocabulary projection ran"))
    manifest = _index.build_index(tmp_path, documents, model=model, tokenizer=tokenizer,
                                  dictionary=dictionary, provenance={"partition": "index",
                                  "publisher_training_overlap": "unknown"}, max_tokens=8,
                                  context_length=8, memory=memory(40))
    assert manifest["tokens"] == 8
    with _index.FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary,
                             memory=memory(40)) as index:
        actual = index.examples(0, feature)
        expected_values = sorted((float(v) for v in all_features[:, feature] if v > 0), reverse=True)
        assert [row["activation"] for row in actual] == pytest.approx(expected_values)
        assert all(len(row["token_ids"]) == 33 for row in actual)


def test_gemma_transfer_cannot_rebaseline_changed_weights(tmp_path, monkeypatch):
    model, _, _ = mock_gemma_load(tmp_path, monkeypatch)
    def transfer(*args, **kwargs):
        model.model.norm.weight.data.add_(1)
        return model
    monkeypatch.setattr(model, "to", transfer)
    with pytest.raises(ValueError, match="identity mismatch"):
        model_load.load_text_model(model_id=GEMMA_ID, device="cpu", memory=memory())


def test_hosted_gemma_tracer_receives_explicit_budget(monkeypatch):
    import caliper.interp as interp
    owner = SimpleNamespace(model=gemma_model(), tokenizer=object(), dictionary=object(), index=None,
                            guard=memory(40), sweep=lambda **kw: None, validate=lambda **kw: None)
    service = _hosted.HostedService(owner)
    calls = []
    sentinel = object()
    monkeypatch.setattr(interp, "Tracer", SimpleNamespace(
        _for_service=lambda *a, **kw: calls.append((a, kw)) or sentinel))
    assert service._gemma_tracer() is sentinel
    assert calls[0][1]["memory_limit_gb"] == 40
    assert calls[0][1]["memory"] is owner.guard
