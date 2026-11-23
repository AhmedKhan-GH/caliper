"""CPU fixture checks for real-model interventions and their lifecycle."""
from copy import deepcopy
import json
import weakref

import pytest
import torch
from torch.nn import functional as F

from caliper.interp._dictionary import Dictionary, TopKTranscoder, model_identity
from caliper.interp._intervene import InterventionError, InterventionService
from tests.interp.fixture import make_fixture


def dictionary_for(model):
    layers = []
    for block in model.layers:
        mlp = block.mlp
        layers.append(TopKTranscoder(*(getattr(mlp, key).detach().clone() for key in
                                      ("W_enc", "b_enc", "W_dec", "b_dec")), mlp.k))
    return Dictionary({"model_hash": model_identity(model), "width": model.config.hidden_size,
                       "read_points": [f"layers.{i}.mlp_in" for i in range(len(layers))],
                       "write_points": [f"layers.{i}.mlp_out" for i in range(len(layers))]}, layers)


@pytest.fixture
def case():
    model, tokenizer = make_fixture()
    service = InterventionService(model)
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, dictionary_id="fixture")
    yield model, tokenizer, service
    service.close()


def explicit_forward(model, ids, *, edit=None):
    """Compute independent feature values from weights, without service or dictionary encode."""
    x = model.embed_tokens(ids)
    activations = []
    for i, block in enumerate(model.layers):
        x = x + block.self_attn(block.input_layernorm(x))
        mlp_input = block.post_attention_layernorm(x)
        z = F.linear(mlp_input, block.mlp.W_enc, block.mlp.b_enc).clamp_min(0)
        indices = torch.argsort(z, dim=-1, descending=True)[..., :block.mlp.k]
        active = torch.zeros_like(z).scatter(-1, indices, z.gather(-1, indices))
        activations.append(active)
        if edit is not None:
            active = edit(i, active)
        x = x + active @ block.mlp.W_dec + block.mlp.b_dec
    return model.lm_head(model.norm(x)), activations


def test_clamp_to_live_value_is_identity(case):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"]
    expected, activations = explicit_forward(model, ids)
    feature = int(activations[2][0, 2].argmax())
    hook = service.install({"kind": "clamp", "point": "layers.2.mlp_out", "feature": feature,
                            "value": activations[2][0, 2, feature].item(), "positions": [2]})
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=1e-12)
    assert hook.calls == 1 and hook.errors == 0


def test_zero_clamp_matches_frozen_replacement_prediction(case):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"]
    baseline, activations = explicit_forward(model, ids)
    position, layer = 3, 3
    feature = int(activations[layer][0, position].argmax())
    # The last MLP has only the final frozen norm and unembedding downstream.
    delta = -activations[layer][0, position, feature] * model.layers[layer].mlp.W_dec[feature]
    predicted = baseline.detach().clone()
    predicted[0, position] += F.linear(delta * model.norm.last_scale[0, position] * model.norm.weight,
                                       model.lm_head.weight)
    with model.frozen_attention(norms=True):
        hook = service.install({"kind": "clamp", "point": "layers.3.mlp_out", "feature": feature,
                                "value": 0, "positions": [position]})
        torch.testing.assert_close(model(ids).logits, predicted, rtol=0, atol=1e-6)
    assert hook.errors == 0


@pytest.mark.parametrize("point", ["layers.1.resid_pre", "layers.1.mlp_in", "layers.1.mlp_out"])
def test_zero_add_is_identity_and_clear_releases_hooks(case, point):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"]
    expected = model(ids).logits
    hook = service.install({"op": "add", "point": point, "vector": [1.0] * 64, "coefficient": 0})
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    hook()
    assert hook.removed and not hook.handles and not service._cache


def test_selected_rows_use_their_own_live_features(case):
    model, tokenizer, service = case
    ids = torch.cat([tokenizer("abc")["input_ids"], tokenizer("xyz")["input_ids"]])
    baseline, activations = explicit_forward(model, ids)
    feature = int(activations[1][1, 2].argmax())
    assert activations[1][0, 2, feature] != activations[1][1, 2, feature]

    def edit(layer, active):
        result = active.clone()
        if layer == 1:
            result[1, 2, feature] *= -2
        return result

    expected, _ = explicit_forward(model, ids, edit=edit)
    service.install({"kind": "scale", "point": "layers.1.mlp_out", "feature": feature,
                     "multiplier": -2, "positions": [2]})
    with service.rows([1]):
        actual = model(ids).logits
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    with service.suspend():
        torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)


def test_nested_row_contexts_restore_after_exception(case):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"].repeat(2, 1)
    baseline = model(ids).logits
    service.install({"kind": "add", "point": "layers.1.mlp_out", "vector": [0.2] * 64, "coefficient": 1})
    with service.rows([1]):
        with pytest.raises(RuntimeError):
            with service.rows([0]):
                torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
                raise RuntimeError("context exit")
        actual = model(ids).logits
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    assert not torch.equal(actual[1], baseline[1])


def test_patch_all_residual_positions_gives_clean_downstream_logits(case):
    model, tokenizer, service = case
    clean, corrupt = tokenizer("abc")["input_ids"], tokenizer("xyz")["input_ids"]
    expected = model(clean).logits
    other = service.install({"kind": "add", "point": "layers.0.mlp_out", "vector": [0.2] * 64,
                             "coefficient": 1})
    hook = service.install({"kind": "patch", "point": "layers.2.resid_pre", "source_prompt": "abc"})
    # Clean capture must suspend the earlier add, which remains active for the corrupt pass.
    assert other.calls == 0 and hook.errors == 0
    torch.testing.assert_close(model(corrupt).logits, expected, rtol=0, atol=0)
    assert other.calls == 1
    cached = weakref.ref(service._cache["layers.2.resid_pre"])
    hook()
    assert cached() is None and not service._cache


def test_patch_source_position_and_batch_mask(case):
    model, tokenizer, service = case
    ids = tokenizer("xyz")["input_ids"].repeat(2, 1)
    clean = tokenizer("abc")["input_ids"]
    baseline = model(ids).logits
    embeddings = model.embed_tokens(ids).detach().clone()
    embeddings[1, 2] = model.embed_tokens(clean)[0, 1]
    expected = model(inputs_embeds=embeddings).logits
    service.install({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc",
                     "source_position": 1, "positions": [2]})
    with service.rows([1]):
        actual = model(ids).logits
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_patch_cache_invalidates_on_weight_or_prompt_change(case):
    model, tokenizer, service = case
    spec = {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"}
    service.install(spec)
    first = service._cache[spec["point"]]
    assert service._source(spec) is first
    with torch.no_grad():
        model.embed_tokens.weight.add_(0.01)
    second = service._source(spec)
    assert second is not first
    torch.testing.assert_close(second, first + 0.01)
    third = service._source(dict(spec, source_prompt="xyz"))
    assert third is not second and len(service._cache) == 1


def test_hook_failure_removes_it_immediately_and_reports_without_throw(case):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"]
    expected = model(ids).logits
    errors = []
    hook = service.install({"kind": "add", "point": "layers.0.mlp_out", "vector": [1.0] * 64,
                            "coefficient": 1, "positions": [31]}, on_error=errors.append)
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    assert hook.errors == 1 and hook.removed and errors
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    assert hook.calls == 1


@pytest.mark.parametrize("change", [
    {"vector": [1.0]}, {"coefficient": float("nan")}, {"coefficient": True},
    {"vector": [float("inf")] * 64}, {"positions": [True]}, {"positions": [-1]},
    {"positions": [1, 1]}, {"positions": [32]}, {"point": "missing"},
    {"point": "layers.0.attn_pattern"}, {"op": "clamp"}, {"unused": 1},
])
def test_invalid_add_keeps_prior_hook(case, change):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"]
    spec = {"kind": "add", "point": "layers.0.mlp_out", "vector": [0.2] * 64, "coefficient": 1}
    hook = service.install(spec)
    prior = model(ids).logits
    with pytest.raises(InterventionError):
        service.install(dict(spec, **change))
    assert not hook.removed
    torch.testing.assert_close(model(ids).logits, prior, rtol=0, atol=0)


@pytest.mark.parametrize("change", [{"dictionary": "unknown"}, {"feature": -1},
                                    {"feature": 256}, {"feature": True}, {"value": float("inf")}])
def test_invalid_feature_refuses_before_hook_install(case, change):
    model, _, service = case
    before = sum(len(m._forward_hooks) for m in model.modules())
    with pytest.raises(InterventionError):
        service.install(dict({"kind": "clamp", "point": "layers.1.mlp_out", "feature": 0,
                              "value": 0}, **change))
    assert sum(len(m._forward_hooks) for m in model.modules()) == before


@pytest.mark.parametrize("point", ["layers.0.mlp", "layers.0.self_attn#0"])
def test_legacy_ablation_obeys_rows_and_suspension(case, point):
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"].repeat(2, 1)
    expected = model(ids).logits
    hook = service.install({"op": "ablate", "point": point})
    with service.rows([1]):
        actual = model(ids).logits
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert not torch.equal(actual[1], expected[1])
    with service.suspend():
        torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    hook()
