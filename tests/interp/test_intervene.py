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
    cached = weakref.ref(service._cache["abc", "layers.2.resid_pre"])
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
    first = service._cache[spec["source_prompt"], spec["point"]]
    assert service._source(spec) is first
    with torch.no_grad():
        model.embed_tokens.weight.add_(0.01)
    second = service._source(spec)
    assert second is not first
    torch.testing.assert_close(second, first + 0.01)
    third = service._source(dict(spec, source_prompt="xyz"))
    assert third is not second and len(service._cache) == 2


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


@pytest.fixture
def controlled():
    from caliper._controls import Controls
    model, tokenizer = make_fixture()
    ctl = Controls()
    service = ctl.interventions_on(model)
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, dictionary_id="fixture")
    yield model, tokenizer, ctl, service
    service.remove()


def add_spec(**fields):
    return dict({"kind": "add", "point": "layers.0.mlp_out", "vector": [0.1] * 64,
                 "coefficient": 1.0}, **fields)


def test_scheduled_activation_and_clear_follow_old_step_semantics(controlled):
    model, tokenizer, ctl, service = controlled
    ids = tokenizer("abc")["input_ids"]
    baseline = model(ids).logits
    ctl.step()
    service.set(add_spec(from_step=4, until_step=6))
    assert not service.specifications()
    ctl.step()  # Latest checkpoint 2, next forward belongs to step 3.
    assert ctl.summary()["interventions"][0]["state"] == "pending"
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    ctl.step()
    assert ctl.summary()["interventions"][0]["since_step"] == 4
    changed = model(ids).logits
    assert not torch.equal(changed, baseline)
    service.clear("layers.0.mlp_out")
    torch.testing.assert_close(model(ids).logits, changed, rtol=0, atol=0)
    assert service.specifications()  # Clear takes effect only at the boundary.
    ctl.step()
    assert not ctl.summary()["interventions"] and not service.specifications()
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)


def test_replacement_waits_for_boundary_and_releases_old_handle(controlled):
    model, tokenizer, ctl, service = controlled
    ids = tokenizer("abc")["input_ids"]
    service.set(add_spec())
    ctl.step()
    first = next(iter(ctl._ivs.values())).hook
    prior = model(ids).logits
    service.set(add_spec(coefficient=0))
    assert service.specifications()[0]["coefficient"] == 1
    assert service.specifications(active_only=False)[0]["coefficient"] == 0
    torch.testing.assert_close(model(ids).logits, prior, rtol=0, atol=0)
    ctl.step()
    assert first.removed and not first.handles
    assert service.specifications()[0]["coefficient"] == 0
    with service.suspend():
        expected = model(ids).logits
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [
    add_spec(coefficient=float("nan")), add_spec(vector=[1.0]), add_spec(positions=[-1]),
    add_spec(op="clamp"), add_spec(kind="unknown"), add_spec(point="missing"),
    {"kind": "clamp", "point": "layers.0.mlp_out", "feature": 0, "value": 0, "dictionary": "missing"},
    {"kind": "scale", "point": "layers.0.mlp_out", "feature": 256, "multiplier": 1},
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc", "source_position": 9},
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "x" * 32},
])
def test_invalid_request_preserves_complete_prior_state(controlled, bad):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    ids = tokenizer("abc")["input_ids"]
    prior = model(ids).logits
    state = service.state_dict()
    hook = next(iter(ctl._ivs.values())).hook
    with pytest.raises(InterventionError):
        service.set(bad)
    assert service.state_dict() == state and not hook.removed
    torch.testing.assert_close(model(ids).logits, prior, rtol=0, atol=0)


@pytest.mark.parametrize("payload", [None, [], {"set": []}, {"set": {}},
                                     {"clear": []}, {"set": add_spec(), "clear_all": True}])
def test_malformed_controls_refuse_without_state_change(controlled, payload):
    _, _, ctl, service = controlled
    state = service.state_dict()
    with ctl.lock:
        assert ctl._intervene(payload) is not None
    assert service.state_dict() == state


def test_failed_hook_ends_before_second_forward_and_reports_structured_error(controlled):
    model, tokenizer, ctl, service = controlled
    ids = tokenizer("abc")["input_ids"]
    baseline = model(ids).logits
    service.set(add_spec(positions=[31]))
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    entry = ctl.controls_meta()["interventions"]["active"][0]
    assert entry["state"] == "ended" and entry["error_code"] == "intervene.hook_failed"
    assert entry["error"] and hook.removed and not service.specifications()
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    assert hook.calls == 1
    with pytest.warns(RuntimeWarning, match="failed and ended"):
        ctl.step()
    assert ctl.summary()["interventions"][0]["state"] == "ended"


def test_fork_state_is_plain_and_preserves_schedule_and_ended_records(controlled):
    from caliper._controls import Controls
    model, tokenizer, ctl, service = controlled
    service.set(add_spec(until_step=3))
    service.set({"kind": "clamp", "point": "layers.1.mlp_out", "feature": 7, "value": 1.2,
                 "positions": [1], "from_step": 6, "until_step": 8})
    service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"})
    ctl.step()
    ctl.step()
    records = service.state_dict()
    assert [r["state"] for r in records["interventions"]] == ["ended", "pending", "active"]
    encoded = json.dumps(records, allow_nan=False)
    other, other_tokenizer = make_fixture()
    fork = Controls()
    fork.n = ctl.n
    restored = fork.interventions_on(other)
    restored.configure(dictionary=dictionary_for(other), tokenizer=other_tokenizer, dictionary_id="fixture")
    try:
        restored.load_state_dict(json.loads(encoded))
        assert restored.state_dict() == records
        ctl.step()
        fork.step()
        ids = tokenizer("xyz")["input_ids"]
        torch.testing.assert_close(other(ids).logits, model(ids).logits, rtol=0, atol=0)
        assert restored.state_dict() == service.state_dict()
        assert fork.summary()["interventions"][0]["state"] == "ended"
        # A later absolute schedule starts at the same step in the fork.
        ctl.step(5)
        fork.step(5)
        assert restored.state_dict() == service.state_dict()
        assert fork.summary()["interventions"][1]["since_step"] == 6
        torch.testing.assert_close(other(ids).logits, model(ids).logits, rtol=0, atol=0)
    finally:
        restored.remove()


def test_invalid_fork_restore_is_atomic_and_snapshots_do_not_alias(controlled):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    prior = service.state_dict()
    state = deepcopy(prior)
    state["interventions"].append({**deepcopy(state["interventions"][0]),
                                  "specification": add_spec(point="missing")})
    hook = next(iter(ctl._ivs.values())).hook
    with pytest.raises(InterventionError):
        service.load_state_dict(state)
    assert service.state_dict() == prior and not hook.removed
    specs = service.specifications()
    specs[0]["vector"][0] = 999
    assert service.specifications()[0]["vector"][0] == 0.1


def test_clean_patch_forward_does_not_advance_managed_step(controlled):
    model, tokenizer, ctl, service = controlled
    callback = model.register_forward_hook(lambda *_: ctl.step())
    try:
        service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"})
        ctl.step()
        assert ctl.n == 1 and not service.executor.preparing_source
        model(**tokenizer("xyz"))
        assert ctl.n == 2
    finally:
        callback.remove()


def test_reconfiguration_validates_first_and_ends_old_effects(controlled):
    model, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook
    state = service.state_dict()
    bad = dictionary_for(model)
    bad.metadata["model_hash"] = "unknown"
    with pytest.raises(InterventionError):
        service.configure(dictionary=bad, tokenizer=tokenizer)
    assert service.state_dict() == state and not hook.removed
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    assert hook.removed and not service.specifications()
    assert ctl.summary()["interventions"][0]["state"] == "ended"


def test_watch_replacement_releases_patch_sources_and_old_handles(controlled):
    model, _, ctl, service = controlled
    service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"})
    ctl.step()
    source = weakref.ref(service.executor._cache["abc", "layers.0.resid_pre"])
    hook = next(iter(ctl._ivs.values())).hook
    next_model, _ = make_fixture()
    replacement = ctl.interventions_on(next_model)
    try:
        assert source() is None and hook.removed
        assert service.dictionary is None
        service.remove()  # A stale handle cannot remove the replacement.
        assert ctl.intervention_service is replacement
    finally:
        replacement.remove()


def test_dtype_overflow_refuses_before_hook_install():
    model, tokenizer = make_fixture(dtype=torch.float32)
    service = InterventionService(model)
    with pytest.raises(InterventionError, match="dtype range"):
        service.install(add_spec(vector=[1e100] * 64))
    assert not any(m._forward_hooks for m in model.modules())


@pytest.fixture
def public_watch(monkeypatch):
    import caliper
    from caliper import _controls, _managed
    ctl = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", ctl)
    monkeypatch.setattr(caliper, "_CTL", ctl)
    monkeypatch.setattr(caliper, "_auto_tried", True)
    monkeypatch.setattr(_managed, "current", lambda: None)
    monkeypatch.setattr(caliper, "serve", lambda: None)
    return caliper, ctl


def test_inference_watch_announces_capabilities_and_releases_resources(public_watch):
    from caliper import _net_programs, _programs
    from caliper.interp import publish
    caliper, ctl = public_watch
    model, tokenizer = make_fixture()
    count = _net_programs._watches
    handle = caliper.watch(model, activations=False, groups=False, scene=False)
    try:
        assert handle._run is None and "lr" not in ctl.params
        assert _net_programs._watches == count + 1
        assert set(_net_programs.capabilities()) == {"programs", "interventions"}
        assert _programs._STATE["model"] is model and _programs._STATE["optimizer"] is None
        service = ctl.intervention_service
        service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, dictionary_id="fixture")
        publish("intervention.example", torch.ones(2, 3))
        service.set(add_spec())
        caliper.step()
        hook = next(iter(ctl._ivs.values())).hook
        assert service.specifications() and not hook.removed
    finally:
        handle.remove()
    assert hook.removed and service.dictionary is None
    assert ctl.intervention_service is None and _net_programs._watches == count
    assert _programs._STATE["model"] is None
    handle.remove()
    assert _net_programs._watches == count


def test_training_watch_enables_capabilities_once(public_watch):
    from caliper import _net_programs
    caliper, ctl = public_watch
    model, _ = make_fixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    count = _net_programs._watches
    with caliper.watch(model, optimizer, activations=False, groups=False, scene=False):
        assert _net_programs._watches == count + 1
        assert ctl.intervention_service.model is model and "lr" in ctl.params
    assert _net_programs._watches == count and "lr" not in ctl.params
    assert ctl.intervention_service is None


def test_old_watch_removal_preserves_new_inference_capabilities(public_watch):
    from caliper import _net_programs
    caliper, ctl = public_watch
    first_model, _ = make_fixture()
    second_model, _ = make_fixture()
    count = _net_programs._watches
    first = caliper.watch(first_model, activations=False, groups=False)
    second = caliper.watch(second_model, activations=False, groups=False)
    try:
        assert _net_programs._watches == count + 2
        first.remove()
        assert _net_programs._watches == count + 1
        assert ctl.intervention_service.model is second_model
    finally:
        first.remove()
        second.remove()
    assert _net_programs._watches == count


@pytest.mark.parametrize("optimizer_watch", [False, True])
def test_failed_public_attach_releases_capabilities_and_hooks(public_watch, monkeypatch, optimizer_watch):
    from caliper import _net_programs, _programs
    caliper, ctl = public_watch
    model, _ = make_fixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if optimizer_watch else None
    count = _net_programs._watches

    def fail(*args):
        raise ValueError("attachment failed")

    monkeypatch.setattr(_programs, "attach", fail)
    with pytest.raises(ValueError, match="attachment failed"):
        caliper.watch(model, optimizer, activations=False, groups=False, scene=False)
    assert _net_programs._watches == count and ctl.intervention_service is None
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert "lr" not in ctl.params


def test_invalid_watch_model_releases_inference_capabilities(public_watch):
    from caliper import _net_programs
    caliper, _ = public_watch
    count = _net_programs._watches
    with pytest.raises(TypeError):
        caliper.watch(object())
    assert _net_programs._watches == count


def test_matching_discriminators_are_accepted_and_missing_dictionary_is_refused():
    from caliper._controls import Controls
    model, _ = make_fixture()
    ctl = Controls()
    service = ctl.interventions_on(model)
    try:
        service.set(add_spec(op="add"))
        before = service.state_dict()
        with pytest.raises(InterventionError) as failure:
            service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1, "value": 0})
        assert failure.value.code == "intervene.unknown_dictionary"
        assert service.state_dict() == before
    finally:
        service.remove()


def test_feature_encoder_failure_preserves_forward_and_ends_hook(controlled, monkeypatch):
    model, tokenizer, ctl, service = controlled
    ids = tokenizer("abc")["input_ids"]
    baseline = model(ids).logits
    service.set({"kind": "clamp", "point": "layers.0.mlp_out", "feature": 1, "value": 0})
    ctl.step()
    hook = next(iter(ctl._ivs.values())).hook

    def fail(*args):
        raise RuntimeError("fixture encoder failure")

    monkeypatch.setattr(service.dictionary, "encode", fail)
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    assert hook.removed and ctl.summary()["interventions"][0]["state"] == "ended"
    torch.testing.assert_close(model(ids).logits, baseline, rtol=0, atol=0)
    assert hook.calls == 1


def test_add_changes_only_selected_tokens_at_the_point(case):
    from caliper.interp import capture
    model, tokenizer, service = case
    ids = tokenizer("abc")["input_ids"].repeat(2, 1)
    with capture(model, ["layers.1.mlp_out"]) as original:
        model(ids)
    expected = original["layers.1.mlp_out"].clone()
    expected[1, 2] += torch.arange(64, dtype=expected.dtype) * 0.01
    service.install({"kind": "add", "point": "layers.1.mlp_out", "vector": list(range(64)),
                     "coefficient": 0.01, "positions": [2]})
    with service.rows([1]), capture(model, ["layers.1.mlp_out"]) as observed:
        model(ids)
    torch.testing.assert_close(observed["layers.1.mlp_out"], expected, rtol=0, atol=1e-12)


def test_managed_clean_source_skips_public_step_and_program_execution(public_watch, monkeypatch):
    from caliper import _managed, _programs
    caliper, ctl = public_watch
    model, tokenizer = make_fixture()
    watch = caliper.watch(model, activations=False, groups=False, scene=False)
    service = ctl.intervention_service
    service.configure(dictionary=dictionary_for(model), tokenizer=tokenizer, dictionary_id="fixture")
    managed = object.__new__(_managed.ManagedRun)
    managed._pkg = caliper
    managed.stopping = False
    callback = model.register_forward_hook(managed._forward)
    program_steps = []
    monkeypatch.setattr(_programs, "ACTIVE", True)
    monkeypatch.setattr(_programs, "at_step", program_steps.append)
    try:
        service.set({"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "abc"})
        caliper.step()
        assert ctl.n == 1 and program_steps == [1]
        assert service.specifications() and not service.executor.preparing_source
        model(**tokenizer("xyz"))
        assert ctl.n == 2 and program_steps == [1, 2]
    finally:
        callback.remove()
        watch.remove()


@pytest.mark.parametrize("identifier", ["", 0, False, "\ud800"])
def test_invalid_dictionary_identifier_preserves_active_hook(controlled, identifier):
    _, tokenizer, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    previous = service.state_dict()
    hook = next(iter(ctl._ivs.values())).hook
    with pytest.raises(InterventionError):
        service.configure(dictionary=service.dictionary, tokenizer=tokenizer, dictionary_id=identifier)
    assert service.state_dict() == previous and not hook.removed


@pytest.mark.parametrize("specification", [add_spec(point="\ud800"),
    {"kind": "patch", "point": "layers.0.resid_pre", "source_prompt": "\ud800"}])
def test_invalid_utf8_refuses_as_a_structured_error(controlled, specification):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    prior = service.state_dict()
    with ctl.lock:
        refused = ctl._intervene({"set": specification})
    assert refused[0] in ("intervene.unknown_point", "intervene.bad_request")
    assert service.state_dict() == prior


@pytest.mark.parametrize("mutation", ["bool_version", "float_version", "active_without_step", "pending_with_step"])
def test_invalid_fork_lifecycle_refuses_before_replacement(controlled, mutation):
    _, _, ctl, service = controlled
    service.set(add_spec())
    ctl.step()
    prior = service.state_dict()
    invalid = deepcopy(prior)
    if mutation.endswith("version"):
        invalid["version"] = True if mutation == "bool_version" else 1.0
    elif mutation == "active_without_step":
        invalid["interventions"][0]["since_step"] = None
    else:
        invalid["interventions"][0]["state"] = "pending"
    hook = next(iter(ctl._ivs.values())).hook
    with pytest.raises(InterventionError):
        service.load_state_dict(invalid)
    assert service.state_dict() == prior and not hook.removed
