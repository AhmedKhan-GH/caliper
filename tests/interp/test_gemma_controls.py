"""Bounded CPU checks for Gemma control refusal and partial hook registration."""
import pytest
import torch

from caliper import _wire
from caliper._controls import Controls
from caliper.interp._intervene import Intervention
from caliper.interp._memory import CleanStop
from caliper.interp._service import InterpretationService, register_service, unregister_service
from .test_controls_admission import assert_callback_unlocked
from .test_gemma_family import gemma_model
from .test_gemma_frozen import Tokenizer
from .test_intervention_replay import hook_objects
from .test_service import Transport
from .test_trace import random_dictionary


@pytest.mark.parametrize("failure_type", [RuntimeError, CleanStop])
@pytest.mark.parametrize("registration", ["read", "enter", "clear", "write"])
def test_partial_registration_preserves_owners_and_terminal_replies(monkeypatch, failure_type, registration):
    model = gemma_model(layers=1).requires_grad_(False)
    model.config._name_or_path = "caliper-test/random-gemma2"
    model.config._commit_hash = "b" * 40
    tokenizer = Tokenizer()
    dictionary = random_dictionary(model)
    controls = Controls()
    service = controls.interventions_on(model)
    service.configure(dictionary=dictionary, tokenizer=tokenizer)
    interpretation = register_service(InterpretationService(model=model), controls=controls)
    timers, observed = [], []
    monkeypatch.setattr(controls._timers, "at", lambda when, fn, *args: timers.append((when, fn, args)))
    route = controls.attach(Transport())
    clamp = {"kind": "clamp", "point": "model.layers.0.mlp_out", "feature": 1, "value": 0}
    patch = {"kind": "patch", "point": "model.layers.0.resid_pre", "source_prompt": "a"}
    try:
        service.set(clamp)
        service.set(patch)
        controls.step()
        record, patch_record = controls._ivs.values()
        hook, applied, hooks = record.hook, record.applied_spec, hook_objects(model)
        executor_hooks = set(service.executor._hooks)
        cache = service.executor._cache
        sources, identity = list(cache.items()), service.executor._cache_identity
        before = model(**tokenizer("a"), use_cache=False).logits
        service.set(dict(clamp, value=1))
        request = record.spec
        service.set({"kind": "add", "point": "model.layers.0.mlp_in",
                     "vector": [1.] * 16, "coefficient": 2})
        add_record = list(controls._ivs.values())[-1]
        action = controls.action("recover", lambda: observed.append(service.specifications()))
        for request_id in (1, 2):
            assert route.on_control(None, _wire.Control(request_id, "run", name="recover").to_meta())["status"] == "queued"
        _, expire, args = next(item for item in timers if item[2] == (action, action.run_id))
        block = model.model.layers[0]
        registrations = [("read", block.mlp, "register_forward_hook"),
                         ("enter", block, "register_forward_pre_hook"),
                         ("clear", block, "register_forward_hook"),
                         ("write", block.post_feedforward_layernorm, "register_forward_hook")]
        reached, partial = [], []
        failure = failure_type("Gemma hook registration refused")

        def intercept(name, register):
            def registered(callback, **kwargs):
                owner = getattr(callback, "__self__", None)
                if isinstance(owner, Intervention):
                    assert_callback_unlocked(controls)
                    reached.append(name)
                    if name == registration:
                        assert len(owner.handles) == len(reached) - 1
                        partial.append(owner)
                        raise failure
                return register(callback, **kwargs)
            return registered

        with monkeypatch.context() as injected:
            for name, module, method in registrations:
                injected.setattr(module, method, intercept(name, getattr(module, method)))
            if failure_type is CleanStop:
                with pytest.raises(CleanStop) as caught:
                    controls.step()
                assert caught.value is failure and interpretation.failure is failure
            else:
                with pytest.warns(RuntimeWarning, match="Gemma hook registration refused"):
                    controls.step()
        assert reached == [name for name, _, _ in registrations][:len(reached)]
        assert reached[-1] == registration and len(partial) == 1
        assert partial[0].removed and not partial[0].handles
        assert partial[0].live_input is None and not partial[0].read_stack
        assert record.hook is hook and record.applied_spec is applied and not hook.removed
        assert record.spec is request and controls._ivs[(record.point, record.op)] is record
        assert hook_objects(model) == hooks and set(service.executor._hooks) == executor_hooks
        assert service.executor._cache is cache and service.executor._cache_identity is identity
        assert list(cache) == [key for key, _ in sources]
        assert all(cache[key] is tensor for key, tensor in sources)
        assert patch_record.error is None and add_record.state == "pending" and add_record.error is None
        torch.testing.assert_close(model(**tokenizer("a"), use_cache=False).logits, before, rtol=0, atol=0)
        assert not observed
        if failure_type is CleanStop:
            assert record.replace and record.spec["value"] == 1 and record.error is None
            assert record.error_code is None
            assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "error"), (2, "error")]
            assert all(reply.error == {"code": "action_failed", "message": str(failure)}
                       for reply in route.transport.replies)
            with pytest.raises(CleanStop) as repeated:
                controls.step()
            assert repeated.value is failure
        else:
            assert record.spec == applied and not record.replace
            assert record.error == "RuntimeError: Gemma hook registration refused"
            assert record.error_code == "intervene.hook_failed" and not route.transport.replies
            controls.step()
            assert observed == [service.specifications()] and observed[0][-1]["coefficient"] == 2
            assert record.hook is hook and patch_record.hook in executor_hooks
            assert [(reply.request_id, reply.status) for reply in route.transport.replies] == [(1, "ok"), (2, "ok")]
            # The exact retained callback must still own this control record.
            hook.on_error("retained owner callback")
            assert record.state == "ended" and record.hook is None and hook.removed
            assert record.error == "retained owner callback"
        expire(*args)
        assert len(route.transport.replies) == 2 and not controls.queue and not action.waiters
    finally:
        route.detach()
        unregister_service(interpretation, controls=controls)
        service.remove()
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())
