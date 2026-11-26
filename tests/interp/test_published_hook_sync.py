"""Program activation hook changes preserve published action identity."""
from threading import Event, Thread

import pytest
import torch

from caliper import _programs
from caliper.interp import _publish, _replication_actions as actions
from .test_replication_groups import grouped, published


@pytest.mark.parametrize("change", ["install", "prune"])
@pytest.mark.parametrize("phase,publication", [("forward", False), ("forward", True), ("publication", True)])
def test_program_hook_changes_preserve_published_action(published, monkeypatch, change, phase, publication):
    model, _, controls, _, runtime = published
    monkeypatch.setitem(_programs._STATE, "model", model)
    monkeypatch.setattr(_programs, "_ACT_HOOKS", {})
    monkeypatch.setattr(_programs, "_ACTS", {})
    monkeypatch.setattr(_programs, "_PROGS", {})
    monkeypatch.setattr(_publish, "_TENSORS", {})
    monkeypatch.setattr(_publish, "_VERSIONS", {})
    if change == "prune":
        with _programs._LOCK, pytest.raises(_programs._Waiting):
            _programs._resolve("lm_head.act")
    start, probed, finished = Event(), Event(), Event()
    acquired, errors = [], []

    def mutate():
        try:
            assert start.wait(5)
            # The nonblocking probe makes the overlap deterministic without sleeps.
            held = _programs._LOCK.acquire(blocking=False)
            acquired.append(held)
            probed.set()
            if not held:
                _programs._LOCK.acquire()
            try:
                # Program mutation must retain the program-before-controls order.
                with controls.lock:
                    if change == "install":
                        with pytest.raises(_programs._Waiting):
                            _programs._resolve("lm_head.act")
                    else:
                        _programs._prune_act_hooks()
            finally:
                _programs._LOCK.release()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=mutate, daemon=True)
    forward = actions.forward_logits

    def synchronize():
        start.set()
        assert probed.wait(5)
        if acquired == [True]:
            assert finished.wait(5)

    def overlap(*args, **kwargs):
        if phase == "forward":
            synchronize()
        return forward(*args, **kwargs)

    def prepare(pair):
        if phase == "publication":
            synchronize()
        return _publish._prepare_many({"fixture.pair": pair})

    monkeypatch.setattr(actions, "forward_logits", overlap)
    worker.start()
    try:
        result = runtime.run(26, prepare=prepare if publication else None)
        assert finished.wait(5)
        worker.join(5)
        assert not worker.is_alive() and not errors
        assert acquired == [phase == "publication"]
        assert result["cell"] == 26 and runtime.last_attempt["status"] == "ok"
        assert runtime._model_snapshot.active
        assert ("lm_head" in _programs._ACT_HOOKS) == (change == "install")
        if publication:
            native = torch.load(runtime.output_directory / result["native_evidence"]["file"], weights_only=True)
            torch.testing.assert_close(_publish._resolve("fixture.pair"), native["logits"], rtol=0, atol=0)
        monkeypatch.setattr(actions, "forward_logits", forward)
        assert runtime.run(10)["cell"] == 10
        assert runtime._model_snapshot.active
    finally:
        start.set()
        worker.join(5)
        with _programs._LOCK:
            for handle in _programs._ACT_HOOKS.values():
                handle.remove()
            _programs._ACT_HOOKS.clear()
