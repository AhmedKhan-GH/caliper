"""Sparse managed checkpoints retain their policy outside hosted pass time."""
import threading
from types import SimpleNamespace

import pytest

import caliper
from caliper import _programs as P, _managed
from caliper.interp import _dictionary
from .test_hosted_cadence import action_run


def managed_fixture(case, monkeypatch, tmp_path, *, state_s):
    run = case.run
    run.model = object()
    run.model_name, run.dictionary_path = 'fixture', 'fixture-dictionary'
    run.index = SimpleNamespace(metadata={'dictionary_hash': 'd', 'tokenizer_hash': 't'})
    run.replication_profile, run._verified_bindings = None, None
    run.checkpoint_prompt = 'fixture'
    run._inference_lock = threading.RLock()
    run.selection, run._owned_clamp = None, None
    run._generation = {'id': 0}
    run.interventions = SimpleNamespace(_ctl=run.controls, _state_dict=lambda: {})
    managed = _managed.ManagedRun.__new__(_managed.ManagedRun)
    managed.stopping, managed.inference = False, run
    managed.id, managed.parent, managed._pkg = 'audit', None, caliper
    managed.kept = _managed.KeptCheckpoints(str(tmp_path / 'kept'), clock=lambda: case.clock.now)
    managed.kept.extension = '.json'
    run.managed = managed
    calls = []
    def identity(*args, **kwargs):
        calls.append(run.controls.n)
        case.clock.now += state_s
        return 'fixture-identity'
    monkeypatch.setattr(_dictionary, 'model_identity', identity)
    return calls


@pytest.mark.parametrize("every", [1, 7])
def test_sparse_checkpoints_preserve_both_cadence_phases(action_run, monkeypatch, tmp_path, every):
    case = action_run
    case.clock.scheduler_s, case.clock.publish_s = .0099, .0001
    case.clock.idle_s, case.clock.waves = 61., 18
    case.program.every_min = every
    calls = managed_fixture(case, monkeypatch, tmp_path, state_s=.6)
    case.drive()
    assert len(calls) == 18 < case.run.controls.n
    assert case.run.managed.kept._spent == pytest.approx(10.8)
    assert case.shown == list(range(18))
    assert case.program.state == "running" and case.program.error is None
    assert P._CAD.step_s() == pytest.approx(.01)
    assert P._CAD.cost_s() == pytest.approx(.00002)
    assert P._CAD.n == P.N_MIN
    assert P._CAD.inference_s == pytest.approx(case.run.controls.n * .01 + 18 * .00002)
