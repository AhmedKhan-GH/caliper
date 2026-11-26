"""Published output changes preserve the accepted graph and its trace status."""
from copy import deepcopy

import pytest
import torch

from caliper.interp import _publish
from .test_hosted import _trace_at_boundary, action_order_run, hosted_run
from .test_replication_groups import replicated_hosted


def baseline(run):
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0,
                               profile="circuit_tracer_82996f14", batch_size=1)
    traced = deepcopy((run.traced_prompt, run.traced_specifications, run.traced_state, run.service.status))
    public = {name: (tensor, tensor.clone(), _publish._VERSIONS[name])
              for name, tensor in _publish._TENSORS.items() if name.startswith("trace.")}
    files = {path: path.read_bytes() for path in run.replication.directory.iterdir()}

    def unchanged():
        assert run.service.graph is graph
        assert (run.traced_prompt, run.traced_specifications, run.traced_state, run.service.status) == traced
        for name, (tensor, value, version) in public.items():
            assert _publish._resolve(name) is tensor and _publish._VERSIONS[name] == version
            torch.testing.assert_close(tensor, value, rtol=0, atol=0)
        for path, value in files.items():
            assert path.read_bytes() == value

    return unchanged


@pytest.mark.parametrize("change", ["toggle", "prompt", "intervention"])
def test_later_published_retrace_preserves_reference_graph(replicated_hosted, change):
    run, _ = replicated_hosted
    unchanged = baseline(run)
    run.apply_published_case(18)
    comparison = deepcopy(run._comparison)
    result = deepcopy(run.replication.last_result)
    run.retrace.set(True)
    run.boundary()
    unchanged()
    assert run.status.value == "Automatic retrace is unsupported for published actions. The current graph is unchanged."
    assert run._comparison == comparison and run.replication.last_result == result
    if change == "prompt":
        run.prompt.set("CD")
    elif change == "intervention":
        run.interventions.clear()
    run.boundary()
    unchanged()
    assert "unsupported" in run.status.value
    assert run._output_key == run._input_state()
    if change != "toggle":
        assert run._comparison != comparison
    run.retrace.set(False)
    run.boundary()
    unchanged()
    assert run.status.value == "ready" and not run._retrace


def test_published_retrace_notice_clears_after_action(replicated_hosted):
    run, _ = replicated_hosted
    unchanged = baseline(run)
    run.retrace.set(True)
    run.apply_published_case(18)
    comparison = deepcopy(run._comparison)
    assert "unsupported" in run.status.value
    run.retrace.set(False)
    run.boundary()
    unchanged()
    assert run.status.value == "ready" and not run._retrace
    assert run._comparison == comparison


def test_disabled_retrace_preserves_an_unrelated_output_error(replicated_hosted):
    run, _ = replicated_hosted
    baseline(run)
    run.retrace.set(True)
    run.apply_published_case(18)
    run.status.set("Unrelated output refusal.")
    run.retrace.set(False)
    run.boundary()
    assert run.status.value == "Unrelated output refusal."


def test_ordinary_retrace_retains_its_control_behavior(action_order_run):
    run, _ = action_order_run
    graph = _trace_at_boundary(run, prompt="AB", max_targets=0)
    run.prompt.set("CD")
    run.retrace.set(True)
    run.boundary()
    assert run.service.graph is not graph and run.traced_prompt == "CD"
    assert run.service.status["state"] == "done" and run.status.value == "ready"
    graph = run.service.graph
    run.retrace.set(False)
    run.prompt.set("EF")
    run.boundary()
    assert run.service.graph is graph and run.traced_prompt == "CD"
    assert run._output_key == run._input_state()
