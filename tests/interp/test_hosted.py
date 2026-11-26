"""Tests for HostedInference controls, service handlers, checkpoints, and readiness."""
import caliper
from caliper import _controls, _programs as P, _wire
from caliper.interp import _publish
from caliper.interp._hosted import (
    CHECKPOINT_FORMAT,
    DEFAULT_PROMPT,
    READ_ONLY_PARAMS,
    HostedInference,
    fixture_resources,
)
from .test_token_labels import execute


def test_hosted_inference_lifecycle_clamp_generate_sweep_validate_and_checkpoint(tmp_path, monkeypatch):
    from caliper._net import current_server

    monkeypatch.setenv("CALIPER_SESSION", "0")
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    ctl = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", ctl)
    monkeypatch.setattr(caliper, "_CTL", ctl)
    monkeypatch.setattr(P, "_PROGS", {})
    monkeypatch.setattr(P, "_IMPORTS", {})
    monkeypatch.setattr(P, "_LIVE", {})

    dict_dir = fixture_resources(tmp_path / "tiny-fixture")
    assert (dict_dir / "layer_0.safetensors").is_file()
    assert (dict_dir / "index" / "index.json").is_file()

    run = HostedInference("tiny-lm", dict_dir, "ABCD")
    run.start()
    try:
        status = run.service.submit({"version": 1, "method": "trace.status", "params": {}}, None)[1]
        assert status["inference"]["state"] == "ready"
        assert {"output.probs", "output.tokens", "output.label_text"} <= set(published)

        trace_res = execute(run.service, "trace.run", {"prompt": "ABCD", "max_targets": 16})
        assert trace_res["state"] == "done"
        feat_idx = next(i for i, n in enumerate(run.service.graph.nodes) if n["kind"] == "feature")
        feat_node = run.service.graph.nodes[feat_idx]

        # Select feature node and apply slider clamp
        run.selected.set(feat_idx)
        run.clamp.set(1.5)
        msg = run.apply_clamp()
        assert "applied" in msg
        assert "Live slider clamp" in run.clamp_status.value

        # Checkpoint state_dict / load_checkpoint round-trip
        saved = {
            "format": CHECKPOINT_FORMAT,
            "version": 1,
            "step": ctl.n,
            "params": {"prompt": run.prompt.value},
            **run.state_dict(),
        }
        assert saved["owned_clamp"] is not None
        assert saved["selection"] == {
            "layer": feat_node["layer"],
            "feature": feat_node["feature"],
            "position": feat_node["position"],
        }

        # Generate 24 tokens
        gen_msg = run.generate()
        assert "24 tokens" in gen_msg
        assert run.inference_status()["generation"]["counts"] == [24, 24]

        # Sweep and validate through HostedService handlers
        sweep_res = execute(
            run.service,
            "patch.sweep",
            {
                "clean_prompt": "ABCD",
                "corrupted_prompt": "EFGH",
                "metric": {"positive_token": 40, "negative_token": 41},
            },
        )
        assert sweep_res["layers"] == 2 and "patch.grid" in published

        val_res = execute(run.service, "trace.validate", {})
        assert val_res["trace_id"] == run.service.graph.trace_id

        # Reset slider clamp
        run.reset()
        assert run.clamp.value == 0.0
        assert "No slider clamp" in run.clamp_status.value
    finally:
        run.close()

    # Verify checkpoint restore on a fresh HostedInference instance
    ctl2 = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", ctl2)
    monkeypatch.setattr(caliper, "_CTL", ctl2)
    restored = HostedInference("tiny-lm", dict_dir, "ABCD")
    restored.load_checkpoint(saved)
    restored.start()
    try:
        assert restored.selection == saved["selection"]
        assert restored._owned_clamp == saved["owned_clamp"]
    finally:
        restored.close()
        P.detach()
        srv = current_server()
        if srv is not None:
            srv.close()


def _isolated_hosted(tmp_path, monkeypatch, prompt="carrot"):
    monkeypatch.setenv("CALIPER_SESSION", "0")
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    ctl = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", ctl)
    monkeypatch.setattr(caliper, "_CTL", ctl)
    monkeypatch.setattr(P, "_PROGS", {})
    monkeypatch.setattr(P, "_IMPORTS", {})
    monkeypatch.setattr(P, "_LIVE", {})
    dict_dir = fixture_resources(tmp_path / "tiny-fixture")
    run = HostedInference("tiny-lm", dict_dir, prompt)
    return run, ctl, published


def test_prompt_change_retraces_regenerates_text_refreshes_examples_and_progress(tmp_path, monkeypatch):
    from caliper._net import current_server

    assert DEFAULT_PROMPT == "He saw a carrot and had to grab it, his hunger was like a starving"
    run, ctl, published = _isolated_hosted(tmp_path, monkeypatch, prompt="carrot")
    run.start()
    try:
        assert run.status.value == "ready"
        assert run.traced_prompt == "carrot"
        assert run.traced_show_more is False
        assert run.selected.value >= 0 and run.selection is not None
        assert run.baseline_text.value != "" and run.steered_text.value != ""
        assert {"trace.counts", "output.probs", "examples.tokens"} <= set(published)

        trace_calls = []
        status_history = []
        examples_calls = []
        orig_trace = run.service.tracer.trace
        orig_set_progress = run._set_progress
        orig_publish_examples = run.service.publish_feature_examples

        def spy_set_progress(state, message=None):
            status_history.append(state)
            return orig_set_progress(state, message)

        def spy_publish_examples(layer, feature, *, limit=16):
            examples_calls.append((layer, feature))
            return orig_publish_examples(layer, feature, limit=limit)

        def spy_trace(prompt, *, show_more=False, **kwargs):
            trace_calls.append({
                "prompt": prompt,
                "show_more": show_more,
                "status_during_trace": run.status.value,
                "inference_state_during_trace": run.inference_status()["state"],
            })
            return orig_trace(prompt, **kwargs)

        monkeypatch.setattr(run, "_set_progress", spy_set_progress)
        monkeypatch.setattr(run.service, "publish_feature_examples", spy_publish_examples)
        monkeypatch.setattr(run.service.tracer, "trace", spy_trace)

        initial_baseline = run.baseline_text.value
        initial_gen_hash = run.inference_status()["generation"]["prompt_hash"]
        initial_cmp_hash = run.inference_status()["comparison"]["prompt_hash"]

        # 1. Prompt change via Prompt field (Enter) retraces, recomputes bars, regenerates text, refreshes examples.
        run.prompt.set("hunger")
        run.boundary()

        assert trace_calls[-1] == {
            "prompt": "hunger",
            "show_more": False,
            "status_during_trace": "tracing",
            "inference_state_during_trace": "tracing",
        }
        assert status_history[:2] == ["tracing", "ready"]
        assert run.status.value == "ready"
        assert run.traced_prompt == "hunger"
        assert run.selected.value >= 0 and run.selection is not None
        assert run.service.graph.nodes[run.selected.value]["kind"] == "feature"
        assert examples_calls[-1] == (run.selection["layer"], run.selection["feature"])
        hunger_status = run.inference_status()
        assert hunger_status["comparison"]["prompt_hash"] != initial_cmp_hash
        assert hunger_status["generation"]["prompt_hash"] == hunger_status["prompt_hash"]
        assert hunger_status["generation"]["prompt_hash"] != initial_gen_hash
        expected_hunger = run.output.generate("hunger")
        assert run.baseline_text.value == expected_hunger["baseline"]
        assert run.steered_text.value == expected_hunger["steered"]
        assert run.baseline_text.value != initial_baseline

        # 2. Out-of-range clamp position on a shorter prompt is safely cleared without crashing.
        deep_feat_idx = max(
            (i for i, n in enumerate(run.service.graph.nodes) if n["kind"] == "feature"),
            key=lambda i: run.service.graph.nodes[i]["position"],
        )
        run.selected.set(deep_feat_idx)
        run.clamp.set(2.0)
        run.apply_clamp()
        assert run._owned_clamp is not None

        run.prompt.set("ox")
        run.boundary()
        assert run.status.value == "ready"
        assert run.traced_prompt == "ox"
        assert run._owned_clamp is None
        assert run.clamp.value == 0.0
        assert run.baseline_text.value == run.output.generate("ox")["baseline"]

        # 3. Prompt change via agent trace.run command retraces, regenerates text, refreshes examples, emits progress.
        status_history.clear()
        trace_res = execute(run.service, "trace.run", {"prompt": "lantern"})
        assert trace_res["state"] == "done"
        assert trace_calls[-1]["prompt"] == "lantern"
        assert trace_calls[-1]["show_more"] is False
        assert trace_calls[-1]["status_during_trace"] == "tracing"
        assert status_history == ["tracing", "ready"]
        assert run.prompt.value == "lantern"
        assert run.status.value == "ready"
        expected_lantern = run.output.generate("lantern")
        assert run.baseline_text.value == expected_lantern["baseline"]
        assert run.steered_text.value == expected_lantern["steered"]
        assert examples_calls[-1] == (run.selection["layer"], run.selection["feature"])

        # 4. Toggling show_more retraces the current prompt with show_more=True.
        status_history.clear()
        run.show_more.set(True)
        run.boundary()
        assert trace_calls[-1]["prompt"] == "lantern"
        assert trace_calls[-1]["show_more"] is True
        assert status_history == ["tracing", "ready"]
        assert run.traced_show_more is True

        # 5. Run action recomputes graph, output bars, generated text, and examples for current prompt and clamp.
        assert any(a.name == "run" for a in run.actions)
        run.clamp.set(1.5)
        run.apply_clamp()
        calls_before_run = len(trace_calls)
        status_history.clear()
        run_msg = run.run_prompt()
        assert "Traced" in run_msg
        assert len(trace_calls) == calls_before_run + 1
        assert trace_calls[-1] == {
            "prompt": "lantern",
            "show_more": True,
            "status_during_trace": "tracing",
            "inference_state_during_trace": "tracing",
        }
        assert status_history == ["tracing", "ready"]
        assert run.status.value == "ready"
        expected_steered = run.output.generate("lantern")
        assert run.baseline_text.value == expected_steered["baseline"]
        assert run.steered_text.value == expected_steered["steered"]
        assert examples_calls[-1] == (run.selection["layer"], run.selection["feature"])
    finally:
        run.close()
        P.detach()
        srv = current_server()
        if srv is not None:
            srv.close()


def test_output_rows_are_readonly(tmp_path, monkeypatch):
    from caliper._net import current_server

    assert {"output.baseline", "output.steered", "output.status", "clamp.status"} <= READ_ONLY_PARAMS
    run, ctl, _published = _isolated_hosted(tmp_path, monkeypatch, prompt="carrot")
    run.start()
    try:
        original_baseline = run.baseline_text.value
        original_steered = run.steered_text.value
        assert original_baseline != "" and original_steered != ""

        class _Transport:
            def __init__(self):
                self.replies = []

            def may_control(self, conn):
                return True

            def reply(self, conn, metadata):
                self.replies.append(metadata)

            def broadcast(self, *args, **kwargs):
                pass

            def controllers(self):
                return 1

        route = ctl.attach(_Transport())
        try:
            # Attempt wire edits to Baseline and Steered output rows.
            route.on_control(None, _wire.Control(1, "set", name="output.baseline", value="typed baseline").to_meta())
            route.on_control(None, _wire.Control(2, "set", name="output.steered", value="typed steered").to_meta())
            run.boundary()
            assert run.baseline_text.value == original_baseline
            assert run.steered_text.value == original_steered

            # Direct external edits are also reverted on the next boundary.
            run.baseline_text.set("manual edit")
            run.steered_text.set("manual steered edit")
            run.boundary()
            assert run.baseline_text.value == original_baseline
            assert run.steered_text.value == original_steered
        finally:
            route.detach()
    finally:
        run.close()
        P.detach()
        srv = current_server()
        if srv is not None:
            srv.close()


def test_staged_progress_covers_load_trace_generation_examples_and_errors(tmp_path, monkeypatch):
    import json
    from caliper._net import current_server

    monkeypatch.setenv("CALIPER_SESSION", "0")
    published = {}
    monkeypatch.setattr(_publish, "_TENSORS", published)
    ctl = _controls.Controls()
    monkeypatch.setattr(_controls, "CONTROLS", ctl)
    monkeypatch.setattr(caliper, "_CTL", ctl)
    monkeypatch.setattr(P, "_PROGS", {})
    monkeypatch.setattr(P, "_IMPORTS", {})
    monkeypatch.setattr(P, "_LIVE", {})

    dict_dir = fixture_resources(tmp_path / "tiny-fixture")
    assert "output.progress" in READ_ONLY_PARAMS

    seen: list[dict] = []
    orig_publish = ctl._publish

    def record_publish(*args, **kwargs):
        param = ctl.params.get("output.progress")
        if param is not None and isinstance(param.value, str) and param.value:
            seen.append(json.loads(param.value))
        return orig_publish(*args, **kwargs)

    monkeypatch.setattr(ctl, "_publish", record_publish)

    run = HostedInference("tiny-lm", dict_dir, "ABCD", defer_load=True)
    run.start()
    try:
        stages = [item["stage"] for item in seen]
        assert "weights" in stages
        assert "transcoders" in stages
        assert "index" in stages
        assert stages.index("weights") < stages.index("transcoders") < stages.index("index")
        assert "attribution" in stages
        assert "pruning" in stages
        assert "publishing" in stages
        assert stages.index("attribution") < stages.index("pruning") < stages.index("publishing")
        assert "output" in stages
        assert "generation" in stages
        assert stages[-1] == "ready"

        # Real determinate attribution batch progress and 24-token generation progress
        attr_counts = [
            (item["done"], item["total"])
            for item in seen
            if item["stage"] == "attribution" and item["total"] is not None
        ]
        assert attr_counts and all(0 <= d <= t and t > 0 for d, t in attr_counts)
        assert attr_counts[-1][0] == attr_counts[-1][1]

        gen_counts = [
            (item["done"], item["total"])
            for item in seen
            if item["stage"] == "generation" and item["total"] is not None
        ]
        assert (1, 24) in gen_counts and (24, 24) in gen_counts

        # trace.status mirrors the same progress state
        status = run.service.submit({"version": 1, "method": "trace.status", "params": {}}, None)[1]
        assert status["progress"]["state"] == "ready"
        assert status["inference"]["progress"]["state"] == "ready"

        # Standalone feature example lookup reports progress on the examples panel
        seen.clear()
        feat_idx = next(i for i, n in enumerate(run.service.graph.nodes) if n["kind"] == "feature")
        feat = run.service.graph.nodes[feat_idx]
        run.service.publish_feature_examples(feat["layer"], feat["feature"], limit=4)
        example_updates = [item for item in seen if item["stage"] == "examples"]
        assert example_updates and "examples" in example_updates[0]["panels"]
        assert seen[-1]["state"] == "ready"

        # Errors populate both overall and panel error state
        seen.clear()
        run._set_progress("error", "Test failure")
        err = json.loads(run.progress.value)
        assert err["state"] == "error"
        assert err["error"] == "Test failure"
        assert err["panels"] and all(p["error"] == "Test failure" for p in err["panels"].values())
    finally:
        run.close()
        P.detach()
        srv = current_server()
        if srv is not None:
            srv.close()
