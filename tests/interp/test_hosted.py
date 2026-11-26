"""Tests for HostedInference controls, service handlers, checkpoints, and readiness."""
import caliper
from caliper import _controls, _programs as P
from caliper.interp import _publish
from caliper.interp._hosted import CHECKPOINT_FORMAT, HostedInference, fixture_resources
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
