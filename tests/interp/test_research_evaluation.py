"""Independent CPU evidence for versioned research loss and source accounting."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from caliper.interp import capture, load_dictionary
from tools.interp import eval_dictionary as evaluation
from tools.interp import evaluation_protocol as protocol
from tools.interp.memory_limits import CleanStop, MemoryGuard
from tools.interp.text_stream import TextStream
from .test_dictionary import memory_readers
from .test_gemma_import_binding import artifact
from .test_gemma_runtime import gemma_model, runtime_dictionary
from .test_points import hooks
from .test_trace import qwen_for_trace, random_dictionary
from .test_training_stream import documents


def guard(limit=12):
    return MemoryGuard(limit, "cpu", pressure=lambda: 50, resident=lambda: 0,
                       footprint=lambda: 0, peak=lambda: 0, driver=lambda: 0)


def tokenizer(model, tokens=None):
    def encode(text, **kwargs):
        if tokens is not None:
            return tokens
        digest = hashlib.sha256(text.encode()).digest()
        return [4 + value % (model.config.vocab_size - 4) for value in digest[:4]]

    return SimpleNamespace(bos_token_id=getattr(model.config, "bos_token_id", None),
                           eos_token_id=getattr(model.config, "eos_token_id", None) or 1,
                           pad_token_id=getattr(model.config, "pad_token_id", None) or 0,
                           add_bos_token=False, encode=encode)


def fixture(family):
    model = gemma_model() if family == "gemma" else qwen_for_trace(layers=2)
    dictionary = runtime_dictionary(model) if family == "gemma" else random_dictionary(model, latents=8, k=3)
    return model, dictionary


@contextmanager
def oracle(model, dictionary, active, mask):
    """Patch original forward methods independently of evaluator hooks."""
    methods, reads = [], {}
    try:
        for i in active:
            block = model.model.layers[i]
            old = block.mlp.forward
            methods.append((block.mlp, old))
            if model.config.model_type == "gemma2":
                def remember(x, *args, old=old, i=i, **kwargs):
                    reads[i] = x
                    return old(x, *args, **kwargs)
                block.mlp.forward = remember
                norm = block.post_feedforward_layernorm
                old_norm = norm.forward
                methods.append((norm, old_norm))
                def replace(x, *args, old_norm=old_norm, i=i, **kwargs):
                    actual = old_norm(x, *args, **kwargs)
                    replacement = dictionary.layers[i](reads.pop(i))
                    return torch.where(mask[..., None], replacement, actual)
                norm.forward = replace
            else:
                def replace(x, *args, old=old, i=i, **kwargs):
                    actual = old(x, *args, **kwargs)
                    return torch.where(mask[..., None], dictionary.layers[i](x), actual)
                block.mlp.forward = replace
        yield
    finally:
        for module, method in reversed(methods):
            module.forward = method


def independent_ce(model, prepared, excluded):
    logits = model(prepared.input_ids, attention_mask=prepared.attention_mask, use_cache=False).logits
    losses = []
    for row in range(len(prepared.input_ids)):
        for target in range(1, prepared.input_ids.shape[1]):
            token = prepared.input_ids[row, target].item()
            if (prepared.attention_mask[row, target] and prepared.attention_mask[row, target - 1]
                    and token not in excluded):
                losses.append(F.cross_entropy(logits[row, target - 1][None], torch.tensor([token])))
    return sum(loss.item() for loss in losses), len(losses)


@pytest.mark.parametrize("family", ["gemma", "qwen"])
def test_independent_isolated_and_all_layer_loss_oracle(family):
    model, dictionary = fixture(family)
    policy = protocol.research_policy(model, 6, tokenizer(model))
    # EOS, source BOS, source padding, partial rows, and synthetic padding coexist.
    rows = [[4, 5, policy["eos_ids"][0], 0], [6, 7]]
    batch = protocol.research_contexts(rows, policy, device="cpu", pad_to=6)
    before_hooks = hooks(model)
    with torch.no_grad():
        baseline, targets = independent_ce(model, batch, policy["excluded_ids"])
        expected = []
        for active in ([0], [1], [0, 1]):
            with oracle(model, dictionary, active, batch.content_mask):
                loss, n = independent_ce(model, batch, policy["excluded_ids"])
                assert n == targets
                expected.append(loss)
        points = dictionary.metadata["read_points"] + dictionary.metadata["write_points"]
        with capture(model, points) as saved:
            model(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        means, variance, errors, hits = [], [], [], []
        for i, layer in enumerate(dictionary.layers):
            x = saved[dictionary.metadata["read_points"][i]][batch.content_mask]
            y = saved[dictionary.metadata["write_points"][i]][batch.content_mask].double()
            features = layer.encode(x)
            means.append(y.mean(0))
            variance.append((y - y.mean(0)).square().sum())
            errors.append((y - layer.decode(features).double()).square().sum())
            hits.append((features > 0).sum(0))
        saved.close(clear=True)
    actual = evaluation.evaluate_research_batch(model, dictionary, batch, 6, chunk_tokens=2,
                                                health=lambda: None, check_binding=lambda: None)
    assert actual.count == 4 and actual.source_tokens == 6
    assert actual.ce_tokens == targets
    assert actual.original_ce == pytest.approx(baseline, rel=1e-6)
    assert actual.isolated_ce.tolist() == pytest.approx(expected[:2], rel=1e-6)
    assert actual.replaced_ce == pytest.approx(expected[2], rel=1e-6)
    assert abs(expected[2] - expected[0]) > 1e-5
    assert abs(expected[2] - expected[1]) > 1e-5
    torch.testing.assert_close(actual.mean, torch.stack(means))
    torch.testing.assert_close(actual.variance_sum, torch.stack(variance))
    torch.testing.assert_close(actual.error_sum, torch.stack(errors), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(actual.hits, torch.stack(hits))
    metrics = actual.metrics()
    assert metrics["per_layer_l0"] == [x.sum().item() / 4 for x in hits]
    assert metrics["per_layer_fvu"] == pytest.approx([e.item() / v.item() for e, v in zip(errors, variance)], rel=1e-6)
    assert hooks(model) == before_hooks


def test_genuine_special_masks_and_native_no_bos():
    model, _ = fixture("gemma")
    model.config.bos_token_id, model.config.eos_token_id = 2, 1
    policy = protocol.research_policy(model, 7, tokenizer(model))
    batch = protocol.research_contexts([[4, 1, 5, 2, 0], [6]], policy, device="cpu", pad_to=7)
    assert batch.input_ids.tolist() == [[2, 4, 1, 5, 2, 0, 0], [2, 6, 0, 0, 0, 0, 0]]
    assert batch.content_mask.tolist() == [[False, True, False, True, False, False, False],
                                          [False, True, False, False, False, False, False]]
    assert batch.prediction_mask.tolist() == [[True, False, True, False, False, False],
                                             [True, False, False, False, False, False]]
    model, _ = fixture("qwen")
    assert getattr(model.config, "bos_token_id", None) is None
    policy = protocol.research_policy(model, 1024, tokenizer(model))
    assert policy["prefix_ids"] == [] and policy["bos_policy"] == "native_no_bos"
    assert policy["content_context"] == 1024 and policy["first_token_loss"] == "no_predictor"
    batch = protocol.research_contexts([[4, 5, 1], [6]], policy, device="cpu")
    assert batch.input_ids.tolist() == [[4, 5, 1], [6, 0, 0]]
    assert batch.prediction_mask.tolist() == [[True, False], [False, False]]
    invalid = tokenizer(model)
    invalid.bos_token_id = invalid.pad_token_id
    with pytest.raises(ValueError, match="BOS"):
        protocol.research_policy(model, 1024, invalid)


@pytest.mark.parametrize("family", ["gemma", "qwen"])
def test_special_outputs_remain_actual_and_features_stay_chunked(family, monkeypatch):
    model, dictionary = fixture(family)
    policy = protocol.research_policy(model, 6, tokenizer(model))
    batch = protocol.research_contexts([[4, 5, policy["eos_ids"][0]], [7]], policy, device="cpu", pad_to=6)
    original, replaced, handles = {}, {}, []
    from caliper.interp._points import _architecture
    probes = _architecture(model)
    try:
        for i, layer in enumerate(dictionary.layers):
            encode = layer.encode
            def bounded(x, encode=encode):
                assert x.shape[0] <= 2 and x.ndim == 2
                return encode(x)
            monkeypatch.setattr(layer, "encode", bounded)
            module = model.get_submodule(probes[dictionary.metadata["write_points"][i]].module)
            handles.append(module.register_forward_hook(lambda m, a, out, i=i: original.__setitem__(i, out.clone())))
        with torch.no_grad(), protocol.research_mlps(model, dictionary, [0, 1], batch.content_mask, chunk_tokens=2):
            for i in range(2):
                module = model.get_submodule(probes[dictionary.metadata["write_points"][i]].module)
                handles.append(module.register_forward_hook(lambda m, a, out, i=i: replaced.__setitem__(i, out.clone())))
            evaluation.hidden_forward(model, batch.input_ids, attention_mask=batch.attention_mask)
        for i in range(2):
            torch.testing.assert_close(original[i][~batch.content_mask], replaced[i][~batch.content_mask], rtol=0, atol=0)
            assert not torch.equal(original[i][batch.content_mask], replaced[i][batch.content_mask])
    finally:
        for handle in handles:
            handle.remove()


def persist(folder, model, dictionary):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    for i, layer in enumerate(dictionary.layers):
        save_file(layer.state_dict(), folder / f"layer_{i:02d}.safetensors")


def run(folder, model, dictionary, **kwargs):
    with TextStream(documents() * 3, tokenizer(model), "quality") as stream:
        return evaluation.evaluate(model, dictionary, stream, folder, protocol="research_v1", tokens=13,
                                   context=4, batch_tokens=8, chunk_tokens=2, memory=guard(), **kwargs)


@pytest.mark.parametrize("family", ["gemma", "qwen"])
@pytest.mark.parametrize("interruption", ["baseline", "isolated", "all"])
def test_exact_resume_after_partial_layer_sweep(artifact, tmp_path, monkeypatch, family, interruption):
    if family == "gemma":
        folder, model, dictionary, _ = artifact
    else:
        model, dictionary = fixture("qwen")
        folder = tmp_path / "qwen"
        persist(folder, model, dictionary)
    clean = tmp_path / "uninterrupted"
    import shutil
    shutil.copytree(folder, clean)
    expected = run(clean, model, dictionary)
    legacy = (folder / "dictionary.json").read_bytes()
    implementation = evaluation.hidden_forward
    calls = 0
    at = {"baseline": 5, "isolated": 6, "all": 8}[interruption]
    before_hooks = hooks(model)
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == at:
            raise CleanStop("partial layer sweep")
        return implementation(*args, **kwargs)
    monkeypatch.setattr(evaluation, "hidden_forward", interrupted)
    partial = run(folder, model, dictionary)
    assert 0 < partial["source_tokens"] < 13
    assert hooks(model) == before_hooks
    state_folder = folder / "evaluation-research-v1"
    state = json.loads((state_folder / "state.json").read_text())
    assert state["source_tokens"] == partial["source_tokens"]
    assert state["pending_ids"]["ids"]
    monkeypatch.setattr(evaluation, "hidden_forward", implementation)
    actual = run(folder, model, dictionary, resume=True)
    assert actual["source_tokens"] == 13
    assert actual["metric_observations"] == 11
    assert not actual["source_budget_complete"]
    for name in ("source_tokens", "metric_observations", "cross_entropy_tokens", "fvu", "per_layer_fvu",
                 "per_layer_l0", "per_layer_dead_fraction", "original_cross_entropy", "replaced_cross_entropy",
                 "per_layer_cross_entropy", "per_layer_cross_entropy_delta", "content_segmentation"):
        assert actual[name] == expected[name], name
    for directory in (clean / "evaluation-research-v1", state_folder):
        state = json.loads((directory / "state.json").read_text())
        tensors = load_file(directory / state["tensors"])
        if directory == state_folder:
            for name in tensors:
                torch.testing.assert_close(tensors[name], expected_tensors[name], rtol=0, atol=0)
        expected_tensors = tensors
    assert (folder / "dictionary.json").read_bytes() == legacy
    assert not (folder / "evaluation").exists()
    assert actual["checkpoint_sha256"] == hashlib.sha256((state_folder / "state.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["protocol", "mask", "source", "context", "tensor", "model", "legacy"])
def test_resume_refuses_changed_settings_before_tensor_load(artifact, monkeypatch, change):
    folder, model, dictionary, _ = artifact
    run(folder, model, dictionary)
    path = folder / "evaluation-research-v1/state.json"
    state = json.loads(path.read_text())
    if change == "protocol":
        state["settings"]["protocol"]["version"] = 2
    elif change == "mask":
        state["settings"]["context_policy"]["excluded_ids"] = []
    elif change == "source":
        state["settings"]["source"]["source_id"] = "changed"
    elif change == "context":
        state["settings"]["context"] = 8
    elif change == "tensor":
        state["tensor_binding"]["tensor_hash"] = "changed"
    elif change == "model":
        state["model_binding"]["forward_settings_hash"] = "changed"
    else:
        state["version"] = 1
    path.write_text(json.dumps(state))
    before = path.read_bytes(), (folder / "evaluation-research-v1/quality.json").read_bytes()
    import safetensors
    original = safetensors.safe_open
    def refuse_stats(filename, **kwargs):
        if "stats-" in str(filename):
            pytest.fail("statistics load preceded protocol validation")
        return original(filename, **kwargs)
    monkeypatch.setattr(safetensors, "safe_open", refuse_stats)
    with pytest.raises(ValueError, match="changed across resume"):
        run(folder, model, dictionary, resume=True)
    assert before == (path.read_bytes(), (folder / "evaluation-research-v1/quality.json").read_bytes())


def test_memory_refusal_precedes_context_allocation_and_preserves_cursor(artifact, monkeypatch):
    folder, model, dictionary, _ = artifact
    run(folder, model, dictionary)
    path = folder / "evaluation-research-v1/state.json"
    before = path.read_bytes()
    monkeypatch.setattr(protocol, "research_contexts", lambda *a, **kw: pytest.fail("context allocation ran"))
    monkeypatch.setattr(evaluation, "EvaluationBinding", lambda *a, **kw: pytest.fail("binding allocation ran"))
    with TextStream(documents(), tokenizer(model), "quality") as stream:
        with pytest.raises(CleanStop, match="workspace"):
            evaluation.evaluate(model, dictionary, stream, folder, protocol="research_v1", tokens=13,
                                context=1024, batch_tokens=1024, memory=guard(.001), resume=True)
    assert path.read_bytes() == before


def test_host_reserve_and_historical_peak_remain_lower_limits(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8_000_000_999))
    with pytest.raises(CleanStop, match="workspace"):
        protocol.admit_workspace(guard(40), 1000)
    memory = guard(1)
    memory.peak = 2_000_000_000
    with pytest.raises(CleanStop, match="memory limit"):
        protocol.admit_workspace(memory, 1)


@pytest.mark.parametrize("family", ["gemma", "qwen"])
@pytest.mark.parametrize("failure", ["encode", "forward", "install", "nonfinite"])
def test_replacement_exception_restores_every_hook_and_module(family, failure, monkeypatch):
    model, dictionary = fixture(family)
    policy = protocol.research_policy(model, 4, tokenizer(model))
    batch = protocol.research_contexts([[4, 5]], policy, device="cpu", pad_to=4)
    originals = tuple(model.modules())
    before = hooks(model)
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")
    if failure == "encode":
        monkeypatch.setattr(dictionary.layers[1], "encode", fail)
    elif failure == "forward":
        monkeypatch.setattr(model.model.layers[1].mlp, "forward", fail)
    elif failure == "install":
        monkeypatch.setattr(model.model.layers[1].mlp, "register_forward_pre_hook", fail)
    else:
        with torch.no_grad():
            dictionary.layers[1].W_dec[0, 0] = float("nan")
    with torch.no_grad(), pytest.raises((RuntimeError, ValueError), match="injected|nonfinite"):
        with protocol.research_mlps(model, dictionary, [0, 1], batch.content_mask, chunk_tokens=2):
            evaluation.hidden_forward(model, batch.input_ids, attention_mask=batch.attention_mask)
    assert hooks(model) == before
    assert tuple(model.modules()) == originals


def test_periodic_checkpoint_during_sweep_replays_only_uncommitted_batch(artifact, monkeypatch, tmp_path):
    folder, model, dictionary, _ = artifact
    import shutil
    clean = tmp_path / "periodic_clean"
    shutil.copytree(folder, clean)
    expected = run(clean, model, dictionary)
    implementation = evaluation.hidden_forward
    now, calls = [0.0], [0]
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: now[0])
    checkpoint_counts = []
    save = evaluation.save_research
    def recorded(*args, **kwargs):
        checkpoint_counts.append((now[0], args[1].source_tokens))
        return save(*args, **kwargs)
    monkeypatch.setattr(evaluation, "save_research", recorded)
    def delayed(*args, **kwargs):
        calls[0] += 1
        now[0] += 121
        if calls[0] == 8:
            raise RuntimeError("simulated process failure during all-layer pass")
        return implementation(*args, **kwargs)
    monkeypatch.setattr(evaluation, "hidden_forward", delayed)
    with pytest.raises(RuntimeError, match="simulated process"):
        run(folder, model, dictionary)
    assert len(checkpoint_counts) >= 7
    assert max(b[0] - a[0] for a, b in zip(checkpoint_counts, checkpoint_counts[1:])) < 240
    # Multiple heartbeats can carry the same committed count without a layer merge.
    assert len({count for _, count in checkpoint_counts}) < len(checkpoint_counts)
    path = folder / "evaluation-research-v1/state.json"
    state = json.loads(path.read_text())
    assert 0 < state["source_tokens"] < 13 and state["pending_ids"]["ids"]
    assert hooks(model) == 0
    monkeypatch.setattr(evaluation, "hidden_forward", implementation)
    actual = run(folder, model, dictionary, resume=True)
    for name in ("fvu", "per_layer_l0", "per_layer_cross_entropy", "replaced_cross_entropy", "source_tokens"):
        assert actual[name] == expected[name]


@pytest.mark.parametrize("field", ["original_ce", "replaced_ce", "isolated_ce", "mean", "variance_sum", "error_sum"])
def test_nonfinite_evidence_cannot_form_a_report(field):
    stats = protocol.ResearchStats(evaluation.QualityStats(1, 2, 3))
    stats.source_tokens = stats.count = stats.ce_tokens = 200_000
    stats.variance_sum.fill_(2)
    stats.hits.fill_(1)
    if field in ("original_ce", "replaced_ce"):
        setattr(stats, field, float("nan"))
    else:
        getattr(stats, field).flatten()[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        stats.metrics()


def test_dataset_mean_fvu_and_source_completion_are_independent_of_metric_count():
    first = protocol.ResearchStats(evaluation.QualityStats(1, 2, 2))
    second = protocol.ResearchStats(evaluation.QualityStats(1, 2, 2))
    a = torch.tensor([[0., 0.], [2., 0.]], dtype=torch.float64)
    b = torch.tensor([[10., 0.], [12., 0.]], dtype=torch.float64)
    for stats, values in ((first, a), (second, b)):
        stats.count, stats.ce_tokens, stats.source_tokens = 2, 1, 100_000
        stats.mean[0] = values.mean(0)
        stats.variance_sum[0] = (values - values.mean(0)).square().sum()
        stats.error_sum[0] = 1
        stats.hits[:] = torch.tensor([[2, 1]])
        stats.original_ce, stats.replaced_ce = 1, 2
        stats.isolated_ce.fill_(2)
    first.merge(second)
    expected_variance = (torch.cat([a, b]) - torch.cat([a, b]).mean(0)).square().sum().item()
    metrics = first.metrics()
    assert metrics["per_layer_fvu"] == [2 / expected_variance]
    assert metrics["per_layer_l0"] == [1.5]
    assert metrics["metric_observations"] == 4 and metrics["source_tokens"] == 200_000
    assert metrics["source_budget_complete"] is True
    # The unchanged CE threshold rejects this complete research measurement.
    assert metrics["status"] == "failed" and metrics["thresholds"]["cross_entropy_delta"] == .5


def test_all_special_batch_consumes_source_without_metric_or_loss_observations():
    model, dictionary = fixture("gemma")
    policy = protocol.research_policy(model, 4, tokenizer(model))
    batch = protocol.research_contexts([[policy["eos_ids"][0], policy["prefix_ids"][0], 0]], policy, device="cpu")
    result = evaluation.evaluate_research_batch(model, dictionary, batch, 3, chunk_tokens=2,
                                                health=lambda: None, check_binding=lambda: None)
    assert result.source_tokens == 3 and result.count == result.ce_tokens == 0
    aggregate = protocol.ResearchStats(evaluation.quality_stats(dictionary.metadata))
    aggregate.merge(result)
    assert aggregate.source_tokens == 3
    assert aggregate.metrics()["per_layer_fvu"] == [None, None]


def test_existing_legacy_report_and_resume_survive_research_mode(artifact):
    folder, model, dictionary, _ = artifact
    with TextStream(documents() * 3, tokenizer(model), "quality") as stream:
        evaluation.evaluate(model, dictionary, stream, folder, tokens=13, context=4, batch_tokens=8, chunk_tokens=2)
    legacy_paths = [folder / "dictionary.json", folder / "evaluation/state.json"]
    before = [p.read_bytes() for p in legacy_paths]
    run(folder, model, dictionary)
    assert [p.read_bytes() for p in legacy_paths] == before
    with TextStream(documents() * 3, tokenizer(model), "quality") as stream:
        resumed = evaluation.evaluate(model, dictionary, stream, folder, tokens=13, context=4,
                                      batch_tokens=8, chunk_tokens=2, resume=True)
    assert resumed["evaluated_tokens"] == 13


@pytest.mark.parametrize("family", ["gemma", "qwen"])
def test_context_rows_follow_documents_and_final_source_budget(family):
    from tools.interp.text_stream import document_identity
    model, _ = fixture(family)
    policy = protocol.research_policy(model, 4, tokenizer(model))
    pending = {"ids": [], "documents": [], "positions": [], "exhausted": False}
    segments, predictions, observed, consumed = [], 0, 0, 0
    with TextStream(documents() * 3, tokenizer(model), "quality") as stream:
        while consumed < 13:
            batch, spans = evaluation._gemma_batch(stream, pending, 13 - consumed, policy, 8,
                                                    device="cpu", health=lambda: None,
                                                    prepare=protocol.research_contexts)
            count = sum(span[2] for span in spans)
            for field in ("ids", "documents", "positions"):
                del pending[field][:count]
            consumed += count
            segments.extend(spans)
            predictions += batch.prediction_mask.sum().item()
            observed += batch.content_mask.sum().item()
    ids = [document_identity(row["text"])[1] for row in documents() * 3
           if document_identity(row["text"])[0] == "quality"]
    capacity = 3 if family == "gemma" else 4
    assert segments == [[ids[0], 0, capacity], [ids[0], capacity, 5 - capacity],
                        [ids[1], 0, capacity], [ids[1], capacity, 5 - capacity], [ids[2], 0, 3]]
    assert consumed == 13 and observed == 11
    assert predictions == (11 if family == "gemma" else 8)


@pytest.mark.parametrize("which", ["baseline", "isolated", "all"])
def test_nonfinite_loss_keeps_prior_cursor_and_cleans_model(artifact, monkeypatch, which):
    folder, model, dictionary, _ = artifact
    batch = evaluation.evaluate_research_batch
    batches = [0]
    def partial(*args, **kwargs):
        batches[0] += 1
        if batches[0] == 2:
            raise CleanStop("preserve committed batch")
        return batch(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate_research_batch", partial)
    run(folder, model, dictionary)
    monkeypatch.setattr(evaluation, "evaluate_research_batch", batch)
    path = folder / "evaluation-research-v1/state.json"
    report = folder / "evaluation-research-v1/quality.json"
    before = path.read_bytes(), report.read_bytes()
    original = evaluation.cross_entropy_sum
    calls = [0]
    def nonfinite(*args, **kwargs):
        calls[0] += 1
        result, count = original(*args, **kwargs)
        if calls[0] == {"baseline": 1, "isolated": 2, "all": 4}[which]:
            return float("nan"), count
        return result, count
    monkeypatch.setattr(evaluation, "cross_entropy_sum", nonfinite)
    with pytest.raises(ValueError, match="nonfinite"):
        run(folder, model, dictionary, resume=True)
    assert (path.read_bytes(), report.read_bytes()) == before and hooks(model) == 0


def test_stop_control_after_resume_keeps_committed_observations(artifact, monkeypatch):
    from tools.interp.memory_limits import StopControl
    folder, model, dictionary, _ = artifact
    original = evaluation.hidden_forward
    calls = [0]
    stop = StopControl()
    # StopControl uses its signal handler to set the public stop state.
    import signal
    def interrupted(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 6:
            stop._signal(signal.SIGINT, None)
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, "hidden_forward", interrupted)
    partial = run(folder, model, dictionary, stop=stop)
    assert 0 < partial["source_tokens"] < 13 and hooks(model) == 0
    path = folder / "evaluation-research-v1/state.json"
    before = path.read_bytes()
    with pytest.raises(CleanStop):
        run(folder, model, dictionary, stop=stop, resume=True)
    assert path.read_bytes() == before
    monkeypatch.setattr(evaluation, "hidden_forward", original)
    assert run(folder, model, dictionary, resume=True)["source_tokens"] == 13


@pytest.mark.parametrize("family", ["gemma", "qwen"])
def test_head_chunks_and_hidden_states_release_before_next_pass(family):
    import weakref
    model, dictionary = fixture(family)
    batch = protocol.research_contexts([[4, 5, 6]], protocol.research_policy(model, 5, tokenizer(model)),
                                       device="cpu", pad_to=5)
    logits, states, passes, handles = [], [], [], []
    def before_head(module, args):
        assert args[0].ndim == 2 and len(args[0]) <= 2
        assert all(ref() is None for ref in logits)
    def after_head(module, args, output):
        logits.append(weakref.ref(output))
    def before_model(module, args):
        assert all(ref() is None for ref in states)
        passes.append(1)
    def after_model(module, args, output):
        states.append(weakref.ref(output.last_hidden_state))
    try:
        head = model.get_output_embeddings()
        handles.append(head.register_forward_pre_hook(before_head))
        handles.append(head.register_forward_hook(after_head))
        handles.append(model.model.register_forward_pre_hook(before_model))
        handles.append(model.model.register_forward_hook(after_model))
        result = evaluation.evaluate_research_batch(model, dictionary, batch, 3, chunk_tokens=2,
                                                    health=lambda: None, check_binding=lambda: None)
        assert len(passes) == 4  # Baseline, two isolated layers, and all layers.
        assert result.source_tokens == 3
        assert all(ref() is None for ref in [*states, *logits])
    finally:
        for handle in handles:
            handle.remove()


def test_qwen_upper_scope_keeps_isolated_and_all_active_fields(monkeypatch):
    from .test_upper_half import fallback_manifest, tiny_model
    model, dictionary = fixture("qwen")
    fallback = fallback_manifest(tiny_model(2))["training"]["fallback"]
    dictionary.metadata["training"].update(fallback=fallback, source_id=fallback["source_id"])
    policy = protocol.research_policy(model, 4, tokenizer(model))
    prepared = protocol.research_contexts([[4, 5, 6]], policy, device="cpu")
    with torch.no_grad():
        baseline, targets = independent_ce(model, prepared, policy["excluded_ids"])
        with oracle(model, dictionary, [1], prepared.content_mask):
            expected, _ = independent_ce(model, prepared, policy["excluded_ids"])
    monkeypatch.setattr(dictionary.layers[0], "encode", lambda *a: pytest.fail("inactive dictionary ran"))
    result = evaluation.evaluate_research_batch(model, dictionary, prepared, 3, chunk_tokens=2,
                                                health=lambda: None, check_binding=lambda: None).metrics()
    assert result["active_layers"] == [1] and result["status"] == "fallback_partial"
    assert result["per_layer_cross_entropy"] == [None, pytest.approx(expected / targets, rel=1e-6)]
    assert result["replaced_cross_entropy"] == pytest.approx(expected / targets, rel=1e-6)
    assert result["cross_entropy_delta"] == pytest.approx((expected - baseline) / targets, abs=1e-6)
    for key in ("per_layer_fvu", "per_layer_l0", "per_layer_dead_fraction", "per_layer_cross_entropy_delta"):
        assert result[key][0] is None


def test_research_supplied_guard_cannot_raise_family_ceiling(artifact, monkeypatch):
    folder, model, dictionary, _ = artifact
    memory = guard(40)
    memory.limit = 50_000_000_000
    memory.peak = 40_000_000_001
    monkeypatch.setattr(evaluation, "EvaluationBinding", lambda *a, **kw: pytest.fail("binding allocation ran"))
    with TextStream(documents(), tokenizer(model), "quality") as stream:
        with pytest.raises(CleanStop, match="memory limit"):
            evaluation.evaluate(model, dictionary, stream, folder, protocol="research_v1", tokens=13,
                                context=4, batch_tokens=8, memory=memory)
