"""CPU refusal evidence for source bounds, masks, and evaluation cursors."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from caliper.interp._context import context_policy, source_segments
from tools.interp import eval_dictionary as evaluation
from tools.interp import evaluation_protocol as protocol
from tools.interp.memory_limits import CleanStop
from tools.interp.text_stream import TextStream, TokenBatch
from .test_gemma_import_binding import artifact
from .test_points import hooks
from .test_research_evaluation import fixture, guard, tokenizer
from .test_training_stream import documents


def run(folder, model, dictionary, mode, **kwargs):
    with TextStream(documents() * 3, tokenizer(model), "quality") as stream:
        return evaluation.evaluate(model, dictionary, stream, folder, protocol=mode, tokens=13,
                                   context=4, batch_tokens=8, chunk_tokens=2, memory=guard(), **kwargs)


@pytest.mark.parametrize("capacity", [True, 0, -1, 2.0, "3", 2**63])
def test_source_capacity_refusal(capacity):
    with pytest.raises(ValueError, match="capacity"):
        list(source_segments([7], [0], capacity, final=True))


@pytest.mark.parametrize("docs,positions", [([7, 8], [0, 5]), ([7, 7], [0, 3]),
                                           ([7, 2**63], [0, 0]), ([7, 7], [0, True])])
def test_source_coordinate_refusal(docs, positions):
    with pytest.raises(ValueError):
        list(source_segments(docs, positions, 3, final=True))
    assert list(source_segments([7, 7, 7], [12, 13, 0], 3, final=True)) == [(0, 2), (2, 3)]


@pytest.mark.parametrize("field,value", [("bos_token_id", 1.0), ("pad_token_id", False),
                                        ("eos_token_id", 2.0), ("add_bos_token", "false")])
def test_research_tokenizer_type_refusal(field, value):
    model, _ = fixture("gemma")
    tok = tokenizer(model)
    setattr(tok, field, value)
    with pytest.raises(ValueError):
        protocol.research_policy(model, 4, tok)


@pytest.mark.parametrize("field", ["bos_token_id", "pad_token_id"])
@pytest.mark.parametrize("change", ["absent", "float"])
def test_legacy_tokenizer_declarations_are_strict(field, change):
    model, _ = fixture("gemma")
    tok = tokenizer(model)
    if change == "absent":
        delattr(tok, field)
    else:
        setattr(tok, field, float(getattr(tok, field)))
    with pytest.raises(ValueError, match="special IDs"):
        context_policy(model, 4, tokenizer=tok)


@pytest.mark.parametrize("mode", ["legacy", "research_v1"])
@pytest.mark.parametrize("kind", ["dtype", "shape", "device"])
def test_mask_refusal_restores_original_model(mode, kind):
    model, dictionary = fixture("gemma")
    mask = torch.ones((1, 4), dtype=torch.bool)
    if kind == "dtype":
        mask = mask.long()
    elif kind == "shape":
        mask = mask[:, :-1]
    else:
        mask = mask.to("meta")
    before = tuple(model.modules()), hooks(model)
    with pytest.raises(ValueError, match="mask|context"):
        replacement = (evaluation.replace_all_mlps(model, dictionary, content_mask=mask) if mode == "legacy"
                       else protocol.research_mlps(model, dictionary, [0, 1], mask, chunk_tokens=2))
        with torch.no_grad(), replacement:
            evaluation.hidden_forward(model, torch.tensor([[1, 4, 5, 6]]))
    assert (tuple(model.modules()), hooks(model)) == before


def test_legacy_qwen_rejects_mask_instead_of_ignoring_it():
    model, dictionary = fixture("qwen")
    with pytest.raises(ValueError, match="require Gemma"):
        with evaluation.replace_all_mlps(model, dictionary, content_mask=torch.ones((1, 4), dtype=torch.bool)):
            pytest.fail("mask was ignored")
    assert hooks(model) == 0


@pytest.mark.parametrize("mode", ["legacy", "research_v1"])
@pytest.mark.parametrize("change", ["count", "targets", "loss", "elapsed", "policy_bool", "version_float",
                                    "segments_gap", "segments_document", "pending_vocab", "pending_bool",
                                    "pending_gap", "stream_vocab", "stream_offset", "stream_rows",
                                    "stream_emitted", "stream_buffer", "stream_version", "source_cursor",
                                    "stream_offset_replay", "stream_document"])
def test_invalid_resume_preserves_publications_before_tensor_load(artifact, monkeypatch, mode, change):
    folder, model, dictionary, _ = artifact
    batch_name = "evaluate_research_batch" if mode == "research_v1" else "evaluate_batch"
    implementation = getattr(evaluation, batch_name)
    calls = [0]
    def interrupt(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise CleanStop("retain a pending source batch")
        return implementation(*args, **kwargs)
    monkeypatch.setattr(evaluation, batch_name, interrupt)
    run(folder, model, dictionary, mode)
    monkeypatch.setattr(evaluation, batch_name, implementation)
    target = folder / ("evaluation-research-v1" if mode == "research_v1" else "evaluation")
    path = target / "state.json"
    state = json.loads(path.read_text())
    if change == "count":
        state["count"] = True
    elif change == "targets":
        state["ce_tokens"] -= 1
        if mode == "research_v1":
            state["cross_entropy_tokens"] -= 1
    elif change == "loss":
        state["original_ce"] = float("nan")
    elif change == "elapsed":
        state["elapsed_seconds"] = "one"
    elif change == "policy_bool":
        state["settings"]["context_policy"]["pad_id"] = False
    elif change == "version_float":
        state["version"] = float(state["version"])
    elif change == "segments_gap":
        state["content_segments"][1][1] += 1
    elif change == "segments_document":
        state["content_segments"][1][0] += 1
    elif change == "pending_vocab":
        state["pending_ids"]["ids"][0] = model.config.vocab_size
    elif change == "pending_bool":
        state["pending_ids"]["ids"][0] = True
    elif change == "pending_gap":
        state["pending_ids"]["positions"][0] = 11
    elif change == "stream_vocab":
        state["stream"]["pending"][0] = model.config.vocab_size
    elif change == "stream_offset":
        state["stream"]["offset"] += 100
    elif change == "stream_offset_replay":
        state["stream"]["offset"] -= 1
    elif change == "stream_document":
        state["stream"]["document_id"] += 1
    elif change == "stream_rows":
        state["stream"]["rows"] = 10000
    elif change == "stream_emitted":
        state["stream"]["emitted"] += 1
    elif change == "stream_buffer":
        state["stream"]["buffer"] = [[4], [7], []]
    elif change == "stream_version":
        state["stream"]["version"] = True
    else:
        state["stream"]["source_state"] = {"row": 10000}
    path.write_text(json.dumps(state))
    report = target / "quality.json" if mode == "research_v1" else folder / "dictionary.json"
    before = path.read_bytes(), report.read_bytes()
    import safetensors
    original = safetensors.safe_open
    def refuse_stats(filename, **kwargs):
        if "stats-" in str(filename):
            pytest.fail("statistics file opened before cursor validation")
        return original(filename, **kwargs)
    monkeypatch.setattr(safetensors, "safe_open", refuse_stats)
    monkeypatch.setattr(evaluation, "load_file", lambda *a, **kw: pytest.fail("statistics payload loaded"))
    with pytest.raises(ValueError):
        run(folder, model, dictionary, mode, resume=True)
    assert (path.read_bytes(), report.read_bytes()) == before
    assert hooks(model) == 0


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite"])
def test_invalid_tensor_evidence_cannot_publish(artifact, monkeypatch, change):
    folder, model, dictionary, _ = artifact
    run(folder, model, dictionary, "research_v1")
    target = folder / "evaluation-research-v1"
    state = json.loads((target / "state.json").read_text())
    tensors = load_file(target / state["tensors"])
    if change == "shape":
        tensors["hits"] = tensors["hits"][:, :1].contiguous()
    elif change == "dtype":
        tensors["hits"] = tensors["hits"].float()
    else:
        tensors["mean"][0, 0] = float("inf")
    save_file(tensors, target / state["tensors"])
    before = (target / "state.json").read_bytes(), (target / "quality.json").read_bytes()
    if change != "nonfinite":
        monkeypatch.setattr(evaluation, "load_file", lambda *a, **kw: pytest.fail("malformed tensor payload loaded"))
    with pytest.raises(ValueError):
        run(folder, model, dictionary, "research_v1", resume=True)
    assert ((target / "state.json").read_bytes(), (target / "quality.json").read_bytes()) == before


def test_out_of_range_source_refuses_before_context_allocation():
    model, _ = fixture("gemma")
    policy = protocol.research_policy(model, 4, tokenizer(model))
    values = [torch.tensor(v) for v in ([4, model.config.vocab_size], [7, 7], [0, 1])]
    stream = SimpleNamespace(next_tokens=lambda *a, **kw: TokenBatch(*values))
    pending = {"ids": [], "documents": [], "positions": [], "exhausted": False}
    with pytest.raises(ValueError, match="token IDs"):
        evaluation._gemma_batch(stream, pending, 2, policy, 4, device="cpu", health=lambda: None,
                                 vocab_size=model.config.vocab_size,
                                 prepare=lambda *a, **kw: pytest.fail("context allocation ran"))


def test_source_buffer_checkpoint_resume_is_exact(artifact, monkeypatch, tmp_path):
    import shutil
    folder, model, dictionary, _ = artifact
    clean = tmp_path / "source_buffer_clean"
    shutil.copytree(folder, clean)
    expected = run(clean, model, dictionary, "research_v1")
    original = TextStream.next_tokens
    now = [0.0]
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: now[0])
    def interrupted(self, count, *, health):
        def during_source():
            if self.buffer[0]:
                now[0] = 121.0
                health()
                raise RuntimeError("source buffer interruption")
            health()
        return original(self, count, health=during_source)
    monkeypatch.setattr(TextStream, "next_tokens", interrupted)
    with pytest.raises(RuntimeError, match="source buffer"):
        run(folder, model, dictionary, "research_v1")
    state = json.loads((folder / "evaluation-research-v1/state.json").read_text())
    assert state["source_tokens"] == 0 and state["stream"]["buffer"][0]
    assert not state["pending_ids"]["ids"] and hooks(model) == 0
    monkeypatch.setattr(TextStream, "next_tokens", original)
    actual = run(folder, model, dictionary, "research_v1", resume=True)
    for key in ("per_layer_fvu", "per_layer_cross_entropy", "replaced_cross_entropy", "content_segmentation"):
        assert actual[key] == expected[key]
