"""CPU checks for document partitions, cursor state, and memory stops."""
import json

import pytest
import torch

from tools.interp.memory_limits import CleanStop, MemoryGuard, physical_footprint
from tools.interp.model_load import hidden_forward
from tools.interp.text_stream import TextStream, document_identity, partition_metadata
from .fixture import TinyLM, TinyTokenizer


def documents():
    return [{"text": f"Example document {i} describes a blue square and a green circle."} for i in range(300)]


def test_disjoint_partitions_and_duplicate_assignment():
    groups = {name: set() for name in ("train", "quality", "index")}
    for row in documents():
        name, identity = document_identity(row["text"])
        groups[name].add(identity)
        assert document_identity(row["text"]) == (name, identity)
    assert all(groups.values())
    assert not groups["train"] & groups["quality"]
    assert not groups["train"] & groups["index"]
    assert not groups["quality"] & groups["index"]
    metadata = partition_metadata()
    metadata["train"][0] = 17
    assert partition_metadata()["train"][0] == 0


@pytest.mark.parametrize("partition", ["train", "quality", "index"])
def test_exact_resume_with_partial_document(partition):
    source = documents()
    tokenizer = TinyTokenizer()
    stream = TextStream(source, tokenizer, partition)
    first = stream.next_tokens(17)
    state = json.loads(json.dumps(stream.state_dict()))
    assert 0 < state["offset"] < len(state["pending"])
    expected = stream.next_tokens(127)
    resumed = TextStream(source, tokenizer, partition, state=state)
    actual = resumed.next_tokens(127)
    assert len(first) == 17
    for name in ("input_ids", "document_ids", "positions"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    assert resumed.state_dict() == stream.state_dict()
    with pytest.raises(ValueError, match="source or partition"):
        TextStream(source, tokenizer, "index" if partition != "index" else "train", state=state)


def test_hf_style_source_cursor():
    class Source:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            rows = documents()
            while self.position < len(rows):
                row = rows[self.position]
                self.position += 1
                yield row

        def state_dict(self):
            return {"row": self.position}

        def load_state_dict(self, state):
            self.position = state["row"]

    stream = TextStream(Source(), TinyTokenizer(), "train")
    stream.next_tokens(80)
    state = stream.state_dict()
    assert state["source_state"]["row"] > 0
    resumed = TextStream(Source(), TinyTokenizer(), "train", state=state)
    assert torch.equal(stream.next_tokens(200).input_ids, resumed.next_tokens(200).input_ids)


def test_installed_hf_iterable_cursor_without_download():
    from datasets import IterableDataset
    def source():
        yield from documents()
    stream = TextStream(IterableDataset.from_generator(source), TinyTokenizer(), "train")
    stream.next_tokens(83)
    state = json.loads(json.dumps(stream.state_dict()))
    resumed = TextStream(IterableDataset.from_generator(source), TinyTokenizer(), "train", state=state)
    assert torch.equal(stream.next_tokens(311).input_ids, resumed.next_tokens(311).input_ids)


def test_no_vocabulary_projection_in_hidden_forward():
    model = TinyLM()
    tokens = torch.tensor([[2, 3, 4, 5]])
    expected = model(tokens).logits
    handle = model.lm_head.register_forward_pre_hook(lambda *_: pytest.fail("vocabulary projection ran"))
    try:
        hidden = hidden_forward(model, tokens)
    finally:
        handle.remove()
    torch.testing.assert_close(model.lm_head(hidden), expected, rtol=0, atol=0)


def test_memory_stop_thresholds():
    values = iter([21, 19, 21, 19, 19])
    guard = MemoryGuard(1, "cpu", resident=lambda: 0, pressure=lambda: next(values))
    for _ in range(4):
        guard.check("fixture")
    with pytest.raises(CleanStop, match="pressure"):
        guard.check("fixture")
    guard = MemoryGuard(1, "cpu", resident=lambda: 0, pressure=lambda: 14)
    with pytest.raises(CleanStop, match="pressure"):
        guard.check("fixture")
    guard = MemoryGuard(1, "cpu", resident=lambda: 1_000_000_001, pressure=lambda: 50)
    with pytest.raises(CleanStop, match="memory limit"):
        guard.check("fixture")


def test_peak_and_fallback_memory_limits():
    for options in (
        {"footprint": lambda: 1_000_000_001, "peak": lambda: 0, "driver": lambda: 0},
        {"footprint": lambda: 0, "peak": lambda: 1_000_000_001, "driver": lambda: 0},
        {"footprint": lambda: 0, "peak": lambda: 0, "driver": lambda: 1_000_000_001},
        {"footprint": lambda: None, "peak": lambda: 0, "driver": lambda: 600_000_001},
    ):
        guard = MemoryGuard(1, "cpu", resident=lambda: 400_000_000, pressure=lambda: 50, **options)
        with pytest.raises(CleanStop, match="memory limit"):
            guard.check("fixture")
        guard.resident = lambda: 0
        guard.driver_reader = lambda: 0
        guard.footprint = lambda: 0
        guard.peak_reader = lambda: 0
        with pytest.raises(CleanStop, match="memory limit"):
            guard.check("past_peak")


def test_physical_footprint_api():
    import sys
    if sys.platform == "darwin":
        assert physical_footprint() > 0
    else:
        assert physical_footprint() is None


def test_autorelease_scope_balances_success_and_failure(monkeypatch):
    from caliper.interp import _memory
    calls = []
    monkeypatch.setattr(_memory.sys, "platform", "darwin")
    monkeypatch.setattr(_memory, "_autorelease_functions", lambda: (
        lambda: calls.append("push") or 71, lambda pool: calls.append(("pop", pool)),
    ))
    with _memory.autorelease_scope("cpu"):
        assert calls == []
    with _memory.autorelease_scope("mps"):
        pass
    with pytest.raises(ValueError):
        with _memory.autorelease_scope("mps"):
            raise ValueError("fixture")
    assert calls == ["push", ("pop", 71), "push", ("pop", 71)]


def test_interrupted_document_scan_keeps_buffered_tokens():
    tokenizer = TinyTokenizer()
    expected = TextStream(documents(), tokenizer, "train").next_tokens(120)
    source = TextStream(documents(), tokenizer, "train")
    def stop():
        if len(source.buffer[0]) > 0:
            raise CleanStop("fixture scan interruption")
    with pytest.raises(CleanStop):
        source.next_tokens(120, health=stop)
    resumed = TextStream(documents(), tokenizer, "train", state=source.state_dict())
    actual = resumed.next_tokens(120)
    assert torch.equal(actual.input_ids, expected.input_ids)
    assert torch.equal(actual.document_ids, expected.document_ids)
    assert torch.equal(actual.positions, expected.positions)
