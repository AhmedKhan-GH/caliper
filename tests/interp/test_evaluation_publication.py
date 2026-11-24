"""Small CPU fixtures for memory refusal at evaluation publication boundaries."""
import json

import pytest

from tools.interp import eval_dictionary as evaluation
from tools.interp.checkpoint_state import dictionary_fingerprint, write_json
from tools.interp.evaluation_binding import EvaluationBinding
from tools.interp.memory_limits import CleanStop, MemoryGuard
from tools.interp.text_stream import TextStream
from .fixture import TinyTokenizer
from .test_dictionary import memory_readers
from .test_evaluation_binding import fixture, publications
from .test_training_stream import documents


@pytest.mark.parametrize("prior_count", [0, 4, 200_000])
@pytest.mark.parametrize("verification", [1, 2, 3])
@pytest.mark.parametrize("stop_kind", ["peak", "reserve", "pressure", "sustained_pressure"])
def test_memory_refusal_preserves_published_evidence(
        tmp_path, monkeypatch, capsys, prior_count, verification, stop_kind):
    model, dictionary = fixture(tmp_path)
    stream = TextStream([], TinyTokenizer(), "quality")
    settings = {"tokens": 200_000, "context": 4, "batch_tokens": 4, "chunk_tokens": 2}
    if prior_count:
        identity = dictionary_fingerprint(tmp_path, dictionary.metadata)
        binding = EvaluationBinding(tmp_path, dictionary, identity, model=model)
        # Artificial aggregates cover a completed report without a long token pass.
        stats = evaluation.quality_stats(dictionary.metadata)
        stats.count, stats.ce_tokens = prior_count, prior_count // 2
        stats.mean.fill_(0.5)
        stats.variance_sum.fill_(prior_count)
        stats.error_sum.fill_(prior_count / 10)
        stats.hits.fill_(1)
        stats.original_ce = stats.replaced_ce = stats.ce_tokens
        stream.emitted = prior_count
        evaluation.save_evaluation(tmp_path / "evaluation", stats, stream, None,
                                   identity, settings, 10, binding)
        dictionary.metadata["quality"] = stats.metrics()
        write_json(tmp_path / "dictionary.json", dictionary.metadata)
    before = publications(tmp_path)
    prior_tensors = None
    if prior_count:
        cursor = json.loads(before[1])
        prior_tensors = tmp_path / "evaluation" / cursor["tensors"]
        tensor_bytes = prior_tensors.read_bytes()
    readings = {"bytes": 8_000_000, "free": 80}
    guard = MemoryGuard(.064, "cpu", interval=0, pressure=lambda: readings["free"],
                        resident=lambda: readings["bytes"], footprint=lambda: readings["bytes"],
                        peak=lambda: readings["bytes"])
    final = False
    checks = 0
    state_dict, verify = stream.state_dict, EvaluationBinding.verify

    def final_state():
        nonlocal final
        final = True
        return state_dict()

    def memory_boundary(self):
        nonlocal checks
        if final:
            checks += 1
            if checks == verification:
                if stop_kind == "peak":
                    readings["bytes"] = guard.limit + 1
                elif stop_kind == "reserve":
                    readings["bytes"] = guard.limit - 1
                else:
                    readings["free"] = 14 if stop_kind == "pressure" else 19
        return verify(self)

    monkeypatch.setattr(stream, "state_dict", final_state)
    monkeypatch.setattr(EvaluationBinding, "verify", memory_boundary)
    capsys.readouterr()
    with pytest.raises(CleanStop, match="memory limit|memory pressure"):
        evaluation.evaluate(model, dictionary, stream, tmp_path, memory=guard,
                            resume=bool(prior_count), **settings)
    assert checks == verification
    assert publications(tmp_path) == before
    assert dictionary.metadata["quality"] == json.loads(before[0])["quality"]
    if prior_tensors:
        assert prior_tensors.read_bytes() == tensor_bytes
    assert "quality_end" not in capsys.readouterr().out
    if stop_kind == "peak":
        assert guard.peak > guard.limit
    elif stop_kind == "reserve":
        assert guard.peak == guard.limit - 1
    elif stop_kind == "sustained_pressure":
        assert guard.low == 2


def test_batch_memory_stop_preserves_prior_cursor_without_more_tensor_writes(tmp_path, monkeypatch, capsys):
    model, dictionary = fixture(tmp_path)

    class Stream(TextStream):
        def next_tokens(self, *args, **kwargs):
            if self.emitted == 4:
                raise CleanStop("fixture partial batch")
            return super().next_tokens(*args, **kwargs)

    settings = {"tokens": 8, "context": 4, "batch_tokens": 4, "chunk_tokens": 2}
    prior = evaluation.evaluate(model, dictionary, Stream(documents(), TinyTokenizer(), "quality"),
                                tmp_path, **settings)
    assert prior["evaluated_tokens"] == 4
    before = publications(tmp_path)
    cursor = json.loads(before[1])
    tensors = tmp_path / "evaluation" / cursor["tensors"]
    tensor_bytes = tensors.read_bytes()
    readings = {"bytes": 8_000_000}
    guard = MemoryGuard(.064, "cpu", interval=0, pressure=lambda: 80,
                        resident=lambda: readings["bytes"], footprint=lambda: readings["bytes"],
                        peak=lambda: readings["bytes"])

    def memory_stop(*args, **kwargs):
        readings["bytes"] = guard.limit + 1
        try:
            kwargs["health"]()
        finally:
            # A lower live footprint does not erase the process peak.
            readings["bytes"] = 8_000_000

    monkeypatch.setattr(evaluation, "evaluate_batch", memory_stop)
    monkeypatch.setattr(evaluation, "_save_tensors", lambda *a, **k: pytest.fail("tensor write after memory stop"))
    capsys.readouterr()
    with pytest.raises(CleanStop, match="memory limit"):
        evaluation.evaluate(model, dictionary, TextStream(documents(), TinyTokenizer(), "quality"),
                            tmp_path, memory=guard, resume=True, **settings)
    assert guard.peak > guard.limit
    assert publications(tmp_path) == before
    assert tensors.read_bytes() == tensor_bytes
    assert "quality_end" not in capsys.readouterr().out


def test_cursor_write_drift_precedes_final_identity_check(tmp_path, monkeypatch):
    model, dictionary = fixture(tmp_path)
    before = publications(tmp_path)
    write = evaluation.write_json

    def changed(path, value):
        write(path, value)
        if path.name == "state.pending.json":
            dictionary.layers[0].W_dec.data[0, 0] += 1

    monkeypatch.setattr(evaluation, "write_json", changed)
    with pytest.raises(ValueError, match="artifact binding"):
        evaluation.evaluate(model, dictionary, TextStream([], TinyTokenizer(), "quality"),
                            tmp_path, tokens=8, context=4, batch_tokens=4, chunk_tokens=2)
    assert publications(tmp_path) == before
