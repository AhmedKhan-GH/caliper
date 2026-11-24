"""Paired output checks on the seeded CPU fixture."""
import pytest
import torch

from caliper._controls import Controls
from caliper.interp._output import Output, is_internal_forward, packed_labels
from caliper.interp._publish import _clear, _resolve
from .fixture import make_fixture
from .test_intervene import dictionary_for, explicit_forward


@pytest.fixture
def output_case():
    model, tokenizer = make_fixture()
    controls = Controls()
    interventions = controls.interventions_on(model)
    interventions.configure(dictionary=dictionary_for(model), tokenizer=tokenizer)
    output = Output(model, tokenizer, interventions)
    yield model, tokenizer, controls, interventions, output
    interventions.remove()
    _clear()


def test_paired_probabilities_use_one_batch_and_preserve_baseline(output_case):
    model, tokenizer, controls, service, output = output_case
    ids = tokenizer("AB")["input_ids"]
    logits, acts = explicit_forward(model, ids)
    feature = int(acts[-1][0, -1].argmax())
    service.set({"kind": "clamp", "point": "layers.3.mlp_out", "feature": feature, "value": 0, "positions": [2]})
    controls.step()
    expected, _ = explicit_forward(model, ids, edit=lambda layer, a: a.scatter(
        -1, torch.full_like(a[..., :1], feature, dtype=torch.long), 0) if layer == 3 else a)
    # The oracle changes every position in the last MLP. Only its last position reaches these logits.
    calls = []
    handle = model.register_forward_hook(lambda m, a, y: calls.append((y.logits.shape, is_internal_forward())))
    result = output.compare("AB")
    handle.remove()
    assert calls == [(torch.Size([2, 3, 512]), True)]
    assert not is_internal_forward()
    selected = result["tokens"]
    assert int(logits[0, -1].argmax()) in selected
    torch.testing.assert_close(_resolve("output.probs")[0], logits[0, -1].softmax(-1)[selected].float())
    torch.testing.assert_close(_resolve("output.probs")[1], expected[0, -1].softmax(-1)[selected].float())
    assert _resolve("output.tokens").dtype == torch.int32
    assert _resolve("output.probs").dtype == torch.float32
    assert _resolve("output.label_text").shape == (10, 16)
    assert controls.n == 1
    assert service.specifications()[0]["feature"] == feature


def test_generation_has_exactly_24_greedy_tokens_per_row(output_case):
    model, tokenizer, controls, service, output = output_case
    service.set({"kind": "ablate", "point": "layers.3.mlp"})
    controls.step()
    before = service.state_dict()
    calls = []
    handle = model.register_forward_hook(lambda m, a, y: calls.append(y.logits.detach()))
    result = output.generate("A")
    handle.remove()
    assert len(calls) == 24
    assert [len(row) for row in result["token_ids"]] == [24, 24]
    for step, logits in enumerate(calls):
        assert logits.shape[:2] == (2, 2 + step)
        assert logits[:, -1].argmax(-1).tolist() == [row[step] for row in result["token_ids"]]
    assert result["baseline"] == tokenizer.decode(result["token_ids"][0])
    assert result["steered"] == tokenizer.decode(result["token_ids"][1])
    assert service.state_dict() == before
    assert controls.n == 1


def test_utf8_labels_end_on_character_boundaries():
    labels = packed_labels(["a" * 63 + "é", "😀" * 17, "A\0B"], "cpu")
    raw = [b"".join(int(i).to_bytes(4, "little") for i in row) for row in labels.tolist()]
    assert raw[0].rstrip(b"\0").decode() == "a" * 63
    assert raw[1].decode() == "😀" * 16
    assert raw[2].rstrip(b"\0").decode() == "AB"


@pytest.mark.parametrize("prompt", ["", "x" * 32, "x" * 4097])
def test_prompt_caps_do_not_truncate(output_case, prompt):
    with pytest.raises(ValueError):
        output_case[-1].compare(prompt)


def test_failed_forward_restores_row_and_internal_scope(output_case, monkeypatch):
    model, tokenizer, controls, service, output = output_case
    from caliper.interp._intervene import selected_rows
    def fail(*a, **kw):
        assert selected_rows(model) == (1,)
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        output.compare("A")
    assert selected_rows(model) is None
    assert not is_internal_forward()
