"""NUL-free display labels with unchanged token IDs and model inputs."""
import torch

from caliper import _interpret_protocol as protocol
from caliper.interp import Tracer
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._labels import display_text, pack_labels
from caliper.interp._service import InterpretationService
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_trace import exact_dictionary


def unpack_label(row):
    raw = b"".join(int(word).to_bytes(4, "little") for word in row)
    return raw.split(b"\0", 1)[0].decode("utf-8")


def execute(service, method, params):
    replies = []
    assert service.submit({"version": 1, "method": method, "params": params},
                          lambda *args: replies.append(args)) == ("queued", None)
    assert service.step(1) == 1
    assert replies[0][0] == "ok", replies
    assert protocol.bounded(replies[0][1])
    return replies[0][1]


def test_display_labels_escape_nul_and_keep_utf8_boundaries():
    value = "a\0é"
    assert display_text(value) == r"a\x00" + "é"
    assert protocol.text(display_text("\ud800\0"))
    labels = pack_labels([value, "é" * 32, "token"], 4)
    assert labels.dtype == torch.uint32 and labels.shape == (4, 16)
    assert unpack_label(labels[0]) == r"a\x00" + "é"
    assert unpack_label(labels[1]) == "é" * 31
    assert unpack_label(labels[2]) == "token"
    assert labels[3].tolist() == [0] * 16
    assert not protocol.bounded({"text": value})
    assert not protocol.request({"version": 1, "method": "trace.run", "params": {"prompt": value}})


def test_dead_feature_show_and_examples_keep_zero_token_id(tmp_path, monkeypatch):
    model, tokenizer = TinyLM(layers=1, width=8, heads=2, latents=8, k=2), TinyTokenizer()
    with torch.no_grad():
        model.layers[0].mlp.W_dec[0].zero_()
    write_fixture(tmp_path, model)
    dictionary = exact_dictionary(model)
    build_index(tmp_path, [("fixture", [0, 65, 0, 66])], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"})
    published = {}
    monkeypatch.setattr("caliper.interp.publish", lambda name, tensor: published.update({name: tensor}))
    with FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        service = InterpretationService(model=model, dictionary=dictionary, index=index)
        shown = execute(service, "feature.show", {"layer": 0, "feature": 0})
        assert shown["promoted_tokens"][0] == {"token_id": 0, "token": r"\x00", "score": 0.0}
        feature = int(index.counts[0].argmax())
        result = execute(service, "feature.examples", {"layer": 0, "feature": feature})
        assert result["examples"]
        for row in result["examples"]:
            assert 0 in row["token_ids"]
            assert all(label == r"\x00" for token, label in zip(row["token_ids"], row["tokens"]) if token == 0)
        assert 0 in published["examples.tokens"][0].tolist()
        assert r"\x00" in [unpack_label(row) for row in published["examples.label_text"]]
        service.close()


def test_trace_read_escapes_nul_without_changing_prompt_ids(monkeypatch):
    model, tokenizer = TinyLM(layers=1, width=8, heads=2, latents=8, k=2), TinyTokenizer()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    dictionary = exact_dictionary(model)
    monkeypatch.setattr("caliper.interp.publish", lambda *args: None)
    tracer = Tracer(model, tokenizer, dictionary)
    prompt = "A\0B"
    graph = tracer.trace(prompt, max_targets=0, max_nodes=32, max_edges=0)
    assert tracer._prompt == prompt
    assert [token["token_id"] for token in graph.tokens] == tokenizer.encode(prompt)
    assert next(token for token in graph.tokens if token["token_id"] == 0)["text"] == r"\x00"
    assert next(node for node in graph.nodes if node["kind"] == "logit" and node["token_id"] == 0)["label"] == r"\x00"
    service = InterpretationService(model=model)
    service.graph = graph
    execute(service, "trace.read", {})
    service.close()
