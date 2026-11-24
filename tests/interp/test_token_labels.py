"""NUL-free display labels with unchanged token IDs and model inputs."""
from types import SimpleNamespace

import pytest
import torch

from caliper import _interpret_protocol as protocol
from caliper.interp import Tracer
from caliper.interp._graph import Graph, TOKEN_CAP
from caliper.interp._index import FeatureIndex, _tokenizer_hash, build_index
from caliper.interp._labels import display_text, pack_labels, token_label
from caliper.interp._service import InterpretationService
from .fixture import TinyLM, TinyTokenizer
from .test_dictionary import write_fixture
from .test_trace import exact_dictionary


class SparseTokenizer(TinyTokenizer):
    """Vocabulary holes and unused output rows coexist with high valid IDs."""

    def __init__(self):
        self.tokens = {0: "\0", 1: "", 2: "None", 65: "A", 256: "<bos>",
                       257: "<eos>", 258: "<pad>", 500: "Ġtail"}

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id):
        return self.tokens.get(token_id)

    def decode(self, ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(" tail" if self.tokens[i] == "Ġtail" else self.tokens[i] for i in ids)


@pytest.fixture
def byte_level_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    vocabulary = {token: i for i, token in enumerate([
        "ĠFrance", "ĠBelgium", "Ċ", "ĉ", "Ġ.", "cafÃ©", "ĠĠ", "Ā", "<special>",
    ])}
    backend = Tokenizer(models.BPE(vocabulary, merges=[]))
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<special>",
                                  clean_up_tokenization_spaces=True)


@pytest.mark.parametrize("token_id,expected", [
    (0, " France"), (1, " Belgium"), (2, r"\n"), (3, r"\t"), (4, " ."),
    (5, "café"), (6, "  "), (7, r"\x00"), (8, "<special>"),
])
def test_byte_level_decoder_controls_display_text(byte_level_tokenizer, token_id, expected):
    assert token_label(byte_level_tokenizer, token_id) == expected
    assert unpack_label(pack_labels([expected], 1)[0]) == expected


def test_token_labels_support_ids_only_decoders_without_vocabulary_lookup():
    tokenizer = SimpleNamespace(decode=lambda ids: " decoded " if ids == [17] else "wrong ID")
    assert token_label(tokenizer, 17) == " decoded "


def test_token_labels_support_special_token_option_without_cleanup_option():
    def decode(ids, skip_special_tokens=True):
        assert ids == [17]
        return "" if skip_special_tokens else "<special>"

    assert token_label(SimpleNamespace(decode=decode), 17) == "<special>"


def test_token_labels_pass_both_options_to_keyword_decoder():
    def decode(ids, **kwargs):
        assert ids == [17]
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return " \n "

    assert token_label(SimpleNamespace(decode=decode), 17) == r" \n "


@pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
def test_token_labels_propagate_decoder_errors_without_retry(error):
    calls = []
    def decode(ids, **kwargs):
        calls.append(ids)
        raise error("decoder failure")

    tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda i: "raw", decode=decode)
    with pytest.raises(error, match="decoder failure"):
        token_label(tokenizer, 17)
    assert calls == [[17]]


def test_token_labels_propagate_vocabulary_errors():
    def lookup(token_id):
        raise ValueError("vocabulary failure")

    with pytest.raises(ValueError, match="vocabulary failure"):
        token_label(SimpleNamespace(convert_ids_to_tokens=lookup), 17)


@pytest.mark.parametrize("token_id,expected", [
    (0, r"\x00"), (1, ""), (2, "None"), (65, "A"), (256, "<bos>"),
    (500, " tail"), (66, "[unmapped token 66]"), (511, "[unmapped token 511]"),
])
def test_sparse_token_labels_decode_text_and_identify_absent_rows(token_id, expected):
    tokenizer = SparseTokenizer()
    index = SimpleNamespace(tokenizer=tokenizer)
    label = FeatureIndex._token(index, token_id)
    assert label == expected
    assert protocol.text(label)
    assert unpack_label(pack_labels([label], 1)[0]) == expected


def test_sparse_token_labels_keep_packed_utf8_limit():
    tokenizer = SparseTokenizer()
    tokenizer.tokens[500] = "é" * 40 + "\0\ud800"
    label = FeatureIndex._token(SimpleNamespace(tokenizer=tokenizer), 500)
    assert label == "é" * 40 + r"\x00" + "?"
    assert protocol.text(label)
    assert unpack_label(pack_labels([label], 1)[0]) == "é" * 31


def test_missing_token_text_remains_distinct_in_content_identity():
    tokenizer = SparseTokenizer()
    original = _tokenizer_hash(tokenizer, 512)
    tokenizer.tokens[511] = "[unmapped token 511]"
    assert FeatureIndex._token(SimpleNamespace(tokenizer=tokenizer), 511) == tokenizer.tokens[511]
    assert _tokenizer_hash(tokenizer, 512) != original
    tokenizer.tokens[511] = "None"
    assert _tokenizer_hash(tokenizer, 512) != original
    del tokenizer.tokens[511]
    assert _tokenizer_hash(tokenizer, 512) == original


@pytest.mark.parametrize("tokenizer,expected", [
    (TinyTokenizer, "bb16a6bab25d041fa2738a8f106be27372e2d1a3e52472bf4bb9bf6a18f37f32"),
    (SparseTokenizer, "24fd05bebf61119419a0067190c0ae2d8c6c1df1c7e400665154aa064b2a907e"),
])
def test_current_tokenizer_identities_ignore_runtime_display_decoding(tokenizer, expected):
    tokenizer = tokenizer()
    assert _tokenizer_hash(tokenizer, 512) == expected
    vocabulary = [tokenizer.convert_ids_to_tokens(i) for i in range(512)]
    tokenizer.convert_ids_to_tokens = vocabulary.__getitem__
    tokenizer.decode = lambda ids: "different display"
    assert token_label(tokenizer, 65) == "different display"
    assert _tokenizer_hash(tokenizer, 512) == expected


def test_tokenizer_backend_remains_part_of_identity(byte_level_tokenizer):
    from tokenizers import decoders

    before = _tokenizer_hash(byte_level_tokenizer, len(byte_level_tokenizer))
    byte_level_tokenizer.backend_tokenizer.decoder = decoders.WordPiece()
    assert _tokenizer_hash(byte_level_tokenizer, len(byte_level_tokenizer)) != before


def test_padded_rows_keep_exact_scores_ids_and_example_labels(tmp_path, monkeypatch):
    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2)
    tokenizer = SparseTokenizer()
    scores = torch.zeros(512, dtype=torch.float64, device="cpu")
    scores[66] = scores[500] = 2.0
    scores[511] = torch.nextafter(scores[500], torch.tensor(3.0, dtype=torch.float64))
    with torch.no_grad():
        model.layers[0].mlp.W_enc[0].zero_()
        model.layers[0].mlp.b_enc[0] = 100.0
        model.layers[0].mlp.W_dec[0].zero_()
        model.layers[0].mlp.W_dec[0, 0] = 1.0
        model.lm_head.weight.zero_()
        model.lm_head.weight[:, 0] = scores
    write_fixture(tmp_path, model)
    dictionary = exact_dictionary(model)
    source_ids = [66, 500, 511]
    build_index(tmp_path, [("sparse", source_ids)], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"})
    contents = {file.name: file.read_bytes() for file in (tmp_path / "index").iterdir()}
    published = {}
    monkeypatch.setattr("caliper.interp.publish", lambda name, tensor: published.update({name: tensor}))
    with FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        assert index.metadata["tokenizer_hash"] == "24fd05bebf61119419a0067190c0ae2d8c6c1df1c7e400665154aa064b2a907e"
        service = InterpretationService(model=model, dictionary=dictionary, index=index)
        try:
            shown = execute(service, "feature.show", {"layer": 0, "feature": 0})
            assert shown["promoted_tokens"] == [
                {"token_id": 511, "token": "[unmapped token 511]", "score": scores[511].item()},
                {"token_id": 66, "token": "[unmapped token 66]", "score": 2.0},
                {"token_id": 500, "token": " tail", "score": 2.0},
            ]
            assert shown["example_tokens"] == [
                {"token_id": 66, "token": "[unmapped token 66]", "count": 3},
                {"token_id": 500, "token": " tail", "count": 3},
                {"token_id": 511, "token": "[unmapped token 511]", "count": 3},
            ]
            assert shown["label"] == (
                "Output: [unmapped token 511] | [unmapped token 66] |  tail"
                " / Examples: [unmapped token 66] |  tail | [unmapped token 511]"
            )
            result = execute(service, "feature.examples", {"layer": 0, "feature": 0})
            assert len(result["examples"]) == 3
            labels = ["[unmapped token 66]", " tail", "[unmapped token 511]"]
            for row in result["examples"]:
                assert [i for i in row["token_ids"] if i is not None] == source_ids
                assert [t for t in row["tokens"] if t is not None] == labels
                assert sum(row["valid"]) == 3
            assert published["examples.tokens"][0, :3].tolist() == source_ids
            assert [unpack_label(row) for row in published["examples.label_text"][:3]] == labels
        finally:
            service.close()
    assert {file.name: file.read_bytes() for file in (tmp_path / "index").iterdir()} == contents


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


def test_control_character_labels_remain_visible_after_repacking():
    value = " \n\r\t\0 "
    expected = r" \n\r\t\x00 "
    assert display_text(value) == expected
    assert display_text(expected) == expected
    assert unpack_label(pack_labels([value], 1)[0]) == expected
    assert unpack_label(pack_labels([display_text(value)], 1)[0]) == expected


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


def test_trace_axes_nodes_and_bars_use_decoded_text(byte_level_tokenizer, monkeypatch):
    source_ids = list(range(10)) + [11]
    expected = [" France", " Belgium", r"\n", r"\t", " .", "café", "  ", r"\x00", "<special>", ""]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "display fixture"
            return {"input_ids": torch.tensor([source_ids])}

        def convert_ids_to_tokens(self, token_id):
            return "<empty>" if token_id == 9 else byte_level_tokenizer.convert_ids_to_tokens(token_id)

        def decode(self, ids, **kwargs):
            return "" if ids == [9] else byte_level_tokenizer.decode(ids, **kwargs)

    model = TinyLM(layers=1, width=8, heads=2, latents=8, k=2, vocab=12)
    with torch.no_grad():
        model.lm_head.weight.mul_(.001)
        model.lm_head.bias = torch.nn.Parameter(torch.tensor([0.] * 10 + [-20.] * 2, dtype=torch.float64))
    tokenizer = Tokenizer()
    dictionary = exact_dictionary(model)
    published = {}
    monkeypatch.setattr("caliper.interp.publish", lambda name, tensor: published.update({name: tensor}))
    graph = Tracer(model, tokenizer, dictionary).trace("display fixture", max_targets=8)
    labels = expected + ["[unmapped token 11]"]
    assert [token["token_id"] for token in graph.tokens] == source_ids
    assert [token["text"] for token in graph.tokens] == labels
    assert [unpack_label(row) for row in published["trace.label_text"][:len(labels)]] == labels
    assert any(node["kind"] == "token" for node in graph.nodes)
    for node in graph.nodes:
        label = node["label"]
        if node["kind"] == "token":
            assert label == labels[node["position"]]
        elif node["kind"] == "logit":
            assert label == expected[node["token_id"]]
        assert unpack_label(published["trace.label_text"][TOKEN_CAP + node["index"]]) == label
    assert sorted(published["trace.logit_tokens"].tolist()) == list(range(10))
    bar_labels = [expected[token] for token in published["trace.logit_tokens"].tolist()]
    assert [unpack_label(row) for row in published["trace.logit_label_text"]] == bar_labels
    probabilities = torch.tensor([node["probability"] for node in graph.logits], dtype=torch.float32)
    assert torch.equal(published["trace.probs"], probabilities.expand(2, 10))
    assert not any("Ġ" in text for text in labels + bar_labels + [node["label"] for node in graph.nodes])

    def raw_label(tokenizer, token_id):
        value = tokenizer.convert_ids_to_tokens(token_id)
        return display_text(value) if value is not None else f"[unmapped token {token_id}]"

    monkeypatch.setattr("caliper.interp._trace.token_label", raw_label)
    raw_graph = Tracer(model, tokenizer, dictionary).trace("display fixture", max_targets=8)
    # Graph identity includes display text. Structural values remain exact.
    assert raw_graph.trace_id != graph.trace_id
    assert raw_graph.completeness == graph.completeness
    assert raw_graph.edges == graph.edges
    for name, tensor in graph.tensors.items():
        assert torch.equal(raw_graph.tensors[name], tensor)


def test_graph_panel_keeps_empty_labels_and_falls_back_only_for_absent_text():
    nodes = [{"id": f"token:{i}:{i}", "kind": "token", "layer": -1, "position": i,
              "token_id": i, "activation": 1., "influence": 0., "effect": 0.}
             for i in range(3)]
    nodes[0]["label"] = ""
    nodes[1]["label"] = None
    graph = Graph.build(nodes=nodes, edges=[], tokens=[{"token_id": i, "text": ""} for i in range(3)], layers=1,
                        completeness={}, omitted_influence=0., error_share=0., feature_share=0.,
                        trace_id="empty-label", device=torch.device("cpu"))
    labels = graph._panel_tensors()["trace.label_text"]
    assert unpack_label(labels[0]) == ""
    assert [unpack_label(row) for row in labels[TOKEN_CAP:TOKEN_CAP + 3]] == ["", "token:1:1", "token:2:2"]
