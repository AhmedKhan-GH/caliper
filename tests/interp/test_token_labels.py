"""NUL-free display labels with unchanged token IDs and model inputs."""
from types import SimpleNamespace

import pytest
import torch

from caliper import _interpret_protocol as protocol
from caliper.interp import Tracer
from caliper.interp._backend import make_tiny_backend
from caliper.interp._graph import Graph, TOKEN_CAP
from caliper.interp._index import FeatureIndex, _tokenizer_hash, build_index
from caliper.interp._labels import display_text, pack_labels, token_label
from caliper.interp._service import InterpretationService


class SparseTokenizer:
    """Vocabulary holes and unused output rows coexist with high valid IDs."""

    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258

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


def unpack_label(row):
    raw = b"".join(int(word).to_bytes(4, "little") for word in row)
    return raw.split(b"\0", 1)[0].decode("utf-8")


def execute(service, method, params):
    replies = []
    assert service.submit(
        {"version": 1, "method": method, "params": params},
        lambda *args: replies.append(args),
    ) == ("queued", None)
    assert service.step(1) == 1
    assert replies[0][0] == "ok", replies
    assert protocol.bounded(replies[0][1])
    return replies[0][1]


@pytest.fixture
def byte_level_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    vocabulary = {token: i for i, token in enumerate([
        "ĠFrance", "ĠBelgium", "Ċ", "ĉ", "Ġ.", "cafÃ©", "ĠĠ", "Ā", "<special>",
    ])}
    backend = Tokenizer(models.BPE(vocabulary, merges=[]))
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<special>", clean_up_tokenization_spaces=True
    )


@pytest.mark.parametrize("token_id,expected", [
    (0, " France"), (1, " Belgium"), (2, r"\n"), (3, r"\t"), (4, " ."),
    (5, "café"), (6, "  "), (7, r"\x00"), (8, "<special>"),
])
def test_byte_level_decoder_controls_display_text(byte_level_tokenizer, token_id, expected):
    assert token_label(byte_level_tokenizer, token_id) == expected
    assert unpack_label(pack_labels([expected], 1)[0]) == expected


def test_token_labels_options_and_error_propagation():
    assert token_label(SimpleNamespace(decode=lambda ids: " decoded "), 17) == " decoded "
    assert token_label(SimpleNamespace(decode=lambda ids: ""), 17) == "#17"

    def decode_kw(ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return " \n "

    assert token_label(SimpleNamespace(decode=decode_kw), 17) == r" \n "

    def bad_decode(ids, **kwargs):
        raise RuntimeError("decoder failure")

    with pytest.raises(RuntimeError, match="decoder failure"):
        token_label(SimpleNamespace(convert_ids_to_tokens=lambda i: "raw", decode=bad_decode), 17)


@pytest.mark.parametrize("token_id,expected", [
    (0, r"\x00"), (1, "#1"), (2, "None"), (65, "A"), (256, "<bos>"),
    (500, " tail"), (66, "[unmapped token 66]"), (511, "[unmapped token 511]"),
])
def test_sparse_token_labels_decode_text_and_identify_absent_rows(token_id, expected):
    tokenizer = SparseTokenizer()
    label = token_label(tokenizer, token_id)
    assert label == expected
    assert protocol.text(label)
    assert unpack_label(pack_labels([label], 1)[0]) == expected


def test_tokenizer_identity_and_backend_hash(byte_level_tokenizer):
    from tokenizers import decoders

    tokenizer = SparseTokenizer()
    assert _tokenizer_hash(tokenizer, 512) == "24fd05bebf61119419a0067190c0ae2d8c6c1df1c7e400665154aa064b2a907e"
    before = _tokenizer_hash(byte_level_tokenizer, len(byte_level_tokenizer))
    byte_level_tokenizer.backend_tokenizer.decoder = decoders.WordPiece()
    assert _tokenizer_hash(byte_level_tokenizer, len(byte_level_tokenizer)) != before


def test_sparse_index_and_display_label_packing(tmp_path, monkeypatch):
    backend = make_tiny_backend(d_vocab=512)
    tokenizer = SparseTokenizer()
    scores = torch.zeros(512, dtype=torch.float32)
    scores[66] = scores[500] = 2.0
    scores[511] = 2.5
    with torch.no_grad():
        backend.dictionary.raw[0].W_enc[0].zero_()
        backend.dictionary.raw[0].b_enc[0] = 10.0
        backend.dictionary.raw[0].W_dec[0].zero_()
        backend.dictionary.raw[0].W_dec[0, 0] = 1.0
        backend.model.W_U.zero_()
        backend.model.W_U[0, :] = scores
    source_ids = [66, 500, 511]
    build_index(
        tmp_path, [("sparse", source_ids)], model=backend.model, tokenizer=tokenizer,
        dictionary=backend.dictionary, provenance={"partition": "fixture"},
    )
    published = {}
    monkeypatch.setattr("caliper.interp._publish._TENSORS", published)
    with FeatureIndex(tmp_path, model=backend.model, tokenizer=tokenizer, dictionary=backend.dictionary) as index:
        service = InterpretationService(model=backend.model, tokenizer=tokenizer, dictionary=backend.dictionary, index=index)
        try:
            shown = execute(service, "feature.show", {"layer": 0, "feature": 0})
            assert [t["token_id"] for t in shown["promoted_tokens"]] == [511, 66, 500]
            result = execute(service, "feature.examples", {"layer": 0, "feature": 0})
            assert len(result["examples"]) == 3
            assert published["examples.tokens"][0, :3].tolist() == source_ids
            assert [unpack_label(row) for row in published["examples.label_text"][:3]] == [
                "[unmapped token 66]", " tail", "[unmapped token 511]"
            ]
        finally:
            service.close()


def test_graph_panel_keeps_empty_labels_and_falls_back_only_for_absent_text():
    nodes = [
        {"id": f"token:{i}:{i}", "kind": "token", "layer": -1, "position": i,
         "token_id": i, "activation": 1.0, "influence": 0.0, "effect": 0.0}
        for i in range(3)
    ]
    nodes[0]["label"] = ""
    nodes[1]["label"] = None
    graph = Graph.build(
        nodes=nodes, edges=[], tokens=[{"token_id": i, "text": ""} for i in range(3)], layers=1,
        completeness={}, omitted_influence=0.0, error_share=0.0, feature_share=0.0,
        trace_id="empty-label", device=torch.device("cpu"),
    )
    labels = graph._panel_tensors()["trace.label_text"]
    assert unpack_label(labels[0]) == ""
    assert [unpack_label(row) for row in labels[TOKEN_CAP:TOKEN_CAP + 3]] == ["", "token:1:1", "token:2:2"]
