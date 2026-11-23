"""Exact CPU evidence for feature tables and document windows."""
from collections import Counter

import numpy as np
import pytest
import torch

from caliper.interp import capture, load_dictionary
from caliper.interp._index import FeatureIndex, _TopTable, build_index
from .fixture import TinyTokenizer
from .test_dictionary import write_fixture


@pytest.fixture
def indexed(tmp_path):
    model, _ = write_fixture(tmp_path)
    tokenizer = TinyTokenizer()
    documents = [("first", [65, 66, 65, 67] * 12), ("second", [70, 71, 72]), ("last", [65])]
    dictionary = load_dictionary(tmp_path, model=model)
    build_index(tmp_path, documents, model=model, tokenizer=tokenizer, dictionary=dictionary,
                provenance={"partition": "fixture-examples"}, context_length=17)
    return FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary), documents


def test_streamed_table_equals_brute_force(indexed):
    index, documents = indexed
    activations = [[] for _ in index.dictionary.layers]
    with torch.inference_mode():
        for _, tokens in documents:
            for start in range(0, len(tokens), 17):
                with capture(index.model, index.dictionary.metadata["read_points"]) as saved:
                    index.model(torch.tensor([tokens[start:start + 17]]))
                    for layer, point in enumerate(index.dictionary.metadata["read_points"]):
                        activations[layer].append(index.dictionary.encode(layer, saved[point])[0].numpy())
    for layer, chunks in enumerate(activations):
        values = np.concatenate(chunks)
        for feature in range(values.shape[1]):
            positive = [i for i in range(len(values)) if values[i, feature] > 0]
            ranked = sorted(positive, key=lambda i: (-values[i, feature], i))[:16]
            expected_values = [values[i, feature] for i in ranked] + [0] * (16 - len(ranked))
            expected_offsets = ranked + [-1] * (16 - len(ranked))
            np.testing.assert_array_equal(index.table[layer, feature]["activation"], expected_values)
            np.testing.assert_array_equal(index.table[layer, feature]["offset"], expected_offsets)
            assert index.counts[layer, feature] == len(positive)
    assert isinstance(index.table, np.memmap)
    assert isinstance(index.tokens, np.memmap)
    assert index.tokens.dtype == np.dtype("uint32")


def test_boundary_ties_across_batches(tmp_path):
    table = _TopTable(tmp_path, 1, 3)
    values = torch.tensor([[2., 0., -1.]] * 37, dtype=torch.float64)
    values[22, 0] = 4
    values[29, 0] = 3
    values[25, 1] = 1
    table.update(0, values[:20], 0)
    table.update(0, values[20:], 20)
    np.testing.assert_array_equal(table.table[0, 0]["offset"], [22, 29] + list(range(14)))
    assert table.counts.tolist() == [[37, 1, 0]]
    assert table.table[0, 1, 0]["offset"] == 25
    assert np.all(table.table[0, 2]["offset"] == -1)
    assert np.all(table.table[0, 2]["activation"] == 0)


def test_document_windows_and_real_token_activations(indexed):
    index, documents = indexed
    short_offset = len(documents[0][1])
    candidates = np.argwhere(index.table["offset"] == short_offset)
    layer, feature, _ = map(int, candidates[0])
    rows = index.examples(layer, feature)
    assert len(rows) <= 16
    assert [r["activation"] for r in rows] == sorted((r["activation"] for r in rows), reverse=True)
    source = dict(documents)
    for row in rows:
        assert all(len(row[k]) == 33 for k in ("token_ids", "tokens", "activations", "valid", "positions"))
        assert row["positions"][16] == row["position"]
        tokens = source[row["document_id"]]
        for slot, position in enumerate(row["positions"]):
            if position is None:
                assert row["token_ids"][slot] is None and row["activations"][slot] is None
            else:
                assert row["token_ids"][slot] == tokens[position]
    point = index.dictionary.metadata["read_points"][layer]
    # Independent batch reconstruction preserves the exact CPU kernel shape.
    ids = torch.full((len(rows), 33), 258)
    for i, row in enumerate(rows):
        valid = [v for v in row["token_ids"] if v is not None]
        ids[i, :len(valid)] = torch.tensor(valid)
    with torch.inference_mode(), capture(index.model, [point]) as saved:
        index.model(ids)
        expected = index.dictionary.encode(layer, saved[point])[..., feature]
    for row, values in zip(rows, expected):
        actual = [v for v in row["activations"] if v is not None]
        assert actual == values[:len(actual)].tolist()
    short = next(r for r in rows if r["document_id"] == "second" and r["position"] == 0)
    assert short["token_ids"][:16] == [None] * 16
    assert short["token_ids"][16:19] == [70, 71, 72]
    assert short["token_ids"][19:] == [None] * 14


def test_label_values_and_query_cache(indexed):
    index, _ = indexed
    feature = int(np.argmax(index.counts[0]))
    info = index.show(0, feature)
    weight = index.model.lm_head.weight.detach()
    direction = index.dictionary.layers[0].W_dec[feature]
    scores = (weight @ direction).tolist()
    promoted = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
    assert [t["token_id"] for t in info["promoted_tokens"]] == promoted
    assert [t["score"] for t in info["promoted_tokens"]] == [scores[i] for i in promoted]
    calls = []
    handle = index.model.register_forward_pre_hook(lambda module, args: calls.append(True))
    rows = index.examples(0, feature, limit=1)
    assert len(calls) == 1
    all_rows = index.examples(0, feature)
    assert len(calls) == 1 and len(all_rows) == min(16, info["count"])
    frequencies = Counter(i for row in all_rows for i in row["token_ids"] if i is not None)
    expected = sorted(frequencies, key=lambda i: (-frequencies[i], i))[:3]
    assert [t["token_id"] for t in info["example_tokens"]] == expected
    assert info["label"] == "Output: " + " | ".join(index._token(i) for i in promoted) + " / Examples: " + " | ".join(index._token(i) for i in expected)
    rows[0]["token_ids"][16] = -1
    info["maximum"] = -1
    assert index.examples(0, feature, 1)[0]["token_ids"][16] >= 0
    assert index.show(0, feature)["maximum"] >= 0
    index.clear_cache()
    index.examples(0, feature)
    assert len(calls) == 2
    handle.remove()


@pytest.mark.parametrize("layer,feature,limit", [(-1, 0, 16), (4, 0, 16), (0, -1, 16),
                                               (0, 256, 16), (0, 0, 17), (0, 0, -1),
                                               (True, 0, 16), (0, 1.5, 16)])
def test_invalid_queries(indexed, layer, feature, limit):
    index, _ = indexed
    with pytest.raises(ValueError):
        index.examples(layer, feature, limit)


def test_empty_index_and_zero_features(tmp_path):
    model, _ = write_fixture(tmp_path)
    build_index(tmp_path, [], model=model, tokenizer=TinyTokenizer(), provenance={"partition": "empty"})
    index = FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer())
    assert index.examples(0, 0) == []
    assert index.examples(0, 0, 0) == []
    info = index.show(0, 0)
    assert info["count"] == info["maximum"] == 0
    assert info["example_tokens"] == [] and info["label"].endswith("Examples: none")
    assert len(info["promoted_tokens"]) == 3
    assert len(index.documents) == len(index.tokens) == 0


def test_one_token_document_and_budget_edge(tmp_path):
    model, _ = write_fixture(tmp_path)
    build_index(tmp_path, [("one", [65]), ("cut", [66, 67, 68])], model=model,
                tokenizer=TinyTokenizer(), provenance={"partition": "fixture"}, max_tokens=3)
    index = FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer())
    assert index.tokens.tolist() == [65, 66, 67]
    assert index.documents["length"].tolist() == [1, 2]
    assert index.documents["truncated"].tolist() == [False, True]
    for offset in (0, 2):
        layer, feature, _ = map(int, np.argwhere(index.table["offset"] == offset)[0])
        rows = index.examples(layer, feature)
        row = next(r for r in rows if r["document_id"] == ("one" if offset == 0 else "cut"))
        assert sum(row["valid"]) == (1 if offset == 0 else 2)
        assert row["document_truncated"] == (offset != 0)
        assert len(rows) <= 3


def test_true_lru_eviction(indexed, monkeypatch):
    index, _ = indexed
    calls = []
    original = index._windows
    def windows(layer, feature):
        calls.append((layer, feature))
        return original(layer, feature) if feature == 0 else []
    monkeypatch.setattr(index, "_windows", windows)
    index.examples(0, 0)
    for feature in range(1, 256):
        index.examples(0, feature)
    assert len(index._cache) == 256
    index.examples(0, 0)  # A hit protects the oldest entry.
    index.examples(1, 1)
    assert (0, 0) in index._cache and (0, 1) not in index._cache
    index.examples(0, 1)
    assert calls.count((0, 0)) == 1 and calls.count((0, 1)) == 2
    assert len(index._cache) == 256


@pytest.mark.parametrize("damage", ["missing_manifest", "partial_manifest", "wrong_hash", "version", "incomplete",
                                    "table", "counts", "tokens", "documents", "source", "weights", "tokenizer", "model"])
def test_incompatible_or_incomplete_index(indexed, damage):
    import json
    from caliper.interp._index import _json
    import hashlib
    index, _ = indexed
    manifest_path = index.root / "index" / "index.json"
    if damage == "missing_manifest":
        manifest_path.unlink()
    elif damage == "partial_manifest":
        manifest_path.write_text('{"complete":')
    elif damage in ("wrong_hash", "version", "incomplete"):
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("index_hash")
        if damage == "version":
            manifest["format_version"] = 99
        elif damage == "incomplete":
            manifest["complete"] = False
        manifest["index_hash"] = "bad" if damage == "wrong_hash" else hashlib.sha256(_json(manifest).encode()).hexdigest()
        manifest_path.write_text(_json(manifest))
    elif damage in ("table", "counts", "tokens", "documents"):
        name = damage + (".npy" if damage in ("table", "counts") else ".bin")
        with (index.root / "index" / name).open("r+b") as output:
            output.truncate(1)
    elif damage == "source":
        with (index.root / "dictionary.json").open("a") as output:
            output.write(" ")
    elif damage == "weights":
        with torch.no_grad():
            index.dictionary.layers[0].W_enc[0, 0] += 1
    elif damage == "tokenizer":
        index.tokenizer.eos_token_id = 0
    else:
        with torch.no_grad():
            index.model.lm_head.weight[-1, -1] += 1
    with pytest.raises(ValueError):
        FeatureIndex(index.root, model=index.model, tokenizer=index.tokenizer, dictionary=index.dictionary)


def test_promoted_chunk_ties_and_zero_direction(indexed):
    index, _ = indexed
    with torch.no_grad():
        index.dictionary.layers[0].W_dec[0].zero_()
    tokens = index._promoted(0, 0, chunk_size=37)
    assert [row["token_id"] for row in tokens] == [0, 1, 2]
    assert [row["score"] for row in tokens] == [0, 0, 0]


def test_token_store_preserves_32_bits(tmp_path):
    from .fixture import TinyLM
    model = TinyLM(layers=1, width=4, heads=1, latents=4, k=2, vocab=70_003)
    model, _ = write_fixture(tmp_path, model)
    class Tokenizer(TinyTokenizer):
        vocab_size = 70_003
    build_index(tmp_path, [("wide-vocab", [70_000, 70_001, 70_002])], model=model,
                tokenizer=Tokenizer(), provenance={"partition": "fixture"})
    index = FeatureIndex(tmp_path, model=model, tokenizer=Tokenizer())
    assert index.tokens.tolist() == [70_000, 70_001, 70_002]
    assert (tmp_path / "index" / "tokens.bin").stat().st_size == 12


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_table_refusal(tmp_path, bad):
    table = _TopTable(tmp_path, 1, 1)
    with pytest.raises(ValueError, match="finite"):
        table.update(0, torch.tensor([[bad]]), 0)


def test_failed_build_has_no_completion_manifest(tmp_path):
    model, _ = write_fixture(tmp_path)
    def fail(stage):
        if stage == "model batch":
            raise RuntimeError("fixture stop")
    with pytest.raises(RuntimeError, match="fixture stop"):
        build_index(tmp_path, [("doc", [65, 66])], model=model, tokenizer=TinyTokenizer(),
                    provenance={"partition": "fixture"}, memory_guard=fail)
    assert not (tmp_path / "index" / "index.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer())
    assert all(not block.mlp._forward_pre_hooks for block in model.layers)


def test_existing_index_is_not_overwritten(indexed):
    index, documents = indexed
    before = (index.root / "index" / "index.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_index(index.root, documents, model=index.model, tokenizer=index.tokenizer,
                    provenance={"partition": "fixture"})
    assert (index.root / "index" / "index.json").read_bytes() == before


def test_t2_stream_restores_documents_and_skips_duplicates():
    from tools.interp.build_index import stream_documents
    from tools.interp.text_stream import TextStream, document_identity
    source = [{"text": f"Example document {i} describes a blue square and a green circle."} for i in range(100)]
    selected = [row for row in source if document_identity(row["text"])[0] == "index"]
    assert len(selected) >= 2
    source = [selected[0], selected[0], *source, selected[-1]]
    expected = {}
    tokenizer = TinyTokenizer()
    for row in source:
        partition, document_id = document_identity(row["text"])
        if partition == "index":
            expected[str(document_id)] = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
    stream = TextStream(source, tokenizer, "index")
    assert dict(stream_documents(stream, batch_tokens=7)) == expected
    assert stream.emitted > sum(map(len, expected.values()))
    with pytest.raises(ValueError, match="index partition"):
        list(stream_documents(TextStream(source, tokenizer, "quality")))
    stream = TextStream(selected, tokenizer, "index")
    stream.next_tokens(1)
    with pytest.raises(ValueError, match="contiguous"):
        list(stream_documents(stream, batch_tokens=7))


@pytest.mark.parametrize("option,value", [("--tokens", "2000001"), ("--tokens", "0"),
                                         ("--memory-gb", "12.1"), ("--memory-gb", "nan"),
                                         ("--memory-gb", "0"), ("--partition", "train"),
                                         ("--partition", "quality"), ("--context", "257")])
def test_cli_caps(option, value):
    from tools.interp.build_index import parser
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["--dictionary", "dictionary", option, value])
    assert error.value.code == 2


def _t2_source_manifest(folder, model):
    import json
    from tools.interp.text_stream import DATASET_CONFIG, DATASET_ID, DATASET_REVISION, partition_metadata
    _, metadata = write_fixture(folder, model)
    metadata["training"].update(dataset_id=DATASET_ID, dataset_config=DATASET_CONFIG,
                                 dataset_revision=DATASET_REVISION, split="train", partitions=partition_metadata())
    (folder / "dictionary.json").write_text(json.dumps(metadata))
    return metadata


def test_cli_fixture_build_and_query(tmp_path, monkeypatch, capsys):
    import json
    from tools.interp import build_index as command
    from .fixture import TinyLM
    model = TinyLM()
    _t2_source_manifest(tmp_path, model)
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({"text": f"Example document {i} describes a blue square and a green circle."}) for i in range(100)))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: (model, TinyTokenizer()))
    stages = []
    class Memory:
        peak = 0
        def __init__(self, gb, device):
            assert gb == 12 and device == "cpu"
        def check(self, stage, **kwargs):
            stages.append(stage)
    monkeypatch.setattr(command, "MemoryGuard", Memory)
    args = ["--dictionary", str(tmp_path), "--device", "cpu"]
    assert command.main([*args, "--text-jsonl", str(source), "--tokens", "73", "--context", "17"]) == 0
    index = FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer())
    assert index.metadata["tokens"] == 73
    assert index.metadata["provenance"]["partition"] == "index"
    assert index.metadata["provenance"]["source_kind"] == "staged_jsonl"
    assert "before batch" in stages and "model batch" in stages and "table" in stages
    assert command.main([*args, "--query", "0", "0", "--limit", "1"]) == 0
    response = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert response["show"]["feature"] == 0
    assert len(response["examples"]) <= 1
    with pytest.raises(ValueError, match="already has an index"):
        command.main(args)


def test_training_and_quality_partition_overlap_refusal(tmp_path, monkeypatch):
    import json
    from tools.interp import build_index as command
    from .fixture import TinyLM
    metadata = _t2_source_manifest(tmp_path, TinyLM())
    metadata["training"]["partitions"]["index"] = [9400, 10000]
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("model load before partition validation"))
    with pytest.raises(ValueError, match="partitions differ"):
        command.main(["--dictionary", str(tmp_path), "--device", "cpu"])


def test_cli_clean_memory_stop(tmp_path, monkeypatch):
    from tools.interp import build_index as command
    from .fixture import TinyLM
    model = TinyLM()
    _t2_source_manifest(tmp_path, model)
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: (model, TinyTokenizer()))
    monkeypatch.setattr(command, "fineweb_documents", lambda **kwargs: [{"text": "fixture"}])
    class Memory:
        def __init__(self, *args):
            pass
        def check(self, stage, **kwargs):
            if stage == "dictionary":
                raise command.CleanStop("fixture memory limit")
    monkeypatch.setattr(command, "MemoryGuard", Memory)
    assert command.main(["--dictionary", str(tmp_path), "--device", "cpu"]) == 2
    assert not (tmp_path / "index" / "index.json").exists()
