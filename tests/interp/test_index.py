"""Exact CPU evidence for feature tables and document windows."""
from collections import Counter
from types import SimpleNamespace

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
    with FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        yield index, documents


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


@pytest.mark.parametrize("stop", [False, True])
def test_empty_document_refuses_before_the_stream_advances(tmp_path, stop):
    from caliper.interp._memory import CleanStop
    model, _ = write_fixture(tmp_path)
    stages = []
    def documents():
        yield "empty", []
        pytest.fail("the builder consumed another empty document")
    def check(stage):
        stages.append(stage)
        if stop and stage == "document":
            raise CleanStop("fixture document stop")
    error = CleanStop if stop else ValueError
    message = "document stop" if stop else "must not be empty"
    with pytest.raises(error, match=message):
        build_index(tmp_path, documents(), model=model, tokenizer=TinyTokenizer(),
                    provenance={"partition": "fixture"}, max_tokens=1, memory_guard=check)
    assert stages[-1] == "document"
    assert not (tmp_path / "index" / "index.json").exists()


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
    index.close()
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


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 9, 4096])
def test_sparse_promoted_chunks_keep_exact_scores_and_ties(monkeypatch, chunk_size):
    from caliper.interp import _index
    from caliper.interp._labels import token_label

    scores = torch.tensor([0., 2., 0., 2., 0., 0., 2., 0., 2.], dtype=torch.float64,
                          device="cpu")
    scores[8] = torch.nextafter(scores[8], torch.tensor(3., dtype=torch.float64))
    output = SimpleNamespace(weight=scores[:, None])
    dictionary = SimpleNamespace(layers=[SimpleNamespace(W_dec=torch.ones(1, 1, dtype=torch.float64))])
    vocabulary = {3: "three", 6: "six"}
    tokenizer = SimpleNamespace(convert_ids_to_tokens=vocabulary.get,
                                decode=lambda ids: "".join(vocabulary[i] for i in ids))
    index = SimpleNamespace(model=SimpleNamespace(get_output_embeddings=lambda: output),
                            dictionary=dictionary, _token=lambda i: token_label(tokenizer, i))
    original = _index._stable_topk
    shapes = []

    def bounded_topk(values, offsets, k):
        shapes.append(values.shape[1])
        assert values.shape[1] <= (chunk_size if len(shapes) % 2 else 6)
        return original(values, offsets, k)

    monkeypatch.setattr(_index, "_stable_topk", bounded_topk)
    rows = FeatureIndex._promoted(index, 0, 0, chunk_size=chunk_size)
    assert rows == [
        {"token_id": 8, "token": "[unmapped token 8]", "score": scores[8].item()},
        {"token_id": 1, "token": "[unmapped token 1]", "score": 2.0},
        {"token_id": 3, "token": "three", "score": 2.0},
    ]
    assert shapes[::2] == [min(chunk_size, 9 - start) for start in range(0, 9, chunk_size)]


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
                                 dataset_revision=DATASET_REVISION, split="train", partitions=partition_metadata(),
                                 source_id=f"{DATASET_ID}/{DATASET_CONFIG}@{DATASET_REVISION}:train")
    (folder / "dictionary.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize("source_kind", ["jsonl", "dataset"])
def test_cli_fixture_build_and_query(tmp_path, monkeypatch, capsys, source_kind):
    import hashlib
    import json
    from tools.interp import build_index as command
    from tools.interp.text_stream import document_identity
    from .fixture import TinyLM
    model = TinyLM()
    metadata = _t2_source_manifest(tmp_path, model)
    records = [{"text": f"Example document {i} describes a blue square and a green circle."} for i in range(300)]
    source_args = []
    if source_kind == "jsonl":
        source = tmp_path / "source.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in records))
        metadata["training"]["source_id"] = "jsonl:sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        source_args = ["--text-jsonl", str(source)]
    else:
        monkeypatch.setattr(command, "fineweb_documents", lambda **kwargs: records)
    source_id = metadata["training"]["source_id"]
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: (model, TinyTokenizer()))
    stages = []
    class Memory:
        peak = 0
        def __init__(self, gb, device):
            assert gb == 12 and device == "cpu"
            self.limit = int(gb * 1_000_000_000)
            self.device = torch.device(device)
        def check(self, stage, **kwargs):
            stages.append(stage)
    monkeypatch.setattr(command, "MemoryGuard", Memory)
    args = ["--dictionary", str(tmp_path), "--device", "cpu"]
    assert command.main([*args, *source_args, "--tokens", "73", "--context", "17"]) == 0
    index = FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer())
    assert index.metadata["tokens"] == 73
    batches = sum((int(length) + 16) // 17 for length in index.documents["length"])
    assert stages.count("before batch") == stages.count("model batch") == batches
    assert index.metadata["provenance"]["partition"] == "index"
    assert index.metadata["provenance"]["source_kind"] == ("staged_jsonl" if source_kind == "jsonl" else "pinned_dataset")
    assert index.metadata["provenance"]["source_id"] == source_id
    groups = {name: set() for name in ("train", "quality", "index")}
    for row in records:
        partition, identity = document_identity(row["text"])
        groups[partition].add(str(identity))
    assert all(groups.values())
    indexed_ids = {bytes(identity).decode("utf-8") for identity in index.documents["id"]}
    assert indexed_ids and indexed_ids <= groups["index"]
    assert not indexed_ids & (groups["train"] | groups["quality"])
    index.close()
    if source_kind == "jsonl":
        source.unlink()
    monkeypatch.setattr(command, "fineweb_documents", lambda **kwargs: pytest.fail("query reopened the source"))
    assert "before batch" in stages and "model batch" in stages and "table" in stages
    capsys.readouterr()
    assert command.main([*args, "--query", "0", "0", "--limit", "1"]) == 0
    # The query emits one JSON result. Library diagnostics use stderr.
    response = json.loads(capsys.readouterr().out)
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
        device = torch.device("cpu")
        limit = 12_000_000_000
        def __init__(self, *args):
            pass
        def check(self, stage, **kwargs):
            if stage == "dictionary":
                raise command.CleanStop("fixture memory limit")
    monkeypatch.setattr(command, "MemoryGuard", Memory)
    assert command.main(["--dictionary", str(tmp_path), "--device", "cpu"]) == 2
    assert not (tmp_path / "index" / "index.json").exists()


def test_near_ties_retain_float64_values(tmp_path):
    table = _TopTable(tmp_path, 1, 1)
    values = torch.ones(31, 1, dtype=torch.float64)
    values[25, 0] = np.nextafter(1.0, 2.0)
    values[26, 0] = np.nextafter(1.0, 0.0)
    table.update(0, values[:12], 0)
    table.update(0, values[12:], 12)
    expected = sorted(range(31), key=lambda i: (-values[i, 0].item(), i))[:16]
    assert table.table[0, 0]["offset"].tolist() == expected
    assert table.table[0, 0]["activation"].tolist() == values[expected, 0].tolist()


def test_zero_feature_skips_forward_and_short_feature_keeps_its_count(indexed):
    index, _ = indexed
    layer, feature = map(int, np.argwhere(index.counts == 0)[0])
    handle = index.model.register_forward_pre_hook(lambda *_: pytest.fail("zero feature ran the model"))
    try:
        assert index.examples(layer, feature) == []
        assert index.show(layer, feature)["maximum"] == 0
    finally:
        handle.remove()
    layer, feature = map(int, np.argwhere((index.counts > 0) & (index.counts < 16))[0])
    assert len(index.examples(layer, feature)) == int(index.counts[layer, feature])


def test_topk_identity_and_eval_mode_refusal(indexed):
    index, _ = indexed
    index.dictionary.layers[0].k += 1
    with pytest.raises(ValueError, match="TopK mismatch"):
        FeatureIndex(index.root, model=index.model, tokenizer=index.tokenizer, dictionary=index.dictionary)
    index.model.train()
    with pytest.raises(ValueError, match="eval mode"):
        index.show(0, 0)


@pytest.mark.parametrize("data", [None, [], "invalid", 7])
def test_malformed_manifest_root(indexed, data):
    import json
    index, _ = indexed
    (index.root / "index" / "index.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="malformed"):
        FeatureIndex(index.root, model=index.model, tokenizer=index.tokenizer, dictionary=index.dictionary)


def test_last_resource_check_precedes_completion(tmp_path):
    model, _ = write_fixture(tmp_path)
    def check(stage):
        if stage == "before completion":
            raise RuntimeError("last resource check")
    with pytest.raises(RuntimeError, match="last resource check"):
        build_index(tmp_path, [("doc", [65])], model=model, tokenizer=TinyTokenizer(),
                    provenance={"partition": "fixture"}, memory_guard=check)
    assert not (tmp_path / "index" / "index.json").exists()


def test_budget_does_not_consume_another_document(tmp_path):
    model, _ = write_fixture(tmp_path)
    def documents():
        yield "first", [65, 66, 67]
        pytest.fail("source advanced after token budget")
    manifest = build_index(tmp_path, documents(), model=model, tokenizer=TinyTokenizer(),
                           provenance={"partition": "fixture"}, max_tokens=2)
    assert manifest["tokens"] == 2


def test_cli_refuses_invalid_query_before_model_load(tmp_path, monkeypatch):
    from tools.interp import build_index as command
    from .fixture import TinyLM
    _t2_source_manifest(tmp_path, TinyLM())
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("invalid query loaded a model"))
    with pytest.raises(ValueError, match="outside the dictionary"):
        command.main(["--dictionary", str(tmp_path), "--query", "4", "0"])


def test_nonfinite_window_refuses_cache_entry(indexed, monkeypatch):
    index, _ = indexed
    layer, feature = map(int, np.argwhere(index.counts > 0)[0])
    encoder = index.dictionary.layers[layer]
    encode = encoder.encode
    monkeypatch.setattr(encoder, "encode", lambda x: encode(x) * float("nan"))
    with pytest.raises(ValueError, match="finite"):
        index.examples(layer, feature)
    assert "examples" not in index._cache[layer, feature]
    assert all(not block.mlp._forward_pre_hooks for block in index.model.layers)


def test_index_close_releases_maps_and_refuses_queries(indexed):
    index, _ = indexed
    maps = [getattr(index, name)._mmap for name in ("table", "counts", "tokens", "documents")]
    index.close()
    index.close()
    assert all(mapping.closed for mapping in maps)
    with pytest.raises(ValueError, match="closed"):
        index.examples(0, 0)
    with pytest.raises(ValueError, match="closed"):
        with index:
            pass
    # This unlink also succeeds on Windows because no mapping remains open.
    (index.root / "index" / "table.npy").unlink()


def test_index_partial_load_closes_earlier_maps(indexed, monkeypatch):
    index, _ = indexed
    index.close()
    actual_load = np.load
    maps = []
    def load(path, **kwargs):
        if maps:
            raise ValueError("fixture map failure")
        value = actual_load(path, **kwargs)
        maps.append(value._mmap)
        return value
    monkeypatch.setattr(np, "load", load)
    with pytest.raises(ValueError, match="fixture map failure"):
        FeatureIndex(index.root, model=index.model, tokenizer=index.tokenizer, dictionary=index.dictionary)
    assert len(maps) == 1 and maps[0].closed
    (index.root / "index" / "table.npy").unlink()


def test_table_partial_allocation_closes_earlier_map(tmp_path, monkeypatch):
    actual_open = np.lib.format.open_memmap
    maps = []
    def open_map(path, **kwargs):
        if maps:
            raise OSError("fixture map failure")
        value = actual_open(path, **kwargs)
        maps.append(value._mmap)
        return value
    monkeypatch.setattr(np.lib.format, "open_memmap", open_map)
    with pytest.raises(OSError, match="fixture map failure"):
        _TopTable(tmp_path, 1, 1)
    assert len(maps) == 1 and maps[0].closed
    (tmp_path / "table.npy").unlink()


def test_cli_shared_guard_enforces_twelve_gb_before_model_load(tmp_path, monkeypatch):
    from tools.interp import build_index as command
    from caliper.interp._memory import MemoryGuard
    from .fixture import TinyLM
    _t2_source_manifest(tmp_path, TinyLM())
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("memory limit loaded a model"))
    monkeypatch.setattr(command, "MemoryGuard", lambda gb, device: MemoryGuard(
        gb, device, resident=lambda: 0, footprint=lambda: 0, peak=lambda: 12_000_000_001,
        driver=lambda: 0, pressure=lambda: 50))
    assert command.main(["--dictionary", str(tmp_path), "--device", "cpu"]) == 2
    assert not (tmp_path / "index").exists()


def test_text_scan_observes_resource_stops():
    from tools.interp.build_index import stream_documents
    from tools.interp.text_stream import TextStream
    stream = TextStream([{"text": "Example document"}], TinyTokenizer(), "index")
    def check(stage):
        if stage == "text document":
            raise RuntimeError("fixture scan stop")
    with pytest.raises(RuntimeError, match="fixture scan stop"):
        list(stream_documents(stream, check=check))


def test_index_identity_checks_the_supplied_memory_limit_before_export_and_open(tmp_path):
    from caliper.interp._memory import CleanStop, MemoryGuard

    model, _ = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model)
    tokenizer = TinyTokenizer()
    memory = MemoryGuard(.001, "cpu", resident=lambda: 0, footprint=lambda: 0,
                         peak=lambda: 0, driver=lambda: 0, pressure=lambda: 50)
    arguments = dict(model=model, tokenizer=tokenizer, dictionary=dictionary,
                     provenance={"partition": "fixture-examples"})
    with pytest.raises(CleanStop, match="identity allocation"):
        build_index(tmp_path, [("first", [65, 66])], memory=memory, **arguments)
    assert not (tmp_path / "index").exists()
    build_index(tmp_path, [("first", [65, 66])], **arguments)
    with pytest.raises(CleanStop, match="identity allocation"):
        FeatureIndex(tmp_path, model=model, tokenizer=tokenizer, dictionary=dictionary, memory=memory)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="device dictionary identity requires MPS")
def test_device_dictionary_hash_retains_cpu_identity(tmp_path):
    from caliper.interp._index import _dictionary_hash
    from caliper.interp._memory import MemoryGuard, autorelease_scope

    model, _ = write_fixture(tmp_path)
    dictionary = load_dictionary(tmp_path, model=model, dtype=torch.float32)
    expected = _dictionary_hash(dictionary)
    memory = MemoryGuard(12, "mps")
    with autorelease_scope("mps"):
        dictionary.to("mps")
        assert _dictionary_hash(dictionary, memory=memory) == expected
        assert _dictionary_hash(dictionary) == expected


def test_cli_local_parquet_preserves_source_and_closes_stream(tmp_path, monkeypatch):
    import json
    from tools.interp import build_index as command
    from tools.interp.text_stream import fineweb_documents, TextStream
    from .fixture import TinyLM
    from .test_training_stream import local_manifest

    model = TinyLM()
    metadata = _t2_source_manifest(tmp_path, model)
    manifest = local_manifest(tmp_path)
    identity = fineweb_documents(local_manifest=manifest).source_id
    metadata["training"]["source_id"] = identity
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: (model, TinyTokenizer()))
    opened = []
    def stream(*args, **kwargs):
        value = TextStream(*args, **kwargs)
        opened.append(value)
        return value
    monkeypatch.setattr(command, "TextStream", stream)
    assert command.main(["--dictionary", str(tmp_path), "--device", "cpu", "--tokens", "73",
                         "--context", "17", "--local-dataset-manifest", str(manifest)]) == 0
    assert len(opened) == 1 and opened[0].closed
    with FeatureIndex(tmp_path, model=model, tokenizer=TinyTokenizer()) as index:
        assert index.metadata["tokens"] == 73
        assert index.metadata["provenance"]["source_id"] == identity
        assert index.metadata["provenance"]["source_kind"] == "pinned_local_parquet"


@pytest.mark.parametrize("source_kind", ["dataset", "local_parquet", "jsonl"])
def test_cli_refuses_a_different_training_source(tmp_path, monkeypatch, source_kind):
    import json
    from tools.interp import build_index as command
    from tools.interp.text_stream import TextStream
    from .fixture import TinyLM
    from .test_training_stream import local_manifest

    model = TinyLM()
    metadata = _t2_source_manifest(tmp_path, model)
    metadata["training"]["source_id"] = "different-source"
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    args = ["--dictionary", str(tmp_path), "--device", "cpu"]
    if source_kind == "local_parquet":
        args.extend(["--local-dataset-manifest", str(local_manifest(tmp_path))])
    elif source_kind == "jsonl":
        source = tmp_path / "source.jsonl"
        source.write_text('{"text": "fixture"}\n')
        args.extend(["--text-jsonl", str(source)])
    else:
        monkeypatch.setattr(command, "fineweb_documents", lambda **kwargs: [])
    opened = []
    def stream(source, tokenizer, *args, **kwargs):
        assert tokenizer is None
        value = TextStream(source, tokenizer, *args, **kwargs)
        opened.append(value)
        return value
    monkeypatch.setattr(command, "TextStream", stream)
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("source mismatch loaded a model"))
    with pytest.raises(ValueError, match="source differs"):
        command.main(args)
    assert len(opened) == 1 and opened[0].closed
    assert opened[0].state_dict()["rows"] == 0
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("source_id", [None, "", 17])
def test_cli_requires_training_source_identity(tmp_path, monkeypatch, source_id):
    import json
    from tools.interp import build_index as command
    from .fixture import TinyLM

    metadata = _t2_source_manifest(tmp_path, TinyLM())
    if source_id is None:
        del metadata["training"]["source_id"]
    else:
        metadata["training"]["source_id"] = source_id
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("missing source identity loaded a model"))
    with pytest.raises(ValueError, match="requires training source identity"):
        command.main(["--dictionary", str(tmp_path), "--device", "cpu"])
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("source_kind", ["dataset", "local_parquet", "jsonl"])
def test_cli_jsonl_requires_fixture_dictionary(tmp_path, monkeypatch, source_kind):
    import hashlib
    import json
    from tools.interp import build_index as command
    from tools.interp.model_load import MODEL_ID
    from tools.interp.text_stream import fineweb_documents
    from .fixture import TinyLM
    from .test_training_stream import local_manifest

    metadata = _t2_source_manifest(tmp_path, TinyLM())
    metadata["model_id"] = MODEL_ID
    source = tmp_path / "source.jsonl"
    source.write_text('{"text": "fixture"}\n')
    if source_kind == "local_parquet":
        metadata["training"]["source_id"] = fineweb_documents(local_manifest=local_manifest(tmp_path)).source_id
    elif source_kind == "jsonl":
        metadata["training"]["source_id"] = "jsonl:sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "dictionary.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(command, "load_text_model", lambda **kwargs: pytest.fail("incompatible JSONL source loaded a model"))
    with pytest.raises(ValueError, match="only supported for tiny-lm fixture dictionaries"):
        command.main(["--dictionary", str(tmp_path), "--device", "cpu", "--text-jsonl", str(source)])
    assert not (tmp_path / "index").exists()
