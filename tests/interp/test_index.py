"""CPU tests for FeatureIndex, LiveFeatureIndex, _TopTable, and build_index."""
from collections import Counter
import json
import shutil

import numpy as np
import pytest
import torch

from caliper.interp._backend import make_tiny_backend
from caliper.interp._index import FeatureIndex, LiveFeatureIndex, _TopTable, build_index, resolve_index_dir
from caliper.interp._memory import CleanStop
from tools.interp import build_index as cli_build_index


@pytest.fixture
def indexed(tmp_path):
    backend = make_tiny_backend()
    documents = [
        ("first", [2, 37, 38, 37, 39] * 10),
        ("second", [42, 43, 44]),
        ("last", [37]),
    ]
    build_index(
        tmp_path, documents, model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, provenance={"partition": "fixture-examples"}, context_length=17,
    )
    with FeatureIndex(
        tmp_path, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary
    ) as index:
        yield index, documents, backend


def test_boundary_ties_across_batches(tmp_path):
    table = _TopTable(tmp_path, 1, 3)
    values = torch.tensor([[2.0, 0.0, -1.0]] * 37, dtype=torch.float64)
    values[22, 0] = 4.0
    values[29, 0] = 3.0
    values[25, 1] = 1.0
    table.update(0, values[:20], 0)
    table.update(0, values[20:], 20)
    np.testing.assert_array_equal(table.table[0, 0]["offset"], [22, 29] + list(range(14)))
    assert table.counts.tolist() == [[37, 1, 0]]
    assert table.table[0, 1, 0]["offset"] == 25
    assert np.all(table.table[0, 2]["offset"] == -1)
    assert np.all(table.table[0, 2]["activation"] == 0)
    table.close()


def test_document_windows_promoted_labels_and_cache(indexed):
    index, documents, _backend = indexed
    feature = int(np.argmax(index.counts[0]))
    info = index.show(0, feature)
    weight = index.model.W_U.T.detach()
    direction = index.dictionary.layers[0].W_dec[feature]
    scores = (weight @ direction).tolist()
    promoted = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
    assert [t["token_id"] for t in info["promoted_tokens"]] == promoted

    rows = index.examples(0, feature)
    assert 1 <= len(rows) <= 16
    assert [r["activation"] for r in rows] == sorted((r["activation"] for r in rows), reverse=True)
    source = dict(documents)
    for row in rows:
        assert all(len(row[k]) == 33 for k in ("token_ids", "tokens", "activations", "valid", "positions"))
        assert row["positions"][16] == row["position"]
        toks = source[row["document_id"]]
        for slot, pos in enumerate(row["positions"]):
            if pos is None:
                assert row["token_ids"][slot] is None and row["activations"][slot] is None
            else:
                assert row["token_ids"][slot] == toks[pos]

    frequencies = Counter(i for row in rows for i in row["token_ids"] if i is not None)
    expected = sorted(frequencies, key=lambda i: (-frequencies[i], i))[:3]
    assert [t["token_id"] for t in info["example_tokens"]] == expected


@pytest.mark.parametrize("layer,feature,limit", [
    (-1, 0, 16), (2, 0, 16), (0, -1, 16), (0, 32, 16),
    (0, 0, 17), (0, 0, -1), (True, 0, 16), (0, 1.5, 16),
])
def test_invalid_queries(indexed, layer, feature, limit):
    index, _, _ = indexed
    with pytest.raises(ValueError):
        index.examples(layer, feature, limit)


def test_empty_index_and_empty_document_refusal(tmp_path):
    backend = make_tiny_backend()
    build_index(
        tmp_path, [], model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, provenance={"partition": "empty"},
    )
    with FeatureIndex(
        tmp_path, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary
    ) as index:
        assert index.examples(0, 0) == []
        info = index.show(0, 0)
        assert info["count"] == 0 and info["maximum"] == 0.0
        assert info["label"].endswith("Examples: none")

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="must not be empty"):
        build_index(
            other, [("empty", [])], model=backend.model, tokenizer=backend.tokenizer,
            dictionary=backend.dictionary, provenance={"partition": "fixture"},
        )
    assert not (other / "index" / "index.json").exists()


def test_live_feature_index_and_cli_query(tmp_path, capsys):
    backend = make_tiny_backend()
    backend.record_prompt("ABCD")
    with LiveFeatureIndex(backend, allow_online=False) as live_idx:
        _, acts = backend.activations("ABCD")
        feat = int(acts[0, 2].argmax().item())
        shown = live_idx.show(0, feat)
        assert shown["maximum"] > 0.0 and len(shown["promoted_tokens"]) == 3
        rows = live_idx.examples(0, feat, limit=4)
        assert rows and len(rows[0]["token_ids"]) == 33
        lbls = live_idx.labels([(0, feat)])
        assert lbls[(0, feat)].startswith("Output: ")

    backend.dictionary.raw.to_safetensors(str(tmp_path))
    build_index(
        tmp_path, [("doc", [2, 37, 38, 39])], model=backend.model,
        tokenizer=backend.tokenizer, dictionary=backend.dictionary,
        provenance={"partition": "cli"},
    )
    rc = cli_build_index.main(["--model", "tiny-lm", "--dictionary", str(tmp_path), "--device", "cpu", "--query", "0", "0", "--limit", "2"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"show"' in out and '"examples"' in out


def test_features_manifest_key_and_index_resolution(tmp_path, monkeypatch):
    backend = make_tiny_backend()
    dict_dir = tmp_path / "dict"
    dict_dir.mkdir()
    documents = [
        ("alpha", [2, 37, 38, 39, 37] * 8),
        ("beta", [42, 43, 44, 45]),
    ]
    build_index(
        dict_dir, documents, model=backend.model, tokenizer=backend.tokenizer,
        dictionary=backend.dictionary, provenance={"partition": "compat-fixture"}, context_length=16,
    )

    nested_index = dict_dir / "index"
    manifest_path = nested_index / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latent_count = manifest.pop("latents")
    manifest["features"] = latent_count
    manifest.pop("dictionary_id", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert resolve_index_dir(nested_index) == nested_index
    assert FeatureIndex.resolve(dict_dir) == nested_index
    assert resolve_index_dir(None) is None
    assert resolve_index_dir("") is None

    with FeatureIndex(
        nested_index, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary
    ) as direct_index:
        assert direct_index.index_dir == nested_index
        feature = int(np.argmax(direct_index.counts[0]))
        direct_info = direct_index.show(0, feature)
        direct_rows = direct_index.examples(0, feature, limit=4)
        direct_labels = direct_index.labels([(0, feature)])
        assert direct_info["count"] > 0 and direct_info["maximum"] > 0.0
        assert 1 <= len(direct_rows) <= 4
        assert all(len(row["token_ids"]) == 33 for row in direct_rows)
        assert direct_labels[(0, feature)] == direct_info["label"]

    sibling_index = tmp_path / "index"
    shutil.move(str(nested_index), str(sibling_index))
    assert FeatureIndex.resolve(dict_dir) == sibling_index

    # Nonexistent hub aliases must not match an unrelated ./index folder in cwd.
    monkeypatch.chdir(tmp_path)
    assert FeatureIndex.resolve("gemma") is None

    with FeatureIndex(
        dict_dir, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary
    ) as sibling_loaded:
        assert sibling_loaded.index_dir == sibling_index
        assert sibling_loaded.show(0, feature) == direct_info
        assert sibling_loaded.examples(0, feature, limit=4) == direct_rows

    sibling_manifest = sibling_index / "index.json"
    bad_manifest = dict(manifest, features=latent_count + 1)
    sibling_manifest.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions differ"):
        FeatureIndex(dict_dir, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary)

    conflict_manifest = dict(manifest, latents=latent_count, features=latent_count + 1)
    sibling_manifest.write_text(json.dumps(conflict_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        FeatureIndex(dict_dir, model=backend.model, tokenizer=backend.tokenizer, dictionary=backend.dictionary)
