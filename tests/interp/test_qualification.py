"""Exact live-resource qualification with bounded local fixture artifacts."""
from copy import deepcopy
import json

import pytest

from caliper import _interpret_protocol as protocol
from caliper.interp._dictionary import dictionary_fingerprint
from caliper.interp._qualification import dictionary_response
from .test_service import actual


def response(service):
    return service._execute("feature.dictionary", {}, 0)


def test_qualification_binds_live_weights_files_index_and_tokenizer(actual):
    result = response(actual)
    assert protocol.bounded(result)
    assert result["metadata"] == actual.dictionary.metadata
    assert result["index"] == actual.index.metadata
    assert result["identity"]["model_hash"] == actual.dictionary.metadata["model_hash"]
    assert result["identity"]["dictionary_hash"] == actual.index.metadata["dictionary_hash"]
    assert result["identity"]["index_hash"] == actual.index.metadata["index_hash"]
    assert result["identity"]["tokenizer_hash"] == actual.index.metadata["tokenizer_hash"]
    assert result["identity"]["dictionary_fingerprint"] == dictionary_fingerprint(
        actual.index.root, actual.dictionary.metadata)
    result["metadata"]["quality"]["status"] = "changed"
    result["index"]["tokens"] = -1
    assert actual.dictionary.metadata["quality"].get("status") != "changed"
    assert actual.index.metadata["tokens"] > 0


@pytest.mark.parametrize("change", ["dictionary_alias", "model_alias", "metadata", "source_file",
                                   "index_file", "index_manifest", "tokenizer", "closed"])
def test_qualification_refuses_changed_resources(actual, change):
    response(actual)
    root = actual.index.root
    if change == "dictionary_alias":
        value = actual.dictionary.layers[0].W_dec
        version = value._version
        value.data.add_(.01)
        assert value._version == version
    elif change == "model_alias":
        actual.model.lm_head.weight.data.add_(.01)
    elif change == "metadata":
        actual.dictionary.metadata["quality"]["status"] = "passed"
    elif change in ("source_file", "index_file"):
        path = root / ("layer_00.safetensors" if change == "source_file" else "index/tokens.bin")
        with path.open("r+b") as output:
            output.seek(-1, 2)
            byte = output.read(1)
            output.seek(-1, 2)
            output.write(bytes([byte[0] ^ 1]))
    elif change == "index_manifest":
        path = root / "index/index.json"
        data = json.loads(path.read_text())
        data["tokens"] += 1
        path.write_text(json.dumps(data))
    elif change == "tokenizer":
        actual.tokenizer.eos_token_id += 1
    else:
        actual.index.close()
    with pytest.raises(ValueError, match="qualification"):
        response(actual)


def test_qualification_refuses_unrelated_live_resources(actual):
    with pytest.raises(ValueError, match="current live"):
        dictionary_response(deepcopy(actual.dictionary), actual.index,
                            model=actual.model, tokenizer=actual.tokenizer)


def test_dictionary_without_index_keeps_metadata_only(actual):
    actual.index = None
    assert response(actual) == {"metadata": actual.dictionary.metadata}


def test_tool_and_runtime_fingerprints_keep_the_same_format(actual):
    from tools.interp.checkpoint_state import dictionary_fingerprint as tool_fingerprint
    assert tool_fingerprint is dictionary_fingerprint
    before = dictionary_fingerprint(actual.index.root, actual.dictionary.metadata)
    metadata = deepcopy(actual.dictionary.metadata)
    metadata["quality"]["fvu"] = .01
    assert dictionary_fingerprint(actual.index.root, metadata) == before
    metadata["training"]["tokens"] += 1
    assert dictionary_fingerprint(actual.index.root, metadata) != before
