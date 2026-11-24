"""Read-only identities for the live dictionary and its completed example index."""
from copy import deepcopy
import hashlib
import json

from ._dictionary import dictionary_fingerprint, model_identity
from ._evaluation_identity import evaluation_dictionary_evidence, evaluation_model_evidence
from ._gemma_import_identity import verified_import_binding
from ._index import FeatureIndex, _FILES, _dictionary_hash, _hash_file, _json, _source_hash, _tokenizer_hash


def dictionary_response(dictionary, index=None, *, model=None, tokenizer=None):
    result = {"metadata": deepcopy(dictionary.metadata)}
    if index is None:
        return result
    if (not isinstance(index, FeatureIndex) or index._closed or index.dictionary is not dictionary
            or index.model is not model or index.tokenizer is not tokenizer):
        raise ValueError("dictionary qualification requires the current live index resources")
    memory = index.memory
    memory.check("dictionary qualification")
    metadata = dictionary.metadata
    ceiling = 28 if getattr(model.config, "model_type", None) == "gemma2" else 12
    identity = model_identity(model, memory_limit_gb=min(ceiling, memory.limit / 1e9))
    if identity != metadata["model_hash"]:
        raise ValueError("dictionary qualification model identity mismatch")
    model_binding = evaluation_model_evidence(model, identity, memory=memory)
    import_binding = verified_import_binding(index.root, metadata, model=model, memory=memory)
    manifest_file = index.root / "index" / "index.json"
    if manifest_file.stat().st_size > 1024 * 1024:
        raise ValueError("index qualification manifest exceeds its bound")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest != index.metadata:
        raise ValueError("index qualification manifest changed")
    payload = {key: value for key, value in manifest.items() if key != "index_hash"}
    if manifest["index_hash"] != hashlib.sha256(_json(payload).encode()).hexdigest():
        raise ValueError("index qualification identity mismatch")
    content = _dictionary_hash(dictionary, memory=memory)
    if (content != manifest["dictionary_hash"] or manifest["model_hash"] != identity
            or _source_hash(index.root, metadata["layers"]) != manifest["dictionary_source_hash"]):
        raise ValueError("dictionary qualification content changed")
    tokenizer_hash = _tokenizer_hash(tokenizer, model.get_output_embeddings().weight.shape[0])
    if tokenizer_hash != manifest["tokenizer_hash"]:
        raise ValueError("index qualification tokenizer changed")
    for name in _FILES:
        memory.check("index qualification file", force=False)
        path = index.root / "index" / name
        record = manifest["files"][name]
        if path.stat().st_size != record["bytes"] or _hash_file(path) != record["sha256"]:
            raise ValueError("index qualification file changed")
    fingerprint = dictionary_fingerprint(index.root, metadata, memory=memory)
    tensor_binding = evaluation_dictionary_evidence(dictionary, memory=memory)
    result.update(index=deepcopy(manifest), model_binding=model_binding, tensor_binding=tensor_binding, identity={
        "model_hash": identity, "dictionary_hash": content, "dictionary_fingerprint": fingerprint,
        "index_hash": manifest["index_hash"], "tokenizer_hash": tokenizer_hash,
    })
    if import_binding is not None:
        result["import_binding"] = import_binding
    if "fallback_source" in metadata:
        result["fallback_source"] = deepcopy(metadata["fallback_source"])
    memory.check("dictionary qualification complete")
    return result
