"""Read-only identities for the live dictionary and its completed example index."""
from copy import deepcopy
import hashlib
import json

from ._context import context_policy
from ._dictionary import dictionary_fingerprint, model_identity
from ._evaluation_identity import evaluation_dictionary_evidence, evaluation_model_evidence
from ._gemma_import_identity import _REFERENCE_INPUT, verified_import_binding
from ._index import (FeatureIndex, WINDOW_TOKENS, _FILES, _dictionary_hash, _encoding_policy,
                     _hash_file, _json, _source_hash, _tokenizer_hash)
from ._memory import family_memory_limit
from ._reference_snapshots import reference_import_arguments
from ._research_evidence import research_evidence


def qualify_import(folder, metadata, *, model, memory):
    """Select reference evidence only from the current qualified model profile."""
    arguments = reference_import_arguments(model)
    return verified_import_binding(folder, metadata, model=model, memory=memory, **arguments)


def require_reference_index(manifest):
    if (manifest.get("format_version") != 2 or manifest.get("encoder_policy") != "gemma_scope_pre_gain_v1"
            or manifest.get("reference_input") != _REFERENCE_INPUT):
        raise ValueError("reference qualification requires an original-coordinate index")


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
    identity = model_identity(model, memory=memory, memory_limit_gb=family_memory_limit(model, memory=memory))
    if identity != metadata["model_hash"]:
        raise ValueError("dictionary qualification model identity mismatch")
    model_binding = evaluation_model_evidence(model, identity, memory=memory)
    import_binding = qualify_import(index.root, metadata, model=model, memory=memory)
    manifest_file = index.root / "index" / "index.json"
    if manifest_file.stat().st_size > 1024 * 1024:
        raise ValueError("index qualification manifest exceeds its bound")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest != index.metadata:
        raise ValueError("index qualification manifest changed")
    reference = import_binding is not None and import_binding.get("version") == 2
    if reference:
        require_reference_index(manifest)
        _encoding_policy(manifest, model, dictionary, memory)
    policy = context_policy(model, manifest["context_length"], tokenizer=tokenizer)
    example_policy = context_policy(model, WINDOW_TOKENS + 1, tokenizer=tokenizer)
    if (_json(manifest.get("context_policy")) != _json(policy)
            or _json(manifest.get("example_context_policy")) != _json(example_policy)):
        raise ValueError("index qualification context policy changed")
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
    if reference and manifest.get("dictionary_fingerprint") != fingerprint:
        raise ValueError("reference qualification dictionary fingerprint changed")
    tensor_options = {"folder": index.root} if reference else {}
    tensor_binding = evaluation_dictionary_evidence(dictionary, memory=memory, **tensor_options)
    result.update(index=deepcopy(manifest), model_binding=model_binding, tensor_binding=tensor_binding, identity={
        "model_hash": identity, "dictionary_hash": content, "dictionary_fingerprint": fingerprint,
        "index_hash": manifest["index_hash"], "tokenizer_hash": tokenizer_hash,
    })
    if import_binding is not None:
        result["import_binding"] = import_binding
    research = research_evidence(index.root, fingerprint, tensor_binding, model_binding, memory=memory)
    if research is not None:
        result["research_quality"] = research
    if "fallback_source" in metadata:
        result["fallback_source"] = deepcopy(metadata["fallback_source"])
    memory.check("dictionary qualification complete")
    return result
