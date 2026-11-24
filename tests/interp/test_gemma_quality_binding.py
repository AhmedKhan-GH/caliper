"""Exact quality identities from small verified random CPU checkpoints."""
from copy import deepcopy
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from caliper.interp._dictionary import dictionary_fingerprint, model_identity, verify_hf_model
from caliper.interp._evaluation_identity import evaluation_model_evidence
from caliper.interp._index import FeatureIndex, build_index
from caliper.interp._qualification import dictionary_response
from demos.circuit_trace._qualification import _gemma_model_binding
from tools.interp import eval_dictionary
from tools.interp.evaluation_binding import EvaluationBinding
from tools.interp.text_stream import TextStream
from .test_gemma_import_binding import artifact
from .test_gemma_runtime import local_gemma
from .test_training_stream import documents


def evidence(model):
    return evaluation_model_evidence(model, model_identity(model))


def reply(folder, model, dictionary):
    tokenizer = SimpleNamespace(eos_token_id=2, convert_ids_to_tokens=lambda value: str(value))
    build_index(folder, [("fixture", [3, 4, 5, 6])], model=model, tokenizer=tokenizer,
                dictionary=dictionary, provenance={"partition": "fixture"}, context_length=4)
    with FeatureIndex(folder, model=model, tokenizer=tokenizer, dictionary=dictionary) as index:
        return dictionary_response(dictionary, index, model=model, tokenizer=tokenizer)


def test_quality_and_live_response_share_verified_evidence(artifact):
    folder, model, dictionary, _ = artifact
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [3 + ord(ch) % 34 for ch in text], eos_token_id=2)
    with TextStream(documents(), tokenizer, "quality") as stream:
        quality = eval_dictionary.evaluate(model, dictionary, stream, folder, tokens=8,
                                          context=4, batch_tokens=8, chunk_tokens=2)
    assert quality["status"] == "partial"
    assert quality["evaluated_tokens"] == 8
    with TextStream(documents(), tokenizer, "quality") as stream:
        resumed = eval_dictionary.evaluate(model, dictionary, stream, folder, tokens=8,
                                          context=4, batch_tokens=8, chunk_tokens=2, resume=True)
    for field in ("fvu", "dead_fraction", "cross_entropy_delta", "evaluated_tokens", "model_binding"):
        assert resumed[field] == quality[field]
    assert resumed["publisher_training_overlap"] == "unknown"
    result = reply(folder, model, dictionary)
    assert result["model_binding"] == quality["model_binding"] == evidence(model)
    _gemma_model_binding(result["metadata"], result)
    result["model_binding"]["tensor_hash"] = "0" * 64
    assert result["metadata"]["quality"]["model_binding"] == evidence(model)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_verified_precision_change_cannot_reuse_quality(artifact, dtype):
    _, model, _, snapshot = artifact
    quality_binding = evidence(model)
    changed = deepcopy(model).to(dtype=dtype)
    verify_hf_model(changed, snapshot)
    current = evidence(changed)
    assert current["model_hash"] == quality_binding["model_hash"]
    assert current["tensor_hash"] != quality_binding["tensor_hash"]
    with pytest.raises(ValueError, match="tensors or forward settings differ"):
        _gemma_model_binding({"quality": {"model_binding": quality_binding}}, {"model_binding": current})


@pytest.mark.parametrize("setting", ["epsilon", "softcap", "activation"])
def test_fresh_index_cannot_reuse_quality_from_other_settings(artifact, tmp_path, setting):
    folder, model, dictionary, _ = artifact
    quality_binding = EvaluationBinding(
        folder, dictionary, dictionary_fingerprint(folder, dictionary.metadata), model=model).model_evidence
    changed = deepcopy(model)
    if setting == "epsilon":
        norm = changed.model.layers[0].pre_feedforward_layernorm
        norm.eps = math.nextafter(norm.eps, math.inf)
        changed.config.rms_norm_eps = norm.eps
        for module in changed.modules():
            if hasattr(module, "eps"):
                module.eps = norm.eps
    elif setting == "softcap":
        changed.config.final_logit_softcapping = math.nextafter(changed.config.final_logit_softcapping, math.inf)
    else:
        changed.config.hidden_activation = "relu"
        for block in changed.model.layers:
            block.mlp.act_fn = torch.nn.ReLU().eval()
    verify_hf_model(changed, local_gemma(tmp_path / "changed", changed))
    dictionary.metadata["quality"]["model_binding"] = quality_binding
    (folder / "dictionary.json").write_text(json.dumps(dictionary.metadata))
    result = reply(folder, changed, dictionary)
    assert result["model_binding"]["model_hash"] == quality_binding["model_hash"]
    assert result["model_binding"]["tensor_hash"] == quality_binding["tensor_hash"]
    assert result["model_binding"]["forward_settings_hash"] != quality_binding["forward_settings_hash"]
    with pytest.raises(ValueError, match="tensors or forward settings differ"):
        _gemma_model_binding(result["metadata"], result)


def test_model_binding_keeps_activation_type_stable_across_processes(artifact):
    _, model, _, snapshot = artifact
    expected = evidence(model)
    code = """
import json
import sys
import torch
from caliper.interp._dictionary import model_identity, verify_hf_model
from caliper.interp._evaluation_identity import evaluation_model_evidence
from tests.interp.test_gemma_runtime import gemma_model
model = gemma_model()
with torch.no_grad():
    for block in model.model.layers:
        block.pre_feedforward_layernorm.weight.copy_(torch.linspace(-.3, .7, 16))
verify_hf_model(model, sys.argv[1])
print(json.dumps(evaluation_model_evidence(model, model_identity(model))))
"""
    process = subprocess.run([sys.executable, "-c", code, str(snapshot)], capture_output=True, text=True, check=True)
    assert json.loads(process.stdout) == expected


def test_unversioned_alias_cannot_forge_a_current_binding(artifact):
    folder, model, dictionary, _ = artifact
    before = evidence(model)
    weight = model.lm_head.weight
    version = weight._version
    weight.data[0, 0] += .125
    assert weight._version == version
    dictionary.metadata["quality"]["model_binding"] = before
    with pytest.raises(ValueError, match="identity mismatch"):
        EvaluationBinding(folder, dictionary, dictionary_fingerprint(folder, dictionary.metadata), model=model)
    with pytest.raises(ValueError, match="identity mismatch"):
        evidence(model)
