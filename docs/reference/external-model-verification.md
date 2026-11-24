# External Model Verification

Caliper LIB verifies external Hugging Face weights through `caliper.interp.verify_hf_model`.
Call this function before `load_dictionary(path, model=model)` for a model with a pinned Hugging Face identity.
The verifier accepts an existing local safetensors snapshot and makes no network request.
The model configuration and snapshot path must name the same repository and exact commit.

## Snapshot Trust

The verifier trusts the local snapshot as source data.
It compares each model weight with the corresponding checkpoint weight after conversion to the model dtype.
It does not authenticate files against Hugging Face, a signature, or an independent remote digest.
A snapshot directory name identifies the declared revision but does not prove the source of its contents.
Use snapshot data whose source you trust.

Verification does not change checkpoint files or model weights.
Each comparison uses slices of at most 1 MiB before dtype conversion.
Missing weights, unexpected weights, duplicate keys, ambiguous mappings, and byte differences cause refusal.
An omitted tied weight is valid only when the model uses the same Parameter object for both names.
Nonpersistent model buffers enter the baseline for later checks, because the checkpoint does not contain them.

## Pinned Qwen Text Subset

The supported text subset is `qwen3.5-0.8b-text` for `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`.
The model type must be `qwen3_5_text`.
The subset maps the literal checkpoint prefix `model.language_model.` to `model.`.
It declares exactly 153 vision keys and 15 MTP keys outside the text model.
The verifier checks their names but does not read their tensor payloads.
All other checkpoint keys must match model weights.

The pinned snapshot header contains 488 tensor entries in one safetensors file.
Of these, 320 entries have the `model.language_model.` prefix.
The snapshot omits `lm_head.weight`, and the text configuration ties that parameter to the token embedding.
The 59,840-byte header and shard index confirm these namespaces without a tensor load.
The pinned checkpoint passed end-to-end identity verification in the local MPS circuit trace run.
This result covers the declared text subset.

The following example assumes that `model` already contains the text model in its final dtype and device.
The `snapshot` path must end with the standard Hugging Face cache structure shown below.
The example does not load or download a model.

```python
from pathlib import Path
from caliper.interp import load_dictionary, verify_hf_model

revision = "2fc06364715b967f1860aea9cf38778875588b17"
snapshot = Path("cache/models--Qwen--Qwen3.5-0.8B/snapshots") / revision

identity = verify_hf_model(
    model,
    snapshot,
    checkpoint_subset="qwen3.5-0.8b-text",
    memory_limit_gb=12,
)
dictionary = load_dictionary("dictionary", model=model)
```

The named subset supplies its own mapping, thus it cannot accompany `key_mapping`.
A missing declared non-text key also causes refusal.
This subset does not permit additional vision or MTP keys through a wildcard.
For another complete checkpoint, omit `checkpoint_subset` and supply literal prefix replacements through `key_mapping` if necessary.
Those replacements use plain prefixes, not regular expressions.

## Identity And Resource Limits

Verification records provenance for one model instance.
Later identity checks hash all current weights and buffers in bounded chunks, including changes through aliases.
A model copy requires its own verification.
A dtype change requires verification against the snapshot again.
A real device change also requires verification again when the previous memory guard does not cover that device.
The spellings `mps` and `mps:0` refer to the same sole MPS device.

Verification enforces a total process limit of at most 12 GB.
The MPS process limit remains active after verification, as required by the run policy.
Library memory diagnostics use stderr, and tool result records retain their stdout channel.
The local MPS run uses this verification path.
CPU tests cover refusal cases and bounded transfers.
