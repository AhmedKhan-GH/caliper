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
The verifier reconstructs supported nonpersistent buffers from the checkpoint configuration before they enter the baseline for later checks.
Unknown buffers and nondefault RoPE modes cause refusal.

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

## Pinned Gemma Model

Gemma uses `google/gemma-2-2b` at revision `c5ebcd40d208330abc697524c919956e692655cf`, with model type `gemma2`.
The model has 26 layers, residual width 2,304, and vocabulary size 256,000.
It uses the complete checkpoint without the Qwen text-subset mapping.
The snapshot path must end with `models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf`.

Verification compares weights, tied parameters, and forward settings with the trusted local snapshot.
The settings include the four block norms, GeGLU activation, attention and final softcaps, sliding-window configuration, and embedding scale.
The verifier reconstructs default Gemma RoPE buffers and the nonpersistent embedding-scale scalar within the fixed CPU allocation bound.
It does not allocate another embedding matrix for that scalar check.

The Gemma runtime loader verifies the CPU model before device transfer.
It admits the transfer separately, retains the verified digest and settings, then checks identity again with the destination memory guard.
Model and tokenizer loads use local files only.
The official model requires gated access under the Gemma license.

CPU tests with random configurations cover these verification paths and refusals.
They do not establish verification of the actual gated checkpoint or its memory use.
The [Gemma Scope reference](gemma-scope.md) records conversion coordinates, source pins, and the remaining acceptance work.

## Identity And Resource Limits

Verification records provenance for one model instance.
Later identity checks hash all current weights and buffers in bounded chunks, including changes through aliases.
A model copy requires its own verification.
A dtype change requires verification against the snapshot again.
A real device change also requires verification again when the previous memory guard does not cover that device.
The spellings `mps` and `mps:0` refer to the same sole MPS device.

Verification defaults to a total process limit of 12 GB.
Only the exact pinned Gemma identity permits an explicit `memory_limit_gb` above 12, with a maximum of 40 GB.
Qwen and other identities retain the 12 GB ceiling.
The MPS process limit remains active after verification, as required by the run policy.
Library memory diagnostics use stderr, and tool result records retain their stdout channel.
The recorded Qwen MPS run uses this verification path.
CPU tests cover refusal cases and bounded transfers.
