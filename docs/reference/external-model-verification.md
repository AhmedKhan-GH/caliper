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

## Pinned Gemma Model

Gemma uses `google/gemma-2-2b` at revision `c5ebcd40d208330abc697524c919956e692655cf`, with model type `gemma2`.
The model has 26 layers, residual width 2,304, and vocabulary size 256,000.
Verification requires the complete checkpoint.
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
The [Gemma Scope reference](gemma-scope.md) defines conversion coordinates, source pins, evaluation requirements, and acceptance limits.

## Identity And Resource Limits

Verification records provenance for one model instance.
Later identity checks hash all current weights and buffers in bounded chunks, including changes through aliases.
A model copy requires its own verification.
A dtype change requires verification against the snapshot again.
A real device change also requires verification again when the previous memory guard does not cover that device.
The spellings `mps` and `mps:0` refer to the same sole MPS device.

Verification defaults to a total process limit of 12 GB.
Only the exact pinned Gemma identity permits an explicit `memory_limit_gb` above 12, with a maximum of 40 GB.
Other identities retain the 12 GB ceiling.
The MPS process limit remains active after verification, as required by the run policy.
Library memory diagnostics use stderr, and tool result records retain their stdout channel.
CPU tests cover refusal cases and bounded transfers.
