# Trace Ablation Support

The attribution engine supports the ablations in the table below.
An intervention point can be valid for the live model and outside this traced subset.
The tracer refuses other ablation points before identity checks or model execution.
The refusal also applies when a previous trace remains in the cache.
The standalone frozen path applies the same check.

## Supported Points

`<block>` is a decoder block, such as `layers.0` on the fixture or `model.layers.0` on Qwen3.5 and Gemma 2.
`<attention>` is `<block>.self_attn` for full attention or `<block>.linear_attn` for DeltaNet.
The output projection is `o_proj` for full attention and `out_proj` for DeltaNet.

| Point | Families | Frozen Effect |
|---|---|---|
| `<block>.mlp` | Fixture, Qwen3.5 | Zero features, decoder bias, and MLP output |
| `<block>.mlp_out` | Fixture, Qwen3.5, Gemma 2 | Zero features, decoder bias, and semantic residual write |
| `<block>.resid_pre` | Fixture, Qwen3.5, Gemma 2 | Zero the residual before the block |
| `<block>.mlp_in` | Fixture, Qwen3.5, Gemma 2 | Zero the transcoder read input |
| `<attention>` | Fixture, Qwen3.5, Gemma 2 | Zero the attention output |
| `<attention>.<output_projection>` | Fixture, Qwen3.5, Gemma 2 | Zero the attention output |
| `<attention>#<head>` | Resolver-dependent | Zero the same output-projection input slice as the live head ablation |

A head point must pass the live ablation resolver for that attention module.
The tracer does not infer head support from a point name alone.
A whole MLP ablation removes its feature nodes, because the MLP makes no feature contribution.
A complete forward check and direct-effect checks verify the supported paths on the CPU fixture.
The local MPS run traces the pinned Qwen checkpoint and tests feature clamps.
The supported module and head ablations in this table retain CPU fixture coverage.

## Gemma Coordinates

Gemma `mlp_in` reads the actual GeGLU input after the pre-feedforward norm and its learned gain.
Gemma `mlp_out` writes after `post_feedforward_layernorm`, at the residual sum.
The raw `.mlp` output precedes this norm, and the tracer refuses edits at that raw output.
An edit at `mlp_out` does not apply the post-feedforward norm a second time.

The frozen attention path retains the captured softcapped probabilities and sliding-window masks.
Its value path passes through the frozen post-attention norm before the residual sum.
The final logit map uses the local affine softcap tangent and an explicit centered bias.
The [Gemma Scope reference](gemma-scope.md) gives the complete coordinate and softcap contracts.

Random Gemma CPU configurations cover semantic ablation, add, patch, feature clamps, and direct-effect checks.
They do not establish real Gemma quality, GPU memory, speed, or acceptance.
Generic Llama support remains deferred.

## Unsupported Points

The tracer refuses MLP submodules, including `down_proj`, `gate_proj`, and `up_proj`.
It also refuses attention input projections, internal attention normalization, decoder normalization, whole decoder blocks, and the final normalization or output head.
Examples include `self_attn.v_proj`, `linear_attn.in_proj_qkv`, `linear_attn.norm`, and `<block>.input_layernorm`.
The error identifies the unsupported ablation or intervention point.

These ablations need separate frozen-path semantics.
A zero live MLP output alone does not justify active transcoder features with an opposite error contribution.
The tracer thus refuses such a path before it can emit phantom feature nodes.
This restriction does not change the live ablation interface.
It does not add support for attention tuple `add` or `patch` operations.
