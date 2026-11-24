# Trace Ablation Support

The attribution engine supports the ablations in the table below.
An intervention point can be valid for the live model and outside this traced subset.
The tracer refuses other ablation points before identity checks or model execution.
The refusal also applies when a previous trace remains in the cache.
The standalone frozen path applies the same check.

## Supported Points

`<block>` is a decoder block, such as `layers.0` on the fixture or `model.layers.0` on Qwen3.5.
`<attention>` is `<block>.self_attn` for full attention or `<block>.linear_attn` for DeltaNet.
The output projection is `o_proj` for full attention and `out_proj` for DeltaNet.

| Point | Frozen Effect |
|---|---|
| `<block>.mlp` | Zero features, decoder bias, and MLP output |
| `<block>.mlp_out` | Same effect as the whole MLP ablation |
| `<block>.resid_pre` | Zero the residual before the block |
| `<block>.mlp_in` | Zero the transcoder read input |
| `<attention>` | Zero the attention output |
| `<attention>.<output_projection>` | Zero the attention output |
| `<attention>#<head>` | Zero the same output-projection input slice as the live head ablation |

A head point must pass the live ablation resolver for that attention module.
The tracer does not infer head support from a point name alone.
A whole MLP ablation removes its feature nodes, because the MLP makes no feature contribution.
A complete forward check and direct-effect checks verify the supported paths on the CPU fixture.
The local MPS run traces the pinned Qwen checkpoint and tests feature clamps.
The supported module and head ablations in this table retain CPU fixture coverage.

## Unsupported Points

The tracer refuses MLP submodules, including `down_proj`, `gate_proj`, and `up_proj`.
It also refuses attention input projections, internal attention normalization, decoder normalization, whole decoder blocks, and the final normalization or output head.
Examples include `self_attn.v_proj`, `linear_attn.in_proj_qkv`, `linear_attn.norm`, and `<block>.input_layernorm`.
The error identifies the unsupported point through `trace does not support ablation point`.

These ablations need separate frozen-path semantics.
A zero live MLP output alone does not justify active transcoder features with an opposite error contribution.
The tracer thus refuses such a path before it can emit phantom feature nodes.
This restriction does not change the live ablation interface.
It does not add support for attention tuple `add` or `patch` operations.
