"""Experimental Qwen text family metadata, capture, and frozen value paths.

Shared callers supply lifecycle and affine primitives. This module does not
import its dispatch sites or load Transformers until a Qwen operation needs it.
"""
from __future__ import annotations


FAMILY_NAME = "qwen3_5_text"
FAMILY_METADATA = dict(name=FAMILY_NAME, decoder_class="Qwen3_5TextModel",
                       block_class="Qwen3_5DecoderLayer",
                       module_name="transformers.models.qwen3_5.modeling_qwen3_5",
                       unit_offset_norm=True)


def mixer(block):
    """Return the module name and pattern kind for one Qwen block."""
    return ("linear_attn", "delta") if block.block_type == "linear_attention" else ("self_attn", "qwen_full")


def attention_gate(projection, x, heads, head_dim):
    """Freeze the output gate from the captured query projection."""
    projected = projection.reshape(1, x.shape[1], heads, head_dim * 2)
    return projected.chunk(2, dim=-1)[1].reshape(1, x.shape[1], -1).sigmoid().detach()


def capture_frozen_inputs(block, index, recorded, install, pre, out):
    """Install Qwen projection and gate hooks in the shared capture scope."""
    if block.block_type == "linear_attention":
        install(block.linear_attn.in_proj_qkv.register_forward_hook(out(f"{index}.qkv")))
        name = f"{index}.delta_norm"
        record_input = pre(name)

        def norm_input(module, args, kwargs):
            record_input(module, args, kwargs)
            recorded[name + ".gate"] = (args[1] if len(args) > 1 else kwargs["gate"]).detach()

        install(block.linear_attn.norm.register_forward_pre_hook(norm_input, with_kwargs=True))
    else:
        install(block.self_attn.q_proj.register_forward_hook(out(f"{index}.q_gate")))


def frozen_mixer(block, index, recorded, pattern, output, *, attention, delta):
    """Construct the Qwen value path with the shared affine output."""
    x = recorded[f"{index}.input_norm"]
    if block.block_type == "linear_attention":
        return delta(block.linear_attn, x, pattern, recorded[f"{index}.qkv"],
                     recorded[f"{index}.delta_norm"], recorded[f"{index}.delta_norm.gate"], output=output)
    return attention(block.self_attn, x, pattern,
                     gate_projection=recorded.get(f"{index}.q_gate"), output=output)


def _delta_pattern(module, values):
    """Compute the frozen signed value operator from the actual projection outputs.

    The functional convolution bypasses module hooks in Transformers. Repeat that convolution
    from its captured input and weights. The recurrence below transports value coefficients.
    It uses the same float32 normalization, decay, and beta as the installed DeltaNet kernels.
    """
    import torch
    from torch.nn import functional as F
    from transformers.activations import ACT2FN

    qkv = values["qkv"].transpose(1, 2)
    length = qkv.shape[-1]
    qkv = F.conv1d(qkv.to(module.conv1d.weight.dtype), module.conv1d.weight,
                   module.conv1d.bias, padding=module.conv_kernel_size - 1,
                   groups=module.conv_dim)[..., :length]
    qkv = ACT2FN[module.activation](qkv).to(values["qkv"].dtype).transpose(1, 2)
    q, k, _ = qkv.split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
    batch = q.shape[0]
    q, k = [x.reshape(batch, length, module.num_k_heads, module.head_k_dim)
            .transpose(1, 2).float() for x in (q, k)]
    q, k = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    repeat = module.num_v_heads // module.num_k_heads
    q, k = [x.repeat_interleave(repeat, dim=1) for x in (q, k)]
    q = q * module.head_k_dim ** -0.5
    beta = values["b"].sigmoid().transpose(1, 2).float()
    decay = (-module.A_log.float().exp() * F.softplus(values["a"].float() + module.dt_bias))
    decay = decay.transpose(1, 2).exp()
    # Column j of state is the coefficient of value j in the recurrent state.
    state = q.new_zeros(batch, module.num_v_heads, module.head_k_dim, length)
    rows = []
    for t in range(length):
        state = state * decay[:, :, t, None, None]
        kt = k[:, :, t, :, None]
        memory = (state * kt).sum(-2)
        delta = -memory * beta[:, :, t, None]
        delta[:, :, t] += beta[:, :, t]
        state = state + kt * delta.unsqueeze(-2)
        rows.append((state * q[:, :, t, :, None]).sum(-2))
    return torch.stack(rows, dim=-2)


def capture_delta(self, probe, module, lifecycle_lock, pattern):
    """Capture DeltaNet projections under the shared lifecycle lock."""
    values = self._pending[probe.point] = {}

    def start(mod, args, kwargs=None):
        with lifecycle_lock():
            if self._closed:
                return
            kwargs = kwargs or {}
            values.clear()
            cache = kwargs.get("cache", kwargs.get("cache_params", args[1] if len(args) > 1 else None))
            if cache is not None and cache.get_conv_state(mod.layer_idx, state_idx=0) is not None:
                raise ValueError("DeltaNet pattern capture requires a complete prompt without a populated cache")
            if kwargs.get("cu_seq_lens_q") is not None:
                raise ValueError("DeltaNet pattern capture does not support packed sequences")

    self._handles.append(module.register_forward_pre_hook(start, with_kwargs=True))
    for name, child in (("qkv", module.in_proj_qkv), ("a", module.in_proj_a), ("b", module.in_proj_b)):
        def keep(mod, args, output, name=name):
            with lifecycle_lock():
                if not self._closed:
                    values[name] = output.detach()
        self._handles.append(child.register_forward_hook(keep))

    def finish(mod, args, output):
        import torch
        with lifecycle_lock():
            if self._closed:
                return
            observed = dict(values)
            values.clear()
        # Do not hold the lifecycle lock through the tensor computation.
        with torch.no_grad():
            self._save(probe.point, pattern(mod, observed))
    self._handles.append(module.register_forward_hook(finish))


class FrozenDeltaNet:
    """Linear value projection, causal convolution, and signed recurrence.

    The captured operator fixes queries, keys, decay, and beta. Value SiLU
    retains its linear input with a fixed sigmoid. Output RMS and SiLU gates
    use their recorded factors. No routing gradient enters this path.
    """

    def __init__(self, module, x, pattern, qkv, norm_input, output_gate, *, output, output_factory, bias):
        import torch
        from torch.nn import functional as F

        if module.activation not in ("silu", "swish"):
            raise NotImplementedError("DeltaNet tracing requires the SiLU value gate")
        start = 2 * module.key_dim
        self.value_dim = module.value_dim
        self.head_dim = module.head_v_dim
        self.heads = module.num_v_heads
        self.kernel = module.conv_kernel_size
        self.v_weight = module.in_proj_qkv.weight.detach()[start:]
        projection_bias = bias(module.in_proj_qkv)
        self.v_bias = projection_bias[start:] if projection_bias is not None else None
        self.conv_weight = module.conv1d.weight.detach()[start:]
        conv_bias = bias(module.conv1d)
        self.conv_bias = conv_bias[start:] if conv_bias is not None else None
        self.output = output if output is not None else output_factory(module.out_proj)
        self.pattern = pattern.detach()
        length = x.shape[1]
        with torch.no_grad():
            projected = qkv[..., start:].transpose(1, 2).to(self.conv_weight.dtype)
            convolved = F.conv1d(projected, self.conv_weight, self.conv_bias,
                                 padding=self.kernel - 1, groups=self.value_dim)[..., :length]
            self.value_gate = convolved.sigmoid().transpose(1, 2).detach()
            core = norm_input.reshape(1, length, self.heads, self.head_dim).float()
            self.norm_scale = torch.rsqrt(core.square().mean(-1, keepdim=True) + module.norm.variance_epsilon)
            self.norm_weight = module.norm.weight.detach()
            self.output_gate = F.silu(output_gate.float()).reshape(1, length, self.heads, self.head_dim).detach()
            self.offset = self._apply(torch.zeros_like(x), True).detach()
            self.intervention_offset = self._apply(torch.zeros_like(x), False, True).detach()

    def _apply(self, x, biases, intervention=False):
        from torch.nn import functional as F

        batch, length, _ = x.shape
        values = F.linear(x, self.v_weight, self.v_bias if biases else None).transpose(1, 2)
        values = F.conv1d(values.to(self.conv_weight.dtype), self.conv_weight,
                          self.conv_bias if biases else None, padding=self.kernel - 1,
                          groups=self.value_dim)[..., :length].transpose(1, 2)
        values = (values * self.value_gate).to(x.dtype)
        values = values.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        core = (self.pattern @ values.float()).transpose(1, 2).to(x.dtype)
        normalized = (core.float() * self.norm_scale).to(x.dtype) * self.norm_weight
        result = (normalized * self.output_gate).to(x.dtype).reshape(batch, length, self.value_dim)
        return self.output(result, biases, intervention)

    def __call__(self, x, bias=1., intervention=1.):
        return self._apply(x, False) + bias * self.offset + intervention * self.intervention_offset
