"""Frozen value paths for the fixture and the Qwen3.5 text decoder."""
from __future__ import annotations

from contextlib import ExitStack
import json

import torch
from torch.nn import functional as F

from ._points import _architecture, capture


class MemoryGuard:
    """Record MPS allocation at stage boundaries and enforce the demo limit."""

    limit = 12_000_000_000

    def __init__(self, device):
        self.device = device
        self.records = []
        if device.type == "mps":
            torch.mps.set_per_process_memory_fraction(min(1., self.limit / torch.mps.recommended_max_memory()))
            self.check("start")

    def check(self, stage):
        if self.device.type != "mps":
            return 0
        allocated = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        record = {"stage": stage, "allocated_bytes": allocated, "driver_bytes": driver,
                  "limit_bytes": self.limit}
        self.records.append(record)
        print("caliper.trace.memory " + json.dumps(record), flush=True)
        if max(allocated, driver) > self.limit:
            raise MemoryError("trace exceeded the 12 GB MPS memory limit")
        return max(allocated, driver)

    def batch_size(self, requested, *, layers, tokens, width, mixer_width):
        if self.device.type != "mps":
            return requested
        current = self.check("target batch admission")
        # Bound live values, adjoints, convolution work, and allocator headroom.
        per_target = 4 * tokens * (width * (layers + 1) * 12 + mixer_width * layers * 8)
        available = self.limit - current - 512_000_000
        admitted = min(requested, int(available // max(1, per_target)))
        if admitted < 1:
            raise MemoryError("no target batch fits the 12 GB MPS memory limit")
        return admitted


def _tensor_input(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))


def _bias(module):
    value = getattr(module, "bias", None)
    return value.detach() if value is not None else None


class FrozenNorm:
    """A recorded RMS denominator, fixed gain, and explicit affine bias."""

    def __init__(self, module, x, *, qwen=False):
        x = x.detach()
        work = x.float() if qwen else x
        scale = torch.rsqrt(work.square().mean(-1, keepdim=True) + module.eps)
        if not qwen and getattr(module, "frozen_scale", None) is not None:
            scale = module.frozen_scale.detach()
        self.scale = scale.detach()
        self.weight = (1 + module.weight.detach().float()) if qwen else module.weight.detach()
        self.bias = _bias(module)
        self.qwen = qwen

    def __call__(self, x, bias=1.):
        work = x.float() if self.qwen else x
        result = ((work * self.scale) * self.weight).to(x.dtype)
        if self.bias is not None:
            result = result + bias * self.bias
        return result


class FrozenAttention:
    """Linear values with fixed softmax and optional Qwen output gate."""

    def __init__(self, module, x, pattern, *, gate_projection=None):
        self.pattern = pattern.detach()
        self.head_dim = module.head_dim
        self.heads = pattern.shape[1]
        self.v_weight = module.v_proj.weight.detach()
        self.v_bias = _bias(module.v_proj)
        self.o_weight = module.o_proj.weight.detach()
        self.o_bias = _bias(module.o_proj)
        self.gate = None
        if gate_projection is not None:
            projected = gate_projection.reshape(1, x.shape[1], self.heads, self.head_dim * 2)
            self.gate = projected.chunk(2, dim=-1)[1].reshape(1, x.shape[1], -1).sigmoid().detach()
        with torch.no_grad():
            self.offset = self._apply(torch.zeros_like(x), True).detach()

    def _apply(self, x, biases):
        batch, length, _ = x.shape
        values = F.linear(x, self.v_weight, self.v_bias if biases else None)
        values = values.reshape(batch, length, -1, self.head_dim).transpose(1, 2)
        values = values.repeat_interleave(self.heads // values.shape[1], dim=1)
        result = (self.pattern @ values).transpose(1, 2).reshape(batch, length, -1)
        if self.gate is not None:
            result = result * self.gate
        return F.linear(result, self.o_weight, self.o_bias if biases else None)

    def __call__(self, x, bias=1.):
        return self._apply(x, False) + bias * self.offset


class FrozenDeltaNet:
    """Linear value projection, causal convolution, and signed recurrence.

    The captured operator fixes queries, keys, decay, and beta. Value SiLU
    retains its linear input with a fixed sigmoid. Output RMS and SiLU gates
    use their recorded factors. No routing gradient enters this path.
    """

    def __init__(self, module, x, pattern, qkv, norm_input, output_gate):
        if module.activation not in ("silu", "swish"):
            raise NotImplementedError("DeltaNet tracing requires the SiLU value gate")
        start = 2 * module.key_dim
        self.value_dim = module.value_dim
        self.head_dim = module.head_v_dim
        self.heads = module.num_v_heads
        self.kernel = module.conv_kernel_size
        self.v_weight = module.in_proj_qkv.weight.detach()[start:]
        projection_bias = _bias(module.in_proj_qkv)
        self.v_bias = projection_bias[start:] if projection_bias is not None else None
        self.conv_weight = module.conv1d.weight.detach()[start:]
        conv_bias = _bias(module.conv1d)
        self.conv_bias = conv_bias[start:] if conv_bias is not None else None
        self.o_weight = module.out_proj.weight.detach()
        self.o_bias = _bias(module.out_proj)
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

    def _apply(self, x, biases):
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
        return F.linear(result, self.o_weight, self.o_bias if biases else None)

    def __call__(self, x, bias=1.):
        return self._apply(x, False) + bias * self.offset


class FrozenPath:
    """Prompt-local affine residual path with independent MLP output writes.

    ``run`` never calls an MLP or a dictionary encoder. Feature targets read
    recorded MLP inputs. Its write arguments are independent source leaves.
    """

    def __init__(self, model, input_ids, dictionary, *, guard=None):
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.guard = guard or MemoryGuard(self.device)
        if model.training:
            raise ValueError("trace requires an evaluation model")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("trace requires float32 or float64 model weights")
        probes = _architecture(model)
        self.qwen = model.config.model_type == "qwen3_5_text"
        if self.qwen and self.dtype != torch.float32:
            raise ValueError("Qwen tracing requires float32 model weights")
        modules = dict(model.named_modules())
        prefix = next(iter(probes)).rsplit(".", 2)[0]
        self.blocks = modules[prefix]
        root = model.model if self.qwen else model
        if len(dictionary.layers) != len(self.blocks):
            raise ValueError("dictionary layer count differs from the model")
        self.layers = len(self.blocks)
        self.tokens = input_ids.shape[1]
        self.width = model.config.hidden_size
        self.mixer_width = max((getattr(getattr(b, "linear_attn", None), "value_dim", self.width)
                                for b in self.blocks), default=self.width)
        recorded = {}
        self.norms, self.post_norms, self.mixers = [], [], []
        with ExitStack() as stack:
            values = stack.enter_context(capture(model, list(probes)))
            def pre(name):
                def hook(mod, args, kwargs):
                    recorded[name] = _tensor_input(args, kwargs).detach()
                    if name.endswith(".delta_norm"):
                        recorded[name + ".gate"] = (args[1] if len(args) > 1 else kwargs["gate"]).detach()
                return hook
            def out(name):
                def hook(mod, args, result):
                    recorded[name] = result.detach()
                return hook
            def install(handle):
                stack.callback(handle.remove)
            frames = []
            def enter_forward(mod, args):
                # Clean patch sources can run a nested model forward inside an intervention.
                frames.append((dict(values), {key: dict(value) for key, value in values._pending.items()},
                               dict(recorded)) if frames else None)
            def exit_forward(mod, args, output):
                frame = frames.pop()
                if frame is not None:
                    saved_values, saved_pending, saved_recorded = frame
                    values.clear()
                    values.update(saved_values)
                    for key, pending in values._pending.items():
                        pending.clear()
                        pending.update(saved_pending[key])
                    recorded.clear()
                    recorded.update(saved_recorded)
            install(model.register_forward_pre_hook(enter_forward, prepend=True))
            install(model.register_forward_hook(exit_forward, always_call=True))
            install(root.norm.register_forward_pre_hook(pre("final_norm"), with_kwargs=True))
            install(root.embed_tokens.register_forward_hook(out("embedding")))
            for i, block in enumerate(self.blocks):
                install(block.input_layernorm.register_forward_pre_hook(pre(f"{i}.input_norm"), with_kwargs=True))
                install(block.post_attention_layernorm.register_forward_pre_hook(pre(f"{i}.post_norm"), with_kwargs=True))
                if self.qwen and block.block_type == "linear_attention":
                    install(block.linear_attn.in_proj_qkv.register_forward_hook(out(f"{i}.qkv")))
                    install(block.linear_attn.norm.register_forward_pre_hook(pre(f"{i}.delta_norm"), with_kwargs=True))
                elif self.qwen:
                    install(block.self_attn.q_proj.register_forward_hook(out(f"{i}.q_gate")))
                def stage_hook(*_, i=i):
                    self.guard.check(f"record layer {i}")
                install(block.register_forward_hook(stage_hook))
            with torch.no_grad():
                kwargs = {"use_cache": False, "logits_to_keep": 1} if self.qwen else {}
                self.logits = model(input_ids=input_ids, **kwargs).logits[:, -1].detach()
        self.guard.check("record complete")
        self.embedding = recorded["embedding"]
        self.inputs, self.outputs, self.features, self.errors, self.decoder_biases = [], [], [], [], []
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                base = f"{prefix}.{i}"
                x = values[f"{base}.mlp_in"]
                y = values[f"{base}.mlp_out"]
                layer = dictionary.layers[i]
                if layer.W_enc.shape[1] != self.width or layer.W_dec.shape != layer.W_enc.shape:
                    raise ValueError("dictionary width or decoder shape differs from the model")
                if layer.W_enc.device != self.device:
                    raise ValueError("dictionary and model must share one device")
                a = dictionary.encode(i, x.to(layer.W_enc.dtype)).detach()
                reconstruction = dictionary.decode(i, a).to(self.dtype)
                self.inputs.append(x)
                self.outputs.append(y)
                self.features.append(a.to(self.dtype))
                self.errors.append((y - reconstruction).detach())
                self.decoder_biases.append(layer.b_dec.detach().to(self.dtype))
                self.norms.append(FrozenNorm(block.input_layernorm, recorded[f"{i}.input_norm"], qwen=self.qwen))
                self.post_norms.append(FrozenNorm(block.post_attention_layernorm, recorded[f"{i}.post_norm"], qwen=self.qwen))
                pattern = values[f"{base}.attn_pattern"]
                if self.qwen and block.block_type == "linear_attention":
                    mixer = FrozenDeltaNet(block.linear_attn, recorded[f"{i}.input_norm"], pattern,
                                           recorded[f"{i}.qkv"], recorded[f"{i}.delta_norm"],
                                           recorded[f"{i}.delta_norm.gate"])
                else:
                    mixer = FrozenAttention(block.self_attn, recorded[f"{i}.input_norm"], pattern,
                                            gate_projection=recorded.get(f"{i}.q_gate"))
                self.mixers.append(mixer)
                self.guard.check(f"freeze layer {i}")
            self.final_norm = FrozenNorm(root.norm, recorded["final_norm"], qwen=self.qwen)
            head = model.get_output_embeddings()
            self.unembedding = head.weight.detach()
            self.unembedding_mean = self.unembedding.mean(0)
            self.unembedding_bias = _bias(head)
            self.writes = [(y - b).detach() for y, b in zip(self.outputs, self.decoder_biases)]
            reads, final = self.run(self.embedding, self.writes)
            # Compare every read and all next-token logits, before any edge calculation.
            self.forward_residual = max([float((a - b).abs().max()) for a, b in zip(reads, self.inputs)] +
                                        [float((F.linear(final[:, -1], self.unembedding,
                                                         self.unembedding_bias) - self.logits).abs().max())])
            self.direct_forward_residual = self.forward_residual
            replacement_reads, replacement_final = self.replacement(self.embedding, dictionary)
            self.replacement_forward_residual = max(
                max(float((a - b).abs().max()) for a, b in zip(replacement_reads, self.inputs)),
                float((F.linear(replacement_final[:, -1], self.unembedding,
                                self.unembedding_bias) - self.logits).abs().max()))
            self.forward_residual = max(self.direct_forward_residual, self.replacement_forward_residual)
            tolerance = 1e-9 if self.dtype == torch.float64 else 2e-2
            if not self.forward_residual <= tolerance:
                raise ValueError(f"frozen replacement forward mismatch: {self.forward_residual:.6g}")
            self.guard.check("forward agreement")
            # Three independent bias families. Encoder and unembedding biases are target-local.
            zero = torch.zeros_like(self.embedding).expand(3, -1, -1)
            masks = torch.eye(3, device=self.device, dtype=self.dtype).reshape(3, 3, 1, 1)
            self.bias_reads, self.bias_final = self.run(zero, [zero] * self.layers, bias_masks=masks)

    def run(self, embedding, writes, *, bias_masks=None):
        """Return MLP reads and final normalized residual for independent writes.

        Bias mask rows are decoder, normalization, and attention contributions.
        A zero mask gives the strictly linear residual path.
        """
        decoder, norm, attention = (1., 1., 1.) if bias_masks is None else bias_masks
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer, write) in enumerate(zip(self.norms, self.post_norms, self.mixers, writes)):
            x = x + mixer(input_norm(x, norm), attention)
            reads.append(post_norm(x, norm))
            x = x + write + decoder * self.decoder_biases[i]
        return reads, self.final_norm(x, norm)


    def replacement(self, embedding, dictionary):
        """Evaluate transcoders with the recorded active masks and constant errors.

        This affine forward verifies the complete local replacement. Direct
        edges use ``run`` with independent writes and never differentiate here.
        """
        x = embedding
        reads = []
        for i, (input_norm, post_norm, mixer) in enumerate(zip(self.norms, self.post_norms, self.mixers)):
            x = x + mixer(input_norm(x))
            read = post_norm(x)
            reads.append(read)
            layer = dictionary.layers[i]
            indices = self.features[i][0].topk(layer.k, dim=-1).indices
            mask = self.features[i][0].gather(-1, indices) > 0
            encoder = layer.W_enc.detach()[indices].to(self.dtype)
            decoder = layer.W_dec.detach()[indices].to(self.dtype)
            activation = (torch.einsum("btw,tkw->btk", read, encoder) +
                          layer.b_enc.detach()[indices].to(self.dtype)) * mask
            x = x + torch.einsum("btk,tkw->btw", activation, decoder) + self.decoder_biases[i] + self.errors[i]
            self.guard.check(f"replacement layer {i}")
        return reads, self.final_norm(x)
