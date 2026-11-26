"""Decoder families and their semantic residual locations."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelFamily:
    """One supported decoder layout and its normalization convention."""

    name: str
    decoder_class: str | None
    block_class: str | None
    module_name: str | None
    unit_offset_norm: bool = False
    mlp_read_norm: str = "post_attention_layernorm"
    mlp_write_module: str = "mlp"
    attention_write_norm: str | None = None
    memory_max_gb: int = 12

    def locate(self, model):
        modules = dict(model.named_modules())
        if self.decoder_class is None:
            root_name, root = "", model
            block_type = None
        else:
            from importlib import import_module
            implementation = import_module(self.module_name)
            decoder_type = getattr(implementation, self.decoder_class)
            block_type = getattr(implementation, self.block_class)
            decoders = [(name, module) for name, module in modules.items() if isinstance(module, decoder_type)]
            if len(decoders) != 1:
                raise ValueError(f"{self.name} probes require exactly one text decoder")
            root_name, root = decoders[0]
        prefix = f"{root_name}.layers".lstrip(".")
        layers = modules.get(prefix)
        if layers is None or len(layers) != model.config.num_hidden_layers:
            raise ValueError("decoder layer paths do not match the model config")
        if block_type is not None:
            for index, block in enumerate(layers):
                if not isinstance(block, block_type):
                    raise ValueError(f"unsupported {self.name} decoder block at {prefix}.{index}")
        return root, prefix, layers

    def mixer(self, block):
        return "self_attn", "fixture" if self.name == "caliper_tiny" else "gemma_full"

    def memory_limit(self, requested):
        if (isinstance(requested, bool) or not isinstance(requested, (int, float))
                or not math.isfinite(requested) or not 0 < requested <= self.memory_max_gb):
            raise ValueError(f"{self.name} trace memory limit must be positive and at most {self.memory_max_gb} GB")
        return requested


_FAMILIES = {
    "caliper_tiny": ModelFamily("caliper_tiny", None, None, None),
    "gemma2": ModelFamily("gemma2", "Gemma2Model", "Gemma2DecoderLayer",
                          "transformers.models.gemma2.modeling_gemma2", unit_offset_norm=True,
                          mlp_read_norm="pre_feedforward_layernorm",
                          mlp_write_module="post_feedforward_layernorm",
                          attention_write_norm="post_attention_layernorm", memory_max_gb=40),
}


def model_family(model):
    """Resolve a supported family without a generic decoder fallback."""
    name = getattr(getattr(model, "config", None), "model_type", None)
    if name not in _FAMILIES:
        raise NotImplementedError(f"semantic probes do not support decoder {name or type(model).__name__!r}")
    return _FAMILIES[name]
