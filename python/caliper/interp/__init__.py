"""Model probes, sparse features, and circuit attribution for Caliper LIB.

The package imports each implementation only when its API is used.
"""

__all__ = ["points", "capture", "publish", "load_dictionary", "verify_hf_model", "Tracer", "Graph"]


def points(model):
    """Return the probe points of a supported decoder model."""
    from ._points import points as implementation
    return implementation(model)


def capture(model, points):
    """Return a context manager that captures points without a CPU copy."""
    from ._points import capture as implementation
    return implementation(model, points)


def publish(name, tensor):
    """Register a named tensor for read-only use by a derive half."""
    from ._publish import publish as implementation
    return implementation(name, tensor)


def load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None,
                    memory_limit_gb=12):
    """Load a matching dictionary with a default process limit of 12 GB.

    The pinned Gemma model permits an explicit limit of at most 28 GB.
    """
    from ._dictionary import load_dictionary as implementation
    return implementation(path, model=model, model_hash=model_hash, device=device, dtype=dtype,
                          memory_limit_gb=memory_limit_gb)


def verify_hf_model(model, snapshot, *, key_mapping=None, checkpoint_subset=None, memory_limit_gb=12):
    """Verify model weights against trusted local snapshot data, without a download.

    Literal ``key_mapping`` prefixes rename checkpoint keys. The named subset
    ``qwen3.5-0.8b-text`` supplies the mapping for the pinned text checkpoint and
    declares its exact non-text keys. The default requires the full checkpoint.
    This check does not authenticate the snapshot against a remote source.
    """
    from ._dictionary import verify_hf_model as implementation
    return implementation(model, snapshot, key_mapping=key_mapping,
                          checkpoint_subset=checkpoint_subset, memory_limit_gb=memory_limit_gb)


def __getattr__(name):
    if name == "Tracer":
        from ._trace import Tracer
        return Tracer
    if name == "Graph":
        from ._graph import Graph
        return Graph
    raise AttributeError(name)
