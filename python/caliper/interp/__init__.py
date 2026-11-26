"""Live interpretability publication and attribution bridges for Caliper LIB."""

__all__ = [
    "publish",
    "Graph",
    "Tracer",
    "LiveFeatureIndex",
    "patch_sweep",
    "load_backend",
    "make_tiny_backend",
    "make_tiny_sae_backend",
]


def publish(name, tensor):
    """Register a named tensor for read-only use by a derive half."""
    from ._publish import publish as implementation
    return implementation(name, tensor)


def __getattr__(name):
    if name in ("Graph", "Tracer"):
        from ._graph import Graph, Tracer
        return {"Graph": Graph, "Tracer": Tracer}[name]
    if name == "LiveFeatureIndex":
        from ._index import LiveFeatureIndex
        return LiveFeatureIndex
    if name == "patch_sweep":
        from ._sweep import patch_sweep
        return patch_sweep
    if name in ("load_backend", "make_tiny_backend", "make_tiny_sae_backend"):
        from ._backend import load_backend, make_tiny_backend, make_tiny_sae_backend
        return {
            "load_backend": load_backend,
            "make_tiny_backend": make_tiny_backend,
            "make_tiny_sae_backend": make_tiny_sae_backend,
        }[name]
    raise AttributeError(name)
