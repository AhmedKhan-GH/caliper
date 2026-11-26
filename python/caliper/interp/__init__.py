"""Live interpretability publication and attribution bridges for Caliper LIB."""

__all__ = ["publish", "Graph", "Tracer", "load_backend", "make_tiny_backend", "make_tiny_sae_backend"]


def publish(name, tensor):
    """Register a named tensor for read-only use by a derive half."""
    from ._publish import publish as implementation
    return implementation(name, tensor)


def __getattr__(name):
    if name in ("Graph", "Tracer"):
        from ._graph import Graph, Tracer
        return {"Graph": Graph, "Tracer": Tracer}[name]
    if name in ("load_backend", "make_tiny_backend", "make_tiny_sae_backend"):
        from ._backend import load_backend, make_tiny_backend, make_tiny_sae_backend
        return {
            "load_backend": load_backend,
            "make_tiny_backend": make_tiny_backend,
            "make_tiny_sae_backend": make_tiny_sae_backend,
        }[name]
    raise AttributeError(name)
