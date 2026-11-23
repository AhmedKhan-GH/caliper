"""Model probes, sparse features, and circuit attribution for Caliper LIB.

The package imports each implementation only when its API is used.
"""

__all__ = ["points", "capture", "publish", "load_dictionary", "Tracer", "Graph"]


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


def load_dictionary(path, *, model=None, model_hash=None, device=None, dtype=None):
    """Load a dictionary after its model hash matches the supplied model identity."""
    from ._dictionary import load_dictionary as implementation
    return implementation(path, model=model, model_hash=model_hash, device=device, dtype=dtype)


def __getattr__(name):
    if name == "Tracer":
        from ._trace import Tracer
        return Tracer
    if name == "Graph":
        from ._graph import Graph
        return Graph
    raise AttributeError(name)
