"""Live interpretability publication and attribution bridges for Caliper LIB."""

__all__ = ["publish", "Graph"]


def publish(name, tensor):
    """Register a named tensor for read-only use by a derive half."""
    from ._publish import publish as implementation
    return implementation(name, tensor)


def __getattr__(name):
    if name == "Graph":
        from ._graph import Graph
        return Graph
    raise AttributeError(name)
