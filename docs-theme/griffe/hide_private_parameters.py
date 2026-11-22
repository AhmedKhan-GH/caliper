"""Griffe extension for the Python API reference: drops parameters whose names start with an underscore.

A few public calls take a private keyword for tests (``caliper.show(..., _raw=True)``). They stay in the code and
out of the rendered signatures. mkdocs.yml loads this file through the ``extensions`` option of mkdocstrings-python.
"""
import griffe


class HidePrivateParameters(griffe.Extension):
    """Removes ``_name`` parameters from every function and method that Griffe loads."""

    def on_function_instance(self, *, func: griffe.Function, **kwargs) -> None:
        kept = [p for p in func.parameters if not p.name.startswith("_")]
        if len(kept) != len(func.parameters):
            func.parameters = griffe.Parameters(*kept)
