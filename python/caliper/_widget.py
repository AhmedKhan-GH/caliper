"""caliper.widget(): one live view of the stream, inline in a notebook cell.

The widget is an anywidget whose browser module is the packaged
``caliper-widget.js``. It draws from the stream socket directly: the ``src``
trait is the ``ws://`` URL of the running server with the session token, and
the browser opens that socket itself. Jupyter's message channel carries only
the traits:

* ``src``, ``view``, ``kind``, ``height``: what the browser shows;
* ``data_view``: the data view (dict), set from Python or picked in the browser;
* ``selection``: the last selection the browser reported (dict), as
  ``{"view": ..., "type": "select" | "pick" | "hover", "detail": {...}}``.

``caliper.controls()`` is a second widget, the panel of the script's run
controls, parameters and actions (wire protocol 3.1). Its browser module is
the packaged ``caliper-controls-widget.js``, loaded when the first panel is
made; its traits are ``src`` and ``height``, and everything else travels on
the stream.

Needs the notebook extra: ``pip install 'caliper[notebook]'``.
"""
from __future__ import annotations

import sys
import warnings
from urllib.parse import quote

try:
    import anywidget
    import traitlets
except ImportError as e:
    raise ImportError(
        "caliper.widget() needs anywidget, which the notebook extra installs: "
        "pip install 'caliper[notebook]'") from e

from . import _assets

KINDS = ("series", "heatmap", "image", "points", "histogram", "scene")   # scene: 3.2 draft


# ---------------------------------------------------------------------------
# The seam between the widget and caliper.serve().
#
# caliper.serve() keeps its running server in the package attribute
# ``caliper._server`` (None or absent while no server runs). The widget reads
# the stream URL from that object, which has either
#     stream_url: str          the ws:// stream URL with the session token, or
#     port: int, token: str    (the URL is then ws://127.0.0.1:<port>/v3/stream?token=<token>)
# and may have ``closed: bool`` (a closed server counts as none).
# ---------------------------------------------------------------------------
def _current_stream_url() -> str | None:
    """The ws:// URL of the running server's stream, or None when none runs."""
    server = getattr(sys.modules[__package__], "_server", None)
    if server is None or getattr(server, "closed", False):
        return None
    url = getattr(server, "stream_url", None)
    if url:
        return str(url)
    port, token = getattr(server, "port", None), getattr(server, "token", None)
    if port and token:
        return f"ws://127.0.0.1:{int(port)}/v3/stream?token={quote(str(token), safe='')}"
    return None


def _stream_url() -> str:
    """The current stream URL, starting caliper.serve() when no server runs."""
    url = _current_stream_url()
    if url:
        return url
    serve = getattr(sys.modules[__package__], "serve", None)
    if serve is None:
        raise RuntimeError(
            "caliper.widget() draws from the stream server, and no server runs; "
            "caliper.serve() is not available in this build to start one")
    serve()
    url = _current_stream_url()
    if not url:
        raise RuntimeError("caliper.serve() returned, but no stream server is running "
                           "(caliper._server is not set)")
    return url


def _redact(url: str) -> str:
    head, sep, query = url.partition("?")
    if not sep:
        return url
    parts = [p.split("=", 1)[0] + "=..." if p.startswith("token=") else p
             for p in query.split("&")]
    return head + "?" + "&".join(parts)


class CaliperWidget(anywidget.AnyWidget):
    """A notebook widget that shows one live view, returned by [`widget`][caliper.widget].

    The browser draws the view. The traits sync through anywidget.

    The `detail` of an event is in source indices. For a `select` in a points view it holds
    `indices`, the positions in the points that were shown, and `row_ids`, their source indices.
    The [Wire Protocol](../reference/wire-protocol.md) reference lists the detail of every kind.

    Traits travel over the message channel of the notebook, which the kernel reads only while no
    cell runs. A selection made during a long cell reaches `selection` when the cell ends. A
    `data_view` set from Python reaches the widget at once. A data view chosen in the picker of
    the widget goes to the producer over the stream and takes effect during the cell.

    Attributes:
        src (str): The `ws://` URL of the stream, with the token.
        view (str): The name of the view.
        kind (str | None): The kind of the live view.
        height (int): Height in CSS pixels.
        data_view (dict): The data view of the own subscription of the widget, in both directions.
            The browser writes the data view that the user picked. Setting it from Python makes
            the widget request that data view.
        selection (dict): Written by the browser: the latest event it reported, as
            `{"view", "type", "detail"}`.
    """

    _esm = _assets.asset_path("caliper-widget.js")

    src = traitlets.Unicode("").tag(sync=True)
    view = traitlets.Unicode("").tag(sync=True)
    kind = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    height = traitlets.Int(320).tag(sync=True)
    data_view = traitlets.Dict().tag(sync=True)
    selection = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._select_handlers = {}

    @traitlets.validate("kind")
    def _check_kind(self, proposal):
        kind = proposal["value"]
        if kind is not None and kind not in KINDS:
            raise traitlets.TraitError(f"kind must be one of {', '.join(KINDS)} or None, "
                                       f"not {kind!r}")
        return kind

    @traitlets.validate("height")
    def _check_height(self, proposal):
        if proposal["value"] <= 0:
            raise traitlets.TraitError(f"height must be positive, not {proposal['value']}")
        return proposal["value"]

    def on_select(self, callback, remove: bool = False):
        """Call `callback(selection)` each time the browser reports a new selection.

        Args:
            callback (Callable[[dict], object]): The function to call with the new `selection`.
            remove (bool): `True` stops the calls to `callback`.

        Returns:
            Callable[[dict], object]: `callback`, so the method also works as a decorator.
        """
        if remove:
            handler = self._select_handlers.pop(callback, None)
            if handler is not None:
                self.unobserve(handler, names="selection")
            return callback
        if callback not in self._select_handlers:
            def handler(change):
                callback(change["new"])
            self._select_handlers[callback] = handler
            self.observe(handler, names="selection")
        return callback

    def __repr__(self) -> str:
        kind = self.kind if self.kind is not None else "from the stream"
        text = f"<caliper widget {self.view!r} ({kind}), height {self.height}, {_redact(self.src)}"
        if self.selection:
            text += f", selection {self.selection.get('type', '?')}"
        return text + ">"


def widget(name: str, kind: str | None = None, *, height: int = 320) -> CaliperWidget:
    """A widget showing view ``name`` of the running stream (see caliper.widget)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid view name {name!r} (non-empty str)")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)} or None, not {kind!r}")
    return CaliperWidget(src=_stream_url(), view=name, kind=kind, height=int(height))


# ---------------------------------------------------------------------------
# caliper.controls(): the panel of run controls, parameters and actions
# ---------------------------------------------------------------------------
CONTROLS_MODULE = "caliper-controls-widget.js"
# Shown instead of the panel by a build whose assets lack its module.
_CONTROLS_MISSING = (
    "export default { render({ el }) { el.textContent = 'The Caliper controls panel is not in "
    "this build: its module caliper-controls-widget.js is missing.'; } };")
_controls_classes: dict = {}


def _controls_class(esm):
    """The widget class for the module ``esm`` (a path or the note above)."""
    key = str(esm)
    cls = _controls_classes.get(key)
    if cls is None:
        class CaliperControls(anywidget.AnyWidget):
            """The run controls, parameters and actions of the stream's script,
            drawn in the browser by ``<caliper-controls>``."""
            _esm = esm
            src = traitlets.Unicode("").tag(sync=True)
            height = traitlets.Int(160).tag(sync=True)

            @traitlets.validate("height")
            def _check_height(self, proposal):
                if proposal["value"] <= 0:
                    raise traitlets.TraitError(f"height must be positive, not {proposal['value']}")
                return proposal["value"]

            def __repr__(self) -> str:
                return f"<caliper controls, height {self.height}, {_redact(self.src)}>"
        cls = _controls_classes[key] = CaliperControls
    return cls


def controls(*, height: int = 160):
    """The controls panel of the running stream (see caliper.controls). When
    the build lacks the panel's module, warns and returns a panel that says
    so, so that a notebook still runs from top to bottom."""
    try:
        esm = _assets.asset_path(CONTROLS_MODULE)
    except _assets.AssetMissing as e:
        warnings.warn(f"caliper.controls(): {e}", RuntimeWarning, stacklevel=3)
        esm = _CONTROLS_MISSING
    return _controls_class(esm)(src=_stream_url(), height=int(height))
