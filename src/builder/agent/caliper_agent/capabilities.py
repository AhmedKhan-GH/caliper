"""The cache of probe results, the person's overrides, and what the sidecar assumes of a model.

The cache is one small JSON file, `model-capabilities.json`, in a folder that
the app gives with `--cache-dir` (the app's settings folder; never a folder
that holds a key, and the file never holds one). It keeps one entry per base
URL and model:

    {"version": 1, "entries": [
      {"base_url": "http://localhost:11434/v1", "model": "example-model",
       "probed_at": 1763251200.0, "used_at": 1763251200.0,
       "result": {"tools": {"value": true, "source": "probe"}, "vision": {...},
                  "context_length": {...}, "quirks": {"strict": true, ...}},
       "overrides": {"vision": false, "context_length": 32768}}]}

- **Bounded.** At most 256 entries (the least recently used goes first), a
  file of at most 1 MiB, ids and URLs of bounded length, and only the
  capabilities and the quirks: the list of models is not kept.
- **An expiry.** A probe result older than 30 days is stale: the sidecar no
  longer relies on it, and `caliper-agent probe` runs the probe again. The
  person's overrides do not expire; they stay until the person clears them.
- **Corruption.** A file that cannot be read, is too large, is not JSON or is
  not version 1 counts as empty, with a warning, and the next save replaces
  it; an entry that does not check is dropped. A save writes a new file beside
  the old one and renames it over it, readable by the user only.
- **No secret.** A base URL with a user name, a password, a query or a
  fragment is refused before it reaches the cache, and a save that would
  contain a registered secret is not written.

`resolve()` gives what the sidecar assumes of a model, in this order: an
option on the sidecar's command line (the app passing an answer directly),
the person's override, a probe result that is not stale, then the defaults:
tool calling and vision assumed until the server says otherwise, and a
context length of 8,192 tokens for an OpenAI-compatible model.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .keyguard import SCRUBBER
from .probe import MAX_ID_CHARS, Capability, ProbeResult

CACHE_FILE = "model-capabilities.json"
CACHE_VERSION = 1
MAX_ENTRIES = 256
MAX_FILE_BYTES = 1 << 20
MAX_URL_CHARS = 2048
TTL_S = 30 * 24 * 3600
DEFAULT_CONTEXT_LENGTH = 8192
MAX_CONTEXT_LENGTH = 100_000_000
CAPABILITY_NAMES = ("tools", "vision", "context_length")
QUIRK_KEYS = ("strict", "max_tokens_field", "reasoning_field", "streamed_arguments")
# What the probe learned of the model's memory: the bytes of context cache one
# token takes, from an Ollama server's model metadata (probe.py).
FACT_KEYS = ("kv_bytes_per_token",)


class CacheError(ValueError):
    """A request the cache refuses; the message is safe to show."""


def _valid_value(name: str, value: Any) -> bool:
    if value is None:
        return True
    if name == "context_length":
        return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_CONTEXT_LENGTH
    return isinstance(value, bool)


def _clean_quirks(doc: Any) -> dict:
    if not isinstance(doc, dict):
        return {}
    out = {}
    for key in QUIRK_KEYS:
        value = doc.get(key)
        if isinstance(value, bool) or (isinstance(value, str) and len(value) <= 32):
            out[key] = value
    for key in FACT_KEYS:
        value = doc.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < (1 << 30):
            out[key] = value
    return out


@dataclass
class Entry:
    base_url: str
    model: str
    probed_at: float | None = None
    used_at: float = field(default_factory=time.time)
    result: dict = field(default_factory=dict)       # capability name -> Capability, plus "quirks"
    overrides: dict = field(default_factory=dict)    # capability name -> value

    def stale(self, now: float | None = None) -> bool:
        return self.probed_at is None or (now if now is not None else time.time()) - self.probed_at > TTL_S

    def as_dict(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "probed_at": self.probed_at, "used_at": self.used_at,
                "result": {k: (v.as_dict() if isinstance(v, Capability) else v) for k, v in self.result.items()},
                "overrides": dict(self.overrides)}

    @staticmethod
    def from_result(result: ProbeResult) -> "Entry":
        return Entry(result.base_url, result.model, result.probed_at, time.time(),
                     {"tools": result.tools, "vision": result.vision, "context_length": result.context_length,
                      "quirks": _clean_quirks(result.quirks)})

    @staticmethod
    def from_dict(doc: Any) -> "Entry | None":
        if not isinstance(doc, dict):
            return None
        base_url, model = doc.get("base_url"), doc.get("model")
        if not (isinstance(base_url, str) and 0 < len(base_url) <= MAX_URL_CHARS
                and isinstance(model, str) and 0 < len(model) <= MAX_ID_CHARS):
            return None
        probed_at = doc.get("probed_at")
        used_at = doc.get("used_at")
        if probed_at is not None and not isinstance(probed_at, (int, float)):
            return None
        result: dict = {}
        raw = doc.get("result") if isinstance(doc.get("result"), dict) else {}
        for name in CAPABILITY_NAMES:
            cap = Capability.from_dict(raw.get(name))
            if _valid_value(name, cap.value):
                result[name] = cap
        result["quirks"] = _clean_quirks(raw.get("quirks"))
        overrides = {}
        raw_over = doc.get("overrides") if isinstance(doc.get("overrides"), dict) else {}
        for name in CAPABILITY_NAMES:
            if name in raw_over and raw_over[name] is not None and _valid_value(name, raw_over[name]):
                overrides[name] = raw_over[name]
        return Entry(base_url, model, float(probed_at) if probed_at is not None else None,
                     float(used_at) if isinstance(used_at, (int, float)) else 0.0, result, overrides)


class ProbeCache:
    """The cache file in one folder. Every method is safe on a missing or broken file."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.path = self.folder / CACHE_FILE
        self.warnings: list[str] = []
        self._entries: list[Entry] = self._read()

    def _read(self) -> list[Entry]:
        try:
            with self.path.open("rb") as f:
                data = f.read(MAX_FILE_BYTES + 1)
        except FileNotFoundError:
            return []
        except OSError as exc:
            self.warnings.append(f"The capability cache cannot be read ({exc.strerror or 'error'}); it is ignored.")
            return []
        if len(data) > MAX_FILE_BYTES:
            self.warnings.append("The capability cache is larger than 1 MiB; it is ignored and will be replaced.")
            return []
        try:
            doc = json.loads(data)
        except (ValueError, RecursionError):
            self.warnings.append("The capability cache is not JSON; it is ignored and will be replaced.")
            return []
        if not isinstance(doc, dict) or doc.get("version") != CACHE_VERSION or not isinstance(doc.get("entries"), list):
            self.warnings.append("The capability cache is not version 1; it is ignored and will be replaced.")
            return []
        entries = []
        for raw in doc["entries"][:MAX_ENTRIES]:
            entry = Entry.from_dict(raw)
            if entry is None:
                self.warnings.append("An entry of the capability cache was malformed and was dropped.")
            else:
                entries.append(entry)
        return entries

    # ---- lookups -------------------------------------------------------------------------------------

    def get(self, base_url: str, model: str) -> Entry | None:
        return next((e for e in self._entries if e.base_url == base_url and e.model == model), None)

    def entries(self) -> list[Entry]:
        return list(self._entries)

    # ---- changes -------------------------------------------------------------------------------------

    def _entry(self, base_url: str, model: str) -> Entry:
        check_cache_key(base_url, model)
        entry = self.get(base_url, model)
        if entry is None:
            entry = Entry(base_url, model)
            self._entries.append(entry)
        entry.used_at = time.time()
        return entry

    def put_result(self, result: ProbeResult) -> Entry:
        entry = self._entry(result.base_url, result.model)
        entry.probed_at = result.probed_at
        entry.result = Entry.from_result(result).result
        return entry

    def set_override(self, base_url: str, model: str, name: str, value: Any) -> Entry:
        if name not in CAPABILITY_NAMES:
            raise CacheError(f"No capability is called {name[:32]}; use tools, vision or context_length.")
        if not _valid_value(name, value):
            raise CacheError(f"{name} cannot be set to that value.")
        entry = self._entry(base_url, model)
        if value is None:
            entry.overrides.pop(name, None)
        else:
            entry.overrides[name] = value
        return entry

    def save(self) -> bool:
        """Writes the file; False (with a warning) when it is not written."""
        self._entries.sort(key=lambda e: e.used_at, reverse=True)
        del self._entries[MAX_ENTRIES:]
        text = json.dumps({"version": CACHE_VERSION, "entries": [e.as_dict() for e in self._entries]},
                          indent=1, sort_keys=True)
        while len(text.encode("utf-8")) > MAX_FILE_BYTES and self._entries:
            self._entries.pop()
            text = json.dumps({"version": CACHE_VERSION, "entries": [e.as_dict() for e in self._entries]},
                              indent=1, sort_keys=True)
        if SCRUBBER.contains_secret(text):
            self.warnings.append("The capability cache was not written, because it would have held a secret.")
            return False
        try:
            self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp = tempfile.mkstemp(prefix=".model-capabilities.", suffix=".tmp", dir=self.folder)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            self.warnings.append(f"The capability cache cannot be written ({exc.strerror or 'error'}).")
            return False
        return True


def check_cache_key(base_url: str, model: str) -> None:
    from urllib.parse import urlsplit
    if not isinstance(model, str) or not model or len(model) > MAX_ID_CHARS or any(ord(c) < 0x20 for c in model):
        raise CacheError("A model id is 1 to 256 printable characters.")
    if not isinstance(base_url, str) or len(base_url) > MAX_URL_CHARS:
        raise CacheError("A base URL is at most 2,048 characters.")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CacheError("A base URL is an http or https address.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise CacheError("A base URL in the cache has no user name, password, query or fragment.")


# ---- what the sidecar assumes ----------------------------------------------------------------------


@dataclass
class Resolved:
    """What the sidecar assumes of one model, with where each answer came from."""

    tools: Capability
    vision: Capability
    context_length: Capability
    quirks: dict = field(default_factory=dict)

    def as_meta(self, model: str) -> dict:
        """The `_meta.caliper.model` of `session/new`'s answer."""
        return {"id": model, "tools": bool(self.tools.value), "vision": bool(self.vision.value),
                "context_length": self.context_length.value, "context_source": self.context_length.source,
                "sources": {"tools": self.tools.source, "vision": self.vision.source}}

    def as_dict(self) -> dict:
        return {"tools": self.tools.as_dict(), "vision": self.vision.as_dict(),
                "context_length": self.context_length.as_dict()}


def resolve(entry: Entry | None, *, options: dict | None = None, default_context: int | None = DEFAULT_CONTEXT_LENGTH,
            now: float | None = None) -> Resolved:
    """The answers for one model; see the module's docstring for the order."""
    options = options or {}
    fresh = entry is not None and not entry.stale(now)
    out = {}
    for name in CAPABILITY_NAMES:
        if options.get(name) is not None:
            out[name] = Capability(options[name], "option")
            continue
        if entry is not None and name in entry.overrides:
            out[name] = Capability(entry.overrides[name], "override")
            continue
        cap = entry.result.get(name) if fresh else None
        if isinstance(cap, Capability) and cap.value is not None:
            out[name] = Capability(cap.value, cap.source, cap.detail)
            continue
        if name == "context_length":
            out[name] = Capability(default_context, "default" if default_context is not None else "unknown")
        else:
            out[name] = Capability(True, "default", "Assumed until the server says otherwise.")
    quirks = dict(entry.result.get("quirks", {})) if fresh and entry is not None else {}
    return Resolved(out["tools"], out["vision"], out["context_length"], quirks)


def provider_capabilities(resolved: Resolved) -> "Capabilities":
    """The provider's view of resolved answers (provider.Capabilities)."""
    from .provider import Capabilities
    return Capabilities(tools=bool(resolved.tools.value), vision=bool(resolved.vision.value),
                        context_length=resolved.context_length.value,
                        sources={"tools": resolved.tools.source, "vision": resolved.vision.source,
                                 "context_length": resolved.context_length.source})


def claude_facts(cache: "ProbeCache | None", base_url: str, context_option: int | None = None):
    """For the Claude provider: every model calls tools; vision and the context
    length come from the Models API as the probe cached it, or an override."""
    def facts(model: str):
        entry = cache.get(base_url, model) if cache is not None else None
        resolved = resolve(entry, options={"context_length": context_option}, default_context=None)
        caps = provider_capabilities(resolved)
        caps.tools = True
        caps.sources["tools"] = "known"
        if resolved.vision.source == "default":
            caps.sources["vision"] = "known"
        return caps
    return facts
