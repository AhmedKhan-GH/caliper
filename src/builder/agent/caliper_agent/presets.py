"""The provider presets, read from `presets.json` beside this module, and the checks of a provider choice.

`presets.json` is the one table of the presets (design/widget-agent.md,
Presets): Claude, OpenAI, Gemini, OpenRouter, Ollama, LM Studio and Custom,
each with its base URL, whether it needs a key and the variable the key
arrives in. The key store embeds the same file at build time, so the app and
the sidecar agree on the ids, and a test keeps the table equal to the design
page's.

`choose()` turns the sidecar's start-up options (`--preset`, `--provider`,
`--base-url`, `--model`) into one choice, checked as hostile input:

- a preset must be one of the table's;
- the Claude preset's base URL is fixed; the hosted presets (OpenAI, Gemini,
  OpenRouter) take no other base URL, so a preset's key only ever goes to that
  preset's documented host; a local preset may point at another host or
  port, and Custom needs one;
- every base URL is checked by `check_base_url()` (http or https, no user
  name, password, query or fragment);
- `warnings()` reports what the pane should warn about: a non-loopback host
  for a local preset or Custom (prompts and workspace data leave the
  machine), and plain http to a host that is not loopback (they travel
  unencrypted).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .offline import is_loopback_host

PRESETS_FILE = Path(__file__).resolve().with_name("presets.json")
MAX_PRESETS_BYTES = 64 << 10
KEY_NEEDS = ("required", "optional", "none")
APIS = ("claude", "openai-compatible")
CHAT_APIS = ("openai", "ollama")
CUSTOM_ACCOUNT_PREFIX = "custom-"


class BaseUrlError(ValueError):
    pass


MAX_BASE_URL = 2048


def check_base_url(url: str) -> str:
    """The base URL as hostile input: an http or https address of at most 2,048
    printable characters, with a host, and without a user name, a password, a
    query or a fragment, which could carry a credential into the cache or a log."""
    if not isinstance(url, str) or not url or len(url) > MAX_BASE_URL or any(
            ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        raise BaseUrlError("--base-url must be an http or https address of at most 2,048 characters, "
                           "without spaces.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise BaseUrlError("--base-url is not a valid address.") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BaseUrlError("--base-url must be an http or https address, such as http://127.0.0.1:11434/v1.")
    if parts.username or parts.password:
        raise BaseUrlError("--base-url must not carry a user name or password; pass a key in "
                           "CALIPER_MODEL_API_KEY instead.")
    if parts.query or parts.fragment or url.endswith(("?", "#")):
        raise BaseUrlError("--base-url must not carry a query or a fragment.")
    del port
    return url.rstrip("/")


class PresetError(ValueError):
    """A provider choice that cannot be used; the message is safe to show."""


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    api: str
    base_url: str | None
    key: str               # required, optional, none
    key_env: str | None
    local: bool
    # "ollama": chat requests go to Ollama's own /api/chat, with the context
    # length below as options.num_ctx (ollama_native.py); "openai" otherwise.
    chat_api: str = "openai"
    # The context length the preset asks the server to load the model with,
    # unless the person sets another; None leaves it to the server.
    context_length: int | None = None


def load_presets(path: str | Path | None = None) -> dict[str, Preset]:
    p = Path(path) if path is not None else PRESETS_FILE
    with p.open("rb") as f:
        data = f.read(MAX_PRESETS_BYTES + 1)
    if len(data) > MAX_PRESETS_BYTES:
        raise PresetError("The presets file is larger than 64 KiB.")
    doc = json.loads(data)
    if not isinstance(doc, dict) or doc.get("version") != 0 or not isinstance(doc.get("presets"), list):
        raise PresetError("The presets file is not version 0 of the table.")
    out: dict[str, Preset] = {}
    for raw in doc["presets"]:
        try:
            preset = Preset(id=raw["id"], name=raw["name"], api=raw["api"], base_url=raw["base_url"],
                            key=raw["key"], key_env=raw["key_env"], local=raw["local"],
                            chat_api=raw.get("chat_api", "openai"), context_length=raw.get("context_length"))
        except (KeyError, TypeError, AttributeError):
            raise PresetError("A preset of the presets file lacks a field.") from None
        context = preset.context_length
        if (not isinstance(preset.id, str) or preset.api not in APIS or preset.key not in KEY_NEEDS
                or not isinstance(preset.local, bool) or (preset.key == "none") != (preset.key_env is None)
                or preset.chat_api not in CHAT_APIS
                or (preset.chat_api == "ollama" and preset.api != "openai-compatible")
                or not (context is None or (isinstance(context, int) and not isinstance(context, bool)
                                            and 0 < context <= 100_000_000))
                or preset.id in out):
            raise PresetError(f"The preset {str(preset.id)[:32]} of the presets file is malformed.")
        out[preset.id] = preset
    return out


PRESETS = load_presets()


def custom_key_account(base_url: str) -> str:
    """The credential store's account of a Custom entry's key: custom- and 12 hex digits of the base URL's SHA-256.

    The base URL is taken as `check_base_url()` gives it, without a trailing slash."""
    return CUSTOM_ACCOUNT_PREFIX + hashlib.sha256(check_base_url(base_url).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Choice:
    """Where the sidecar's model requests go."""

    preset: Preset | None
    api: str                   # claude or openai-compatible
    base_url: str | None       # checked, without a trailing slash; None for Claude

    def warnings(self) -> list[dict]:
        if self.base_url is None:
            return []
        parts = urlsplit(self.base_url)
        host = parts.hostname or ""
        out = []
        if not is_loopback_host(host) and (self.preset is None or self.preset.local or self.preset.id == "custom"):
            what = (f"The {self.preset.name} preset" if self.preset is not None and self.preset.local
                    else "This base URL")
            out.append({"code": "remote_host",
                        "message": f"{what} points at {host}, another machine: prompts and workspace data (samples, "
                                   "spec text, renders) leave this machine."})
        if parts.scheme == "http" and not is_loopback_host(host):
            out.append({"code": "unencrypted",
                        "message": f"The address of {host} is plain http, so prompts and workspace data travel "
                                   "unencrypted."})
        return out

    def as_meta(self) -> dict:
        host = urlsplit(self.base_url).hostname if self.base_url else "api.anthropic.com"
        return {"preset": self.preset.id if self.preset is not None else None, "api": self.api,
                "base_url": self.base_url, "host": host, "loopback": is_loopback_host(host),
                "warnings": self.warnings()}


def choose(preset_id: str | None, provider: str | None, base_url: str | None) -> Choice:
    """The provider choice of the start-up options; raises PresetError or BaseUrlError."""
    if preset_id is None:
        api = provider or "claude"
        if api not in APIS:
            raise PresetError(f"--provider must be one of {', '.join(APIS)}.")
        if api == "claude":
            if base_url:
                raise PresetError("The Claude API's base URL is fixed; --base-url is for an OpenAI-compatible server.")
            return Choice(PRESETS.get("anthropic"), "claude", None)
        if not base_url:
            raise PresetError("--provider openai-compatible needs --base-url.")
        return Choice(None, api, check_base_url(base_url))
    preset = PRESETS.get(preset_id)
    if preset is None:
        raise PresetError(f"--preset must be one of {', '.join(PRESETS)}.")
    if provider is not None and provider != preset.api:
        raise PresetError(f"The {preset.name} preset uses the {preset.api} provider, not {provider[:32]}.")
    if preset.api == "claude":
        if base_url:
            raise PresetError("The Claude preset's base URL is fixed.")
        return Choice(preset, "claude", None)
    if preset.id == "custom":
        if not base_url:
            raise PresetError("The Custom preset needs --base-url.")
        return Choice(preset, preset.api, check_base_url(base_url))
    assert preset.base_url is not None
    fixed = check_base_url(preset.base_url)
    if base_url and not preset.local and check_base_url(base_url) != fixed:
        raise PresetError(f"The {preset.name} preset's base URL is {preset.base_url}; use the Custom preset for "
                          "another server.")
    return Choice(preset, preset.api, check_base_url(base_url) if base_url else fixed)
