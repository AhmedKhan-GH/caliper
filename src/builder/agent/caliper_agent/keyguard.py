"""The API key's life inside the sidecar, and the scrubbing of every text that leaves it.

The app places the user's key in the sidecar's environment at spawn, under the
one variable the Claude API SDK reads. `take_key_from_environment()` reads it
once, removes it from `os.environ` together with every variable that could make
the SDK use another credential, another endpoint or extra headers, and returns
it wrapped in a `Secret` whose repr never shows it.

Every text that leaves the process (ACP messages, stderr lines, log records,
exception reports) passes through the `Scrubber`, which replaces each
registered secret (the key, Caliper API's session token) and anything shaped
like a Claude, OpenRouter or Google API key with a marker. `install_guards()` routes uncaught
exceptions and log records through it.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import traceback
from typing import Iterable

# The variable the Claude API SDK reads, and the only way the Claude key enters
# the sidecar.
KEY_ENV = "ANTHROPIC_API_KEY"
# The key of an OpenAI-compatible server, when it needs one (openai_compat.py).
MODEL_KEY_ENV = "CALIPER_MODEL_API_KEY"

# Variables that would let the SDK pick another credential, send requests to
# another host, add headers or log requests. They are removed at start so that
# only the key the app passed is ever used, and only against the Claude API.
OVERRIDE_ENV = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_LOG",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY",
    # The OpenAI SDK, which Pydantic AI uses for OpenAI-compatible servers.
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_WEBHOOK_SECRET",
    "OPENAI_LOG",
    # Pydantic AI's own gateway and telemetry switches.
    "PYDANTIC_AI_GATEWAY_API_KEY",
    "LOGFIRE_TOKEN",
)

MAX_KEY_LENGTH = 1024
REDACTED = "[redacted]"

# Anything shaped like a provider's key, even one that is not ours; each
# shape was read in the provider's own documentation (design/widget-agent.md,
# Presets): a Claude API key starts with sk-ant-; an OpenRouter key is sk-or-v1-
# and 64 lowercase hexadecimal digits (its authentication page); a Google API
# key, which the Gemini preset takes, is AIza and 35 more characters (the
# example key of Google Cloud's API keys page). OpenAI's documentation that
# could be read shows no key's shape, so only the exact key is replaced there.
_KEY_SHAPES = (
    ("sk-ant-", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("sk-or-v1-", re.compile(r"sk-or-v1-[0-9a-f]{64}")),
    ("AIza", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
)


class Secret:
    """A secret string that never shows itself in a repr, a str or a format."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The value, for the one place that needs it: the SDK client."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(****)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "****"

    def __reduce__(self):  # refuse pickling, which would copy the value out
        raise TypeError("A Secret cannot be pickled.")


class KeyStatus:
    """What the environment held: a usable key, a malformed one, or none."""

    MISSING = "missing"
    MALFORMED = "malformed"
    PRESENT = "present"


def _is_well_formed(value: str) -> bool:
    if not value or len(value) > MAX_KEY_LENGTH:
        return False
    # A key is printable ASCII without spaces; anything else is refused without
    # being shown.
    return all(0x21 <= ord(c) <= 0x7E for c in value)


def take_key_from_environment(environ: dict | None = None, name: str = KEY_ENV) -> tuple[Secret | None, str]:
    """Reads the key in `name` once and removes it, and every override, from the environment.

    Returns the key (or None) and a KeyStatus value. The status never carries
    any part of the value.
    """
    env = os.environ if environ is None else environ
    raw = env.pop(name, None)
    for name in OVERRIDE_ENV:
        env.pop(name, None)
    if raw is None or raw == "":
        return None, KeyStatus.MISSING
    if not _is_well_formed(raw):
        return None, KeyStatus.MALFORMED
    return Secret(raw), KeyStatus.PRESENT


class Scrubber:
    """Replaces registered secrets, and anything shaped like a key, in text."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._secrets: list[str] = []

    def register(self, secret: str | Secret | None) -> None:
        if secret is None:
            return
        value = secret.reveal() if isinstance(secret, Secret) else secret
        if len(value) < 8:
            return  # too short to replace safely; such values are refused earlier
        with self._lock:
            if value not in self._secrets:
                self._secrets.append(value)
                # Longest first, so that a secret containing another is replaced whole.
                self._secrets.sort(key=len, reverse=True)

    def scrub(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            secrets = list(self._secrets)
        for value in secrets:
            if value in text:
                text = text.replace(value, REDACTED)
        for marker, shape in _KEY_SHAPES:
            if marker in text:
                text = shape.sub(REDACTED, text)
        return text

    def contains_secret(self, text: str) -> bool:
        with self._lock:
            secrets = list(self._secrets)
        return any(value in text for value in secrets) or any(
            marker in text and shape.search(text) for marker, shape in _KEY_SHAPES)


# The process-wide scrubber. Modules that register a secret use this one.
SCRUBBER = Scrubber()


def scrub(text: str) -> str:
    return SCRUBBER.scrub(text)


def describe_exception(exc: BaseException) -> str:
    """One scrubbed line naming an exception, for messages that reach the pane."""
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    name = type(exc).__name__
    return scrub(f"{name}: {message}" if message else name)


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:
            text = str(record.msg)
        record.msg = scrub(text)
        record.args = None
        if record.exc_info:
            record.exc_text = scrub("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return True


def _write_scrubbed(lines: Iterable[str]) -> None:
    try:
        sys.stderr.write(scrub("".join(lines)))
        sys.stderr.flush()
    except Exception:
        pass


def _excepthook(exc_type, exc, tb) -> None:
    _write_scrubbed(["caliper-agent: uncaught exception\n", *traceback.format_exception(exc_type, exc, tb)])


def _thread_excepthook(args) -> None:
    _write_scrubbed(
        ["caliper-agent: uncaught exception in a thread\n",
         *traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)]
    )


def _unraisablehook(unraisable) -> None:
    exc = unraisable.exc_value
    if exc is None:
        return
    _write_scrubbed(["caliper-agent: unraisable exception\n", *traceback.format_exception(type(exc), exc, exc.__traceback__)])


def install_guards(level: int = logging.WARNING) -> None:
    """Routes uncaught exceptions and log records through the scrubber, to stderr."""
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    sys.unraisablehook = _unraisablehook
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("caliper-agent: %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_ScrubFilter())
    root.addHandler(handler)
    root.setLevel(level)
    # The SDK and its HTTP library log request details at debug level; keep them quiet.
    for name in ("anthropic", "openai", "pydantic_ai", "httpx2", "httpcore2", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
