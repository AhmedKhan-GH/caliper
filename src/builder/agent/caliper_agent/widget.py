"""The widget role: one target, a tool scope and a stacked preview for each session.

A widget session is asked for in `session/new` with `_meta.caliper.widget`
(design/widget-agent.md, How it stays an ACP agent):

    {"target": {"kind": "panel", "panel": "curves"},
     "scope": ["spec.get", "panel.patch"],
     "preview": "widget-7f3a"}

- The **target** is a panel (`panel`), a selection with its panel
  (`selection`, `panel`), a dropped file's loader and its sources
  (`dropped_file`, `loader`, `catalog`), or one property of a panel or a source
  (`property`, `panel` or `source`, `path`).
- The **scope** is the commands the model may use. It is derived from the
  command table by one rule, kept as data in `widget_scopes.json` beside the
  table, which the app's context bundle reads too (`ScopeRule`). A scope that
  the app sends must be a subset of what the rule gives for the target, so a
  mistake in the app cannot widen what the widget agent may do; without one,
  the sidecar uses the rule's whole scope.
- The **preview** is the id of the stacked preview that the app opened for the
  session (`preview.begin`). Every call the sidecar makes for the session
  carries it in `X-Caliper-Preview`, so edits stack on the preview and never
  reach the open spec; only the person commits it, in the app.

Everything in `_meta` is hostile input: its size, its depth, every id and every
name are checked, and a widget field that is malformed refuses the session
instead of falling back to the unscoped role. A request without the field gets
the general session.

This module also turns the content blocks of `session/prompt` into what the
model reads: text, embedded text resources and resource links as text, and
images as images when the model reads them, else as a line that says one was
left out.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent

from .tools import PREVIEW_ARG, Command, ToolSpec, choose_strict, derive_tools

SCOPE_FILE = "widget_scopes.json"
MAX_SCOPE_FILE_BYTES = 64 << 10
MAX_META_BYTES = 64 << 10
MAX_META_DEPTH = 16
MAX_SCOPE = 64
MAX_CATALOG_IDS = 16
MAX_PROMPT_IMAGES = 4
# The most a prompt image may hold: the base64 text of one then stays under the
# 5 MB that one image of a Claude API request takes (caliper_api.py has the same).
MAX_PROMPT_IMAGE_BYTES = (5 * 1000 * 1000 * 3) // 4

TARGET_KINDS = ("panel", "selection", "dropped_file", "property")

# The turn budget of a widget prompt (design/widget-agent.md, The turn budget):
# six model requests, enough for a model that makes the edits of a selection's
# table one call at a time (three edits, a check and the reply), a few tool
# calls and a bounded reply. `caliper-agent --widget-requests` sets another.
WIDGET_MAX_REQUESTS = 6
WIDGET_MAX_TOOL_CALLS = 8
WIDGET_MAX_TOKENS = 8000

# Why a request leaves the widget agent for the side chat, as the `handoff`
# tool names it; the app turns each into "Ask in chat".
HANDOFF_KINDS = ("build_from_scratch", "loader", "project_code", "explanation", "another_target", "outside_scope")
HANDOFF_TOOL = "handoff"
MAX_HANDOFF_REASON = 500

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")              # spec ids: panels, sources, instruments
_LOADER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CATALOG_ID = re.compile(r"^(table|tensor|live|feed|metric|run):[^\x00-\x1f]{1,256}$")
_PREVIEW_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")      # the command table's preview id
_PROPERTY_PATH = re.compile(r"^[a-z_][a-z0-9_]{0,63}(\.[a-z_][a-z0-9_]{0,63}){0,7}$")
_COMMAND = re.compile(r"^[a-z][a-z0-9]*\.[a-z][a-z0-9_]*$")


class WidgetError(ValueError):
    """A widget field that cannot be used; the message is safe to show."""


# ---- the scope rule ------------------------------------------------------------------------------


def default_scope_file(table: Path | None) -> Path | None:
    """`widget_scopes.json` beside the command table, else beside the package."""
    candidates = []
    if table is not None:
        candidates.append(Path(table).resolve().parent / SCOPE_FILE)
    here = Path(__file__).resolve().parent
    candidates += [here / SCOPE_FILE, here.parent.parent / "commands" / SCOPE_FILE]
    return next((c for c in candidates if c.is_file()), None)


def _name_list(doc: dict, key: str, where: str) -> tuple[str, ...]:
    value = doc.get(key, [])
    if not isinstance(value, list) or len(value) > 256 or not all(
            isinstance(v, str) and len(v) <= 64 for v in value):
        raise WidgetError(f"{where}{key} is not a list of names.")
    return tuple(value)


@dataclass(frozen=True)
class ScopeRule:
    """The data of the rule (`widget_scopes.json`); `scope()` applies it to a table."""

    never: frozenset[str]
    never_groups: frozenset[str]
    vision_only: frozenset[str]
    targets: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]   # key -> (reads, edits)

    @staticmethod
    def parse(data: bytes | str) -> "ScopeRule":
        if isinstance(data, str):
            data = data.encode("utf-8")
        if len(data) > MAX_SCOPE_FILE_BYTES:
            raise WidgetError("The widget scope file is larger than 64 KiB.")
        try:
            doc = json.loads(data)
        except (ValueError, RecursionError):
            raise WidgetError("The widget scope file is not JSON.") from None
        if not isinstance(doc, dict) or doc.get("version") != 0 or not isinstance(doc.get("targets"), dict):
            raise WidgetError("The widget scope file is not version 0 of the rule.")
        targets = {}
        for key, spec in doc["targets"].items():
            if not isinstance(spec, dict) or not isinstance(key, str):
                raise WidgetError(f"The widget scope of {key!s:.32} is not an object.")
            targets[key] = (_name_list(spec, "reads", f"targets.{key}."), _name_list(spec, "edits", f"targets.{key}."))
        return ScopeRule(never=frozenset(_name_list(doc, "never", "")),
                         never_groups=frozenset(_name_list(doc, "never_groups", "")),
                         vision_only=frozenset(_name_list(doc, "vision_only", "")), targets=targets)

    @staticmethod
    def load(path: str | Path) -> "ScopeRule":
        p = Path(path)
        try:
            with p.open("rb") as f:
                data = f.read(MAX_SCOPE_FILE_BYTES + 1)
        except OSError as exc:
            raise WidgetError(f"The widget scope file cannot be read: {exc.strerror or exc}") from None
        return ScopeRule.parse(data)

    def _excluded(self, c: Command) -> bool:
        return c.name in self.never or c.group in self.never_groups

    def scope(self, key: str, commands: list[Command], *, vision: bool = True) -> list[str]:
        """The command names of one kind of target, in the table's order."""
        if key not in self.targets:
            raise WidgetError(f"The widget scope file has no target {key}.")
        reads, edits = self.targets[key]
        out = []
        for c in commands:
            if self._excluded(c) or (not vision and c.name in self.vision_only):
                continue
            if c.name in reads and not c.mutates:
                out.append(c.name)
            elif c.name in edits and c.mutates and c.previewable and not c.destructive:
                out.append(c.name)
        return out


# ---- the target ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class WidgetSpec:
    """What a widget session acts on: its target, its scope and its preview."""

    target: dict                 # the target, normalised: only the checked keys
    scope: tuple[str, ...]       # command names, in the table's order
    preview: str

    @property
    def kind(self) -> str:
        return self.target["kind"]

    @property
    def scope_key(self) -> str:
        """The key of the target's entry in the scope rule."""
        return target_key(self.target)


def _depth(value: Any, limit: int, depth: int = 0) -> bool:
    if depth > limit:
        return False
    if isinstance(value, dict):
        return all(_depth(v, limit, depth + 1) for v in value.values())
    if isinstance(value, list):
        return all(_depth(v, limit, depth + 1) for v in value)
    return True


def _str(target: dict, key: str, pattern: re.Pattern, what: str, *, required: bool = True) -> str | None:
    value = target.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise WidgetError(f"The widget target needs {what} in {key}.")
    return value


def target_key(target: dict) -> str:
    """The key of a target's entry in the scope rule."""
    kind = target["kind"]
    if kind == "property":
        return "panel_property" if "panel" in target else "source_property"
    return kind


def parse_target(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise WidgetError("The widget target is not an object.")
    kind = raw.get("kind")
    if kind not in TARGET_KINDS:
        raise WidgetError(f"The widget target's kind must be one of {', '.join(TARGET_KINDS)}.")
    target: dict = {"kind": kind}
    if kind in ("panel", "selection"):
        target["panel"] = _str(raw, "panel", _ID, "a panel id")
    elif kind == "dropped_file":
        target["loader"] = _str(raw, "loader", _LOADER_ID, "a loader id")
        ids = raw.get("catalog", [])
        if not isinstance(ids, list) or len(ids) > MAX_CATALOG_IDS or not all(
                isinstance(i, str) and _CATALOG_ID.fullmatch(i) for i in ids):
            raise WidgetError(f"The widget target's catalog must be at most {MAX_CATALOG_IDS} catalog ids.")
        target["catalog"] = list(dict.fromkeys(ids))
    else:  # property
        panel = _str(raw, "panel", _ID, "a panel id", required=False)
        source = _str(raw, "source", _ID, "a source id", required=False)
        if (panel is None) == (source is None):
            raise WidgetError("A property belongs to a panel or a source: give panel or source, not both.")
        target["panel" if panel is not None else "source"] = panel if panel is not None else source
        target["path"] = _str(raw, "path", _PROPERTY_PATH, "a property path such as options.colormap")
    return target


def parse_widget_meta(meta: Any, commands: list[Command], rule: ScopeRule | None) -> WidgetSpec | None:
    """The widget session that `session/new`'s `_meta` asks for, or None for the general role.

    Raises WidgetError for a widget field that cannot be used."""
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise WidgetError("_meta is not an object.")
    caliper = meta.get("caliper")
    if caliper is None:
        return None
    if not isinstance(caliper, dict):
        raise WidgetError("_meta.caliper is not an object.")
    widget = caliper.get("widget")
    if widget is None:
        return None
    try:
        size = len(json.dumps(widget, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        raise WidgetError("_meta.caliper.widget cannot be read.") from None
    if size > MAX_META_BYTES or not _depth(widget, MAX_META_DEPTH):
        raise WidgetError("_meta.caliper.widget is larger or deeper than the sidecar takes.")
    if not isinstance(widget, dict):
        raise WidgetError("_meta.caliper.widget is not an object.")
    if rule is None:
        raise WidgetError("This sidecar has no widget scope file, so it cannot open a widget session.")
    target = parse_target(widget.get("target"))
    preview = widget.get("preview")
    if not isinstance(preview, str) or not _PREVIEW_ID.fullmatch(preview):
        raise WidgetError("A widget session needs the id of the preview the app opened, in preview.")
    allowed = rule.scope(target_key(target), commands, vision=True)
    scope = widget.get("scope")
    if scope is None:
        chosen = allowed
    else:
        if not isinstance(scope, list) or not scope or len(scope) > MAX_SCOPE or not all(
                isinstance(n, str) and _COMMAND.fullmatch(n) for n in scope):
            raise WidgetError(f"The widget scope must be a list of 1 to {MAX_SCOPE} command names.")
        outside = [n for n in scope if n not in allowed]
        if outside:
            raise WidgetError(f"The widget scope names {outside[0][:64]}, which the rule does not allow for a "
                              f"{target['kind'].replace('_', ' ')}.")
        wanted = set(scope)
        chosen = [n for n in allowed if n in wanted]
    return WidgetSpec(target=target, scope=tuple(chosen), preview=preview)


# ---- the scoped tools ----------------------------------------------------------------------------


def _without_preview_arg(schema: dict) -> dict:
    props = schema.get("properties")
    if not isinstance(props, dict) or PREVIEW_ARG not in props:
        return schema
    out = dict(schema)
    out["properties"] = {k: v for k, v in props.items() if k != PREVIEW_ARG}
    if isinstance(schema.get("required"), list):
        out["required"] = [r for r in schema["required"] if r != PREVIEW_ARG]
    return out


def widget_tools(commands: list[Command], scope: tuple[str, ...], *, vision: bool,
                 rule: ScopeRule) -> list[ToolSpec]:
    """The tools of a widget session: its scope, in the table's order, without
    `dry_run` (every edit stacks on the preview), and without the vision-only
    commands for a model that reads no images. The strict tools are chosen again
    among them."""
    wanted = {n for n in scope if vision or n not in rule.vision_only}
    tools = derive_tools(commands, lambda c: c.name in wanted)
    return choose_strict([replace(t, input_schema=_without_preview_arg(t.input_schema)) for t in tools])


HANDOFF_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(HANDOFF_KINDS),
                 "description": "Why the request needs the side chat."},
        "reason": {"type": "string", "description": "One sentence for the person on what the request needs."},
    },
    "required": ["kind", "reason"],
    "additionalProperties": False,
}

HANDOFF_DESCRIPTION = (
    "Hand the request to the side chat instead of doing it. Use it when the request does not fit this one target "
    "or your tools: building a dashboard from nothing, making or changing a loader, changing project code, a long "
    "explanation, another panel or source, or anything your tools cannot do. Caliper then offers the person to ask "
    "in the side chat. Make no edits before or after it.")


# ---- the prompt's content blocks -----------------------------------------------------------------


@dataclass
class PromptContent:
    """What the model reads of one prompt."""

    parts: list = field(default_factory=list)   # str and BinaryContent, in order
    chars: int = 0
    images: int = 0
    images_left_out: int = 0

    def user_prompt(self) -> Any:
        """A single string when there is no image (as the general role always sent), else the list."""
        if all(isinstance(p, str) for p in self.parts):
            return "\n\n".join(self.parts)
        return self.parts


def _image_kind(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _for_the_assistant(block: dict) -> bool:
    """False for a block that annotations mark as meant for the person only."""
    annotations = block.get("annotations")
    audience = annotations.get("audience") if isinstance(annotations, dict) else None
    return not (isinstance(audience, list) and audience and "assistant" not in audience)


def prompt_content(blocks: Any, *, vision: bool, max_chars: int) -> PromptContent:
    """The prompt's ACP content blocks as the model's input; raises WidgetError for one
    that cannot be used (the caller answers -32602)."""
    if not isinstance(blocks, list) or not blocks:
        raise WidgetError("session/prompt needs a non-empty prompt.")
    out = PromptContent()

    def add_text(text: str) -> None:
        if text:
            out.parts.append(text)
            out.chars += len(text)

    for block in blocks:
        if not isinstance(block, dict) or not _for_the_assistant(block):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            add_text(block["text"])
        elif kind == "resource_link" and isinstance(block.get("uri"), str):
            name = block.get("name") if isinstance(block.get("name"), str) else block["uri"]
            add_text(f"[Linked resource: {name} ({block['uri']})]")
        elif kind == "resource" and isinstance(block.get("resource"), dict):
            resource = block["resource"]
            uri = resource.get("uri") if isinstance(resource.get("uri"), str) else "resource"
            if isinstance(resource.get("text"), str):
                add_text(resource["text"])
            else:
                add_text(f"[An embedded resource that is not text ({uri[:200]}) was left out.]")
        elif kind == "image":
            data = block.get("data")
            if not isinstance(data, str):
                raise WidgetError("An image block has no data.")
            if out.images + out.images_left_out >= MAX_PROMPT_IMAGES:
                raise WidgetError(f"A prompt holds at most {MAX_PROMPT_IMAGES} images.")
            if len(data) > (MAX_PROMPT_IMAGE_BYTES * 4) // 3 + 4:
                raise WidgetError("An image of the prompt is larger than the sidecar takes.")
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                raise WidgetError("An image of the prompt is not valid base64.") from None
            media = _image_kind(raw)
            if media is None:
                raise WidgetError("An image of the prompt is not a PNG, JPEG, GIF or WebP.")
            if vision:
                out.parts.append(BinaryContent(data=raw, media_type=media))
                out.images += 1
            else:
                out.images_left_out += 1
                add_text("[An image of the panel was left out, because this model cannot read images.]")
        if out.chars > max_chars:
            raise WidgetError(f"The prompt is longer than {max_chars} characters.")
    if not any(isinstance(p, str) and p.strip() for p in out.parts):
        raise WidgetError("The prompt holds no text.")
    return out
