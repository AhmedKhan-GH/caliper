"""One prompt turn: model requests, tool calls, approvals, and their ACP updates.

The loop is the sidecar's own (the SDK's tool runner is not used; the design
page says why). For each request it lets the provider stream the reply to the
pane, then acts on how the reply ended:

- `end_turn`: the turn ends with ACP stop reason `end_turn`.
- `tool_use`: each tool call runs in order (validated, approved when it must
  be, then sent to Caliper API) and all results go back in one message.
- `max_tokens` or `context_full`: no tool of that reply runs; each gets an error
  result so the history stays valid, and the turn ends with `max_tokens`.
- `refusal`: no tool runs; the turn ends with `refusal`, and the prompt and
  everything after it are left out of the history, as ACP's `refusal` means.
- cancellation: whatever is running stops; every tool call of the reply gets a
  result (`cancelled` for those that did not finish) and the turn ends with
  `cancelled`.

Approval: before a destructive command runs, and before a command that can
preview its change (one that takes `dry_run`) when the session reviews spec
changes, the loop sends `session/request_permission` and acts on the answer. For
a previewable command it first sends the same command with `dry_run: true`: a
preview that is refused is the call's result and nothing is asked; an empty diff
runs the command without asking; otherwise the request carries the change as an
ACP diff, and the command runs only when it is allowed. Caliper itself may also
hold a destructive command for the person's approval; the call then just takes
longer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .acp import Connection, PeerClosed, RpcError
from .caliper_api import CaliperApi, ToolOutput
from .claude import tool_input_text
from .keyguard import scrub
from .provider import AgentError, Provider, RequestResult, Settings, ToolCall, ToolOutcome
from .schema import validate
from .tools import PREVIEW_ARG, ToolSpec, acp_tool_kind
from .udiff import apply_unified_diff

log = logging.getLogger("caliper_agent.loop")

MAX_REQUESTS_PER_TURN = 50
MAX_PANE_TEXT = 4096

REVIEW_SPEC_CHANGES = "spec_changes"
REVIEW_DESTRUCTIVE_ONLY = "destructive_only"


@dataclass
class SessionState:
    """What a turn needs from its session."""

    id: str
    cwd: str
    conversation: Any
    settings: Settings
    review: str = REVIEW_SPEC_CHANGES
    turns: int = 0
    usage: dict = field(default_factory=dict)


class AcpSink:
    """Turns provider events into ACP session updates."""

    def __init__(self, conn: Connection, session_id: str) -> None:
        self._conn = conn
        self._sid = session_id
        self.started_calls: dict[str, str] = {}   # tool call id -> tool name, as shown in the pane
        self.open_calls: set[str] = set()         # shown and not yet completed or failed

    def update(self, update: dict) -> None:
        self._conn.notify("session/update", {"sessionId": self._sid, "update": update})

    def message(self, text: str) -> None:
        self.update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}})

    def thought(self, text: str) -> None:
        self.update({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}})

    def status(self, kind: str, text: str, **meta: Any) -> None:
        # A status reaches the pane as a thought that carries `_meta`, so any ACP
        # client shows it, and Caliper's pane can show it as a status line.
        self.update({"sessionUpdate": "agent_thought_chunk",
                     "content": {"type": "text", "text": scrub(text)},
                     "_meta": {"caliper": {"status": kind, **meta}}})

    def tool_started(self, call_id: str, name: str) -> None:
        if call_id in self.started_calls:
            return
        self.started_calls[call_id] = name
        self.open_calls.add(call_id)
        self.update({"sessionUpdate": "tool_call", "toolCallId": call_id, "title": name, "name": name,
                     "kind": "other", "status": "pending"})

    def ensure_started(self, call_id: str, name: str) -> None:
        self.tool_started(call_id, name)

    def finish(self, call_id: str, status: str, content: list[dict] | None = None,
               raw_output: Any = None) -> None:
        update: dict = {"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": status}
        if content:
            update["content"] = content
        if raw_output is not None:
            update["rawOutput"] = raw_output
        self.update(update)
        if status in ("completed", "failed"):
            self.open_calls.discard(call_id)

    def fail_open_calls(self, reason: str, keep: set[str] | None = None) -> None:
        for call_id in sorted(self.open_calls - (keep or set())):
            self.finish(call_id, "failed", [text_content(reason)])


def text_content(text: str) -> dict:
    return {"type": "content", "content": {"type": "text", "text": text}}


def _pane_text(text: str) -> str:
    return text if len(text) <= MAX_PANE_TEXT else text[:MAX_PANE_TEXT] + "\n[cut]"


def _describe_call(command: str, args: Any) -> str:
    try:
        pretty = json.dumps(args, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        pretty = repr(args)
    return _pane_text(f"{command}\n{pretty}")


def _title(command: str, args: Any) -> str:
    if isinstance(args, dict):
        for key in ("id", "name", "panel", "path"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return f"{command} {value[:64]}"
    return command


def _diff_path(cwd: str, relative: str) -> str:
    """An absolute path under the session's folder for a spec diff."""
    rel = PurePosixPath(relative.replace("\\", "/"))
    parts = [p for p in rel.parts if p not in ("", ".", "..", "/")]
    return posixpath.join(cwd, *parts) if parts else posixpath.join(cwd, "open.spec.toml")


class TurnRunner:
    def __init__(self, conn: Connection, provider: Provider, api: CaliperApi,
                 tools: dict[str, ToolSpec], session: SessionState, usage_log: str | None = None) -> None:
        self._usage_log = usage_log
        self._conn = conn
        self._provider = provider
        self._api = api
        self._tools = tools
        self._s = session
        self._sink = AcpSink(conn, session.id)
        self._usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "per_request": []}

    @property
    def usage(self) -> dict:
        return self._usage

    def _add_usage(self, result: RequestResult) -> None:
        self._usage["requests"] += 1
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            self._usage[key] += int(result.usage.get(key, 0))
        self._usage["per_request"].append({**result.usage, "model": result.model})
        if self._usage_log:
            # Token counts only, for the manual check of prompt caching; no content.
            line = json.dumps({"session": self._s.id, "turn": self._s.turns, "model": result.model, **result.usage},
                              sort_keys=True)
            try:
                with open(self._usage_log, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                log.warning("the usage log cannot be written")

    async def run(self, prompt_text: str) -> str:
        """Runs the turn and returns its ACP stop reason. AgentError ends it with an error."""
        s = self._s
        s.turns += 1
        group = f"{s.id}-t{s.turns}"
        conv = s.conversation
        conv.add_user_prompt(prompt_text)
        pending: list[ToolCall] = []
        done: dict[str, ToolOutcome] = {}
        try:
            for index in range(MAX_REQUESTS_PER_TURN):
                pending, done = [], {}
                result = await self._provider.request(conv, s.settings, self._sink)
                self._add_usage(result)
                if result.served_by_fallback:
                    self._sink.status("served_by", f"{result.model} answered this request.", model=result.model)
                kept = {c.id for c in result.tool_calls}
                if result.stop == "refusal":
                    self._sink.fail_open_calls("Not run: the model declined the request.")
                    rolled_back = conv.rollback_turn()
                    text = "The model declined this request."
                    if result.refusal_category:
                        text += f" Category: {result.refusal_category}."
                    text += " The prompt and what followed it are left out of the conversation, so you can rephrase it."
                    if index > 0:
                        text += " Changes this turn already made in Caliper stay; undo them in the app if needed."
                    self._sink.status("refusal", text, category=result.refusal_category,
                                      explanation=result.refusal_explanation, turn_removed=rolled_back)
                    return "refusal"
                # Calls the pane saw that are not in the reply any more (a fallback
                # replaced the reply that proposed them) never run.
                self._sink.fail_open_calls("Not run: the model that proposed this call declined, "
                                           "and another model continued.", keep=kept)
                if result.stop == "pause":
                    continue
                if result.stop != "tool_use":
                    if result.tool_calls:
                        reason = ("Not run: the reply was cut off at the output limit."
                                  if result.stop in ("max_tokens", "context_full")
                                  else "Not run: the reply ended without asking for tools.")
                        conv.add_tool_results([ToolOutcome(c.id, reason, is_error=True) for c in result.tool_calls])
                        self._sink.fail_open_calls(reason)
                    if result.stop in ("max_tokens", "context_full"):
                        return "max_tokens"
                    return "end_turn"
                if not result.tool_calls:
                    return "end_turn"
                pending = list(result.tool_calls)
                for call in result.tool_calls:
                    outcome = await self._run_tool(call, group)
                    done[call.id] = outcome
                    if outcome.cancelled:
                        raise asyncio.CancelledError()
                conv.add_tool_results([done[c.id] for c in result.tool_calls])
                pending = []
            self._sink.status("limit", f"The turn stopped after {MAX_REQUESTS_PER_TURN} model requests.")
            return "max_turn_requests"
        except asyncio.CancelledError:
            if pending:
                outcomes = [done.get(c.id) or ToolOutcome(c.id, "Cancelled by the user before it finished.",
                                                          is_error=True, cancelled=True) for c in pending]
                conv.add_tool_results(outcomes)
            self._sink.fail_open_calls("Cancelled.")
            return "cancelled"
        except AgentError:
            if pending:
                conv.add_tool_results([done.get(c.id) or ToolOutcome(c.id, "Not run: the turn failed.", is_error=True)
                                       for c in pending])
            self._sink.fail_open_calls("Not run: the turn failed.")
            raise

    # ---- tools ------------------------------------------------------------------

    async def _run_tool(self, call: ToolCall, group: str) -> ToolOutcome:
        sink = self._sink
        spec = self._tools.get(call.name)
        sink.ensure_started(call.id, call.name)
        if spec is None:
            sink.finish(call.id, "failed", [text_content(f"There is no tool named {call.name}.")])
            return ToolOutcome(call.id, json.dumps({"error": f"There is no tool named {call.name}."}), is_error=True)

        command = spec.command
        args = call.input
        problem = "the input is not a JSON object" if not isinstance(args, dict) else validate(args, command.args)
        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call.id,
                     "title": _title(command.name, args), "kind": acp_tool_kind(command),
                     "rawInput": args if isinstance(args, (dict, list)) else None})
        if problem:
            # Treated like invalid JSON: not run, and the model sees what it sent.
            text = json.dumps({"INVALID_JSON": tool_input_text(args), "problem": problem}, ensure_ascii=False)
            sink.finish(call.id, "failed", [text_content(f"Not run: {problem}")])
            return ToolOutcome(call.id, text, is_error=True)

        previewable = command.previewable and args.get(PREVIEW_ARG) is not True
        before = await self._api.spec_text() if previewable else None

        preview_diff: str | None = None
        if previewable and self._s.review == REVIEW_SPEC_CHANGES:
            preview = await self._api.call(command.name, {**args, PREVIEW_ARG: True}, group)
            if not preview.ok:
                # The change itself is refused; the real call would be too.
                sink.finish(call.id, "failed", [text_content(_pane_text(preview.text))],
                            raw_output={"error": preview.error_code})
                return ToolOutcome(call.id, scrub(preview.text), is_error=True)
            diff = preview.result.get("diff") if isinstance(preview.result, dict) else None
            preview_diff = diff if isinstance(diff, str) else None
        # A preview that changes nothing needs no approval.
        ask = command.destructive or (previewable and self._s.review == REVIEW_SPEC_CHANGES and preview_diff != "")
        if ask:
            decision = await self._ask(call, command, args, before, preview_diff)
            if decision == "cancelled":
                sink.finish(call.id, "failed", [text_content("Cancelled.")])
                return ToolOutcome(call.id, "Cancelled by the user.", is_error=True, cancelled=True)
            if decision == "rejected":
                sink.finish(call.id, "failed", [text_content("Denied by the user.")])
                return ToolOutcome(call.id, json.dumps({"error": {
                    "code": "agent.denied",
                    "message": "The user denied this call. Do not try it again in another form; "
                               "say what you wanted to do and ask how to go on."}}), is_error=True)

        sink.update({"sessionUpdate": "tool_call_update", "toolCallId": call.id, "status": "in_progress"})
        out = await self._api.call(command.name, args, group)
        return await self._report(call, command, out, before, previewable)

    def _diff_content(self, before: tuple[str, str] | None, diff: str | None) -> dict | None:
        """The change as an ACP diff (old and new text), else as the unified diff in a text block."""
        if not isinstance(diff, str) or diff == "":
            return None
        if before is not None:
            after = apply_unified_diff(before[0], diff)
            if after is not None:
                return {"type": "diff", "path": _diff_path(self._s.cwd, before[1]), "oldText": before[0],
                        "newText": after}
        return text_content(_pane_text("```diff\n" + diff.rstrip("\n") + "\n```"))

    async def _ask(self, call: ToolCall, command, args: dict, before: tuple[str, str] | None,
                   preview_diff: str | None) -> str:
        """Sends session/request_permission; returns allowed, rejected or cancelled."""
        content = [text_content(_describe_call(command.name, args))]
        shown = self._diff_content(before, preview_diff)
        if shown is not None:
            content.insert(0, shown)
        if command.destructive:
            options = [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                       {"optionId": "reject", "name": "Deny", "kind": "reject_once"}]
        else:
            options = [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                       {"optionId": "allow-session", "name": "Allow spec changes for this session",
                        "kind": "allow_always"},
                       {"optionId": "reject", "name": "Deny", "kind": "reject_once"}]
        tool_call = {"toolCallId": call.id, "title": _title(command.name, args), "kind": acp_tool_kind(command),
                     "status": "pending", "rawInput": args, "content": content}
        try:
            answer = await self._conn.request("session/request_permission",
                                              {"sessionId": self._s.id, "toolCall": tool_call, "options": options})
        except (RpcError, PeerClosed):
            return "rejected"
        outcome = answer.get("outcome") if isinstance(answer, dict) else None
        if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
            return "cancelled"
        chosen = outcome.get("optionId")
        kinds = {o["optionId"]: o["kind"] for o in options}
        kind = kinds.get(chosen)
        if kind == "allow_always":
            self._s.review = REVIEW_DESTRUCTIVE_ONLY
            return "allowed"
        if kind == "allow_once":
            return "allowed"
        return "rejected"

    async def _report(self, call: ToolCall, command, out: ToolOutput, before, previewable: bool) -> ToolOutcome:
        content: list[dict] = []
        if out.ok and previewable and before is not None:
            diff = out.result.get("diff") if isinstance(out.result, dict) else None
            shown = self._diff_content(before, diff) if isinstance(diff, str) else None
            if shown is None and not isinstance(diff, str):
                # An app whose commands return no diff: compare the spec before and after.
                after = await self._api.spec_text()
                if after is not None and after[0] != before[0]:
                    shown = {"type": "diff", "path": _diff_path(self._s.cwd, after[1]),
                             "oldText": before[0], "newText": after[0]}
            if shown is not None:
                content.append(shown)
        content.append(text_content(_pane_text(out.text)))
        if out.png is not None:
            content.append({"type": "content", "content": {"type": "image", "mimeType": "image/png",
                                                           "data": base64.b64encode(out.png).decode("ascii")}})
        raw = out.result if out.ok else {"error": out.error_code}
        self._sink.finish(call.id, "completed" if out.ok else "failed", content, raw_output=raw)
        # Workspace data could hold a secret; the model never sees one.
        return ToolOutcome(call.id, scrub(out.text), is_error=not out.ok, png=out.png)
