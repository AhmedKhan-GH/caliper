#!/usr/bin/env python3
"""Generate or verify docs/reference/mcp-commands.md from command_table.json.

Reads src/builder/commands/command_table.json and renders the complete MCP tool
and command-layer lookup reference. In --check mode (the default), exits with
status 1 when command_table.json is invalid or docs/reference/mcp-commands.md
drifts from the generated output.

Usage:
  generate_mcp_reference.py [--root <repo_root>] [--check | --write]
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys

TABLE_REL = "src/builder/commands/command_table.json"
OUTPUT_REL = "docs/reference/mcp-commands.md"

GROUP_INFO = {
    "app": ("Application Commands", "Application commands inspect host versions, list applets, and switch the active applet."),
    "catalog": ("Catalog Commands", "Catalog commands list, describe, and sample data sources across workspace stores and live feeds."),
    "data": ("Data Store Commands", "Data store commands run read-only SQL queries and import tabular files into the workspace store."),
    "loader": ("Loader Commands", "Loader commands manage and execute workspace file and Python data loaders."),
    "spec": ("Spec Document Commands", "Spec document commands create, open, read, replace, and validate the active spec document."),
    "source": ("Source Commands", "Source commands add, patch, and remove data sources and transform chains in the active spec."),
    "panel": ("Panel Commands", "Panel commands create, patch, remove, render, and read data from spec panels."),
    "layout": ("Layout Commands", "Layout commands place panels, set split trees and shares, and lock or unlock the layout."),
    "preview": ("Preview Commands", "Preview commands stage, diff, commit, or discard multi-step spec edits before workspace mutation."),
    "applet": ("Workspace Applet Commands", "Workspace applet commands save, open, and list spec applets in the builder workspace."),
    "review": ("Provenance Review Commands", "Provenance review commands report foreign spec review status for custom GPU programs."),
    "capture": ("Frame Capture Commands", "Frame capture commands render the current application view to a PNG image."),
    "control": ("Spec Control Commands", "Spec control commands create, patch, remove, and set interactive controls on the active spec."),
    "state": ("State Variable Commands", "State variable commands define and update shared state values across panels, controls, and programs."),
    "selection": ("Selection Commands", "Selection commands read and set linked entity or program item selections."),
    "entity": ("Scene Entity Commands", "Scene entity commands add, update, and remove 3D scene entities on scene panels."),
    "annotation": ("Annotation Commands", "Annotation commands add, update, and remove labels, thresholds, bands, and markers on panels."),
    "cursor": ("Step Cursor Commands", "Step cursor commands read and move the step history cursor across recorded snapshots."),
    "camera": ("Camera Commands", "Camera commands read, set, and save scene panel camera poses and presets."),
    "run": ("Run Control and Managed Run Commands", "Run commands inspect and steer attached scripts and start, fork, or stop managed workspace runs."),
    "events": ("Event Stream Commands", "Event commands wait for live run, pick, and program events from the host ring."),
    "program": ("GPU Program Commands", "Program commands attach, patch, remove, read, and inspect GPU programs and piece instances."),
    "piece": ("Piece Library Commands", "Piece commands validate, list, inspect, save, copy, update, and delete WGSL pieces."),
    "feature": ("Feature Dictionary Commands", "Feature commands inspect attached sparse dictionaries, feature summaries, and ranked corpus examples."),
    "trace": ("Circuit Trace Commands", "Trace commands run circuit attribution, poll status, page graph records, and validate faithfulness."),
    "patch": ("Activation Patch Commands", "Activation patch commands run batched residual-stream patching sweeps across layers and token positions."),
    "intervene": ("Model Intervention Commands", "Intervention commands list model probe points and set, list, or clear live feature and activation edits."),
    "history": ("History Commands", "History commands inspect undo and redo steps in the session journal."),
}


def md_cell(text: str) -> str:
    escaped = html.escape(" ".join(str(text).split()), quote=False)
    return escaped.replace("|", "\\|")


def tool_name_of(command_name: str) -> str:
    return command_name.replace(".", "_", 1)


def schema_type_label(prop: dict) -> str:
    if "enum" in prop and isinstance(prop["enum"], list):
        return "enum"
    if "const" in prop:
        return "const"
    t = prop.get("type")
    if isinstance(t, list):
        return "|".join(str(x) for x in t)
    if isinstance(t, str):
        return t
    if "oneOf" in prop or "anyOf" in prop:
        return "union"
    return "value"


def format_args(args_schema: dict) -> str:
    props = args_schema.get("properties") if isinstance(args_schema, dict) else None
    if not props:
        return "None"
    req = set(args_schema.get("required") or [])
    items = []
    for name, spec in props.items():
        tlabel = schema_type_label(spec if isinstance(spec, dict) else {})
        if name in req:
            items.append("`%s` (%s, required)" % (name, tlabel))
        else:
            items.append("`%s` (%s)" % (name, tlabel))
    return ", ".join(items)


def format_flags(cmd: dict) -> str:
    name = cmd["name"]
    effect = cmd.get("effect", "change")
    mutates = bool(cmd.get("mutates"))
    read_only = (not mutates) and (name != "app.launch") and (effect != "run")
    flags = []
    if read_only:
        flags.append("read-only")
    if mutates:
        flags.append("mutates")
    if cmd.get("destructive"):
        flags.append("destructive")
    if cmd.get("frame_thread"):
        flags.append("frame-thread")
    res_props = cmd.get("result", {}).get("properties")
    if isinstance(res_props, dict) and "png_base64" in res_props:
        flags.append("png-image")
    return ", ".join("`%s`" % f for f in flags) if flags else "None"


def load_commands(root: pathlib.Path) -> tuple[list[dict], list[str]]:
    path = root / TABLE_REL
    if not path.is_file():
        return [], ["%s: file does not exist" % TABLE_REL]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], ["%s: invalid JSON: %s" % (TABLE_REL, exc)]
    if not isinstance(data, dict) or data.get("version") != 0 or not isinstance(data.get("commands"), list):
        return [], ["%s: expected version 0 command table object" % TABLE_REL]
    commands = data["commands"]
    if not commands:
        return [], ["%s: command list is empty" % TABLE_REL]
    return commands, []


def render_markdown(commands: list[dict]) -> str:
    agent_cmds = [c for c in commands if c.get("agents", True)]
    session_only = [c for c in commands if not c.get("agents", True)]

    group_order: list[str] = []
    by_group: dict[str, list[dict]] = {}
    for cmd in agent_cmds:
        grp = cmd["name"].split(".", 1)[0]
        if grp not in by_group:
            group_order.append(grp)
            by_group[grp] = []
        by_group[grp].append(cmd)

    lines = [
        "<!-- Generated by scripts/generate_mcp_reference.py. Do not edit by hand. -->",
        "",
        "# MCP Command Reference",
        "",
        "The `caliper-mcp` server exposes every agent-enabled command in `src/builder/commands/command_table.json` as one Model Context Protocol tool.",
        "The script `scripts/generate_mcp_reference.py` generates this page directly from `src/builder/commands/command_table.json`.",
        "Continuous integration runs `python scripts/generate_mcp_reference.py --check` and fails when this page drifts from the command table.",
        "After any edit to `src/builder/commands/command_table.cpp`, run `caliper_builder_command_table --write` and then run `python3 scripts/generate_mcp_reference.py --write`.",
        "",
        "## Tool Mapping and Execution",
        "",
        "Each command with `agents: true` becomes one MCP tool whose name replaces the first dot with an underscore.",
        "For example, `panel.create` becomes `panel_create`, while its MCP `title` keeps the dotted command name.",
        "The tool input schema matches the command `args` schema, and the structured output schema matches `result`.",
        "Commands that return `png_base64` (`capture_frame` and `panel_render`) return MCP `image/png` content alongside the JSON metadata.",
        "The server marks a tool with `readOnlyHint: true` only when `mutates` is false, `effect` is not `run`, and the command is not `app.launch`.",
        "See [MCP Server](../agents/mcp-server.md) and [Caliper API](../agents/http-api.md) for transport, token, and approval rules.",
        "",
        "## Group Summary",
        "",
        "The command table defines %d commands across %d groups: %d MCP tools for agents and %d session-only undo commands."
        % (len(commands), len(group_order), len(agent_cmds), len(session_only)),
        "",
        "| Group | Prefix | MCP Tools | Effects |",
        "|---|---|---:|---|",
    ]

    for grp in group_order:
        items = by_group[grp]
        title = GROUP_INFO.get(grp, ("%s Commands" % grp.capitalize(), ""))[0]
        effects = ", ".join("`%s`" % e for e in sorted({c.get("effect", "change") for c in items}))
        lines.append("| %s | `%s.*` | %d | %s |" % (md_cell(title), grp, len(items), effects))
    lines.append("")

    for grp in group_order:
        items = by_group[grp]
        title, intro = GROUP_INFO.get(
            grp,
            ("%s Commands" % grp.capitalize(), "Commands in the `%s` group." % grp),
        )
        lines.append("## %s" % title)
        lines.append("")
        lines.append(intro)
        lines.append("")
        lines.extend(
            [
                "| MCP Tool | Command | Effect | Flags | Arguments | Summary |",
                "|---|---|---|---|---|---|",
            ]
        )
        for cmd in items:
            tname = tool_name_of(cmd["name"])
            effect = cmd.get("effect", "change")
            flags = format_flags(cmd)
            args_str = format_args(cmd.get("args") or {})
            summary = md_cell(cmd.get("summary", ""))
            lines.append(
                "| `%s` | `%s` | `%s` | %s | %s | %s |"
                % (tname, cmd["name"], effect, flags, args_str, summary)
            )
        lines.append("")

    if session_only:
        lines.extend(
            [
                "## Session-Only Commands",
                "",
                "Commands with `agents: false` belong to interactive user history in the desktop application.",
                "The `caliper-mcp` server omits them from `tools/list` and refuses them for `mcp` and `agent` origins.",
                "",
                "| Command | Effect | Flags | Arguments | Summary |",
                "|---|---|---|---|---|",
            ]
        )
        for cmd in session_only:
            lines.append(
                "| `%s` | `%s` | %s | %s | %s |"
                % (
                    cmd["name"],
                    cmd.get("effect", "change"),
                    format_flags(cmd),
                    format_args(cmd.get("args") or {}),
                    md_cell(cmd.get("summary", "")),
                )
            )
        lines.append("")

    return "\n".join(lines)


def run(root: pathlib.Path, write: bool) -> tuple[list[str], str]:
    commands, problems = load_commands(root)
    if problems:
        return problems, ""
    rendered = render_markdown(commands)
    out_path = root / OUTPUT_REL
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        return [], rendered
    if not out_path.is_file():
        return ["%s: generated file does not exist (run scripts/generate_mcp_reference.py --write)" % OUTPUT_REL], rendered
    if out_path.read_text(encoding="utf-8") != rendered:
        return ["%s: out of date (run scripts/generate_mcp_reference.py --write)" % OUTPUT_REL], rendered
    return [], rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify docs/reference/mcp-commands.md is up to date")
    mode.add_argument("--write", action="store_true", help="write docs/reference/mcp-commands.md")
    args = parser.parse_args(argv)

    problems, _ = run(args.root.resolve(), write=args.write)
    if problems:
        for p in problems:
            print(p)
        print("%d problem(s) in MCP command reference" % len(problems))
        return 1
    if args.write:
        print("Wrote %s" % OUTPUT_REL)
    else:
        print("MCP command reference complete and up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
