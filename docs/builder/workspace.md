---
kind: concept
---

# Workspace

The workspace is the folder that the builder session works on, and this page describes the session, the spec page and the command script.

The app, `caliper dev` and `caliper shot` hold one builder session for a workspace. The workspace holds loaders, specs, saved spec applets, the command journal and the workspace's database. The builder's commands act on it. [CLI](../reference/cli.md#workspace) describes the `--workspace` option and `CALIPER_WORKSPACE`, which name the folder.

## Session Lifecycle

The session costs nothing at start-up. The app opens a workspace that `--workspace` or `CALIPER_WORKSPACE` names on a worker thread, once it has drawn its first frame. It creates the folder and its layout when they are missing. The default workspace opens only when something first needs it, such as a command through Caliper API. An app started without either option thus creates nothing there. When the workspace is open, standard error says so with the time it took, and the Activity pane shows a `workspace` line:

```text
[session] workspace /home/me/work/ws (--workspace) opened in 6 ms
```

A workspace that cannot be opened, such as a path that names a file, leaves the app running without the builder. Standard error and the Activity pane say why, in a `workspace not opened` line.

Each command that changes the workspace adds a line to the Activity pane. The event column shows the command's origin (`ui command`, `agent command`, `mcp command` or `api command`), and the object is the command's name. When the command failed or was denied, the detail is its error code. The command also adds a `[session] command <name> (<origin>)` line to standard error. When the app closes, the session answers the commands still waiting with `command.shutdown` and waits for a running loader to finish. Then it closes the journal and the workspace's database before the process exits.

## Spec Page

Every successful `applet.open` shows the spec page. It makes no difference whether the command came from the command script, Caliper API or an agent through `caliper-mcp`. An agent's `spec.new` or `spec.open` shows the page too, so you see the spec whose edits you review.

One exception applies when a compiled applet's page is showing. An agent never ends your running applet, so the spec opens in the session and the applet's page stays. Activity says `agent opened` with the spec's title and how to reach it. The status bar names the spec as `dashboard <title> open`. Press esc to go Home, then F6 Dashboard on the landing screen to show it.

In all other cases, the page of the applet shown before goes, as with Home. The spec runtime draws the spec in the work area, from the data that the workspace's loaders made. The page bar holds Home and the spec's title, where the applet page has the applet's name. The status bar shows the renderer and the device, the workspace's folder name, and the numbers of panels and sources. It adds `unsaved` when the open spec changed since it was saved. The jobs tray keeps its place above the status bar. The window's title becomes `Caliper - <title>`, and the Activity pane shows an `open` line.

A click inside a panel focuses it, and its frame turns bronze. Esc first gives up a panel's focus, and then goes back to the landing screen when no panel has focus. Home goes back at once. ⌘B Build in the page bar turns on build mode, the builder pane, where you build and change the spec by hand. F3 New dashboard on the landing screen starts a new spec in build mode. Files dropped on the window are imported into the workspace and shown there ([Builder pane](builder-pane.md)). ⌘O Open in browser in the page bar opens the saved spec applet in your default browser. It starts [Caliper API](../agents/http-api.md#app-startup) on loopback when the API does not run.

The page reads the open spec again after each command that changes the workspace, so a later edit from any origin shows at the next frame. It reads a panel's data again only when the panel changed or a loader ran. A panel over a source that the app serves (a view of the attached live session, a feed channel or a metric) also reads again when its data has a new generation. A new snapshot of the view gives a new generation.

A live session that you attach, from the live page or with `--url`, stays attached while the spec page shows. It stays attached after Home and on every other page, so its views keep feeding the spec's `live:` sources. Only `⌘D Detach` on the live page, attaching another session or quitting detaches it. The spec's live panels then show their offline state. When the session's script ends, its sources keep their last data.

Launching a spec applet opens the same page. You launch it with a click or Enter on its row on the landing screen, `CALIPER_AUTOLAUNCH`, `app.launch` or `POST /v0/applets/{id}/launch` with its manifest id. The app first validates the spec, as `caliper check` does. A spec saved in the workspace opens through the session's `applet.open`. That command comes from the `ui` origin for your own launches. It comes from the caller's origin for `app.launch` and `POST /v0/applets/{id}/launch`, so an agent's launch over unsaved changes waits on an approval card. The page appears a frame or two after the command runs, and the launch opens the default workspace if it is not open yet. A spec applet found in another search folder is drawn from its file as it is, over the workspace's data, and does not become the open spec.

The applet is `ACTIVE` in the list while its page shows, and `READY` again after Home, which returns to its row. The landing screen lists the workspace's spec applets under Projects and the others under Instruments. After each `applet.save`, the app lists the workspace's `applets/` folder again. A saved applet thus appears at once, `ACTIVE` and selected while its page shows. While a compiled applet's page is open, this waits until the page closes. In place of a compiled applet's epoch, services and binary, the Inspector shows a spec applet's kind, the spec's format version, its numbers of panels and sources, and its spec file. The file is relative to the workspace for an applet saved there, and otherwise has the home folder written `~`.

`CALIPER_HOME_AFTER` goes Home from the spec page as from other pages. `caliper shot` draws the same page ([caliper shot](../reference/shot-and-serve-commands.md#caliper-shot)). `panel.render` draws one panel of the open spec on a canvas of its own. With `--api`, a saved spec applet of the workspace also opens in a browser, at the link that the app prints: `http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=<id>` ([Caliper API](../agents/http-api.md#get-v0spec)). The browser draws the same layout and look from the data that the app sends.

## Command Script

A testing and demo aid drives the session without Caliper API. `CALIPER_COMMAND_SCRIPT` names a file of JSON lines, one command each, such as `{"name": "spec.new", "args": {"id": "board", "title": "Board"}}`. The origin is `ui` unless a line gives another. The script skips empty lines and lines that start with `#`. The app, `dev` and `shot` read the variables only when they start.

When the workspace is open, the commands run one after another through the command queue, as a client's commands do. If `CALIPER_COMMAND_RESULTS` names a file, each result goes there as one JSON line: `{"line", "name", "id", "ok", "result"}`, or with `error` in place of `result`. The last line is `{"done": true, "commands", "failed"}`. Standard error ends with `[session] script done: <n> commands, <m> failed`. With a script, the workspace opens after the first frame even when it is the default, and `shot` keeps drawing frames until the script is done.

A command can ask for the person's approval. Examples are a destructive command and, from `mcp` and `agent`, an edit of the open spec under review. Such a command never holds the script. Its result is `command.needs_approval` with the approval id, the approval stays pending, and the script goes on. A line with `"approve": true` approves its command as the person would, when it asks. Its result then carries `"approved"` with the approval id.

A script may approve as the person does, so it runs only where nobody watches a window. These places are the [hidden window](../reference/cli.md#hidden-window) (`--hidden`, with the app or `dev`) and `shot`. An app with a visible window ignores the script. It says so on standard error with a `[session]` line, and lists "command script refused" in Activity.

## Related Pages

- [CLI](../reference/cli.md)
- [Builder Pane](builder-pane.md)
- [Shot and Serve Commands](../reference/shot-and-serve-commands.md)
- [Caliper API](../agents/http-api.md)
