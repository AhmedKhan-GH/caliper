---
kind: reference
---

# Shot and Serve Commands

This page describes `caliper shot`, which writes a PNG of an applet page, and `caliper serve`, which runs the hub of wire protocol v3.

## `caliper shot`

Draws the app's applet page for one applet without a window, and writes the last frame as a PNG. It draws the same page that the app draws for the applet. The page has the page bar and the status bar. While a job runs or a failure is kept, it also has the jobs tray's strip above the status bar. All three are absent when `CALIPER_CHROMELESS=1`. The dock space fills the area they leave, with docking by role and the applet's own frame. The tray's strip appears once a job has run for half a second. Thus a shot of an applet that starts a job when it launches needs more than 30 frames to show it.

`shot` finds the applet in [the search order](cli.md#search-order), with the `--applets` folders first. It launches the applet as the app does, draws the frames at about 60 a second, and writes the last one:

```text
caliper shot --applets build --launch com.example.my-viewer --out page.png
wrote /home/me/my-viewer/page.png: com.example.my-viewer at 1600x1000 after 60 frames
```

| Option | Meaning |
|---|---|
| `--launch <id>` | The manifest id of the applet to launch, a spec applet's included, or the workspace id of a spec applet that a command script saves in the same run ([Spec Page](../builder/workspace.md#spec-page)). Required. |
| `--out <file.png>` | The image to write. Its folder must exist, and a file already there is replaced. Required. |
| `--size <W>x<H>` | The size of the image in pixels, each side from 1 to 16384. The default is `1600x1000`. |
| `--frames <N>` | The number of frames drawn, from 1 to 100000. The image is the last one. The default is 60, one second. |

Each option takes its value as the next argument or after `=`, as in `--size=800x500`, and you may give it once. Any of them with another subcommand is a usage error.

The page is drawn at a scale of 1, so a 1600x1000 image shows what a window of 1600 by 1000 points shows on a display without scaling. The image is opaque. As in the app, an applet's windows do not appear in the first frame and are docked in place from the second. Give enough frames for the applet to load what it shows.

Nothing appears on screen. `shot` draws into the renderer's offscreen canvas, the canvas that Caliper Embed uses in `CALIPER_CANVAS_OFFSCREEN` mode, and reads the frame back from it. It creates no window. On macOS the process has no Dock icon and no menu bar. The offscreen canvas needs Metal on macOS or Vulkan on Windows, and there is no fallback renderer. When that renderer cannot start, with `CALIPER_RENDERER=gl`, which asks for the removed OpenGL renderer, and on Linux, `shot` prints `caliper: shot cannot draw offscreen: ` followed by the reason, as in `caliper: shot cannot draw offscreen: the Metal renderer could not start (no Metal device), and there is no fallback renderer until the WebGPU fallback through Dawn lands`, and exits with 1.

`shot` never reads or writes the saved window layout, `imgui.ini`. Every image thus shows the layout of a first run, with each window where its dock role puts it. `CALIPER_DATA_DIR` decides where the stores and the applet's data folder go:

- When `CALIPER_DATA_DIR` is set and not empty, the stores and the applet's data folder `data/<id>/` are in that folder, as they are for the app. An image can thus show data that the applet saved there.
- When it is not set, `shot` creates a temporary folder under the system's temporary folder (`TMPDIR` on macOS and Linux, `TMP` or `TEMP` on Windows) and uses it as the app-data folder. It removes the folder before it exits. The applet starts with an empty data folder, and `shot` writes nothing in the real app-data folder.

In both cases `shot` searches the user applets folder, as the app does, and only reads it. `--workspace` and `CALIPER_WORKSPACE` name the builder's workspace as they do for the app ([Workspace](cli.md#workspace)). Without them the workspace is `workspace/` in the app-data folder, which is the temporary one when `CALIPER_DATA_DIR` is not set. It opens only when a command script, a spec applet or a command of Caliper API needs it. The session's `capture.frame` reads the canvas back at the end of the next frame. Only `CALIPER_DISPLAY_FONT` loads the display face, which only the landing screen's wordmark uses.

For a spec applet, `shot` draws the spec page. The applet is one of these:

- A spec applet that the search finds by its manifest id, such as `local.run42-dashboard`.
- The workspace's applet of that id, when no search folder has an applet with the id and the id is a workspace's applet id (a lowercase letter, then up to 63 lowercase letters, digits, `_` and `-`, with no dots).

Once the workspace is open and the command script, if any, is done, `shot` opens the applet. It uses `applet.open` when the applet is in the workspace and its file otherwise. Then it draws at least 10 more frames so that the plots fit their data, and writes the last one. A command script can therefore build and save the applet in the same run. When the workspace has no such applet, `shot` exits with 1. Whatever `--launch` named, an `applet.open` that succeeds while `shot` runs shows the spec page from then on, as in the app. It can come from the command script or through Caliper API.

```text
CALIPER_COMMAND_SCRIPT=build.jsonl caliper shot --workspace ws --launch run42-dashboard --size 2560x1440 --out board.png
wrote /home/me/board.png: Run 42 at 2560x1440 after 60 frames
```

The applet runs as it runs in the app: its `on_init`, its frames and any jobs it starts. Its own variables, such as `CALIPER_EMBED_AUTOTRAIN`, apply as usual. Leave them unset for an image of the applet's first state.

`shot` exits with 0 when it has written the image, and with 2 for a wrong command line. It exits with 1 in these cases:

- No folder holds an applet with the id: `caliper: no applet with the id <id> in the search folders (caliper list shows them)`. For an id written as a workspace's applet id, the message is `caliper: no applet with the id <id> in the search folders or the workspace (<code>: <reason>)`.
- The applet cannot be launched, with its status from `caliper list`: `caliper: <id> cannot be launched (REFUSED): <reason>`.
- Its `on_init` fails or it faults while launching: `caliper: <id> failed to launch: <reason>`.
- It faults in a frame and is quarantined: `caliper: <id> was quarantined: <reason>`.
- The renderer cannot draw offscreen, as described above.
- The image's folder does not exist, or the image cannot be written: `caliper: cannot write <file>: <reason>`.

`shot` checks the id and the image's folder before it draws or creates anything.

## `caliper serve`

Runs the hub of [wire protocol v3](wire-protocol.md). Producers, such as training scripts, connect out to the hub and each becomes a session. Viewers (the browser viewer page, the notebook widget, the Caliper app) connect to it, list the sessions and attach to one. The hub needs no window, renderer or GPU, and listens on 127.0.0.1 only. [Run a hub](../python/hub-setup.md) explains how to use it, what it keeps and its limits.

| Option | Meaning |
|---|---|
| `--port <n>` | The port to listen on, 0 to 65535. The default is 7340. 0 takes any free port. |
| `--token-file <file>` | Read the session token from the first line of `<file>` instead of making a random one. The token must be 22 to 128 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`. On macOS and Linux, the file must not be readable or writable by other users (`chmod 600`). |
| `--assets <dir>` | Serve the viewer page and its files from `<dir>`, such as the `caliper/_web/` folder of Caliper LIB. Without it, `/` answers with a short note that says where the page is. |
| `--allow-control` | Forward viewers' controls, of [wire protocol 3.1](run-control-messages.md#hub-forwarding), to the script of each viewer's own session: pause, resume, step, setting a registered parameter and running a registered action. Off by default. Without it the hub answers every control with `forbidden`, and viewers show the controls without letting them be changed. |

The options with a value also take the form `--port=<n>`. They apply only to `serve`, except `--allow-control`, which takes no value and also applies with `--api` (see [Caliper API](cli.md#caliper-api)). The token is never given on the command line, where other users could read it from the process list. Without `--token-file`, the hub makes a random token of 32 characters (24 random bytes) for this run.

`serve` prints the viewer link and the producer address. Then it prints one line for each session that starts, resumes, ends or is forgotten:

```text
viewers   http://127.0.0.1:7340/?token=<token>
producers ws://127.0.0.1:7340/v3/stream (address and token in <hub file>)
controls are not forwarded; start with --allow-control to allow them
stop with Ctrl-C
session clp-41222-9f03ab started (train.py, pid 41222)
```

The **hub file** tells producers on the same machine where the hub is. It is `hub.json` in the app-data folder, or the path in `CALIPER_HUB_FILE` when that is set and not empty. The hub writes it when it starts, readable and writable by its owner only (mode 0600), and removes it when it stops. It holds one JSON object:

| Key | Value |
|---|---|
| `version` | 1 |
| `pid` | The hub's process id |
| `port` | The port it listens on |
| `token` | The session token |
| `stream_url` | `ws://127.0.0.1:<port>/v3/stream`, without the token |
| `url` | The viewer link, with the token |

SIGINT (Ctrl-C) or SIGTERM stops the hub. The hub sends BYE `shutdown` on every connection, closes them with code 1001, removes the hub file, prints `hub stopped` and exits with 0. It exits with 1 when the port is taken, the token file cannot be used, or the `--assets` folder does not exist. When the default port is taken, another hub is probably running, so start this one with `--port`.
