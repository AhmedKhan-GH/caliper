# CLI

The `caliper` executable starts the app and also takes subcommands, which make up Caliper CLI. There is no separate program, so there is one thing to install. The executable reads its command line before it opens a window. Every subcommand except `dev` runs without a window, and thus is safe in scripts and CI. Only `caliper` with no subcommand and `caliper dev` start the app. `serve` runs until you stop it, and the other subcommands exit when they finish.

```text
usage: caliper [--instruments <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]] [--hidden]
               [--workspace <dir>]
       caliper dev <folder> [--instruments <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]]
                   [--hidden] [--workspace <dir>]
       caliper list [--instruments <dir>]... [--workspace <dir>]
       caliper check <folder>
       caliper install <folder>
       caliper new <name> [<dir>]
       caliper build <folder>
       caliper info
       caliper shot [--instruments <dir>]... --launch <id> [--size <W>x<H>]
                    [--frames <N>] --out <file.png> [--api [--api-port <n>] [--allow-control]]
                    [--workspace <dir>]
       caliper serve [--port <n>] [--token-file <file>] [--assets <dir>] [--allow-control]
       caliper --help
```

Results go to standard output. Errors go to standard error and start with `caliper:`. Only `shot` opens the metrics, data or artifacts store, in a temporary folder or in `CALIPER_DATA_DIR` (see [`caliper shot`](shot-and-serve-commands.md#caliper-shot)). `list`, `check`, `info`, `new` and `--help` create nothing in the app-data folder. `install` writes only the user instruments folder. `shot` without `CALIPER_DATA_DIR` creates nothing there either.

## App Startup

`caliper` with no subcommand starts the app. It draws with Metal on macOS and Vulkan on Windows and has no fallback renderer. Where that renderer cannot start, the app prints `caliper: the Metal renderer could not start (<reason>), and there is no fallback renderer until the WebGPU fallback through Dawn lands` (Vulkan on Windows) and exits with 1. With `CALIPER_RENDERER=gl` it prints `caliper: CALIPER_RENDERER=gl asks for the OpenGL renderer, which was removed; unset it, and Caliper draws with Metal on macOS and Vulkan on Windows` and exits with 1 ([Renderer Refusals](refusal-messages.md#renderer-refusals)). `--instruments <dir>`, or `--instruments=<dir>`, adds a development folder to the search. You may give it more than once, and the app searches the folders in the order given. It is accepted when starting the app and with `dev`, `list` and `shot`. With any other subcommand it is a usage error, and so is `--instruments` followed by an argument that starts with `--`, such as `--instruments --help`. The app ignores an argument that starts with `-psn_`, which Finder passes on older versions of macOS.

`--url <link>`, or `--url=<link>`, attaches the app to a stream of [wire protocol v3](wire-protocol.md) as soon as it starts. It works as pasting the link into the attach pane does. The link is one of these:

- The page link `http://127.0.0.1:<port>/?token=<token>` that `caliper.serve()` prints.
- The stream link `ws://127.0.0.1:<port>/v3/stream?token=<token>` of a script or a hub.

The app opens on the stream's live page. If the link does not attach, the app stays on its landing page, and the Activity pane and standard error say why. The option is accepted when starting the app and with `dev`. It needs a value and you may give it once. The link holds the stream's session token, which other users of the machine can read from the process list while the app runs. Thus use it only on a machine you do not share. See [Stream Links](../python/stream-links.md).

`caliper dev <folder>` starts the app with `<folder>` searched first, before any `--instruments` folders. The folder is usually an instrument's build folder, such as `build`, or its build output, `build/instruments` (see [Build Folders](#build-folders)). The `scanned` line in the Activity pane names the folder. If `<folder>` is not a folder, `dev` prints `caliper: not a folder: <folder>` and exits with 1 without starting the app. `dev` does not reload an instrument whose library is rebuilt, so quit and start it again.

### Caliper API

`--api` serves [Caliper API](../agents/http-api.md), the HTTP interface to the running app, on 127.0.0.1 while the app runs. It is accepted when starting the app and with `dev` and `shot`. With any other subcommand it is a usage error. `CALIPER_API=1` turns the API on as `--api` does.

| Option | Meaning |
|---|---|
| `--api` | Serve Caliper API. The process prints the address, your session token and the link of the browser viewer of spec-based instruments. Until it exits, it also writes the address, a token for outside agents and the link in `api.json` in the app-data folder, readable by its owner only. Commands from outside agents are reviewed. Your session token is in no file ([Command Origin](../agents/http-api.md#command-origin)). |
| `--api-port <n>` | The port, from 0 to 65535. The default is 7341, and 0 takes any free port. It needs `--api`, and you may give it once. |
| `--allow-control` | The API's stream, which is a hub, forwards viewers' controls to the scripts, as `caliper serve --allow-control` does. It needs `--api`. Off by default. `CALIPER_API=1` alone never turns it on. |

`--api-port` also takes the form `--api-port=<n>`. `--api` takes no value. When the port is taken, the app starts without the API and says why on standard error and in the Activity pane. In the same case `shot` exits with 1. With `--api`, SIGINT and SIGTERM end the frames of `shot` early, and it then writes its image as usual.

### Workspace

The app, `caliper dev` and `caliper shot` hold one builder session for a workspace, a folder that the builder's commands act on. `--workspace <dir>`, or `--workspace=<dir>`, names the folder. If the option is absent, `CALIPER_WORKSPACE` names it when it is set and not empty. Otherwise the workspace is `workspace/` in the app-data folder, which `CALIPER_DATA_DIR` moves ([App-Data Folder](folders-and-environment.md#app-data-folder)). A relative path starts from the working directory.

The option is accepted when starting the app and with `dev`, `shot` and `list`. It needs a value and you may give it once. With any other subcommand it is a usage error. In a build without the builder (`CALIPER_WITH_BUILDER` off, the default on Windows), the option is still accepted. The app then reports the workspace as not opened. [Workspace](../builder/workspace.md) describes the session, the spec page and the command script.

### Hidden Window

`--hidden` is a testing aid. It runs the real app, with its renderer and its frame loop, but never shows its window. `CALIPER_HIDDEN=1` does the same. It is accepted when starting the app and with `dev`, and it takes no value. With any other subcommand it is a usage error. With `--api`, a test or an agent can drive the whole app through [Caliper API](../agents/http-api.md), captures included, with nothing on screen.

In this mode the app starts as the test binaries that open hidden windows do ([Testing](../development/testing.md#pixel-exact-graphics-suite)):

- On macOS the process has no Dock icon and no menu bar, and the system never activates it.
- The window is created invisible and unfocused. If anything makes it visible, the app hides it again at once and says so on standard error.
- The window is 1280 by 800 in screen coordinates on every display, unless `CALIPER_WINDOW_SIZE` sets another size. Thus captures have the same size everywhere, and the framebuffer is that size times the display's scale.
- The app reads and writes no `imgui.ini`, so every run shows the layout of a first run and the saved layout stays as it is.
- The app needs `CALIPER_DATA_DIR` because a hidden app never uses your own app-data folder. Without the variable it stops before it creates anything, prints `caliper: the hidden app (--hidden) has no data folder of its own ...` on standard error and exits with 1 ([App-Data Folder](folders-and-environment.md#app-data-folder)).
- SIGINT and SIGTERM close the window as a click on its close button does, so the full teardown runs and the app removes the API file. `CALIPER_EXIT_AFTER` works as usual.
- `CALIPER_TEST_UI` plays scripted clicks, typing and drags into the live page ([Host Variables](folders-and-environment.md#host-variables)). With `--url`, a test attaches the hidden app to a stream and drives its run bar and its views as a person does.
- `CALIPER_TEST_APPROVALS` names a file of decisions on the approval cards, such as `approve panel.create mcp` ([Host Variables](folders-and-environment.md#host-variables)). A test thus approves or denies an agent's held request as the person does.

Both test variables work only in this mode. At start-up the app prints one line to standard error. It says that the window is not shown, and gives its visibility, its focus, whether the background policy holds and the framebuffer's size:

```text
[hidden] the window is not shown (visible 0, focused 0, background policy 1, framebuffer 2560x1600)
```

## Search Order

The app, `caliper dev`, `caliper list` and `caliper shot` search these folders, in this order:

1. The development folders: the `dev` folder, then the `--instruments` folders in the order given.
2. The folders in `CALIPER_INSTRUMENT_PATH`, separated by `:` (`;` on Windows), in the order given.
3. The user instruments folder: `CALIPER_USER_INSTRUMENTS_DIR` when it is set and not empty, otherwise `instruments/` under the app-data folder.
4. The `instruments/` folder beside the executable, which holds the instruments shipped with the app.

The same folders, and the folders one level down, hold [spec-based instruments](../instruments/manifest.md#spec-search-order). The app, `caliper dev`, `caliper shot` and `caliper list` also search the `instruments/` folder of the builder's workspace ([Workspace](#workspace)) between items 2 and 3. The workspace is the folder of `--workspace`, else of `CALIPER_WORKSPACE`, else `workspace/` in the app-data folder. They only read it, so a workspace that is not open yet is not created.

The host resolves each development folder from items 1 and 2 as [Build Folders](#build-folders) describes. It searches a folder named twice once, at its first position, and skips empty entries. When several folders hold copies of one instrument id, the host loads the highest version that passes every check. At equal versions it loads the copy found first, and the other copies that pass are `SHADOWED`. [Several Copies of One Instrument](../development/compatibility-and-epochs.md#several-copies-of-one-instrument) states the rule, and [Folders and Environment](folders-and-environment.md) gives the folders on each platform.

## Build Folders

`caliper_add_instrument()` puts an instrument's library and manifest in the `instruments/` folder of its build folder. You can name the build folder itself, because a development folder with no manifest (`<stem>.caliper.toml`) of its own stands for its `instruments/` subfolder when that subfolder holds one. This applies to the `dev` folder, the `--instruments` folders, the folders in `CALIPER_INSTRUMENT_PATH`, and the folders given to `check` and `install`. A folder that holds a manifest of its own is used as given, even when it also has an `instruments/` subfolder. The user instruments folder and the `instruments/` folder beside the executable are always used as given.

## Subcommands

Two other pages describe the subcommands:

- [Instrument Commands](instrument-commands.md) covers `caliper list`, `caliper check`, `caliper install`, `caliper new`, `caliper build` and `caliper info`.
- [Shot and Serve Commands](shot-and-serve-commands.md) covers `caliper shot` and `caliper serve`.

### `caliper --help`

Prints the usage to standard output and exits with 0. `-h` and `help` do the same.

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A check failed, nothing was found, or a file operation failed. The app and `dev` also exit with 1 when the renderer cannot start or `CALIPER_RENDERER=gl` is set. `build` also exits with 1 when CMake, a compiler or the SDK is missing or the build fails. `dev` exits with 1 when its folder is not a folder. `shot` exits with 1 when its instrument cannot be launched or is quarantined, when the renderer cannot draw offscreen, or when Caliper API cannot listen. `serve` exits with 1 when it cannot listen or cannot use its token file or assets folder. On Windows, `serve`, `build`, `--api` and `--url` exit with 1 before anything starts ([Windows Limits](#windows-limits)). A test or hidden run that would use your own app-data folder exits with 1 too: the hidden app, and `shot` in a test run, without `CALIPER_DATA_DIR` ([App-Data Folder](folders-and-environment.md#app-data-folder)). |
| 2 | The command line is wrong. Causes are an unknown command or option, a missing or extra operand, and `--instruments`, `--url`, `--api`, `--api-port`, `--hidden` or `--workspace` with a subcommand that does not take it. More causes are `--url` without a link or given twice, `--workspace` without a folder or given twice, and `--api-port` or `--allow-control` without `--api` when starting the app, `dev` or `shot`. Others are an option of `shot` or `serve` with another subcommand, a missing or unusable value of `--launch`, `--out`, `--size` or `--frames`, a port out of range, a URL given to `install`, and a name that `new` cannot use. For an unknown command or option and for wrong operands, the usage follows the error on standard error. |

## Windows Limits

The Windows build refuses `caliper serve`, `caliper build`, `--api` (when starting the app, with `dev` and with `shot`) and `--url` before anything starts. It prints one line, such as `caliper: caliper serve is not available on Windows yet`, and exits with 1. A wrong command line is still a usage error with exit code 2. `CALIPER_API=1` is ignored on Windows, and a line on standard error says so. Every other subcommand behaves as this page describes.

## Environment Variables

| Variable | Effect |
|---|---|
| `CALIPER_INSTRUMENT_PATH` | Adds development folders after the `--instruments` folders. A build folder in it stands for its `instruments/` subfolder, as [Build Folders](#build-folders) describes. Read when starting the app, and by `dev`, `list` and `shot`. |
| `CALIPER_USER_INSTRUMENTS_DIR` | Replaces the user instruments folder, for the app, `dev`, `list`, `install` and `shot`. |
| `CALIPER_DATA_DIR` | Moves the whole app-data folder: the stores, `imgui.ini`, `data/`, `live/` and, unless `CALIPER_USER_INSTRUMENTS_DIR` is set, the user instruments folder `instruments/`. `check` reads its data-schema stamps there and `install` copies into its `instruments/`. `list`, `check`, `info`, `new` and `--help` create nothing in it, whether it is set or not. `shot` keeps its stores and the instrument's data folder there when it is set, and in a temporary folder when it is not. |
| `CALIPER_CHROMELESS` | `1` draws the instrument page without the page bar, the status bar and the jobs tray, in the app and in `shot`. |
| `CALIPER_HUB_FILE` | Replaces the path of the hub file that `serve` writes, `hub.json` in the app-data folder, when it is set and not empty. |
| `CALIPER_API` | `1` serves [Caliper API](../agents/http-api.md) as `--api` does, when starting the app and with `dev` and `shot`. Ignored on Windows ([Windows Limits](#windows-limits)). |
| `CALIPER_HIDDEN` | `1` runs the app with its window never shown, as `--hidden` does: a testing aid (see [Hidden Window](#hidden-window)). |
| `CALIPER_WORKSPACE` | The builder's workspace folder, when it is set and not empty and `--workspace` is not given, for the app, `dev` and `shot` (see [Workspace](#workspace)). |
| `CALIPER_COMMAND_SCRIPT`, `CALIPER_COMMAND_RESULTS` | A file of commands that the hidden app (`--hidden`, with the app or `dev`) or `shot` runs through the builder session once its workspace is open, and the file that receives their results: a testing and demo aid (see [Command Script](../builder/workspace.md#command-script)). A visible app ignores them. |
| `CALIPER_SDK_DIR` | The prefix of the installed SDK that `build` uses, when it is set and not empty. |
| `CALIPER_CMAKE` | The cmake that `build` runs, when it is set and not empty. |

[Folders and Environment](folders-and-environment.md) lists every variable the host reads.
