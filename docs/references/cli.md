# Caliper CLI

The `caliper` executable starts the app and also takes subcommands, which make up Caliper CLI. There is no separate program, so there is one thing to install. The executable reads its command line before anything opens a window. Every subcommand except `dev` runs without opening a window: `list`, `check`, `install`, `new`, `build`, `info`, `shot`, `serve` and `--help` never start the app, so they are safe in scripts and CI. `serve` runs until it is stopped; the others exit when they are done. Only `caliper` with no subcommand and `caliper dev` start the app. `shot` draws the app's applet page into an offscreen canvas and writes it as a PNG, and shows nothing on screen. The parser and the subcommands are in `src/host/host_cli.h`, and `src/main.cpp` dispatches to them.

```text
usage: caliper [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]] [--hidden]
               [--workspace <dir>]
       caliper dev <folder> [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]]
                   [--hidden] [--workspace <dir>]
       caliper list [--applets <dir>]... [--workspace <dir>]
       caliper check <folder>
       caliper install <folder>
       caliper new <name> [<dir>]
       caliper build <folder>
       caliper info
       caliper shot [--applets <dir>]... --launch <id> [--size <W>x<H>]
                    [--frames <N>] --out <file.png> [--api [--api-port <n>] [--allow-control]]
                    [--workspace <dir>]
       caliper serve [--port <n>] [--token-file <file>] [--assets <dir>] [--allow-control]
       caliper --help
```

Results go to standard output, and errors go to standard error, each starting with `caliper:`. Apart from `shot`, which runs an applet and opens the stores in a temporary folder or in `CALIPER_DATA_DIR` (see [caliper shot](#caliper-shot)), no subcommand opens the metrics, data or artifacts store. `list`, `check`, `info`, `new` and `--help` create nothing in the app-data folder, `install` writes only the user applets folder, and `shot` without `CALIPER_DATA_DIR` creates nothing there either.

## Starting the app

`caliper` with no subcommand starts the app. `--applets <dir>`, or `--applets=<dir>`, adds a development folder to the search. It may be given more than once, and the folders are searched in the order given. It is accepted when starting the app and with `dev`, `list` and `shot`; with any other subcommand it is a usage error, and so is `--applets` followed by an argument that starts with `--`, such as `--applets --help`. An argument that starts with `-psn_`, which Finder passes to an app it launches on older versions of macOS, is ignored.

`--url <link>`, or `--url=<link>`, attaches the app to a stream of [wire protocol v3](wire-protocol.md) as soon as it starts, as pasting the link into the attach pane does: the page link `http://127.0.0.1:<port>/?token=<token>` that `caliper.serve()` prints, or the stream link `ws://127.0.0.1:<port>/v3/stream?token=<token>` of a script or a hub. The app opens on the stream's live page. A link that does not attach leaves the app on its landing page, and the Activity pane and standard error say why. It is accepted when starting the app and with `dev`, may be given once, and needs a value. The link holds the stream's session token, which other users of the machine can read from the process list while the app runs, so use it on a machine you do not share. See [Attach to a running script](../guides/attach-to-a-running-script.md#attach-to-a-stream-by-its-link).

`caliper dev <folder>` starts the app with `<folder>` searched first, before any `--applets` folders. The folder is usually an applet's build folder, such as `build`, or its build output, `build/applets` (see [Build folders](#build-folders)). The Activity pane's `scanned` line names the folder that was searched. If `<folder>` is not a folder, `dev` prints `caliper: not a folder: <folder>` and exits with 1 without starting the app. `dev` does not reload an applet whose library is rebuilt; quit and start it again.

### Caliper API

`--api` serves [Caliper API](api.md), the HTTP interface to the running app, on 127.0.0.1 while the app runs. It is accepted when starting the app and with `dev` and `shot`, whose offscreen canvas is the app's windowless mode; with any other subcommand it is a usage error. Setting `CALIPER_API=1` turns the API on as `--api` does.

| Option | Meaning |
|---|---|
| `--api` | Serve Caliper API. The process prints the address and the session token, and the link of the browser viewer of spec applets, and writes them in `api.json` in the app-data folder, readable by its owner only, until it exits. |
| `--api-port <n>` | The port, from 0 to 65535. The default is 7341, and 0 takes any free port. It needs `--api`, and may be given once. |
| `--allow-control` | The API's stream, which is a hub, forwards viewers' controls to the scripts, as `caliper serve --allow-control` does. It needs `--api`. Off by default. `CALIPER_API=1` alone never turns it on. |

`--api-port` also takes the form `--api-port=<n>`. `--api` takes no value. When the port is taken, the app starts without the API and says why on standard error and in the Activity pane, and `shot` exits with 1. With `--api`, SIGINT and SIGTERM end the frames of `shot` early, and it then writes its image as usual.

### The workspace

The app, `caliper dev` and `caliper shot` hold one builder session for a workspace: a folder of loaders, specs, saved spec applets, the command journal and the workspace's database, which the builder's commands act on. `--workspace <dir>`, or `--workspace=<dir>`, names the folder, and `CALIPER_WORKSPACE`, when it is set and not empty, names it when the option is not given. Without either, the workspace is `workspace/` in the app-data folder, which `CALIPER_DATA_DIR` moves with the rest ([Folders and environment variables](environment.md#the-app-data-folder)). A relative path is taken from the working directory. The option is accepted when starting the app and with `dev`, `shot` and `list`, may be given once and needs a value; with any other subcommand it is a usage error. The code is in `src/host/builder/app_session.h`.

The session costs nothing at start-up. A workspace named by `--workspace` or `CALIPER_WORKSPACE` is opened on a worker thread once the app has drawn its first frame, and the folder and its layout are created when they are missing. The default workspace is opened only when something first needs it, such as a command through Caliper API, so the app started without either creates nothing there. Once the workspace is open, standard error says so with the time it took, and the Activity pane shows a `workspace` line:

```text
[session] workspace /home/me/work/ws (--workspace) opened in 6 ms
```

A workspace that cannot be opened, such as a path that names a file, leaves the app running without the builder: standard error and the Activity pane say why, in a `workspace not opened` line. Each command that changes the workspace adds a line to the Activity pane, with the command's origin in the event column (`ui command`, `agent command`, `mcp command` or `api command`), its name as the object and, when it failed or was denied, its error code as the detail, and a `[session] command <name> (<origin>)` line to standard error. When the app closes, the session answers the commands still waiting with `command.shutdown`, waits for a running loader to finish, and closes the journal and the workspace's database before the process exits.

In a build without the builder (`CALIPER_WITH_BUILDER` off, the default on Windows), the option is still accepted, and a workspace that it or `CALIPER_WORKSPACE` names is reported as not opened, because the builder is not in the build.

#### The spec page

Every `applet.open` that succeeds, whether it came from the command script, from Caliper API or from an agent through `caliper-mcp`, shows the spec page: an applet's page shown before goes, as with Home, and the spec of that applet is drawn in the work area by the spec runtime, from the data that the workspace's loaders made. The page bar holds Home and the spec's title, where an applet's name goes on the applet page, and the status bar shows the renderer and the device, then the workspace's folder name and the numbers of panels and sources, and `unsaved` when the open spec has changed since it was saved. The jobs tray keeps its place above the status bar. A click inside a panel focuses it, and its frame turns bronze. Esc first gives a panel's focus up, and goes back to the landing screen when no panel has it; Home goes back at once. ⌘B Build in the page bar turns on build mode, the builder pane, in which the spec is built and changed by hand; F3 New dashboard on the landing screen starts a new spec in build mode, and files dropped on the window are imported into the workspace and shown there ([Builder pane](builder-pane.md)). The page reads the open spec again after each command that changed the workspace, so a later edit, from any origin, shows at the next frame; it reads a panel's data again only when the panel changed, a loader ran, or, for a panel over a source that the app serves (a view of the attached live session, a feed channel or a metric), when that data has a new generation, which a new snapshot of the view gives. A live session that was attached when the page opened, from the live page or with `--url`, stays attached while the spec page shows, so that its views keep feeding the spec's `live:` sources; Home from the spec page, or launching another applet, detaches it, and the spec's live panels then show their offline state. The window's title becomes `Caliper - <title>`, and the Activity pane shows an `open` line.

Launching a spec applet opens the same page: a click or Enter on its row on the landing screen, `CALIPER_AUTOLAUNCH`, `app.launch` and `POST /v0/applets/{id}/launch` with its manifest id. Its spec is validated first, as `caliper check` does. One saved in the workspace opens through the session's `applet.open` from the `ui` origin, so the page appears once that command has run, a frame or two later, and opens the default workspace when it is not open yet; one found in another search folder is drawn from its file as it is, over the workspace's data, without becoming the open spec. The applet is `ACTIVE` in the list while the page shows it and `READY` again after Home, which returns to its row. After each `applet.save` the app lists the workspace's `applets/` folder again, so a saved applet appears on the landing screen at once; while a compiled applet's page is open this waits until it closes. The Inspector shows a spec applet's kind, the spec's format version, its numbers of panels and sources, and its spec file, relative to the workspace for an applet saved there and with the home folder written `~` otherwise, in place of a compiled applet's epoch, services and binary. `CALIPER_HOME_AFTER` goes Home from the spec page as it does from the other pages. `caliper shot` draws the same page ([caliper shot](#caliper-shot)), and `panel.render` draws one panel of the open spec with the same code on a canvas of its own. The code is in `src/host/builder/spec_page.h`. With `--api`, a saved spec applet of the workspace also opens in a browser, drawn with the same layout and look from the data the app sends, at the link the app prints, `http://127.0.0.1:<port>/v0/spec?token=<viewer token>&applet=<id>` ([Caliper API](api.md#get-v0spec)).

A testing and demo aid drives the session without Caliper API: `CALIPER_COMMAND_SCRIPT` names a file of JSON lines, one command each, such as `{"name": "spec.new", "args": {"id": "board", "title": "Board"}}`. The origin is `ui` unless a line gives another, and empty lines and lines that start with `#` are skipped. Once the workspace is open, the commands run one after another through the command queue, as a client's do, and when `CALIPER_COMMAND_RESULTS` names a file, each result is written there as one JSON line, `{"line", "name", "id", "ok", "result"}` or with `error` in place of `result`, followed by `{"done": true, "commands", "failed"}`. Standard error ends with `[session] script done: <n> commands, <m> failed`. A command that asks for the person's approval, such as a destructive one or, from `mcp` and `agent`, an edit of the open spec under review, never holds the script: its result is `command.needs_approval` with the approval id, the approval stays pending, and the script goes on. A line with `"approve": true` approves its command as the person would, when it asks, and its result then carries `"approved"` with the approval id. With a script, the workspace opens after the first frame even when it is the default, and `shot` keeps drawing frames until the script is done. The variables are read only when the app, `dev` or `shot` start.

### The hidden window

`--hidden` is a testing aid: it runs the real app, with its renderer and its frame loop, but never shows its window. Setting `CALIPER_HIDDEN=1` does the same. It is accepted when starting the app and with `dev`; with any other subcommand it is a usage error, and it takes no value. With `--api` it lets a test or an agent drive the whole app through [Caliper API](api.md), captures included, with nothing on screen.

In this mode the app starts as the test binaries that open hidden windows do ([Testing](../explanations/testing.md#the-pixel-exact-graphics-suite)):

- On macOS the process has no Dock icon and no menu bar, and it is never activated: the activation policy is `NSApplicationActivationPolicyProhibited` before and after GLFW starts, and GLFW builds no menu bar.
- The window is created invisible and unfocused, and nothing shows it later. Should anything make it visible, the app hides it again at once and says so on standard error.
- The window is 1280 by 800 in screen coordinates on every display, unless `CALIPER_WINDOW_SIZE` sets another size, so captures have the same size everywhere: the framebuffer is that size times the display's scale.
- The app reads and writes no `imgui.ini`, so every run shows the layout of a first run and the saved layout is left alone. Set `CALIPER_DATA_DIR` as well to keep the stores and the API file out of the app-data folder.
- SIGINT and SIGTERM close the window as a click on its close button would, so the full teardown runs and the API file is removed. `CALIPER_EXIT_AFTER` works as usual.
- `CALIPER_TEST_UI` plays scripted clicks, typing and drags into the live page, only in this mode ([Host variables](environment.md#host-variables)). With `--url`, a test attaches the hidden app to a stream and drives its run bar and its views as a person would.
- `CALIPER_TEST_APPROVALS` names a file of decisions on the approval cards, such as `approve panel.create mcp`, only in this mode ([Host variables](environment.md#host-variables)), so a test approves or denies an agent's held request as the person would.

At start-up the app prints one line to standard error that says the window is not shown, with its visibility, its focus, whether the background policy holds and the framebuffer's size:

```text
[hidden] the window is not shown (visible 0, focused 0, background policy 1, framebuffer 2560x1600)
```

## The search order

The app, `caliper dev`, `caliper list` and `caliper shot` search these folders, in this order:

1. The development folders: the `dev` folder, then the `--applets` folders in the order given.
2. The folders in `CALIPER_APPLET_PATH`, separated by `:` (`;` on Windows), in the order given.
3. The user applets folder: `CALIPER_USER_APPLETS_DIR` when it is set and not empty, otherwise `applets/` under the app-data folder.
4. The `applets/` folder beside the executable, which holds the applets shipped with the app.

[Spec applets](manifest.md#where-spec-applets-are-found) are found in the same folders and one folder down. The app, `caliper dev`, `caliper shot` and `caliper list` also search the builder's workspace's `applets/` folder ([The workspace](#the-workspace)) between items 2 and 3: the folder of `--workspace`, else of `CALIPER_WORKSPACE`, else `workspace/` in the app-data folder. They only read it, so a workspace that has not been opened yet is not created.

Each development folder, from items 1 and 2, is resolved as described in [Build folders](#build-folders). A folder named twice is searched once, at its first position, and empty entries are skipped. When several folders hold copies of one applet id, the host loads the highest version that passes every check, and at equal versions the copy found first; the other copies that pass are `SHADOWED`. [Several copies of one applet](../explanations/compatibility.md#several-copies-of-one-applet) states the rule, and [Folders and environment variables](environment.md) gives the folders on each platform.

## Build folders

`caliper_add_applet()` puts an applet's library and manifest in the `applets/` folder of its build folder. So that the build folder itself can be named, a development folder that holds no manifest (`<stem>.caliper.toml`) of its own, but whose `applets/` subfolder holds one, stands for that subfolder. This applies to the `dev` folder, the `--applets` folders, the folders in `CALIPER_APPLET_PATH`, and the folders given to `check` and `install`. A folder that holds a manifest of its own is used as given, even when it also has an `applets/` subfolder. The user applets folder and the `applets/` folder beside the executable are always used as given.

## Subcommands

### caliper list

Lists every applet in the search folders, without opening any library. Applets are ordered by the position of their folder in the search order, then by id, then from the highest version down. Each applet takes one line, with its status, id, version, kind and folder in columns. The kind is `library` for a compiled applet and `spec` for a [spec applet](manifest.md#spec-applets), whose folder is the applet's own folder. A refused, failed or shadowed applet has its reason on the next line, indented under the id:

```text
READY     dev.caliper.pulse-scope  0.1.0  library  <development folder>
SHADOWED  dev.caliper.pulse-scope  0.1.0  library  <user applets folder>
          Version 0.1.0 in <development folder> is used instead.
READY     local.loss-view          0.1.0  spec     <user applets folder>/loss_view
READY     dev.caliper.hello        0.1.0  library  <applets folder beside the executable>
```

`--workspace <dir>` names the workspace whose spec applets are listed; without it `list` lists those of `CALIPER_WORKSPACE` or of the default workspace, as the app does.

The status is `READY`, `REFUSED`, `FAILED` or `SHADOWED`; the reasons are those in [Refusal messages](refusals.md). Folders are printed as absolute paths. `list` makes only the checks of the listing: it reads a spec applet's manifest and the first lines of its spec, and does not validate the spec. When no folder holds an applet, `list` prints `no applets found`. It always exits with 0.

### caliper check <folder>

Runs the checks that the host makes before it opens a library on every applet in one folder, and only that folder, with this host's capabilities (see `caliper info`). It opens no library. The data-schema check reads the stamps in the app-data folder. The output is the same lines as `list`, then a summary:

```text
READY  dev.caliper.pulse-scope  0.1.0  library  <folder>
1 of 1 applets pass the checks
```

When `<folder>` is a build folder whose applets are in its `applets/` subfolder (see [Build folders](#build-folders)), `check` uses that subfolder and says so first:

```text
using the applets in <folder>/applets
READY  dev.caliper.pulse-scope  0.1.0  library  <folder>/applets
1 of 1 applets pass the checks
```

`check` validates each [spec applet](manifest.md#spec-applets) that passes the listing checks in full, with the builder's spec library, as a launch does, and prints every error and warning of the validator under it, each with the spec's file name, its line and column, its severity and its stable code. A spec with an error fails, and its first error is its reason; warnings do not fail it. `<folder>` may be the folder of one spec applet, or a folder that holds spec applets one folder down, such as a workspace's `applets/` folder:

```text
FAILED  local.broken     0.1.0  spec  <folder>/broken
        broken.spec.toml:9:14: error spec.unknown_source: The binding "nope.loss" names "nope", which is not a source of this spec.
READY   local.loss-view  0.1.0  spec  <folder>/loss_view
        loss_view.spec.toml:6:1: warning spec.unused_source: No panel binds the source "metrics", directly or through from.
1 of 2 applets pass the checks
```

A host built without the builder lists every spec applet as `REFUSED`, so it does not pass there.

An applet passes when it is `READY` or `SHADOWED`. `check` exits with 0 when every applet passes, and with 1 when any does not, when `<folder>` is not a folder (`caliper: not a folder: <folder>`), or when neither it nor its `applets/` subfolder holds an applet (`caliper: no applets in <folder>`).

### caliper install <folder>

Runs the same checks on a local folder, and when every applet passes, copies each applet into the user applets folder: its library, then its asset folder `assets/<stem>/` beside the library when there is one, then its manifest. An applet that reads files from beside its library keeps them in that folder, and its own build stages them there, as the in-tree applets do; an installed asset folder replaces the old one as a whole. Other files beside the library are not copied. A build folder is resolved as for `check`, with the same `using the applets in` line.

The install is all or nothing for the whole folder. Every file and asset folder is first copied to a hidden temporary name beside its destination, and only then renamed into place, each manifest after its library and assets, so the host never sees half a library or a manifest without its library. When a step fails, `install` removes what it placed, puts back what it replaced, and leaves the user applets folder as it was. The output is the lines of `check`, then one line per applet:

```text
installed dev.caliper.pulse-scope 0.1.0 in <user applets folder>
```

When any applet fails a check, `install` copies nothing, prints `caliper: nothing installed; N of M applets do not pass the checks`, and exits with 1. `install` does not take spec applets, which live in a workspace: when the folder holds one, it copies nothing, prints `caliper: nothing installed; <id> is a spec applet, which install does not take`, and exits with 1. It also exits with 1 when `<folder>` is the user applets folder itself, when that folder cannot be created, or when a copy fails, which it reports as `caliper: nothing installed; <reason>`. A URL in place of a folder is a usage error (`caliper: install takes a local folder, not <url>`), with exit code 2.

### caliper new <name> [<dir>]

Creates an applet project from a template built into the executable. `<name>` is either a short name or a full applet id:

- A short name is lowercase letters, digits and `-`, starts with a letter, does not end with `-`, and has at most 64 characters. The applet's id is `dev.example.<name>`.
- The stem names the CMake targets `<stem>` and `<stem>_tests`, so neither may be a name that CMake or one of its generators reserves (`all`, `all_build`, `clean`, `depend`, `edit_cache`, `help`, `install`, `list_install_components`, `package`, `package_source`, `preinstall`, `rebuild_cache`, `run_tests`, `test` and `zero_check`) or start with `caliper_`, which the SDK uses. `new` refuses such a name with `caliper: not a usable applet name: <name> (the CMake target "<target>" is reserved; choose another name)`.
- A full id, such as `com.example.my-viewer`, is used as given, and its last part is the short name.

From the short name, `new` derives the stem (hyphens become underscores: `my_viewer`), the display name (`My Viewer`) and the class name (`MyViewerApplet`). The project goes into `<dir>`, or into a folder named after the short name in the current folder. `new` writes nothing and exits with 1 when the folder already exists, and exits with 2 for a name that breaks the rules above.

| File | Contents |
|---|---|
| `CMakeLists.txt` | Finds `caliper-sdk` with this host's SDK version as the minimum, builds the applet with `caliper_add_applet()`, and builds and registers the test |
| `<stem>.caliper.toml` | The manifest: the id, the display name, version `0.1.0`, this host's ABI epoch, this host's version as `min_host`, and `caliper.ui.v1` and `caliper.log.v1` as required services |
| `<stem>.cpp` | One window, docked in the centre through a dock hint, with a Play and Pause button and a sine plot, and the `CALIPER_APPLET` descriptor |
| `tests/test_<stem>.cpp` | A doctest case on the SDK's fixture host that checks the descriptor against the manifest and runs the lifecycle without a window |

```text
created <dir> for applet dev.example.my-viewer
  CMakeLists.txt
  my_viewer.caliper.toml
  my_viewer.cpp
  tests/test_my_viewer.cpp
```

The comment at the top of the generated `CMakeLists.txt` gives the next steps: configure with `-DCMAKE_PREFIX_PATH=<sdk prefix>`, build, run `ctest --test-dir build`, and run `caliper dev build/applets`. [caliper build](#caliper-build) does the first two and the checks in one step.

### caliper build <folder>

Configures and builds an applet's folder against the installed SDK with CMake, then runs the checks of [caliper check](#caliper-check) on the result. The folder must hold a `CMakeLists.txt` that finds `caliper-sdk`, such as one that `caliper new` writes. The build goes into `<folder>/build`, in the Release configuration, and the library and its manifest land in `<folder>/build/applets`, where `caliper dev` and `caliper install` take them. It runs these steps, and prints what CMake prints as it comes:

```sh
cmake -S <folder> -B <folder>/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<sdk prefix>
cmake --build <folder>/build --config Release --parallel <cores>
caliper check <folder>/build
```

```text
caliper build my-viewer
using the Caliper SDK in /home/me/caliper-sdk
configuring /home/me/my-viewer/build with /usr/local/bin/cmake
...
using the applets in /home/me/my-viewer/build/applets
READY  dev.example.my-viewer  0.1.0  /home/me/my-viewer/build/applets
1 of 1 applets pass the checks
```

The SDK's prefix is found without being named. `build` takes the first of these that holds `lib/cmake/caliper-sdk/caliper-sdk-config.cmake`:

1. `CALIPER_SDK_DIR`, when it is set and not empty. A folder without the SDK is an error.
2. The prefix the executable is installed in: the folder above the executable's folder, then the executable's folder.
3. `caliper-sdk` in the home folder, where the [tutorials](../tutorials/development-basics.md#the-development-loop) install the SDK.

When none of them holds the SDK, `build` says where it looked and lets CMake search its own places, such as `CMAKE_PREFIX_PATH` in the environment and the system prefixes.

`build` runs `CALIPER_CMAKE` when it is set and not empty, else `cmake` from `PATH`, else, on macOS, the command of CMake.app in `/Applications`. CMake chooses the generator and the compiler as usual, so `CMAKE_GENERATOR` and `CXX` in the environment apply. A folder that was built before is configured and built again incrementally.

`build` exits with 0 when the build passes the checks, with 2 for a wrong command line, and with 1 when:

- `<folder>` is not a folder or holds no `CMakeLists.txt`;
- no cmake is found: `caliper: cmake was not found: install CMake 3.18 or later and put it on PATH, or set CALIPER_CMAKE to its path`, or `CALIPER_CMAKE` names something that is not a program;
- `CALIPER_SDK_DIR` names a folder without the SDK;
- the configure fails. When CMake found no working C++ compiler, no build tool for its generator, or no SDK, a last line says which and how to install it, such as `caliper: no Caliper SDK was found. Install it from a Caliper build with cmake --install <build folder> --prefix <prefix> --component sdk, then set CALIPER_SDK_DIR=<prefix> or use the prefix ~/caliper-sdk`;
- the build fails, or an applet in the build folder fails the checks.

`build` is built and tested on macOS. The Windows build refuses it (see [On Windows](#on-windows)); configure and build there with CMake as [SDK package](sdk-package.md#using-the-package) describes.

### caliper info

Prints the host's version, its ABI epoch, the SDK version it was built with, its libtorch version, its UI-stack hash, and one line for each service it serves, in sorted order. A value the build does not define prints as `unknown` for the SDK and `none` for libtorch and the UI-stack hash. From a release build:

```text
host 0.6.0
abi_epoch 2
sdk 0.2.0
libtorch 2.5.1
ui_stack 03c557182adc5cd9
service caliper.artifacts.v1
service caliper.data.v1
...
service caliper.ui.v1_1
```

### caliper shot

Draws the app's applet page for one applet without a window, and writes the last frame as a PNG. It runs the code that the app runs for the applet page (`src/host/ui/applet_page.h`): the page bar, the status bar and, while a job runs or a failure is kept, the jobs tray's strip above the status bar, all three unless `CALIPER_CHROMELESS=1`, then the dock space over the area they leave, docking by role and the applet's own frame. The tray's strip appears once a job has run for half a second, so a shot of an applet that starts a job when it launches needs more than 30 frames to show it. It finds the applet in [the search order](#the-search-order), with the `--applets` folders first, launches it as the app launches an applet, draws the frames at about 60 a second, and writes the last one:

```text
caliper shot --applets build --launch com.example.my-viewer --out page.png
wrote /home/me/my-viewer/page.png: com.example.my-viewer at 1600x1000 after 60 frames
```

| Option | Meaning |
|---|---|
| `--launch <id>` | The manifest id of the applet to launch, a spec applet's included, or the workspace id of a spec applet that a command script saves in the same run ([The spec page](#the-spec-page)). Required. |
| `--out <file.png>` | The image to write. Its folder must exist, and a file already there is replaced. Required. |
| `--size <W>x<H>` | The size of the image in pixels, each side from 1 to 16384. The default is `1600x1000`. |
| `--frames <N>` | The number of frames drawn, from 1 to 100000; the image is the last one. The default is 60, one second. |

Each option takes its value as the next argument or after `=`, as in `--size=800x500`, and may be given once. Any of them with another subcommand is a usage error.

The page is drawn at a scale of 1, so a 1600x1000 image shows what a window of 1600 by 1000 points shows on a display without scaling, and the image is opaque. As in the app, an applet's windows do not appear in the first frame and are docked in place from the second; give enough frames for the applet to load what it shows.

Nothing appears on screen. `shot` draws into the renderer's offscreen canvas, the canvas that Caliper Embed uses in `CALIPER_CANVAS_OFFSCREEN` mode, and reads the frame back from it. It starts no GLFW and creates no window, and on macOS it first sets the activation policy to `NSApplicationActivationPolicyProhibited`, so the process has no Dock icon and no menu bar. The offscreen canvas needs Metal on macOS or Vulkan on Windows. With the OpenGL fallback (`CALIPER_RENDERER=gl`, and on Linux), or when no GPU can make the canvas, `shot` prints `caliper: shot cannot draw offscreen: ...` and exits with 1.

`shot` never reads or writes the saved window layout, `imgui.ini`, so every image shows the layout of a first run, with each window where its dock role puts it. Where the stores and the applet's data folder go depends on `CALIPER_DATA_DIR`:

- When `CALIPER_DATA_DIR` is set and not empty, the stores and the applet's data folder `data/<id>/` are in that folder, as they are for the app, so an image can show data that the applet saved there.
- When it is not set, `shot` creates a temporary folder under the system's temporary folder (`TMPDIR` on macOS and Linux, `TMP` or `TEMP` on Windows), uses it as the app-data folder, and removes it before it exits. The applet starts with an empty data folder, and nothing is written in the app-data folder.

In both cases `shot` searches the user applets folder, as the app does, and only reads it. `--workspace` and `CALIPER_WORKSPACE` name the builder's workspace as they do for the app ([The workspace](#the-workspace)); without them the workspace is `workspace/` in the folder above, the temporary one when `CALIPER_DATA_DIR` is not set, and it opens only when a command script, a spec applet or a command of Caliper API needs it. The session's `capture.frame` reads the canvas back at the end of the frame drawn next. The display face, which only the landing screen's wordmark uses, is loaded only when `CALIPER_DISPLAY_FONT` names it.

For a spec applet, `shot` draws the spec page: a spec applet that the search finds by its manifest id, such as `local.run42-dashboard`, or, when no search folder has an applet with the id and the id is written as a workspace's applet id (a lowercase letter, then up to 63 lowercase letters, digits, `_` and `-`, with no dots), the workspace's applet of that id. Once the workspace is open and the command script, if any, is done, it opens the applet, with `applet.open` when it is in the workspace and from its file otherwise, then draws at least 10 more frames so that the plots fit their data, and writes the last one. A command script can therefore build and save the applet in the same run. When the workspace has no such applet, `shot` exits with 1. Whatever `--launch` named, an `applet.open` that succeeds while `shot` runs, from the command script or through Caliper API, shows the spec page from then on, as in the app:

```text
CALIPER_COMMAND_SCRIPT=build.jsonl caliper shot --workspace ws --launch run42-dashboard --size 2560x1440 --out board.png
wrote /home/me/board.png: Run 42 at 2560x1440 after 60 frames
```

The applet runs as it runs in the app: its `on_init`, its frames and any jobs it starts. Its own variables, such as `CALIPER_EMBED_AUTOTRAIN`, apply as usual, so leave them unset for an image of the applet's first state.

`shot` exits with 0 when it has written the image, with 2 for a wrong command line, and with 1 when:

- no folder holds an applet with the id: `caliper: no applet with the id <id> in the search folders (caliper list shows them)`, or, for an id written as a workspace's applet id, the workspace has none: `caliper: no applet with the id <id> in the search folders or the workspace (<code>: <reason>)`;
- the applet cannot be launched, with its status from `caliper list`: `caliper: <id> cannot be launched (REFUSED): <reason>`;
- its `on_init` fails or it faults while launching: `caliper: <id> failed to launch: <reason>`;
- it faults in a frame and is quarantined: `caliper: <id> was quarantined: <reason>`;
- the renderer cannot draw offscreen, as described above;
- the image's folder does not exist, or the image cannot be written: `caliper: cannot write <file>: <reason>`.

The id and the image's folder are checked before anything is drawn or created.

### caliper serve

Runs the hub of [wire protocol v3](wire-protocol.md): producers, such as training scripts, connect out to it and each becomes a session, and viewers (the browser viewer page, the notebook widget, the Caliper app) connect to it, list the sessions and attach to one. It needs no window, renderer or GPU, and listens on 127.0.0.1 only. [Run a hub](../guides/run-a-hub.md) explains how to use it, what it keeps and its limits.

| Option | Meaning |
|---|---|
| `--port <n>` | The port to listen on, 0 to 65535. The default is 7340. 0 takes any free port. |
| `--token-file <file>` | Read the session token from the first line of `<file>` instead of making a random one. The token must be 22 to 128 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`. On macOS and Linux, the file must not be readable or writable by other users (`chmod 600`). |
| `--assets <dir>` | Serve the viewer page and its files from `<dir>`, such as the `caliper/_web/` folder of Caliper LIB. Without it, `/` answers with a short note that says where the page is. |
| `--allow-control` | Forward viewers' controls, of [wire protocol 3.1](wire-protocol.md#controls-through-a-hub), to the script of each viewer's own session: pause, resume, step, setting a registered parameter and running a registered action. Off by default: without it the hub answers every control with `forbidden`, and viewers show the controls without letting them be changed. |

The options with a value also take the form `--port=<n>`. They apply only to `serve`, except `--allow-control`, which also applies with `--api` (see [Caliper API](#caliper-api)), and takes no value. The token is never given on the command line, where other users could read it from the process list. Without `--token-file`, the hub makes a random token of 32 characters (24 random bytes) for this run.

`serve` prints the viewer link and the producer address, then one line for each session that starts, resumes, ends or is forgotten:

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

SIGINT (Ctrl-C) or SIGTERM stops the hub: it sends BYE `shutdown` on every connection, closes them with code 1001, removes the hub file, prints `hub stopped` and exits with 0. It exits with 1 when the port is taken, the token file cannot be used, or the `--assets` folder does not exist. When the default port is taken, another hub is probably running; start this one with `--port`.

### caliper --help

Prints the usage to standard output and exits with 0. `-h` and `help` do the same.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A check failed, nothing was found, or a file operation failed. `build` also exits with 1 when CMake, a compiler or the SDK is missing or the build fails, `dev` when its folder is not a folder, `shot` when its applet cannot be launched or is quarantined, when the renderer cannot draw offscreen, or when Caliper API cannot listen, and `serve` when it cannot listen or cannot use its token file or assets folder. On Windows, `serve`, `build`, `--api` and `--url` exit with 1 before anything starts ([On Windows](#on-windows)). |
| 2 | The command line is wrong: an unknown command or option, a missing or extra operand, `--applets`, `--url`, `--api`, `--api-port`, `--hidden` or `--workspace` with a subcommand that does not take it, `--url` without a link or given twice, `--workspace` without a folder or given twice, `--api-port` or `--allow-control` without `--api` when starting the app, `dev` or `shot`, an option of `shot` or `serve` with another subcommand, a missing or unusable value of `--launch`, `--out`, `--size` or `--frames`, a port out of range, a URL given to `install`, or a name that `new` cannot use. For an unknown command or option and for wrong operands, the usage follows the error on standard error. |

## On Windows

The Windows build refuses `caliper serve`, `caliper build`, `--api` (when starting the app, with `dev` and with `shot`) and `--url` before anything starts. It prints one line, such as `caliper: caliper serve is not available on Windows yet`, and exits with 1. A wrong command line is still a usage error with exit code 2. The code of these pieces compiles on Windows but is untested there: the socket tests of the WebSocket layer, the hub and Caliper API do not run in the Windows CI lane, and the network source that `--url` attaches through exists on macOS only. `CALIPER_API=1` is ignored on Windows, and a line on standard error says so. Every other subcommand behaves as this page describes.

## Environment variables

| Variable | Effect |
|---|---|
| `CALIPER_APPLET_PATH` | Adds development folders after the `--applets` folders. A build folder in it stands for its `applets/` subfolder, as described in [Build folders](#build-folders). Read when starting the app, and by `dev`, `list` and `shot`. |
| `CALIPER_USER_APPLETS_DIR` | Replaces the user applets folder, for the app, `dev`, `list`, `install` and `shot`. |
| `CALIPER_DATA_DIR` | Moves the whole app-data folder: the stores, `imgui.ini`, `data/`, `live/` and, unless `CALIPER_USER_APPLETS_DIR` is set, the user applets folder `applets/`. `check` reads its data-schema stamps there and `install` copies into its `applets/`; `list`, `check`, `info`, `new` and `--help` create nothing in it, whether it is set or not. `shot` keeps its stores and the applet's data folder there when it is set, and in a temporary folder when it is not. |
| `CALIPER_CHROMELESS` | `1` draws the applet page without the page bar, the status bar and the jobs tray, in the app and in `shot`. |
| `CALIPER_HUB_FILE` | Replaces the path of the hub file that `serve` writes, `hub.json` in the app-data folder, when it is set and not empty. |
| `CALIPER_API` | `1` serves [Caliper API](api.md) as `--api` does, when starting the app and with `dev` and `shot`. Ignored on Windows ([On Windows](#on-windows)). |
| `CALIPER_HIDDEN` | `1` runs the app with its window never shown, as `--hidden` does: a testing aid (see [The hidden window](#the-hidden-window)). |
| `CALIPER_WORKSPACE` | The builder's workspace folder, when it is set and not empty and `--workspace` is not given, for the app, `dev` and `shot` (see [The workspace](#the-workspace)). |
| `CALIPER_COMMAND_SCRIPT`, `CALIPER_COMMAND_RESULTS` | A file of commands that the app, `dev` or `shot` runs through the builder session once its workspace is open, and the file that receives their results: a testing and demo aid (see [The workspace](#the-workspace)). |
| `CALIPER_SDK_DIR` | The prefix of the installed SDK that `build` uses, when it is set and not empty. |
| `CALIPER_CMAKE` | The cmake that `build` runs, when it is set and not empty. |

[Folders and environment variables](environment.md) lists every variable the host reads.
