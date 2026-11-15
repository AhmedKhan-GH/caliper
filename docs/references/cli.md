# Caliper CLI

The `caliper` executable starts the app and also takes subcommands. It reads its command line before anything opens a window. Every subcommand except `dev` runs and exits without opening a window: `list`, `check`, `install`, `new`, `info`, `shot` and `--help` never start the app, so they are safe in scripts and CI. Only `caliper` with no subcommand and `caliper dev` start the app. `shot` draws the app's applet page into an offscreen canvas and writes it as a PNG, and shows nothing on screen. The parser and the subcommands are in `src/host/host_cli.h`, and `src/main.cpp` dispatches to them.

```text
usage: caliper [--applets <dir>]...
       caliper dev <folder> [--applets <dir>]...
       caliper list [--applets <dir>]...
       caliper check <folder>
       caliper install <folder>
       caliper new <name> [<dir>]
       caliper info
       caliper shot [--applets <dir>]... --launch <id> [--size <W>x<H>]
                    [--frames <N>] --out <file.png>
       caliper --help
```

Results go to standard output, and errors go to standard error, each starting with `caliper:`. Apart from `shot`, which runs an applet and opens the stores in a temporary folder or in `CALIPER_DATA_DIR` (see [caliper shot](#caliper-shot)), no subcommand opens the metrics, data or artifacts store. `list`, `check`, `info`, `new` and `--help` create nothing in the app-data folder, `install` writes only the user applets folder, and `shot` without `CALIPER_DATA_DIR` creates nothing there either.

## Starting the app

`caliper` with no subcommand starts the app. `--applets <dir>`, or `--applets=<dir>`, adds a development folder to the search. It may be given more than once, and the folders are searched in the order given. It is accepted when starting the app and with `dev`, `list` and `shot`; with any other subcommand it is a usage error, and so is `--applets` followed by an argument that starts with `--`, such as `--applets --help`. An argument that starts with `-psn_`, which Finder passes to an app it launches on older versions of macOS, is ignored.

`caliper dev <folder>` starts the app with `<folder>` searched first, before any `--applets` folders. The folder is usually an applet's build folder, such as `build`, or its build output, `build/applets` (see [Build folders](#build-folders)). The Activity pane's `scanned` line names the folder that was searched. If `<folder>` is not a folder, `dev` prints `caliper: not a folder: <folder>` and exits with 1 without starting the app. `dev` does not reload an applet whose library is rebuilt; quit and start it again.

## The search order

The app, `caliper dev`, `caliper list` and `caliper shot` search these folders, in this order:

1. The development folders: the `dev` folder, then the `--applets` folders in the order given.
2. The folders in `CALIPER_APPLET_PATH`, separated by `:` (`;` on Windows), in the order given.
3. The user applets folder: `CALIPER_USER_APPLETS_DIR` when it is set and not empty, otherwise `applets/` under the app-data folder.
4. The `applets/` folder beside the executable, which holds the applets shipped with the app.

Each development folder, from items 1 and 2, is resolved as described in [Build folders](#build-folders). A folder named twice is searched once, at its first position, and empty entries are skipped. When several folders hold copies of one applet id, the host loads the highest version that passes every check, and at equal versions the copy found first; the other copies that pass are `SHADOWED`. [Several copies of one applet](../explanations/compatibility.md#several-copies-of-one-applet) states the rule, and [Folders and environment variables](environment.md) gives the folders on each platform.

## Build folders

`caliper_add_applet()` puts an applet's library and manifest in the `applets/` folder of its build folder. So that the build folder itself can be named, a development folder that holds no manifest (`<stem>.caliper.toml`) of its own, but whose `applets/` subfolder holds one, stands for that subfolder. This applies to the `dev` folder, the `--applets` folders, the folders in `CALIPER_APPLET_PATH`, and the folders given to `check` and `install`. A folder that holds a manifest of its own is used as given, even when it also has an `applets/` subfolder. The user applets folder and the `applets/` folder beside the executable are always used as given.

## Subcommands

### caliper list

Lists every applet in the search folders, without opening any library. Applets are ordered by the position of their folder in the search order, then by id, then from the highest version down. Each applet takes one line, with its status, id, version and folder in columns. A refused, failed or shadowed applet has its reason on the next line, indented under the id:

```text
READY     dev.caliper.pulse-scope  0.1.0  <development folder>
SHADOWED  dev.caliper.pulse-scope  0.1.0  <user applets folder>
          Version 0.1.0 in <development folder> is used instead.
READY     dev.caliper.hello        0.1.0  <applets folder beside the executable>
```

The status is `READY`, `REFUSED`, `FAILED` or `SHADOWED`; the reasons are those in [Refusal messages](refusals.md). Folders are printed as absolute paths. When no folder holds an applet, `list` prints `no applets found`. It always exits with 0.

### caliper check <folder>

Runs the checks that the host makes before it opens a library on every applet in one folder, and only that folder, with this host's capabilities (see `caliper info`). It opens no library. The data-schema check reads the stamps in the app-data folder. The output is the same lines as `list`, then a summary:

```text
READY  dev.caliper.pulse-scope  0.1.0  <folder>
1 of 1 applets pass the checks
```

When `<folder>` is a build folder whose applets are in its `applets/` subfolder (see [Build folders](#build-folders)), `check` uses that subfolder and says so first:

```text
using the applets in <folder>/applets
READY  dev.caliper.pulse-scope  0.1.0  <folder>/applets
1 of 1 applets pass the checks
```

An applet passes when it is `READY` or `SHADOWED`. `check` exits with 0 when every applet passes, and with 1 when any does not, when `<folder>` is not a folder (`caliper: not a folder: <folder>`), or when neither it nor its `applets/` subfolder holds an applet (`caliper: no applets in <folder>`).

### caliper install <folder>

Runs the same checks on a local folder, and when every applet passes, copies each applet into the user applets folder: its library, then its asset folder `assets/<stem>/` beside the library when there is one, then its manifest. An applet that reads files from beside its library keeps them in that folder, and its own build stages them there, as the in-tree applets do; an installed asset folder replaces the old one as a whole. Other files beside the library are not copied. A build folder is resolved as for `check`, with the same `using the applets in` line.

The install is all or nothing for the whole folder. Every file and asset folder is first copied to a hidden temporary name beside its destination, and only then renamed into place, each manifest after its library and assets, so the host never sees half a library or a manifest without its library. When a step fails, `install` removes what it placed, puts back what it replaced, and leaves the user applets folder as it was. The output is the lines of `check`, then one line per applet:

```text
installed dev.caliper.pulse-scope 0.1.0 in <user applets folder>
```

When any applet fails a check, `install` copies nothing, prints `caliper: nothing installed; N of M applets do not pass the checks`, and exits with 1. It also exits with 1 when `<folder>` is the user applets folder itself, when that folder cannot be created, or when a copy fails, which it reports as `caliper: nothing installed; <reason>`. A URL in place of a folder is a usage error (`caliper: install takes a local folder, not <url>`), with exit code 2.

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

The comment at the top of the generated `CMakeLists.txt` gives the next steps: configure with `-DCMAKE_PREFIX_PATH=<sdk prefix>`, build, run `ctest --test-dir build`, and run `caliper dev build/applets`.

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
| `--launch <id>` | The manifest id of the applet to launch. Required. |
| `--out <file.png>` | The image to write. Its folder must exist, and a file already there is replaced. Required. |
| `--size <W>x<H>` | The size of the image in pixels, each side from 1 to 16384. The default is `1600x1000`. |
| `--frames <N>` | The number of frames drawn, from 1 to 100000; the image is the last one. The default is 60, one second. |

Each option takes its value as the next argument or after `=`, as in `--size=800x500`, and may be given once. Any of them with another subcommand is a usage error.

The page is drawn at a scale of 1, so a 1600x1000 image shows what a window of 1600 by 1000 points shows on a display without scaling, and the image is opaque. As in the app, an applet's windows do not appear in the first frame and are docked in place from the second; give enough frames for the applet to load what it shows.

Nothing appears on screen. `shot` draws into the renderer's offscreen canvas, the canvas that Caliper Embed uses in `CALIPER_CANVAS_OFFSCREEN` mode, and reads the frame back from it. It starts no GLFW and creates no window, and on macOS it first sets the activation policy to `NSApplicationActivationPolicyProhibited`, so the process has no Dock icon and no menu bar. The offscreen canvas needs Metal on macOS or Vulkan on Windows. With the OpenGL fallback (`CALIPER_RENDERER=gl`, and on Linux), or when no GPU can make the canvas, `shot` prints `caliper: shot cannot draw offscreen: ...` and exits with 1.

`shot` never reads or writes the saved window layout, `imgui.ini`, so every image shows the layout of a first run, with each window where its dock role puts it. Where the stores and the applet's data folder go depends on `CALIPER_DATA_DIR`:

- When `CALIPER_DATA_DIR` is set and not empty, the stores and the applet's data folder `data/<id>/` are in that folder, as they are for the app, so an image can show data that the applet saved there.
- When it is not set, `shot` creates a temporary folder under the system's temporary folder (`TMPDIR` on macOS and Linux, `TMP` or `TEMP` on Windows), uses it as the app-data folder, and removes it before it exits. The applet starts with an empty data folder, and nothing is written in the app-data folder.

In both cases `shot` searches the user applets folder, as the app does, and only reads it. The display face, which only the landing screen's wordmark uses, is loaded only when `CALIPER_DISPLAY_FONT` names it.

The applet runs as it runs in the app: its `on_init`, its frames and any jobs it starts. Its own variables, such as `CALIPER_EMBED_AUTOTRAIN`, apply as usual, so leave them unset for an image of the applet's first state.

`shot` exits with 0 when it has written the image, with 2 for a wrong command line, and with 1 when:

- no folder holds an applet with the id: `caliper: no applet with the id <id> in the search folders (caliper list shows them)`;
- the applet cannot be launched, with its status from `caliper list`: `caliper: <id> cannot be launched (REFUSED): <reason>`;
- its `on_init` fails or it faults while launching: `caliper: <id> failed to launch: <reason>`;
- it faults in a frame and is quarantined: `caliper: <id> was quarantined: <reason>`;
- the renderer cannot draw offscreen, as described above;
- the image's folder does not exist, or the image cannot be written: `caliper: cannot write <file>: <reason>`.

The id and the image's folder are checked before anything is drawn or created.

### caliper --help

Prints the usage to standard output and exits with 0. `-h` and `help` do the same.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A check failed, nothing was found, or a file operation failed. `dev` also exits with 1 when its folder is not a folder, and `shot` when its applet cannot be launched or is quarantined, or when the renderer cannot draw offscreen. |
| 2 | The command line is wrong: an unknown command or option, a missing or extra operand, `--applets` with a subcommand that does not take it, an option of `shot` with another subcommand, a missing or unusable value of `--launch`, `--out`, `--size` or `--frames`, a URL given to `install`, or a name that `new` cannot use. For an unknown command or option and for wrong operands, the usage follows the error on standard error. |

## Environment variables

| Variable | Effect |
|---|---|
| `CALIPER_APPLET_PATH` | Adds development folders after the `--applets` folders. A build folder in it stands for its `applets/` subfolder, as described in [Build folders](#build-folders). Read when starting the app, and by `dev`, `list` and `shot`. |
| `CALIPER_USER_APPLETS_DIR` | Replaces the user applets folder, for the app, `dev`, `list`, `install` and `shot`. |
| `CALIPER_DATA_DIR` | Moves the whole app-data folder: the stores, `imgui.ini`, `data/`, `live/` and, unless `CALIPER_USER_APPLETS_DIR` is set, the user applets folder `applets/`. `check` reads its data-schema stamps there and `install` copies into its `applets/`; `list`, `check`, `info`, `new` and `--help` create nothing in it, whether it is set or not. `shot` keeps its stores and the applet's data folder there when it is set, and in a temporary folder when it is not. |
| `CALIPER_CHROMELESS` | `1` draws the applet page without the page bar, the status bar and the jobs tray, in the app and in `shot`. |

[Folders and environment variables](environment.md) lists every variable the host reads.
