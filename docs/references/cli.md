# Command line

The `caliper` executable starts the app and also takes subcommands. It reads its command line before anything opens a window. Every subcommand except `dev` runs and exits without opening a window: `list`, `check`, `install`, `new`, `info` and `--help` never start the app, so they are safe in scripts and CI. Only `caliper` with no subcommand and `caliper dev` start the app. The parser and the subcommands are in `src/host/host_cli.h`, and `src/main.cpp` dispatches to them.

```text
usage: caliper [--applets <dir>]...
       caliper dev <folder> [--applets <dir>]...
       caliper list [--applets <dir>]...
       caliper check <folder>
       caliper install <folder>
       caliper new <name> [<dir>]
       caliper info
       caliper --help
```

Results go to standard output, and errors go to standard error, each starting with `caliper:`.

## Starting the app

`caliper` with no subcommand starts the app. `--applets <dir>`, or `--applets=<dir>`, adds a development folder to the search. It may be given more than once, and the folders are searched in the order given. It is accepted when starting the app and with `dev` and `list`; with any other subcommand it is a usage error.

`caliper dev <folder>` starts the app with `<folder>` searched first, before any `--applets` folders. The folder is usually an applet's build output, such as `build/applets`. If `<folder>` is not a folder, `dev` prints `caliper: not a folder: <folder>` and exits with 1 without starting the app. `dev` does not reload an applet whose library is rebuilt; quit and start it again.

## The search order

The app, `caliper dev` and `caliper list` search these folders, in this order:

1. The development folders: the `dev` folder, then the `--applets` folders in the order given.
2. The folders in `CALIPER_APPLET_PATH`, separated by `:` (`;` on Windows), in the order given.
3. The user applets folder: `CALIPER_USER_APPLETS_DIR` when it is set and not empty, otherwise `applets/` under the app-data folder.
4. The `applets/` folder beside the executable, which holds the applets shipped with the app.

A folder named twice is searched once, at its first position, and empty entries are skipped. When several folders hold copies of one applet id, the host loads the highest version that passes every check, and at equal versions the copy found first; the other copies that pass are `SHADOWED`. [Several copies of one applet](../explanations/compatibility.md#several-copies-of-one-applet) states the rule, and [Folders and environment variables](environment.md) gives the folders on each platform.

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

An applet passes when it is `READY` or `SHADOWED`. `check` exits with 0 when every applet passes, and with 1 when any does not, when `<folder>` is not a folder (`caliper: not a folder: <folder>`), or when it holds no applet (`caliper: no applets in <folder>`).

### caliper install <folder>

Runs the same checks on a local folder, and when every applet passes, copies each library and then its manifest into the user applets folder. Each file is copied to a temporary name and renamed over the destination, so the host never sees half a library. The output is the lines of `check`, then one line per applet:

```text
installed dev.caliper.pulse-scope 0.1.0 in <user applets folder>
```

When any applet fails a check, `install` copies nothing, prints `caliper: nothing installed; N of M applets do not pass the checks`, and exits with 1. It also exits with 1 when `<folder>` is the user applets folder itself, when that folder cannot be created, or when a copy fails. A URL in place of a folder is a usage error (`caliper: install takes a local folder, not <url>`), with exit code 2.

### caliper new <name> [<dir>]

Creates an applet project from a template built into the executable. `<name>` is either a short name or a full applet id:

- A short name is lowercase letters, digits and `-`, starts with a letter, does not end with `-`, and has at most 64 characters. The applet's id is `dev.example.<name>`.
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

### caliper --help

Prints the usage to standard output and exits with 0. `-h` and `help` do the same.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A check failed, nothing was found, or a file operation failed. `dev` also exits with 1 when its folder is not a folder. |
| 2 | The command line is wrong: an unknown command or option, a missing or extra operand, `--applets` with a subcommand that does not take it, a URL given to `install`, or a name that `new` cannot use. For an unknown command or option and for wrong operands, the usage follows the error on standard error. |

## Environment variables

| Variable | Effect |
|---|---|
| `CALIPER_APPLET_PATH` | Adds development folders after the `--applets` folders. Read when starting the app, by `dev` and by `list`. |
| `CALIPER_USER_APPLETS_DIR` | Replaces the user applets folder, for the app, `dev`, `list` and `install`. |

[Folders and environment variables](environment.md) lists every variable the host reads.
