# Manifest

Every instrument ships a manifest, a TOML file named `<stem>.caliper.toml`. The host parses it before it loads any code. The manifest declares the instrument's identity, the ABI epoch of its build when it has a library, an optional minimum host version, and the framework services the instrument needs. The host is strict about required fields and their formats, and lenient about everything else.

The key `[instrument].kind` declares how the instrument is built. The default, `"library"`, is an instrument written in C++ as a shared library that the host opens and calls. The value `"spec"` is an instrument written as a spec file with no library, which the app draws as a page and never executes. [Spec-Based Instruments](#spec-based-instruments) describes the manifest for `kind = "spec"`, and the rest of this page describes the manifest for `kind = "library"`.

## Search Order

For an instrument with a library, the manifest is a sidecar file in the same folder as the library, and the two share a stem. For an instrument with the stem `mnist`, the folder holds:

```text
mnist.caliper.toml
libmnist.dylib           (macOS; the library prefix is `lib`)
mnist.dll                (Windows)
libmnist.so              (Linux)
```

The layout is flat. One folder holds any number of library instruments, and there are no subfolders or bundles. The loader parses each file that ends in `.caliper.toml` and looks beside it for the library, with or without the `lib` prefix. The host shows a library manifest without its library as a failed instrument.

The host scans these folders at startup, in this order:

1. The development folders: the `caliper dev` folder and those given with `--instruments <dir>`, then those in `CALIPER_INSTRUMENT_PATH`.
2. The user instruments folder, `instruments/` under the app-data root.
3. `instruments/` next to the `caliper` executable.

The host also finds spec-based instruments one folder down, as [Spec Search Order](#spec-search-order) says.

To install an instrument with a library, put its manifest and library in the user instruments folder or run `caliper install <folder>`. When several folders hold copies of one instrument id, the host loads one of them, as [Several Copies of One Instrument](../development/compatibility-and-epochs.md#several-copies-of-one-instrument) describes. [Caliper CLI](../reference/cli.md#search-order) gives the search order in full, and [Folders and Environment](../reference/folders-and-environment.md) gives the folders on each platform.

## Schema

| Section        | Key         | Type              | Required | Meaning                                                                 |
|----------------|-------------|-------------------|----------|-------------------------------------------------------------------------|
| `[instrument]` | `id`        | string            | Yes      | Lowercase reverse-DNS identity, e.g. `dev.caliper.mnist`. See [Instrument IDs](#instrument-ids). |
| `[instrument]` | `name`      | string            | Yes      | Human-readable display name. Must be non-empty.                         |
| `[instrument]` | `version`   | string            | Yes      | Instrument version, strict semver `x.y.z` (three numeric parts).        |
| `[instrument]` | `summary`   | string            | No       | Short one-line description shown in the landing screen's Inspector.     |
| `[instrument]` | `tag`       | string            | No       | Free-form category label, e.g. `EDA`. Shown in the Instruments table, which sorts by it by default. |
| `[instrument]` | `kind`      | string            | No       | `"library"`, the default, for an instrument with a library, or `"spec"` for an [instrument written as a spec](#spec-based-instruments). Any other value is a manifest error. |
| `[compat]`     | `abi_epoch` | integer           | Yes      | ABI epoch the instrument was built against. Must be an integer ≥ 1.     |
| `[compat]`     | `min_host`  | string            | No       | Minimum host version floor, strict semver `x.y.z`. Omit for no floor.   |
| `[compat]`     | `torch`     | string            | No       | Exact libtorch version the instrument was built against, e.g. `"2.5.1"`. Present only for instruments that link libtorch. |
| `[services]`   | `required`  | array of strings  | No       | Service IDs the instrument cannot run without, e.g. `["caliper.ui.v1"]`. |
| `[services]`   | `optional`  | array of strings  | No       | Service IDs the instrument uses if present, e.g. `["caliper.log.v1"]`.  |
| `[build]`      | `sdk`       | string            | No       | SDK version that built the instrument. Written by the build.            |
| `[build]`      | `ui_stack`  | string            | No       | Hash of the UI stack the instrument was compiled against: the ImGui version, the checked-out commits of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog, `imconfig.h` and the UI-stack definitions. Written by the build. See [UI-Stack Hash](sdk-package.md#ui-stack-hash). |
| `[data]`       | `schema`    | integer           | No       | Version of the instrument's stored data. Must be an integer ≥ 1.        |

Notes:

- **Strict semver** means exactly three dot-separated numeric parts. The host rejects `1.0` and `v1.0.0`.
- **Unknown keys and tables are ignored.** The host skips extra fields, such as `authors` or a whole `[future]` table, so manifests for newer schema revisions still load on older hosts.
- **A missing or malformed required field is a hard error.** The host lists the instrument as `FAILED` on the landing screen, under the manifest's file name, and does not load it. The Inspector shows the reason.
- **`[services].required` is checked twice.** The host checks it before loading, against this list, and after loading, against the `required_services` of the binary (the `.services` of the `CALIPER_INSTRUMENT` macro). The host refuses the instrument if the binary requires a service that the manifest omits and the host lacks. Keep the two lists the same.
- **`[compat].torch` must match exactly.** The host refuses the instrument when the value differs from the version of the libtorch that the host provides, or when the host provides no libtorch. `caliper_add_instrument(... TORCH)` writes the key into the built manifest. A value that is not a non-empty string is a manifest error. See [Instrument Versions and Requirements](../development/compatibility-and-epochs.md#instrument-versions-and-requirements).
- **The `[build]` table is written by the build, never by hand.** `caliper_add_instrument()` adds the table to a copy of the manifest beside the built library. It refuses a source manifest that already has the table ([Stamped Manifest](sdk-package.md#stamped-manifest)). The host shows `sdk` in the Inspector and never compares it. The host refuses the instrument unless `ui_stack`, when present, is a non-empty string equal to the host's own hash. See [SDK Version](../development/compatibility-and-epochs.md#sdk-version).
- **`[data].schema` versions the instrument's stored data.** The host refuses an instrument whose schema is lower than the highest schema that has opened its data folder. It never checks an instrument without the key. A value that is not an integer of at least 1 is a manifest error. See [Stored Data Versions](../development/compatibility-and-epochs.md#stored-data-versions).

## Spec-Based Instruments

An instrument written as a spec is a folder that holds its manifest and its spec, and no library. The spec is a TOML file of sources, panels, controls, state and a layout, in version 2 of the spec format (`caliper_spec = 2`). The builder's `instrument.save` command writes a spec-based instrument into a workspace as `instruments/<id>/<id>.caliper.toml` and `instruments/<id>/<id>.spec.toml`:

```toml
[instrument]
id = "local.attention-entropy"
name = "Attention entropy"
version = "0.1.0"
summary = "Entropy of each attention head while the model trains."
kind = "spec"
spec = "attention-entropy.spec.toml"
```

The keys have the same meaning as for a library instrument, with these differences:

| Section | Key | Required | Meaning for `kind = "spec"` |
|---|---|---|---|
| `[instrument]` | `id`, `name`, `version` | Yes | As for a library instrument. The id follows the same [rules](#instrument-ids). |
| `[instrument]` | `summary`, `tag` | No | As for a library instrument. |
| `[instrument]` | `kind` | Yes | `"spec"`. |
| `[instrument]` | `spec` | Yes | The spec's file name, in the same folder as the manifest. It is a plain file name: not empty, not `.` or `..`, at most 255 bytes, with no `/`, `\`, `:` or control character. |
| `[compat]` | `min_host` | No | As for a library instrument: a host older than it refuses the instrument. The `[compat]` table itself is optional. |
| `[compat]` | `abi_epoch`, `torch` | Refused | An instrument with `kind = "spec"` has no library, so there is no binary to check. Either key is a manifest error. |
| `[build]` | any | Refused | The build of a library writes this table. Its presence is a manifest error. |
| `[services]` | any | Refused | An instrument with `kind = "spec"` calls no service. The table's presence is a manifest error. |
| `[data]` | `schema` | Refused | An instrument with `kind = "spec"` keeps no data of its own, and the host gives it no data folder. The table's presence is a manifest error. |

The host refuses the keys that only a library can honor instead of ignoring them. Thus a library manifest with `kind = "spec"` added by mistake is never taken for a spec-based instrument. A manifest without `kind`, or with `kind = "library"`, is an instrument with a library, and a `spec` key in it is an unknown key.

A spec names the version of the spec format that it needs with the key `caliper_spec`, written before its first table, as `caliper_spec = 2`. The host reads that line when it lists the instrument. It refuses a version that it cannot read (see [Refusal Messages](../reference/refusal-messages.md#spec-based-instruments)).

### Workspace Instrument IDs

A workspace names its instruments with workspace ids, `[a-z][a-z0-9_-]{0,63}`, which name their folders and files. A manifest id has no `_`. Thus `instrument.save` writes the manifest id `local.<id>`, with each `_` of the workspace id replaced by `-`. For example, the workspace instrument `loss_view` is the instrument `local.loss-view` in its folder `instruments/loss_view/`. The loader never compares the folder name with the manifest id.

### Spec Search Order

The loader looks for spec-based instruments in the folders of the [search order](../reference/cli.md#search-order), in two places:

- A manifest of kind `"spec"` directly in a searched folder, as for a library instrument. This is how `caliper check <folder>` takes the folder of one spec-based instrument.
- A manifest of kind `"spec"` one folder down, in `<folder>/<name>/`. This is how a workspace keeps its instruments.

Outside a workspace, the loader does not look at any other manifest one folder down, because library instruments are flat. It skips folders whose names start with `.` and folders that are symbolic links. It also skips the folders after the first 4096 in one searched folder.

A host with an open workspace also searches the workspace's `instruments/` folder, after the development folders and before the user instruments folder. A workspace is data that anyone may hand to the user, so that folder holds spec-based instruments only. The host lists a library instrument's manifest there, at any depth that the loader looks at, as refused and never opens its library.

Before it parses a manifest read one folder down or in the workspace's folder, the host limits the manifest to 64 KiB and each dotted key to 32 parts. The spec file must be a regular file in the manifest's folder, not a symbolic link, and at most 512 KiB, the limit of the spec format. When it lists the instrument, the host reads only the first lines of the spec, up to `caliper_spec`. It parses and validates the whole spec when the instrument is launched or checked with `caliper check`.

## Instrument IDs

The `id` names the instrument's data folder (`data/<id>/` under the app-data root). Thus it has a form that is always a single, safe path component:

- Only lowercase letters, digits, `.` and `-` (`[a-z0-9.-]`).
- Dot-separated segments, none empty: no leading or trailing dot, no `..`.
- At most 128 characters.

`dev.caliper.mnist` is valid. `Dev.Example.App` (uppercase), `dev.example.my_app` (underscore), `dev/app`, `a..b` and `../../escape` are not. A manifest with an invalid id is listed as `FAILED` with the message `instrument.id is not a valid instrument id (lowercase reverse-DNS, [a-z0-9.-]): <id>` (see [Refusal Messages](../reference/refusal-messages.md#failed-messages)). The host creates no data folder for it.

## Worked Example

The minimal manifest below is the one that the **Hello World** fixture instrument ships. The page embeds it from the source, so it cannot drift:

```toml
--8<-- "examples/hello_world/hello_world.caliper.toml"
```

See [First Instrument](first-instrument.md) for a full walk of the instrument that this manifest belongs to.
