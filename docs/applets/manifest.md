# Manifest

Every applet ships a manifest, a TOML file named `<stem>.caliper.toml`. The host parses it before it loads any code. The manifest declares the applet's identity, the ABI epoch of the build, an optional minimum host version, and the framework services the applet needs. The host is strict about required fields and their formats, and lenient about everything else.

The key `[applet].kind` sets one of two kinds of applet. A compiled applet, the default, is a library that the host opens and calls. A spec applet has no library. It is data that the app draws as a page and never executes. [Spec Applets](#spec-applets) describes its manifest, and the rest of this page describes a compiled applet's.

## Search Order

The manifest is a sidecar file in the same folder as the applet's library, and the two share a stem. For an applet with the stem `sine_scope`, the folder holds:

```text
sine_scope.caliper.toml
libsine_scope.dylib      (macOS; the library prefix is `lib`)
sine_scope.dll           (Windows)
libsine_scope.so         (Linux)
```

The layout is flat. One folder holds any number of applets, and there are no subfolders or bundles. The loader parses each file that ends in `.caliper.toml` and looks beside it for the library, with or without the `lib` prefix. The host shows a manifest without its library as a failed applet.

The host scans these folders at startup, in this order:

1. The development folders: the `caliper dev` folder and those given with `--applets <dir>`, then those in `CALIPER_APPLET_PATH`.
2. The user applets folder, `applets/` under the app-data root.
3. `applets/` next to the `caliper` executable.

The host also finds spec applets one folder down, as [Spec Applet Search Order](#spec-applet-search-order) says.

To install an applet, put its manifest and library in the user applets folder or run `caliper install <folder>`. When several folders hold copies of one applet id, the host loads one of them, as [Several copies of one applet](../development/compatibility-and-epochs.md#several-copies-of-one-applet) describes. [Caliper CLI](../reference/cli.md#search-order) gives the search order in full, and [Folders and environment variables](../reference/folders-and-environment.md) gives the folders on each platform.

## Schema

| Section     | Key        | Type              | Required | Meaning                                                                 |
|-------------|------------|-------------------|----------|-------------------------------------------------------------------------|
| `[applet]`  | `id`       | string            | Yes      | Lowercase reverse-DNS identity, e.g. `dev.example.sine-scope`. See [Applet ids](#applet-ids). |
| `[applet]`  | `name`     | string            | Yes      | Human-readable display name. Must be non-empty.                          |
| `[applet]`  | `version`  | string            | Yes      | Applet version, strict semver `x.y.z` (three numeric parts).            |
| `[applet]`  | `summary`  | string            | No       | Short one-line description shown in the landing screen's Inspector.      |
| `[applet]`  | `tag`      | string            | No       | Free-form category label, e.g. `EDA`. Shown in the Instruments table, which sorts by it by default. |
| `[applet]`  | `kind`     | string            | No       | `"library"`, the default, for a compiled applet, or `"spec"` for a [spec applet](#spec-applets). Any other value is a manifest error. |
| `[compat]`  | `abi_epoch`| integer           | Yes      | ABI epoch the applet was built against. Must be an integer ≥ 1.          |
| `[compat]`  | `min_host` | string            | No       | Minimum host version floor, strict semver `x.y.z`. Omit for no floor.   |
| `[compat]`  | `torch`    | string            | No       | Exact libtorch version the applet was built against, e.g. `"2.5.1"`. Present only for applets that link libtorch. |
| `[services]`| `required` | array of strings  | No       | Service IDs the applet cannot run without, e.g. `["caliper.ui.v1"]`.    |
| `[services]`| `optional` | array of strings  | No       | Service IDs the applet uses if present, e.g. `["caliper.log.v1"]`.      |
| `[build]`   | `sdk`      | string            | No       | SDK version that built the applet. Written by the build.                |
| `[build]`   | `ui_stack` | string            | No       | Hash of the UI stack the applet was compiled against: the ImGui version, the checked-out commits of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog, `imconfig.h` and the UI-stack definitions. Written by the build. See [The UI-stack hash](sdk-package.md#ui-stack-hash). |
| `[data]`    | `schema`   | integer           | No       | Version of the applet's stored data. Must be an integer ≥ 1.            |

Notes:

- **Strict semver** means exactly three dot-separated numeric parts. The host rejects `1.0` and `v1.0.0`.
- **Unknown keys and tables are ignored.** The host skips extra fields, such as `authors` or a whole `[future]` table, so manifests for newer schema revisions still load on older hosts.
- **A missing or malformed required field is a hard error.** The host lists the applet as `FAILED` on the landing screen, under the manifest's file name, and does not load it. The Inspector shows the reason.
- **`[services].required` is checked twice.** The host checks it before loading, against this list, and after loading, against the `required_services` of the binary (the `.services` of the `CALIPER_APPLET` macro). The host refuses the applet if the binary requires a service that the manifest omits and the host lacks. Keep the two lists the same.
- **`[compat].torch` must match exactly.** The host refuses the applet when the value differs from the version of the libtorch that the host provides, or when the host provides no libtorch. `caliper_add_applet(... TORCH)` writes the key into the built manifest. A value that is not a non-empty string is a manifest error. See [Each applet's version and what it requires](../development/compatibility-and-epochs.md#applet-versions-and-requirements).
- **The `[build]` table is written by the build, never by hand.** `caliper_add_applet()` adds the table to a copy of the manifest beside the built library. It refuses a source manifest that already has the table ([SDK package](sdk-package.md#stamped-manifest)). The host shows `sdk` in the Inspector and never compares it. The host refuses the applet unless `ui_stack`, when present, is a non-empty string equal to the host's own hash. See [The SDK version](../development/compatibility-and-epochs.md#sdk-version).
- **`[data].schema` versions the applet's stored data.** The host refuses an applet whose schema is lower than the highest schema that has opened its data folder. It never checks an applet without the key. A value that is not an integer of at least 1 is a manifest error. See [The version of each applet's stored data](../development/compatibility-and-epochs.md#stored-data-versions).

## Spec Applets

A spec applet is a folder that holds its manifest and its spec, and no library. The spec is a TOML file of sources, panels and a layout, in version 0 of the spec format. The builder's `applet.save` command writes a spec applet into a workspace as `applets/<id>/<id>.caliper.toml` and `applets/<id>/<id>.spec.toml`:

```toml
[applet]
id = "local.attention-entropy"
name = "Attention entropy"
version = "0.1.0"
summary = "Entropy of each attention head while the model trains."
kind = "spec"
spec = "attention-entropy.spec.toml"
```

The keys have the same meaning as for a compiled applet, with these differences:

| Section | Key | Required | Meaning for a spec applet |
|---|---|---|---|
| `[applet]` | `id`, `name`, `version` | Yes | As for a compiled applet. The id follows the same [rules](#applet-ids). |
| `[applet]` | `summary`, `tag` | No | As for a compiled applet. |
| `[applet]` | `kind` | Yes | `"spec"`. |
| `[applet]` | `spec` | Yes | The spec's file name, in the same folder as the manifest. It is a plain file name: not empty, not `.` or `..`, at most 255 bytes, with no `/`, `\`, `:` or control character. |
| `[compat]` | `min_host` | No | As for a compiled applet: a host older than it refuses the applet. The `[compat]` table itself is optional. |
| `[compat]` | `abi_epoch`, `torch` | Refused | A spec applet has no library, so there is no binary to check. Either key is a manifest error. |
| `[build]` | any | Refused | The build of a library writes this table. Its presence is a manifest error. |
| `[services]` | any | Refused | A spec applet calls no service. The table's presence is a manifest error. |
| `[data]` | `schema` | Refused | A spec applet keeps no data of its own, and the host gives it no data folder. The table's presence is a manifest error. |

The host refuses the keys that only a library can honor instead of ignoring them. Thus a library manifest with `kind = "spec"` added by mistake is never taken for a spec applet. A manifest without `kind`, or with `kind = "library"`, is a compiled applet, and a `spec` key in it is an unknown key.

A spec names the version of the spec format that it needs with the key `caliper_spec`, written before its first table, as `caliper_spec = 0`. The host reads that line when it lists the applet. It refuses a version that it cannot read (see [Refusal messages](../reference/refusal-messages.md#spec-applets)).

### Spec Applet IDs

A workspace names its applets with workspace ids, `[a-z][a-z0-9_-]{0,63}`, which name their folders and files. A manifest id has no `_`. Thus `applet.save` writes the manifest id `local.<id>`, with each `_` of the workspace id replaced by `-`. For example, the workspace applet `loss_view` is the applet `local.loss-view` in its folder `applets/loss_view/`. The loader never compares the folder name with the manifest id.

### Spec Applet Search Order

The loader looks for spec applets in the folders of the [search order](../reference/cli.md#search-order), in two places:

- A manifest of kind `"spec"` directly in a searched folder, as for a compiled applet. This is how `caliper check <folder>` takes the folder of one spec applet.
- A manifest of kind `"spec"` one folder down, in `<folder>/<name>/`. This is how a workspace keeps its applets.

Outside a workspace, the loader does not look at any other manifest one folder down, because compiled applets are flat. It skips folders whose names start with `.` and folders that are symbolic links. It also skips the folders after the first 4096 in one searched folder.

A host with an open workspace also searches the workspace's `applets/` folder, after the development folders and before the user applets folder. A workspace is data that anyone may hand to the user, so that folder holds spec applets only. The host lists a compiled applet's manifest there, at any depth that the loader looks at, as refused and never opens its library.

Before it parses a manifest read one folder down or in the workspace's folder, the host limits the manifest to 64 KiB and each dotted key to 32 parts. The spec file must be a regular file in the manifest's folder, not a symbolic link, and at most 512 KiB, the limit of the spec format. When it lists the applet, the host reads only the first lines of the spec, up to `caliper_spec`. It parses and validates the whole spec when the applet is launched or checked with `caliper check`.

## Applet IDs

The `id` names the applet's data folder (`data/<id>/` under the app-data root). Thus it has a form that is always a single, safe path component:

- Only lowercase letters, digits, `.` and `-` (`[a-z0-9.-]`).
- Dot-separated segments, none empty: no leading or trailing dot, no `..`.
- At most 128 characters.

`dev.example.sine-scope` is valid. `Dev.Example.App` (uppercase), `dev.example.my_app` (underscore), `dev/app`, `a..b` and `../../escape` are not. A manifest with an invalid id is listed as `FAILED` with the message `applet.id is not a valid applet id (lowercase reverse-DNS, [a-z0-9.-]): <id>` (see [Refusal messages](../reference/refusal-messages.md#failed-messages)). The host creates no data folder for it.

## Worked Example

The minimal manifest below is the one that the **Hello World** fixture applet ships. The page embeds it from the source, so it cannot drift:

```toml
--8<-- "examples/hello_world/hello_world.caliper.toml"
```

See [First Applet](first-applet.md) for a full walk of the applet that this manifest belongs to.
