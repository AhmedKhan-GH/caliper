# Manifest

Every applet ships a manifest, a TOML file named `<stem>.caliper.toml`. The host parses it before loading
any code: it declares the applet's identity, the ABI epoch it was built against,
an optional minimum host version, and the framework services it needs. Parsing is
strict about required fields and their formats, and lenient about everything
else — unknown keys and tables are ignored so newer manifests keep loading on
older hosts (forward compatibility).

An applet is one of two kinds, which `[applet].kind` names. A compiled applet, the default, is a library that the host opens and calls. A spec applet is a folder with a manifest and a spec, the text file that the builder writes, and no library: it is data, which the app is to draw as a page and never executes. [Spec applets](#spec-applets) describes its manifest; the rest of this page describes a compiled applet's.

## Where the host looks

The manifest is a sidecar file in the same folder as the applet's library, and the two share a stem. For an applet with the stem `sine_scope`, the folder holds:

```text
sine_scope.caliper.toml
libsine_scope.dylib      (macOS; the library prefix is `lib`)
sine_scope.dll           (Windows)
libsine_scope.so         (Linux)
```

The layout is flat: one folder holds any number of applets, and there are no subfolders or bundles. The loader lists each folder for files that end in `.caliper.toml`, parses each one, and then looks for `<stem>.dylib`, `<stem>.dll` or `<stem>.so`, with or without the `lib` prefix, beside it. A manifest without its library is shown as a failed applet.

The host scans these folders at startup, in this order:

1. The development folders: the `caliper dev` folder and those given with `--applets <dir>`, then those in `CALIPER_APPLET_PATH`.
2. The user applets folder, `applets/` under the app-data root: `~/Library/Application Support/Caliper/applets` on macOS and `%APPDATA%\Caliper\applets` on Windows.
3. `applets/` next to the `caliper` executable.

Spec applets are found in the same folders and also one folder down, as [Where spec applets are found](#where-spec-applets-are-found) says.

Dropping a manifest and its library into the user applets folder, or running `caliper install <folder>`, installs an applet. When several folders hold copies of one applet id, the host loads one of them, as [Several copies of one applet](../explanations/compatibility.md#several-copies-of-one-applet) describes. [Caliper CLI](../references/cli.md#the-search-order) gives the search order in full, and [Folders and environment variables](../references/environment.md) gives the folders on each platform and the variables that change them.

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
| `[build]`   | `ui_stack` | string            | No       | Hash of the ImGui version number, the checked-out commits of ImGui, ImPlot, ImPlot3D and ImGuiFileDialog, the `imconfig.h` and the UI-stack definitions the applet was compiled against. Written by the build. See [The UI-stack hash](sdk-package.md#the-ui-stack-hash). |
| `[data]`    | `schema`   | integer           | No       | Version of the applet's stored data. Must be an integer ≥ 1.            |

Notes:

- **Strict semver** means exactly three dot-separated numeric parts (`x.y.z`).
  Values like `1.0` (too few parts) or `v1.0.0` (leading `v`) are rejected.
- **Unknown keys and tables are ignored.** A manifest may carry extra fields
  (for example `authors`, or a whole `[future]` table); the host skips them so that
  manifests targeting newer schema revisions still load.
- Missing or malformed required fields are hard errors: the host lists the
  applet as `FAILED` on the landing screen, under the manifest's file name,
  with a human-readable reason in the Inspector, rather than loading it.
- **`[services].required` is checked twice.** Before loading, against this
  list; after loading, against the binary's own `required_services` (the
  `CALIPER_APPLET` macro's `.services`). A service the binary requires but the
  manifest leaves out is still refused when the host lacks it, so keep the two
  lists the same.
- **`[compat].torch` must match exactly.** The host refuses the applet when the
  value differs from the version of the libtorch that the host provides, or
  when the host provides no libtorch. `caliper_add_applet(... TORCH)` writes
  the key into the built manifest. A value that is not a non-empty string is a
  manifest error. See
  [Each applet's version and what it requires](../explanations/compatibility.md#each-applets-version-and-what-it-requires).
- **The `[build]` table is written by the build, never by hand.**
  `caliper_add_applet()` copies the manifest that the author writes to the
  folder beside the built library and adds the table to that copy, and it
  refuses a source manifest that already has one
  ([SDK package](sdk-package.md#the-stamped-manifest)). `sdk` is a record for
  diagnosis, which the host shows in the Inspector and never compares.
  `ui_stack`, when present, must be a non-empty string equal to the host's own
  hash, or the applet is refused. See
  [The SDK version](../explanations/compatibility.md#the-sdk-version).
- **`[data].schema` versions the applet's stored data.** The host refuses an
  applet whose schema is lower than the highest schema that has opened its
  data folder. An applet without the key is never checked, and a value that
  is not an integer of at least 1 is a manifest error. See
  [The version of each applet's stored data](../explanations/compatibility.md#the-version-of-each-applets-stored-data).

## Spec applets

A spec applet is a folder that holds its manifest and its spec, and no library. The spec is a TOML file of sources, panels and a layout, in version 0 of the spec format, which the builder's library in `src/builder/spec/` reads and checks. The builder's `applet.save` command writes a spec applet into a workspace as `applets/<id>/<id>.caliper.toml` and `applets/<id>/<id>.spec.toml`:

```toml
[applet]
id = "local.attention-entropy"
name = "Attention entropy"
version = "0.1.0"
summary = "Entropy of each attention head while the model trains."
kind = "spec"
spec = "attention-entropy.spec.toml"
```

Its keys mean what they mean for a compiled applet, with these differences:

| Section | Key | Required | Meaning for a spec applet |
|---|---|---|---|
| `[applet]` | `id`, `name`, `version` | Yes | As for a compiled applet. The id follows the same [rules](#applet-ids). |
| `[applet]` | `summary`, `tag` | No | As for a compiled applet. |
| `[applet]` | `kind` | Yes | `"spec"`. |
| `[applet]` | `spec` | Yes | The spec's file name, in the same folder as the manifest. It is a plain file name: not empty, not `.` or `..`, at most 255 bytes, with no `/`, `\`, `:` or control character. |
| `[compat]` | `min_host` | No | As for a compiled applet: a host older than it refuses the applet. The `[compat]` table itself is optional. |
| `[compat]` | `abi_epoch`, `torch` | Refused | A spec applet has no library, so there is no binary to check. Either key is a manifest error. |
| `[build]` | any | Refused | The table is written by the build of a library. Its presence is a manifest error. |
| `[services]` | any | Refused | A spec applet calls no service. The table's presence is a manifest error. |
| `[data]` | `schema` | Refused | A spec applet keeps no data of its own: the host gives it no data folder, and the data it draws are the workspace's sources. The table's presence is a manifest error. |

The keys that only a library can honour are refused rather than ignored, so that a manifest written for a library, with `kind = "spec"` added by mistake, is never taken for a spec applet. Unknown keys and tables are ignored, as for a compiled applet. A manifest without `kind`, or with `kind = "library"`, is a compiled applet exactly as before, and a `spec` key in it is an unknown key.

A spec applet names the version of the spec format it needs in the spec itself, with the key `caliper_spec`, written before the spec's first table as `caliper_spec = 0`. The host reads that line when it lists the applet and refuses a version it cannot read (see [Refusal messages](../references/refusals.md#spec-applets)).

### Spec applet ids

A workspace names its applets with workspace ids, `[a-z][a-z0-9_-]{0,63}`, which name their folders and files. A manifest id has its own grammar, which has no `_`. `applet.save` therefore writes the manifest id `local.<id>`, with each `_` of the workspace id replaced by `-`: the workspace applet `loss_view` is the applet `local.loss-view` in its folder `applets/loss_view/`. The loader never compares the folder name with the manifest id; the copies of one applet are those with one manifest id, whatever their kind.

### Where spec applets are found

The loader looks for spec applets in the folders of the [search order](../references/cli.md#the-search-order), in two places:

- a manifest of kind `"spec"` directly in a searched folder, as a compiled applet's manifest is, which is how `caliper check <folder>` takes the folder of one spec applet;
- a manifest of kind `"spec"` one folder down, in `<folder>/<name>/`, which is how a workspace keeps its applets. Outside a workspace, any other manifest one folder down is not looked at, as before, since compiled applets are flat. Folders whose names start with `.`, folders that are symbolic links, and more than 4096 folders of one searched folder are skipped.

A host that has a workspace open also searches the workspace's `applets/` folder, after the development folders and before the user applets folder. A workspace is data that anyone may hand to the person, so that folder holds spec applets only: a compiled applet's manifest there, at any depth the loader looks at, is listed as refused and its library is never opened. The app does not open a workspace yet; the loader's interface for it is `AppletLoader::set_workspace` and the last argument of `applet_search_dirs` in `src/host/applet_loader.h`.

A manifest read one folder down or in the workspace's folder may be at most 64 KiB, and none of its dotted keys may have more than 32 parts, before it is parsed. The spec's file must be a regular file in the manifest's folder, not a symbolic link wherever it points, of at most 512 KiB, the spec format's limit. When the applet is listed the host reads only the spec's first lines, up to `caliper_spec`; the whole spec is parsed and validated when the applet is launched or checked with `caliper check`.

## Applet ids

The `id` names the applet's data folder (`data/<id>/` under the app-data root),
so it is restricted to a form that is always a single, safe path component:

- only lowercase letters, digits, `.` and `-` (`[a-z0-9.-]`);
- dot-separated segments, none empty: no leading or trailing dot, no `..`;
- at most 128 characters.

`dev.example.sine-scope` and `dev.caliper.gpt-scope` are valid;
`Dev.Example.App` (uppercase), `dev.example.my_app` (underscore), `dev/app`,
`a..b` and `../../escape` are not. A manifest with an invalid id is listed as
`FAILED` with the message
`applet.id is not a valid applet id (lowercase reverse-DNS, [a-z0-9.-]): <id>`
(see [Refusal messages](../references/refusals.md#failed-messages)), and no data folder is
created for it.

## Example

The canonical minimal manifest is the one shipped by the **hello** fixture
applet (`examples/hello/`), embedded here verbatim so it can never drift from the
source of truth:

```toml
--8<-- "examples/hello/hello.caliper.toml"
```

See [tutorials/first-applet.md](first-applet.md) for a full walk of
the applet this manifest belongs to.
