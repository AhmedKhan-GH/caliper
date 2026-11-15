# Manifest (caliper.toml)

Every applet ships a `caliper.toml` manifest. The host parses it before loading
any code: it declares the applet's identity, the ABI epoch it was built against,
an optional minimum host version, and the framework services it needs. Parsing is
strict about required fields and their formats, and lenient about everything
else — unknown keys and tables are ignored so newer manifests keep loading on
older hosts (forward compatibility).

## Schema

| Section     | Key        | Type              | Required | Meaning                                                                 |
|-------------|------------|-------------------|----------|-------------------------------------------------------------------------|
| `[applet]`  | `id`       | string            | Yes      | Lowercase reverse-DNS identity, e.g. `dev.ahmed.circuitnet`. See [Applet ids](#applet-ids). |
| `[applet]`  | `name`     | string            | Yes      | Human-readable display name. Must be non-empty.                          |
| `[applet]`  | `version`  | string            | Yes      | Applet version, strict semver `x.y.z` (three numeric parts).            |
| `[applet]`  | `summary`  | string            | No       | Short one-line description shown in the landing screen's Inspector.      |
| `[applet]`  | `tag`      | string            | No       | Free-form category label, e.g. `EDA`. Shown in the Instruments table, which sorts by it by default. |
| `[compat]`  | `abi_epoch`| integer           | Yes      | ABI epoch the applet was built against. Must be an integer ≥ 1.          |
| `[compat]`  | `min_host` | string            | No       | Minimum host version floor, strict semver `x.y.z`. Omit for no floor.   |
| `[services]`| `required` | array of strings  | No       | Service IDs the applet cannot run without, e.g. `["caliper.ui.v1"]`.    |
| `[services]`| `optional` | array of strings  | No       | Service IDs the applet uses if present, e.g. `["caliper.log.v1"]`.      |

Notes:

- **Strict semver** means exactly three dot-separated numeric parts (`x.y.z`).
  Values like `1.0` (too few parts) or `v1.0.0` (leading `v`) are rejected.
- **Unknown keys and tables are ignored.** A manifest may carry extra fields
  (e.g. `authors`, or a whole `[future]` table); the host skips them so that
  manifests targeting newer schema revisions still load.
- Missing or malformed required fields are hard errors: the host lists the
  applet as `FAILED` on the landing screen, under the manifest's file name,
  with a human-readable reason in the Inspector, rather than loading it.
- **`[services].required` is checked twice.** Before loading, against this
  list; after loading, against the binary's own `required_services` (the
  `CALIPER_APPLET` macro's `.services`). A service the binary requires but the
  manifest leaves out is still refused when the host lacks it, so keep the two
  lists the same.

## Applet ids

The `id` names the applet's data folder (`data/<id>/` under the app-data root),
so it is restricted to a form that is always a single, safe path component:

- only lowercase letters, digits, `.` and `-` (`[a-z0-9.-]`);
- dot-separated segments, none empty: no leading or trailing dot, no `..`;
- at most 128 characters.

`dev.ahmed.circuitnet` and `dev.example.sine-scope` are valid;
`Dev.Ahmed.App` (uppercase), `dev.ahmed.my_app` (underscore), `dev/app`,
`a..b` and `../../escape` are not. A manifest with an invalid id is listed as
`FAILED` with the message
`applet.id is not a valid applet id (lowercase reverse-DNS, [a-z0-9.-]): <id>`
(see [Refusal messages](refusals.md#failed-messages)), and no data folder is
created for it.

## Example

The canonical minimal manifest is the one shipped by the **hello** fixture
applet (`examples/hello/`), embedded here verbatim so it can never drift from the
source of truth:

```toml
--8<-- "examples/hello/hello.caliper.toml"
```

See [tutorials/first-applet.md](../tutorials/first-applet.md) for a full walk of
the applet this manifest belongs to.
