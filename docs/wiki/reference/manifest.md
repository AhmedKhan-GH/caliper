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
| `[applet]`  | `id`       | string            | Yes      | Reverse-DNS identity, e.g. `dev.ahmed.circuitnet`. Must be non-empty.    |
| `[applet]`  | `name`     | string            | Yes      | Human-readable display name. Must be non-empty.                          |
| `[applet]`  | `version`  | string            | Yes      | Applet version, strict semver `x.y.z` (three numeric parts).            |
| `[applet]`  | `summary`  | string            | No       | Short one-line description shown in the launcher.                        |
| `[applet]`  | `tag`      | string            | No       | Free-form category label, e.g. `EDA`.                                    |
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
- Missing or malformed required fields are hard errors: the host surfaces a
  human-readable reason on the applet's failure card rather than loading it.

## Example

The canonical minimal manifest is the one shipped by the **hello** fixture
applet (`examples/hello/`), embedded here verbatim so it can never drift from the
source of truth:

```toml
--8<-- "examples/hello/hello.caliper.toml"
```

See [tutorials/first-applet.md](../tutorials/first-applet.md) for a full walk of
the applet this manifest belongs to.
