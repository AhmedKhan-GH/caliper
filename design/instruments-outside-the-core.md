# Instruments outside the core

The host should ship with no instruments and name none. Instruments are built, stored
and installed outside the core repository, and adding one never means editing
the host. This page holds the parts of that plan that are not built yet. The
registry is a separate plan, [Instrument registry](registry.md).

The built parts are described in the living pages: the SDK package and
`caliper_add_instrument()` in SDK package (`docs/instruments/sdk-package.md`), the
search order and the choice among copies of one instrument in
Compatibility and epochs (`docs/development/compatibility-and-epochs.md`),
dock hints in caliper.ui.v1_1 (`docs/instruments/services/caliper-ui-v1-1.md`), and the
folders and variables in Folders and environment variables (`docs/reference/folders-and-environment.md`).
The `caliper` subcommands, including `dev` and `install <folder>`, are in
Caliper CLI (`docs/reference/cli.md`). The roadmap lists the `.caliperapp` bundle
format and the entry points for hot reload under [Undecided](roadmap.md#undecided).

## The changes

### Moving the instruments out

- First-party instruments move to one `caliper-instruments` repository with their
  history, rather than one repository each. The history is extracted with
  `git filter-repo` rather than copied as a fresh start, because it is the
  record of how each instrument was built.
- `CALIPER_BUILD_INSTRUMENTS` goes away with them.
- `examples/hello_world` and `examples/mnist/instrument` stay, because the tutorials embed
  them and the tests load them.

### Shipping the SDK

The SDK ships as a release asset beside the app. A separate SDK repository can
wait until outside contributors exist.

### Hot reload

`caliper dev <folder>` starts the app with that folder searched first, but it
does not reload an instrument whose library is rebuilt. Reloading is proposed for
later, as hot reload.

### Installing from a URL

- `caliper install <url>` is proposed for later. It would download a
  `.caliperapp` bundle, check its sha256, place it and handle the quarantine
  flag under the [signing policy](#signing-on-macos).
- A Browse tab reading the registry comes later ([Instrument registry](registry.md)).

## Where instruments are stored

| What | Where |
|---|---|
| Source | The instrument's own repository. First-party instruments in `caliper-instruments`; private instruments in private repositories |
| Releases | The instrument repository's releases, as `<id>-<version>-<platform>.caliperapp.zip`; the registry points to them |
| Shipped with the app | `Caliper.app/Contents/PlugIns/`, signed with the app |
| Installed by the user | macOS `~/Library/Application Support/Caliper/instruments/<id>.caliperapp/`; Windows `%APPDATA%\Caliper\instruments\` |
| Instrument data | `~/Library/Application Support/Caliper/data/<id>/` |
| Rebuildable caches | `~/Library/Caches/Caliper/<id>/` |
| Shared runtimes | `~/Library/Application Support/Caliper/runtimes/`, if [runtime packs](roadmap.md#undecided) are adopted |
| During development | The instrument's build folder, found through `--instruments`, `CALIPER_INSTRUMENT_PATH` or `caliper dev` |

The host scans `<exe>/instruments/` today, which inside a macOS app is
`Contents/MacOS/instruments/`. Apple expects loadable code in `Contents/PlugIns/`,
and anything written into the app bundle breaks its signature. Shipped instruments
therefore move to `Contents/PlugIns/`, and the app bundle is never written after
it ships.

## Signing on macOS

A notarized host with the hardened runtime refuses to load libraries signed by a
different team. Two options:

- The host carries the `com.apple.security.cs.disable-library-validation`
  entitlement and accepts third-party instruments, relying on the manifest checks,
  the sha256 and the trust model.
- Only instruments signed by the same team load; others must be re-signed locally.

Downloaded bundles also carry Gatekeeper's quarantine flag, which
`caliper install <url>` must handle. The policy must be decided before the first third-party instrument ships.

## The empty-state check

CI builds the host, starts it headless with an empty instruments folder, and checks
that the launcher shows its empty state.
