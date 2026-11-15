# Applet registry

This page proposes a registry for applets. Nothing described here is built. The host has no Browse tab, no registry client, no install command and no index format. Today an applet is installed by copying its manifest and library into an applets folder; see [Manifest](../references/manifest.md).

## Proposal

The registry is served at `caliperapp.ai`. A registry repository backs it, and the repository takes publications by pull request. The site serves the index that the repository holds, so the repository stays the record of what was published and who published it. A served registry lets users find and install applets without knowing about git, and it keeps review of publications by pull request. This is decision [D41](../decisions/d041-registry-at-caliperapp.md), which supersedes [D6](../decisions/d006-registry-as-a-git-repository.md). D6 proposed a git repository alone as the registry, with no served site.

Sideloading continues to work and is not replaced. A registry lists applets and helps the host find them, and it never gates what the host will load. The host loads what is in its applets folders.

## What the host would do

- **Browse tab.** The landing screen gains a tab that reads the registry index, filters it to releases that match the host's platform and ABI epoch, and lists them with their summary and tags.
- **Install by id and version.** A release is installed by its applet id and version. The host downloads the artifact for its platform, checks the sha256 digest against the index, and places the manifest and library in the user's applets folder. The existing loader then finds it like any other applet.
- **Update and uninstall.** The tab marks an installed applet whose index entry has a newer compatible release, and removes an installed applet's files on request.
- **Compatibility filtering.** A release the host cannot load is listed as incompatible, with the reason, instead of being hidden. The refusal text is the one the loader already uses for an epoch or `min_host` mismatch; see [Refusal messages](../references/refusals.md).

## Metadata for each release

A release entry carries what the host needs to decide whether to offer it and to install it without running any applet code:

| Field | Meaning |
|---|---|
| `version` | The applet's semver version. |
| `abi_epoch` | The ABI epoch the release was built against. The host offers only releases whose epoch equals its own. |
| `min_host` | The oldest host version that can run it, the same value the manifest carries. |
| Per-platform artifacts | For each platform, such as `macos-arm64` and `windows-x64`, a download URL and the sha256 digest of the file at that URL. |

The entry for the applet itself carries its id, display name, summary and tags, and the location of its source repository. Everything in a release entry duplicates data that the manifest already holds, and a registry check compares the two.

## Publishing

An applet's author opens a pull request against the registry repository that adds or updates the entry. Automatic checks in the registry repository validate the entry against the index schema, confirm that each URL is reachable and that its digest matches, and check that the epoch and `min_host` agree with the manifest inside the artifact. Review by a maintainer is a separate step that a registry may add; the proposal does not require it.

## Open questions

- Whether signing a release is required, and by whom. This is the open question in [D22](../decisions/d022-third-party-applet-signing.md), and the registry would be where a signature is checked.
- Where the host stores the downloaded index and how long it trusts the cached copy.
- How a release that depends on a libtorch version, which the applets link today, is described in the index. See [D5](../decisions/d005-runtime-packs.md).
- Whether the installed layout stays the flat sidecar folder or changes with the plan in [Applets outside the core](applets-outside-the-core.md).
