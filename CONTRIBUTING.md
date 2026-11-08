# Contributing to Caliper

## Building and testing

The README's [Quick start](README.md#quick-start) and [Prerequisites](README.md#prerequisites) cover a local build. Run the tests with:

```sh
ctest --test-dir cmake-build-release --output-on-failure
```

Every pull request is built and tested on macOS and Windows. [docs/ci.md](docs/ci.md) describes what runs, what is cached, and how to handle a failed run.

## Where things live

| Path | Contents |
|---|---|
| `src/`, `sdk/`, `applets/`, `tests/` | the host, the SDK, the applets and their tests |
| `docs/wiki/` | the product documentation, published as the docs site (`mkdocs.yml`) |
| `docs/` (outside `wiki/`) | engineering notes: CI, architecture, verification and postmortems. Not published. |
| `site/` | the landing page, published with the docs |
| `.github/workflows/` | the Build and Deploy Pages workflows |

## Submodules

Third-party code is vendored as git submodules under `third_party/`. Initialize them with `./init-submodules.sh` rather than `git submodule update`, because some pins need the script's fallbacks. Don't commit changed submodule pointers unless the change is the point of the commit.
