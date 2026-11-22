# Contributing to Caliper

## Build and test

The README's [Quick start](README.md#quick-start) and [Prerequisites](README.md#prerequisites) cover a local build. In short:

```sh
./init-submodules.sh
cmake --preset release
cmake --build --preset release
ctest --test-dir cmake-build-release --output-on-failure
```

The GPU suites skip when the machine has no usable device, and ctest reports them as skipped. Set `CALIPER_REQUIRE_GPU=1` to turn those skips into failures. [Testing](docs/development/testing.md) describes the test layers, and [Verify on Windows hardware](docs/development/windows-verification.md) describes the checks for the Vulkan and CUDA paths.

Every pull request is built and tested on macOS and Windows. [Continuous integration](docs/development/ci.md) describes what runs, what is cached, and how to handle a failed run.

The shaders of the graphics layer are written once in WGSL under `src/host/gfx/shaders/`, and their MSL and SPIR-V are generated and committed. After changing one, run `python3 tools/shaders/generate.py` (it needs a Rust toolchain) and commit the generated files with it; [Shader Authoring](docs/applets/shader-authoring.md) describes the generator and its checks.

To check the documentation, build it the way the Docs workflow does, from the repository root and with the same versions. The first command builds the docs site. Its API reference comes from the code: mkdocstrings renders the Python calls from the docstrings of Caliper LIB, and the hook `docs-theme/hooks/c_reference.py` renders the C tables from the `/** */` comments of the headers. `python3 scripts/check_api_reference.py` fails when a public symbol has no docstring or comment, or no page. The second command checks the design pages, which are built apart from the docs site and are not published.

```sh
uvx --with mkdocs-material==9.8.1 --with mkdocs-awesome-nav==3.3.0 --with pymdown-extensions==13.1 --with mkdocstrings==1.0.8 --with mkdocstrings-python==2.0.10 --with griffelib==2.4.1 --with mkdocs-autorefs==1.4.5 --with ruff==0.17.3 mkdocs@1.6.1 build --strict
uvx --with mkdocs-material==9.8.1 --with mkdocs-awesome-nav==3.3.0 --with pymdown-extensions==13.1 mkdocs@1.6.1 build --strict -f design/mkdocs.yml
```

## Submodules

Third-party code is vendored as git submodules under `third_party/`. Initialize them with `./init-submodules.sh` rather than `git submodule update`, because some pins need the script's fallbacks. Don't commit changed submodule pointers unless the change is the point of the commit.

## Commits

- One logical change per commit. Code, its tests and its documentation go in the same commit.
- The subject line says what changed, in plain words. Start it with the area when that helps, as in `Site: ...`, `CI: ...` or `Docs: ...`.
- The body says why, and anything a reviewer cannot see in the diff.

## Where things live

| Path | Contents |
|---|---|
| `src/`, `include/` | the host, the embeddable core and its C interfaces |
| `sdk/` | the applet SDK: the ABI, the service headers, the C++ sugar, the adapters, the test fixture host and the `caliper-sdk` CMake package |
| `applets/`, `examples/` | the first-party applets and the tutorial and embedding examples; each applet builds against the installed SDK, and the core build compiles none of them |
| `python/` | the `caliper` Python package |
| `tests/` | the test suites, including the pixel-exact `tests/gfx` suite |
| `docs/` | the documentation, published as the docs site (`mkdocs.yml`) |
| `design/` | proposals for work that is not built yet, with the roadmap; built on its own (`design/mkdocs.yml`) and not published |
| `docs-theme/` | the docs site's icons, fonts and stylesheet |
| `site/` | the landing page, published with the docs |
| `.github/workflows/` | the Build and Deploy Pages workflows |

## Documentation rules

- **One home per kind of document.** The docs site lives under `docs/`, in one folder per kind: `tutorials/`, `guides/`, `references/` and `explanations/`. A page covers one topic, has a lowercase hyphenated file name, and sits directly in its folder, with no deeper folders. Each folder's `index.md` introduces the section and lists its pages. The site navigation follows the folders; a `.nav.yml` file in a folder sets its order. A design decision is not a document of its own: the rule goes, with its reason, in the explanation or reference page it shapes, and an open question goes in the design pages.
- **Nothing new at the root.** The repository root holds only `README.md`, `CONTRIBUTING.md` and `LICENSE` among documents. Proposals go in `design/`, and every other document in `docs/`.
- **Code and docs change together.** A change to a header, a service or a behaviour updates its doc comment or docstring, and its page, in the same commit. Reference pages render headers and docstrings and embed examples with snippets, so a moved or renamed file fails `mkdocs build --strict`.
- **Living pages say what is true now.** Tutorials, guides, references and explanations describe the current code. They hold no status tables, schedules or history.
- **Proposals are pages in `design/`, apart from the docs site.** The docs site is the reference for the people who build on Caliper and the people who use it, and it holds no plans. `design/` is built on its own and is not published, so no page under `docs/` links into it; a reference page that must mention planned work says so in a sentence, and a design page mentions a docs page by its path. The [roadmap](design/roadmap.md) is the only place for order and status. When a proposal is built, its content moves into the living pages and the proposal page is deleted.
- **Old material is deleted, not archived.** When a document goes stale, move what is still true into the right page and delete the rest. Git history is the archive, also of how each decision changed, and the deleting commit says where the content went.
- **The documentation is technical.** Business plans, personal paths and process notes do not belong in this repository.
