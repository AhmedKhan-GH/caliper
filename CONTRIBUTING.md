# Contributing to Caliper

## Build and Test

[Quick Start](README.md#quick-start) and [Prerequisites](README.md#prerequisites) in `README.md` describe a local build. Run these commands from the repository root:

```sh
./init-submodules.sh
cmake --preset release
cmake --build --preset release
ctest --test-dir cmake-build-release --output-on-failure
```

The GPU suites skip when the machine has no usable device, and ctest reports them as skipped. Set `CALIPER_REQUIRE_GPU=1` to turn those skips into failures. [Testing](docs/development/testing.md) describes the test layers, and [Verify on Windows hardware](docs/development/windows-verification.md) describes the checks for the Vulkan and CUDA paths.

Every pull request is built and tested on macOS and Windows. [Continuous integration](docs/development/ci.md) describes what runs, what is cached, and how to handle a failed run.

### Python and Builder Tests

The suites in `python/tests/` and the builder suites (`builder_session` and `builder_agent_py`) require a Python 3.12 virtual environment. Create the environment from the repository root and install the same packages that CI installs:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "pydantic-ai-slim[anthropic]>=2.68,<3" "anthropic>=2.8,<3" "openai>=3.51,<4" \
  "agent-client-protocol>=1.0,<2" \
  torch numpy pyarrow pyyaml anywidget Pillow websockets nanobind "pytest>=8" jsonschema
.venv/bin/python scripts/build_netio.py
```

Install the pinned interpretability packages from `[project.optional-dependencies]` in `pyproject.toml` to run the interpretability suites in `tests/interp/` and `python/tests/test_net_controls.py`.

CMake uses `CALIPER_BUILDER_TEST_PYTHON` for the Python loader, session scope, and `caliper-agent` tests. When `CALIPER_BUILDER_TEST_PYTHON` is empty, CMake selects `src/builder/agent/.venv`, `.venv`, or `CALIPER_ACCEPTANCE_PYTHON` if that interpreter imports the required agent packages. When you pass `-DCALIPER_BUILDER_TEST_PYTHON` explicitly, CMake stops at configure time if that interpreter lacks `numpy`, `pydantic_ai`, `anthropic`, `openai`, `pytest`, or `acp.schema`. Pass both variables to match the macOS CI lane:

```sh
cmake --preset release \
  -DCALIPER_ACCEPTANCE_PYTHON="$PWD/.venv/bin/python" \
  -DCALIPER_BUILDER_TEST_PYTHON="$PWD/.venv/bin/python"
cmake --build --preset release
ctest --test-dir cmake-build-release -R "^(builder_session|builder_agent_py)$" --output-on-failure
```

Run the Python tests with `PYTHONPATH=python`. Use the marker filter below to run the CPU subset from `.github/workflows/build.yml`. Omit `-m` to include the local Metal GPU tests:

```sh
PYTHONPATH=python CALIPER_TEST_CALIPER_EXE="$PWD/cmake-build-release/caliper" \
  CALIPER_VIEWER="$PWD/cmake-build-release/caliper_viewer" \
  .venv/bin/python -m pytest -q python/tests \
  -m "not gpu and not window and not notebook and not timing" --strict-markers
```

The shaders of the graphics layer are written once in WGSL under `src/host/gfx/shaders/`, and their MSL and SPIR-V are generated and committed. After changing one, run `python3 tools/shaders/generate.py` (it needs a Rust toolchain) and commit the generated files with it; [Shader Authoring](docs/applets/shader-authoring.md) describes the generator and its checks.

To check the documentation, build it from the repository root with the pinned `docs` extra in `pyproject.toml` (`[project.optional-dependencies].docs`). Install those packages directly into a virtual environment instead of `pip install ".[docs]"` so `scikit-build-core` does not compile the C++ extension. The first `mkdocs` command builds the docs site. Its API reference comes from the code: mkdocstrings renders the Python calls from the docstrings of Caliper LIB, and the hook `docs-theme/hooks/c_reference.py` renders the C tables from the `/** */` comments of the headers. `python3 scripts/check_api_reference.py` fails when a public symbol has no docstring or comment, or no page. The second `mkdocs` command checks the design pages, which are built apart from the docs site and are not published.

```sh
uv venv --python 3.12 .docs-venv
python3 -c 'import subprocess, tomllib; deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["optional-dependencies"]["docs"]; subprocess.check_call(["uv", "pip", "install", "--python", ".docs-venv/bin/python", *deps])'
.docs-venv/bin/mkdocs build --strict
.docs-venv/bin/mkdocs build --strict -f design/mkdocs.yml
```

## Submodules

Caliper keeps third-party code as git submodules in `third_party/`. Run `./init-submodules.sh`, not `git submodule update`, because some submodule pins need the script fallbacks. On Windows without Git Bash, run `init-submodules.bat` in Command Prompt. Both scripts delete `third_party/` and local build folders (`cmake-build-*`, `build/`) before they initialize submodules. Do not commit changed submodule pointers unless that update is the purpose of the commit.

## Commits and Pull Requests

- One logical change per commit. Code, its tests and its documentation go in the same commit.
- Every pull request updates the documentation that its change affects, or states in the pull request description why no documentation page changes.
- When a change updates `src/builder/commands/command_table.json`, `src/builder/include/caliper_builder/spec/piece.h`, `src/builder/include/caliper_builder/programs/pieces.h`, `src/builder/programs/piece_checks.cpp`, `src/builder/spec/schema/caliper-spec-v1.schema.json`, `pyproject.toml`, `.gitmodules`, `cmake/Dependencies.cmake`, `web/package.json`, or `web/wasm/emscripten-version.txt`, regenerate the matching reference page (`scripts/generate_mcp_reference.py --write`, `scripts/generate_piece_reference.py --write`, or `scripts/generate_packages_reference.py --write`) in the same pull request.
- The subject line says what changed, in plain words. Start it with the area when that helps, as in `Site: ...`, `CI: ...` or `Docs: ...`.
- The body says why, and anything a reviewer cannot see in the diff.

## Repository Layout

| Path | Contents |
|---|---|
| `src/`, `include/` | the host, the embeddable core and its C interfaces |
| `sdk/` | the applet SDK: the ABI, the service headers, the C++ sugar, the adapters, the test fixture host and the `caliper-sdk` CMake package |
| `examples/` | the tutorial, embedding and MNIST examples; each example applet builds against the installed SDK, and the core build compiles none of them |
| `python/` | the `caliper` Python package |
| `tests/` | the test suites, including the pixel-exact `tests/gfx` suite |
| `docs/` | the documentation, published as the docs site (`mkdocs.yml`) |
| `design/` | proposals for work that is not built yet, with the roadmap; built on its own (`design/mkdocs.yml`) and not published |
| `docs-theme/` | the docs site's C reference hook, API reference templates and docstring extension; the theme and roadmap come from the `sternbild-docs` package (pinned under `docs` in `pyproject.toml`) |
| `site/` | the landing page, published with the docs |
| `.github/workflows/` | the Build and Deploy Pages workflows |

## Documentation Rules

- **One home per kind of document.** The docs site lives under `docs/`, in one folder per area: `introduction/`, `python/`, `applets/`, `builder/`, `agents/`, `web/`, `reference/`, `development/` and `roadmap/`, whose feature pages are generated from the roadmap registry. A page covers one topic, has a lowercase hyphenated file name, and sits in its area's folder; only `applets/services/` goes deeper. Each folder's `index.md` introduces the section and lists its pages. The site navigation follows the folders; a `.nav.yml` file in a folder sets its order. A design decision is not a document of its own: the rule goes, with its reason, in the page it shapes, and an open question goes in the design pages.
- **Nothing new at the root.** The repository root holds only `README.md`, `CONTRIBUTING.md` and `LICENSE` among documents. Proposals go in `design/`, and every other document in `docs/`.
- **Code and docs change together.** A change to a header, a service or a behaviour updates its doc comment or docstring, and its page, in the same commit. Reference pages render headers and docstrings and embed examples with snippets, so a moved or renamed file fails `mkdocs build --strict`.
- **Living pages say what is true now.** Tutorials, guides, references and explanations describe the current code. They hold no status tables, schedules or history.
- **Proposals are pages in `design/`, apart from the docs site.** The docs site is the reference for the people who build on Caliper and the people who use it, and it holds no plans. `design/` is built on its own and is not published, so no page under `docs/` links into it; a reference page that must mention planned work says so in a sentence, and a design page mentions a docs page by its path. The [roadmap](design/roadmap.md) is the only place for order and status. When a proposal is built, its content moves into the living pages and the proposal page is deleted.
- **Old material is deleted, not archived.** When a document goes stale, move what is still true into the right page and delete the rest. Git history is the archive, also of how each decision changed, and the deleting commit says where the content went.
- **The documentation is technical.** Business plans, personal paths and process notes do not belong in this repository.
