"""Unit tests for generate_mcp_reference.py, generate_piece_reference.py, and generate_packages_reference.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_mcp_reference as gen_mcp  # noqa: E402
import generate_packages_reference as gen_pkgs  # noqa: E402
import generate_piece_reference as gen_piece  # noqa: E402


class McpGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        table = {
            "version": 0,
            "commands": [
                {
                    "name": "app.info",
                    "agents": True,
                    "effect": "read",
                    "mutates": False,
                    "destructive": False,
                    "frame_thread": False,
                    "args": {"type": "object", "properties": {}, "required": []},
                    "result": {"type": "object", "properties": {"host": {"type": "string"}}},
                    "summary": "Return host info.",
                },
                {
                    "name": "history.undo",
                    "agents": False,
                    "effect": "edit",
                    "mutates": True,
                    "destructive": False,
                    "frame_thread": False,
                    "args": {"type": "object", "properties": {}, "required": []},
                    "result": {"type": "object", "properties": {}},
                    "summary": "Undo last step.",
                },
            ],
        }
        p = self.root / gen_mcp.TABLE_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(table), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_check_passes(self):
        problems, rendered = gen_mcp.run(self.root, write=True)
        self.assertEqual(problems, [])
        self.assertIn("`app_info`", rendered)
        self.assertIn("`history.undo`", rendered)
        problems, _ = gen_mcp.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_drift_fails(self):
        gen_mcp.run(self.root, write=True)
        out = self.root / gen_mcp.OUTPUT_REL
        out.write_text("changed\n", encoding="utf-8")
        problems, _ = gen_mcp.run(self.root, write=False)
        self.assertTrue(any("out of date" in p for p in problems), problems)


class PieceGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write(
            gen_piece.PIECE_H_REL,
            """\
namespace piece_limits {
inline constexpr std::size_t kInputs = 8; // max inputs
}
inline constexpr std::array<const char*, 3> kPieceFiles{"common.wgsl", "derive.wgsl", "draw.wgsl"};
inline constexpr std::array<const char*, 2> kTensorDtypes{"f32", "i32"};
inline constexpr std::array<const char*, 1> kTextureFormats{"rgba8unorm"};
enum class InputKind { tensor, buffer };
""",
        )
        self._write(
            gen_piece.PIECES_H_REL,
            """\
inline constexpr const char* kPieceBadManifest = "piece.bad_manifest"; // invalid manifest
// ---- bind groups of a composed module
inline constexpr std::uint32_t kPreludeVersion = 1; // prelude version
""",
        )
        self._write(
            gen_piece.CHECKS_CPP_REL,
            'Verdict f() { return fail(kPieceBadManifest, "bad"); }\n',
        )
        defs = {k: {"type": "object", "required": [], "properties": {}} for k, _, _ in gen_piece.SCHEMA_SECTIONS}
        self._write(gen_piece.SCHEMA_REL, json.dumps({"$defs": defs}))

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_write_and_check_passes(self):
        problems, rendered = gen_piece.run(self.root, write=True)
        self.assertEqual(problems, [])
        self.assertIn("`piece.bad_manifest`", rendered)
        problems, _ = gen_piece.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_undocumented_check_code_fails(self):
        self._write(gen_piece.CHECKS_CPP_REL, 'Verdict f() { return fail(kPieceMystery, "bad"); }\n')
        problems, _ = gen_piece.run(self.root, write=True)
        self.assertTrue(any("kPieceMystery" in p for p in problems), problems)


class PackagesGeneratorTest(unittest.TestCase):
    GLFW_SHA = "3d89365fdb7449487993515156cf81011dc6121c"
    DUCKDB_SHA = "069cc9f9b5be802405797faecc284961b07c70ef"
    TOKENS_SHA = "f88dc88a2223cdeb029625639d23be1ab5755f95"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write(
            gen_pkgs.PYPROJECT_REL,
            """\
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2"]

[project]
name = "caliper"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "websockets", "Pillow"]

[project.optional-dependencies]
torch = ["torch"]
examples = ["torch", "torchvision"]
notebook = ["anywidget"]
interp = ["transformer-lens==3.9.0", "circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer.git@82996f144b3371ae858ea5713590d004c218f0c2"]
agent = ["pydantic-ai-slim[anthropic]>=2.68,<3"]
test = ["pytest>=8"]
docs = ["mkdocs==1.6.1"]

[project.scripts]
caliper-agent = "caliper_agent.cli:main"
""",
        )
        self._write(
            gen_pkgs.GITMODULES_REL,
            """\
[submodule "third_party/glfw"]
\tpath = third_party/glfw
\turl = https://github.com/glfw/glfw.git

[submodule "third_party/duckdb"]
\tpath = third_party/duckdb
\turl = https://github.com/duckdb/duckdb.git
\tbranch = v1.5-variegata
""",
        )
        self._write(
            gen_pkgs.CMAKE_DEPS_REL,
            f"""\
set(CALIPER_TOMLPLUSPLUS_TAG v3.4.0)
set(CALIPER_NLOHMANN_JSON_VERSION 3.11.3)
set(CALIPER_NLOHMANN_JSON_SHA256 d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(CALIPER_DOCTEST_TAG v2.4.11)
set(STERNBILD_TOKENS_REPOSITORY https://github.com/AhmedKhan-GH/organization.git)
set(STERNBILD_TOKENS_TAG design-v2.0.0)
set(STERNBILD_TOKENS_COMMIT {self.TOKENS_SHA})

FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG ${{CALIPER_TOMLPLUSPLUS_TAG}})
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v${{CALIPER_NLOHMANN_JSON_VERSION}}/json.tar.xz
    URL_HASH SHA256=${{CALIPER_NLOHMANN_JSON_SHA256}})
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG ${{CALIPER_DOCTEST_TAG}})
FetchContent_Declare(sternbild_tokens
    GIT_REPOSITORY ${{STERNBILD_TOKENS_REPOSITORY}}
    GIT_TAG ${{STERNBILD_TOKENS_COMMIT}}
    SOURCE_SUBDIR packages/sternbild-tokens-c)
set(_VK_TAG "vulkan-sdk-1.3.290.0")
FetchContent_Declare(zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz)
FetchContent_Declare(curl
    URL https://github.com/curl/curl/releases/download/curl-8_11_1/curl-8.11.1.tar.gz)
FetchContent_Declare(vulkan-headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${{_VK_TAG}}.tar.gz)

set(PYTORCH_VARIANT "cu124")
set(PYTORCH_VARIANT "cpu")
set(PYTORCH_VERSION "2.5.1")
set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${{PYTORCH_VERSION}}.zip")
set(PYTORCH_URL "https://download.pytorch.org/libtorch/${{PYTORCH_VARIANT}}/libtorch-win-shared-with-deps-${{PYTORCH_VERSION}}%2B${{PYTORCH_VARIANT}}.zip")
""",
        )
        self._write(
            gen_pkgs.WEB_PACKAGE_REL,
            json.dumps(
                {
                    "name": "@caliper/view",
                    "version": "0.1.0",
                    "devDependencies": {
                        "@playwright/test": "1.65.0",
                        "typescript": "7.0.2",
                    },
                }
            ),
        )
        self._write(gen_pkgs.EMSCRIPTEN_VERSION_REL, "6.0.13\n")
        self._init_git_with_uninitialized_submodules()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def _init_git_with_uninitialized_submodules(self) -> None:
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True)
        # Register gitlinks in the index with empty working directories, matching CI actions/checkout.
        for path, sha in (
            ("third_party/glfw", self.GLFW_SHA),
            ("third_party/duckdb", self.DUCKDB_SHA),
        ):
            (self.root / path).mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.root),
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"160000,{sha},{path}",
                ],
                check=True,
            )

    def test_write_and_check_round_trip_on_uninitialized_submodules(self):
        problems, rendered = gen_pkgs.run(self.root, write=True)
        self.assertEqual(problems, [])
        self.assertIn("<!-- Generated by scripts/generate_packages_reference.py. Do not edit by hand. -->", rendered)
        self.assertIn("## Shipped Packages", rendered)
        self.assertIn("## External Dependencies", rendered)
        self.assertIn("`caliper[examples]` | Users", rendered)
        self.assertIn("`caliper[test]` | **Contributors only**", rendered)
        self.assertIn("`caliper[docs]` | **Contributors only**", rendered)
        self.assertIn("| Instrument SDK | `caliper-sdk`", rendered)
        self.assertNotIn("applet", rendered.lower())
        self.assertIn("`caliper-agent` (`caliper_agent.cli:main`)", rendered)
        self.assertIn(f"`{self.GLFW_SHA}`", rendered)
        self.assertIn(f"`{self.DUCKDB_SHA}` (branch `v1.5-variegata`)", rendered)
        self.assertIn("`design-v2.0.0` (commit `%s`)" % self.TOKENS_SHA, rendered)
        self.assertIn("`v3.11.3` (`SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`)", rendered)
        self.assertIn("`8.11.1` (tag `curl-8_11_1`)", rendered)
        self.assertIn("`vulkan-sdk-1.3.290.0`", rendered)
        self.assertIn("libtorch-macos-arm64-2.5.1.zip", rendered)
        self.assertIn("`@playwright/test`", rendered)
        self.assertIn("`6.0.13`", rendered)
        problems, _ = gen_pkgs.run(self.root, write=False)
        self.assertEqual(problems, [])

    def test_drift_fails(self):
        problems, _ = gen_pkgs.run(self.root, write=True)
        self.assertEqual(problems, [])
        out = self.root / gen_pkgs.OUTPUT_REL
        out.write_text("stale content\n", encoding="utf-8")
        problems, _ = gen_pkgs.run(self.root, write=False)
        self.assertTrue(any("out of date" in p for p in problems), problems)

    def test_unresolved_cmake_pin_fails(self):
        self._write(
            gen_pkgs.CMAKE_DEPS_REL,
            """\
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG ${MISSING_TAG_VAR})
set(PYTORCH_VERSION "2.5.1")
set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip")
""",
        )
        problems, _ = gen_pkgs.run(self.root, write=True)
        self.assertTrue(any("unresolved CMake variable" in p for p in problems), problems)

    def test_unparseable_url_pin_and_missing_pytorch_version_fail(self):
        self._write(
            gen_pkgs.CMAKE_DEPS_REL,
            """\
FetchContent_Declare(mystery
    URL https://example.com/download/latest.tar.gz)
""",
        )
        problems, _ = gen_pkgs.run(self.root, write=True)
        self.assertTrue(any("cannot parse version or tag pin" in p for p in problems), problems)
        self.assertTrue(any("PYTORCH_VERSION not found" in p for p in problems), problems)

    def test_missing_submodule_commit_fails(self):
        # Create a non-git directory where no gitlink or .git exists for submodules.
        with tempfile.TemporaryDirectory() as plain_tmp:
            plain = Path(plain_tmp)
            for rel in (
                gen_pkgs.PYPROJECT_REL,
                gen_pkgs.GITMODULES_REL,
                gen_pkgs.CMAKE_DEPS_REL,
                gen_pkgs.WEB_PACKAGE_REL,
                gen_pkgs.EMSCRIPTEN_VERSION_REL,
            ):
                dst = plain / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text((self.root / rel).read_text(encoding="utf-8"), encoding="utf-8")
            problems, _ = gen_pkgs.run(plain, write=False)
            self.assertTrue(any("cannot resolve 40-char pinned commit" in p for p in problems), problems)

            # Fallback via local .git/HEAD works when not in a parent Git repo.
            for sub_path, sha in (
                ("third_party/glfw", self.GLFW_SHA),
                ("third_party/duckdb", self.DUCKDB_SHA),
            ):
                head_file = plain / sub_path / ".git" / "HEAD"
                head_file.parent.mkdir(parents=True, exist_ok=True)
                head_file.write_text(sha + "\n", encoding="utf-8")
            problems, rendered = gen_pkgs.run(plain, write=True)
            self.assertEqual(problems, [])
            self.assertIn(f"`{self.GLFW_SHA}`", rendered)

    def test_invalid_emscripten_version_fails(self):
        self._write(gen_pkgs.EMSCRIPTEN_VERSION_REL, "not-a-version\n")
        problems, _ = gen_pkgs.run(self.root, write=False)
        self.assertTrue(any("invalid Emscripten version" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
