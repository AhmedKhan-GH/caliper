#!/usr/bin/env python3
"""Build the runtime WGSL compiler, the static library caliper_wgsl, from source.

    build_library.py --out DIR [--target-dir DIR] [--jobs N] [--cargo PATH] [--offline] [--archive] [--test-hooks]

It runs cargo on tools/shaders/caliper-wgsl with the locked, pinned Naga and copies the
library to DIR/lib/. With --archive it also writes DIR/caliper_wgsl-<version>-<platform>.tar.gz,
the file that a release publishes, and DIR/<archive>.sha256. It prints the platform, the size
and the SHA-256 of the library and of the archive. CI and the CMake option
CALIPER_WGSL_FROM_SOURCE both build through this script, so that every build uses the same
flags:

- with --test-hooks, the library is built with the cargo feature test-hooks, which adds
  caliper_wgsl_test_panic for the C test of the panic guard; the archives that a release
  publishes are built without it;
- the source paths of the build machine are mapped to fixed prefixes, so the library holds
  no path of the machine that built it;
- macOS builds for the deployment target that Caliper's wheel uses (14.0) unless
  MACOSX_DEPLOYMENT_TARGET is set;
- Windows uses the dynamic C runtime (/MD), as Caliper does when CALIPER_STATIC_CRT is OFF.

Exit status: 0 on success, 1 when cargo is missing or the build fails.
"""

import argparse
import gzip
import hashlib
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile

CRATE = pathlib.Path(__file__).resolve().parent
REPO = CRATE.parents[2]


def crate_version():
    """The package version of Cargo.toml; the release tag is caliper-wgsl-<version>."""
    text = (CRATE / "Cargo.toml").read_text()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def platform_name():
    machine = platform.machine().lower()
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine, machine)
    system = {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux" if sys.platform.startswith("linux") else sys.platform)
    return "%s-%s" % (system, arch)


def library_name():
    return "caliper_wgsl.lib" if sys.platform == "win32" else "libcaliper_wgsl.a"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=pathlib.Path, help="folder that receives lib/ and the archive")
    ap.add_argument("--target-dir", type=pathlib.Path, help="cargo's target folder (default: OUT/cargo-target)")
    ap.add_argument("--jobs", type=int, default=0, help="cargo build jobs (default: cargo's choice)")
    ap.add_argument("--offline", action="store_true", help="build from the crates already in CARGO_HOME")
    ap.add_argument("--archive", action="store_true", help="also write the release archive and its hash")
    ap.add_argument("--test-hooks", action="store_true", help="build with the cargo feature test-hooks (developer and test builds only)")
    ap.add_argument("--cargo", type=pathlib.Path, help="path to the cargo executable (default: cargo on PATH)")
    args = ap.parse_args(argv)

    if args.cargo is not None:
        # Preserve the path without dereferencing symlinks so rustup proxies keep argv[0] as cargo.
        cargo = os.path.abspath(str(args.cargo))
        if not os.path.isfile(cargo):
            print("build_library.py: cargo does not exist: %s" % args.cargo, file=sys.stderr)
            return 1
    else:
        cargo = shutil.which("cargo")
        if cargo is None:
            print("build_library.py: cargo is not on PATH; install Rust or use a prebuilt library", file=sys.stderr)
            return 1
    out = args.out.resolve()
    target_dir = (args.target_dir or out / "cargo-target").resolve()

    env = dict(os.environ)
    cargo_dir = os.path.dirname(cargo)
    if cargo_dir:
        path_entries = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
        if cargo_dir not in path_entries:
            env["PATH"] = os.pathsep.join([cargo_dir] + path_entries)
    rustc = env.get("RUSTC")
    if not rustc:
        rustc_name = "rustc.exe" if sys.platform == "win32" else "rustc"
        sibling_rustc = os.path.join(cargo_dir, rustc_name) if cargo_dir else ""
        if sibling_rustc and os.path.isfile(sibling_rustc):
            rustc = sibling_rustc
        else:
            rustc = shutil.which("rustc", path=env.get("PATH"))
    if not rustc:
        print("build_library.py: rustc is not found beside cargo or on PATH", file=sys.stderr)
        return 1
    env["RUSTC"] = rustc

    sysroot = subprocess.run([rustc, "--print", "sysroot"], capture_output=True, text=True, check=True, env=env).stdout.strip()
    cargo_home = env.get("CARGO_HOME") or str(pathlib.Path.home() / ".cargo")
    remaps = [(cargo_home, "/cargo"), (sysroot, "/rustc-sysroot"), (str(REPO), "/caliper"), (str(target_dir), "/target")]
    remap_flags = ["--remap-path-prefix=%s=%s" % pair for pair in remaps]
    if env.get("CARGO_ENCODED_RUSTFLAGS"):
        existing = [f for f in env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if f]
        env.pop("RUSTFLAGS", None)
    else:
        existing = env.pop("RUSTFLAGS", "").split()
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(existing + remap_flags)
    if sys.platform == "darwin":
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")

    cmd = [cargo, "build", "--release", "--locked", "--manifest-path", str(CRATE / "Cargo.toml"),
           "--target-dir", str(target_dir)]
    if args.jobs:
        cmd += ["-j", str(args.jobs)]
    if args.offline:
        cmd.append("--offline")
    if args.test_hooks:
        cmd += ["--features", "test-hooks"]
    if subprocess.run(cmd, env=env).returncode != 0:
        print("build_library.py: cargo build failed", file=sys.stderr)
        return 1

    built = target_dir / "release" / library_name()
    lib_dir = out / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    library = lib_dir / library_name()
    shutil.copy2(built, library)
    print("platform %s" % platform_name())
    print("library %s %d bytes sha256 %s" % (library, library.stat().st_size, sha256(library)))

    if args.archive:
        archive = out / ("caliper_wgsl-%s-%s.tar.gz" % (crate_version(), platform_name()))

        def clean(info):
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            return info

        # A fixed gzip time and fixed tar fields, so the same library gives the same archive.
        with open(archive, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            tar.add(library, arcname="lib/" + library_name(), filter=clean)
        digest = sha256(archive)
        (out / (archive.name + ".sha256")).write_text("%s  %s\n" % (digest, archive.name))
        print("archive %s %d bytes sha256 %s" % (archive, archive.stat().st_size, digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
