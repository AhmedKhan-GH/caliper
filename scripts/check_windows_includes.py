#!/usr/bin/env python3
"""Check on macOS that the code the Windows lane compiles reaches no macOS-only or POSIX-only header.

    python3 scripts/check_windows_includes.py --build cmake-build-release [--list] [--keep <folder>] [unit...]

It preprocesses every first-party C and C++ translation unit that the Windows
lane of the Build workflow compiles, as clang would for MSVC
(-target x86_64-pc-windows-msvc, so _WIN32, _WIN64 and _MSC_VER are defined
and __APPLE__ is not), with none of the macOS SDK's or the toolchain's headers
reachable (-nostdinc). Each header comes instead from a folder of stand-ins
that this script writes:

- the C++ standard library, the C library and the CRT headers that MSVC
  ships (sys/stat.h, fcntl.h, io.h, process.h, ...), as empty files;
- the Windows SDK headers that the code includes (windows.h, winsock2.h, ...),
  as empty files, and the headers the Windows build generates or fetches
  (volk, the Vulkan headers, the SPIR-V headers, and zlib, which the core
  build provides there), as empty files;
- every macOS-only or POSIX-only header that MSVC lacks (unistd.h, libproc.h,
  pthread.h, dlfcn.h, sys/mman.h, sys/socket.h, mach/..., ...), as a file
  whose only line is an #error.

So an include of such a header on the Windows path fails here as it fails
with MSVC, with the chain of includes that reached it. A header in none of
these lists stops its translation unit with "file not found", which is
reported too. Then it scans what each first-party file contributes to the
preprocessed output, on the Windows path only, for identifiers that MSVC's
headers do not declare although the headers exist (sigaction, kill, SIGKILL,
O_CLOEXEC, S_ISDIR, clock_gettime, M_PI without _USE_MATH_DEFINES, ...) and
for GCC and Clang extensions MSVC does not have (__attribute__, __builtin_*,
__atomic_*).

The translation units are those of the macOS build's compile_commands.json,
minus the macOS-only ones and plus the Windows-only ones, as CMakeLists.txt,
tests/CMakeLists.txt and the applet folders choose them (WINDOWS_ONLY and
MACOS_ONLY below), and the applets that scripts/build_applets_against_sdk.sh
builds in the Windows lane. It needs a macOS build configured with libtorch
and the tests (the release preset). It cannot see a type error, an overload that
differs, a missing declaration in a Windows header, a link error or anything
the C++ standard library headers decide, since every one of those headers is
empty here; MSVC itself is the check for those. The exit code is 1 when
anything is found.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- which translation units the Windows lane compiles --------------------------------------
# macOS-only translation units of the macOS build (if(APPLE), if(NOT WIN32) and the
# live-session filter of CMakeLists.txt; the live and socket tests of tests/CMakeLists.txt).
MACOS_ONLY = {
    "src/host/live/live_core.cpp",
    "src/host/live/live_attach.cpp",
    "tests/test_net_ws.cpp",
    "tests/test_net_hub.cpp",
    "tests/test_api.cpp",
    "tests/test_live.cpp",
    "tests/test_live_net.cpp",
    "tests/test_live_views.cpp",
}
# Windows-only translation units, each with the macOS translation unit whose
# include paths and definitions it shares (its target's).
WINDOWS_ONLY = {
    "src/host/feed_provider_win.cpp": "src/host/applet_loader.cpp",
    "src/host/device_query_cuda.cpp": "src/host/applet_loader.cpp",
    "src/host/cuda_driver.cpp": "src/host/applet_loader.cpp",
    "src/host/live/live_unsupported.cpp": "src/host/live/live_controls.cpp",
    "src/host/renderer/vulkan_renderer.cpp": "src/host/renderer/gl_renderer.cpp",
    "examples/embed_host/main_win32.cpp": "tests/test_embed.cpp",
    "tests/sdk_install_probe/probe.cpp": "tests/test_sugar.cpp",
}
# tests/sdk_install_probe/no_embed/no_embed.cpp is left out: it is built to fail
# (an applet that includes <caliper/embed.h> must not compile).
# The applets the Windows lane builds against the installed SDK
# (scripts/build_applets_against_sdk.sh with no folder: every scope but these).
APPLETS_SKIPPED_ON_WINDOWS = {"embed_scope", "gpt_scope", "legacy", "legacy-dev", "common"}
# An applet gets the SDK, UI-stack and export definitions of examples/hello and the libtorch
# include paths of tests/test_sculpt.cpp.
APPLET_DONORS = ["examples/hello/hello.cpp", "tests/test_sculpt.cpp"]
EXTRA_DEFINES = ["WIN32", "_WINDOWS", "_MBCS", "VK_USE_PLATFORM_WIN32_KHR", "IMGUI_IMPL_VULKAN_USE_VOLK"]
MACOS_DEFINES = {"CALIPER_HAVE_METAL"}

# ---- the stand-in headers -----------------------------------------------------------------
CXX_STD = """algorithm any array atomic barrier bit bitset charconv chrono codecvt compare complex
concepts condition_variable coroutine deque exception execution expected filesystem format
forward_list fstream functional future initializer_list iomanip ios iosfwd iostream istream
iterator latch limits list locale map memory memory_resource mutex new numbers numeric optional
ostream print queue random ranges ratio regex scoped_allocator semaphore set shared_mutex
source_location span spanstream sstream stack stacktrace stdexcept stop_token streambuf string
string_view strstream syncstream system_error thread tuple type_traits typeindex typeinfo
unordered_map unordered_set utility valarray variant vector version cassert cctype cerrno cfenv
cfloat cinttypes ciso646 climits clocale cmath csetjmp csignal cstdarg cstdbool cstddef cstdint
cstdio cstdlib cstring ctgmath ctime cuchar cwchar cwctype""".split()
# The C library and the CRT headers MSVC ships, POSIX-like ones included.
C_AND_CRT = """assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h
math.h setjmp.h signal.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdio.h
stdlib.h stdnoreturn.h string.h tgmath.h threads.h time.h uchar.h wchar.h wctype.h
io.h direct.h process.h fcntl.h share.h malloc.h memory.h search.h conio.h crtdbg.h dos.h
excpt.h eh.h new.h tchar.h mbstring.h mbctype.h sys/stat.h sys/types.h sys/timeb.h sys/utime.h
sys/locking.h intrin.h immintrin.h xmmintrin.h emmintrin.h pmmintrin.h tmmintrin.h smmintrin.h
nmmintrin.h wmmintrin.h ammintrin.h mmintrin.h""".split()
# Windows SDK headers the code includes; add a name here when a Windows file needs one.
WINDOWS_SDK = """windows.h winsock2.h ws2tcpip.h mswsock.h afunix.h iphlpapi.h shlobj.h shlwapi.h
knownfolders.h objbase.h combaseapi.h psapi.h tlhelp32.h dbghelp.h powrprof.h winternl.h ntstatus.h
pdh.h pdhmsg.h wbemidl.h comdef.h comutil.h oleauto.h shellapi.h commdlg.h winreg.h synchapi.h
processthreadsapi.h sysinfoapi.h winbase.h windef.h winuser.h wingdi.h wincrypt.h bcrypt.h sddl.h
aclapi.h userenv.h intsafe.h mmsystem.h timeapi.h dwmapi.h shellscalingapi.h versionhelpers.h
winioctl.h setupapi.h devguid.h cfgmgr32.h initguid.h dxgi.h dxgi1_2.h dxgi1_4.h dxgi1_6.h
d3d11.h d3d12.h wrl.h wrl/client.h unknwn.h objidl.h propidl.h rpc.h rpcndr.h guiddef.h
specstrings.h sal.h winerror.h winnt.h minwindef.h basetsd.h errhandlingapi.h fileapi.h handleapi.h
heapapi.h libloaderapi.h memoryapi.h namedpipeapi.h profileapi.h securitybaseapi.h stringapiset.h
GL/gl.h GL/glu.h powerbase.h windowsx.h""".split()
# Headers the Windows build generates or fetches (FetchContent volk and Vulkan-Headers, glslang
# SPIR-V headers, the PTX header) and that the macOS build does not have.
WINDOWS_GENERATED = """volk.h vulkan/vulkan.h vulkan/vulkan_core.h vulkan/vulkan_win32.h
vulkan/vk_platform.h colormap_spv.h points_vert_spv.h points_frag_spv.h geom_vert_spv.h
geom_frag_spv.h geom_tex_frag_spv.h cmap_direct_ptx.h
zlib.h""".split()   # zlib.h: the core build provides zlib on Windows (ZLIB::ZLIB)
# macOS-only or POSIX-only headers that MSVC lacks.
POISON = """unistd.h libproc.h pthread.h sched.h semaphore.h dlfcn.h spawn.h poll.h netdb.h dirent.h
libgen.h pwd.h grp.h termios.h execinfo.h xlocale.h crt_externs.h copyfile.h strings.h fnmatch.h
glob.h syslog.h ucontext.h ifaddrs.h iconv.h langinfo.h utime.h
sys/mman.h sys/socket.h sys/sysctl.h sys/time.h sys/wait.h sys/un.h sys/ioctl.h sys/select.h
sys/event.h sys/param.h sys/resource.h sys/uio.h sys/file.h sys/utsname.h sys/xattr.h sys/attr.h
sys/clonefile.h sys/sem.h sys/shm.h sys/ipc.h sys/statvfs.h sys/mount.h sys/times.h sys/ptrace.h
sys/sysctl.h sys/kern_control.h sys/proc_info.h netinet/in.h netinet/tcp.h arpa/inet.h net/if.h
mach/mach.h mach/mach_time.h mach/mach_init.h mach/mach_host.h mach/task.h mach/task_info.h
mach/vm_map.h mach/vm_statistics.h mach/thread_act.h mach/semaphore.h mach-o/dyld.h
os/log.h os/lock.h os/signpost.h os/proc.h os/object.h dispatch/dispatch.h objc/objc.h
objc/runtime.h objc/message.h CoreFoundation/CoreFoundation.h Foundation/Foundation.h
AppKit/AppKit.h Cocoa/Cocoa.h Metal/Metal.h MetalKit/MetalKit.h QuartzCore/QuartzCore.h
QuartzCore/CAMetalLayer.h IOSurface/IOSurface.h IOSurface/IOSurfaceRef.h IOKit/IOKitLib.h
IOKit/ps/IOPowerSources.h CoreVideo/CoreVideo.h CoreGraphics/CoreGraphics.h Accelerate/Accelerate.h
TargetConditionals.h Availability.h AvailabilityMacros.h malloc/malloc.h libkern/OSAtomic.h
simd/simd.h arm_neon.h""".split()

# Identifiers that MSVC's headers do not declare, or extensions it does not have.
SUSPECT = """sigaction sigemptyset sigaddset sigprocmask pthread_sigmask kill killpg SIGKILL SIGHUP
SIGPIPE SIGUSR1 SIGUSR2 SIGCHLD SIGALRM SIGQUIT SA_RESTART SA_SIGINFO O_CLOEXEC O_NONBLOCK
O_NOFOLLOW O_DIRECTORY O_SYNC F_GETFL F_SETFL F_SETFD F_GETFD FD_CLOEXEC fcntl S_IRUSR S_IWUSR
S_IXUSR S_IRWXU S_IRGRP S_IWGRP S_IROTH S_IWOTH S_ISREG S_ISDIR S_ISLNK lstat fchmod fstatat
mkfifo mknod fsync fdatasync ftruncate pread pwrite flock lockf symlink readlink realpath mkstemp
mkdtemp getpagesize sysconf nanosleep usleep sleep clock_gettime clock_gettime_nsec_np
CLOCK_MONOTONIC CLOCK_REALTIME CLOCK_UPTIME_RAW gettimeofday localtime_r gmtime_r timegm strptime
setenv unsetenv strerror_r strtok_r strcasecmp strncasecmp strsep strlcpy strlcat getline fork
waitpid posix_spawn posix_spawnp getppid getuid geteuid getpwuid mmap munmap mprotect madvise
shm_open shm_unlink sem_open dlopen dlsym dlclose dlerror socketpair poll MSG_NOSIGNAL
SO_NOSIGPIPE sockaddr_un mach_absolute_time os_unfair_lock proc_pidpath proc_pidinfo getprogname
_NSGetExecutablePath arc4random pipe environ __attribute__ __builtin_expect __builtin_clz
__builtin_clzll __builtin_ctz __builtin_ctzll __builtin_popcount __builtin_popcountll
__builtin_bswap16 __builtin_bswap32 __builtin_bswap64 __builtin_unreachable __builtin_trap
__builtin_assume __builtin_prefetch __builtin_memcpy __atomic_load_n __atomic_store_n
__atomic_fetch_add __atomic_fetch_sub __atomic_exchange_n __atomic_compare_exchange_n
__sync_synchronize __sync_fetch_and_add __thread __int128 typeof __typeof__ __PRETTY_FUNCTION__
M_E M_LOG2E M_LOG10E M_LN2 M_LN10 M_PI M_PI_2 M_PI_4 M_1_PI M_2_PI M_2_SQRTPI M_SQRT2 M_SQRT1_2""".split()
# (MSVC's math.h defines the M_ constants only with _USE_MATH_DEFINES, as the stand-in does.)


# What the preprocessor needs from the empty standard headers, as MSVC on x64 (LLP64) defines it.
STD_MACROS = {
    "climits": "limits.h", "limits.h": """#pragma once
#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 0xff
#define CHAR_MIN SCHAR_MIN
#define CHAR_MAX SCHAR_MAX
#define SHRT_MIN (-32768)
#define SHRT_MAX 32767
#define USHRT_MAX 0xffff
#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647
#define UINT_MAX 0xffffffff
#define LONG_MIN (-2147483647L - 1)
#define LONG_MAX 2147483647L
#define ULONG_MAX 0xffffffffUL
#define LLONG_MAX 9223372036854775807i64
#define LLONG_MIN (-9223372036854775807i64 - 1)
#define ULLONG_MAX 0xffffffffffffffffui64
""",
    # As the UCRT's math.h: no include guard around the constants, so a later
    # #define _USE_MATH_DEFINES and #include <math.h> still defines them (libtorch does that).
    "cmath": "math.h", "math.h": """#if defined(_USE_MATH_DEFINES) && !defined(_MATH_DEFINES_DEFINED)
#define _MATH_DEFINES_DEFINED
#define M_E 2.71828182845904523536
#define M_LOG2E 1.44269504088896340736
#define M_LOG10E 0.434294481903251827651
#define M_LN2 0.693147180559945309417
#define M_LN10 2.30258509299404568402
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.785398163397448309616
#define M_1_PI 0.318309886183790671538
#define M_2_PI 0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2 0.707106781186547524401
#endif
""",
    "cstdint": "stdint.h", "stdint.h": """#pragma once
#define INT8_MAX 127
#define INT16_MAX 32767
#define INT32_MAX 2147483647
#define INT64_MAX 9223372036854775807LL
#define UINT8_MAX 0xff
#define UINT16_MAX 0xffff
#define UINT32_MAX 0xffffffffU
#define UINT64_MAX 0xffffffffffffffffULL
#define SIZE_MAX UINT64_MAX
#define INTPTR_MAX INT64_MAX
#define UINTPTR_MAX UINT64_MAX
""",
}


def write_standins(base: str) -> list[str]:
    dirs = []
    for name, files, text in (("std", CXX_STD + C_AND_CRT, ""),
                              ("windows", WINDOWS_SDK + WINDOWS_GENERATED, ""),
                              ("poison", POISON, None)):
        d = os.path.join(base, name)
        for f in files:
            p = os.path.join(d, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            body = STD_MACROS.get(f, "")
            if body in STD_MACROS:   # a C++ header that includes its C header
                body = f"#include <{body}>\n"
            with open(p, "w") as fh:
                fh.write(body if text is not None and name == "std" else text if text is not None else
                         f'#error "MSVC has no <{f}>: an unguarded include on the Windows path"\n')
        dirs.append(d)
    return dirs


# ---- the commands -------------------------------------------------------------------------
def first_party(path: str) -> bool:
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    return not rel.startswith("..") and "third_party" not in parts and "_deps" not in parts


def flags_of(entry: dict) -> tuple[list[str], str]:
    """The include paths and definitions of a compile command (system and toolchain ones dropped), and its language standard."""
    args = shlex.split(entry["command"]) if "command" in entry else entry["arguments"]
    out, std = [], "c++20"
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-I", "-isystem", "-iquote") and i + 1 < len(args):
            if args[i + 1].startswith(ROOT):
                out += [a, args[i + 1]]
            i += 2
            continue
        if a.startswith("-I") and a[2:].startswith(ROOT):
            out.append(a)
        elif a.startswith("-D") and a[2:].split("=")[0] not in MACOS_DEFINES:
            out.append(a)
        elif a.startswith("-std="):
            std = a[5:].replace("gnu", "c").replace("c++2a", "c++20")
        i += 1
    return out, std


def translation_units(build: str) -> list[tuple[str, list[str], str]]:
    cc = json.load(open(os.path.join(build, "compile_commands.json")))
    by_rel = {}
    for e in cc:
        f = os.path.normpath(os.path.join(e["directory"], e["file"]))
        by_rel[os.path.relpath(f, ROOT)] = e
    units = []
    for rel, e in sorted(by_rel.items()):
        if not first_party(os.path.join(ROOT, rel)) or rel in MACOS_ONLY:
            continue
        if not rel.endswith((".cpp", ".cc", ".c")):
            continue   # Objective-C++ is macOS-only
        fl, std = flags_of(e)
        units.append((rel, fl, std if not rel.endswith(".c") else "c11"))
    for rel, donor in WINDOWS_ONLY.items():
        if donor not in by_rel:
            sys.exit(f"the donor {donor} of {rel} is not in compile_commands.json")
        fl, std = flags_of(by_rel[donor])
        units.append((rel, fl, std))
    afl, astd = [], "c++17"
    for donor in APPLET_DONORS:
        if donor not in by_rel:
            sys.exit(f"{donor} is not in compile_commands.json (a build with libtorch and tests is needed)")
        fl, astd = flags_of(by_rel[donor])
        k = 0
        while k < len(fl):   # merge, keeping "-isystem <dir>" pairs together
            item = fl[k:k + 2] if fl[k] in ("-I", "-isystem", "-iquote") else fl[k:k + 1]
            k += len(item)
            if not any(afl[m:m + len(item)] == item for m in range(len(afl))):
                afl += item
    for scope in sorted(os.listdir(os.path.join(ROOT, "applets"))):
        d = os.path.join(ROOT, "applets", scope)
        if scope in APPLETS_SKIPPED_ON_WINDOWS or not os.path.exists(os.path.join(d, "CMakeLists.txt")):
            continue
        for dirpath, _, files in os.walk(d):
            if "/tests" in dirpath[len(d):]:
                continue
            for f in sorted(files):
                if f.endswith((".cpp", ".c")):
                    rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                    units.append((rel, afl + ["-I" + d, "-I" + os.path.join(ROOT, "applets", "common")], astd))
    return units


INCLUDED_FROM = re.compile(r"^In file included from (.+?):(\d+):")
DIAG = re.compile(r"^(.+?):(\d+):\d+: (fatal error|error): (.*)$")
LINEMARK = re.compile(r'^# (\d+) "(.*)"')
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_][A-Za-z_0-9]*|->|::|\S')
SUSPECT_SET = set(SUSPECT)
# A function name counts only where it is called as a free function: not after
# '.', '->' or 'X::', and not where a type or a name precedes it (a declaration).
NOT_A_CALL_AFTER = {".", "->", ">", "*", "&", "~"}
# Sites read and found harmless: file, identifier and a piece of the source line.
REVIEWED = {
    ("src/host/net/ws.cpp", "poll", "ev = poll((int)"),   # Server::poll, called inside the class
    # caliper_gfx::Backend::poll(), with no arguments, called inside the class
    # (src/host/gfx/gfx.h); POSIX poll takes three.
    ("src/host/gfx/core.cpp", "poll", "poll();"),
    ("src/host/gfx/recording_backend.cpp", "poll", "poll();"),
}


def rel_of(path: str) -> str:
    return os.path.relpath(path, ROOT)


def reviewed(path: str, lineno: int, ident: str) -> bool:
    try:
        with open(path, errors="replace") as fh:
            text = fh.read().splitlines()[lineno - 1]
    except (OSError, IndexError):
        return False
    return any(f == rel_of(path) and i == ident and piece in text for f, i, piece in REVIEWED)


def scan_line(tokens: list[str]) -> list[str]:
    hits = []
    for k, t in enumerate(tokens):
        if t not in SUSPECT_SET:
            continue
        prev = tokens[k - 1] if k > 0 else ""
        prev2 = tokens[k - 2] if k > 1 else ""
        if prev in (".", "->"):
            continue
        if prev == "::" and (prev2[:1].isalnum() or prev2[:1] == "_" or prev2 == ">"):
            continue   # qualified: std::getline, Server::poll
        function_like = t[:1].islower() and not t.startswith("__")
        if function_like:
            nxt = tokens[k + 1] if k + 1 < len(tokens) else ""
            if nxt != "(":
                continue
            if prev in NOT_A_CALL_AFTER or (prev[:1].isalnum() or prev[:1] == "_") and prev not in (
                    "return", "else", "do", "case", "throw", "co_return", "co_await"):
                continue
        hits.append(t)
    return hits


def check(unit, standins, keep):
    rel, fl, std = unit
    lang = ["-x", "c"] if std.startswith("c1") else ["-x", "c++"]
    cmd = (["clang", "-E", "-target", "x86_64-pc-windows-msvc", "-fms-extensions", "-fms-compatibility",
            "-fms-compatibility-version=19.40", "-nostdinc", "-nostdinc++", "-U__clang__",
            "-Wno-builtin-macro-redefined", "-Wno-pragma-system-header-outside-header"]
           + lang + ["-std=" + std] + ["-D" + d for d in EXTRA_DEFINES] + fl
           + sum((["-isystem", d] for d in standins), []) + [os.path.join(ROOT, rel)])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    findings = []
    chain = []
    for line in r.stderr.splitlines():
        m = INCLUDED_FROM.match(line)
        if m:
            chain.append(f"{rel_of(m.group(1))}:{m.group(2)}")
            continue
        m = DIAG.match(line)
        if m:
            via = " -> ".join(chain + [f"{rel_of(m.group(1))}:{m.group(2)}"])
            findings.append(("include", f"{m.group(4)}; via {via}"))
        if not line.startswith(" "):
            chain = []
    # What first-party files contribute on the Windows path.
    current, lineno, seen = None, 0, set()
    for line in r.stdout.splitlines():
        m = LINEMARK.match(line)
        if m:
            current, lineno = m.group(2), int(m.group(1)) - 1
            continue
        lineno += 1
        if not current or not first_party(current) or not os.path.isabs(current):
            continue
        for ident in scan_line(TOKENS.findall(line)):
            key = (rel_of(current), ident)
            if (key, lineno) in seen or reviewed(current, lineno, ident):
                continue
            seen.add((key, lineno))
            findings.append(("identifier", f"{ident} at {rel_of(current)}:{lineno}"))
    if keep:
        with open(os.path.join(keep, rel.replace("/", "__") + ".i"), "w") as fh:
            fh.write(r.stdout)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", default="cmake-build-release", help="a configured macOS build folder")
    ap.add_argument("--list", action="store_true", help="only list the translation units")
    ap.add_argument("--keep", help="write each preprocessed unit into this folder")
    ap.add_argument("only", nargs="*", help="check only the units whose path contains one of these")
    args = ap.parse_args()
    build = os.path.abspath(args.build)
    units = [u for u in translation_units(build) if not args.only or any(o in u[0] for o in args.only)]
    if args.list:
        for u in units:
            print(u[0])
        print(f"{len(units)} translation units")
        return 0
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    base = tempfile.mkdtemp(prefix="caliper-wincheck-", dir=build)
    try:
        standins = write_standins(base)
        bad = 0
        for u in units:
            found = check(u, standins, args.keep)
            if found:
                bad += 1
                print(f"FAIL  {u[0]}")
                for kind, text in found:
                    print(f"      {kind}: {text}")
            else:
                print(f"ok    {u[0]}")
        print(f"{len(units)} translation units, {bad} with findings")
        return 1 if bad else 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
