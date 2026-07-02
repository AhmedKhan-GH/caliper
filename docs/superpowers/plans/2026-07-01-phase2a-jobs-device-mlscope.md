# Phase 2A — `jobs.v1` + `device.v1` + ml_scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the first two Phase-2 services — background jobs with progress/cancel and host-negotiated device identity — validated by a new torch-linked exemplar applet (`examples/ml_scope`) that trains a tiny MLP off the frame thread. Step 1 of the ratified Phase-2 sequencing (PLATFORM.md §17 Phase 2).

**Architecture:** Service tables follow the frozen epoch-2 pattern (immutable C structs behind `get_service`). `JobSystem` (thread-per-job, poll-based cancel — the exact pattern extracted from repnet's `train_engine.cpp`: `std::atomic<bool>` stop flag + mutex-published state) lives in `src/host/` as a UI-free testable unit; the jobs tray is main.cpp glue over its `views()`. Device detection uses Metal directly (`MTLCreateSystemDefaultDevice`) — the host never links torch (D11); applets map `CALIPER_DEV_METAL` to their framework's device themselves.

**Tech Stack:** as Phase 0+1 (C++20 consumers, doctest, existing build tree in `build/`), plus Objective-C++ (`.mm`, `enable_language(OBJCXX)` on APPLE) and the Metal framework for device query; ml_scope links the in-tree libtorch exactly as repnet_demo does.

## Global Constraints

- **Carried from the Phase 0+1 plan verbatim:** TDD for every `src/host/` unit and sugar; every task ends green (`cmake --build build` + full `ctest` + strict mkdocs when docs change); conventional commits ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; docs ride in the same commit per the mapping below; build dir `build/`; never `git add -A`.
- **Branch:** `platform/phase-2a` from `main`; merge `--no-ff` at plan end.
- **Versions (fixed):** service ids `"caliper.jobs.v1"`, `"caliper.device.v1"`; ml_scope id `dev.caliper.ml-scope`, version `0.1.0`; device kinds `CALIPER_DEV_CPU=0, CALIPER_DEV_CUDA=1, CALIPER_DEV_METAL=2` (METAL names the memory domain — spec §7.2 amended accordingly; never "MPS" in ABI).
- **Contract (§16):** cancel honored ≤ 100 ms is a tested guarantee, not advisory.
- **Honesty rule (§15 interplay):** job functions run on host worker threads UNGUARDED — the crash guard is UI-thread-only by documented precondition. Trusted code; a fault in a job kills the process. This is stated in the header, the docs, and trust-model.md — never softened.
- **The host never links torch/DuckDB-for-this/any ML framework** (D11). Device detection is Metal-API-only on Apple; CUDA detection is Phase 4 (no hardware here).
- **ml_scope teaching rule:** no CPU-staged weight visualization — the weight-matrix view arrives with `tensor_bridge.v1` (Plan 2C). The exemplar must never demonstrate a pattern the platform is about to obsolete. Loss curve is applet-local ImPlot (that's applet data, not a service concern until metrics.v1 in Plan 2B).
- **Do not touch:** `applets/*` internals, `examples/hello`, `examples/signal_scope`, `sdk/include/caliper/abi.h` (frozen), `third_party/`, `cmake-build-debug/`.

## File Map

```
sdk/include/caliper/services/jobs_v1.h     A1 (frozen once shipped)
sdk/include/caliper/services/device_v1.h   A1 (frozen once shipped)
tests/test_abi.cpp, tests/abi_c_check.c    A1 (extended)
src/host/job_system.h/.cpp                 A2
tests/test_jobs.cpp                        A2
src/host/device_query.h                    A3
src/host/device_query_apple.mm             A3 (APPLE)
src/host/device_query_stub.cpp             A3 (non-APPLE)
tests/test_device.cpp                      A3
src/host/host_services.h/.cpp              A4 (vend both; expose JobSystem to tray)
sdk/include/caliper/caliper.hpp            A4 (Jobs/Device typed wrappers)
sdk/testing/caliper/fixture_host.h         A4 (generic provide() injection)
tests/test_sugar_services.cpp              A4
src/main.cpp                               A5 (jobs tray overlay)
examples/ml_scope/{CMakeLists.txt,ml_scope.cpp,ml_scope.caliper.toml}  A6
CMakeLists.txt                             A3 (OBJCXX+Metal), A6 (examples wiring)
docs/wiki/reference/services/jobs-v1.md    A6 } new pages + nav entries
docs/wiki/reference/services/device-v1.md  A6 } (mkdocs.yml nav edit allowed)
docs/wiki/explanation/trust-model.md       A6 (jobs-unguarded paragraph)
```

Docs mapping (same-commit rule): A1 → stub the two service pages with header embeds; A6 → full semantics + tutorials cross-link + trust-model paragraph + nav.

---

### Task A1: Service headers + ABI test extension

**Files:** Create `sdk/include/caliper/services/jobs_v1.h`, `sdk/include/caliper/services/device_v1.h`; Modify `tests/test_abi.cpp`, `tests/abi_c_check.c`; Create stub docs pages `docs/wiki/reference/services/jobs-v1.md`, `device-v1.md` (H1 + one-line intro + `--8<--` embed of the header + `*Semantics: written at Task A6.*`), add both to `mkdocs.yml` nav under Services.

**Interfaces — Produces:** `CALIPER_JOBS_V1`, `CaliperJobControl{struct_size, cancelled, progress}`, `CaliperJobFn`, `CaliperJobsV1{struct_size, submit, request_cancel, is_running, progress_of}`; `CALIPER_DEVICE_V1`, `CaliperDeviceKind{CPU=0,CUDA=1,METAL=2}`, `CaliperDeviceV1{struct_size, kind, index, name, free_memory_hint}`. Consumed by every later task.

- [ ] **Step 1: extend the tests first (RED)**

Append to `tests/test_abi.cpp` (with the existing includes pattern):
```cpp
#include <caliper/services/jobs_v1.h>
#include <caliper/services/device_v1.h>

static_assert(std::is_standard_layout_v<CaliperJobControl>);
static_assert(std::is_standard_layout_v<CaliperJobsV1>);
static_assert(std::is_standard_layout_v<CaliperDeviceV1>);
static_assert(offsetof(CaliperJobControl, struct_size) == 0);
static_assert(offsetof(CaliperJobsV1, struct_size) == 0);
static_assert(offsetof(CaliperDeviceV1, struct_size) == 0);
static_assert(CALIPER_DEV_CPU == 0 && CALIPER_DEV_CUDA == 1 && CALIPER_DEV_METAL == 2);

TEST_CASE("abi: phase-2a service ids are fixed") {
    CHECK(std::string(CALIPER_JOBS_V1) == "caliper.jobs.v1");
    CHECK(std::string(CALIPER_DEVICE_V1) == "caliper.device.v1");
}
```
Append to `tests/abi_c_check.c` includes: `#include <caliper/services/jobs_v1.h>` and `#include <caliper/services/device_v1.h>`.

- [ ] **Step 2: run to verify RED** — build `caliper_tests` → FAIL, `caliper/services/jobs_v1.h` not found.

- [ ] **Step 3: create the headers**

`sdk/include/caliper/services/jobs_v1.h`:
```c
#pragma once
/* caliper.jobs.v1 — background compute with progress + cancel (PLATFORM.md
 * §7.5). IMMUTABLE once published: new capability = jobs_v2, alongside.
 *
 * THREADING HONESTY (§15): job functions run on HOST WORKER THREADS as
 * trusted code. They are NOT crash-guarded — the signal guard is
 * UI-thread-only by documented precondition — so a fault in a job takes the
 * process down. Cancellation is cooperative: poll cancelled() in your inner
 * loop and return promptly. */
#include <stdint.h>
#include <stdbool.h>

#define CALIPER_JOBS_V1 "caliper.jobs.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperJobControl CaliperJobControl;
struct CaliperJobControl {
    uint32_t struct_size;
    /* Poll in loops; return promptly when true. */
    bool (*cancelled)(const CaliperJobControl* ctl);
    /* frac in [0,1]; msg_utf8 may be NULL. Surfaced in the host jobs tray. */
    void (*progress)(const CaliperJobControl* ctl, float frac,
                     const char* msg_utf8);
};

/* Runs on a host worker thread. user must outlive the job. */
typedef void (*CaliperJobFn)(void* user, const CaliperJobControl* ctl);

typedef struct CaliperJobsV1 {
    uint32_t struct_size;
    /* Returns a job id; 0 = error (never a valid id). */
    uint64_t (*submit)(const char* label_utf8, CaliperJobFn fn, void* user);
    void     (*request_cancel)(uint64_t job);
    bool     (*is_running)(uint64_t job);
    float    (*progress_of)(uint64_t job);  /* last reported frac; 0 if none */
} CaliperJobsV1;

#ifdef __cplusplus
}
#endif
```

`sdk/include/caliper/services/device_v1.h`:
```c
#pragma once
/* caliper.device.v1 — the host's negotiated compute device (PLATFORM.md
 * §7.3). IMMUTABLE once published. Kinds name the MEMORY/API DOMAIN, not a
 * framework backend: METAL covers torch-MPS, MLX, and ggml-Metal alike. The
 * host detects without linking any ML framework (D11); applets map the kind
 * to their framework's device (torch: METAL -> torch::kMPS). */
#include <stdint.h>

#define CALIPER_DEVICE_V1 "caliper.device.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperDeviceKind {
    CALIPER_DEV_CPU   = 0,
    CALIPER_DEV_CUDA  = 1,
    CALIPER_DEV_METAL = 2
} CaliperDeviceKind;

typedef struct CaliperDeviceV1 {
    uint32_t struct_size;
    CaliperDeviceKind (*kind)(void);
    int32_t           (*index)(void);             /* 0 for CPU/METAL */
    const char*       (*name)(void);              /* host-owned, e.g. "Apple M3 Max" */
    uint64_t          (*free_memory_hint)(void);  /* bytes, best-effort; 0 = unknown */
} CaliperDeviceV1;

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 4: GREEN** — `cmake --build build --target caliper_tests -j && ctest --test-dir build --output-on-failure` all pass (C file included, so the headers are C-verified too). Docs stubs: strict mkdocs green.
- [ ] **Step 5: Commit** — `feat(sdk): jobs.v1 + device.v1 service headers (Phase 2A)` (headers, both test files, two doc stubs, mkdocs.yml).

---

### Task A2: JobSystem — TDD

**Files:** Create `src/host/job_system.h`, `src/host/job_system.cpp`, `tests/test_jobs.cpp`; Modify root `CMakeLists.txt` (add `job_system.cpp` to `caliper_host_lib`), `tests/CMakeLists.txt` (add test file).

**Interfaces — Produces:** `caliper_host::JobSystem{submit(label,fn,user)→id, request_cancel(id), is_running(id), progress_of(id), views()→vector<JobView{id,label,progress,message,running}>, cancel_all_and_join()}`. Consumed by A4 (service thunks) and A5 (tray).

- [ ] **Step 1: tests first**

`tests/test_jobs.cpp`:
```cpp
#include <doctest/doctest.h>
#include "job_system.h"
#include <atomic>
#include <chrono>
#include <thread>
using namespace caliper_host;
using namespace std::chrono;

namespace {
bool wait_until(const std::function<bool()>& pred, milliseconds timeout) {
    auto deadline = steady_clock::now() + timeout;
    while (steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(milliseconds(1));
    }
    return pred();
}
} // namespace

TEST_CASE("jobs: runs to completion, progress reported") {
    JobSystem js;
    std::atomic<bool> ran{false};
    auto fn = [](void* user, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 1.0f, "done");
        static_cast<std::atomic<bool>*>(user)->store(true);
    };
    uint64_t id = js.submit("t", fn, &ran);
    REQUIRE(id != 0);
    CHECK(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(ran.load());
    CHECK(js.progress_of(id) == doctest::Approx(1.0f));
}

TEST_CASE("jobs: cancel honored within 100ms (§16 contract)") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        while (!ctl->cancelled(ctl))
            std::this_thread::sleep_for(milliseconds(1));
    };
    uint64_t id = js.submit("spin", fn, nullptr);
    REQUIRE(wait_until([&] { return js.is_running(id); }, milliseconds(2000)));
    auto t0 = steady_clock::now();
    js.request_cancel(id);
    CHECK(wait_until([&] { return !js.is_running(id); }, milliseconds(100)));
    CHECK(duration_cast<milliseconds>(steady_clock::now() - t0).count() <= 100);
}

TEST_CASE("jobs: progress + message visible in views") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 0.5f, "halfway");
        while (!ctl->cancelled(ctl))
            std::this_thread::sleep_for(milliseconds(1));
    };
    uint64_t id = js.submit("labelled", fn, nullptr);
    REQUIRE(wait_until([&] { return js.progress_of(id) == 0.5f; },
                       milliseconds(2000)));
    auto vs = js.views();
    REQUIRE(!vs.empty());
    bool found = false;
    for (auto& v : vs)
        if (v.id == id) {
            found = true;
            CHECK(v.label == "labelled");
            CHECK(v.message == "halfway");
            CHECK(v.running);
        }
    CHECK(found);
    js.request_cancel(id);
}

TEST_CASE("jobs: unknown id is inert") {
    JobSystem js;
    CHECK_FALSE(js.is_running(424242));
    CHECK(js.progress_of(424242) == 0.0f);
    js.request_cancel(424242);  // no crash
}

TEST_CASE("jobs: destructor cancels and joins") {
    std::atomic<bool> exited{false};
    {
        JobSystem js;
        auto fn = [](void* user, const CaliperJobControl* ctl) {
            while (!ctl->cancelled(ctl))
                std::this_thread::sleep_for(milliseconds(1));
            static_cast<std::atomic<bool>*>(user)->store(true);
        };
        js.submit("forever", fn, &exited);
        std::this_thread::sleep_for(milliseconds(20));
    }   // ~JobSystem must cancel + join, not hang
    CHECK(exited.load());
}

TEST_CASE("jobs: concurrent jobs get distinct ids and all finish") {
    JobSystem js;
    std::atomic<int> done{0};
    auto fn = [](void* user, const CaliperJobControl*) {
        static_cast<std::atomic<int>*>(user)->fetch_add(1);
    };
    uint64_t a = js.submit("a", fn, &done), b = js.submit("b", fn, &done),
             c = js.submit("c", fn, &done);
    CHECK(a != b);
    CHECK(b != c);
    CHECK(wait_until([&] { return done.load() == 3; }, milliseconds(2000)));
}
```
(Note: the lambdas are captureless → convert to `CaliperJobFn`. `#include <functional>` for the helper.)

- [ ] **Step 2: RED** — missing `job_system.h`.
- [ ] **Step 3: implement**

`src/host/job_system.h`:
```cpp
#pragma once
#include <caliper/services/jobs_v1.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caliper_host {

// Thread-per-job v1 (PLATFORM.md §7.5), the pattern extracted from repnet's
// train_engine (atomic stop flag polled by the loop; state published under a
// mutex). Job fns are trusted code on worker threads — NOT crash-guarded
// (the guard is UI-thread-only, see crash_guard.h preconditions).
class JobSystem {
public:
    JobSystem() = default;
    ~JobSystem() { cancel_all_and_join(); }

    uint64_t submit(std::string label, CaliperJobFn fn, void* user);
    void     request_cancel(uint64_t id);
    bool     is_running(uint64_t id) const;
    float    progress_of(uint64_t id) const;

    struct JobView {
        uint64_t id = 0;
        std::string label;
        float progress = 0.0f;
        std::string message;
        bool running = false;
    };
    std::vector<JobView> views() const;   // running jobs first, then recent

    void cancel_all_and_join();

private:
    struct Job;
    Job* find(uint64_t id) const;         // caller holds mutex_
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Job>> jobs_;
    uint64_t next_id_ = 1;
};

} // namespace caliper_host
```

`src/host/job_system.cpp`:
```cpp
#include "job_system.h"
#include <atomic>

namespace caliper_host {

// The C control block handed to job fns. ctl is the FIRST member of a
// standard-layout struct, so the thunks recover the block from the ctl
// pointer with a cast — the same first-member idiom the fixture host uses.
struct ControlBlock {
    CaliperJobControl ctl{};
    std::atomic<bool> cancelled{false};
    std::atomic<float> progress{0.0f};
    std::mutex msg_mutex;
    std::string message;
};
static_assert(offsetof(ControlBlock, ctl) == 0);

struct JobSystem::Job {
    uint64_t id = 0;
    std::string label;
    ControlBlock block;
    std::atomic<bool> running{true};
    std::thread thread;
};

namespace {
bool ctl_cancelled(const CaliperJobControl* ctl) {
    return reinterpret_cast<const ControlBlock*>(ctl)->cancelled.load();
}
void ctl_progress(const CaliperJobControl* ctl, float frac, const char* msg) {
    auto* b = const_cast<ControlBlock*>(
        reinterpret_cast<const ControlBlock*>(ctl));
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    b->progress.store(frac);
    if (msg) {
        std::lock_guard<std::mutex> lk(b->msg_mutex);
        b->message = msg;
    }
}
} // namespace

uint64_t JobSystem::submit(std::string label, CaliperJobFn fn, void* user) {
    if (!fn) return 0;
    std::lock_guard<std::mutex> lk(mutex_);
    auto job = std::make_unique<Job>();
    job->id = next_id_++;
    job->label = std::move(label);
    job->block.ctl.struct_size = sizeof(CaliperJobControl);
    job->block.ctl.cancelled = &ctl_cancelled;
    job->block.ctl.progress = &ctl_progress;
    Job* raw = job.get();
    job->thread = std::thread([raw, fn, user] {
        fn(user, &raw->block.ctl);      // trusted code, unguarded (§15)
        raw->running.store(false);
    });
    jobs_.push_back(std::move(job));
    return raw->id;
}

JobSystem::Job* JobSystem::find(uint64_t id) const {
    for (auto& j : jobs_)
        if (j->id == id) return j.get();
    return nullptr;
}

void JobSystem::request_cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (Job* j = find(id)) j->block.cancelled.store(true);
}

bool JobSystem::is_running(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j && j->running.load();
}

float JobSystem::progress_of(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j ? j->block.progress.load() : 0.0f;
}

std::vector<JobSystem::JobView> JobSystem::views() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<JobView> out;
    out.reserve(jobs_.size());
    for (auto& j : jobs_) {
        JobView v;
        v.id = j->id;
        v.label = j->label;
        v.progress = j->block.progress.load();
        {
            std::lock_guard<std::mutex> mk(j->block.msg_mutex);
            v.message = j->block.message;
        }
        v.running = j->running.load();
        if (v.running) out.insert(out.begin(), v); else out.push_back(v);
    }
    return out;
}

void JobSystem::cancel_all_and_join() {
    std::vector<std::unique_ptr<Job>> taken;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& j : jobs_) j->block.cancelled.store(true);
        taken.swap(jobs_);
    }
    for (auto& j : taken)
        if (j->thread.joinable()) j->thread.join();
}

} // namespace caliper_host
```

- [ ] **Step 4: GREEN** — targeted `--test-case="jobs*"` then full ctest (all prior cases still green; the suite gains 6 cases).
- [ ] **Step 5: Commit** — `feat(host): JobSystem — thread-per-job with poll-cancel and progress (jobs.v1 core)`.

---

### Task A3: Device query (Metal, no torch) — TDD

**Files:** Create `src/host/device_query.h`, `src/host/device_query_apple.mm`, `src/host/device_query_stub.cpp`, `tests/test_device.cpp`; Modify root `CMakeLists.txt` (OBJCXX language, Metal framework, conditional source), `tests/CMakeLists.txt`.

**Interfaces — Produces:** `caliper_host::DeviceInfo{kind,index,name,free_memory_hint}` and `const DeviceInfo& device_info()` (detect-once, cached). Consumed by A4's service thunks and (read-only) by the tray/main.

- [ ] **Step 1: test first**

`tests/test_device.cpp`:
```cpp
#include <doctest/doctest.h>
#include "device_query.h"
using namespace caliper_host;

TEST_CASE("device: detection is stable and sane on this machine") {
    const DeviceInfo& a = device_info();
    const DeviceInfo& b = device_info();
    CHECK(&a == &b);                       // detect-once, cached
#ifdef __APPLE__
    // This repo's suite runs on Apple Silicon (environment-dependent by
    // design — the host's job is to detect THIS machine).
    CHECK(a.kind == CALIPER_DEV_METAL);
    CHECK_FALSE(a.name.empty());
    CHECK(a.free_memory_hint > 0);
#else
    CHECK(a.kind == CALIPER_DEV_CPU);
#endif
    CHECK(a.index == 0);
}
```

- [ ] **Step 2: RED** — missing `device_query.h`.
- [ ] **Step 3: implement**

`src/host/device_query.h`:
```cpp
#pragma once
#include <caliper/services/device_v1.h>
#include <cstdint>
#include <string>

namespace caliper_host {

struct DeviceInfo {
    CaliperDeviceKind kind = CALIPER_DEV_CPU;
    int32_t index = 0;
    std::string name = "CPU";
    uint64_t free_memory_hint = 0;   // bytes; 0 = unknown
};

// Detect-once, cached for the process lifetime. Never links an ML framework
// (D11): Metal is queried directly on Apple; CUDA detection arrives with
// Phase 4 hardware (until then non-Apple reports CPU).
const DeviceInfo& device_info();

} // namespace caliper_host
```

`src/host/device_query_apple.mm`:
```objc++
#include "device_query.h"
#import <Metal/Metal.h>

namespace caliper_host {

const DeviceInfo& device_info() {
    static const DeviceInfo info = [] {
        DeviceInfo d;
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (dev) {
                d.kind = CALIPER_DEV_METAL;
                d.index = 0;
                d.name = [[dev name] UTF8String];
                d.free_memory_hint = [dev recommendedMaxWorkingSetSize];
            }
        }
        return d;
    }();
    return info;
}

} // namespace caliper_host
```

`src/host/device_query_stub.cpp`:
```cpp
#include "device_query.h"
// Non-Apple fallback until Phase 4 brings CUDA detection (needs hardware/CI).
namespace caliper_host {
const DeviceInfo& device_info() {
    static const DeviceInfo info{};   // CPU defaults
    return info;
}
} // namespace caliper_host
```

Root `CMakeLists.txt` — after `project(...)`, add:
```cmake
if(APPLE)
    enable_language(OBJCXX)
endif()
```
In the `caliper_host_lib` sources, add conditionally:
```cmake
if(APPLE)
    target_sources(caliper_host_lib PRIVATE src/host/device_query_apple.mm)
    target_link_libraries(caliper_host_lib PUBLIC "-framework Metal" "-framework Foundation")
else()
    target_sources(caliper_host_lib PRIVATE src/host/device_query_stub.cpp)
endif()
```
`tests/CMakeLists.txt`: add `test_device.cpp`.

- [ ] **Step 4: GREEN** — `--test-case="device*"` then full ctest.
- [ ] **Step 5: Commit** — `feat(host): Metal device query — device.v1 core, no ML framework linked (D11)`.

---

### Task A4: Vend both services + typed sugar + fixture injection — TDD

**Files:** Modify `src/host/host_services.h` (expose `JobSystem& host_job_system()`), `src/host/host_services.cpp` (two new tables + registry entries), `sdk/include/caliper/caliper.hpp` (Jobs/Device wrappers), `sdk/testing/caliper/fixture_host.h` (`provide()` injection); Create `tests/test_sugar_services.cpp`; Modify `tests/CMakeLists.txt`.

**Interfaces — Produces:** `caliper_host::host_job_system()`; `services_get` now answers 4 ids; `caliper::Jobs{explicit Jobs(const Host&), operator bool, submit(label,fn,user)→id, request_cancel, is_running, progress_of}` and `caliper::Device{kind,index,name,free_memory_hint; static Device query(const Host&)}`; `FixtureHost::provide(const char* id, const void* table)`. Consumed by A5/A6.

- [ ] **Step 1: tests first** — `tests/test_sugar_services.cpp`:
```cpp
#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <string>

// Fake tables: prove the sugar wrappers route through get_service correctly.
namespace {
uint64_t last_submit_seen = 0;
uint64_t fake_submit(const char* label, CaliperJobFn, void*) {
    last_submit_seen = (label && std::string(label) == "train") ? 7u : 1u;
    return last_submit_seen;
}
void fake_cancel(uint64_t) {}
bool fake_running(uint64_t id) { return id == 7; }
float fake_progress(uint64_t) { return 0.25f; }
const CaliperJobsV1 kFakeJobs = {sizeof(CaliperJobsV1), &fake_submit,
                                 &fake_cancel, &fake_running, &fake_progress};

CaliperDeviceKind fake_kind(void) { return CALIPER_DEV_METAL; }
int32_t fake_index(void) { return 0; }
const char* fake_name(void) { return "FakeGPU"; }
uint64_t fake_hint(void) { return 42; }
const CaliperDeviceV1 kFakeDev = {sizeof(CaliperDeviceV1), &fake_kind,
                                  &fake_index, &fake_name, &fake_hint};
} // namespace

TEST_CASE("sugar: Jobs wrapper routes through the service table") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_JOBS_V1, &kFakeJobs);
    caliper::Host host(fx.host());
    caliper::Jobs jobs(host);
    REQUIRE(static_cast<bool>(jobs));
    CHECK(jobs.submit("train", nullptr, nullptr) == 7);
    CHECK(jobs.is_running(7));
    CHECK(jobs.progress_of(7) == doctest::Approx(0.25f));
}

TEST_CASE("sugar: Jobs wrapper is falsy without the service") {
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Jobs jobs(host);
    CHECK_FALSE(static_cast<bool>(jobs));
    CHECK(jobs.submit("x", nullptr, nullptr) == 0);   // inert, not UB
}

TEST_CASE("sugar: Device::query snapshots the table") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_DEVICE_V1, &kFakeDev);
    caliper::Host host(fx.host());
    auto dev = caliper::Device::query(host);
    CHECK(dev.kind == CALIPER_DEV_METAL);
    CHECK(std::string(dev.name) == "FakeGPU");
    CHECK(dev.free_memory_hint == 42);
}

TEST_CASE("sugar: Device::query defaults to CPU without the service") {
    caliper::testing::FixtureHost fx;
    auto dev = caliper::Device::query(caliper::Host(fx.host()));
    CHECK(dev.kind == CALIPER_DEV_CPU);
}
```

- [ ] **Step 2: RED** — `provide` and the wrappers don't exist.
- [ ] **Step 3: implement**

`fixture_host.h` — add below the existing members:
```cpp
    // Inject any service table (e.g. a fake jobs.v1) for wrapper tests.
    void provide(const char* id, const void* table) {
        provided_.emplace_back(id, table);
    }
```
extend `get_service_thunk` to scan `active_->provided_` after the log check, and add the member `std::vector<std::pair<std::string, const void*>> provided_;`.

`caliper.hpp` — add after the `Host` class (inside `namespace caliper`):
```cpp
#include <caliper/services/jobs_v1.h>     // (with the other service includes)
#include <caliper/services/device_v1.h>

class Jobs {
public:
    Jobs() = default;
    explicit Jobs(const Host& host)
        : t_(static_cast<const CaliperJobsV1*>(host.service(CALIPER_JOBS_V1))) {}
    explicit operator bool() const { return t_ && t_->submit; }
    uint64_t submit(const char* label, CaliperJobFn fn, void* user) const {
        return (t_ && t_->submit) ? t_->submit(label, fn, user) : 0;
    }
    void request_cancel(uint64_t id) const {
        if (t_ && t_->request_cancel) t_->request_cancel(id);
    }
    bool is_running(uint64_t id) const {
        return (t_ && t_->is_running) ? t_->is_running(id) : false;
    }
    float progress_of(uint64_t id) const {
        return (t_ && t_->progress_of) ? t_->progress_of(id) : 0.0f;
    }
private:
    const CaliperJobsV1* t_ = nullptr;
};

struct Device {
    CaliperDeviceKind kind = CALIPER_DEV_CPU;
    int32_t index = 0;
    const char* name = "CPU";              // host-owned string
    uint64_t free_memory_hint = 0;
    static Device query(const Host& host) {
        Device d;
        auto* t = static_cast<const CaliperDeviceV1*>(
            host.service(CALIPER_DEVICE_V1));
        if (t && t->kind) {
            d.kind = t->kind();
            d.index = t->index ? t->index() : 0;
            d.name = t->name ? t->name() : "";
            d.free_memory_hint = t->free_memory_hint ? t->free_memory_hint() : 0;
        }
        return d;
    }
};
```

`host_services.h`: add forward decl + accessor `class JobSystem; JobSystem& host_job_system();`
`host_services.cpp`: add includes (`job_system.h`, `device_query.h`, the two service headers), a file-static `JobSystem g_jobs;`, `JobSystem& host_job_system() { return g_jobs; }`, the thunk functions:
```cpp
uint64_t jobs_submit(const char* label, CaliperJobFn fn, void* user) {
    return g_jobs.submit(label ? label : "(job)", fn, user);
}
void jobs_cancel(uint64_t id)   { g_jobs.request_cancel(id); }
bool jobs_running(uint64_t id)  { return g_jobs.is_running(id); }
float jobs_progress(uint64_t id){ return g_jobs.progress_of(id); }
const CaliperJobsV1 kJobs = {sizeof(CaliperJobsV1), &jobs_submit, &jobs_cancel,
                             &jobs_running, &jobs_progress};

CaliperDeviceKind dev_kind(void) { return device_info().kind; }
int32_t dev_index(void)          { return device_info().index; }
const char* dev_name(void)       { return device_info().name.c_str(); }
uint64_t dev_hint(void)          { return device_info().free_memory_hint; }
const CaliperDeviceV1 kDevice = {sizeof(CaliperDeviceV1), &dev_kind, &dev_index,
                                 &dev_name, &dev_hint};
```
extend `services_get` with the two ids and `kIds` to all four.

- [ ] **Step 4: GREEN** — targeted + full ctest (loader tests keep passing: `service_ids()` growing to 4 only widens `HostCaps`).
- [ ] **Step 5: Commit** — `feat(host): vend jobs.v1 + device.v1; typed sugar wrappers; fixture-host service injection`.

---

### Task A5: Jobs tray (main.cpp glue)

**Files:** Modify `src/main.cpp` only. No unit tests (UI glue per Global Constraints); verified by build + the A6 demo checklist.

- [ ] **Step 1:** In `run()`, after the page-branch block and before `ImGui::Render()`, render the tray on BOTH pages:
```cpp
            // Jobs tray (§7.5): visible on every page while jobs exist.
            {
                auto views = caliper_host::host_job_system().views();
                if (!views.empty()) {
                    ImGuiIO& tio = ImGui::GetIO();
                    ImGui::SetNextWindowPos(
                        {tio.DisplaySize.x - 330.0f, tio.DisplaySize.y - 10.0f},
                        ImGuiCond_Always, {0.0f, 1.0f});
                    ImGui::SetNextWindowSize({320.0f, 0.0f});
                    ImGui::Begin("Jobs", nullptr,
                                 ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoCollapse);
                    for (auto& v : views) {
                        ImGui::PushID((int)v.id);
                        ImGui::Text("%s", v.label.c_str());
                        ImGui::ProgressBar(v.progress, {-60.0f, 0.0f},
                                           v.message.empty() ? nullptr
                                                             : v.message.c_str());
                        if (v.running) {
                            ImGui::SameLine();
                            if (ImGui::SmallButton("cancel"))
                                caliper_host::host_job_system().request_cancel(v.id);
                        }
                        ImGui::PopID();
                    }
                    ImGui::End();
                }
            }
```
plus `#include "host/host_services.h"` is already present and add `#include "host/job_system.h"`.

- [ ] **Step 2:** Build green; full ctest green; headless app check (launch ~10s, crash-free — the tray is invisible with no jobs, which is correct).
- [ ] **Step 3: Commit** — `feat(host): jobs tray overlay (label, progress, cancel)`.

---

### Task A6: `examples/ml_scope` — the ML exemplar's birth (+ docs)

**Files:** Create `examples/ml_scope/CMakeLists.txt`, `examples/ml_scope/ml_scope.cpp`, `examples/ml_scope/ml_scope.caliper.toml`; Modify root `CMakeLists.txt` (examples block: add_subdirectory + `add_dependencies` + `_active_applet_libs` registration for `ml_scope`); Docs: fill `## Semantics` on `jobs-v1.md` + `device-v1.md`, add jobs paragraph to `trust-model.md`, link ml_scope from `tutorials/first-applet.md` as "the ML exemplar".

**Interfaces — Consumes:** everything above. The applet is the acceptance vehicle: if writing it fights the API, the API is wrong — report DONE_WITH_CONCERNS with specifics rather than working around.

- [ ] **Step 1:** `examples/ml_scope/CMakeLists.txt`:
```cmake
# The ML exemplar (PLATFORM.md §17 Phase 2). Links the in-tree libtorch the
# same way repnet_demo does — runtime packs replace this at Phase 4.
add_library(ml_scope_applet SHARED ml_scope.cpp)
target_link_libraries(ml_scope_applet PRIVATE
    caliper::sdk caliper::ui_stack "${TORCH_LIBRARIES}")
target_compile_definitions(ml_scope_applet PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(ml_scope_applet PROPERTIES
    OUTPUT_NAME ml_scope
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)
    set_target_properties(ml_scope_applet PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_custom_command(TARGET ml_scope_applet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ml_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/ml_scope.caliper.toml)
```

- [ ] **Step 2:** `examples/ml_scope/ml_scope.caliper.toml`:
```toml
[applet]
id      = "dev.caliper.ml-scope"
name    = "MLScope"
version = "0.1.0"
summary = "ML exemplar: trains a tiny MLP off the frame thread via caliper.jobs.v1, device-negotiated, with live loss. Weight visualization arrives with tensor_bridge (Phase 2C)."
tag     = "ML"

[compat]
abi_epoch = 2
min_host  = "0.6.0"

[services]
required = ["caliper.ui.v1", "caliper.log.v1", "caliper.jobs.v1", "caliper.device.v1"]
optional = ["caliper.metrics.v1", "caliper.tensor_bridge.v1"]
```

- [ ] **Step 3:** `examples/ml_scope/ml_scope.cpp` — complete file:
```cpp
// ============================================================================
// MLScope — the ML exemplar (PLATFORM.md §17 Phase 2, step 1 of the ratified
// sequencing). Shows the idioms of ML on the platform:
//   ML-EXEMPLAR 1 — never train on the frame thread: submit to caliper.jobs.v1
//     and poll cancelled() in the epoch loop (cooperative cancel).
//   ML-EXEMPLAR 2 — the host picks the device (caliper.device.v1); the applet
//     maps the KIND to its framework: METAL -> torch::kMPS here.
//   ML-EXEMPLAR 3 — publish training state to the UI under a mutex; the frame
//     reads a copy. (repnet's snapshot pattern, minimal form.)
//   ML-EXEMPLAR 4 — deliberately NO weight-matrix visualization yet: that is
//     tensor_bridge.v1's job (Plan 2C). A CPU-staged copy here would teach the
//     exact pattern the platform exists to delete.
// ============================================================================
#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kEpochs = 300;
constexpr int kN = 512;   // two-moons points

// Synthetic two-moons, generated on the training device.
std::pair<torch::Tensor, torch::Tensor> make_moons(torch::Device dev) {
    auto t = torch::rand({kN}) * M_PI;
    auto x0 = torch::stack({torch::cos(t), torch::sin(t)}, 1);
    auto x1 = torch::stack({1.0f - torch::cos(t), 0.5f - torch::sin(t)}, 1);
    auto X = torch::cat({x0, x1}, 0) + torch::randn({2 * kN, 2}) * 0.08f;
    auto y = torch::cat({torch::zeros({kN}), torch::ones({kN})}, 0)
                 .to(torch::kLong);
    return {X.to(dev), y.to(dev)};
}
} // namespace

class MLScope final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        jobs_ = caliper::Jobs(host);          // required -> present (manifest)
        device_ = caliper::Device::query(host);
        host.log_info("ml-scope: on_init");
        return true;
    }

    void on_frame(const caliper::Frame&) override {
        ImGui::SetNextWindowPos({60, 80}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({560, 420}, ImGuiCond_FirstUseEver);
        ImGui::Begin("MLScope");

        // ML-EXEMPLAR 2 — the negotiated device, and what torch calls it.
        ImGui::TextDisabled("device: %s (%s)  |  free mem hint: %.1f GB",
                            device_.name,
                            device_.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
                            : device_.kind == CALIPER_DEV_CUDA ? "CUDA"
                                                               : "CPU",
                            device_.free_memory_hint / 1073741824.0);

        const bool running = job_id_ != 0 && jobs_.is_running(job_id_);
        if (!running) {
            if (ImGui::Button("start training")) start_training();
        } else {
            if (ImGui::Button("cancel")) jobs_.request_cancel(job_id_);
            ImGui::SameLine();
            ImGui::ProgressBar(jobs_.progress_of(job_id_), {-1, 0});
        }

        // ML-EXEMPLAR 3 — read a copy of worker-published state.
        std::vector<float> loss;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss = loss_history_;
        }
        if (ImPlot::BeginPlot("loss", {-1, 260})) {
            ImPlot::SetupAxes("epoch", "NLL");
            if (!loss.empty())
                ImPlot::PlotLine("train", loss.data(), (int)loss.size());
            ImPlot::EndPlot();
        }
        ImGui::TextWrapped("Weight-matrix visualization arrives with "
                           "caliper.tensor_bridge.v1 — GPU-resident, no CPU "
                           "staging. Watch this space (Plan 2C).");
        ImGui::End();
    }

    void on_cleanup() override {
        if (job_id_ != 0) {
            jobs_.request_cancel(job_id_);
            // ML-EXEMPLAR 1b — `user` (this object) must outlive the job
            // (jobs_v1.h contract): wait for the worker to exit BEFORE
            // destroy() frees us. Cancel is honored <=100 ms by tested
            // contract, so this bounded wait cannot hang teardown.
            for (int i = 0; i < 300 && jobs_.is_running(job_id_); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (host_) host_->log_info("ml-scope: on_cleanup");
    }

private:
    void start_training() {
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss_history_.clear();
        }
        // ML-EXEMPLAR 1 — static trampoline + this: the raw C job contract.
        job_id_ = jobs_.submit("ml_scope: train MLP", &MLScope::train_job, this);
        if (job_id_ == 0 && host_) host_->log_error("ml-scope: submit failed");
    }

    static void train_job(void* user, const CaliperJobControl* ctl) {
        auto* self = static_cast<MLScope*>(user);
        torch::Device dev = self->device_.kind == CALIPER_DEV_METAL &&
                                    torch::hasMPS()
                                ? torch::Device(torch::kMPS)
                                : torch::Device(torch::kCPU);
        torch::manual_seed(7);
        auto [X, y] = make_moons(dev);
        auto model = torch::nn::Sequential(
            torch::nn::Linear(2, 16), torch::nn::ReLU(),
            torch::nn::Linear(16, 16), torch::nn::ReLU(),
            torch::nn::Linear(16, 2));
        model->to(dev);
        torch::optim::Adam opt(model->parameters(),
                               torch::optim::AdamOptions(1e-2));
        for (int epoch = 0; epoch < kEpochs; epoch++) {
            if (ctl->cancelled(ctl)) break;         // ML-EXEMPLAR 1
            opt.zero_grad();
            auto out = torch::log_softmax(model->forward(X), 1);
            auto loss = torch::nll_loss(out, y);
            loss.backward();
            opt.step();
            float l = loss.item<float>();
            {
                std::lock_guard<std::mutex> lk(self->state_mutex_);
                self->loss_history_.push_back(l);
            }
            char msg[64];
            std::snprintf(msg, sizeof msg, "epoch %d/%d  loss %.4f", epoch + 1,
                          kEpochs, l);
            ctl->progress(ctl, (float)(epoch + 1) / kEpochs, msg);
        }
    }

    caliper::Host* host_ = nullptr;
    caliper::Jobs jobs_;
    caliper::Device device_;
    uint64_t job_id_ = 0;
    std::mutex state_mutex_;
    std::vector<float> loss_history_;
};

CALIPER_APPLET(MLScope,
    .id       = "dev.caliper.ml-scope",
    .version  = "0.1.0",
    .name     = "MLScope",
    .summary  = "ML exemplar: trains a tiny MLP off the frame thread via "
                "caliper.jobs.v1, device-negotiated, with live loss. Weight "
                "visualization arrives with tensor_bridge (Phase 2C).",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
```

- [ ] **Step 4:** Root CMake examples block: add `add_subdirectory(examples/ml_scope)`, extend `add_dependencies(caliper ... ml_scope_applet)`, append `libml_scope` registration to `_active_applet_libs` (same pattern as hello/signal_scope — placement inside the existing `CALIPER_BUILD_EXAMPLES` block).

- [ ] **Step 5: verify** — build (ml_scope links torch; minutes); full ctest; `ls build/applets/` gains `libml_scope.dylib` + manifest; `nm` exports descriptor; reconfigure survival (stale-cleanup registration); headless app alive ~10s. Card count (6) + train/cancel/tray behavior = human demo checklist:
  1. MLScope card appears; launch → device line reads your GPU name + "METAL->torch MPS".
  2. Start training → loss curve draws live, jobs tray shows label + progress + message, frame stays fluid (nothing blocks).
  3. Cancel mid-run → job ends promptly, tray row completes, applet stays live.
  4. ESC out mid-training → returns to landing; tray still shows the job briefly (cancel requested by cleanup); relaunch works.

- [ ] **Step 6: docs (same commit)** — `jobs-v1.md` + `device-v1.md` `## Semantics` sections (threading honesty verbatim from the header comments; cancel ≤100 ms contract; METAL naming rationale; `user`-outlives-job rule); `trust-model.md` gains a "Jobs run unguarded" paragraph tying to the crash-guard preconditions; `tutorials/first-applet.md` links ml_scope as "the ML exemplar". `mkdocs build --strict` exit 0.

- [ ] **Step 7: Commit** — `feat(examples): MLScope — jobs.v1 + device.v1 exemplar, MLP training off the frame thread`. Then merge: `git checkout main && git merge --no-ff platform/phase-2a -m "Phase 2A: jobs.v1 + device.v1 + MLScope (PLATFORM.md §17 Phase 2, step 1)"` (+ trailer).

---

## Exit Criteria (Plan 2A)

| Requirement | Proof |
|---|---|
| jobs.v1/device.v1 headers frozen, C-clean, struct_size-first | A1 static_asserts + abi_c_check.c |
| Cancel honored ≤ 100 ms (§16) | A2 timed contract test |
| Worker lifecycle sound (dtor joins; concurrent jobs) | A2 tests |
| Device detected without torch (D11) | A3 (Metal-only link) + test on this machine |
| Sugar wrappers degrade gracefully absent the service | A4 fixture tests |
| Host vends 4 services; negotiation set grows | A4 + existing loader tests stay green |
| Training off the frame thread, live loss, tray, cancel | A6 human demo checklist |
| Docs ride along; strict build | A1 + A6 |

## Spec Deviations (deliberate)

1. `CALIPER_DEV_METAL` replaces the spec sketch's `CALIPER_DEV_MPS` — memory-domain naming (spec §7.2 amended in the same session; MLX/ggml rationale).
2. Sugar keeps the **raw `CaliperJobFn`** (static trampoline + `this`), no `std::function` convenience — YAGNI, and the C pattern is the teaching point; a closure helper can be additive later.
3. Device detection is Metal-or-CPU only; CUDA is Phase 4 (hardware). `index()` is fixed 0 until multi-GPU exists.
4. Jobs tray is untested glue (Global Constraints rule).
5. `progress_of` on a finished job returns its last reported value (not forced to 1.0) — the fn owns its progress semantics.

## Risks / Environment Notes

- `enable_language(OBJCXX)` triggers a compiler re-detection on next configure — expected one-time configure noise.
- ml_scope's first build links libtorch (~minutes). MPS torch ops for this MLP are all supported; if an op falls back with a console warning, that's torch's known MPS fallback chatter — note it, don't chase it.
- The A2 cancel-latency test is timing-based; 100 ms against a 1 ms poll loop has ~100× margin, safe on any loaded machine.
- Threads + doctest: all worker threads are joined inside each test (dtor semantics) — no cross-test thread leakage.

