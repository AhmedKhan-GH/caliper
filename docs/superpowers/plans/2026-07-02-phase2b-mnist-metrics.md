# Phase 2B — MNIST Exemplar + `metrics.v1` + Runs Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Upgrade MLScope from the two-moons toy to **real MNIST CNN training** (user directive: the ubiquitous benchmark), then ship `metrics.v1` (run/tag/step store on embedded DuckDB) with a host **Runs dashboard**, using MNIST's loss/accuracy streams as the acceptance test. Step 2 of the ratified Phase-2 sequencing.

**Architecture:** MNIST arrives at first run *inside the job* (download → gunzip → cache in the applet's data dir — heavy IO is job work too, cancellable via curl's progress callback), parsed by a tiny TDD'd IDX reader. `metrics.v1` follows the frozen-table pattern; its host implementation (`MetricsStore`) embeds DuckDB in the host per spec §11 (applets never see DuckDB types — the service is the boundary). The dashboard is host UI over tested store queries. `caliper/tensor.h` (§7.2) ships now because the frozen metrics table's `image()` needs `CaliperTensor` — the type arrives early, the bridge that animates it stays Plan 2C.

**Tech Stack:** as 2A; plus system libcurl + zlib (ml_scope only), `duckdb_static` into `caliper_host_lib` (host-internal, §11), MNIST from the standard mirror `https://ossci-datasets.s3.amazonaws.com/mnist/`.

## Global Constraints

- All Phase-2A plan constraints carry over verbatim (TDD, green tasks, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, docs-ride-along, explicit-path staging, build/, no merge by agents — orchestrator merges).
- **Branch:** `platform/phase-2b` from `main`.
- **Ids (fixed):** `"caliper.metrics.v1"`; MNIST files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` (+`.gz` on the wire); IDX magics: images `2051`, labels `2049`.
- **§16 contract:** metrics store writes 10k scalars and queries them back ordered — a tested guarantee.
- **Threading:** `metrics.v1` is callable from job threads (MLScope will) — the store serializes internally (mutex over one DuckDB connection, v1).
- **Exemplar rules:** still NO CPU-staged weight/kernel visualization (that's 2C); download/caching happens inside the job, never on the frame thread; offline behavior is a clear status message, never a hang or crash.
- **Do not touch:** `applets/*` internals, other examples, `abi.h`, shipped service headers, `third_party/`, `cmake-build-debug/`.

## File Map

```
examples/ml_scope/mnist_idx.h            B1 (header-only parser: gunzip + IDX, TDD'd)
examples/ml_scope/ml_scope.cpp           B1 rewrite (MNIST CNN), B6 (metrics streaming)
examples/ml_scope/CMakeLists.txt         B1 (+CURL/ZLIB)
tests/test_mnist_idx.cpp                 B1
sdk/include/caliper/tensor.h             B2 (frozen §7.2)
sdk/include/caliper/services/metrics_v1.h B2 (frozen §7.6)
tests/test_abi.cpp, tests/abi_c_check.c  B2 (extended)
src/host/metrics_store.h/.cpp            B3 (DuckDB-backed, TDD)
tests/test_metrics_store.cpp             B3
src/host/host_services.h/.cpp            B4 (vend metrics.v1; open store at init)
sdk/include/caliper/caliper.hpp          B4 (caliper::Metrics wrapper)
tests/test_sugar_services.cpp            B4 (extended)
src/host/runs_dashboard.h/.cpp           B5 (UI over tested queries)
src/main.cpp                             B5 (menu toggle + render call)
CMakeLists.txt                           B1 (nothing), B3 (duckdb into host_lib), B5 (dashboard src)
docs/wiki/reference/services/metrics-v1.md  B2 stub → B6 semantics; + nav (mkdocs.yml B2)
docs/wiki/reference/tensor.md            B2 (embed tensor.h) + nav
docs/wiki/tutorials/first-applet.md      B6 (MNIST mention)
```

---

### Task B1: MLScope → MNIST CNN

**Files:** Create `examples/ml_scope/mnist_idx.h`, `tests/test_mnist_idx.cpp`; Rewrite `examples/ml_scope/ml_scope.cpp`; Modify `examples/ml_scope/CMakeLists.txt`, `tests/CMakeLists.txt`.

**Interfaces — Produces:** `mnist_idx::gunzip(bytes)→optional<vector<uint8_t>>`, `parse_images(bytes)→optional<Images{n,rows,cols,pixels}>`, `parse_labels(bytes)→optional<vector<uint8_t>>`. MLScope trains a CNN (conv 1→8→16, fc 400→10) on MNIST with per-batch loss + per-epoch test accuracy, all inside the job. B6 consumes the training loop's recording points.

- [ ] **Step 1: parser tests first** — `tests/test_mnist_idx.cpp`:
```cpp
#include <doctest/doctest.h>
#include "mnist_idx.h"
#include <zlib.h>
#include <cstring>
using namespace mnist_idx;

namespace {
void be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
std::vector<uint8_t> gzip_compress(const std::vector<uint8_t>& in) {
    z_stream s{};
    deflateInit2(&s, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8,
                 Z_DEFAULT_STRATEGY);                     // gzip wrapper
    std::vector<uint8_t> out(deflateBound(&s, in.size()));
    s.next_in = const_cast<uint8_t*>(in.data());
    s.avail_in = (uInt)in.size();
    s.next_out = out.data();
    s.avail_out = (uInt)out.size();
    deflate(&s, Z_FINISH);
    out.resize(out.size() - s.avail_out);
    deflateEnd(&s);
    return out;
}
} // namespace

TEST_CASE("mnist_idx: images parse (magic, dims, pixel order)") {
    std::vector<uint8_t> raw;
    be32(raw, 2051); be32(raw, 2); be32(raw, 2); be32(raw, 3);  // 2 imgs, 2x3
    for (int i = 0; i < 12; i++) raw.push_back((uint8_t)(i * 10));
    auto img = parse_images(raw);
    REQUIRE(img.has_value());
    CHECK(img->n == 2); CHECK(img->rows == 2); CHECK(img->cols == 3);
    REQUIRE(img->pixels.size() == 12);
    CHECK(img->pixels[0] == 0); CHECK(img->pixels[11] == 110);
}

TEST_CASE("mnist_idx: labels parse; wrong magic refused") {
    std::vector<uint8_t> raw;
    be32(raw, 2049); be32(raw, 3);
    raw.push_back(7); raw.push_back(0); raw.push_back(9);
    auto lab = parse_labels(raw);
    REQUIRE(lab.has_value());
    CHECK((*lab)[0] == 7); CHECK((*lab)[2] == 9);
    // images magic fed to labels parser (and vice versa) must refuse
    std::vector<uint8_t> wrong; be32(wrong, 2051); be32(wrong, 1);
    CHECK_FALSE(parse_labels(wrong).has_value());
}

TEST_CASE("mnist_idx: truncated payload refused") {
    std::vector<uint8_t> raw;
    be32(raw, 2051); be32(raw, 2); be32(raw, 28); be32(raw, 28);  // promises 1568
    raw.push_back(1);                                              // delivers 1
    CHECK_FALSE(parse_images(raw).has_value());
}

TEST_CASE("mnist_idx: gunzip round-trips") {
    std::vector<uint8_t> original;
    for (int i = 0; i < 5000; i++) original.push_back((uint8_t)(i % 251));
    auto un = gunzip(gzip_compress(original));
    REQUIRE(un.has_value());
    CHECK(*un == original);
    CHECK_FALSE(gunzip({0x00, 0x01, 0x02}).has_value());  // not gzip
}
```
`tests/CMakeLists.txt`: add `test_mnist_idx.cpp`; add `target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/examples/ml_scope)`; `find_package(ZLIB REQUIRED)` + link `ZLIB::ZLIB`.

- [ ] **Step 2: RED** — missing `mnist_idx.h`.
- [ ] **Step 3: implement `examples/ml_scope/mnist_idx.h`**:
```cpp
#pragma once
// Minimal MNIST IDX reader (+gunzip). Header-only so the test suite can
// exercise it without linking the applet. IDX: big-endian magic + dims,
// then raw bytes. Images magic 2051, labels magic 2049.
#include <zlib.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace mnist_idx {

inline std::optional<std::vector<uint8_t>> gunzip(
    const std::vector<uint8_t>& gz) {
    z_stream s{};
    if (inflateInit2(&s, 15 + 32) != Z_OK) return std::nullopt;  // auto gzip/zlib
    std::vector<uint8_t> out;
    out.resize(gz.size() * 4 + 1024);
    s.next_in = const_cast<uint8_t*>(gz.data());
    s.avail_in = (uInt)gz.size();
    size_t written = 0;
    int rc;
    do {
        if (written == out.size()) out.resize(out.size() * 2);
        s.next_out = out.data() + written;
        s.avail_out = (uInt)(out.size() - written);
        rc = inflate(&s, Z_NO_FLUSH);
        written = out.size() - s.avail_out;
        if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&s); return std::nullopt; }
    } while (rc != Z_STREAM_END);
    inflateEnd(&s);
    out.resize(written);
    return out;
}

namespace detail {
inline std::optional<uint32_t> be32_at(const std::vector<uint8_t>& b, size_t i) {
    if (i + 4 > b.size()) return std::nullopt;
    return ((uint32_t)b[i] << 24) | ((uint32_t)b[i + 1] << 16) |
           ((uint32_t)b[i + 2] << 8) | (uint32_t)b[i + 3];
}
} // namespace detail

struct Images {
    int n = 0, rows = 0, cols = 0;
    std::vector<uint8_t> pixels;   // n*rows*cols, row-major
};

inline std::optional<Images> parse_images(const std::vector<uint8_t>& raw) {
    auto magic = detail::be32_at(raw, 0);
    if (!magic || *magic != 2051) return std::nullopt;
    auto n = detail::be32_at(raw, 4), r = detail::be32_at(raw, 8),
         c = detail::be32_at(raw, 12);
    if (!n || !r || !c) return std::nullopt;
    size_t need = (size_t)*n * *r * *c;
    if (raw.size() < 16 + need) return std::nullopt;
    Images out;
    out.n = (int)*n; out.rows = (int)*r; out.cols = (int)*c;
    out.pixels.assign(raw.begin() + 16, raw.begin() + 16 + need);
    return out;
}

inline std::optional<std::vector<uint8_t>> parse_labels(
    const std::vector<uint8_t>& raw) {
    auto magic = detail::be32_at(raw, 0);
    if (!magic || *magic != 2049) return std::nullopt;
    auto n = detail::be32_at(raw, 4);
    if (!n || raw.size() < 8 + *n) return std::nullopt;
    return std::vector<uint8_t>(raw.begin() + 8, raw.begin() + 8 + *n);
}

} // namespace mnist_idx
```
- [ ] **Step 4: GREEN on parser tests** (`--test-case="mnist*"`), full ctest.
- [ ] **Step 5: rewrite `ml_scope.cpp`** — keep the 2A skeleton (macro block, `on_cleanup` bounded-wait now `1000` iterations with the same comment plus "covers a cancel that lands mid-download", state mutex pattern, device line) and replace the data/model/UI middles:
  - **Acquisition (inside the job; ML-EXEMPLAR 5 comment: "heavy data is job work too — download once into data_dir, cache forever, cancellable"):** for each of the four MNIST files: if `<data_dir>/<name>` exists, skip; else curl `https://ossci-datasets.s3.amazonaws.com/mnist/<name>.gz` into memory (write-callback appends to a `std::vector<uint8_t>`; `CURLOPT_XFERINFOFUNCTION` returns 1 when `ctl->cancelled(ctl)` so cancel aborts the transfer; `CURLOPT_FOLLOWLOCATION`, `CURLOPT_FAILONERROR`), `mnist_idx::gunzip`, write raw bytes to `path + ".tmp"` then `std::rename` to the cache path (**atomic cache — an interrupted write must never leave a truncated file at the canonical name**; B1-review finding). A cached file that later fails to parse is deleted with status "cached MNIST file was corrupt — press start to re-download" (self-healing, never a permanent wedge). Any download failure → `ctl->progress(ctl, 0.f, "MNIST download failed (offline?) — press start to retry")`, log via a status string under the mutex, return from the job cleanly. `curl_global_init(CURL_GLOBAL_DEFAULT)` once in `on_init` / `curl_global_cleanup` in `on_cleanup` (lazy init from a worker thread is not thread-safe per libcurl docs; B1-review finding).
  - **Tensors:** parse cached files → train X `(60000,1,28,28)` float/255 on CPU, y long; test likewise; `.to(dev)` once (fits comfortably in unified memory; comment says so).
  - **Model:** `nn::Sequential(Conv2d(1,8,3), ReLU, MaxPool2d(2), Conv2d(8,16,3), ReLU, MaxPool2d(2), Flatten, Linear(400,10))`; Adam 1e-3; **3 epochs, batch 256** via `torch::randperm` indexing; per-batch: poll `cancelled`, `nll_loss(log_softmax(...))`, record loss point under mutex; per-epoch: eval test accuracy in 1000-image no_grad batches, record accuracy point; `ctl->progress(ctl, global_step/total_steps, "epoch e/3  loss L  test acc A%")`.
  - **UI:** status line (device + dataset state); Start/cancel + tray-mirroring progress bar as in 2A; TWO plots — "train loss" (per step) and "test accuracy %" (per epoch, 0–100 y-axis); the 2C deferral text now says *first-layer conv kernels* arrive with the bridge.
- [ ] **Step 6: CMake** — `examples/ml_scope/CMakeLists.txt`: add `find_package(CURL REQUIRED)` + `find_package(ZLIB REQUIRED)` and link `CURL::libcurl ZLIB::ZLIB` (system libs on macOS; exemplar-only deps, the host links neither).
- [ ] **Step 7: verify** — build; full ctest; headless app 10s; then the ONE machine-runnable end-to-end: launch the app headless is insufficient for training, so verification of convergence is the human demo (state it). Machine checks: dylib+manifest present, descriptor exported, reconfigure survival.
- [ ] **Step 8: Commit** — `feat(ml_scope): MNIST CNN training — cancellable download to data_dir, per-batch loss, per-epoch test accuracy`.

---

### Task B2: `caliper/tensor.h` + `metrics_v1.h` (frozen) + ABI tests

**Files:** Create `sdk/include/caliper/tensor.h`, `sdk/include/caliper/services/metrics_v1.h`; extend `tests/test_abi.cpp` + `tests/abi_c_check.c`; doc stubs `docs/wiki/reference/tensor.md` + `docs/wiki/reference/services/metrics-v1.md` (H1 + intro + `--8<--` embed + `*Semantics: written at Task B6.*`) + both in `mkdocs.yml` nav.

- [ ] **Step 1 (RED):** append to `test_abi.cpp`: standard-layout + `offsetof(...,struct_size)==0` for `CaliperTensor` and `CaliperMetricsV1`; `CALIPER_DT_F32==0`; id string test `"caliper.metrics.v1"`; include both headers in `abi_c_check.c`. Build → RED.
- [ ] **Step 2:** `sdk/include/caliper/tensor.h`:
```c
#pragma once
/* caliper/tensor.h — the tensor interchange TYPE (PLATFORM.md §7.2), not a
 * service. DLPack-aligned on purpose: torch/numpy/mlx interop is a cast away.
 * FROZEN once shipped. Reuses CaliperDeviceKind (memory-domain naming). */
#include <stdint.h>
#include <caliper/services/device_v1.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperDType {
    CALIPER_DT_F32 = 0, CALIPER_DT_F16 = 1, CALIPER_DT_BF16 = 2,
    CALIPER_DT_I64 = 3, CALIPER_DT_I32 = 4, CALIPER_DT_U8 = 5
} CaliperDType;

typedef struct CaliperTensor {
    uint32_t          struct_size;
    void*             data;            /* device or host pointer */
    CaliperDType      dtype;
    int32_t           ndim;            /* <= 8 */
    int64_t           shape[8];
    int64_t           strides[8];      /* in elements */
    CaliperDeviceKind device;
    int32_t           device_index;
    void*             stream;          /* cudaStream_t / MTLCommandQueue* / NULL */
} CaliperTensor;

#ifdef __cplusplus
}
#endif
```
`sdk/include/caliper/services/metrics_v1.h`:
```c
#pragma once
/* caliper.metrics.v1 — TensorBoard vocabulary (experiment/run/tag/step),
 * ImPlot immediacy (PLATFORM.md §7.6). IMMUTABLE once published. Callable
 * from applet job threads; the host serializes internally. image() accepts
 * CPU-resident HWC u8 tensors in v1 (GPU-resident paths arrive with the
 * tensor bridge). */
#include <stdint.h>
#include <caliper/tensor.h>

#define CALIPER_METRICS_V1 "caliper.metrics.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperMetricsV1 {
    uint32_t struct_size;
    uint64_t (*begin_run)(const char* experiment, const char* run_name); /* 0 = error */
    void     (*end_run)(uint64_t run);
    void     (*scalar)(uint64_t run, const char* tag, int64_t step, double value);
    void     (*histogram)(uint64_t run, const char* tag, int64_t step,
                          const float* values, int64_t count);
    void     (*image)(uint64_t run, const char* tag, int64_t step,
                      const CaliperTensor* hwc_u8);
    void     (*hparams_json)(uint64_t run, const char* json_utf8);
} CaliperMetricsV1;

#ifdef __cplusplus
}
#endif
```
- [ ] **Step 3:** GREEN (full ctest, C TU included); docs stubs + nav; strict mkdocs. Commit `feat(sdk): tensor.h interchange type + metrics.v1 service header (Phase 2B)`.

---

### Task B3: `MetricsStore` (DuckDB in the host) — TDD

**Files:** Create `src/host/metrics_store.h`, `src/host/metrics_store.cpp`, `tests/test_metrics_store.cpp`; Modify root `CMakeLists.txt` (host_lib sources + link `duckdb_static duckdb_generated_extension_loader parquet_extension core_functions_extension` — spec §11: DuckDB is a host implementation detail), `tests/CMakeLists.txt`.

**Interfaces — Produces:** `caliper_host::MetricsStore{open(path|":memory:"), begin_run(exp,name)→u64, end_run, scalar, histogram(run,tag,step,const float*,int64), image(run,tag,step,bytes,w,h,c), hparams_json, runs()→vector<RunInfo{id,experiment,name,done,hparams}>, scalar_tags(run), scalars(run,tag)→vector<pair<int64,double>> step-ordered}`. Internally: one connection + one mutex (thread-callable per the header's promise).

- [ ] **Step 1 (RED):** `tests/test_metrics_store.cpp` — cases: **§16 contract** (open `:memory:`, begin_run, write 10,000 scalars on one tag with shuffled step order, `scalars()` returns 10,000 pairs strictly step-ascending with matching values); two runs isolate (same tag, different values — each queries only its own); `scalar_tags` lists exactly the written tags; `hparams_json` round-trips via `runs()`; `end_run` flips `done`; unknown-run calls inert (no throw, empty queries); histogram blob survives (write 64 floats, count them back via a `histogram_count(run,tag)` helper or store-level query — implementer may expose a minimal `histograms(run,tag)` reader for the test's assertion). Threaded smoke: 4 threads × 500 scalars on distinct tags, total count correct.
- [ ] **Step 2:** implement — schema on `open`: `runs(id BIGINT, experiment VARCHAR, name VARCHAR, done BOOLEAN, hparams VARCHAR)`, `scalars(run BIGINT, tag VARCHAR, step BIGINT, value DOUBLE)`, `histograms(run BIGINT, tag VARCHAR, step BIGINT, count BIGINT, data BLOB)`, `images(run BIGINT, tag VARCHAR, step BIGINT, w INT, h INT, c INT, data BLOB)`; prepared statements under the mutex; `scalars()` = `... ORDER BY step`. All queries via the DuckDB C++ API; **no DuckDB type in the header** (pimpl or void* impl pointer).
- [ ] **Step 3:** GREEN targeted + full; commit `feat(host): MetricsStore — DuckDB-backed run/tag/step store (§16 contract tested)`.

---

### Task B4: Vend `metrics.v1` + `caliper::Metrics` sugar — TDD

**Files:** Modify `src/host/host_services.h/.cpp` (store instance opened at `services_init()` → `caliper::app_data_path("metrics.duckdb")`; expose `MetricsStore& host_metrics_store()`; thunks + registry entry; `kIds` → 5), `sdk/include/caliper/caliper.hpp` (`caliper::Metrics` wrapper: begin_run/end_run/scalar/histogram/image/hparams_json, falsy-inert like `Jobs`), `sdk/testing` untouched; extend `tests/test_sugar_services.cpp` (fake metrics table records calls; wrapper inert absent the service).

- [ ] TDD: extend sugar tests (RED) → wrapper + thunks (image thunk accepts only `dtype==CALIPER_DT_U8 && ndim==3 && device==CALIPER_DEV_CPU`, else logs + drops — documented) → GREEN full suite (loader green with 5 ids); strict mkdocs (caliper.hpp embed reflects). Commit `feat(host): vend metrics.v1; caliper::Metrics sugar`.

---

### Task B5: Runs dashboard (host UI glue)

**Files:** Create `src/host/runs_dashboard.h/.cpp` (`void render_runs_dashboard(MetricsStore&, bool* p_open)`); Modify `src/main.cpp` (menu-bar "Runs" toggle on BOTH pages — place next to the existing menu items; render call before the jobs tray), root `CMakeLists.txt` (add dashboard .cpp to the `caliper` executable sources).

- [ ] Layout: left pane — run list from `runs()` (`experiment/name`, ● while not done), selectable; right pane — hparams line, then one ImPlot per `scalar_tags(run)` plotting `scalars(run,tag)` with an EMA smoothing slider (0–0.99, default 0; smoothed line over raw faint line). Poll queries each frame only while the window is open (store is mutex-guarded; row counts here are trivial).
- [ ] No unit tests (glue over B3-tested queries — sanctioned). Build + full ctest + headless 10s. Commit `feat(host): Runs dashboard (run list, per-tag plots, EMA smoothing)`.

---

### Task B6: MLScope streams to `metrics.v1` + docs semantics (+ orchestrator merge)

**Files:** Modify `examples/ml_scope/ml_scope.cpp`; docs: `metrics-v1.md` semantics, `tensor.md` one-paragraph intro above the embed, `tutorials/first-applet.md` MNIST/metrics mention.

- [ ] MLScope: probe `caliper::Metrics metrics_{host}` in `on_init` (manifest already lists it optional); in the job: if truthy — `begin_run("mnist", "cnn")`, `hparams_json` (`{"lr":0.001,"batch":256,"epochs":3,"model":"conv8-16-fc"}`), `scalar("train/loss", global_step, l)` per batch, `scalar("test/accuracy", epoch, acc)` per epoch, `end_run` on every exit path (completion, cancel, download-failure). Status line gains `metrics: run #N` / `metrics: absent (ok)`. ML-EXEMPLAR 6 comment: probe-optional pattern now pays off — same binary works on hosts with and without metrics.
- [ ] Docs: metrics-v1.md `## Semantics` — the run/tag/step vocabulary, thread-callability, the CPU-u8 image limitation until the bridge, "every applet that logs a scalar inherits the dashboard"; tensor.md intro (a type not a service, DLPack alignment, who consumes it when). Strict mkdocs.
- [ ] Verify: build, full ctest, artifacts, headless 10s. **Human demo checklist:** launch MLScope → start → open **Runs** from the menu bar → the `mnist/cnn` run appears live, loss curve fills per batch, accuracy per epoch (expect >95% by epoch 3 on MNIST), smoothing slider works; SignalScope's status line now reads `metrics service: present`; cancel mid-run → run ends (`done` ●→gone) with partial curves preserved; relaunch + retrain → a second run listed, both comparable.
- [ ] Commit `feat(ml_scope): stream MNIST training to metrics.v1 — the Runs dashboard acceptance test`. NO merge — the orchestrator merges after gates.

---

## Exit Criteria (Plan 2B)

| Requirement | Proof |
|---|---|
| MNIST parsed correctly (magics, dims, truncation, gzip) | B1 parser tests |
| Download cancellable, cached, offline-graceful | B1 code path + human demo |
| tensor.h + metrics table frozen, C-clean | B2 asserts + C TU |
| 10k scalars back ordered (§16) | B3 contract test |
| Store thread-callable | B3 threaded smoke |
| Sugar inert without service; loader green with 5 ids | B4 |
| Dashboard renders live runs with smoothing | B6 human demo |
| MLScope = acceptance vehicle (real benchmark >95%) | B6 human demo |

## Spec Deviations (deliberate)

1. `tensor.h` ships in 2B (the frozen metrics table needs `CaliperTensor`); the bridge that animates it stays 2C.
2. `image()` v1 accepts CPU HWC u8 only — GPU-resident images arrive with the bridge; stated in the header.
3. MetricsStore v1 = one connection + mutex (not a writer queue) — §16 contract test is the perf floor; revisit if profiling ever says so.
4. MNIST via the ossci S3 mirror (canonical torchvision source; original LeCun URLs are unreliable).
5. curl/zlib are exemplar-only system deps; the host links neither.
6. Dashboard is untested UI glue over tested queries (standing rule).

## Risks / Environment Notes

- DuckDB linking into `caliper_host_lib` grows test-binary link time noticeably (one-time; already compiled objects).
- First MLScope run needs network (~11 MB once); offline first-run is a clear in-UI message, not an error state.
- MNIST train tensors on MPS: ~180 MB device-resident — trivial for unified memory; comment in code.
- `metrics.duckdb` schema is v1-fresh (no migration story needed yet; note for 2E/artifacts to reuse the pattern).

