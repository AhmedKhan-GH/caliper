// ============================================================================
// MLScope — the ML exemplar (PLATFORM.md §17 Phase 2, step 1 of the ratified
// sequencing). Shows the idioms of ML on the platform:
//   ML-EXEMPLAR 1 — never train on the frame thread: submit to caliper.jobs.v1
//     and poll cancelled() in the batch loop (cooperative cancel).
//   ML-EXEMPLAR 2 — the host picks the device (caliper.device.v1); the applet
//     maps the KIND to its framework: METAL -> torch::kMPS here.
//   ML-EXEMPLAR 3 — publish training state to the UI under a mutex; the frame
//     reads a copy. (repnet's snapshot pattern, minimal form.)
//   ML-EXEMPLAR 4 — weight visualization crosses caliper.tensor_bridge.v1, never
//     a CPU-staged copy hand-rolled in the applet (that would teach the exact
//     pattern the platform exists to delete). Delivered form: ML-EXEMPLAR 7.
//   ML-EXEMPLAR 5 — heavy data is job work too — download once into data_dir,
//     cache forever, cancellable. The frame thread never touches the network.
//   ML-EXEMPLAR 6 — probe-optional pays off: caliper.metrics.v1 is optional in
//     the manifest, so we probe it (caliper::Metrics, falsy-inert if absent)
//     and stream the run only when it is truthy. The SAME binary runs on a host
//     with metrics (loss/accuracy land in the Runs dashboard) and on one without
//     (training is unchanged; status says `metrics: absent (ok)`). Every applet
//     that logs a scalar this way inherits the Runs dashboard for free.
//   ML-EXEMPLAR 7 — GPU-resident visualization, the platform's reason to exist:
//     the worker snapshots conv1's 8 first-layer filters ON THE TRAINING DEVICE
//     every kEvalEvery batches (a tiny owned per-kernel clone, MPS drained once
//     so the frame never syncs) and publishes them under the mutex. The FRAME
//     thread — the only place tensor_bridge.v1 may be called (UI-thread-only
//     contract) — turns each into a live texture. On the Metal renderer the MPS
//     buffer is colormapped ON-GPU: zero CPU staging, the whole USP. On GL the
//     bridge rejects the device tensor; the applet relocates it to CPU and the
//     BRIDGE stages it — identical applet code, and no pixel work in the applet
//     either way (§6c). tensor_bridge.v1 is OPTIONAL: absent -> the grid says so
//     and training is unchanged (the EXEMPLAR 6 probe-optional pattern, again).
//   ML-EXEMPLAR 8 — real-data visualization: inputs and activations, not just
//     weights. Every eval tick the worker also snapshots ONE fixed probe digit
//     (t10k index 0) and forwards it through conv1 alone -> its 8 (26,26)
//     feature maps, plus the predicted/true labels — the SAME worker-snapshot
//     discipline as EXEMPLAR 7, just bigger owned tensors, MPS drained once. The
//     frame turns the digit into a VIRIDIS texture (fixed 0..1: create-once,
//     update-after) and each map into an RdBu texture. Activations rescale hard
//     as the net learns, so unlike the pinned-range kernels the maps use a fresh
//     symmetric per-snapshot range — and since v1 update_texture has no range
//     channel, a new range means fresh textures (release + recreate the 8, once
//     per eval tick, never per frame). Still zero applet pixel work on either
//     path: the bridge colormaps on-GPU (Metal) or CPU-stages (the shared C8
//     GL fallback). Bridge absent -> the panel says so; training is unchanged.
// ============================================================================
#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>   // ML-EXEMPLAR 7 — torch::Tensor -> CaliperTensor
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <curl/curl.h>

#include "mnist_idx.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kEpochs = 3;
constexpr int kBatch = 256;
// Evaluate test accuracy every kEvalEvery training batches (plus a step-0
// baseline and each epoch end). Per-epoch cadence hides the learning transient
// on fast-converging datasets like MNIST — 3 points that snap to ~98%.
constexpr int kEvalEvery = 50;
// ML-EXEMPLAR 8 — the probe digit is a FIXED test index so you watch the SAME
// digit's conv1 activations sharpen across the run (t10k[0]).
constexpr int64_t kProbeIdx = 0;

// The four IDX files MNIST ships as (host-side names in data_dir; `.gz` on the
// wire). Mirror on S3 — the classic yann.lecun.com host 403s from many nets.
const char* kFiles[4] = {
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"};
constexpr const char* kBaseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist/";

// libcurl write callback: append received bytes to a std::vector<uint8_t>.
size_t write_to_vec(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::vector<uint8_t>*>(userdata);
    size_t n = size * nmemb;
    buf->insert(buf->end(), (uint8_t*)ptr, (uint8_t*)ptr + n);
    return n;
}
} // namespace

class MLScope final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        jobs_ = caliper::Jobs(host);          // required -> present (manifest)
        device_ = caliper::Device::query(host);
        // ML-EXEMPLAR 6 — metrics is OPTIONAL: probe it here; the wrapper is
        // falsy-inert if the host does not vend it. No branching in on_init —
        // the job checks truthiness before it streams.
        metrics_ = caliper::Metrics(host);
        // ML-EXEMPLAR 7 — tensor_bridge.v1 is OPTIONAL too: probe it here (same
        // falsy-inert wrapper). The worker snapshots kernels only when it is
        // present; the frame renders the grid only when it is present.
        bridge_ = caliper::Bridge(host);
        // curl global init MUST happen once here on the frame thread: lazy init
        // from curl_easy_init on a worker thread is not thread-safe (libcurl docs).
        curl_global_init(CURL_GLOBAL_DEFAULT);
        host.log_info("ml-scope: on_init");
        return true;
    }

    void on_frame(const caliper::Frame&) override {
        ImGui::SetNextWindowPos({60, 80}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({620, 760}, ImGuiCond_FirstUseEver);
        ImGui::Begin("MLScope");

        // ML-EXEMPLAR 2 — the negotiated device, and what torch calls it.
        ImGui::TextDisabled("device: %s (%s)  |  free mem hint: %.1f GB",
                            device_.name,
                            device_.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
                            : device_.kind == CALIPER_DEV_CUDA ? "CUDA"
                                                               : "CPU",
                            device_.free_memory_hint / 1073741824.0);

        // ML-EXEMPLAR 3 — read a copy of worker-published state under the mutex.
        std::vector<float> loss, acc, acc_x;
        std::string status;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss = loss_history_;
            acc_x = acc_steps_;
            acc = acc_history_;
            status = status_line_;
        }
        ImGui::TextWrapped("%s", status.c_str());

        // ML-EXEMPLAR 6 — surface the optional service's state. When present and
        // a run is live, show its id; when absent, say so and reassure (ok).
        if (metrics_) {
            uint64_t run = run_id_.load();
            if (run != 0) ImGui::TextDisabled("metrics: run #%llu",
                                              (unsigned long long)run);
            else          ImGui::TextDisabled("metrics: present (open Runs)");
        } else {
            ImGui::TextDisabled("metrics: absent (ok)");
        }

        const bool running = job_id_ != 0 && jobs_.is_running(job_id_);
        if (!running) {
            if (ImGui::Button("start training")) start_training();
        } else {
            if (ImGui::Button("cancel")) jobs_.request_cancel(job_id_);
            ImGui::SameLine();
            ImGui::ProgressBar(jobs_.progress_of(job_id_), {-1, 0});
        }

        if (ImPlot::BeginPlot("train loss", {-1, 200})) {
            ImPlot::SetupAxes("step", "NLL");
            if (!loss.empty())
                ImPlot::PlotLine("loss", loss.data(), (int)loss.size());
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("test accuracy %", {-1, 200})) {
            // x-axis is the global step (same domain as the loss plot), so the
            // step-0 baseline and mid-epoch samples trace the true learning ramp.
            ImPlot::SetupAxes("step", "acc %");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImPlotCond_Always);
            if (!acc.empty())
                ImPlot::PlotLine("acc", acc_x.data(), acc.data(),
                                 (int)acc.size());
            ImPlot::EndPlot();
        }
        ImGui::Separator();
        render_kernels();   // ML-EXEMPLAR 7 — the live conv1 filter grid
        ImGui::Separator();
        render_probe();     // ML-EXEMPLAR 8 — the live probe digit + feature maps
        ImGui::End();
    }

    void on_cleanup() override {
        if (job_id_ != 0) {
            jobs_.request_cancel(job_id_);
            // ML-EXEMPLAR 1b — `user` (this object) must outlive the job
            // (jobs_v1.h contract): wait for the worker to exit BEFORE
            // destroy() frees us. Cancel is honored <=100 ms by tested
            // contract; the 1000 ms ceiling also covers a cancel that lands
            // mid-download (curl's xferinfo poll aborts the transfer, but the
            // socket teardown adds slack), so this bounded wait cannot hang.
            for (int i = 0; i < 1000 && jobs_.is_running(job_id_); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // ML-EXEMPLAR 7 — the kernel textures are frame-thread-owned. Release
        // them AFTER the job wait above (the worker never touched the bridge, and
        // the frame loop is stopped by the time on_cleanup runs, so nothing races
        // this) and BEFORE the host tears the renderer down.
        for (auto id : kernel_tex_)
            if (id) bridge_.release_texture(id);
        kernel_tex_.clear();
        // ML-EXEMPLAR 8 — the probe digit + feature-map textures are frame-thread
        // owned too; release them in the same window (after the job wait, before
        // renderer teardown).
        if (probe_digit_tex_) bridge_.release_texture(probe_digit_tex_);
        probe_digit_tex_ = 0;
        for (auto id : probe_map_tex_)
            if (id) bridge_.release_texture(id);
        probe_map_tex_.clear();
        // Pairs with the on_init curl_global_init; only safe once the worker
        // (the sole curl user) has exited, which the bounded wait above ensures.
        curl_global_cleanup();
        if (host_) host_->log_info("ml-scope: on_cleanup");
    }

private:
    void set_status(const std::string& s) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        status_line_ = s;
    }

    // ML-EXEMPLAR 7 — the payoff, drawn on the FRAME thread (tensor_bridge.v1 is
    // UI-thread-only). Read the worker's latest snapshot under the mutex; the
    // vector copy bumps each torch::Tensor's refcount, so the storage the bridge
    // reads stays alive across the (synchronous) upload even if the worker
    // publishes the next snapshot meanwhile. Ownership chain: worker owns the
    // clones in kernel_snap_; this copy co-owns them for the duration of the
    // frame; the CaliperTensor descriptors below point into that live storage.
    void render_kernels() {
        if (!bridge_) {
            ImGui::TextDisabled(
                "kernels: tensor_bridge.v1 absent (ok) — grid needs it");
            return;
        }
        std::vector<torch::Tensor> ks;
        uint64_t gen = 0; float wmax = 0.f; bool on_dev = false;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            gen = kernel_gen_;
            if (gen != 0) {
                ks = kernel_snap_;            // refcount bump, keeps storage alive
                wmax = kernel_wmax_;
                on_dev = kernel_on_device_;
            }
        }
        if (gen == 0) {
            ImGui::TextDisabled(
                "kernels: start training to watch conv1 sharpen from noise");
            return;
        }
        if (gen != kernel_tex_gen_) {         // new snapshot -> (re)upload
            upload_kernels(ks, wmax, on_dev);
            kernel_tex_gen_ = gen;
        }
        // The RdBu range is fixed at the first snapshot: v1 update_texture has no
        // range channel (the frozen ABI is the point — you live within it). The
        // filters still visibly sharpen; only the color scale is pinned.
        ImGui::Text("conv1 filters (3x3, RdBu +/-%.3f, range set at first "
                    "snapshot; weights live)", kernel_tex_range_);
        for (size_t k = 0; k < kernel_tex_.size(); ++k) {
            if (k % 4 != 0) ImGui::SameLine();
            // ~16x nearest-sampled upscale (host sampler); a 3x3 filter at 48px.
            ImGui::Image(caliper::Bridge::imtex(kernel_tex_[k]),
                         ImVec2(48, 48));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("kernel %zu / 8  (3x3 -> 48px, nearest)", k);
        }
        ImGui::TextDisabled("kernels: %s",
            (kernel_tex_on_device_ && !stage_cpu_)
                ? "GPU-resident (Metal, zero CPU staging)"
                : "CPU-staged (GL fallback)");
    }

    // ML-EXEMPLAR 8 — the real-data payoff, drawn on the FRAME thread. Same
    // read-a-copy-under-the-mutex discipline as render_kernels: the vector copy
    // refcount-bumps the worker's owned clones so their storage outlives the
    // (synchronous) uploads even if the worker publishes the next snapshot mid-
    // frame.
    void render_probe() {
        if (!bridge_) {
            ImGui::TextDisabled(
                "probe: tensor_bridge.v1 absent (ok) — panel needs it");
            return;
        }
        torch::Tensor digit;
        std::vector<torch::Tensor> maps;
        uint64_t gen = 0; float amax = 0.f; bool on_dev = false;
        int pred = -1, truth = -1;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            gen = probe_gen_;
            if (gen != 0) {
                digit = probe_digit_;         // refcount bumps, keep storage alive
                maps = probe_maps_;
                amax = probe_amax_;
                pred = probe_pred_;
                truth = probe_true_;
                on_dev = probe_on_device_;
            }
        }
        if (gen == 0) {
            ImGui::TextDisabled(
                "probe: start training to watch a digit flow through conv1");
            return;
        }
        if (gen != probe_tex_gen_) {          // new snapshot -> (re)upload
            upload_probe(digit, maps, amax, on_dev);
            probe_tex_gen_ = gen;
        }
        ImGui::Text("probe digit t10k[%lld] -> conv1 feature maps "
                    "(RdBu +/-%.3f, per-snapshot range; live)",
                    (long long)kProbeIdx, probe_amax_shown_);
        // Digit (VIRIDIS 0..1, ~112px) + caption on the left; the 4x2 map grid
        // (~52px cells) beside it on the right.
        ImGui::BeginGroup();
        if (probe_digit_tex_)
            ImGui::Image(caliper::Bridge::imtex(probe_digit_tex_),
                         ImVec2(112, 112));
        const ImVec4 ok{0.35f, 0.85f, 0.40f, 1.f}, bad{0.90f, 0.35f, 0.35f, 1.f};
        ImGui::TextColored(pred == truth ? ok : bad, "pred %d / true %d",
                           pred, truth);
        ImGui::EndGroup();
        ImGui::SameLine();
        ImGui::BeginGroup();
        for (size_t m = 0; m < probe_map_tex_.size(); ++m) {
            if (m % 4 != 0) ImGui::SameLine();
            ImGui::Image(caliper::Bridge::imtex(probe_map_tex_[m]),
                         ImVec2(52, 52));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("conv1 map %zu / 8  (26x26 -> 52px, nearest)", m);
        }
        ImGui::EndGroup();
        ImGui::TextDisabled("feature maps: %s",
            (probe_tex_on_device_ && !stage_cpu_)
                ? "GPU-resident (Metal, zero CPU staging)"
                : "CPU-staged (GL fallback)");
    }

    // Upload the probe snapshot. The digit is create-once/update-after (its
    // VIRIDIS 0..1 range never changes). The 8 feature maps carry a fresh
    // symmetric RdBu range every snapshot; v1 update_texture cannot rescale (no
    // range channel), so we release + recreate them — cheap at eval cadence, and
    // it also covers a shape change for free. Zero applet pixel work either way.
    void upload_probe(const torch::Tensor& digit,
                      const std::vector<torch::Tensor>& maps,
                      float amax, bool on_dev) {
        probe_tex_on_device_ = on_dev;
        probe_amax_shown_ = (amax > 0.f) ? amax : 1e-6f;
        probe_digit_tex_ = upload_mapped(probe_digit_tex_, digit,
                                         CALIPER_CMAP_VIRIDIS, 0.f, 1.f);
        if (probe_map_tex_.size() != maps.size()) {
            for (auto id : probe_map_tex_)
                if (id) bridge_.release_texture(id);
            probe_map_tex_.assign(maps.size(), 0);
        }
        for (size_t m = 0; m < maps.size(); ++m) {
            if (probe_map_tex_[m]) bridge_.release_texture(probe_map_tex_[m]);
            probe_map_tex_[m] = upload_mapped(0, maps[m], CALIPER_CMAP_RDBU,
                                              -probe_amax_shown_,
                                              +probe_amax_shown_);
        }
    }

    // Turn the 8 owned (3,3) kernel clones into 8 textures: create on the first
    // snapshot, update thereafter (same id/shape). We hand the bridge the
    // TRAINING-device tensor first — on the Metal renderer it is accepted and
    // colormapped on-GPU (zero CPU staging, the USP). On a non-Metal renderer
    // the bridge's active device is CPU, so a device tensor is rejected; we then
    // relocate the clone to CPU and the BRIDGE stages it. The applet never
    // touches a pixel on either path (§6c) — it only chooses where the tensor
    // lives, and the bridge's own accept/reject drives that choice.
    void upload_kernels(const std::vector<torch::Tensor>& dev_ks,
                        float wmax, bool on_dev) {
        if (kernel_tex_.empty()) {
            kernel_tex_.assign(dev_ks.size(), 0);
            kernel_tex_range_ = (wmax > 0.f) ? wmax : 1e-6f;
        }
        kernel_tex_on_device_ = on_dev;
        for (size_t k = 0; k < dev_ks.size(); ++k)
            // Create-once/update-after per kernel, range pinned at the first
            // snapshot. upload_mapped owns the shared GL relocate-to-CPU fallback.
            kernel_tex_[k] = upload_mapped(kernel_tex_[k], dev_ks[k],
                                           CALIPER_CMAP_RDBU,
                                           -kernel_tex_range_, +kernel_tex_range_);
    }

    // Shared bridge upload for ONE mapped (H,W) f32 tensor, with the C8 GL
    // relocate-to-CPU fallback factored out (reused by kernels and the probe).
    // id == 0 -> create; else update in place. We hand the bridge the TRAINING
    // -device tensor first: the Metal renderer accepts and colormaps it on-GPU
    // (zero CPU staging, the USP); a non-Metal bridge rejects the device tensor,
    // so we relocate the clone to CPU and the BRIDGE stages it — the applet never
    // touches a pixel (§6c), it only chooses where the tensor lives. stage_cpu_
    // latches true on the first device rejection so every later upload skips
    // straight to the CPU path. Returns the (possibly new) texture id.
    CaliperTextureId upload_mapped(CaliperTextureId id, const torch::Tensor& dev_t,
                                   int32_t cmap, float vmin, float vmax) {
        // host_t keeps a CPU copy alive across the synchronous bridge call when
        // the staging path is taken; the CaliperTensor descriptor aliases it.
        torch::Tensor host_t;
        auto view = [&](bool cpu) -> std::optional<CaliperTensor> {
            if (cpu) { host_t = dev_t.to(torch::kCPU);
                       return caliper::adapters::to_tensor(host_t); }
            return caliper::adapters::to_tensor(dev_t);
        };
        auto ct = view(stage_cpu_);
        if (!ct) return id;                  // clones are offset-0 contig; defensive
        if (id != 0) { bridge_.update_texture(id, &*ct); return id; }
        id = bridge_.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
        if (id == 0 && !stage_cpu_) {        // device rejected -> non-Metal bridge
            stage_cpu_ = true;
            ct = view(true);
            if (!ct) return 0;
            id = bridge_.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
        }
        return id;
    }

    void start_training() {
        // Re-entrancy latch: ignore a double-click that lands in the submit
        // window before is_running() flips true (the review's flagged race).
        if (job_id_ != 0 && jobs_.is_running(job_id_)) return;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss_history_.clear();
            acc_steps_.clear();
            acc_history_.clear();
            status_line_ = "starting…";
        }
        // ML-EXEMPLAR 1 — static trampoline + this: the raw C job contract.
        job_id_ = jobs_.submit("ml_scope: train MNIST CNN", &MLScope::train_job,
                               this);
        if (job_id_ == 0 && host_) host_->log_error("ml-scope: submit failed");
    }

    // Per-transfer state for curl's xferinfo callback: lets a cancel abort the
    // download mid-flight (returns non-zero -> curl aborts with CURLE_ABORTED).
    struct XferCtx { const CaliperJobControl* ctl; };
    static int xferinfo(void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
        auto* x = static_cast<XferCtx*>(p);
        return (x->ctl && x->ctl->cancelled(x->ctl)) ? 1 : 0;
    }

    // ML-EXEMPLAR 5 — acquisition INSIDE the job. Returns false on any failure
    // (offline, cancel, corrupt) after posting a clear status; caller returns.
    static bool ensure_dataset(MLScope* self, const CaliperJobControl* ctl) {
        std::string dir = self->host_ ? self->host_->data_dir() : "";
        for (int i = 0; i < 4; i++) {
            if (ctl->cancelled(ctl)) return false;
            std::string path = dir + "/" + kFiles[i];
            {
                std::ifstream f(path, std::ios::binary);
                if (f.good()) continue;   // cached — skip
            }
            self->set_status(std::string("downloading ") + kFiles[i] + "…");
            ctl->progress(ctl, (float)i / 4.f,
                          (std::string("downloading ") + kFiles[i]).c_str());

            std::string url = std::string(kBaseUrl) + kFiles[i] + ".gz";
            std::vector<uint8_t> gz;
            XferCtx xc{ctl};
            CURL* c = curl_easy_init();
            if (!c) { self->fail_dl(ctl); return false; }
            curl_easy_setopt(c, CURLOPT_URL, url.c_str());
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_to_vec);
            curl_easy_setopt(c, CURLOPT_WRITEDATA, &gz);
            curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
            curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, xferinfo);
            curl_easy_setopt(c, CURLOPT_XFERINFODATA, &xc);
            CURLcode rc = curl_easy_perform(c);
            curl_easy_cleanup(c);
            if (rc != CURLE_OK) {
                if (rc == CURLE_ABORTED_BY_CALLBACK) return false;  // cancel
                self->fail_dl(ctl);
                return false;
            }
            auto raw = mnist_idx::gunzip(gz);
            if (!raw) { self->fail_dl(ctl); return false; }
            // Atomic cache write: write to `.tmp`, then rename onto the canonical
            // name only on success. An interrupted write leaves a stray `.tmp`,
            // never a truncated file at `path` a later run would trust. Copy me.
            std::string tmp = path + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary);
                if (!out) { self->fail_dl(ctl); return false; }
                out.write((const char*)raw->data(), (std::streamsize)raw->size());
                out.flush();
                if (!out.good()) {
                    out.close();
                    std::remove(tmp.c_str());
                    self->fail_dl(ctl);
                    return false;
                }
            }
            if (std::rename(tmp.c_str(), path.c_str()) != 0) {
                std::remove(tmp.c_str());
                self->fail_dl(ctl);
                return false;
            }
        }
        return true;
    }

    void fail_dl(const CaliperJobControl* ctl) {
        set_status("MNIST download failed (offline?) — press start to retry");
        ctl->progress(ctl, 0.f,
                      "MNIST download failed (offline?) — press start to retry");
        if (host_) host_->log_error("ml-scope: MNIST download failed");
    }

    // Load a cached IDX pair into tensors: X (n,1,28,28) float/255, y long.
    // Self-healing: if a CACHED file fails to parse it is deleted so the next
    // start re-downloads it (the corrupt-cache wedge fixes itself). Copy me.
    static bool load_split(const std::string& dir, const char* img_name,
                           const char* lab_name, torch::Tensor& X,
                           torch::Tensor& y) {
        std::string ipath = dir + "/" + img_name, lpath = dir + "/" + lab_name;
        auto rd = [&](const std::string& p) -> std::optional<std::vector<uint8_t>> {
            std::ifstream f(p, std::ios::binary);
            if (!f) return std::nullopt;
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
        };
        auto ib = rd(ipath), lb = rd(lpath);
        if (!ib || !lb) return false;
        auto imgs = mnist_idx::parse_images(*ib);
        auto labs = mnist_idx::parse_labels(*lb);
        if (!imgs || !labs || (int)labs->size() != imgs->n) {
            std::remove(ipath.c_str());   // drop corrupt cache -> re-download
            std::remove(lpath.c_str());
            return false;
        }
        int n = imgs->n, r = imgs->rows, c = imgs->cols;
        X = torch::from_blob(imgs->pixels.data(), {n, 1, r, c}, torch::kUInt8)
                .to(torch::kFloat32)
                .div_(255.0f)
                .clone();   // clone: pixels vector is about to go out of scope
        std::vector<int64_t> ly(labs->begin(), labs->end());
        y = torch::from_blob(ly.data(), {n}, torch::kInt64).clone();
        return true;
    }

    static void train_job(void* user, const CaliperJobControl* ctl) {
        auto* self = static_cast<MLScope*>(user);
        torch::Device dev = self->device_.kind == CALIPER_DEV_METAL &&
                                    torch::hasMPS()
                                ? torch::Device(torch::kMPS)
                                : torch::Device(torch::kCPU);
        const bool on_mps = dev.type() == torch::kMPS;

        // ML-EXEMPLAR 5 — download+cache before training (both are job work).
        if (!ensure_dataset(self, ctl)) return;   // offline/cancel: clean exit
        if (ctl->cancelled(ctl)) return;

        self->set_status("parsing MNIST…");
        std::string d = self->host_ ? self->host_->data_dir() : "";
        torch::Tensor Xtr, ytr, Xte, yte;
        // ensure_dataset guaranteed all four files exist, so a parse failure here
        // means a cached file is corrupt (load_split already deleted it). Post
        // the self-heal message, not the offline one — the next start re-fetches.
        if (!load_split(d, kFiles[0], kFiles[1], Xtr, ytr) ||
            !load_split(d, kFiles[2], kFiles[3], Xte, yte)) {
            self->set_status(
                "cached MNIST file was corrupt — press start to re-download");
            ctl->progress(ctl, 0.f,
                "cached MNIST file was corrupt — press start to re-download");
            if (self->host_)
                self->host_->log_error("ml-scope: corrupt MNIST cache, deleted");
            return;
        }
        // Whole dataset fits comfortably in unified memory: move once.
        Xtr = Xtr.to(dev); ytr = ytr.to(dev);
        Xte = Xte.to(dev); yte = yte.to(dev);

        torch::manual_seed(7);
        // conv1 is held separately so ML-EXEMPLAR 7 can snapshot its (8,1,3,3)
        // weights; the Sequential shares the same module (holder = shared_ptr),
        // so model->to(dev) moves this exact tensor onto the training device.
        auto conv1 = torch::nn::Conv2d(1, 8, 3);
        auto model = torch::nn::Sequential(
            conv1, torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(8, 16, 3), torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Flatten(), torch::nn::Linear(400, 10));
        model->to(dev);
        torch::optim::Adam opt(model->parameters(),
                               torch::optim::AdamOptions(1e-3));

        // ML-EXEMPLAR 6 — begin the run now that data is loaded (so download /
        // corrupt-cache exits above never leave a dangling run). begin_run
        // returns 0 on error OR when metrics is absent (falsy wrapper): both
        // mean "do not stream", so the same `run != 0` guard covers both.
        // metrics.v1 is callable from this job thread — the host serializes
        // internally; this is exactly the write the teardown-order fix protects
        // (g_metrics outlives g_jobs, so a late scalar cannot fault).
        uint64_t run = self->metrics_.begin_run("mnist", "cnn");
        self->run_id_.store(run);
        if (run != 0)
            self->metrics_.hparams_json(
                run, R"({"lr":0.001,"batch":256,"epochs":3,"model":"conv8-16-fc"})");

        const int64_t n = Xtr.size(0);
        const int64_t batches_per_epoch = (n + kBatch - 1) / kBatch;
        const int64_t total_steps = batches_per_epoch * kEpochs;
        int64_t step = 0;

        // Full-t10k test accuracy in no_grad 1000-image batches. Returns nullopt
        // on cancel (caller ends the run and returns). One routine, reused for the
        // step-0 baseline, the mid-epoch cadence, and each epoch end. Leaves the
        // model in eval mode — training loops call model->train() before stepping.
        auto evaluate = [&]() -> std::optional<float> {
            model->eval();
            int64_t correct = 0, seen = Xte.size(0);
            torch::NoGradGuard ng;
            for (int64_t b = 0; b < seen; b += 1000) {
                if (ctl->cancelled(ctl)) return std::nullopt;
                int64_t hi = std::min<int64_t>(b + 1000, seen);
                auto xb = Xte.slice(0, b, hi);
                auto pred = model->forward(xb).argmax(1);
                correct += pred.eq(yte.slice(0, b, hi)).sum().item<int64_t>();
            }
            return seen ? 100.f * (float)correct / (float)seen : 0.f;
        };
        // Publish one evaluation at a global step: append to the paired UI arrays
        // (xs = step, ys = acc %) under the mutex, and stream it to metrics.
        // ML-EXEMPLAR 6 — test/accuracy is sampled every kEvalEvery steps (plus a
        // step-0 baseline); per-epoch cadence hides the learning transient on
        // fast-converging datasets. Step-indexed so it shares the loss x-axis.
        auto record_acc = [&](int64_t at_step, float accpct) {
            {
                std::lock_guard<std::mutex> lk(self->state_mutex_);
                self->acc_steps_.push_back((float)at_step);
                self->acc_history_.push_back(accpct);
            }
            if (run != 0)
                self->metrics_.scalar(run, "test/accuracy", at_step, accpct);
        };

        // ML-EXEMPLAR 7 — snapshot conv1's 8 filters for the frame thread. Runs
        // ONLY in the worker, and NEVER calls the bridge (UI-thread-only). Each
        // (3,3) kernel is an OWNED device clone: a raw select() view carries a
        // nonzero storage offset the MPS adapter rejects (offset-0 contract), and
        // the live weight keeps mutating as training continues — the clone (9
        // floats x 8, tiny) decouples both. Drain MPS ONCE here so the frame
        // thread never pays the device barrier. Publish under the mutex + bump a
        // generation the frame diffs against.
        auto snapshot_kernels = [&]() {
            if (!self->bridge_) return;          // no consumer -> skip the copy
            torch::NoGradGuard ng;
            auto w4 = conv1->weight.detach();    // (8,1,3,3), shares live storage
            float wmax = w4.abs().max().item<float>();   // forces MPS->CPU read
            std::vector<torch::Tensor> ks;
            ks.reserve(w4.size(0));
            for (int64_t k = 0; k < w4.size(0); ++k)
                ks.push_back(w4[k][0].clone());  // (3,3) OWNED, offset-0, contig
            if (on_mps) torch::mps::synchronize();  // pay the barrier once, here
            {
                std::lock_guard<std::mutex> lk(self->state_mutex_);
                self->kernel_snap_ = std::move(ks);
                self->kernel_wmax_ = wmax;
                self->kernel_on_device_ = on_mps;
                self->kernel_gen_++;
            }
        };

        // ML-EXEMPLAR 8 — snapshot the real-data panel: the fixed probe digit,
        // its conv1 feature maps, and the predicted/true labels. Same discipline
        // as snapshot_kernels — worker-only, NEVER touches the bridge, owned
        // offset-0 clones (raw select() views carry a nonzero storage offset the
        // MPS adapter rejects), one MPS drain here so the frame never syncs.
        // Runs in eval mode (every caller evaluates first), under no_grad.
        auto snapshot_probe = [&]() {
            if (!self->bridge_) return;          // no consumer -> skip the copy
            torch::NoGradGuard ng;
            auto x = Xte.slice(0, kProbeIdx, kProbeIdx + 1);   // (1,1,28,28) on dev
            auto feat = conv1->forward(x);                     // (1,8,26,26)
            int pred = (int)model->forward(x).argmax(1).item<int64_t>();
            int truth = (int)yte[kProbeIdx].item<int64_t>();
            float amax = feat.abs().max().item<float>();       // symmetric range
            auto digit = x[0][0].clone();        // (28,28) OWNED, offset-0, contig
            std::vector<torch::Tensor> maps;
            maps.reserve(feat.size(1));
            for (int64_t c = 0; c < feat.size(1); ++c)
                maps.push_back(feat[0][c].clone());  // (26,26) OWNED, offset-0
            if (on_mps) torch::mps::synchronize();   // drain the clones once, here
            {
                std::lock_guard<std::mutex> lk(self->state_mutex_);
                self->probe_digit_ = std::move(digit);
                self->probe_maps_ = std::move(maps);
                self->probe_amax_ = amax;
                self->probe_pred_ = pred;
                self->probe_true_ = truth;
                self->probe_on_device_ = on_mps;
                self->probe_gen_++;
            }
        };

        // Baseline BEFORE the first training step: an untrained net scores ~10%
        // (chance on 10 classes), anchoring the curve so the ramp is visible.
        if (auto acc0 = evaluate()) {
            record_acc(step, *acc0);              // step == 0 here
            snapshot_kernels();                  // random init: pure noise
            snapshot_probe();                    // untrained conv1 on a real digit
        } else { self->end_metrics_run(run); return; } // cancel during baseline

        for (int epoch = 0; epoch < kEpochs; epoch++) {
            model->train();
            auto perm = torch::randperm(n, torch::TensorOptions(dev).dtype(
                                               torch::kInt64));
            for (int64_t b = 0; b < n; b += kBatch) {
                if (ctl->cancelled(ctl)) {         // ML-EXEMPLAR 1 (+6: end_run)
                    self->end_metrics_run(run);
                    return;
                }
                int64_t hi = std::min<int64_t>(b + kBatch, n);
                auto idx = perm.slice(0, b, hi);
                auto xb = Xtr.index_select(0, idx);
                auto yb = ytr.index_select(0, idx);
                opt.zero_grad();
                auto out = torch::log_softmax(model->forward(xb), 1);
                auto loss = torch::nll_loss(out, yb);
                loss.backward();
                opt.step();
                float l = loss.item<float>();
                {
                    std::lock_guard<std::mutex> lk(self->state_mutex_);
                    self->loss_history_.push_back(l);
                }
                // ML-EXEMPLAR 6 — one scalar per batch under the global step.
                if (run != 0) self->metrics_.scalar(run, "train/loss", step, l);
                step++;
                char msg[96];
                std::snprintf(msg, sizeof msg, "epoch %d/%d  loss %.4f",
                              epoch + 1, kEpochs, l);
                ctl->progress(ctl, (float)step / (float)total_steps, msg);

                // Mid-epoch accuracy sample: MNIST converges inside epoch 1, so
                // this is where the learning curve actually lives.
                if (step % kEvalEvery == 0) {
                    if (auto acc = evaluate()) { record_acc(step, *acc);
                                                 snapshot_kernels();
                                                 snapshot_probe(); }
                    else { self->end_metrics_run(run); return; }  // cancel in eval
                    model->train();   // evaluate() left the model in eval mode
                }
            }

            // End-of-epoch accuracy (the final epoch end is also the completion
            // point). batches_per_epoch is not a multiple of kEvalEvery, so this
            // does not duplicate a mid-epoch sample at the same step.
            float accpct;
            if (auto acc = evaluate()) { accpct = *acc; record_acc(step, accpct);
                                         snapshot_kernels();
                                         snapshot_probe(); }
            else { self->end_metrics_run(run); return; }   // cancel during eval
            char msg[96];
            std::snprintf(msg, sizeof msg, "epoch %d/%d  test acc %.2f%%",
                          epoch + 1, kEpochs, accpct);
            self->set_status(msg);
            ctl->progress(ctl, (float)step / (float)total_steps, msg);
        }
        self->end_metrics_run(run);            // ML-EXEMPLAR 6: completion path
        self->set_status("training complete");
    }

    // end_run on EVERY exit path from train_job (completion, both cancel points).
    // No-op when run == 0 (metrics absent, or begin_run failed). Clearing
    // run_id_ flips the status line back to "present (open Runs)".
    void end_metrics_run(uint64_t run) {
        if (run != 0) metrics_.end_run(run);
        run_id_.store(0);
    }

    caliper::Host* host_ = nullptr;
    caliper::Jobs jobs_;
    caliper::Device device_;
    caliper::Metrics metrics_;            // ML-EXEMPLAR 6 — optional, falsy-inert
    caliper::Bridge bridge_;              // ML-EXEMPLAR 7 — optional, falsy-inert
    std::atomic<uint64_t> run_id_{0};     // live run id for the status line (0 = none)
    uint64_t job_id_ = 0;
    std::mutex state_mutex_;
    std::vector<float> loss_history_;
    std::vector<float> acc_steps_;    // xs for the accuracy plot (global step)
    std::vector<float> acc_history_;  // ys (test accuracy %), paired with acc_steps_
    std::string status_line_ = "idle — press start to download MNIST + train";

    // ML-EXEMPLAR 7 — conv1 snapshot published by the worker (state_mutex_).
    // OWNED torch storage: the CaliperTensor descriptors the frame builds point
    // into these tensors, so they must outlive each bridge call (the frame holds
    // a refcounted copy for the duration — see render_kernels).
    std::vector<torch::Tensor> kernel_snap_;   // 8 x (3,3), on the training device
    float    kernel_wmax_ = 0.f;               // max|weight| of the snapshot
    bool     kernel_on_device_ = false;        // true when snapshot lives on MPS
    uint64_t kernel_gen_ = 0;                   // bumped per snapshot (0 = none yet)
    // Frame-thread-owned texture state (never touched by the worker).
    std::vector<CaliperTextureId> kernel_tex_; // one texture per filter
    uint64_t kernel_tex_gen_ = 0;              // last generation uploaded
    float    kernel_tex_range_ = 0.f;          // symmetric RdBu range (fixed at 1st)
    bool     kernel_tex_on_device_ = false;    // snapshot was device-resident
    // Shared across kernel + probe uploads: latches true the first time the
    // bridge rejects a device tensor (non-Metal renderer -> CPU staging, §6c).
    bool     stage_cpu_ = false;

    // ML-EXEMPLAR 8 — real-data snapshot published by the worker (state_mutex_).
    // OWNED torch storage, same lifetime contract as kernel_snap_: the frame
    // holds a refcounted copy for the duration of each bridge call.
    torch::Tensor probe_digit_;                 // (28,28) input on training device
    std::vector<torch::Tensor> probe_maps_;     // 8 x (26,26) conv1 activations
    float    probe_amax_ = 0.f;                 // max|activation| of the snapshot
    int      probe_pred_ = -1;                  // argmax of the model on the probe
    int      probe_true_ = -1;                  // ground-truth label of the probe
    bool     probe_on_device_ = false;          // snapshot lives on MPS
    uint64_t probe_gen_ = 0;                     // bumped per snapshot (0 = none yet)
    // Frame-thread-owned probe textures (never touched by the worker).
    CaliperTextureId probe_digit_tex_ = 0;      // VIRIDIS 0..1, create-once
    std::vector<CaliperTextureId> probe_map_tex_; // 8 RdBu maps, recreated/snapshot
    uint64_t probe_tex_gen_ = 0;                // last generation uploaded
    float    probe_amax_shown_ = 0.f;           // symmetric RdBu range in use
    bool     probe_tex_on_device_ = false;      // snapshot was device-resident
};

CALIPER_APPLET(MLScope,
    .id       = "dev.caliper.ml-scope",
    .version  = "0.1.0",
    .name     = "MLScope",
    .summary  = "ML exemplar: trains a small CNN on MNIST off the frame thread "
                "via caliper.jobs.v1, device-negotiated, with live loss and test "
                "accuracy — and a live conv1 kernel grid via tensor_bridge.v1, "
                "GPU-resident (zero CPU staging) on the Metal backend.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
