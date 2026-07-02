// ============================================================================
// MLScope — the ML exemplar (PLATFORM.md §17 Phase 2, step 1 of the ratified
// sequencing). Shows the idioms of ML on the platform:
//   ML-EXEMPLAR 1 — never train on the frame thread: submit to caliper.jobs.v1
//     and poll cancelled() in the batch loop (cooperative cancel).
//   ML-EXEMPLAR 2 — the host picks the device (caliper.device.v1); the applet
//     maps the KIND to its framework: METAL -> torch::kMPS here.
//   ML-EXEMPLAR 3 — publish training state to the UI under a mutex; the frame
//     reads a copy. (repnet's snapshot pattern, minimal form.)
//   ML-EXEMPLAR 4 — deliberately NO weight-matrix visualization yet: that is
//     tensor_bridge.v1's job (Plan 2C). A CPU-staged copy here would teach the
//     exact pattern the platform exists to delete.
//   ML-EXEMPLAR 5 — heavy data is job work too — download once into data_dir,
//     cache forever, cancellable. The frame thread never touches the network.
//   ML-EXEMPLAR 6 — probe-optional pays off: caliper.metrics.v1 is optional in
//     the manifest, so we probe it (caliper::Metrics, falsy-inert if absent)
//     and stream the run only when it is truthy. The SAME binary runs on a host
//     with metrics (loss/accuracy land in the Runs dashboard) and on one without
//     (training is unchanged; status says `metrics: absent (ok)`). Every applet
//     that logs a scalar this way inherits the Runs dashboard for free.
// ============================================================================
#include <caliper/caliper.hpp>
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
        // curl global init MUST happen once here on the frame thread: lazy init
        // from curl_easy_init on a worker thread is not thread-safe (libcurl docs).
        curl_global_init(CURL_GLOBAL_DEFAULT);
        host.log_info("ml-scope: on_init");
        return true;
    }

    void on_frame(const caliper::Frame&) override {
        ImGui::SetNextWindowPos({60, 80}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({620, 560}, ImGuiCond_FirstUseEver);
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
        ImGui::TextWrapped("First-layer conv kernels arrive with "
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
            // contract; the 1000 ms ceiling also covers a cancel that lands
            // mid-download (curl's xferinfo poll aborts the transfer, but the
            // socket teardown adds slack), so this bounded wait cannot hang.
            for (int i = 0; i < 1000 && jobs_.is_running(job_id_); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
        auto model = torch::nn::Sequential(
            torch::nn::Conv2d(1, 8, 3), torch::nn::ReLU(),
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

        // Baseline BEFORE the first training step: an untrained net scores ~10%
        // (chance on 10 classes), anchoring the curve so the ramp is visible.
        if (auto acc0 = evaluate())
            record_acc(step, *acc0);              // step == 0 here
        else { self->end_metrics_run(run); return; }   // cancel during baseline

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
                    if (auto acc = evaluate()) record_acc(step, *acc);
                    else { self->end_metrics_run(run); return; }  // cancel in eval
                    model->train();   // evaluate() left the model in eval mode
                }
            }

            // End-of-epoch accuracy (the final epoch end is also the completion
            // point). batches_per_epoch is not a multiple of kEvalEvery, so this
            // does not duplicate a mid-epoch sample at the same step.
            float accpct;
            if (auto acc = evaluate()) { accpct = *acc; record_acc(step, accpct); }
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
    std::atomic<uint64_t> run_id_{0};     // live run id for the status line (0 = none)
    uint64_t job_id_ = 0;
    std::mutex state_mutex_;
    std::vector<float> loss_history_;
    std::vector<float> acc_steps_;    // xs for the accuracy plot (global step)
    std::vector<float> acc_history_;  // ys (test accuracy %), paired with acc_steps_
    std::string status_line_ = "idle — press start to download MNIST + train";
};

CALIPER_APPLET(MLScope,
    .id       = "dev.caliper.ml-scope",
    .version  = "0.1.0",
    .name     = "MLScope",
    .summary  = "ML exemplar: trains a small CNN on MNIST off the frame thread "
                "via caliper.jobs.v1, device-negotiated, with live loss and test "
                "accuracy. Kernel visualization arrives with tensor_bridge "
                "(Phase 2C).",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
