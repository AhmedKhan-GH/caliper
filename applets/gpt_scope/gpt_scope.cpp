// gpt_scope.cpp — GPTScope, the flagship applet (Task E1, Phase 2E′).
//
// A char-level nanoGPT trained on TinyShakespeare, born native on the service
// stack. It inherits MLScope's idioms verbatim-in-spirit:
//   • never train on the frame thread — submit to caliper.jobs.v1 and poll
//     cancelled() every step (cooperative cancel);
//   • the host picks the device (caliper.device.v1); the applet maps METAL ->
//     torch::kMPS;
//   • publish training state (loss curves + the live sample) under a mutex; the
//     frame reads a copy;
//   • data acquisition is job work — download the corpus once into data_dir,
//     cache forever, cancellable, atomic, self-healing (the B1 recipe, plain
//     text so no gunzip);
//   • caliper.metrics.v1 is OPTIONAL — probe it, stream train/loss + val/loss
//     only when present (the same binary runs with and without the Runs
//     dashboard);
//   • caliper.tensor_bridge.v1 is OPTIONAL and PROBED here for E2 (the live
//     attention panel). E1 builds the capability into the model (the probe
//     forward) and reports the service's presence; it uploads no textures yet.
//
// Checkpoint save/load is DEFERRED by design: it is the first honest demand for
// caliper.artifacts.v1 (the D16 demand-driven clause). The UI shows a disabled
// "save checkpoint" button whose tooltip says exactly that.
#include "gpt_model.h"

#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace gptscope {
namespace {

// --- fixed training recipe (the brief is binding) --------------------------
constexpr int    kMaxSteps   = 3000;
constexpr int    kBatch      = 64;
constexpr int    kBlock      = 128;   // == GPTConfig::block_size
constexpr int    kEvalEvery  = 100;   // val loss + a live sample this often
constexpr int    kSampleLen  = 200;   // chars per sample
constexpr int    kValBatches = 20;    // batches averaged for the val-loss point
constexpr double kLR         = 3e-4;  // AdamW
constexpr double kTemp       = 0.8;   // sampling temperature

// The fixed corpus (phase2e constraints): a single plain-text file, cached as
// <data_dir>/tinyshakespeare.txt.
constexpr const char* kDataUrl =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt";
constexpr const char* kDataFile = "tinyshakespeare.txt";

// libcurl write callback: append received bytes to a std::string.
size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* ud) {
    auto* s = static_cast<std::string*>(ud);
    const size_t n = size * nmemb;
    s->append(ptr, n);
    return n;
}

} // namespace

// ---------------------------------------------------------------------------
// pImpl state — everything heavy the header hides.
// ---------------------------------------------------------------------------
struct GPTScopeState {
    caliper::Host*   host = nullptr;
    caliper::Jobs    jobs;
    caliper::Device  device;
    caliper::Metrics metrics;              // optional — falsy-inert if absent
    caliper::Bridge  bridge;               // optional — probed for E2 (no upload)
    std::atomic<uint64_t> run_id{0};       // live metrics run id (0 = none)
    uint64_t         job_id = 0;

    std::mutex          mtx;               // guards everything below
    std::vector<float>  loss_x, loss_y;    // train cross-entropy (step, value)
    std::vector<float>  val_x, val_y;      // val cross-entropy (step, value)
    std::string         sample;            // latest generated sample text
    std::string         status =
        "idle — press start to download TinyShakespeare + train";
    int                 vocab_size = 0;    // header, once the corpus is read

    void set_status(const std::string& s) {
        std::lock_guard<std::mutex> lk(mtx);
        status = s;
    }
};

namespace {

// Per-transfer state for curl's xferinfo callback: a cancel aborts the download
// mid-flight (returns non-zero -> CURLE_ABORTED_BY_CALLBACK).
struct XferCtx { const CaliperJobControl* ctl; };
int xferinfo(void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* x = static_cast<XferCtx*>(p);
    return (x->ctl && x->ctl->cancelled(x->ctl)) ? 1 : 0;
}

// B1 recipe (plain text): download once into data_dir, cache forever,
// cancellable, atomic .tmp+rename, self-healing on an unreadable/empty cache.
// Returns the corpus text, or nullopt after posting a clear status (offline /
// cancel / empty).
std::optional<std::string> ensure_corpus(GPTScopeState* st,
                                         const CaliperJobControl* ctl) {
    const std::string dir  = st->host ? st->host->data_dir() : "";
    const std::string path = dir + "/" + kDataFile;

    // Cache hit: a non-empty cached file is trusted.
    {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            std::string text((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
            if (!text.empty()) return text;
        }
    }
    if (ctl->cancelled(ctl)) return std::nullopt;

    st->set_status("downloading TinyShakespeare…");
    ctl->progress(ctl, 0.f, "downloading TinyShakespeare");

    std::string body;
    XferCtx xc{ctl};
    CURL* c = curl_easy_init();
    if (!c) { st->set_status("download failed (curl init)"); return std::nullopt; }
    curl_easy_setopt(c, CURLOPT_URL, kDataUrl);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, xferinfo);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, &xc);
    const CURLcode rc = curl_easy_perform(c);
    curl_easy_cleanup(c);
    if (rc != CURLE_OK) {
        if (rc == CURLE_ABORTED_BY_CALLBACK) return std::nullopt;   // cancel
        st->set_status("TinyShakespeare download failed (offline?) — "
                       "press start to retry");
        ctl->progress(ctl, 0.f, "download failed (offline?)");
        if (st->host) st->host->log_error("gpt-scope: corpus download failed");
        return std::nullopt;
    }
    if (body.empty()) {
        st->set_status("download returned an empty file — press start to retry");
        return std::nullopt;
    }

    // Atomic cache write: .tmp then rename onto the canonical name only on
    // success. An interrupted write leaves a stray .tmp, never a truncated file
    // a later run would trust.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) { st->set_status("cache write failed"); return std::nullopt; }
        out.write(body.data(), static_cast<std::streamsize>(body.size()));
        out.flush();
        if (!out.good()) {
            out.close();
            std::remove(tmp.c_str());
            st->set_status("cache write failed");
            return std::nullopt;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        st->set_status("cache rename failed");
        return std::nullopt;
    }
    return body;
}

void end_metrics_run(GPTScopeState* st, uint64_t run) {
    if (run != 0) st->metrics.end_run(run);
    st->run_id.store(0);
}

// ---------------------------------------------------------------------------
// The training job. Runs on a host worker thread (never crash-guarded) — poll
// cancelled() and return promptly. user is the State*.
// ---------------------------------------------------------------------------
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<GPTScopeState*>(user);

    const torch::Device dev =
        (st->device.kind == CALIPER_DEV_METAL && torch::hasMPS())
            ? torch::Device(torch::kMPS)
            : torch::Device(torch::kCPU);

    // ---- data acquisition (job work) --------------------------------------
    auto corpus = ensure_corpus(st, ctl);
    if (!corpus) return;                       // offline / cancel / empty
    if (ctl->cancelled(ctl)) return;

    st->set_status("building char vocabulary…");
    const std::string& text = *corpus;

    // Char-level vocab: sorted unique bytes -> stoi / itos.
    std::vector<char> itos;
    {
        std::vector<bool> seen(256, false);
        for (unsigned char ch : text) seen[ch] = true;
        for (int i = 0; i < 256; ++i)
            if (seen[i]) itos.push_back(static_cast<char>(i));
    }
    std::map<char, int64_t> stoi;
    for (int64_t i = 0; i < (int64_t)itos.size(); ++i) stoi[itos[i]] = i;
    const int64_t V = (int64_t)itos.size();
    { std::lock_guard<std::mutex> lk(st->mtx); st->vocab_size = (int)V; }

    // Encode + 90/10 split. Move each split to the device once (TinyShakespeare
    // is ~1.1M tokens: a couple of MB of int64, comfortable in unified memory).
    std::vector<int64_t> ids(text.size());
    for (size_t i = 0; i < text.size(); ++i) ids[i] = stoi[text[i]];
    const int64_t n = (int64_t)ids.size();
    const int64_t n_train = (int64_t)(0.9 * n);
    auto all = torch::from_blob(ids.data(), {n}, torch::kInt64).clone();
    auto train_ids = all.slice(0, 0, n_train).to(dev);
    auto val_ids   = all.slice(0, n_train, n).to(dev);

    // A random contiguous batch: x (B,block), y (B,block) shifted by one, built
    // entirely on-device via advanced indexing (no per-item host sync).
    auto get_batch = [&](const torch::Tensor& data) {
        const int64_t len = data.size(0);
        auto ix = torch::randint(0, len - kBlock - 1, {kBatch},
                                 torch::TensorOptions(dev).dtype(torch::kLong));
        auto ar = torch::arange(kBlock,
                                torch::TensorOptions(dev).dtype(torch::kLong));
        auto rows = ix.unsqueeze(1) + ar.unsqueeze(0);        // (B,block)
        auto x = data.index({rows});
        auto y = data.index({rows + 1});
        return std::make_pair(x, y);
    };

    // ---- model + optimizer ------------------------------------------------
    torch::manual_seed(1337);
    GPTConfig cfg;
    cfg.vocab_size = V;                         // 4/4/128, block 128, dropout 0.1
    GPT model(cfg);
    model->to(dev);
    torch::optim::AdamW opt(model->parameters(),
                            torch::optim::AdamWOptions(kLR));

    // ML-EXEMPLAR 6 — metrics is optional: begin_run returns 0 when absent or on
    // error; the same `run != 0` guard covers both. Callable from this thread
    // (the host serializes internally).
    const uint64_t run = st->metrics.begin_run("tinyshakespeare", "mini-gpt");
    st->run_id.store(run);
    if (run != 0)
        st->metrics.hparams_json(
            run,
            R"({"lr":3e-4,"batch":64,"block":128,"n_layer":4,"n_head":4,)"
            R"("n_embd":128,"dropout":0.1,"max_steps":3000})");

    // Average cross-entropy over a few val batches (no_grad, eval). nullopt on
    // cancel. Leaves the model in eval mode — callers call train() after.
    auto eval_val = [&]() -> std::optional<float> {
        model->eval();
        torch::NoGradGuard ng;
        double sum = 0.0;
        for (int i = 0; i < kValBatches; ++i) {
            if (ctl->cancelled(ctl)) return std::nullopt;
            auto [xb, yb] = get_batch(val_ids);
            auto logits = model->forward(xb);
            auto loss = torch::nn::functional::cross_entropy(
                logits.view({-1, V}), yb.reshape({-1}));
            sum += loss.item<double>();
        }
        return (float)(sum / kValBatches);
    };

    // Generate a kSampleLen-char sample seeded from a newline (fallback: id 0),
    // decode via itos, and publish it (+ the val-loss point) under the mutex.
    auto eval_and_sample = [&](int64_t step) -> bool {
        auto vloss = eval_val();
        if (!vloss) return false;                              // cancelled
        auto seed_it = stoi.find('\n');
        const int64_t seed = seed_it != stoi.end() ? seed_it->second : 0;
        auto idx = torch::full({1, 1}, seed,
                               torch::TensorOptions(dev).dtype(torch::kLong));
        auto out = model->generate(idx, kSampleLen, kTemp).to(torch::kCPU);
        auto* p = out.data_ptr<int64_t>();
        std::string s;
        s.reserve(out.size(1));
        for (int64_t i = 0; i < out.size(1); ++i) {
            const int64_t id = p[i];
            if (id >= 0 && id < V) s.push_back(itos[id]);
        }
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->val_x.push_back((float)step);
            st->val_y.push_back(*vloss);
            st->sample = std::move(s);
        }
        if (run != 0) st->metrics.scalar(run, "val/loss", step, *vloss);
        model->train();
        return true;
    };

    st->set_status("training…");
    for (int64_t step = 0; step < kMaxSteps; ++step) {
        if (ctl->cancelled(ctl)) { end_metrics_run(st, run); return; }

        // A live sample every kEvalEvery steps, including a step-0 baseline (an
        // untrained net emits noise — it anchors the demo arc).
        if (step % kEvalEvery == 0) {
            if (!eval_and_sample(step)) { end_metrics_run(st, run); return; }
        }

        model->train();
        auto [xb, yb] = get_batch(train_ids);
        opt.zero_grad();
        auto logits = model->forward(xb);                      // (B,T,V)
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, V}), yb.reshape({-1}));
        loss.backward();
        opt.step();

        const float l = loss.item<float>();
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->loss_x.push_back((float)step);
            st->loss_y.push_back(l);
        }
        if (run != 0) st->metrics.scalar(run, "train/loss", step, l);

        char msg[96];
        std::snprintf(msg, sizeof msg, "step %lld/%d  loss %.4f",
                      (long long)(step + 1), kMaxSteps, l);
        ctl->progress(ctl, (float)(step + 1) / (float)kMaxSteps, msg);
    }

    // A final sample + val point at the end of the run.
    if (!eval_and_sample(kMaxSteps)) { end_metrics_run(st, run); return; }
    end_metrics_run(st, run);
    st->set_status("training complete");
}

} // namespace

// ---------------------------------------------------------------------------
// Applet facade.
// ---------------------------------------------------------------------------
GPTScopeApplet::GPTScopeApplet() : s_(std::make_unique<GPTScopeState>()) {}
GPTScopeApplet::~GPTScopeApplet() = default;

bool GPTScopeApplet::initialize(caliper::Host& host) {
    s_->host    = &host;
    s_->jobs    = caliper::Jobs(host);        // required -> present (manifest)
    s_->device  = caliper::Device::query(host);
    s_->metrics = caliper::Metrics(host);     // optional — probe (falsy-inert)
    s_->bridge  = caliper::Bridge(host);      // optional — probe for E2's panel
    // curl global init MUST run once on the frame thread: lazy init from
    // curl_easy_init on a worker is not thread-safe (libcurl docs).
    curl_global_init(CURL_GLOBAL_DEFAULT);
    host.log_info("gpt-scope: initialize");
    return true;
}

void GPTScopeApplet::draw_ui() {
    ImGui::SetNextWindowPos({80, 80}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({660, 760}, ImGuiCond_FirstUseEver);
    ImGui::Begin("GPTScope");

    // Config + device header.
    ImGui::TextDisabled(
        "mini-GPT (char-level): 4 layers · 4 heads · 128 embd · block 128 · "
        "dropout 0.10");
    ImGui::TextDisabled(
        "device: %s (%s)  |  free mem hint: %.1f GB", s_->device.name,
        s_->device.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
        : s_->device.kind == CALIPER_DEV_CUDA ? "CUDA"
                                              : "CPU",
        s_->device.free_memory_hint / 1073741824.0);

    // Snapshot worker-published state under the mutex.
    std::vector<float> lx, ly, vx, vy;
    std::string sample, status;
    int vocab = 0;
    {
        std::lock_guard<std::mutex> lk(s_->mtx);
        lx = s_->loss_x; ly = s_->loss_y;
        vx = s_->val_x;  vy = s_->val_y;
        sample = s_->sample;
        status = s_->status;
        vocab  = s_->vocab_size;
    }
    if (vocab > 0) ImGui::TextDisabled("vocabulary: %d chars", vocab);
    ImGui::TextWrapped("%s", status.c_str());

    // Optional-service status lines (probe-optional pattern).
    if (s_->metrics) {
        const uint64_t run = s_->run_id.load();
        if (run != 0) ImGui::TextDisabled("metrics: run #%llu",
                                          (unsigned long long)run);
        else          ImGui::TextDisabled("metrics: present (open Runs)");
    } else {
        ImGui::TextDisabled("metrics: absent (ok)");
    }
    if (s_->bridge)
        ImGui::TextDisabled(
            "tensor_bridge.v1: present — live attention maps arrive in E2");
    else
        ImGui::TextDisabled("tensor_bridge.v1: absent (ok) — E2 panel needs it");

    // Start / cancel + tray-mirrored progress.
    const bool running = s_->job_id != 0 && s_->jobs.is_running(s_->job_id);
    if (!running) {
        if (ImGui::Button("start training") &&
            !(s_->job_id != 0 && s_->jobs.is_running(s_->job_id))) {
            // Re-entrancy latch: ignore a click that lands in the submit window
            // before is_running() flips true (MLScope's flagged race).
            {
                std::lock_guard<std::mutex> lk(s_->mtx);
                s_->loss_x.clear(); s_->loss_y.clear();
                s_->val_x.clear();  s_->val_y.clear();
                s_->sample.clear();
                s_->status = "starting…";
            }
            s_->job_id = s_->jobs.submit("gpt_scope: train mini-GPT",
                                         &train_job, s_.get());
            if (s_->job_id == 0 && s_->host)
                s_->host->log_error("gpt-scope: submit failed");
        }
    } else {
        if (ImGui::Button("cancel")) s_->jobs.request_cancel(s_->job_id);
        ImGui::SameLine();
        ImGui::ProgressBar(s_->jobs.progress_of(s_->job_id), {-1, 0});
    }

    // Train + val loss, shared step axis.
    if (ImPlot::BeginPlot("loss", {-1, 220})) {
        ImPlot::SetupAxes("step", "cross-entropy");
        if (!ly.empty())
            ImPlot::PlotLine("train", lx.data(), ly.data(), (int)ly.size());
        if (!vy.empty())
            ImPlot::PlotLine("val", vx.data(), vy.data(), (int)vy.size());
        ImPlot::EndPlot();
    }

    // The live sample panel — the demo arc. The default ImGui font is fixed
    // width, so a plain scrollable child reads as monospace.
    ImGui::Text("live sample  (temperature %.1f, seeded from newline)", kTemp);
    ImGui::BeginChild("sample", {-1, 200}, ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    if (sample.empty())
        ImGui::TextDisabled(
            "(samples appear here every %d steps once training starts)",
            kEvalEvery);
    else
        ImGui::TextUnformatted(sample.c_str());
    ImGui::EndChild();

    // Deferred checkpoint save — the first honest demand for artifacts.v1.
    ImGui::BeginDisabled();
    ImGui::Button("save checkpoint");
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(
            "arrives with caliper.artifacts.v1 — the first real demand for it");

    ImGui::End();
}

void GPTScopeApplet::cleanup() {
    if (s_->job_id != 0) {
        s_->jobs.request_cancel(s_->job_id);
        // `user` (State, owned by this object) must outlive the job (jobs_v1.h
        // contract): wait for the worker to exit BEFORE we free it. Cancel is
        // honored <=100 ms by tested contract; the 1000 ms ceiling also covers a
        // cancel that lands mid-download (curl's xferinfo poll aborts the
        // transfer, but socket teardown adds slack), so this cannot hang.
        for (int i = 0; i < 1000 && s_->jobs.is_running(s_->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Pairs with the initialize() curl_global_init; safe only once the worker
    // (the sole curl user) has exited, which the bounded wait above ensures.
    curl_global_cleanup();
    if (s_->host) s_->host->log_info("gpt-scope: cleanup");
}

} // namespace gptscope
