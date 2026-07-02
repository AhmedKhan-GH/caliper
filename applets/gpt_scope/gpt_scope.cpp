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
#include <caliper/adapters/torch.hpp>   // torch::Tensor -> CaliperTensor (E2)
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
constexpr int    kProbeLen   = 64;    // chars in the fixed E2 attention excerpt

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

    // E2 — the selected attention layer: UI writes it, the worker reads it at
    // its next eval tick (the atomic desired-layer the brief specifies).
    std::atomic<int>    att_layer_sel{0};

    // E2 — attention snapshot published by the worker (guarded by mtx). OWNED
    // torch storage: the CaliperTensor descriptors the frame builds point into
    // these, so the frame holds a refcounted copy for the duration of each
    // (synchronous) bridge call (the ml_scope EXEMPLAR-7/8 lifetime contract).
    std::vector<torch::Tensor> att_heads;      // n_head x (T,T) on train device
    std::vector<float>         att_hmax;       // per-head max (vmax; vmin 0)
    std::string                att_probe;      // the fixed kProbeLen-char excerpt
    int                        att_snap_layer = 0;   // layer this snapshot is for
    bool                       att_on_device = false;// snapshot lives on MPS
    uint64_t                   att_gen = 0;    // bumped per snapshot (0 = none)

    // E2 — frame-thread-owned attention textures (the worker never touches these).
    std::vector<CaliperTextureId> att_tex;     // one texture per head
    uint64_t                      att_tex_gen = 0;     // last generation uploaded
    std::vector<float>            att_tex_hmax;        // per-head vmax in use
    bool                          att_tex_on_device = false;
    bool                          att_stage_cpu = false;  // GL device-reject latch

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

// E2 — shared bridge upload for ONE mapped (H,W) f32 tensor, with the C8 GL
// relocate-to-CPU fallback (mirrors ml_scope's upload_mapped). id==0 -> create;
// else update in place. We hand the bridge the TRAINING-device tensor first: the
// Metal renderer accepts and colormaps it on-GPU (zero CPU staging, the USP); a
// non-Metal bridge rejects the device tensor, so we relocate the clone to CPU
// and the BRIDGE stages it — the applet never touches a pixel (§6c), it only
// chooses where the tensor lives. `stage_cpu` latches true on the first device
// rejection so every later upload skips straight to the CPU path.
CaliperTextureId upload_mapped(const caliper::Bridge& bridge, bool& stage_cpu,
                               CaliperTextureId id, const torch::Tensor& dev_t,
                               int32_t cmap, float vmin, float vmax) {
    // host_t keeps a CPU copy alive across the synchronous bridge call when the
    // staging path is taken; the CaliperTensor descriptor aliases it.
    torch::Tensor host_t;
    auto view = [&](bool cpu) -> std::optional<CaliperTensor> {
        if (cpu) { host_t = dev_t.to(torch::kCPU);
                   return caliper::adapters::to_tensor(host_t); }
        return caliper::adapters::to_tensor(dev_t);
    };
    auto ct = view(stage_cpu);
    if (!ct) return id;                 // clones are offset-0 contig; defensive
    if (id != 0) { bridge.update_texture(id, &*ct); return id; }
    id = bridge.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
    if (id == 0 && !stage_cpu) {        // device rejected -> non-Metal bridge
        stage_cpu = true;
        ct = view(true);
        if (!ct) return 0;
        id = bridge.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
    }
    return id;
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
    const bool on_mps = dev.type() == torch::kMPS;

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

    // E2 — a FIXED kProbeLen-char val excerpt, chosen once per run (the first
    // chars of the validation split). Encoded to a (1,plen) device tensor for
    // the probe forward, and decoded to the string the panel highlights per
    // char. The SAME excerpt across the run lets you watch the heads sharpen.
    std::string probe_str;
    torch::Tensor probe_tok;
    {
        const int64_t plen = std::min<int64_t>(kProbeLen, n - n_train);
        std::vector<int64_t> pids((size_t)std::max<int64_t>(plen, 0));
        probe_str.reserve((size_t)std::max<int64_t>(plen, 0));
        for (int64_t i = 0; i < plen; ++i) {
            pids[(size_t)i] = ids[(size_t)(n_train + i)];
            probe_str.push_back(itos[pids[(size_t)i]]);
        }
        if (plen > 0)
            probe_tok = torch::from_blob(pids.data(), {1, plen}, torch::kInt64)
                            .clone().to(dev);
    }

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

    // E2 — snapshot the SELECTED layer's attention on the fixed probe excerpt for
    // the frame thread. Worker-only, NEVER touches the bridge (UI-thread-only).
    // probe_attention runs eval()+no_grad and restores the prior mode. Each head
    // is an OWNED offset-0 (T,T) clone (a raw select() view carries a nonzero
    // storage offset the MPS adapter rejects, and the live weights keep changing
    // — the clone decouples both); MPS is drained ONCE here so the frame never
    // syncs. Per-head max is the VIRIDIS vmax (vmin 0). Publish under the mutex +
    // bump a generation the frame diffs against. Skipped when no bridge consumer.
    auto snapshot_attention = [&]() {
        if (!st->bridge || !probe_tok.defined()) return;
        const int want = st->att_layer_sel.load();
        auto att = model->probe_attention(probe_tok);   // per-layer (nh,T,T)
        if (att.empty()) return;
        const int li = std::clamp(want, 0, (int)att.size() - 1);
        auto layer_att = att[li];                        // (nh,T,T) on device
        const int64_t nh = layer_att.size(0);
        std::vector<torch::Tensor> heads;
        std::vector<float> hmax;
        heads.reserve((size_t)nh); hmax.reserve((size_t)nh);
        for (int64_t h = 0; h < nh; ++h) {
            auto hd = layer_att[h].clone();   // (T,T) OWNED, offset-0, contiguous
            hmax.push_back(hd.max().item<float>());   // forces read; vmax (vmin 0)
            heads.push_back(std::move(hd));
        }
        if (on_mps) torch::mps::synchronize();   // pay the barrier once, here
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->att_heads = std::move(heads);
            st->att_hmax  = std::move(hmax);
            st->att_probe = probe_str;
            st->att_snap_layer = li;
            st->att_on_device = on_mps;
            st->att_gen++;
        }
    };

    st->set_status("training…");
    for (int64_t step = 0; step < kMaxSteps; ++step) {
        if (ctl->cancelled(ctl)) { end_metrics_run(st, run); return; }

        // A live sample every kEvalEvery steps, including a step-0 baseline (an
        // untrained net emits noise — it anchors the demo arc).
        if (step % kEvalEvery == 0) {
            if (!eval_and_sample(step)) { end_metrics_run(st, run); return; }
            snapshot_attention();   // E2 — refresh the live attention panel
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
    snapshot_attention();   // E2 — final attention state of the trained net
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

    // -----------------------------------------------------------------------
    // E2 — live per-head attention heatmaps via tensor_bridge.v1. The manual
    // attention in gpt_model.h keeps the (n_head,T,T) weight matrix reachable;
    // the worker probes the fixed excerpt and snapshots the selected layer's 4
    // heads, and the FRAME thread (the only place the bridge may be called)
    // turns each into a VIRIDIS texture. Bridge absent -> the panel says so.
    // -----------------------------------------------------------------------
    ImGui::Separator();
    if (!s_->bridge) {
        ImGui::TextDisabled(
            "attention: tensor_bridge.v1 absent (ok) — panel needs it");
    } else {
        // Layer radio row L0–L3: writes the atomic the worker reads at its next
        // eval tick (so the map updates one eval cadence after a click).
        int sel = s_->att_layer_sel.load();
        ImGui::TextUnformatted("attention layer:");
        for (int l = 0; l < 4; ++l) {
            ImGui::SameLine();
            char lbl[8];
            std::snprintf(lbl, sizeof lbl, "L%d", l);
            if (ImGui::RadioButton(lbl, sel == l)) s_->att_layer_sel.store(l);
        }

        // Read a copy of the worker's snapshot under the mutex. The vector copy
        // refcount-bumps each clone, so the storage the bridge reads stays alive
        // across the (synchronous) uploads even if the worker publishes meanwhile.
        std::vector<torch::Tensor> heads;
        std::vector<float> hmax;
        std::string probe;
        uint64_t gen = 0;
        int snap_layer = 0;
        bool on_dev = false;
        {
            std::lock_guard<std::mutex> lk(s_->mtx);
            gen = s_->att_gen;
            if (gen != 0) {
                heads = s_->att_heads;
                hmax = s_->att_hmax;
                probe = s_->att_probe;
                snap_layer = s_->att_snap_layer;
                on_dev = s_->att_on_device;
            }
        }
        if (gen == 0) {
            ImGui::TextDisabled(
                "attention: start training to watch the heads light up on a "
                "fixed val excerpt");
        } else {
            // New snapshot -> recreate the 4 maps. Each head carries a fresh
            // per-head VIRIDIS range (vmin 0, vmax per-head max); v1
            // update_texture has no range channel, so a new range means fresh
            // textures — cheap at eval cadence, and it covers a shape change for
            // free. Zero applet pixel work on either path (§6c).
            if (gen != s_->att_tex_gen) {
                s_->att_tex_on_device = on_dev;
                if (s_->att_tex.size() != heads.size()) {
                    for (auto id : s_->att_tex)
                        if (id) s_->bridge.release_texture(id);
                    s_->att_tex.assign(heads.size(), 0);
                }
                s_->att_tex_hmax.assign(heads.size(), 0.f);
                for (size_t h = 0; h < heads.size(); ++h) {
                    if (s_->att_tex[h]) s_->bridge.release_texture(s_->att_tex[h]);
                    const float vmax = hmax[h] > 0.f ? hmax[h] : 1e-6f;
                    s_->att_tex_hmax[h] = vmax;
                    s_->att_tex[h] =
                        upload_mapped(s_->bridge, s_->att_stage_cpu, 0, heads[h],
                                      CALIPER_CMAP_VIRIDIS, 0.f, vmax);
                }
                s_->att_tex_gen = gen;
            }

            const int T = (int)probe.size();
            ImGui::Text("layer %d — 4 heads (VIRIDIS, per-head vmax; rows attend "
                        "to cols)", snap_layer);

            // 2x2 grid of ~140px cells with head captions. Hover computes the
            // (row=source, col=target) index into the probe from the mouse's
            // position within the image rect; all heads share the same indexing.
            int hover_row = -1, hover_col = -1;
            const float cell = 140.f;
            for (size_t h = 0; h < s_->att_tex.size(); ++h) {
                if (h % 2 != 0) ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::TextDisabled("head %zu", h);
                ImGui::Image(caliper::Bridge::imtex(s_->att_tex[h]),
                             ImVec2(cell, cell));
                if (T > 0 && ImGui::IsItemHovered()) {
                    const ImVec2 mn = ImGui::GetItemRectMin();
                    const ImVec2 sz = ImGui::GetItemRectSize();
                    const ImVec2 mp = ImGui::GetIO().MousePos;
                    const int col = (int)((mp.x - mn.x) / sz.x * (float)T);
                    const int row = (int)((mp.y - mn.y) / sz.y * (float)T);
                    hover_row = std::clamp(row, 0, T - 1);
                    hover_col = std::clamp(col, 0, T - 1);
                }
                ImGui::EndGroup();
            }

            // The probe excerpt, per-char highlighted: the hovered cell's ROW is
            // the SOURCE (attending) char, its COL the TARGET (attended) char —
            // the touch that makes attention legible. Control chars render as a
            // space so per-char indices stay aligned with the (T,T) map.
            const ImVec4 src{0.35f, 0.85f, 1.00f, 1.f};   // cyan  = source (row)
            const ImVec4 tgt{1.00f, 0.65f, 0.25f, 1.f};   // amber = target (col)
            const ImVec4 both{0.55f, 1.00f, 0.55f, 1.f};  // green = both
            const ImVec4 def = ImGui::GetStyleColorVec4(ImGuiCol_Text);
            ImGui::TextDisabled("hover a map — highlighted:");
            ImGui::SameLine();
            ImGui::TextColored(src, "source (row, attending)");
            ImGui::SameLine();
            ImGui::TextColored(tgt, "target (col, attended)");
            for (int i = 0; i < T; ++i) {
                const char c = probe[(size_t)i];
                const char buf[2] = {
                    (c == '\n' || c == '\t' || c == '\r') ? ' ' : c, 0};
                ImVec4 colr = def;
                const bool isRow = (i == hover_row), isCol = (i == hover_col);
                if (isRow && isCol) colr = both;
                else if (isRow)     colr = src;
                else if (isCol)     colr = tgt;
                if (i) ImGui::SameLine(0, 0);
                ImGui::TextColored(colr, "%s", buf);
            }
            ImGui::TextDisabled(
                "attention maps: %s",
                (s_->att_tex_on_device && !s_->att_stage_cpu)
                    ? "GPU-resident (Metal, zero CPU staging)"
                    : "CPU-staged (GL fallback)");
        }
    }

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
    // E2 — the attention textures are frame-thread-owned. Release them AFTER the
    // job wait above (the worker never touched the bridge, and the frame loop is
    // stopped by the time cleanup() runs, so nothing races this) and BEFORE the
    // host tears the renderer down.
    for (auto id : s_->att_tex)
        if (id) s_->bridge.release_texture(id);
    s_->att_tex.clear();
    // Pairs with the initialize() curl_global_init; safe only once the worker
    // (the sole curl user) has exited, which the bounded wait above ensures.
    curl_global_cleanup();
    if (s_->host) s_->host->log_info("gpt-scope: cleanup");
}

} // namespace gptscope
