// ============================================================================
// GPTScope 2 — mechanistic insight into a live-training char GPT
// (id dev.caliper.gpt-scope, version 0.2.0).
//
// A 4L/4H/128d char transformer trained on TinyShakespeare, off the frame
// thread via caliper.jobs.v1. Six docked windows, each answering ONE named
// question about the architecture or the data — not "visual slop" attention
// wallpaper:
//
//   Logit Lens  — "when does the model decide?"   ln_f+unembed at every depth
//   Heads       — "what did each head become?"    distance/entropy scatter + drill
//   Embeddings  — "what has it learned about chars?"  W_E PCA, glyph point cloud
//   Residual    — "who writes what, where?"       attn-vs-MLP write norms + grads
//   Sample      — "what does it believe as it speaks?"  confidence-colored text
//   Training    — controls, loss + val perplexity, metrics + Save/Load
//
// The engine is the ml-applet cookbook's threading spine: one worker computes
// and publishes generations under one mutex; the frame thread pulls the latest
// and renders. Streams update at the rate their nature dictates (§2): loss and
// grad norms every step; the probe bundle (lens/heads/residual) ~1 Hz; the
// embedding PCA ~5 s; the live sample ~2 s — all time-gated IN THE WORKER. The
// bridge (head drill-down heatmap) and data.v1 (unused) degrade to a visible
// "absent (ok)" line. Cancel is honored per training batch, per eval batch, and
// inside the sampling loop (§7).
// ============================================================================
#include "gpt_model.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>   // torch::Tensor -> CaliperTensor (heatmap)
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <torch/torch.h>

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace gptscope {
namespace {

// --- fixed training recipe -------------------------------------------------
constexpr int    kMaxSteps   = 3000;
constexpr int    kBatch      = 64;
constexpr int    kBlock      = 128;    // == GPTConfig::block_size
constexpr int    kEvalEvery  = 100;    // val loss + perplexity this often
constexpr int    kValBatches = 20;     // batches averaged for the val point
constexpr int    kSampleLen  = 240;    // chars per live sample
constexpr double kLR         = 3e-4;   // AdamW
constexpr int    kProbeLen   = 48;     // chars in the fixed mechanistic probe
constexpr double kProbeHz    = 1.0;    // probe bundle cadence (lens/heads/resid)
constexpr double kPcaSec     = 5.0;    // embedding PCA cadence
constexpr double kSampleSec  = 2.0;    // auto-sample cadence
constexpr const char* kModelName = "gptscope-model";

// TinyShakespeare — a single plain-text file (no gunzip), cached as
// <data_dir>/tinyshakespeare.txt. Verbatim recipe/URL from the archived v0.1.0.
constexpr const char* kDataUrl =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt";
constexpr const char* kDataFile = "tinyshakespeare.txt";

// The six grad-norm groups: one per block, plus embedding and head as their own.
constexpr int kGroups = 6;
const char* kGroupName[kGroups] = {"blocks.0", "blocks.1", "blocks.2",
                                   "blocks.3", "embed",    "head"};
const ImU32 kGroupCol[kGroups] = {
    IM_COL32( 66,133,244,255), IM_COL32( 52,168, 83,255),
    IM_COL32(251,188,  5,255), IM_COL32(234, 67, 53,255),
    IM_COL32(163, 96,247,255), IM_COL32(120,144,156,255)};

// One color per layer for the Heads scatter.
const ImU32 kLayerCol[4] = {
    IM_COL32( 66,133,244,255), IM_COL32( 52,168, 83,255),
    IM_COL32(251,188,  5,255), IM_COL32(234, 67, 53,255)};

// Character classes for the embedding cloud (the model's emergent phonotactics).
enum CharClass { CC_VOWEL, CC_CONSON, CC_DIGIT, CC_PUNCT, CC_UPPER };
const ImU32 kClassCol[5] = {
    IM_COL32(234, 67, 53,255),   // vowel     — red
    IM_COL32( 66,133,244,255),   // consonant — blue
    IM_COL32(251,188,  5,255),   // digit     — amber
    IM_COL32(120,144,156,255),   // punct     — slate
    IM_COL32( 52,168, 83,255)};  // uppercase — green
const char* kClassName[5] = {"vowel", "consonant", "digit", "punct", "uppercase"};

CharClass classify_char(char c) {
    unsigned char u = (unsigned char)c;
    if (u >= 'A' && u <= 'Z') return CC_UPPER;
    if (u >= '0' && u <= '9') return CC_DIGIT;
    char l = (char)std::tolower(u);
    if (l == 'a' || l == 'e' || l == 'i' || l == 'o' || l == 'u') return CC_VOWEL;
    if (l >= 'a' && l <= 'z') return CC_CONSON;
    return CC_PUNCT;
}

// A printable glyph for the embedding label: space and newline get visible marks.
std::string glyph_of(char c) {
    if (c == ' ')  return "\xE2\x90\xA3";  // ␣  (open box)
    if (c == '\n') return "\xE2\x8F\x8E";  // ⏎  (return symbol)
    if (c == '\t') return "\xE2\x87\xA5";  // ⇥
    if (c == '\r') return "\xC2\xB6";      // ¶
    return std::string(1, c);
}

size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* ud) {
    auto* s = static_cast<std::string*>(ud);
    const size_t n = size * nmemb;
    s->append(ptr, n);
    return n;
}

int grad_group(const std::string& name) {
    if (name.rfind("blocks.", 0) == 0 && name.size() > 7) {
        int n = name[7] - '0';
        return (n >= 0 && n < 4) ? n : -1;
    }
    if (name.rfind("wte", 0) == 0 || name.rfind("wpe", 0) == 0) return 4;  // embed
    if (name.rfind("lm_head", 0) == 0 || name.rfind("ln_f", 0) == 0) return 5;
    return -1;                                                             // head
}

} // namespace

// ---------------------------------------------------------------------------
// pImpl state — everything heavy the header hides. All cross-thread fields live
// under `mtx`; the frame-thread-only block (textures, plot splits) never locks.
// ---------------------------------------------------------------------------
struct GPTScopeState {
    caliper::Host*     host = nullptr;
    caliper::Jobs      jobs;
    caliper::Device    device;
    caliper::Metrics   metrics;    // optional
    caliper::Bridge    bridge;     // optional (head drill-down heatmap)
    caliper::Artifacts artifacts;  // optional (Save/Load)

    GPT model{nullptr};            // persistent so Save/Load reaches it
    std::atomic<uint64_t> run_id{0};
    uint64_t           job_id = 0;

    // frame -> worker controls
    std::atomic<float> temp{0.8f};
    std::atomic<bool>  auto_sample{true};
    std::atomic<bool>  sample_now{false};

    std::mutex mtx;   // guards everything from here to the frame-thread block

    // Training curves.
    std::vector<float> loss_x, loss_y;      // train cross-entropy
    std::vector<float> val_x, val_y;        // val cross-entropy (perplexity = exp)
    std::string status = "idle — press Train to download TinyShakespeare + learn";
    int vocab_size = 0;

    // Per-step grad norms per group (blocks 0..3, embed, head).
    std::vector<float>              grad_x;
    std::array<std::vector<float>, kGroups> grad_y;

    // Probe bundle (~1 Hz). Logit lens: rows = depth (emb, L1..L4), cols =
    // predictive position (0..T-2). Owned flat CPU arrays.
    int lens_depths = 0, lens_T = 0;        // lens_T = predictive positions
    std::vector<int>   lens_top1;           // depths*lens_T char ids
    std::vector<float> lens_pcorrect;       // depths*lens_T p(correct char)
    std::vector<int>   lens_target;         // lens_T actual next-char ids
    std::string        probe_text;          // T chars
    std::vector<char>  itos_snap;           // id -> char (for lens/heads decode)

    // Heads scatter: one point per head (16 = 4 layers x 4 heads).
    std::vector<float> head_dist, head_ent; // mean attended distance / entropy
    std::vector<int>   head_layer;          // layer of each head
    // Per-head (T,T) attention for the drill-down heatmap (owned; on train dev).
    std::vector<torch::Tensor> attn_all;    // 16 x (T,T)
    std::vector<float>         attn_vmax;   // per-head max (MAGMA vmax; vmin 0)
    bool     probe_on_device = false;
    uint64_t probe_gen = 0;                 // bumped per probe bundle (0 = none)

    // Residual accounting: per-layer sublayer write norms.
    std::vector<float> attn_wnorm, mlp_wnorm;   // per layer (n_layer)

    // Embedding PCA (~5 s).
    std::vector<float> pca_x, pca_y, pca_z;     // vocab coords (top-3 PCs)
    std::string        pca_chars;               // vocab chars aligned to coords
    uint64_t           pca_gen = 0;

    // Live sample (~2 s / on demand), confidence-colored.
    std::string        sample_text;
    std::vector<float> sample_prob;             // per-char sampled probability
    std::vector<float> top8_p;                  // last-position top-8 probs
    std::vector<int>   top8_id;                 // last-position top-8 char ids
    uint64_t           sample_gen = 0;

    // ------- frame-thread-only -------
    std::string load_path;                      // set before submitting Load job
    std::string save_status;

    bool   follow_curves = true;
    // Embedding cloud fixed-axes machinery (EmbedScope's policy).
    uint64_t emb_plot_gen = 0;
    bool   emb_refit = false, emb_autofit = false;
    double bmin[3] = {-1,-1,-1}, bmax[3] = {1,1,1};

    // Head drill-down selection + its texture.
    int sel_layer = 0, sel_head = 0;
    CaliperTextureId head_tex = 0;
    uint64_t head_tex_gen = 0;
    int      head_tex_sel = -1;
    float    head_tex_vmax = 0.f;
    bool     head_stage_cpu = false;
};

namespace {

void set_status(GPTScopeState* st, const std::string& s) {
    std::lock_guard<std::mutex> lk(st->mtx);
    st->status = s;
}

// ---- data acquisition (inside the job) ------------------------------------
struct XferCtx { const CaliperJobControl* ctl; };
int xferinfo(void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* x = static_cast<XferCtx*>(p);
    return (x->ctl && x->ctl->cancelled(x->ctl)) ? 1 : 0;
}

// Resolve the corpus path: prefer our own data_dir (which for id
// dev.caliper.gpt-scope IS the archived v0.1.0's cache dir — same id, same
// dir), else a sibling dir under the same id (defensive, EmbedScope's pattern).
std::string corpus_path(GPTScopeState* st) {
    namespace fs = std::filesystem;
    std::string dir = st->host ? st->host->data_dir() : "";
    if (dir.empty()) return "";
    std::error_code ec;
    fs::path mine = fs::path(dir) / kDataFile;
    if (fs::exists(mine, ec)) return mine.string();
    fs::path sib = fs::path(dir).parent_path() / "dev.caliper.gpt-scope" / kDataFile;
    if (fs::exists(sib, ec)) return sib.string();
    return "";
}

// Download once into data_dir, cache forever, cancellable, atomic .tmp+rename,
// self-healing. Returns the corpus text or nullopt (offline/cancel/empty).
std::optional<std::string> ensure_corpus(GPTScopeState* st,
                                         const CaliperJobControl* ctl) {
    const std::string cached = corpus_path(st);
    if (!cached.empty()) {
        std::ifstream f(cached, std::ios::binary);
        if (f.good()) {
            std::string text((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
            if (!text.empty()) return text;
        }
    }
    if (ctl->cancelled(ctl)) return std::nullopt;

    const std::string dir  = st->host ? st->host->data_dir() : "";
    const std::string path = dir + "/" + kDataFile;
    set_status(st, "downloading TinyShakespeare…");
    ctl->progress(ctl, 0.f, "downloading TinyShakespeare");

    std::string body;
    XferCtx xc{ctl};
    CURL* c = curl_easy_init();
    if (!c) { set_status(st, "download failed (curl init)"); return std::nullopt; }
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
        set_status(st, "TinyShakespeare download failed (offline?) — "
                       "press Train to retry");
        ctl->progress(ctl, 0.f, "download failed (offline?)");
        if (st->host) st->host->log_error("gpt-scope: corpus download failed");
        return std::nullopt;
    }
    if (body.empty()) {
        set_status(st, "download returned an empty file — press Train to retry");
        return std::nullopt;
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) { set_status(st, "cache write failed"); return std::nullopt; }
        out.write(body.data(), (std::streamsize)body.size());
        out.flush();
        if (!out.good()) { out.close(); std::remove(tmp.c_str());
                           set_status(st, "cache write failed"); return std::nullopt; }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        set_status(st, "cache rename failed");
        return std::nullopt;
    }
    return body;
}

// Build the char vocab from the corpus: sorted unique bytes -> stoi/itos.
void build_vocab(const std::string& text, std::vector<char>& itos,
                 std::map<char, int64_t>& stoi) {
    std::vector<bool> seen(256, false);
    for (unsigned char ch : text) seen[ch] = true;
    for (int i = 0; i < 256; ++i)
        if (seen[i]) itos.push_back((char)i);
    for (int64_t i = 0; i < (int64_t)itos.size(); ++i) stoi[itos[i]] = i;
}

torch::Device pick_device(GPTScopeState* st) {
    if (st->device.kind == CALIPER_DEV_CUDA && torch::cuda::is_available())
        return torch::Device(torch::kCUDA);
    return (st->device.kind == CALIPER_DEV_METAL && torch::hasMPS())
               ? torch::Device(torch::kMPS)
               : torch::Device(torch::kCPU);
}

// Shared mapped-texture upload with the GL relocate-to-CPU fallback (archived
// upload_mapped). id==0 -> create; else update in place.
CaliperTextureId upload_mapped(const caliper::Bridge& bridge, bool& stage_cpu,
                               CaliperTextureId id, const torch::Tensor& dev_t,
                               int32_t cmap, float vmin, float vmax) {
    torch::Tensor host_t;
    auto view = [&](bool cpu) -> std::optional<CaliperTensor> {
        if (cpu) { host_t = dev_t.to(torch::kCPU);
                   return caliper::adapters::to_tensor(host_t); }
        return caliper::adapters::to_tensor(dev_t);
    };
    auto ct = view(stage_cpu);
    if (!ct) return id;
    if (id != 0) { bridge.update_texture(id, &*ct); return id; }
    id = bridge.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
    if (id == 0 && !stage_cpu) {
        stage_cpu = true;
        ct = view(true);
        if (!ct) return 0;
        id = bridge.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax, 0);
    }
    return id;
}

// ===========================================================================
// The mechanistic probe bundle: one forward_full over the fixed probe, from
// which the logit lens, head roles, and residual writes are all derived. Owned
// CPU copies published under the mutex + a bumped generation the frame diffs.
// (Everything here is tiny: T<=48, V~65, 16 heads.)
// ===========================================================================
void publish_probe(GPTScopeState* st, GPT& model, const torch::Tensor& probe_tok,
                   const std::string& probe_str, const std::vector<char>& itos,
                   bool on_mps) {
    torch::NoGradGuard ng;
    const int64_t V  = (int64_t)itos.size();
    const int64_t T  = probe_tok.size(1);
    if (T < 2) return;
    auto ff = model->forward_full(probe_tok);           // resid/attn/write norms

    // ---- Logit lens: ln_f + unembed at every depth ----
    // logits_d = lm_head(ln_f(resid_d))  — the model's OWN final norm + head.
    const int D  = (int)ff.resid.size();                // n_layer + 1
    const int PT = (int)T - 1;                           // predictive positions
    std::vector<int>   top1((size_t)D * PT);
    std::vector<float> pcorr((size_t)D * PT);
    std::vector<int>   target((size_t)PT);
    // targets: the actual next char at each position.
    auto tok_cpu = probe_tok.to(torch::kCPU).contiguous();
    const int64_t* tp = tok_cpu.data_ptr<int64_t>();
    for (int p = 0; p < PT; ++p) target[(size_t)p] = (int)tp[p + 1];

    for (int d = 0; d < D; ++d) {
        auto logits = model->lm_head()->forward(
            model->ln_f()->forward(ff.resid[(size_t)d]));  // (T,V)
        auto probs = torch::softmax(logits, -1);
        auto pred  = probs.argmax(-1);                     // (T,)
        auto pc    = probs.to(torch::kCPU).contiguous();
        auto pr    = pred.to(torch::kCPU).contiguous();
        const float*   pcp = pc.data_ptr<float>();
        const int64_t* prp = pr.data_ptr<int64_t>();
        for (int p = 0; p < PT; ++p) {
            top1[(size_t)d * PT + p]  = (int)prp[p];
            const int tgt = target[(size_t)p];
            pcorr[(size_t)d * PT + p] =
                (tgt >= 0 && tgt < V) ? pcp[(int64_t)p * V + tgt] : 0.f;
        }
    }

    // ---- Head roles: mean attended distance + entropy, per head ----
    const int L  = (int)ff.attn.size();
    std::vector<float> hdist, hent;
    std::vector<int>   hlayer;
    std::vector<torch::Tensor> heads;
    std::vector<float> hvmax;
    // distance/entropy averaged over queries i, over the causal support j<=i.
    auto ar = torch::arange(T, torch::TensorOptions(torch::kFloat32));  // (T,) CPU
    for (int l = 0; l < L; ++l) {
        auto att = ff.attn[(size_t)l].to(torch::kCPU).to(torch::kFloat32); // (H,T,T)
        const int64_t H = att.size(0);
        // signed offset (i - j): rows i, cols j.
        auto off = ar.unsqueeze(1) - ar.unsqueeze(0);        // (T,T) = i - j
        for (int64_t h = 0; h < H; ++h) {
            auto p = att[h];                                  // (T,T) row-normalized
            auto dist = (p * off).sum(1).mean().item<float>();          // scalar
            auto ent  = -(p * (p + 1e-9f).log()).sum(1).mean().item<float>();
            hdist.push_back(dist);
            hent.push_back(ent);
            hlayer.push_back(l);
        }
    }
    // Owned (T,T) clones for the drill-down heatmap (on train device, offset-0).
    for (int l = 0; l < L; ++l) {
        auto layer_att = ff.attn[(size_t)l];                 // (H,T,T) on device
        const int64_t H = layer_att.size(0);
        for (int64_t h = 0; h < H; ++h) {
            auto hd = layer_att[h].clone();                  // (T,T) owned contig
            hvmax.push_back(hd.max().item<float>());
            heads.push_back(std::move(hd));
        }
    }
    if (on_mps) torch::mps::synchronize();

    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->lens_depths = D; st->lens_T = PT;
        st->lens_top1 = std::move(top1);
        st->lens_pcorrect = std::move(pcorr);
        st->lens_target = std::move(target);
        st->probe_text = probe_str;
        st->itos_snap = itos;
        st->head_dist = std::move(hdist);
        st->head_ent  = std::move(hent);
        st->head_layer = std::move(hlayer);
        st->attn_all = std::move(heads);
        st->attn_vmax = std::move(hvmax);
        st->attn_wnorm = ff.attn_wnorm;
        st->mlp_wnorm  = ff.mlp_wnorm;
        st->probe_on_device = on_mps;
        st->probe_gen++;
    }
}

// Embedding PCA: SVD of the centered W_E on CPU -> top-3 PC scores per char.
void publish_pca(GPTScopeState* st, GPT& model, const std::vector<char>& itos) {
    torch::NoGradGuard ng;
    auto W = model->wte()->weight.detach().to(torch::kCPU).to(torch::kFloat32);
    if (W.size(0) < 3) return;
    auto Wc = W - W.mean(0, /*keepdim=*/true);
    auto svd = torch::linalg_svd(Wc, /*full_matrices=*/false);
    auto U = std::get<0>(svd);                    // (V, k)
    auto S = std::get<1>(svd);                    // (k,)
    auto coords = (U.slice(1, 0, 3) * S.slice(0, 0, 3)).contiguous();  // (V,3)
    const float* cp = coords.data_ptr<float>();
    const int64_t V = coords.size(0);
    std::vector<float> x(V), y(V), z(V);
    std::string chars;
    chars.reserve((size_t)V);
    for (int64_t i = 0; i < V; ++i) {
        x[i] = cp[i * 3 + 0]; y[i] = cp[i * 3 + 1]; z[i] = cp[i * 3 + 2];
        chars.push_back(i < (int64_t)itos.size() ? itos[(size_t)i] : '?');
    }
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pca_x = std::move(x); st->pca_y = std::move(y); st->pca_z = std::move(z);
        st->pca_chars = std::move(chars);
        st->pca_gen++;
    }
}

// Confidence-colored sampling: a custom loop (not model->generate) that records
// each sampled token's probability, plus the last position's top-8. Cancel is
// checked every step (§7). Seeded from a newline.
void publish_sample(GPTScopeState* st, GPT& model, const std::vector<char>& itos,
                    const std::map<char, int64_t>& stoi, const torch::Device& dev,
                    const CaliperJobControl* ctl, double temp) {
    torch::NoGradGuard ng;
    const bool was_training = model->is_training();
    model->eval();
    const int64_t V = (int64_t)itos.size();
    const int64_t block = model->config().block_size;
    auto sit = stoi.find('\n');
    const int64_t seed = sit != stoi.end() ? sit->second : 0;
    auto idx = torch::full({1, 1}, seed,
                           torch::TensorOptions(dev).dtype(torch::kLong));
    std::string out;
    std::vector<float> probs;
    std::vector<float> last8p; std::vector<int> last8id;
    out.reserve(kSampleLen); probs.reserve(kSampleLen);
    for (int i = 0; i < kSampleLen; ++i) {
        if (ctl->cancelled(ctl)) { if (was_training) model->train(); return; }
        auto cond = idx.size(1) <= block ? idx
                                         : idx.slice(1, idx.size(1) - block);
        auto logits = model->forward(cond);
        auto last = logits.select(1, logits.size(1) - 1)          // (1,V)
                    / std::max(temp, 1e-6);
        auto p = torch::softmax(last, -1);                         // (1,V)
        auto next = torch::multinomial(p, 1);                      // (1,1)
        const int64_t nid = next.item<int64_t>();
        float sp = p.select(0, 0).index({next.item<int64_t>()}).item<float>();
        if (nid >= 0 && nid < V) { out.push_back(itos[(size_t)nid]); }
        probs.push_back(sp);
        if (i == kSampleLen - 1) {
            auto topk = p.select(0, 0).topk(std::min<int64_t>(8, V));
            auto tv = std::get<0>(topk).to(torch::kCPU).contiguous();
            auto ti = std::get<1>(topk).to(torch::kCPU).contiguous();
            const float* tvp = tv.data_ptr<float>();
            const int64_t* tip = ti.data_ptr<int64_t>();
            for (int64_t j = 0; j < tv.size(0); ++j) {
                last8p.push_back(tvp[j]); last8id.push_back((int)tip[j]);
            }
        }
        idx = torch::cat({idx, next}, 1);
    }
    if (was_training) model->train();
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->sample_text = std::move(out);
        st->sample_prob = std::move(probs);
        st->top8_p = std::move(last8p); st->top8_id = std::move(last8id);
        st->sample_gen++;
    }
}

void end_metrics_run(GPTScopeState* st, uint64_t run) {
    if (run != 0) st->metrics.end_run(run);
    st->run_id.store(0);
}

// ===========================================================================
// The training job (jobs.v1). Cadences (§2): loss + grad norms every step; the
// probe bundle ~1 Hz; the PCA ~5 s; the sample ~2 s / on demand — all time-gated
// here with steady_clock. Cancel is checked per batch, per eval batch, and in
// the sampling loop.
// ===========================================================================
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<GPTScopeState*>(user);
    const torch::Device dev = pick_device(st);
    const bool on_mps = dev.type() == torch::kMPS;

    auto corpus = ensure_corpus(st, ctl);
    if (!corpus) return;
    if (ctl->cancelled(ctl)) return;

    set_status(st, "building char vocabulary…");
    const std::string& text = *corpus;
    std::vector<char> itos;
    std::map<char, int64_t> stoi;
    build_vocab(text, itos, stoi);
    const int64_t V = (int64_t)itos.size();
    { std::lock_guard<std::mutex> lk(st->mtx); st->vocab_size = (int)V; }

    std::vector<int64_t> ids(text.size());
    for (size_t i = 0; i < text.size(); ++i) ids[i] = stoi[text[i]];
    const int64_t n = (int64_t)ids.size();
    const int64_t n_train = (int64_t)(0.9 * n);
    auto all = torch::from_blob(ids.data(), {n}, torch::kInt64).clone();
    auto train_ids = all.slice(0, 0, n_train).to(dev);
    auto val_ids   = all.slice(0, n_train, n).to(dev);

    // The fixed probe: the first kProbeLen chars of the validation split.
    std::string probe_str;
    torch::Tensor probe_tok;
    {
        const int64_t plen = std::min<int64_t>(kProbeLen, n - n_train);
        std::vector<int64_t> pids((size_t)std::max<int64_t>(plen, 0));
        for (int64_t i = 0; i < plen; ++i) {
            pids[(size_t)i] = ids[(size_t)(n_train + i)];
            probe_str.push_back(itos[(size_t)pids[(size_t)i]]);
        }
        if (plen > 0)
            probe_tok = torch::from_blob(pids.data(), {1, plen}, torch::kInt64)
                            .clone().to(dev);
    }

    auto get_batch = [&](const torch::Tensor& data) {
        const int64_t len = data.size(0);
        auto ix = torch::randint(0, len - kBlock - 1, {kBatch},
                                 torch::TensorOptions(dev).dtype(torch::kLong));
        auto ar = torch::arange(kBlock,
                                torch::TensorOptions(dev).dtype(torch::kLong));
        auto rows = ix.unsqueeze(1) + ar.unsqueeze(0);
        return std::make_pair(data.index({rows}), data.index({rows + 1}));
    };

    torch::manual_seed(1337);
    GPTConfig cfg; cfg.vocab_size = V;
    st->model = GPT(cfg);
    auto model = st->model;
    model->to(dev);
    torch::optim::AdamW opt(model->parameters(),
                            torch::optim::AdamWOptions(kLR));

    const uint64_t run = st->metrics.begin_run("tinyshakespeare", "gpt2-mech");
    st->run_id.store(run);
    if (run != 0)
        st->metrics.hparams_json(run,
            R"({"lr":3e-4,"batch":64,"block":128,"n_layer":4,"n_head":4,)"
            R"("n_embd":128,"dropout":0.1,"max_steps":3000})");

    auto eval_val = [&]() -> std::optional<float> {
        model->eval();
        torch::NoGradGuard ng;
        double sum = 0.0;
        for (int i = 0; i < kValBatches; ++i) {
            if (ctl->cancelled(ctl)) return std::nullopt;
            auto [xb, yb] = get_batch(val_ids);
            auto logits = model->forward(xb);
            sum += torch::nn::functional::cross_entropy(
                       logits.view({-1, V}), yb.reshape({-1})).item<double>();
        }
        model->train();
        return (float)(sum / kValBatches);
    };

    using clock = std::chrono::steady_clock;
    auto now = [] { return clock::now(); };
    auto last_probe  = clock::time_point{};   // fire immediately at step 0
    auto last_pca    = clock::time_point{};
    auto last_sample = clock::time_point{};
    auto elapsed = [&](clock::time_point t) {
        return std::chrono::duration<double>(now() - t).count();
    };

    set_status(st, "training…");
    for (int64_t step = 0; step < kMaxSteps; ++step) {
        if (ctl->cancelled(ctl)) { end_metrics_run(st, run); return; }

        // Val loss + perplexity at the eval cadence (incl. a step-0 baseline).
        if (step % kEvalEvery == 0) {
            auto vloss = eval_val();
            if (!vloss) { end_metrics_run(st, run); return; }
            { std::lock_guard<std::mutex> lk(st->mtx);
              st->val_x.push_back((float)step); st->val_y.push_back(*vloss); }
            if (run != 0) st->metrics.scalar(run, "val/loss", step, *vloss);
        }

        // Probe bundle (lens/heads/residual) ~1 Hz.
        if (probe_tok.defined() && elapsed(last_probe) >= 1.0 / kProbeHz) {
            publish_probe(st, model, probe_tok, probe_str, itos, on_mps);
            last_probe = now();
        }
        // Embedding PCA ~5 s.
        if (elapsed(last_pca) >= kPcaSec) {
            publish_pca(st, model, itos);
            last_pca = now();
        }
        // Live sample ~2 s (auto) or on demand.
        if (st->sample_now.exchange(false) ||
            (st->auto_sample.load() && elapsed(last_sample) >= kSampleSec)) {
            publish_sample(st, model, itos, stoi, dev, ctl,
                           (double)st->temp.load());
            last_sample = now();
            if (ctl->cancelled(ctl)) { end_metrics_run(st, run); return; }
        }

        // ---- the optimizer step ----
        model->train();
        auto [xb, yb] = get_batch(train_ids);
        opt.zero_grad();
        auto logits = model->forward(xb);
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, V}), yb.reshape({-1}));
        loss.backward();

        // Per-group grad norms (blocks 0..3, embed, head) — one device sync.
        auto acc = torch::zeros({kGroups}, torch::TensorOptions(dev));
        for (const auto& it : model->named_parameters()) {
            if (!it.value().grad().defined()) continue;
            const int g = grad_group(it.key());
            if (g >= 0) acc[g] += it.value().grad().detach().pow(2).sum();
        }
        auto gn = acc.sqrt().to(torch::kCPU).contiguous();
        const float* gp = gn.data_ptr<float>();

        opt.step();

        const float l = loss.item<float>();
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->loss_x.push_back((float)step); st->loss_y.push_back(l);
            st->grad_x.push_back((float)step);
            for (int g = 0; g < kGroups; ++g) st->grad_y[g].push_back(gp[g]);
        }
        if (run != 0) st->metrics.scalar(run, "train/loss", step, l);

        char msg[96];
        std::snprintf(msg, sizeof msg, "step %lld/%d  loss %.4f",
                      (long long)(step + 1), kMaxSteps, l);
        ctl->progress(ctl, (float)(step + 1) / (float)kMaxSteps, msg);
    }

    // Final probe + sample of the trained net.
    if (probe_tok.defined()) publish_probe(st, model, probe_tok, probe_str, itos, on_mps);
    publish_pca(st, model, itos);
    publish_sample(st, model, itos, stoi, dev, ctl, (double)st->temp.load());
    end_metrics_run(st, run);
    set_status(st, "training complete — Save the model, or keep sampling");
}

// The Load job (artifacts.v1): rebuild vocab + probe from the corpus, load the
// checkpoint, and run ONE probe + PCA + sample pass. NO training.
void eval_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<GPTScopeState*>(user);
    const torch::Device dev = pick_device(st);
    const bool on_mps = dev.type() == torch::kMPS;

    auto corpus = ensure_corpus(st, ctl);
    if (!corpus) return;
    if (ctl->cancelled(ctl)) return;

    std::vector<char> itos; std::map<char, int64_t> stoi;
    build_vocab(*corpus, itos, stoi);
    const int64_t V = (int64_t)itos.size();
    const std::string& text = *corpus;
    std::vector<int64_t> ids(text.size());
    for (size_t i = 0; i < text.size(); ++i) ids[i] = stoi[text[i]];
    const int64_t n = (int64_t)ids.size();
    const int64_t n_train = (int64_t)(0.9 * n);
    std::string probe_str; torch::Tensor probe_tok;
    {
        const int64_t plen = std::min<int64_t>(kProbeLen, n - n_train);
        std::vector<int64_t> pids((size_t)std::max<int64_t>(plen, 0));
        for (int64_t i = 0; i < plen; ++i) {
            pids[(size_t)i] = ids[(size_t)(n_train + i)];
            probe_str.push_back(itos[(size_t)pids[(size_t)i]]);
        }
        if (plen > 0)
            probe_tok = torch::from_blob(pids.data(), {1, plen}, torch::kInt64)
                            .clone().to(dev);
    }
    { std::lock_guard<std::mutex> lk(st->mtx); st->vocab_size = (int)V; }

    if (st->load_path.empty()) { set_status(st, "load: artifact path missing"); return; }
    try {
        GPTConfig cfg; cfg.vocab_size = V;
        st->model = GPT(cfg);
        torch::load(st->model, st->load_path);   // loads on CPU
    } catch (...) {
        set_status(st, "load: failed to deserialize checkpoint");
        if (st->host) st->host->log_error("gpt-scope: torch::load failed");
        return;
    }
    auto model = st->model;
    model->to(dev);
    if (ctl->cancelled(ctl)) return;
    if (probe_tok.defined()) publish_probe(st, model, probe_tok, probe_str, itos, on_mps);
    publish_pca(st, model, itos);
    publish_sample(st, model, itos, stoi, dev, ctl, (double)st->temp.load());
    set_status(st, "loaded checkpoint — probe/embeddings/sample restored, no training");
}

} // namespace

// ---------------------------------------------------------------------------
// Applet facade.
// ---------------------------------------------------------------------------
GPTScopeApplet::GPTScopeApplet() : s_(std::make_unique<GPTScopeState>()) {}
GPTScopeApplet::~GPTScopeApplet() = default;

bool GPTScopeApplet::initialize(caliper::Host& host) {
    s_->host      = &host;
    s_->jobs      = caliper::Jobs(host);
    s_->device    = caliper::Device::query(host);
    s_->metrics   = caliper::Metrics(host);
    s_->bridge    = caliper::Bridge(host);
    s_->artifacts = caliper::Artifacts(host);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    host.log_info("gpt-scope: on_init");
    return true;
}

namespace {

// Green if the depth's top-1 matches the actual next char; else a red->yellow
// ramp by the probability it assigned the CORRECT char (bright = closer).
ImVec4 lens_color(bool match, float pcorrect) {
    if (match) return ImVec4(0.30f, 0.90f, 0.35f, 1.f);      // green
    float t = std::clamp(pcorrect / 0.5f, 0.f, 1.f);         // 0=red .. 1=yellow
    return ImVec4(0.95f, 0.25f + 0.65f * t, 0.20f, 1.f);
}

// Confidence -> color for the sample: dim red (desperate) -> bright green.
ImVec4 conf_color(float p) {
    float t = std::clamp(p, 0.f, 1.f);
    return ImVec4(0.95f - 0.65f * t, 0.30f + 0.60f * t, 0.30f, 1.f);
}

} // namespace

void GPTScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex ----
    std::vector<float> lx, ly, vx, vy, gx;
    std::array<std::vector<float>, kGroups> gy;
    std::string status; int vocab = 0;
    int lens_D = 0, lens_T = 0;
    std::vector<int> lens_top1, lens_target; std::vector<float> lens_pc;
    std::string probe_text; std::vector<char> itos;
    std::vector<float> hdist, hent; std::vector<int> hlayer;
    std::vector<torch::Tensor> attn_all; std::vector<float> attn_vmax;
    bool probe_on_dev = false; uint64_t probe_gen = 0;
    std::vector<float> attn_wn, mlp_wn;
    std::vector<float> px, py, pz; std::string pca_chars; uint64_t pca_gen = 0;
    std::string sample_text; std::vector<float> sample_prob, top8p;
    std::vector<int> top8id; uint64_t sample_gen = 0;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        lx = st->loss_x; ly = st->loss_y; vx = st->val_x; vy = st->val_y;
        gx = st->grad_x; for (int g = 0; g < kGroups; ++g) gy[g] = st->grad_y[g];
        status = st->status; vocab = st->vocab_size;
        lens_D = st->lens_depths; lens_T = st->lens_T;
        lens_top1 = st->lens_top1; lens_pc = st->lens_pcorrect;
        lens_target = st->lens_target; probe_text = st->probe_text;
        itos = st->itos_snap;
        hdist = st->head_dist; hent = st->head_ent; hlayer = st->head_layer;
        probe_gen = st->probe_gen;
        if (probe_gen != 0) { attn_all = st->attn_all; attn_vmax = st->attn_vmax;
                              probe_on_dev = st->probe_on_device; }
        attn_wn = st->attn_wnorm; mlp_wn = st->mlp_wnorm;
        px = st->pca_x; py = st->pca_y; pz = st->pca_z;
        pca_chars = st->pca_chars; pca_gen = st->pca_gen;
        sample_text = st->sample_text; sample_prob = st->sample_prob;
        top8p = st->top8_p; top8id = st->top8_id; sample_gen = st->sample_gen;
    }
    auto decode = [&](int id) -> char {
        return (id >= 0 && id < (int)itos.size()) ? itos[(size_t)id] : '?';
    };
    auto vis = [](char c) -> char {
        return (c == '\n' || c == '\t' || c == '\r') ? ' ' : c;
    };

    // =====================================================================
    // 1. GPTScope: Logit Lens — "when does the model decide?"
    //    Rows = depth (emb, L1..L4); cols = position. Each cell = the depth's
    //    top-1 next char, GREEN if it matches the actual next char, else a
    //    red->yellow ramp by p(correct). Bottom row = the probe text itself.
    // =====================================================================
    ImGui::Begin("GPTScope: Logit Lens");
    ImGui::TextDisabled(
        "ln_f + unembed applied to the residual stream at every depth — where "
        "predictions crystallize. green = top-1 hits the true next char.");
    if (lens_D == 0 || lens_T == 0) {
        ImGui::TextDisabled("press Train — the lens fills in ~1 s once the "
                            "probe forward runs.");
    } else {
        ImGui::BeginChild("lensgrid", ImVec2(0, 0), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar);
        const char* rowlab[6] = {"emb", "L1", "L2", "L3", "L4", "L5"};
        for (int d = 0; d < lens_D; ++d) {
            ImGui::Text("%-4s", d < 6 ? rowlab[d] : "L?");
            for (int p = 0; p < lens_T; ++p) {
                const int idx = d * lens_T + p;
                if (idx >= (int)lens_top1.size()) break;
                const int t1 = lens_top1[(size_t)idx];
                const int tgt = p < (int)lens_target.size() ? lens_target[(size_t)p] : -1;
                const bool match = (t1 == tgt);
                ImGui::SameLine(0, 0);
                ImGui::TextColored(lens_color(match, lens_pc[(size_t)idx]),
                                   "%c", vis(decode(t1)));
            }
        }
        ImGui::Separator();
        ImGui::Text("%-4s", "txt");
        for (int p = 0; p < lens_T; ++p) {
            const int tgt = p < (int)lens_target.size() ? lens_target[(size_t)p] : -1;
            ImGui::SameLine(0, 0);
            ImGui::TextDisabled("%c", vis(decode(tgt)));
        }
        ImGui::EndChild();
    }
    ImGui::End();

    // =====================================================================
    // 2. GPTScope: Heads — "what did each head become?"
    //    Scatter of all 16 heads: x = mean attended distance Σ p(i,j)(i-j),
    //    y = attention entropy -Σ p log p (both averaged over queries), colored
    //    by layer. Click a point -> that head's (T,T) attention as a MAGMA
    //    bridge texture (the heatmap redeemed: on demand, with a question).
    // =====================================================================
    ImGui::Begin("GPTScope: Heads");
    ImGui::TextDisabled(
        "each point is one head. x = mean attended distance (i-j), y = "
        "attention entropy. watch heads differentiate: local heads fall to the "
        "left, diffuse heads sit high.");
    if (hdist.empty()) {
        ImGui::TextDisabled("press Train — the 16 heads appear once the probe runs.");
    } else {
        if (ImPlot::BeginPlot("##heads", ImVec2(-1, 240))) {
            // Fixed, input-locked frame (viewport policy #6): the POINTS move
            // as heads specialize; the camera must not. Bounds = the metric
            // ranges: distance in [0, T), entropy in [0, ln T].
            const double Tmax = probe_text.empty() ? 64.0
                                                   : (double)probe_text.size();
            ImPlot::SetupAxes("mean attended distance", "entropy (nats)");
            ImPlot::SetupAxesLimits(-1.0, Tmax, -0.1, std::log(Tmax) + 0.3,
                                    ImPlotCond_Always);
            const int H = 4;  // heads per layer
            for (int l = 0; l < 4; ++l) {
                std::vector<float> xs, ys;
                for (size_t i = 0; i < hlayer.size(); ++i)
                    if (hlayer[i] == l) { xs.push_back(hdist[i]); ys.push_back(hent[i]); }
                if (xs.empty()) continue;
                char lab[8]; std::snprintf(lab, sizeof lab, "L%d", l + 1);
                ImPlot::PlotScatter(lab, xs.data(), ys.data(), (int)xs.size(),
                    ImPlotSpec(ImPlotProp_Marker, ImPlotMarker_Circle,
                               ImPlotProp_MarkerSize, 6.0,
                               ImPlotProp_MarkerFillColor, kLayerCol[l],
                               ImPlotProp_MarkerLineColor, kLayerCol[l]));
            }
            // Hover/click pick: nearest head point in pixel space.
            if (ImPlot::IsPlotHovered()) {
                ImVec2 m = ImGui::GetMousePos();
                float best = 18.f * 18.f; int bh = -1;
                for (size_t i = 0; i < hdist.size(); ++i) {
                    ImVec2 sp = ImPlot::PlotToPixels(hdist[i], hent[i]);
                    float dx = sp.x - m.x, dy = sp.y - m.y, d = dx * dx + dy * dy;
                    if (d < best) { best = d; bh = (int)i; }
                }
                if (bh >= 0) {
                    ImGui::BeginTooltip();
                    ImGui::Text("layer %d, head %d", hlayer[bh] + 1, bh % H);
                    ImGui::Text("distance %.2f   entropy %.2f", hdist[bh], hent[bh]);
                    ImGui::EndTooltip();
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        st->sel_layer = hlayer[bh]; st->sel_head = bh % H;
                    }
                }
            }
            ImPlot::EndPlot();
        }

        // Drill-down: the selected head's (T,T) attention as a MAGMA texture.
        const int H = 4;
        const int sel = st->sel_layer * H + st->sel_head;
        ImGui::Text("selected: layer %d, head %d — attention (rows attend to cols)",
                    st->sel_layer + 1, st->sel_head);
        if (!st->bridge) {
            ImGui::TextDisabled("tensor_bridge absent (ok) — head heatmap needs it");
        } else if (probe_gen != 0 && sel < (int)attn_all.size()) {
            const bool need = (probe_gen != st->head_tex_gen) ||
                              (sel != st->head_tex_sel);
            if (need) {
                if (probe_on_dev != !st->head_stage_cpu) { /* handled by helper */ }
                if (st->head_tex) { st->bridge.release_texture(st->head_tex);
                                    st->head_tex = 0; }
                const float vmax = attn_vmax[(size_t)sel] > 0.f
                                       ? attn_vmax[(size_t)sel] : 1e-6f;
                st->head_tex_vmax = vmax;
                // Block-upscale before upload (cookbook #4): a raw (T,T) map
                // stretched to 260 px is linear-filter mush; hard k x k blocks
                // stay sharp. k chosen so the texture >= the drawn size.
                const auto& amap = attn_all[(size_t)sel];
                const int64_t Tt = amap.size(0);
                const int64_t kk = std::max<int64_t>(1, (320 + Tt - 1) / Tt);
                auto blocks = amap.repeat_interleave(kk, 0)
                                  .repeat_interleave(kk, 1).contiguous();
                st->head_tex = upload_mapped(st->bridge, st->head_stage_cpu, 0,
                                             blocks,
                                             CALIPER_CMAP_MAGMA, 0.f, vmax);
                st->head_tex_gen = probe_gen; st->head_tex_sel = sel;
            }
            const int T = (int)probe_text.size();
            int hover_row = -1, hover_col = -1;
            const float side = 260.f;
            if (st->head_tex) {
                ImGui::Image(caliper::Bridge::imtex(st->head_tex), ImVec2(side, side));
                if (T > 0 && ImGui::IsItemHovered()) {
                    const ImVec2 mn = ImGui::GetItemRectMin();
                    const ImVec2 sz = ImGui::GetItemRectSize();
                    const ImVec2 mp = ImGui::GetIO().MousePos;
                    hover_col = std::clamp((int)((mp.x - mn.x) / sz.x * (float)T), 0, T - 1);
                    hover_row = std::clamp((int)((mp.y - mn.y) / sz.y * (float)T), 0, T - 1);
                }
            }
            // The probe on both axes: hovered cell's row = source (attending),
            // col = target (attended). Makes the (T,T) map legible.
            const ImVec4 srcc{0.35f, 0.85f, 1.f, 1.f}, tgtc{1.f, 0.65f, 0.25f, 1.f};
            const ImVec4 both{0.55f, 1.f, 0.55f, 1.f};
            const ImVec4 def = ImGui::GetStyleColorVec4(ImGuiCol_Text);
            for (int i = 0; i < T; ++i) {
                ImVec4 c = def;
                const bool r = (i == hover_row), q = (i == hover_col);
                if (r && q) c = both; else if (r) c = srcc; else if (q) c = tgtc;
                if (i) ImGui::SameLine(0, 0);
                ImGui::TextColored(c, "%c", vis(probe_text[(size_t)i]));
            }
            ImGui::TextDisabled("heatmap: %s",
                (probe_on_dev && !st->head_stage_cpu)
                    ? "GPU-resident (Metal, zero CPU staging)"
                    : "CPU-staged (GL fallback)");
        } else {
            ImGui::TextDisabled("waiting for the first probe…");
        }
    }
    ImGui::End();

    // =====================================================================
    // 3. GPTScope: Embeddings — "what has it learned about characters?"
    //    PCA of W_E to 3-D, each point drawn AS its glyph, colored by class.
    //    Fixed axes after first fit (EmbedScope's policy) + Refit button.
    // =====================================================================
    ImGui::Begin("GPTScope: Embeddings");
    ImGui::TextDisabled(
        "top-3 PCs of the token embedding W_E. each point is a character glyph. "
        "watch vowels cluster, case pair up, punctuation exile itself.");
    for (int c = 0; c < 5; ++c) {
        if (c) ImGui::SameLine();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kClassCol[c]), "%s",
                           kClassName[c]);
    }
    // Recompute fixed bounds when a new PCA arrived.
    if (pca_gen != 0 && pca_gen != st->emb_plot_gen) {
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (size_t i = 0; i < px.size(); ++i) {
            float v[3] = {px[i], py[i], pz[i]};
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], (double)v[k]);
                                          hi[k] = std::max(hi[k], (double)v[k]); }
        }
        for (int k = 0; k < 3; ++k) {
            if (hi[k] <= lo[k]) { lo[k] -= 1; hi[k] += 1; }
            double pad = 0.10 * (hi[k] - lo[k]);
            st->bmin[k] = lo[k] - pad; st->bmax[k] = hi[k] + pad;
        }
        st->emb_refit = (st->emb_plot_gen == 0) || st->emb_autofit;
        st->emb_plot_gen = pca_gen;
    }
    if (pca_gen == 0) {
        ImGui::TextDisabled("press Train — the embedding cloud appears in ~5 s.");
    } else {
        ImGui::Checkbox("auto-fit axes", &st->emb_autofit);
        ImGui::SameLine();
        if (ImGui::Button("Refit")) st->emb_refit = true;
        ImGui::SameLine();
        ImGui::TextDisabled("fixed axes show the geometry settle");
        // Lock pan/zoom/menus (they drift the framing and hide the geometry
        // settling) — but keep ROTATE: orbiting is how you read a 3-D cloud,
        // and it's non-destructive. Axes also Lock'd so nothing rescales them.
        const ImPlot3DFlags kNoDrift = ImPlot3DFlags_NoPan |
                                       ImPlot3DFlags_NoZoom |
                                       ImPlot3DFlags_NoMenus;
        if (ImPlot3D::BeginPlot("##emb", ImVec2(-1, -1), kNoDrift)) {
            ImPlot3D::SetupAxes("PC1", "PC2", "PC3",
                                ImPlot3DAxisFlags_Lock,
                                ImPlot3DAxisFlags_Lock,
                                ImPlot3DAxisFlags_Lock);
            ImPlot3D::SetupAxesLimits(st->bmin[0], st->bmax[0], st->bmin[1],
                                      st->bmax[1], st->bmin[2], st->bmax[2],
                                      st->emb_refit ? ImPlot3DCond_Always
                                                    : ImPlot3DCond_Once);
            st->emb_refit = false;
            for (size_t i = 0; i < px.size() && i < pca_chars.size(); ++i) {
                const char ch = pca_chars[i];
                ImU32 col = kClassCol[classify_char(ch)];
                ImPlot3D::PushStyleColor(ImPlot3DCol_InlayText, col);
                ImPlot3D::PlotText(glyph_of(ch).c_str(), px[i], py[i], pz[i]);
                ImPlot3D::PopStyleColor();
            }
            ImPlot3D::EndPlot();
        }
    }
    ImGui::End();

    // =====================================================================
    // 4. GPTScope: Residual — "who writes what, where?"
    //    (a) per-layer attn-write vs MLP-write norms into the residual stream.
    //    (b) per-layer gradient-norm lines over steps (who is learning NOW).
    // =====================================================================
    ImGui::Begin("GPTScope: Residual");
    ImGui::TextDisabled("division of labor: how big a vector each sublayer adds "
                        "to the residual stream, and which groups are learning now.");
    if (!attn_wn.empty()) {
        const int L = (int)attn_wn.size();
        // Grouped bars: offset the x positions by hand (the xs/ys PlotBars
        // overload has no shift channel in this ImPlot).
        std::vector<double> xa(L), xm(L), a(L), m(L);
        for (int i = 0; i < L; ++i) {
            xa[i] = (i + 1) - 0.17; xm[i] = (i + 1) + 0.17;
            a[i] = attn_wn[i]; m[i] = mlp_wn[i];
        }
        if (ImPlot::BeginPlot("write norms per layer", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("layer", "L2 write norm",  // AutoFit = follow
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);  // + input-lock
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.5, L + 0.5, ImPlotCond_Always);
            ImPlot::SetupAxisFormat(ImAxis_X1, "%.0f");
            ImPlot::PlotBars("attention", xa.data(), a.data(), L, 0.32);
            ImPlot::PlotBars("MLP", xm.data(), m.data(), L, 0.32);
            ImPlot::EndPlot();
        }
    } else {
        ImGui::TextDisabled("write norms appear with the first probe (~1 s).");
    }
    if (!gx.empty()) {
        if (ImPlot::BeginPlot("per-group gradient norm", ImVec2(-1, 200))) {
            const ImPlotAxisFlags f =
                st->follow_curves ? ImPlotAxisFlags_AutoFit : 0;
            ImPlot::SetupAxes("step", "grad L2", f, f);
            for (int g = 0; g < kGroups; ++g) {
                ImPlot::PlotLine(kGroupName[g], gx.data(), gy[g].data(),
                                 (int)gy[g].size(),
                                 ImPlotSpec(ImPlotProp_LineColor, kGroupCol[g]));
            }
            ImPlot::EndPlot();
        }
    } else {
        ImGui::TextDisabled("gradient norms stream once training starts.");
    }
    ImGui::End();

    // =====================================================================
    // 5. GPTScope: Sample — "what does it believe as it speaks?"
    //    Confidence-colored generated text (per-char sampled prob ramp), a
    //    top-8 bar chart for the last position, and the probe text tinted by
    //    per-position loss.
    // =====================================================================
    ImGui::Begin("GPTScope: Sample");
    {
        float temp = st->temp.load();
        if (ImGui::SliderFloat("temperature", &temp, 0.2f, 1.5f, "%.2f"))
            st->temp.store(std::clamp(temp, 0.2f, 1.5f));
        bool as = st->auto_sample.load();
        if (ImGui::Checkbox("auto-sample (~2 s)", &as)) st->auto_sample.store(as);
        ImGui::SameLine();
        if (ImGui::Button("Sample Now")) st->sample_now.store(true);
    }
    ImGui::TextDisabled("each char is tinted by the probability it was sampled "
                        "with: dim red = desperate guess, bright green = confident.");
    ImGui::BeginChild("sampletext", ImVec2(0, 180), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    if (sample_gen == 0) {
        ImGui::TextDisabled("(samples appear here once training starts)");
    } else {
        const float wrap = ImGui::GetContentRegionAvail().x;
        float x0 = ImGui::GetCursorPosX();
        for (size_t i = 0; i < sample_text.size() && i < sample_prob.size(); ++i) {
            const char c = sample_text[i];
            if (c == '\n') { ImGui::NewLine(); continue; }
            char buf[2] = {c, 0};
            float w = ImGui::CalcTextSize(buf).x;
            if (ImGui::GetCursorPosX() - x0 + w > wrap) ImGui::NewLine();
            else if (i) ImGui::SameLine(0, 0);
            ImGui::TextColored(conf_color(sample_prob[i]), "%c", c);
        }
    }
    ImGui::EndChild();
    // Top-8 candidates for the last generated position.
    if (!top8p.empty()) {
        ImGui::SeparatorText("top-8 next-char candidates (last position)");
        if (ImPlot::BeginPlot("##top8", ImVec2(-1, 150))) {
            std::vector<double> xs(top8p.size()), ys(top8p.size());
            for (size_t i = 0; i < top8p.size(); ++i) { xs[i] = (double)i; ys[i] = top8p[i]; }
            ImPlot::SetupAxes("candidate", "probability",
                              ImPlotAxisFlags_AutoFit, 0);  // input-locked
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImPlotCond_Always);
            // tick labels = the actual chars
            std::vector<double> ticks(top8id.size()); std::vector<std::string> lbls;
            std::vector<const char*> lblp;
            for (size_t i = 0; i < top8id.size(); ++i) {
                ticks[i] = (double)i;
                char c = vis(decode(top8id[i]));
                lbls.push_back(std::string(1, c));
            }
            for (auto& s : lbls) lblp.push_back(s.c_str());
            ImPlot::SetupAxisTicks(ImAxis_X1, ticks.data(), (int)ticks.size(),
                                   lblp.data());
            ImPlot::PlotBars("p", xs.data(), ys.data(), (int)xs.size(), 0.6);
            ImPlot::EndPlot();
        }
    }
    // The probe text, tinted by per-position loss (final depth p(correct)).
    ImGui::SeparatorText("probe — per-position loss (redder = harder to predict)");
    if (lens_D > 0 && lens_T > 0) {
        const int last = lens_D - 1;
        for (int p = 0; p < lens_T; ++p) {
            const int idx = last * lens_T + p;
            float pc = idx < (int)lens_pc.size() ? lens_pc[(size_t)idx] : 0.f;
            const int tgt = lens_target[(size_t)p];
            // color: high p(correct) -> green, low -> red (loss = -log p).
            ImVec4 col = conf_color(pc);
            if (p) ImGui::SameLine(0, 0);
            ImGui::TextColored(col, "%c", vis(decode(tgt)));
        }
    } else {
        ImGui::TextDisabled("per-position loss appears with the first probe.");
    }
    ImGui::End();

    // =====================================================================
    // 6. GPTScope: Training — controls, loss + val perplexity, metrics, Save/Load.
    // =====================================================================
    ImGui::Begin("GPTScope: Training");
    ImGui::TextDisabled("mini-GPT (char): 4 layers · 4 heads · 128 embd · block 128");
    ImGui::TextDisabled("device: %s (%s)", st->device.name,
                        st->device.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
                        : st->device.kind == CALIPER_DEV_CUDA ? "CUDA" : "CPU");
    if (vocab > 0) ImGui::TextDisabled("vocabulary: %d chars", vocab);
    ImGui::TextWrapped("%s", status.c_str());
    if (st->metrics) {
        uint64_t run = st->run_id.load();
        if (run) ImGui::TextDisabled("metrics: run #%llu", (unsigned long long)run);
        else     ImGui::TextDisabled("metrics: present (open Runs)");
    } else ImGui::TextDisabled("metrics: absent (ok)");

    const bool running = st->job_id != 0 && st->jobs.is_running(st->job_id);
    // Dev hook: press Train on frame 1 when CALIPER_GPT_AUTOTRAIN=1.
    static bool autotrain_fired = false;
    auto start_train = [&]() {
        { std::lock_guard<std::mutex> lk(st->mtx);
          st->loss_x.clear(); st->loss_y.clear(); st->val_x.clear(); st->val_y.clear();
          st->grad_x.clear(); for (auto& v : st->grad_y) v.clear();
          st->status = "starting…"; }
        st->job_id = st->jobs.submit("gpt_scope: train mini-GPT", &train_job, st);
        if (st->job_id == 0 && st->host) st->host->log_error("gpt-scope: submit failed");
    };
    if (!autotrain_fired && !running && std::getenv("CALIPER_GPT_AUTOTRAIN")) {
        autotrain_fired = true; start_train();
    }
    if (!running) {
        if (ImGui::Button("Train")) start_train();
        ImGui::SameLine();
        const bool can_save = st->artifacts && probe_gen != 0;
        if (!can_save) ImGui::BeginDisabled();
        if (ImGui::Button("Save model")) {
            std::ostringstream oss(std::ios::binary);
            st->model->to(torch::kCPU);
            torch::save(st->model, oss);
            std::string bytes = oss.str();
            std::string dg = st->artifacts.put(kModelName, bytes.data(),
                                               bytes.size(), st->run_id.load());
            st->save_status = dg.empty() ? "save failed"
                                         : ("saved  digest " + dg.substr(0, 16) + "…");
        }
        if (!can_save) ImGui::EndDisabled();
        ImGui::SameLine();
        const bool can_load = st->artifacts && st->artifacts.exists(kModelName);
        if (!can_load) ImGui::BeginDisabled();
        if (ImGui::Button("Load model")) {
            const char* p = st->artifacts.path_of(kModelName);   // frame thread
            if (p) { st->load_path = p;
                     st->job_id = st->jobs.submit(
                         "gpt_scope: load checkpoint (probe only)", &eval_job, st);
                     st->save_status = "loading checkpoint…"; }
        }
        if (!can_load) ImGui::EndDisabled();
    } else {
        if (ImGui::Button("Cancel")) st->jobs.request_cancel(st->job_id);
        ImGui::SameLine();
        ImGui::ProgressBar(st->jobs.progress_of(st->job_id), ImVec2(-1, 0));
    }
    if (!st->artifacts)
        ImGui::TextDisabled("artifacts: absent (ok) — Save/Load need it");
    else if (!st->save_status.empty())
        ImGui::TextDisabled("artifacts: %s", st->save_status.c_str());

    ImGui::Checkbox("follow", &st->follow_curves);
    const ImPlotAxisFlags follow = st->follow_curves ? ImPlotAxisFlags_AutoFit : 0;
    if (ImPlot::BeginPlot("train / val loss", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("step", "cross-entropy", follow, follow);
        if (!ly.empty()) ImPlot::PlotLine("train", lx.data(), ly.data(), (int)ly.size());
        if (!vy.empty()) ImPlot::PlotLine("val", vx.data(), vy.data(), (int)vy.size());
        ImPlot::EndPlot();
    }
    if (!vy.empty()) {
        const float vl = vy.back();
        ImGui::Text("latest val loss %.4f   ·   val perplexity %.2f (exp val_loss)",
                    vl, std::exp(vl));
    } else {
        ImGui::TextDisabled("val perplexity: appears with the first val point");
    }
    ImGui::End();
}

void GPTScopeApplet::cleanup() {
    auto* st = s_.get();
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 1000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->head_tex) { st->bridge.release_texture(st->head_tex); st->head_tex = 0; }
    curl_global_cleanup();
    if (st->host) st->host->log_info("gpt-scope: on_cleanup");
}

} // namespace gptscope
