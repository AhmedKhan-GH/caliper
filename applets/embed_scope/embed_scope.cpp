// ============================================================================
// EmbedScope — the 3D embedding projector (Phase 2F′, Task F5).
//
// The exemplar that un-parks the last two services (D16 demand-driven clause):
//   • jobs.v1     — training runs off the frame thread, cancel honored per step.
//   • device.v1   — the host picks the device; METAL -> torch::kMPS.
//   • metrics.v1  — loss/accuracy stream to the Runs dashboard (optional).
//   • tensor_bridge.v1 — hover a 3-D point -> that digit as a texture (optional).
//   • artifacts.v1 (LOAD-BEARING) — Save serializes the module to a byte buffer
//     and put()s it content-addressed; Load resolves path_of + torch::load and
//     runs ONE eval pass (no training) — the cloud reappears across relaunches.
//   • data.v1 (honest) — each publish rebuilds a DuckDB table of the embeddings
//     and runs SQL for per-class centroids (drawn as 3-D diamonds) and the
//     misclassified count; results cross as Arrow, drained via Data::drain_numeric.
//
// The star is an ImPlot3D scatter of ~2000 test-set embeddings, one series per
// digit class (10 colors), updating live: a gray blob splits into ten lobes as
// the learned 3-D bottleneck sharpens. ImPlot3D (not raw GL) => renderer-
// agnostic (§6c): Metal by default, GL on CALIPER_RENDERER=gl.
//
// Threading discipline mirrors ml_scope: the worker publishes OWNED std::vector
// copies under a mutex (never live tensor memory); the frame reads a copy.
// tensor_bridge.v1 and data.v1 are touched ONLY on the frame thread; artifacts
// path resolution happens on the frame thread (host strings are "valid until the
// next call") and the resolved path is handed to the eval job.
// ============================================================================
#include "embed_model.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>   // torch::Tensor -> CaliperTensor (hover)
#include <torch/torch.h>

#include <curl/curl.h>

#include "mnist_idx.h"                   // reused verbatim from examples/ml_scope

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace embedscope {
namespace {

constexpr int     kEpochs    = 3;
constexpr int     kBatch     = 128;
constexpr int     kEvalEvery = 50;     // eval + publish every N training batches
constexpr int64_t kShowN     = 2000;   // test-set points drawn in the 3-D scatter
constexpr float   kPickPx    = 14.f;   // hover pick radius (screen pixels)
constexpr const char* kModelName = "embedscope-model";

// The four MNIST IDX files (host-side names in data_dir; `.gz` on the wire).
// Mirror on S3 — the classic yann.lecun.com host 403s from many networks.
const char* kFiles[4] = {
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"};
constexpr const char* kBaseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist/";

// tab10 — 10 visually distinct class colors for the scatter + centroids.
const ImU32 kClassCol[10] = {
    IM_COL32( 31,119,180,255), IM_COL32(255,127, 14,255),
    IM_COL32( 44,160, 44,255), IM_COL32(214, 39, 40,255),
    IM_COL32(148,103,189,255), IM_COL32(140, 86, 75,255),
    IM_COL32(227,119,194,255), IM_COL32(127,127,127,255),
    IM_COL32(188,189, 34,255), IM_COL32( 23,190,207,255)};

size_t write_to_vec(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::vector<uint8_t>*>(userdata);
    size_t n = size * nmemb;
    buf->insert(buf->end(), (uint8_t*)ptr, (uint8_t*)ptr + n);
    return n;
}

} // namespace

// Heavy state (pimpl behind EmbedScopeApplet). All cross-thread fields live
// under `mtx`; texture + data-panel fields are frame-thread-only.
struct EmbedScopeState {
    caliper::Host*    host = nullptr;
    caliper::Jobs     jobs;
    caliper::Device   device;
    caliper::Metrics  metrics;     // optional
    caliper::Bridge   bridge;      // optional
    caliper::Artifacts artifacts;  // optional (LOAD-BEARING when present)
    caliper::Data     data;        // optional

    EmbedNet model{nullptr};       // persistent so Save/Load can reach it
    std::atomic<uint64_t> run_id{0};
    uint64_t job_id = 0;

    // --- cross-thread (mtx) ---
    std::mutex mtx;
    std::vector<float> loss_hist;
    std::vector<float> acc_steps, acc_hist;
    std::string status = "idle — press Train to download MNIST + learn a 3-D "
                         "embedding";
    std::vector<float>   ex, ey, ez;   // snap_n embedding coordinates
    std::vector<int>     labels, preds;
    std::vector<uint8_t> pixels;       // snap_n * 28*28 u8 (for hover)
    int64_t  snap_n = 0;
    uint64_t embed_gen = 0;            // bumped per publish (0 = none yet)

    // set on the frame thread before submitting the eval (Load) job
    std::string load_path;

    // --- frame-thread-only ---
    uint64_t plot_gen = 0;             // last gen split into per-class arrays
    std::array<std::vector<float>, 10> cx, cy, cz;
    bool  refit = false;
    double bmin[3] = {-1,-1,-1}, bmax[3] = {1,1,1};

    uint64_t data_gen = 0;             // last gen fed through data.v1 SQL
    bool     centroid_valid[10] = {};
    double   cent[10][3] = {};
    int64_t  misclassified = -1, total_rows = 0;
    std::string data_status;

    CaliperTextureId hover_tex = 0;
    int hover_idx = -1;

    std::string save_status;           // last Save/Load message
};

namespace {

void set_status(EmbedScopeState* st, const std::string& s) {
    std::lock_guard<std::mutex> lk(st->mtx);
    st->status = s;
}

// Resolve one MNIST file: prefer our own data_dir, then reuse MLScope's cache
// (sibling id dir), else "" (needs download). Same data_dir recipe as ml_scope.
std::string mnist_path(EmbedScopeState* st, const char* file) {
    namespace fs = std::filesystem;
    std::string dir = st->host ? st->host->data_dir() : "";
    if (dir.empty()) return "";
    std::error_code ec;
    fs::path mine = fs::path(dir) / file;
    if (fs::exists(mine, ec)) return mine.string();
    fs::path sib = fs::path(dir).parent_path() / "dev.caliper.ml-scope" / file;
    if (fs::exists(sib, ec)) return sib.string();
    return "";
}

// Per-transfer state for curl's xferinfo callback: a cancel aborts the download.
struct XferCtx { const CaliperJobControl* ctl; };
int xferinfo(void* p, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* x = static_cast<XferCtx*>(p);
    return (x->ctl && x->ctl->cancelled(x->ctl)) ? 1 : 0;
}

void fail_dl(EmbedScopeState* st, const CaliperJobControl* ctl) {
    set_status(st, "MNIST download failed (offline?) — press Train to retry");
    ctl->progress(ctl, 0.f,
                  "MNIST download failed (offline?) — press Train to retry");
    if (st->host) st->host->log_error("embed-scope: MNIST download failed");
}

// Acquisition INSIDE the job (ml_scope's recipe): reuse cache if present, else
// download with atomic .tmp+rename and a cancellable curl transfer. Fills
// out[4] with the resolved paths. Returns false on offline/cancel/corrupt.
bool ensure_dataset(EmbedScopeState* st, const CaliperJobControl* ctl,
                    std::string out[4]) {
    namespace fs = std::filesystem;
    std::string dir = st->host ? st->host->data_dir() : "";
    for (int i = 0; i < 4; i++) {
        if (ctl->cancelled(ctl)) return false;
        std::string cached = mnist_path(st, kFiles[i]);
        if (!cached.empty()) { out[i] = cached; continue; }

        set_status(st, std::string("downloading ") + kFiles[i] + "…");
        ctl->progress(ctl, (float)i / 4.f,
                      (std::string("downloading ") + kFiles[i]).c_str());
        std::string url = std::string(kBaseUrl) + kFiles[i] + ".gz";
        std::vector<uint8_t> gz;
        XferCtx xc{ctl};
        CURL* c = curl_easy_init();
        if (!c) { fail_dl(st, ctl); return false; }
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
            fail_dl(st, ctl);
            return false;
        }
        auto raw = mnist_idx::gunzip(gz);
        if (!raw) { fail_dl(st, ctl); return false; }
        fs::path path = fs::path(dir) / kFiles[i];
        std::string tmp = path.string() + ".tmp";
        {
            std::ofstream o(tmp, std::ios::binary);
            if (!o) { fail_dl(st, ctl); return false; }
            o.write((const char*)raw->data(), (std::streamsize)raw->size());
            o.flush();
            if (!o.good()) { o.close(); std::remove(tmp.c_str());
                             fail_dl(st, ctl); return false; }
        }
        if (std::rename(tmp.c_str(), path.string().c_str()) != 0) {
            std::remove(tmp.c_str()); fail_dl(st, ctl); return false;
        }
        out[i] = path.string();
    }
    return true;
}

// Load a cached IDX pair -> X (n,1,28,28) float/255, y long. Self-heals: a
// corrupt cached file is deleted so the next Train re-downloads it.
bool load_split(const std::string& ipath, const std::string& lpath,
                torch::Tensor& X, torch::Tensor& y) {
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
        std::remove(ipath.c_str());
        std::remove(lpath.c_str());
        return false;
    }
    int n = imgs->n, r = imgs->rows, c = imgs->cols;
    X = torch::from_blob(imgs->pixels.data(), {n, 1, r, c}, torch::kUInt8)
            .to(torch::kFloat32).div_(255.0f).clone();
    std::vector<int64_t> ly(labs->begin(), labs->end());
    y = torch::from_blob(ly.data(), {n}, torch::kInt64).clone();
    return true;
}

torch::Device pick_device(EmbedScopeState* st) {
    return (st->device.kind == CALIPER_DEV_METAL && torch::hasMPS())
               ? torch::Device(torch::kMPS)
               : torch::Device(torch::kCPU);
}

// Compute the first kShowN test embeddings/preds/labels/pixels and publish OWNED
// copies under the mutex. Worker-only; NEVER touches the bridge. The .to(CPU)
// copies drain any pending MPS work implicitly.
void publish_embeddings(EmbedScopeState* st, EmbedNet& model,
                        const torch::Tensor& Xte, const torch::Tensor& yte) {
    torch::NoGradGuard ng;
    model->eval();
    int64_t n = std::min<int64_t>(kShowN, Xte.size(0));
    auto xb  = Xte.slice(0, 0, n);
    auto emb = model->embed(xb);                 // (n,3)
    auto pr  = model->fc_out->forward(emb).argmax(1);
    auto emb_c = emb.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto pr_c  = pr.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto lb_c  = yte.slice(0, 0, n).to(torch::kCPU).to(torch::kInt64).contiguous();
    auto px_c  = xb.mul(255.0f).clamp(0, 255)
                    .to(torch::kUInt8).to(torch::kCPU).contiguous();  // (n,1,28,28)

    const float*   ep = emb_c.data_ptr<float>();
    const int64_t* pp = pr_c.data_ptr<int64_t>();
    const int64_t* lp = lb_c.data_ptr<int64_t>();
    const uint8_t* xp = px_c.data_ptr<uint8_t>();

    std::vector<float>   ex(n), ey(n), ez(n);
    std::vector<int>     lbl(n), prd(n);
    std::vector<uint8_t> px((size_t)n * 28 * 28);
    for (int64_t i = 0; i < n; i++) {
        ex[i] = ep[i * 3 + 0]; ey[i] = ep[i * 3 + 1]; ez[i] = ep[i * 3 + 2];
        lbl[i] = (int)lp[i];   prd[i] = (int)pp[i];
    }
    std::memcpy(px.data(), xp, px.size());
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->ex = std::move(ex); st->ey = std::move(ey); st->ez = std::move(ez);
        st->labels = std::move(lbl); st->preds = std::move(prd);
        st->pixels = std::move(px);
        st->snap_n = n;
        st->embed_gen++;
    }
}

// The training job (jobs.v1). Cancel is checked every batch (<=100ms contract).
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<EmbedScopeState*>(user);
    torch::Device dev = pick_device(st);

    std::string paths[4];
    if (!ensure_dataset(st, ctl, paths)) return;   // offline/cancel: clean exit
    if (ctl->cancelled(ctl)) return;

    set_status(st, "parsing MNIST…");
    torch::Tensor Xtr, ytr, Xte, yte;
    if (!load_split(paths[0], paths[1], Xtr, ytr) ||
        !load_split(paths[2], paths[3], Xte, yte)) {
        set_status(st, "cached MNIST file was corrupt — press Train to re-download");
        ctl->progress(ctl, 0.f,
                      "cached MNIST file was corrupt — press Train to re-download");
        if (st->host) st->host->log_error("embed-scope: corrupt MNIST cache");
        return;
    }
    Xtr = Xtr.to(dev); ytr = ytr.to(dev);
    Xte = Xte.to(dev); yte = yte.to(dev);

    torch::manual_seed(7);
    st->model = EmbedNet();      // fresh random init — the blob->lobes is the show
    auto model = st->model;
    model->to(dev);
    torch::optim::Adam opt(model->parameters(),
                           torch::optim::AdamOptions(1e-3));

    uint64_t run = st->metrics.begin_run("mnist-embed", "embed3d");
    st->run_id.store(run);
    if (run != 0)
        st->metrics.hparams_json(run,
            R"({"lr":0.001,"batch":128,"epochs":3,"model":"conv8-16-fc64-emb3"})");

    const int64_t n = Xtr.size(0);
    const int64_t bpe = (n + kBatch - 1) / kBatch;
    const int64_t total = bpe * kEpochs;
    int64_t step = 0;

    auto evaluate = [&]() -> std::optional<float> {
        model->eval();
        int64_t correct = 0, seen = Xte.size(0);
        torch::NoGradGuard ng;
        for (int64_t b = 0; b < seen; b += 1000) {
            if (ctl->cancelled(ctl)) return std::nullopt;
            int64_t hi = std::min<int64_t>(b + 1000, seen);
            auto pred = model->forward(Xte.slice(0, b, hi)).argmax(1);
            correct += pred.eq(yte.slice(0, b, hi)).sum().item<int64_t>();
        }
        return seen ? 100.f * (float)correct / (float)seen : 0.f;
    };
    auto record_acc = [&](int64_t at, float pct) {
        { std::lock_guard<std::mutex> lk(st->mtx);
          st->acc_steps.push_back((float)at); st->acc_hist.push_back(pct); }
        if (run != 0) st->metrics.scalar(run, "test/accuracy", at, pct);
    };
    auto end_run = [&]() { if (run != 0) st->metrics.end_run(run);
                           st->run_id.store(0); };

    // Baseline BEFORE the first step: an untrained net is one gray blob at ~10%.
    if (auto a0 = evaluate()) { record_acc(step, *a0);
                                publish_embeddings(st, model, Xte, yte); }
    else { end_run(); return; }

    for (int epoch = 0; epoch < kEpochs; epoch++) {
        model->train();
        auto perm = torch::randperm(n, torch::TensorOptions(dev).dtype(torch::kInt64));
        for (int64_t b = 0; b < n; b += kBatch) {
            if (ctl->cancelled(ctl)) { end_run(); return; }
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
            { std::lock_guard<std::mutex> lk(st->mtx); st->loss_hist.push_back(l); }
            if (run != 0) st->metrics.scalar(run, "train/loss", step, l);
            step++;
            char msg[96];
            std::snprintf(msg, sizeof msg, "epoch %d/%d  loss %.4f",
                          epoch + 1, kEpochs, l);
            ctl->progress(ctl, (float)step / (float)total, msg);
            if (step % kEvalEvery == 0) {
                if (auto a = evaluate()) { record_acc(step, *a);
                                           publish_embeddings(st, model, Xte, yte); }
                else { end_run(); return; }
                model->train();
            }
        }
        float pct;
        if (auto a = evaluate()) { pct = *a; record_acc(step, pct);
                                   publish_embeddings(st, model, Xte, yte); }
        else { end_run(); return; }
        char msg[96];
        std::snprintf(msg, sizeof msg, "epoch %d/%d  test acc %.2f%%",
                      epoch + 1, kEpochs, pct);
        set_status(st, msg);
        ctl->progress(ctl, (float)step / (float)total, msg);
    }
    end_run();
    set_status(st, "training complete — Save the model, or hover a point");
}

// The eval (Load) job (artifacts.v1 LOAD-BEARING): load the checkpoint and run
// ONE eval pass to repopulate the cloud. NO training. The artifact path was
// resolved on the frame thread (host strings are per-call) into st->load_path.
void eval_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<EmbedScopeState*>(user);
    torch::Device dev = pick_device(st);

    std::string paths[4];
    if (!ensure_dataset(st, ctl, paths)) return;
    if (ctl->cancelled(ctl)) return;
    torch::Tensor Xte, yte, dummyX, dummyY;
    if (!load_split(paths[2], paths[3], Xte, yte)) {
        set_status(st, "cached MNIST file was corrupt — press Train to re-download");
        return;
    }
    Xte = Xte.to(dev); yte = yte.to(dev);

    std::string path = st->load_path;
    if (path.empty()) { set_status(st, "load: artifact path missing"); return; }
    try {
        st->model = EmbedNet();
        torch::load(st->model, path);   // loads on CPU
    } catch (...) {
        set_status(st, "load: failed to deserialize checkpoint");
        if (st->host) st->host->log_error("embed-scope: torch::load failed");
        return;
    }
    auto model = st->model;
    model->to(dev);
    if (ctl->cancelled(ctl)) return;
    publish_embeddings(st, model, Xte, yte);
    set_status(st, "loaded checkpoint — eval only, no training (cloud restored)");
}

// ---- frame-thread helpers -------------------------------------------------

bool data_exec(EmbedScopeState* st, const std::string& sql) {
    ArrowArrayStream s{};
    if (!st->data.query(sql.c_str(), &s)) return false;
    if (s.release) s.release(&s);   // DDL/INSERT: drain the (empty) result
    return true;
}

// Rebuild the DuckDB table from the current snapshot and run the SQL panels.
void refresh_data(EmbedScopeState* st, const std::vector<float>& ex,
                  const std::vector<float>& ey, const std::vector<float>& ez,
                  const std::vector<int>& lbl, const std::vector<int>& prd,
                  int64_t n) {
    for (int c = 0; c < 10; c++) st->centroid_valid[c] = false;
    st->misclassified = -1; st->total_rows = 0;
    if (!st->data) { st->data_status = "data.v1 absent (ok) — SQL panels need it";
                     return; }
    if (!data_exec(st, "CREATE OR REPLACE TABLE embed_points"
                       "(label INTEGER, pred INTEGER, x DOUBLE, y DOUBLE, z DOUBLE)")) {
        st->data_status = std::string("data.v1 error: ") + st->data.last_error();
        return;
    }
    std::string sql;
    sql.reserve((size_t)n * 40 + 64);
    sql = "INSERT INTO embed_points VALUES ";
    char row[128];
    for (int64_t i = 0; i < n; i++) {
        std::snprintf(row, sizeof row, "%s(%d,%d,%.6g,%.6g,%.6g)",
                      i ? "," : "", lbl[i], prd[i], ex[i], ey[i], ez[i]);
        sql += row;
    }
    if (n > 0 && !data_exec(st, sql)) {
        st->data_status = std::string("data.v1 error: ") + st->data.last_error();
        return;
    }
    // Centroids: AVG(x,y,z) GROUP BY label -> drained as numeric doubles.
    ArrowArrayStream cs{};
    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;
    if (st->data.query("SELECT label, AVG(x), AVG(y), AVG(z) FROM embed_points "
                       "GROUP BY label ORDER BY label", &cs) &&
        caliper::Data::drain_numeric(&cs, &names, &cols) && cols.size() >= 4) {
        for (size_t r = 0; r < cols[0].size(); r++) {
            int c = (int)cols[0][r];
            if (c >= 0 && c < 10) {
                st->centroid_valid[c] = true;
                st->cent[c][0] = cols[1][r];
                st->cent[c][1] = cols[2][r];
                st->cent[c][2] = cols[3][r];
            }
        }
    }
    // Misclassified + total.
    ArrowArrayStream ms{};
    std::vector<std::vector<double>> mc;
    if (st->data.query("SELECT SUM(CASE WHEN label<>pred THEN 1 ELSE 0 END), "
                       "COUNT(*) FROM embed_points", &ms) &&
        caliper::Data::drain_numeric(&ms, nullptr, &mc) && mc.size() >= 2 &&
        !mc[0].empty()) {
        st->misclassified = (int64_t)mc[0][0];
        st->total_rows    = (int64_t)mc[1][0];
    }
    st->data_status = "data.v1: table rebuilt, SQL panels live";
}

// Build a (28,28,4) RGBA u8 CPU texture for a hovered digit. CPU tensor => the
// bridge stages it on any renderer (§6c). The local rgba lives across the
// synchronous upload.
CaliperTextureId make_digit_tex(EmbedScopeState* st,
                                const std::vector<uint8_t>& px, int idx) {
    if (!st->bridge) return 0;
    std::vector<uint8_t> rgba((size_t)28 * 28 * 4);
    const uint8_t* p = px.data() + (size_t)idx * 28 * 28;
    for (int i = 0; i < 28 * 28; i++) {
        rgba[i * 4 + 0] = p[i]; rgba[i * 4 + 1] = p[i];
        rgba[i * 4 + 2] = p[i]; rgba[i * 4 + 3] = 255;
    }
    auto t = torch::from_blob(rgba.data(), {28, 28, 4}, torch::kUInt8);
    auto ct = caliper::adapters::to_tensor(t);
    if (!ct) return 0;
    return st->bridge.texture_from_tensor(&*ct);
}

} // namespace

// ---- applet facade --------------------------------------------------------

EmbedScopeApplet::EmbedScopeApplet() : s_(std::make_unique<EmbedScopeState>()) {}
EmbedScopeApplet::~EmbedScopeApplet() = default;

bool EmbedScopeApplet::initialize(caliper::Host& host) {
    s_->host      = &host;
    s_->jobs      = caliper::Jobs(host);
    s_->device    = caliper::Device::query(host);
    s_->metrics   = caliper::Metrics(host);       // optional
    s_->bridge    = caliper::Bridge(host);        // optional
    s_->artifacts = caliper::Artifacts(host);     // optional (load-bearing)
    s_->data      = caliper::Data(host);          // optional
    s_->model     = EmbedNet();                   // constructed on CPU
    // curl_global_init MUST run once on the frame thread (libcurl is not
    // thread-safe to lazy-init from a worker's curl_easy_init).
    curl_global_init(CURL_GLOBAL_DEFAULT);
    host.log_info("embed-scope: on_init");
    return true;
}

void EmbedScopeApplet::draw_ui() {
    auto* st = s_.get();
    ImGui::Begin("EmbedScope");   // no SetNextWindowPos/Size — host dockspace

    ImGui::TextDisabled("device: %s (%s)", st->device.name,
                        st->device.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
                        : st->device.kind == CALIPER_DEV_CUDA ? "CUDA" : "CPU");

    // Read a copy of the worker-published snapshot under the mutex.
    std::vector<float> loss, accx, accy, ex, ey, ez;
    std::vector<int>   lbl, prd;
    std::vector<uint8_t> px;
    std::string status;
    uint64_t gen; int64_t n;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        loss = st->loss_hist; accx = st->acc_steps; accy = st->acc_hist;
        status = st->status;  gen = st->embed_gen;  n = st->snap_n;
        if (gen != 0) { ex = st->ex; ey = st->ey; ez = st->ez;
                        lbl = st->labels; prd = st->preds; px = st->pixels; }
    }
    ImGui::TextWrapped("%s", status.c_str());

    if (st->metrics) {
        uint64_t run = st->run_id.load();
        if (run) ImGui::TextDisabled("metrics: run #%llu", (unsigned long long)run);
        else     ImGui::TextDisabled("metrics: present (open Runs)");
    } else ImGui::TextDisabled("metrics: absent (ok)");

    const bool running = st->job_id != 0 && st->jobs.is_running(st->job_id);
    // Dev hook mirroring the host's CALIPER_AUTOLAUNCH: press Train on the
    // first frame when CALIPER_EMBED_AUTOTRAIN=1 (headless debugging / CI).
    static bool autotrain_fired = false;
    if (!autotrain_fired && !running && std::getenv("CALIPER_EMBED_AUTOTRAIN")) {
        autotrain_fired = true;
        { std::lock_guard<std::mutex> lk(st->mtx);
          st->loss_hist.clear(); st->acc_steps.clear(); st->acc_hist.clear();
          st->status = "starting…"; }
        st->job_id = st->jobs.submit("embed_scope: train MNIST 3-D embedding",
                                     &train_job, st);
    }
    if (!running) {
        if (ImGui::Button("Train")) {
            { std::lock_guard<std::mutex> lk(st->mtx);
              st->loss_hist.clear(); st->acc_steps.clear(); st->acc_hist.clear();
              st->status = "starting…"; }
            st->job_id = st->jobs.submit("embed_scope: train MNIST 3-D embedding",
                                         &train_job, st);
            if (st->job_id == 0 && st->host)
                st->host->log_error("embed-scope: submit failed");
        }
        // artifacts.v1 — Save (needs a trained/loaded model) + Load.
        ImGui::SameLine();
        const bool can_save = st->artifacts && gen != 0;
        if (!can_save) ImGui::BeginDisabled();
        if (ImGui::Button("Save model")) {
            std::ostringstream oss(std::ios::binary);
            st->model->to(torch::kCPU);
            torch::save(st->model, oss);
            std::string bytes = oss.str();
            std::string dg = st->artifacts.put(kModelName, bytes.data(),
                                                bytes.size(), st->run_id.load());
            st->save_status = dg.empty()
                ? "save failed" : ("saved  digest " + dg.substr(0, 16) + "…");
        }
        if (!can_save) ImGui::EndDisabled();
        ImGui::SameLine();
        const bool can_load = st->artifacts && st->artifacts.exists(kModelName);
        if (!can_load) ImGui::BeginDisabled();
        if (ImGui::Button("Load model")) {
            const char* p = st->artifacts.path_of(kModelName);   // frame-thread
            if (p) { st->load_path = p;
                     st->job_id = st->jobs.submit(
                         "embed_scope: load checkpoint (eval only)", &eval_job, st);
                     st->save_status = "loading checkpoint…"; }
        }
        if (!can_load) ImGui::EndDisabled();
    } else {
        if (ImGui::Button("Cancel")) st->jobs.request_cancel(st->job_id);
        ImGui::SameLine();
        ImGui::ProgressBar(st->jobs.progress_of(st->job_id), {-1, 0});
    }
    if (!st->artifacts)
        ImGui::TextDisabled("artifacts: absent (ok) — Save/Load need it");
    else if (!st->save_status.empty())
        ImGui::TextDisabled("artifacts: %s", st->save_status.c_str());

    // Refresh the frame-side per-class split + data.v1 SQL when a new snapshot
    // arrived (once per publish, not per frame).
    if (gen != 0 && gen != st->plot_gen) {
        for (int c = 0; c < 10; c++) { st->cx[c].clear(); st->cy[c].clear();
                                       st->cz[c].clear(); }
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int64_t i = 0; i < n; i++) {
            int c = lbl[i]; if (c < 0 || c > 9) continue;
            st->cx[c].push_back(ex[i]); st->cy[c].push_back(ey[i]);
            st->cz[c].push_back(ez[i]);
            float v[3] = {ex[i], ey[i], ez[i]};
            for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], (double)v[k]);
                                          hi[k] = std::max(hi[k], (double)v[k]); }
        }
        for (int k = 0; k < 3; k++) {
            if (hi[k] <= lo[k]) { lo[k] -= 1; hi[k] += 1; }
            double pad = 0.08 * (hi[k] - lo[k]);
            st->bmin[k] = lo[k] - pad; st->bmax[k] = hi[k] + pad;
        }
        st->plot_gen = gen; st->refit = true;
    }
    if (gen != 0 && gen != st->data_gen) {
        refresh_data(st, ex, ey, ez, lbl, prd, n);
        st->data_gen = gen;
    }

    // The centerpiece: the live 3-D embedding scatter.
    int hovered = -1;
    if (ImPlot3D::BeginPlot("##embed", ImVec2(-1, 380))) {
        ImPlot3D::SetupAxes("z0", "z1", "z2");
        ImPlot3D::SetupAxesLimits(st->bmin[0], st->bmax[0], st->bmin[1],
                                  st->bmax[1], st->bmin[2], st->bmax[2],
                                  st->refit ? ImPlot3DCond_Always
                                            : ImPlot3DCond_Once);
        st->refit = false;
        if (gen == 0) {
            ImPlot3D::EndPlot();
            ImGui::TextDisabled("press Train to watch one blob split into ten "
                                "colored lobes");
        } else {
            for (int c = 0; c < 10; c++) {
                if (st->cx[c].empty()) continue;
                char lab[8]; std::snprintf(lab, sizeof lab, "%d", c);
                ImPlot3D::PlotScatter(lab, st->cx[c].data(), st->cy[c].data(),
                    st->cz[c].data(), (int)st->cx[c].size(),
                    ImPlot3DSpec(ImPlot3DProp_MarkerFillColor, kClassCol[c],
                                 ImPlot3DProp_MarkerLineColor, kClassCol[c],
                                 ImPlot3DProp_MarkerSize, 2.2f));
            }
            // data.v1 centroids as large outlined diamonds.
            for (int c = 0; c < 10; c++) {
                if (!st->centroid_valid[c]) continue;
                float cx = (float)st->cent[c][0], cy = (float)st->cent[c][1],
                      cz = (float)st->cent[c][2];
                char id[8]; std::snprintf(id, sizeof id, "##k%d", c);
                ImPlot3D::PlotScatter(id, &cx, &cy, &cz, 1,
                    ImPlot3DSpec(ImPlot3DProp_Marker, ImPlot3DMarker_Diamond,
                                 ImPlot3DProp_MarkerFillColor, kClassCol[c],
                                 ImPlot3DProp_MarkerLineColor, IM_COL32(15,15,15,255),
                                 ImPlot3DProp_MarkerSize, 8.0f));
            }
            // Hover pick: nearest projected point within kPickPx (not while
            // dragging — that rotates the view).
            if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ImVec2 m = ImGui::GetMousePos();
                float best = kPickPx * kPickPx;
                for (int64_t i = 0; i < n; i++) {
                    ImVec2 sp = ImPlot3D::PlotToPixels(ex[i], ey[i], ez[i]);
                    float dx = sp.x - m.x, dy = sp.y - m.y, d = dx * dx + dy * dy;
                    if (d < best) { best = d; hovered = (int)i; }
                }
            }
            ImPlot3D::EndPlot();
        }
    }

    // Hover: (re)build the digit texture on index change, show it in a tooltip.
    if (hovered >= 0 && !px.empty()) {
        if (hovered != st->hover_idx) {
            if (st->hover_tex) { st->bridge.release_texture(st->hover_tex);
                                 st->hover_tex = 0; }
            st->hover_idx = hovered;
            st->hover_tex = make_digit_tex(st, px, hovered);
        }
        ImGui::BeginTooltip();
        ImGui::Text("label %d / pred %d", lbl[hovered], prd[hovered]);
        if (st->hover_tex)
            ImGui::Image(caliper::Bridge::imtex(st->hover_tex), ImVec2(84, 84));
        else
            ImGui::TextDisabled("tensor_bridge.v1 absent (ok) — no image");
        ImGui::EndTooltip();
    } else if (st->hover_idx != -1) {
        if (st->hover_tex) { st->bridge.release_texture(st->hover_tex);
                             st->hover_tex = 0; }
        st->hover_idx = -1;
    }

    // Loss + accuracy (2-D).
    if (ImPlot::BeginPlot("train loss", {-1, 140})) {
        ImPlot::SetupAxes("step", "NLL");
        if (!loss.empty()) ImPlot::PlotLine("loss", loss.data(), (int)loss.size());
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("test accuracy %", {-1, 140})) {
        ImPlot::SetupAxes("step", "acc %");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImPlotCond_Always);
        if (!accy.empty())
            ImPlot::PlotLine("acc", accx.data(), accy.data(), (int)accy.size());
        ImPlot::EndPlot();
    }

    // data.v1 panel.
    ImGui::SeparatorText("data.v1 — SQL over the live embedding table");
    if (!st->data) {
        ImGui::TextDisabled("data.v1 absent (ok) — centroids/misclassified need it");
    } else if (st->total_rows > 0) {
        double pct = 100.0 * (double)st->misclassified / (double)st->total_rows;
        ImGui::Text("rows %lld   misclassified %lld  (%.1f%%)   "
                    "class centroids drawn as diamonds above",
                    (long long)st->total_rows, (long long)st->misclassified, pct);
    } else {
        ImGui::TextDisabled("%s", st->data_status.empty()
            ? "train to populate the table" : st->data_status.c_str());
    }

    ImGui::End();
}

void EmbedScopeApplet::cleanup() {
    auto* st = s_.get();
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        // The applet object must outlive the job (jobs_v1 contract). Cancel is
        // honored <=100ms; the ceiling also covers a cancel mid-download.
        for (int i = 0; i < 1000 && st->jobs.is_running(st->job_id); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Frame-thread-owned texture: release after the job wait, before renderer
    // teardown (the worker never touched the bridge).
    if (st->hover_tex) { st->bridge.release_texture(st->hover_tex);
                         st->hover_tex = 0; }
    curl_global_cleanup();   // pairs with on_init; safe once the worker exited
    if (st->host) st->host->log_info("embed-scope: on_cleanup");
}

} // namespace embedscope
