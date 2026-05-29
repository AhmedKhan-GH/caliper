#define _USE_MATH_DEFINES
#include "repnet_demo.h"
#include "dataset.h"
#include "app_paths.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <fstream>
#include <sstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <ImGuiFileDialog.h>
#include <duckdb.hpp>
#include <torch/script.h>
#include <torch/csrc/autograd/autograd.h>
#include "model_viz.h"

namespace fs = std::filesystem;

// ============================================================================
// CONSTANTS
// ============================================================================

static const ImVec4 LEAD_COLORS[NUM_LEADS] = {
    {1.0f, 0.30f, 0.30f, 1.0f},  // I    - Red
    {0.2f, 1.00f, 0.30f, 1.0f},  // II   - Green
    {0.3f, 0.55f, 1.00f, 1.0f},  // III  - Blue
    {1.0f, 0.85f, 0.20f, 1.0f},  // aVR  - Yellow
    {1.0f, 0.50f, 0.20f, 1.0f},  // aVL  - Orange
    {0.8f, 0.25f, 1.00f, 1.0f},  // aVF  - Purple
    {0.2f, 0.90f, 0.90f, 1.0f},  // V1   - Cyan
    {1.0f, 0.25f, 0.60f, 1.0f},  // V2   - Pink
    {0.5f, 1.00f, 0.25f, 1.0f},  // V3   - Lime
    {1.0f, 0.70f, 0.30f, 1.0f},  // V4   - Gold
    {0.6f, 0.35f, 1.00f, 1.0f},  // V5   - Violet
    {0.2f, 0.80f, 0.60f, 1.0f},  // V6   - Teal
};

// ============================================================================
// PROCESSING PARAMS
// ============================================================================

struct ProcessingParams {
    bool zscore = true;
    bool baseline_wander_correction = false;
    float baseline_cutoff_hz = 0.0f;
    uint32_t version = 1;
};

// ============================================================================
// SIGNAL PROCESSING
// ============================================================================

namespace dsp {

void compute_stats(const std::vector<float>& data, ECGSample::LeadStats& out) {
    if (data.empty()) return;
    float sum = 0;
    float mn = data[0], mx = data[0];
    for (float v : data) {
        sum += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    out.mean = sum / (float)data.size();
    out.min_val = mn;
    out.max_val = mx;

    float var = 0;
    for (float v : data) {
        float d = v - out.mean;
        var += d * d;
    }
    out.stddev = std::sqrt(var / (float)data.size());
}

void zscore(std::vector<float>& data) {
    if (data.size() < 2) return;
    float sum = 0;
    for (float v : data) sum += v;
    float mean = sum / (float)data.size();

    float var = 0;
    for (float v : data) { float d = v - mean; var += d * d; }
    float sd = std::sqrt(var / (float)data.size());
    if (sd < 1e-8f) sd = 1.0f;

    for (float& v : data) v = (v - mean) / sd;
}

void butterworth_highpass(std::vector<float>& data, float cutoff_hz, float sample_rate) {
    if (data.empty() || cutoff_hz <= 0 || sample_rate <= 0) return;
    if (cutoff_hz >= sample_rate * 0.5f) return;
    int n = (int)data.size();

    double wc = std::tan(M_PI * (double)cutoff_hz / (double)sample_rate);
    double wc2 = wc * wc;

    const double Q[2] = { 0.54119610014620, 1.30656296487638 };

    struct SOS { double b0, b1, b2, a1, a2; };
    SOS sos[2];

    for (int s = 0; s < 2; s++) {
        double alpha = wc / Q[s];
        double denom = 1.0 + alpha + wc2;
        sos[s].b0 =  1.0 / denom;
        sos[s].b1 = -2.0 / denom;
        sos[s].b2 =  1.0 / denom;
        sos[s].a1 =  2.0 * (wc2 - 1.0) / denom;
        sos[s].a2 =  (1.0 - alpha + wc2) / denom;
    }

    // Pad length must cover the filter's transient (mirrors scipy filtfilt padlen=3*order).
    // For 2 cascaded SOS sections (order 4), use 3*order * samples_per_cycle.
    int pad = std::min((int)(3.0 * 4.0 * sample_rate / cutoff_hz), n - 1);
    int pn = n + 2 * pad;
    std::vector<double> buf(pn);

    for (int i = 0; i < pad; i++)
        buf[i] = 2.0 * data[0] - data[pad - i];
    for (int i = 0; i < n; i++)
        buf[pad + i] = data[i];
    for (int i = 0; i < pad; i++)
        buf[pad + n + i] = 2.0 * data[n - 1] - data[n - 2 - i];

    for (int s = 0; s < 2; s++) {
        auto& bq = sos[s];

        double w1 = 0, w2 = 0;
        for (int i = 0; i < pn; i++) {
            double xi = buf[i];
            double yi = bq.b0 * xi + w1;
            w1 = bq.b1 * xi - bq.a1 * yi + w2;
            w2 = bq.b2 * xi - bq.a2 * yi;
            buf[i] = yi;
        }

        w1 = 0; w2 = 0;
        for (int i = pn - 1; i >= 0; i--) {
            double xi = buf[i];
            double yi = bq.b0 * xi + w1;
            w1 = bq.b1 * xi - bq.a1 * yi + w2;
            w2 = bq.b2 * xi - bq.a2 * yi;
            buf[i] = yi;
        }
    }

    for (int i = 0; i < n; i++)
        data[i] = (float)buf[pad + i];
}

void process(ECGSample& sample, const ProcessingParams& params,
             const std::atomic<bool>* cancel = nullptr) {
    sample.processed.resize(NUM_LEADS);
    sample.stats.resize(NUM_LEADS);

    for (int lead = 0; lead < NUM_LEADS; lead++) {
        if (cancel && cancel->load(std::memory_order_relaxed)) return;

        sample.processed[lead] = sample.raw[lead];
        auto& sig = sample.processed[lead];

        if (sample.original_num_samples > 2500) {
            std::vector<float> ds;
            ds.reserve(sig.size() / 2);
            for (size_t i = 0; i < sig.size(); i += 2)
                ds.push_back(sig[i]);
            sig = std::move(ds);
            sample.downsampled = true;
        } else {
            sample.downsampled = false;
        }

        if (params.baseline_wander_correction && params.baseline_cutoff_hz > 0 && sample.sampling_rate > 0) {
            butterworth_highpass(sig, params.baseline_cutoff_hz, sample.sampling_rate);
        }

        if (params.zscore) {
            zscore(sig);
        }

        compute_stats(sig, sample.stats[lead]);
    }
    sample.num_samples = (int)sample.processed[0].size();
    sample.processed_valid = true;
}

void derive_xyz(const ECGSample& s,
                std::vector<float>& vx,
                std::vector<float>& vy,
                std::vector<float>& vz) {
    const int N = s.num_samples;
    vx.assign(N, 0.0f);
    vy.assign(N, 0.0f);
    vz.assign(N, 0.0f);
    if (N <= 0 || (int)s.processed.size() < NUM_LEADS) return;

    const auto& I  = s.processed[0];
    const auto& II = s.processed[1];
    const auto& V1 = s.processed[6];
    const auto& V2 = s.processed[7];
    const auto& V3 = s.processed[8];
    const auto& V4 = s.processed[9];
    const auto& V5 = s.processed[10];
    const auto& V6 = s.processed[11];

    for (int i = 0; i < N; i++) {
        float i_  = (int)I.size()  > i ? I[i]  : 0.0f;
        float ii_ = (int)II.size() > i ? II[i] : 0.0f;
        float v1 = (int)V1.size()  > i ? V1[i] : 0.0f;
        float v2 = (int)V2.size()  > i ? V2[i] : 0.0f;
        float v3 = (int)V3.size()  > i ? V3[i] : 0.0f;
        float v4 = (int)V4.size()  > i ? V4[i] : 0.0f;
        float v5 = (int)V5.size()  > i ? V5[i] : 0.0f;
        float v6 = (int)V6.size()  > i ? V6[i] : 0.0f;

        vx[i] =  0.38f*i_  - 0.07f*ii_ - 0.13f*v1 + 0.05f*v2
               - 0.01f*v3 + 0.14f*v4 + 0.06f*v5 + 0.54f*v6;
        vy[i] = -0.07f*i_  + 0.93f*ii_ + 0.06f*v1 - 0.02f*v2
               - 0.05f*v3 + 0.06f*v4 - 0.17f*v5 + 0.13f*v6;
        vz[i] =  0.11f*i_  - 0.23f*ii_ - 0.43f*v1 - 0.06f*v2
               - 0.14f*v3 - 0.20f*v4 - 0.11f*v5 + 0.31f*v6;
    }
}

} // namespace dsp

// ============================================================================
// HEATMAP UTILITIES
// ============================================================================

namespace heatmap {

static void colormap(float t, uint8_t& r, uint8_t& g, uint8_t& b, bool diverging) {
    t = std::clamp(t, 0.0f, 1.0f);
    if (diverging) {
        if (t < 0.5f) {
            float s = t * 2.0f;
            r = (uint8_t)(s * 240);
            g = (uint8_t)(s * 240);
            b = (uint8_t)(120 + s * 135);
        } else {
            float s = (t - 0.5f) * 2.0f;
            r = (uint8_t)(240 + s * 15);
            g = (uint8_t)(240 * (1.0f - s));
            b = (uint8_t)(255 * (1.0f - s * 0.9f));
        }
    } else {
        if (t < 0.33f) {
            float s = t * 3.0f;
            r = (uint8_t)(10 + 60 * s); g = (uint8_t)(10 + 80 * s); b = (uint8_t)(40 + 140 * s);
        } else if (t < 0.66f) {
            float s = (t - 0.33f) * 3.0f;
            r = (uint8_t)(70 + 130 * s); g = (uint8_t)(90 + 100 * s); b = (uint8_t)(180 - 100 * s);
        } else {
            float s = (t - 0.66f) * 3.0f;
            r = (uint8_t)(200 + 55 * s); g = (uint8_t)(190 + 65 * s); b = (uint8_t)(80 - 50 * s);
        }
    }
}

static GLuint upload_texture(const torch::Tensor& data_2d, bool diverging = true,
                             bool log_scale = false) {
    auto data = data_2d.detach().contiguous().to(torch::kCPU, torch::kFloat);
    int rows = (int)data.size(0);
    int cols = (int)data.size(1);
    auto acc = data.accessor<float, 2>();

    float vmin = data.min().item<float>();
    float vmax = data.max().item<float>();
    if (diverging) {
        float absmax = std::max(std::abs(vmin), std::abs(vmax));
        if (absmax < 1e-8f) absmax = 1.0f;
        vmin = -absmax; vmax = absmax;
    }
    float range = vmax - vmin;
    if (range < 1e-8f) range = 1.0f;

    float log_denom = 1.0f;
    if (log_scale) {
        float absmax = std::max(std::abs(vmin), std::abs(vmax));
        log_denom = std::log1p(absmax);
        if (log_denom < 1e-8f) log_denom = 1.0f;
    }

    std::vector<uint8_t> px(rows * cols * 4);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v = acc[r][c];
            float t;
            if (log_scale && diverging) {
                float sign = v >= 0 ? 1.0f : -1.0f;
                float lv = std::log1p(std::abs(v)) / log_denom;
                t = 0.5f + sign * lv * 0.5f;
            } else if (log_scale) {
                t = std::log1p(v - vmin) / std::log1p(range);
            } else {
                t = (v - vmin) / range;
            }
            int i = (r * cols + c) * 4;
            colormap(t, px[i], px[i+1], px[i+2], diverging);
            px[i+3] = 255;
        }
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static void release_textures(std::vector<GLuint>& texs) {
    for (auto& t : texs) {
        if (t) { glDeleteTextures(1, &t); t = 0; }
    }
}

} // namespace heatmap

// ============================================================================
// BACKGROUND PROCESSOR
// ============================================================================

class BackgroundProcessor {
public:
    BackgroundProcessor() {
        stop_.store(false);
        worker_ = std::thread(&BackgroundProcessor::run, this);
    }

    ~BackgroundProcessor() {
        stop_.store(true);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            queue_.clear();
        }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void enqueue(std::vector<ECGSample>* samples, IDatasetLoader* loader,
                 const ProcessingParams& params, const std::vector<int>& indices) {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.clear();
        samples_ = samples;
        loader_ = loader;
        params_ = params;
        for (int idx : indices) queue_.push_back(idx);
        processed_count_.store(0);
        total_queued_.store((int)indices.size());
        cv_.notify_one();
    }

    void prioritize(const std::vector<int>& indices) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = (int)indices.size() - 1; i >= 0; i--)
            queue_.push_front(indices[i]);
        total_queued_.fetch_add((int)indices.size());
        cv_.notify_one();
    }

    int processed_count() const { return processed_count_.load(); }
    int total_queued() const { return total_queued_.load(); }
    bool busy() const { return total_queued_.load() > processed_count_.load(); }

private:
    void run() {
        while (true) {
            int idx = -1;
            ProcessingParams params;
            ECGSample* sample = nullptr;
            IDatasetLoader* loader = nullptr;

            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_.load() || !queue_.empty(); });
                if (stop_.load()) return;
                if (queue_.empty()) continue;

                idx = queue_.front();
                queue_.pop_front();
                params = params_;
                sample = &(*samples_)[idx];
                loader = loader_;
            }

            if (stop_) { processed_count_.fetch_add(1); continue; }

            if (!sample->loaded && loader) {
                loader->load(*sample);
            }

            if (stop_) { processed_count_.fetch_add(1); continue; }

            if (sample->loaded) {
                dsp::process(*sample, params, &stop_);
            }

            processed_count_.fetch_add(1);
        }
    }

    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
    std::deque<int> queue_;
    std::vector<ECGSample>* samples_ = nullptr;
    IDatasetLoader* loader_ = nullptr;
    ProcessingParams params_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> total_queued_{0};
};

// ============================================================================
// DUCKDB RAW DATA BROWSER
// ============================================================================

namespace {

struct FormatInfo {
    const char* ext;
    const char* label;
};

const FormatInfo kFormats[] = {
    {".csv", "CSV"}, {".tsv", "TSV"}, {".parquet", "PARQ"},
    {".pq", "PARQ"}, {".json", "JSON"}, {".jsonl", "JSONL"},
    {".ndjson", "JSONL"},
};

const FormatInfo* match_format(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return nullptr;
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const auto& f : kFormats)
        if (ext == f.ext) return &f;
    return nullptr;
}

std::string sql_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '\'') out += "''";
        else           out += c;
    }
    return out;
}

struct DiscoveredFile {
    std::string path;
    std::string display_name;
    const FormatInfo* fmt = nullptr;
    uintmax_t size_bytes = 0;
};

void scan_folder(const std::string& dir, std::vector<DiscoveredFile>& out) {
    out.clear();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;
    for (auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const auto p = entry.path().string();
        const FormatInfo* fmt = match_format(p);
        if (!fmt) continue;
        DiscoveredFile f;
        f.path = p;
        f.display_name = fs::relative(entry.path(), dir, ec).string();
        f.fmt = fmt;
        std::error_code se;
        f.size_bytes = fs::file_size(entry.path(), se);
        out.push_back(std::move(f));
    }
    std::sort(out.begin(), out.end(),
        [](const DiscoveredFile& a, const DiscoveredFile& b) {
            return a.display_name < b.display_name;
        });
}

std::string format_size(uintmax_t bytes) {
    char buf[32];
    if (bytes < 1024)
        std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    else if (bytes < 1024ULL * 1024)
        std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024ULL * 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024));
    else
        std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024 * 1024));
    return buf;
}

struct PreviewSnapshot {
    bool ready = false;
    std::string error;
    std::vector<std::string> col_names;
    std::vector<std::string> col_types;
    std::vector<std::vector<std::string>> rows;
    size_t row_count = 0;
};

} // namespace

// ============================================================================
// STATE
// ============================================================================

struct WeightEntry {
    std::string label;
    torch::Tensor tensor;
    GLuint tex = 0;
};

struct RepNetDemoApplet::State {
    // ── ECG dataset ──
    std::vector<ECGSample> samples;
    int selected = -1;
    std::unique_ptr<IDatasetLoader> loader;
    std::string current_dir;

    enum class ScanStatus { Idle, Scanning, Ready, Failed };
    std::atomic<ScanStatus> scan_status{ScanStatus::Idle};
    std::thread scan_thread;
    std::mutex scan_result_mtx;
    std::unique_ptr<IDatasetLoader> pending_loader;
    std::vector<ECGSample> pending_samples;
    std::string scan_dir;
    std::string scan_error;

    ProcessingParams params;
    std::unique_ptr<BackgroundProcessor> bg;

    float panel_w = 280.0f;
    bool lead_visible[NUM_LEADS] = {true,true,true,true,true,true,true,true,true,true,true,true};
    int last_plot_sample = -1;
    bool last_plot_had_data = false;
    char filter_buf[128] = {};
    std::vector<float> time_axis;

    // VCG cache + animation
    std::vector<float> vx, vy, vz;
    int vcg_sample_idx = -1;
    uint32_t vcg_params_version = 0;
    float anim_time = 0.0f;
    float anim_speed = 1.0f;
    int trail_samples = 200;
    bool anim_playing = true;
    double last_anim_tick = 0.0;

    // ── DuckDB raw browser ──
    std::unique_ptr<duckdb::DuckDB> db;
    std::unique_ptr<duckdb::Connection> con;
    std::vector<DiscoveredFile> duck_files;
    int duck_selected = -1;
    PreviewSnapshot preview;
    int preview_limit = 100;
    int col_offset = 0;
    int col_page = 50;

    // ── Model inference ──
    torch::jit::Module model;
    bool model_loaded = false;
    std::string model_path;
    std::string model_error;

    // ── Live inference state ──
    InferenceOverlay inference;
    int inference_sample_idx = -1;
    uint32_t inference_params_ver = 0;

    // ── Architecture visualizer ──
    std::unique_ptr<ModelVisualizer> viz;

    // ── Activation detail view ──
    std::vector<torch::Tensor> detail_acts;   // per-node (batch squeezed)
    std::vector<GLuint> detail_texs;          // cached heatmap textures
    int detail_sample_idx = -1;
    int detail_lead = 0;
    int detail_lead_cached = -1;
    bool detail_texs_dirty = true;

    // ── Weight visualization ──
    std::vector<WeightEntry> weight_entries;
    bool weights_extracted = false;

    // ── Batch statistics ──
    struct SampleResult {
        std::string id;
        int true_class = -1;   // 0=Normal, 1=PE, -1=unknown
        int pred_class = -1;
        float prob_pe = 0;
        float gap_feat[576] = {};
    };
    std::vector<SampleResult> batch_results;
    std::vector<float> pca_x, pca_y;
    bool batch_stale = true;
    bool batch_running = false;
    int batch_progress = 0;

    void run_preview(const DiscoveredFile& f) {
        preview = PreviewSnapshot{};
        if (!con) { preview.error = "DuckDB not initialized"; return; }

        const std::string p = sql_escape(f.path);
        char qbuf[1024];
        std::snprintf(qbuf, sizeof(qbuf),
            "SELECT * FROM '%s' LIMIT %d", p.c_str(), preview_limit);

        auto result = con->Query(qbuf);
        if (result->HasError()) { preview.error = result->GetError(); return; }

        const size_t ncols = result->ColumnCount();
        preview.col_names.reserve(ncols);
        preview.col_types.reserve(ncols);
        for (size_t c = 0; c < ncols; c++) {
            preview.col_names.push_back(result->ColumnName(c));
            preview.col_types.push_back(result->types[c].ToString());
        }

        while (auto chunk = result->Fetch()) {
            const size_t nrows = chunk->size();
            for (size_t r = 0; r < nrows; r++) {
                std::vector<std::string> row;
                row.reserve(ncols);
                for (size_t c = 0; c < ncols; c++) {
                    duckdb::Value v = chunk->GetValue(c, r);
                    row.push_back(v.IsNull() ? "NULL" : v.ToString());
                }
                preview.rows.push_back(std::move(row));
            }
            if (preview.rows.size() >= (size_t)preview_limit) break;
        }
        preview.row_count = preview.rows.size();
        preview.ready = true;
    }
};

// ============================================================================
// LIFECYCLE
// ============================================================================

RepNetDemoApplet::RepNetDemoApplet() = default;
RepNetDemoApplet::~RepNetDemoApplet() = default;

static void extract_weights(torch::jit::Module& model,
                            std::vector<WeightEntry>& out);

bool RepNetDemoApplet::initialize() {
    s_ = std::make_unique<State>();

    s_->params.baseline_wander_correction = true;
    s_->params.baseline_cutoff_hz = 0.5f;

    try {
        s_->db  = std::make_unique<duckdb::DuckDB>(nullptr);
        s_->con = std::make_unique<duckdb::Connection>(*s_->db);
    } catch (const std::exception& e) {
        std::cerr << "[workbench] DuckDB init failed: " << e.what() << std::endl;
    }

    // Restore last dataset if saved
    {
        std::ifstream f(caliper::app_data_path("last_dataset.txt"));
        std::string last_dir;
        if (f.is_open() && std::getline(f, last_dir) && fs::is_directory(last_dir)) {
            open_dataset(last_dir);
        }
    }

    // Restore last model if saved
    {
        std::ifstream f(caliper::app_data_path("last_model.txt"));
        std::string last_model;
        if (f.is_open() && std::getline(f, last_model) && fs::exists(last_model)) {
            try {
                s_->model = torch::jit::load(last_model);
                s_->model.eval();
                s_->model_loaded = true;
                s_->model_path = last_model;
                extract_weights(s_->model, s_->weight_entries);
                s_->weights_extracted = true;
            } catch (const std::exception& e) {
                std::cerr << "[workbench] Model restore failed: " << e.what() << std::endl;
            }
        }
    }

    return true;
}

void RepNetDemoApplet::cleanup() {
    if (!s_) return;
    heatmap::release_textures(s_->detail_texs);
    for (auto& w : s_->weight_entries)
        if (w.tex) { glDeleteTextures(1, &w.tex); w.tex = 0; }
    s_->bg.reset();
    if (s_->scan_thread.joinable()) s_->scan_thread.join();
    s_->con.reset();
    s_->db.reset();
    s_.reset();
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

namespace {

std::vector<int> outward_indices(int center, int count, int total) {
    std::vector<int> out;
    out.reserve(count);
    for (int d = 1; (int)out.size() < count; d++) {
        bool added = false;
        int lo = center - d, hi = center + d;
        if (lo >= 0 && lo < total) { out.push_back(lo); added = true; }
        if (hi >= 0 && hi < total) { out.push_back(hi); added = true; }
        if (!added) break;
    }
    return out;
}

} // namespace

// ============================================================================
// DRAW
// ============================================================================

void RepNetDemoApplet::draw_ui(int /*win_w*/, int /*win_h*/) {
    if (!s_) return;

    // ── Commit async scan results ──
    if (s_->scan_status.load() == State::ScanStatus::Ready) {
        if (s_->scan_thread.joinable()) s_->scan_thread.join();
        s_->bg.reset();
        {
            std::lock_guard<std::mutex> lk(s_->scan_result_mtx);
            s_->loader = std::move(s_->pending_loader);
            s_->samples = std::move(s_->pending_samples);
            s_->current_dir = s_->scan_dir;
        }
        s_->selected = -1;
        s_->batch_stale = true;
        s_->bg = std::make_unique<BackgroundProcessor>();
        s_->scan_status.store(State::ScanStatus::Idle);

        std::cout << "[workbench] Opened " << s_->current_dir << ": "
                  << s_->samples.size() << " samples" << std::endl;

        // Persist last dataset path
        {
            std::ofstream f(caliper::app_data_path("last_dataset.txt"));
            if (f.is_open()) f << s_->current_dir;
        }

        // Also scan folder for DuckDB raw file browser
        scan_folder(s_->current_dir, s_->duck_files);
        s_->duck_selected = -1;
        s_->preview = PreviewSnapshot{};

        // Auto-select first sample
        if (!s_->samples.empty()) {
            s_->selected = 0;
            auto& samp = s_->samples[0];
            auto neighbors = outward_indices(0, (int)s_->samples.size() - 1, (int)s_->samples.size());
            if (samp.loaded) {
                if (!samp.processed_valid) dsp::process(samp, s_->params);
                s_->bg->enqueue(&s_->samples, s_->loader.get(), s_->params, neighbors);
            } else {
                neighbors.insert(neighbors.begin(), 0);
                s_->bg->enqueue(&s_->samples, s_->loader.get(), s_->params, neighbors);
            }
        }
    }

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##WorkbenchRoot", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;
    float sp = ImGui::GetStyle().ItemSpacing.x;
    float splitter_thick = 6.0f;

    s_->panel_w = std::clamp(s_->panel_w, 200.0f, avail_w - 300.0f);
    float plot_w = avail_w - s_->panel_w - sp - splitter_thick;

    // -- Left panel --
    ImGui::BeginChild("##Panel", ImVec2(s_->panel_w, avail_h), true);
    draw_panel();
    ImGui::EndChild();

    // -- Splitter handle --
    ImGui::SameLine();
    ImGui::InvisibleButton("##splitter", ImVec2(splitter_thick, avail_h));
    if (ImGui::IsItemActive())
        s_->panel_w += ImGui::GetIO().MouseDelta.x;
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

    ImGui::SameLine();

    // -- Right: tabs --
    ImGui::BeginChild("##Content", ImVec2(plot_w, avail_h), false);
    if (ImGui::BeginTabBar("##view_tabs")) {
        if (ImGui::BeginTabItem("Leads")) {
            draw_leads();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("3D Vector")) {
            draw_vcg_3d();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Raw Data")) {
            draw_raw_browser();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Model")) {
            draw_model_tab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::End();
}

// ============================================================================
// PANEL (left sidebar)
// ============================================================================

void RepNetDemoApplet::open_dataset(const std::string& dir) {
    if (s_->scan_thread.joinable()) s_->scan_thread.join();
    s_->scan_status.store(State::ScanStatus::Scanning);
    s_->scan_error.clear();
    s_->scan_dir = dir;

    auto* sp = s_.get();
    s_->scan_thread = std::thread([sp, dir]() {
        auto loader = make_dataset_loader();

        std::vector<ECGSample> samples;
        bool ok = loader->scan(dir, samples);
        if (!ok) {
            std::lock_guard<std::mutex> lk(sp->scan_result_mtx);
            sp->scan_error = std::string("No matching files in: ") + dir;
            sp->scan_status.store(RepNetDemoApplet::State::ScanStatus::Failed);
            return;
        }

        std::lock_guard<std::mutex> lk(sp->scan_result_mtx);
        sp->pending_loader = std::move(loader);
        sp->pending_samples = std::move(samples);
        sp->scan_status.store(RepNetDemoApplet::State::ScanStatus::Ready);
    });
}

void RepNetDemoApplet::select_sample(int idx) {
    auto& s = *s_;
    if (idx < 0 || idx >= (int)s.samples.size()) return;
    s.selected = idx;
    if (!s.bg || !s.loader) return;

    auto& samp = s.samples[idx];

    if (samp.loaded && samp.processed_valid) return;

    // If the background processor is busy (e.g. after a params change),
    // don't process on the main thread — it races with the bg thread.
    // Just prioritize this sample in the queue instead.
    if (samp.loaded && !s.bg->busy()) {
        dsp::process(samp, s.params);
        return;
    }

    std::vector<int> to_load;
    to_load.push_back(idx);
    for (int i : outward_indices(idx, 20, (int)s.samples.size())) {
        if (!s.samples[i].loaded || !s.samples[i].processed_valid)
            to_load.push_back(i);
    }
    s.bg->prioritize(to_load);
}

void RepNetDemoApplet::on_params_changed() {
    auto& s = *s_;
    s.params.version++;
    s.last_plot_sample = -1;
    for (auto& samp : s.samples) samp.processed_valid = false;
    if (!s.bg || !s.loader || s.samples.empty()) return;

    int center = std::max(0, s.selected);
    if (s.selected >= 0 && s.selected < (int)s.samples.size()
        && s.samples[s.selected].loaded) {
        dsp::process(s.samples[s.selected], s.params);
        auto neighbors = outward_indices(center, (int)s.samples.size() - 1, (int)s.samples.size());
        s.bg->enqueue(&s.samples, s.loader.get(), s.params, neighbors);
    } else {
        auto idxs = outward_indices(center, (int)s.samples.size() - 1, (int)s.samples.size());
        idxs.insert(idxs.begin(), center);
        s.bg->enqueue(&s.samples, s.loader.get(), s.params, idxs);
    }
}

void RepNetDemoApplet::ensure_vcg_cached() {
    auto& s = *s_;
    if (s.selected < 0 || s.selected >= (int)s.samples.size()) {
        s.vcg_sample_idx = -1;
        s.vx.clear(); s.vy.clear(); s.vz.clear();
        return;
    }
    auto& samp = s.samples[s.selected];
    if (!samp.processed_valid) { s.vcg_sample_idx = -1; return; }
    if (s.vcg_sample_idx == s.selected && s.vcg_params_version == s.params.version
        && (int)s.vx.size() == samp.num_samples) return;
    dsp::derive_xyz(samp, s.vx, s.vy, s.vz);
    s.vcg_sample_idx = s.selected;
    s.vcg_params_version = s.params.version;
}

// ============================================================================
// PANEL (left sidebar)
// ============================================================================

void RepNetDemoApplet::draw_panel() {
    auto& s = *s_;
    // ── Back button ──
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.14f, 0.14f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.20f, 0.20f, 1.0f));
    if (ImGui::Button("<< Back to Menu", ImVec2(-1, 28))) {
        exit_requested_ = true;
    }
    ImGui::PopStyleColor(2);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Dataset ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "DATASET");
    ImGui::Separator();

    if (s.current_dir.empty()) {
        ImGui::TextColored({0.8f, 0.8f, 0.8f, 1.0f}, "(none)");
    } else {
        ImGui::TextWrapped("%s", s.current_dir.c_str());
    }

    ImGui::Spacing();

    auto st = s.scan_status.load();
    bool scanning = (st == State::ScanStatus::Scanning);

    if (scanning) ImGui::BeginDisabled();
    if (ImGui::Button("Open Dataset...", ImVec2(-1, 28))) {
        IGFD::FileDialogConfig cfg;
        cfg.path = s.current_dir.empty() ? "." : s.current_dir;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "OpenDatasetDlg", "Choose dataset directory", nullptr, cfg);
    }
    if (scanning) ImGui::EndDisabled();

    if (scanning) {
        ImGui::TextColored({1.0f, 0.85f, 0.3f, 1.0f}, "Scanning...");
    } else if (st == State::ScanStatus::Failed) {
        std::string err;
        { std::lock_guard<std::mutex> lk(s.scan_result_mtx); err = s.scan_error; }
        ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "Scan failed:");
        ImGui::TextWrapped("%s", err.c_str());
    }

    ImVec2 min_sz(600, 400);
    ImVec2 max_sz(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display("OpenDatasetDlg",
            ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            open_dataset(dir);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (s.samples.empty()) {
        if (!scanning && st != State::ScanStatus::Failed) {
            ImGui::TextColored({1.0f, 0.7f, 0.3f, 1.0f}, "No dataset loaded.");
            ImGui::TextWrapped("Click Open Dataset... and pick the directory.");
        }
        return;
    }

    // ── Sample picker ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "SAMPLE PICKER");
    ImGui::Separator();
    ImGui::Text("Samples: %d", (int)s.samples.size());

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##filter", "Filter by ID...", s.filter_buf, sizeof(s.filter_buf))) {}

    float list_h = std::min(200.0f, ImGui::GetContentRegionAvail().y * 0.35f);
    if (ImGui::BeginListBox("##samples", ImVec2(-1, list_h))) {
        std::string filter(s.filter_buf);
        for (int i = 0; i < (int)s.samples.size(); i++) {
            if (!filter.empty() && s.samples[i].file_id.find(filter) == std::string::npos)
                continue;
            bool is_selected = (i == s.selected);
            auto& si = s.samples[i];
            std::string label;
            if (!si.label.empty()) {
                bool pos = si.label.find("Normal") == std::string::npos;
                label = pos ? "[+] " : "[-] ";
            }
            label += si.file_id;
            if (si.loaded) {
                label += " (" + std::to_string(si.num_samples) + ")";
                if (si.downsampled) label += " [ds]";
            }
            if (!si.processed_valid && si.loaded)
                label += " *";
            if (ImGui::Selectable(label.c_str(), is_selected))
                select_sample(i);
        }
        ImGui::EndListBox();
    }

    float btn_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    if (ImGui::Button("<< Prev", ImVec2(btn_w, 0)) && s.selected > 0)
        select_sample(s.selected - 1);
    ImGui::SameLine();
    if (ImGui::Button("Next >>", ImVec2(btn_w, 0)) && s.selected < (int)s.samples.size() - 1)
        select_sample(s.selected + 1);

    int sel = s.selected;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::SliderInt("##id_scroll", &sel, 0, (int)s.samples.size() - 1))
        select_sample(sel);

    ImGui::Spacing();
    ImGui::Separator();

    // ── Processing controls ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "PROCESSING");
    ImGui::Separator();

    bool changed = false;
    if (ImGui::Checkbox("Z-Score Normalize", &s.params.zscore)) changed = true;
    if (ImGui::Checkbox("Baseline Wander Correction", &s.params.baseline_wander_correction))
        changed = true;

    if (s.params.baseline_wander_correction) {
        ImGui::Indent(12);
        ImGui::SetNextItemWidth(-12);
        if (ImGui::DragFloat("##cutoff", &s.params.baseline_cutoff_hz, 0.01f, 0.0f, 125.0f, "%.2f Hz"))
            changed = true;
        ImGui::Unindent(12);
    }

    if (changed) on_params_changed();

    if (s.bg && s.bg->busy()) {
        ImGui::Spacing();
        int done = s.bg->processed_count();
        int total = s.bg->total_queued();
        float frac = total > 0 ? (float)done / (float)total : 0.0f;
        char buf[64];
        snprintf(buf, sizeof(buf), "BG: %d/%d", done, total);
        ImGui::ProgressBar(frac, ImVec2(-1, 18), buf);
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── Current sample info ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "CURRENT SAMPLE");
    ImGui::Separator();

    if (s.selected >= 0 && s.selected < (int)s.samples.size()) {
        auto& samp = s.samples[s.selected];
        ImGui::Text("ID: %s", samp.file_id.c_str());
        if (!samp.label.empty()) {
            bool positive = samp.label.find("Normal") == std::string::npos;
            ImVec4 col = positive ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
                                  : ImVec4(0.4f, 1.0f, 0.6f, 1.0f);
            ImGui::TextColored(col, "%s", positive ? "Positive" : "Negative");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", samp.label.c_str());
        }
        if (samp.downsampled)
            ImGui::Text("Samples: %d (ds from %d)", samp.num_samples, samp.original_num_samples);
        else
            ImGui::Text("Samples: %d", samp.num_samples);
        ImGui::Text("Rate: %.0f Hz", samp.sampling_rate);
        ImGui::Text("Duration: %.1f sec", samp.num_samples / std::max(1.0f, samp.sampling_rate));

        if (samp.processed_valid && !samp.stats.empty()) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "LEAD STATS");
            ImGui::Separator();

            for (int i = 0; i < NUM_LEADS; i++) {
                ImGui::PushStyleColor(ImGuiCol_Text, LEAD_COLORS[i]);
                ImGui::Checkbox(LEAD_NAMES[i], &s.lead_visible[i]);
                ImGui::PopStyleColor();
                if (i < NUM_LEADS - 1 && (i % 3 != 2)) ImGui::SameLine(0, 15);
            }

            ImGui::Spacing();

            if (ImGui::BeginTable("##stats", 4,
                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
                ImGui::TableSetupColumn("Lead", ImGuiTableColumnFlags_WidthFixed, 40);
                ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Std", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Range", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                for (int i = 0; i < NUM_LEADS; i++) {
                    if (!s.lead_visible[i]) continue;
                    auto& st = samp.stats[i];
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextColored(LEAD_COLORS[i], "%s", LEAD_NAMES[i]);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.1f", st.mean);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.1f", st.stddev);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.0f", st.max_val - st.min_val);
                }
                ImGui::EndTable();
            }
        }
    }
}

// ============================================================================
// LEADS TAB
// ============================================================================

void RepNetDemoApplet::draw_leads() {
    auto& s = *s_;

    bool has_data = s.selected >= 0 && s.selected < (int)s.samples.size()
                    && s.samples[s.selected].loaded && s.samples[s.selected].processed_valid;

    ECGSample* samp = has_data ? &s.samples[s.selected] : nullptr;

    float avail_h = ImGui::GetContentRegionAvail().y;
    float avail_w = ImGui::GetContentRegionAvail().x;

    int visible_count = 0;
    for (int i = 0; i < NUM_LEADS; i++) if (s.lead_visible[i]) visible_count++;
    if (visible_count == 0) visible_count = NUM_LEADS;

    float sp = ImGui::GetStyle().ItemSpacing.y;
    float plot_h = (avail_h - sp * (visible_count - 1)) / (float)visible_count;
    plot_h = std::max(plot_h, 60.0f);

    float duration = samp ? samp->num_samples / std::max(1.0f, samp->sampling_rate) : 10.0f;

    if (samp && (int)s.time_axis.size() != samp->num_samples) {
        s.time_axis.resize(samp->num_samples);
        for (int i = 0; i < samp->num_samples; i++)
            s.time_axis[i] = (float)i / samp->sampling_rate;
    }

    bool sample_changed = (s.selected != s.last_plot_sample)
                        || (has_data && !s.last_plot_had_data);
    s.last_plot_sample = s.selected;
    s.last_plot_had_data = has_data;

    ImPlot::GetInputMap().ZoomRate = 0.15f;

    for (int lead = 0; lead < NUM_LEADS; lead++) {
        if (!s.lead_visible[lead]) continue;

        if (has_data)
            ImPlot::SetNextAxisToFit(ImAxis_Y1);

        char plot_id[64];
        snprintf(plot_id, sizeof(plot_id), "##lead_%d", lead);

        if (ImPlot::BeginPlot(plot_id, ImVec2(avail_w, plot_h),
                ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoInputs)) {
            ImPlot::SetupAxes("Time (s)", LEAD_NAMES[lead],
                ImPlotAxisFlags_NoLabel, ImPlotAxisFlags_NoLabel);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Once);

            if (samp && lead < (int)samp->processed.size() && !samp->processed[lead].empty()
                && lead < (int)samp->stats.size()) {
                int plot_n = std::min(samp->num_samples, (int)samp->processed[lead].size());
                if (plot_n > 0 && plot_n <= (int)s.time_axis.size()) {
                    auto& st = samp->stats[lead];
                    float margin = (st.max_val - st.min_val) * 0.1f;
                    ImPlot::SetupAxisLimits(ImAxis_Y1, st.min_val - margin, st.max_val + margin,
                        ImGuiCond_Always);

                    ImPlot::Annotation(0.0, st.max_val, LEAD_COLORS[lead],
                        ImVec2(5, 5), false, "%s", LEAD_NAMES[lead]);

                    ImPlot::PlotLine("##sig", s.time_axis.data(), samp->processed[lead].data(),
                        plot_n,
                        ImPlotSpec(ImPlotProp_LineColor, LEAD_COLORS[lead], ImPlotProp_LineWeight, 1.2f));
                }
            } else {
                ImPlot::SetupAxisLimits(ImAxis_Y1, -1, 1, ImGuiCond_Once);
                ImPlot::Annotation(0.0, 0.8, LEAD_COLORS[lead],
                    ImVec2(5, 5), false, "%s", LEAD_NAMES[lead]);
            }

            ImPlot::EndPlot();
        }
    }
}

// ============================================================================
// VCG 3D TAB
// ============================================================================

void RepNetDemoApplet::draw_vcg_3d() {
    auto& s = *s_;
    if (s.selected < 0 || s.selected >= (int)s.samples.size()) {
        ImGui::TextColored({0.7f, 0.7f, 0.8f, 1.0f}, "Select a sample from the panel.");
        return;
    }
    auto& samp = s.samples[s.selected];
    if (!samp.loaded || !samp.processed_valid) {
        ImGui::Spacing();
        ImGui::TextColored({1.0f, 0.85f, 0.3f, 1.0f},
            "Loading sample %s ...", samp.file_id.c_str());
        return;
    }

    ensure_vcg_cached();
    const int N = (int)s.vx.size();
    if (N <= 0) { ImGui::Text("No VCG data."); return; }

    const float sr = std::max(1.0f, samp.sampling_rate);
    const float duration = N / sr;

    double now = ImGui::GetTime();
    if (s.last_anim_tick <= 0.0) s.last_anim_tick = now;
    double dt = now - s.last_anim_tick;
    s.last_anim_tick = now;
    if (s.anim_playing) {
        s.anim_time += (float)dt * s.anim_speed;
        if (s.anim_time >= duration) s.anim_time = std::fmod(s.anim_time, duration);
        if (s.anim_time < 0) s.anim_time = 0;
    }

    if (ImGui::Button(s.anim_playing ? "Pause" : "Play", ImVec2(70, 0)))
        s.anim_playing = !s.anim_playing;
    ImGui::SameLine();
    if (ImGui::Button("Restart", ImVec2(70, 0))) s.anim_time = 0;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("speed", &s.anim_speed, 0.05f, 4.0f, "%.2fx");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("trail", &s.trail_samples, 10, std::max(50, N / 2), "%d smp");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat("t (s)", &s.anim_time, 0.0f, duration, "%.3f");

    int cur = (int)(s.anim_time * sr);
    if (cur < 0) cur = 0;
    if (cur >= N) cur = N - 1;

    float rmax = 1e-3f;
    for (int i = 0; i < N; i++) {
        rmax = std::max(rmax, std::fabs(s.vx[i]));
        rmax = std::max(rmax, std::fabs(s.vy[i]));
        rmax = std::max(rmax, std::fabs(s.vz[i]));
    }
    rmax *= 1.1f;

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot3D::BeginPlot("##vcg", avail)) {
        ImPlot3D::SetupAxes("X  (L +, R -)", "Y  (Inf +, Sup -)", "Z  (Post +, Ant -)");
        ImPlot3D::SetupAxesLimits(-rmax, rmax, -rmax, rmax, -rmax, rmax, ImPlot3DCond_Always);

        ImU32 col_loop = IM_COL32(110, 150, 230, 90);
        ImPlot3D::PlotLine("loop", s.vx.data(), s.vy.data(), s.vz.data(), N,
            ImPlot3DSpec(ImPlot3DProp_LineColor, col_loop, ImPlot3DProp_LineWeight, 1.0f));

        int t0 = std::max(0, cur - s.trail_samples);
        int tn = cur - t0 + 1;
        if (tn > 1) {
            ImU32 col_trail = IM_COL32(255, 210, 90, 235);
            ImPlot3D::PlotLine("trail", s.vx.data() + t0, s.vy.data() + t0,
                s.vz.data() + t0, tn,
                ImPlot3DSpec(ImPlot3DProp_LineColor, col_trail, ImPlot3DProp_LineWeight, 2.5f));
        }

        float head_x[2] = {0, s.vx[cur]};
        float head_y[2] = {0, s.vy[cur]};
        float head_z[2] = {0, s.vz[cur]};
        ImU32 col_vec = IM_COL32(255, 255, 255, 200);
        ImPlot3D::PlotLine("##vec", head_x, head_y, head_z, 2,
            ImPlot3DSpec(ImPlot3DProp_LineColor, col_vec, ImPlot3DProp_LineWeight, 1.5f));

        ImU32 col_pt = IM_COL32(255, 80, 80, 255);
        float px = s.vx[cur], py = s.vy[cur], pz = s.vz[cur];
        ImPlot3D::PlotScatter("##pt", &px, &py, &pz, 1,
            ImPlot3DSpec(ImPlot3DProp_MarkerFillColor, col_pt,
                         ImPlot3DProp_MarkerLineColor, col_pt,
                         ImPlot3DProp_MarkerSize, 6.0f));

        ImPlot3D::EndPlot();
    }
}

// ============================================================================
// RAW DATA BROWSER TAB (DuckDB)
// ============================================================================

void RepNetDemoApplet::draw_raw_browser() {
    auto& s = *s_;
    if (s.current_dir.empty()) {
        ImGui::TextDisabled("Open a dataset to browse raw files.");
        return;
    }

    if (s.duck_files.empty()) {
        ImGui::TextDisabled("No supported files in the dataset directory.");
        ImGui::TextDisabled("Supported: .csv .tsv .parquet .json .jsonl");
        return;
    }

    // File list (horizontal layout within tab)
    float file_panel_w = 280.0f;
    float avail_h = ImGui::GetContentRegionAvail().y;

    ImGui::BeginChild("##duck_files", ImVec2(file_panel_w, avail_h), true);
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "FILES");
    ImGui::Separator();

    for (int i = 0; i < (int)s.duck_files.size(); i++) {
        const auto& f = s.duck_files[i];
        char label[512];
        std::snprintf(label, sizeof(label), "[%s] %s  (%s)##%d",
            f.fmt ? f.fmt->label : "?",
            f.display_name.c_str(),
            format_size(f.size_bytes).c_str(), i);

        if (ImGui::Selectable(label, i == s.duck_selected)) {
            s.duck_selected = i;
            s.col_offset = 0;
            s.run_preview(f);
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("##duck_preview", ImVec2(0, avail_h), false);

    if (s.duck_selected < 0 || s.duck_selected >= (int)s.duck_files.size()) {
        ImGui::TextDisabled("Select a file to preview.");
    } else {
        const auto& f = s.duck_files[s.duck_selected];
        ImGui::TextWrapped("%s", f.path.c_str());
        ImGui::TextDisabled("%s  %s", f.fmt ? f.fmt->label : "?",
            format_size(f.size_bytes).c_str());
        ImGui::Spacing();

        ImGui::SetNextItemWidth(180);
        if (ImGui::SliderInt("rows", &s.preview_limit, 10, 5000, "%d"))
            s.run_preview(f);
        ImGui::SameLine();
        if (ImGui::Button("Refresh"))
            s.run_preview(f);
        ImGui::Spacing();

        if (!s.preview.ready) {
            if (!s.preview.error.empty()) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Query failed:");
                ImGui::TextWrapped("%s", s.preview.error.c_str());
            } else {
                ImGui::TextDisabled("(loading...)");
            }
        } else {
            const int total_cols = (int)s.preview.col_names.size();
            const int kMaxPage = 256;
            int& col_page = s.col_page;
            int& col_off = s.col_offset;

            if (col_page < 1) col_page = 1;
            if (col_page > kMaxPage) col_page = kMaxPage;
            const int max_off = std::max(0, total_cols - col_page);
            if (col_off < 0) col_off = 0;
            if (col_off > max_off) col_off = max_off;

            const int ncols = std::min(col_page, total_cols - col_off);
            const int last_col = col_off + ncols;

            ImGui::TextDisabled("%zu rows x %d cols — viewing cols %d-%d",
                s.preview.row_count, total_cols, col_off + 1, last_col);

            if (total_cols > col_page) {
                ImGui::BeginDisabled(col_off == 0);
                if (ImGui::Button("<<", ImVec2(36, 0))) col_off = 0;
                ImGui::SameLine();
                if (ImGui::Button("<", ImVec2(36, 0))) col_off = std::max(0, col_off - col_page);
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::SetNextItemWidth(220);
                ImGui::SliderInt("##col_off", &col_off, 0, max_off, "first col %d");
                ImGui::SameLine();
                ImGui::BeginDisabled(col_off >= max_off);
                if (ImGui::Button(">", ImVec2(36, 0))) col_off = std::min(max_off, col_off + col_page);
                ImGui::SameLine();
                if (ImGui::Button(">>", ImVec2(36, 0))) col_off = max_off;
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("cols/page", &col_page, 1, kMaxPage, "%d");
            }

            if (ncols > 0 && ImGui::BeginTable("##preview_tbl", ncols,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollX |
                    ImGuiTableFlags_ScrollY,
                    ImVec2(0, ImGui::GetContentRegionAvail().y - 6))) {

                ImGui::TableSetupScrollFreeze(0, 1);
                for (int c = 0; c < ncols; c++) {
                    const int gc = col_off + c;
                    ImGui::TableSetupColumn(s.preview.col_names[gc].c_str(),
                        ImGuiTableColumnFlags_WidthFixed, 140.0f);
                }
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                for (int c = 0; c < ncols; c++) {
                    const int gc = col_off + c;
                    ImGui::TableSetColumnIndex(c);
                    ImGui::TextColored(ImVec4(0.65f, 0.85f, 0.65f, 1.0f),
                        "%s", s.preview.col_types[gc].c_str());
                }

                for (const auto& row : s.preview.rows) {
                    ImGui::TableNextRow();
                    for (int c = 0; c < ncols; c++) {
                        const int gc = col_off + c;
                        if (gc >= (int)row.size()) break;
                        ImGui::TableSetColumnIndex(c);
                        ImGui::TextUnformatted(row[gc].c_str());
                    }
                }
                ImGui::EndTable();
            }
        }
    }

    ImGui::EndChild();
}

// ============================================================================
// MODEL TAB
// ============================================================================

static LayerActivation tensor_stats(const torch::Tensor& t) {
    LayerActivation a;
    a.mean = t.mean().item<float>();
    a.stddev = t.std().item<float>();
    a.min_val = t.min().item<float>();
    a.max_val = t.max().item<float>();
    auto sizes = t.sizes();
    a.shape = "(";
    for (int64_t i = 0; i < (int64_t)sizes.size(); i++) {
        if (i > 0) a.shape += ",";
        a.shape += std::to_string(sizes[i]);
    }
    a.shape += ")";
    a.valid = true;
    return a;
}

static torch::Tensor compute_grad_cam(torch::jit::Module& model,
                                       const torch::Tensor& input,
                                       int target_class,
                                       int input_length) {
    int B = (int)input.size(0);
    int nl = (int)input.size(1);
    int T_in = (int)input.size(2);
    torch::Tensor backbone_out;

    {
        torch::NoGradGuard no_grad;
        auto x = input.reshape({B * nl, 1, T_in});
        x = model.attr("backbone").toModule().forward({x}).toTensor();
        backbone_out = x;  // (B*nl, C, T')
    }

    int C = (int)backbone_out.size(1);
    int Tp = (int)backbone_out.size(2);

    // Reshape to (B, nl, C, T') and anchor gradients
    auto A = backbone_out.reshape({B, nl, C, Tp}).detach().clone();
    A.requires_grad_(true);

    // Continue forward: pool, reshape, dropout, FC
    auto a_flat = A.reshape({B * nl, C, Tp});
    auto x = model.attr("pool").toModule().forward({a_flat}).toTensor().squeeze(-1);
    x = x.reshape({B, nl * C});
    x = model.attr("head_drop").toModule().forward({x}).toTensor();
    auto logits = model.attr("fc").toModule().forward({x}).toTensor();

    auto target = logits.select(1, target_class).sum();
    auto grads = torch::autograd::grad(
        {target}, {A}, /*grad_outputs=*/{}, /*retain_graph=*/c10::nullopt,
        /*create_graph=*/false, /*allow_unused=*/true);

    if (grads.empty() || !grads[0].defined()) {
        std::fprintf(stderr, "[grad-cam] No gradient on feature map\n");
        return {};
    }

    // A: [1, nl, C, T'], grads: [1, nl, C, T']
    // Per-lead Grad-CAM: alpha = GAP over T' of grad, cam = sum(alpha * feat, dim=C)
    auto result = torch::zeros({nl, input_length});
    auto res_a = result.accessor<float, 2>();

    for (int l = 0; l < nl; l++) {
        auto grad_l = grads[0].select(1, l);  // [1, C, T']
        auto feat_l = A.select(1, l);          // [1, C, T']
        auto alpha = grad_l.mean(2, true);     // [1, C, 1]
        auto cam = (alpha * feat_l).sum(1).squeeze(0); // [T']
        auto cam_a = cam.accessor<float, 1>();

        for (int i = 0; i < input_length; i++) {
            float src = (float)i * (Tp - 1) / std::max(input_length - 1, 1);
            int lo = std::min((int)src, Tp - 1);
            int hi = std::min(lo + 1, Tp - 1);
            float frac = src - lo;
            res_a[l][i] = cam_a[lo] * (1.0f - frac) + cam_a[hi] * frac;
        }
    }

    std::fprintf(stderr, "[grad-cam] OK: C=%d T'=%d range=[%.4f, %.4f]\n",
        C, Tp, result.min().item<float>(), result.max().item<float>());

    return result.detach();
}

static void run_step_inference(torch::jit::Module& model,
                               ECGSample& samp,
                               InferenceOverlay& inf,
                               std::vector<torch::Tensor>& detail) {
    if (!samp.processed_valid) return;

    const int nl = (int)samp.processed.size();
    if (nl < NUM_LEADS) return;
    const int n = samp.num_samples;
    if (n <= 0) return;

    for (int l = 0; l < nl; l++)
        if ((int)samp.processed[l].size() < n) return;

    auto input = torch::zeros({1, nl, n});
    {
        auto acc = input.accessor<float, 3>();
        for (int l = 0; l < nl; l++)
            for (int i = 0; i < n; i++)
                acc[0][l][i] = samp.processed[l][i];
    }

    inf.valid = false;
    inf.layers.clear();
    inf.layers.resize(9);
    inf.sample_id = samp.file_id;
    detail.clear();
    detail.resize(9);

    inf.layers[0] = tensor_stats(input);

    try {
        torch::Tensor logits;

        {
            torch::NoGradGuard no_grad;
            // Reshape to per-lead: (B*12, 1, T)
            auto x = input.reshape({1 * nl, 1, n});

            // Step through backbone in groups of 3 (Conv + BN + Mish = one stage)
            auto backbone = model.attr("backbone").toModule();
            for (int si = 0; si < 3; si++) {
                int base = si * 3;
                x = backbone.attr(std::to_string(base)).toModule().forward({x}).toTensor();
                x = backbone.attr(std::to_string(base + 1)).toModule().forward({x}).toTensor();
                x = backbone.attr(std::to_string(base + 2)).toModule().forward({x}).toTensor();
                // Reshape to (nl, C, T') for stats and detail
                auto stage_out = x.reshape({nl, x.size(1), x.size(2)});
                inf.layers[1 + si] = tensor_stats(stage_out.unsqueeze(0));
                detail[1 + si] = stage_out;  // (12, C, T')
            }

            // Pool per-lead
            x = model.attr("pool").toModule().forward({x}).toTensor().squeeze(-1);
            auto pool_out = x.reshape({nl, x.size(1)});  // (12, 48)
            inf.layers[4] = tensor_stats(pool_out.unsqueeze(0));
            detail[4] = pool_out;

            // Lead concatenation: (1, 576)
            x = x.reshape({1, nl * x.size(1)});
            inf.layers[5] = tensor_stats(x);
            detail[5] = x.squeeze(0);

            // Dropout
            x = model.attr("head_drop").toModule().forward({x}).toTensor();
            inf.layers[6] = tensor_stats(x);
            detail[6] = x.squeeze(0);

            // FC
            logits = model.attr("fc").toModule().forward({x}).toTensor();
            inf.layers[7] = tensor_stats(logits);
            detail[7] = logits.squeeze(0);

            // Softmax
            auto probs = torch::softmax(logits, 1);
            inf.layers[8] = tensor_stats(probs);
            detail[8] = probs.squeeze(0);
            auto pa = probs.accessor<float, 2>();
            inf.probs[0] = pa[0][0];
            inf.probs[1] = pa[0][1];
            inf.result_class = pa[0][1] > pa[0][0] ? 1 : 0;
            inf.valid = true;
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[model] Step-through failed (%s), trying whole-model\n",
                     e.what());
        try {
            torch::NoGradGuard no_grad;
            auto out = model.forward({input}).toTensor();
            auto probs = torch::softmax(out, 1);
            auto pa = probs.accessor<float, 2>();
            inf.probs[0] = pa[0][0];
            inf.probs[1] = pa[0][1];
            inf.result_class = pa[0][1] > pa[0][0] ? 1 : 0;
            inf.layers[8] = tensor_stats(probs);
            inf.valid = true;
        } catch (const std::exception& e2) {
            std::fprintf(stderr, "[model] Inference failed: %s\n", e2.what());
        }
    }

    if (inf.valid) {
        try {
            auto cam = compute_grad_cam(model, input, 1, n);
            detail[0] = cam;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[grad-cam] Failed: %s\n", e.what());
        }
    }
}

static void extract_weights(torch::jit::Module& model,
                            std::vector<WeightEntry>& out) {
    out.clear();
    try {
        auto backbone = model.attr("backbone").toModule();
        const char* stage_names[] = {"Stage 0", "Stage 1", "Stage 2"};
        for (int i = 0; i < 3; i++) {
            auto conv = backbone.attr(std::to_string(i * 3)).toModule();
            auto cw = conv.attr("weight").toTensor();
            out.push_back({std::string(stage_names[i]) + " Conv1d kernels "
                + std::to_string(cw.size(0)) + "x" + std::to_string(cw.size(1))
                + "x" + std::to_string(cw.size(2)),
                cw.detach().flatten(0, 1), 0});
        }

        auto fc_w = model.attr("fc").toModule().attr("weight").toTensor();
        out.push_back({"FC classifier " + std::to_string(fc_w.size(0))
            + "x" + std::to_string(fc_w.size(1)), fc_w.detach(), 0});

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[model] Weight extraction: %s\n", e.what());
    }
}

void RepNetDemoApplet::draw_model_tab() {
    auto& s = *s_;

    if (!s.viz)
        s.viz = std::make_unique<ModelVisualizer>();

    // ── Top bar: model loading ──
    if (ImGui::Button("Load Model...", ImVec2(130, 0))) {
        IGFD::FileDialogConfig cfg;
        cfg.path = s.current_dir.empty() ? "." : s.current_dir;
        ImGuiFileDialog::Instance()->OpenDialog(
            "LoadModelDlg", "Select TorchScript Model", ".pt,.pth", cfg);
    }

    if (ImGuiFileDialog::Instance()->Display("LoadModelDlg",
            ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            s.model_error.clear();
            s.inference = InferenceOverlay{};
            s.inference_sample_idx = -1;
            try {
                s.model = torch::jit::load(path);
                s.model.eval();
                s.model_loaded = true;
                s.model_path = path;
                extract_weights(s.model, s.weight_entries);
                s.weights_extracted = true;
                s.batch_stale = true;
                std::ofstream f(caliper::app_data_path("last_model.txt"));
                if (f.is_open()) f << path;
            } catch (const c10::Error& e) {
                s.model_loaded = false;
                s.model_error = e.what();
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::SameLine();

    if (s.model_loaded) {
        ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f}, "Model:");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", fs::path(s.model_path).filename().string().c_str());

        // Live inference on sample change
        bool has_sample = s.selected >= 0 && s.selected < (int)s.samples.size()
                          && s.samples[s.selected].processed_valid;
        if (has_sample && (s.inference_sample_idx != s.selected ||
                           s.inference_params_ver != s.params.version)) {
            run_step_inference(s.model, s.samples[s.selected],
                               s.inference, s.detail_acts);
            s.inference_sample_idx = s.selected;
            s.inference_params_ver = s.params.version;
            s.detail_texs_dirty = true;
        }

        if (s.inference.valid) {
            ImGui::SameLine();
            ImGui::Text("|");
            ImGui::SameLine();
            ImGui::TextDisabled("Sample: %s", s.inference.sample_id.c_str());
            ImGui::SameLine();
            ImGui::Text("->");
            ImGui::SameLine();
            if (s.inference.result_class == 1)
                ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f},
                    "PE (%.1f%%)", s.inference.probs[1] * 100);
            else
                ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f},
                    "Normal (%.1f%%)", s.inference.probs[0] * 100);
        } else if (has_sample) {
            ImGui::SameLine();
            ImGui::TextDisabled("| Processing...");
        } else {
            ImGui::SameLine();
            ImGui::TextDisabled("| Select a sample for live inference");
        }
    } else {
        if (!s.model_error.empty()) {
            ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "Load failed.");
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::TextWrapped("%s", s.model_error.c_str());
                if (s.model_error.find("constants.pkl") != std::string::npos)
                    ImGui::TextColored({1.0f, 0.85f, 0.3f, 1.0f},
                        "File appears to be a state_dict. Use "
                        "scripts/export_torchscript.py to convert.");
                ImGui::EndTooltip();
            }
        } else {
            ImGui::TextDisabled("No model loaded — load a TorchScript (.pt) for live inference");
        }
    }

    ImGui::Separator();

    if (ImGui::BeginTabBar("##ModelViews")) {
        if (ImGui::BeginTabItem("Data Flow")) {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            const InferenceOverlay* overlay =
                s.inference.valid ? &s.inference : nullptr;
            s.viz->draw(avail, overlay);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Activations")) {
            draw_activation_detail();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Weights")) {
            draw_weight_view();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Statistics")) {
            draw_statistics_tab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

// ============================================================================
// ACTIVATION DETAIL VIEW
// ============================================================================

static const char* kLayerLabels[] = {
    "Input: 12-Lead ECG",
    "Stage 0 — Conv + BN + Mish (k=31)",
    "Stage 1 — Conv + BN + Mish (k=21)",
    "Stage 2 — Conv + BN + Mish (k=11)",
    "Global Average Pool",
    "Lead Concatenation",
    "Dropout",
    "Classifier (FC)",
    "Output Probabilities",
};

static const char* kLeadNames12[] = {
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
};

void RepNetDemoApplet::draw_activation_detail() {
    auto& s = *s_;

    if (!s.inference.valid || s.detail_acts.empty()
        || s.inference_sample_idx != s.selected) {
        ImGui::TextDisabled("Load a model and select a sample to view activations.");
        return;
    }

    // ── Lead scrubber ──
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(LEAD_COLORS[s.detail_lead], "%s", kLeadNames12[s.detail_lead]);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.35f);
    if (ImGui::SliderInt("##lead_scrub", &s.detail_lead, 0, 11, kLeadNames12[s.detail_lead]))
        s.detail_texs_dirty = true;
    ImGui::SameLine();
    ImGui::TextDisabled("Sample: %s", s.inference.sample_id.c_str());
    ImGui::SameLine();
    if (s.inference.result_class == 1)
        ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f},
            "-> PE (%.1f%%)", s.inference.probs[1] * 100);
    else
        ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f},
            "-> Normal (%.1f%%)", s.inference.probs[0] * 100);

    ImGui::Separator();

    // ── Regenerate textures if needed ──
    if (s.detail_texs_dirty || s.detail_lead_cached != s.detail_lead) {
        heatmap::release_textures(s.detail_texs);
        s.detail_texs.clear();
        s.detail_texs.resize(9, 0);

        for (int i = 0; i < 9; i++) {
            if (!s.detail_acts[i].defined() || s.detail_acts[i].numel() == 0)
                continue;

            auto t = s.detail_acts[i];
            torch::Tensor t2d;

            if (i == 0) {
                if (t.dim() == 2 && t.size(0) > 1) {
                    int lead = std::min(s.detail_lead, (int)t.size(0) - 1);
                    t2d = t[lead].unsqueeze(0);
                } else if (t.dim() == 2) {
                    t2d = t;
                } else if (t.dim() == 1) {
                    t2d = t.unsqueeze(0);
                } else {
                    continue;
                }
            } else if (t.dim() == 1) {
                t2d = t.unsqueeze(0);
            } else if (t.dim() == 2) {
                t2d = t;
            } else if (t.dim() == 3) {
                int lead = std::min(s.detail_lead, (int)t.size(0) - 1);
                t2d = t[lead];
            } else {
                continue;
            }

            bool diverging = (i == 0);
            bool log_scale = (i == 0);
            s.detail_texs[i] = heatmap::upload_texture(t2d, diverging, log_scale);
        }

        s.detail_texs_dirty = false;
        s.detail_lead_cached = s.detail_lead;
    }

    // ── Compute time axis and duration once ──
    float duration = 0.0f;
    bool has_samp = s.selected >= 0 && s.selected < (int)s.samples.size()
                    && s.samples[s.selected].processed_valid;
    if (has_samp) {
        auto& samp = s.samples[s.selected];
        duration = samp.num_samples / std::max(1.0f, samp.sampling_rate);
        if ((int)s.time_axis.size() != samp.num_samples) {
            s.time_axis.resize(samp.num_samples);
            for (int j = 0; j < samp.num_samples; j++)
                s.time_axis[j] = (float)j / samp.sampling_rate;
        }
    }

    // Count temporal subplot rows: waveform + each activation layer with real
    // temporal extent (exclude 2-element classifier/output — those are per-class,
    // not temporal, and belong in the non-temporal section below).
    int n_subplot_rows = 0;
    std::vector<int> temporal_layers;
    if (has_samp && duration > 0.0f) {
        n_subplot_rows = 1; // waveform row
        for (int i = 1; i < 9; i++) {
            if (!s.detail_acts[i].defined()) continue;
            auto& t = s.detail_acts[i];
            torch::Tensor t2d;
            if (t.dim() == 1) t2d = t.unsqueeze(0);
            else if (t.dim() == 2) t2d = t;
            else if (t.dim() == 3) t2d = t[std::min(s.detail_lead, (int)t.size(0)-1)];
            int cols = t2d.dim() >= 2 ? (int)t2d.size(1) : 1;
            if (cols > 2 && s.detail_texs[i]) {
                temporal_layers.push_back(i);
                n_subplot_rows++;
            }
        }
    }

    // ── Scrollable layer view ──
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("##act_scroll", avail, false);

    float content_w = ImGui::GetContentRegionAvail().x;
    float hm_w = std::max(200.0f, content_w - 8.0f);

    if (n_subplot_rows > 0 && has_samp) {
        // Compute row heights: waveform gets 130px, each layer gets dynamic height
        std::vector<float> row_ratios;
        float total_h = 130.0f;
        row_ratios.push_back(130.0f);

        for (int li : temporal_layers) {
            auto& t = s.detail_acts[li];
            torch::Tensor t2d;
            if (t.dim() == 1) t2d = t.unsqueeze(0);
            else if (t.dim() == 2) t2d = t;
            else if (t.dim() == 3) t2d = t[std::min(s.detail_lead, (int)t.size(0)-1)];
            int rows = (int)t2d.size(0);

            float hm_h;
            if (rows <= 2)       hm_h = 40.0f;
            else if (rows <= 16) hm_h = std::max(60.0f, (float)rows * 5.0f);
            else if (rows <= 64) hm_h = std::max(80.0f, (float)rows * 2.5f);
            else                 hm_h = std::min(200.0f, (float)rows * 1.5f);

            row_ratios.push_back(hm_h);
            total_h += hm_h;
        }

        // Normalize ratios
        for (auto& r : row_ratios) r /= total_h;

        float subplot_h = std::max(total_h + n_subplot_rows * 30.0f, 300.0f);

        if (ImPlot::BeginSubplots("##activation_flow", n_subplot_rows, 1,
                ImVec2(hm_w, subplot_h),
                ImPlotSubplotFlags_LinkAllX | ImPlotSubplotFlags_NoResize,
                row_ratios.data())) {

            // Row 0: waveform + saliency overlay
            if (ImPlot::BeginPlot("##waveform", ImVec2(), ImPlotFlags_NoLegend)) {
                auto& samp = s.samples[s.selected];
                int lead = s.detail_lead;
                bool lead_ok = lead < (int)samp.stats.size()
                    && lead < (int)samp.processed.size()
                    && !samp.processed[lead].empty();
                int plot_n = lead_ok
                    ? std::min(samp.num_samples, (int)samp.processed[lead].size()) : 0;
                float y_lo = -1, y_hi = 1;
                if (lead_ok) {
                    auto& st = samp.stats[lead];
                    float margin = (st.max_val - st.min_val) * 0.1f;
                    y_lo = st.min_val - margin;
                    y_hi = st.max_val + margin;
                }

                ImPlot::SetupAxes(nullptr, nullptr,
                    ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels,
                    ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, y_lo, y_hi, ImGuiCond_Always);

                if (s.detail_texs[0]) {
                    ImPlot::PlotImage("##sal",
                        (ImTextureID)(intptr_t)s.detail_texs[0],
                        ImPlotPoint(0, y_hi), ImPlotPoint(duration, y_lo),
                        ImVec2(0, 0), ImVec2(1, 1),
                        ImVec4(1, 1, 1, 0.45f));
                }

                if (plot_n > 0 && plot_n <= (int)s.time_axis.size())
                ImPlot::PlotLine("##sig", s.time_axis.data(),
                    samp.processed[lead].data(), plot_n,
                    ImPlotSpec(ImPlotProp_LineColor, LEAD_COLORS[lead],
                               ImPlotProp_LineWeight, 1.5f));

                ImPlot::Annotation(0.0, y_hi, LEAD_COLORS[lead],
                    ImVec2(5, 5), false, "%s  Grad-CAM", kLeadNames12[lead]);

                ImPlot::EndPlot();
            }

            // Remaining rows: temporal activation heatmaps
            for (int li : temporal_layers) {
                auto& t = s.detail_acts[li];
                torch::Tensor t2d;
                if (t.dim() == 1) t2d = t.unsqueeze(0);
                else if (t.dim() == 2) t2d = t;
                else if (t.dim() == 3) t2d = t[std::min(s.detail_lead, (int)t.size(0)-1)];
                int rows = (int)t2d.size(0);

                char plot_id[32];
                std::snprintf(plot_id, sizeof(plot_id), "##layer_%d", li);
                if (ImPlot::BeginPlot(plot_id, ImVec2(), ImPlotFlags_NoLegend)) {
                    ImPlot::SetupAxes(nullptr, nullptr,
                        ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels,
                        ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, rows, ImGuiCond_Always);

                    ImPlot::PlotImage("##hm",
                        (ImTextureID)(intptr_t)s.detail_texs[li],
                        ImPlotPoint(0, rows), ImPlotPoint(duration, 0));

                    ImPlot::Annotation(0.0, (double)rows, ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                        ImVec2(5, 5), false, "%s", kLayerLabels[li]);

                    if ((li == 7 || li == 8) && rows == 1) {
                        ImPlot::Annotation(duration * 0.25, 0.5,
                            ImVec4(1.0f, 1.0f, 1.0f, 0.9f), ImVec2(0, 0), true, "Normal");
                        ImPlot::Annotation(duration * 0.75, 0.5,
                            ImVec4(1.0f, 1.0f, 1.0f, 0.9f), ImVec2(0, 0), true, "PE");
                    }

                    ImPlot::EndPlot();
                }
            }

            ImPlot::EndSubplots();
        }
    }

    // ── Non-temporal layers (GAP, Dropout, FC, Probs) ──
    for (int i = 0; i < 9; i++) {
        if (i == 0) continue;
        if (!s.detail_acts[i].defined()) continue;

        auto& t = s.detail_acts[i];
        torch::Tensor t2d;
        if (t.dim() == 1) t2d = t.unsqueeze(0);
        else if (t.dim() == 2) t2d = t;
        else if (t.dim() == 3) t2d = t[std::min(s.detail_lead, (int)t.size(0)-1)];
        int cols = t2d.dim() >= 2 ? (int)t2d.size(1) : 1;

        bool is_temporal = (cols > 2 && duration > 0.0f);
        if (is_temporal) continue;

        auto& la = s.inference.layers[i];

        ImGui::PushID(i);

        // Classifier logits / output probs — color intensity scales with weight
        if ((i == 7 || i == 8) && s.detail_acts[i].defined() && s.detail_acts[i].numel() == 2) {
            auto vals = s.detail_acts[i].contiguous().to(torch::kCPU, torch::kFloat);
            auto va = vals.accessor<float, 1>();
            float v0 = va[0], v1 = va[1];

            float p0, p1;
            if (i == 8) {
                p0 = v0; p1 = v1;
            } else {
                float e0 = std::exp(v0), e1 = std::exp(v1);
                float es = e0 + e1;
                p0 = e0 / es; p1 = e1 / es;
            }

            uint8_t c0r, c0g, c0b, c1r, c1g, c1b;
            heatmap::colormap(p0, c0r, c0g, c0b, false);
            heatmap::colormap(p1, c1r, c1g, c1b, false);

            ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "%s", kLayerLabels[i]);
            ImGui::SameLine();
            if (i == 8)
                ImGui::TextDisabled("Normal: %.1f%%  PE: %.1f%%", v0 * 100, v1 * 100);
            else
                ImGui::TextDisabled("Normal: %.4f  PE: %.4f", v0, v1);

            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();

            if (i == 7) {
                // Classifier logits — equal-width cells, color = activation weight
                float field_h = 32.0f;
                float half_w = hm_w * 0.5f;

                dl->AddRectFilled(pos,
                    ImVec2(pos.x + half_w, pos.y + field_h),
                    IM_COL32(c0r, c0g, c0b, 255));
                dl->AddRectFilled(ImVec2(pos.x + half_w, pos.y),
                    ImVec2(pos.x + hm_w, pos.y + field_h),
                    IM_COL32(c1r, c1g, c1b, 255));

                char lbl0[48], lbl1[48];
                std::snprintf(lbl0, sizeof(lbl0), "Normal %.3f", v0);
                std::snprintf(lbl1, sizeof(lbl1), "PE %.3f", v1);
                dl->AddText(ImVec2(pos.x + 6, pos.y + 9),
                    IM_COL32(255, 255, 255, 220), lbl0);
                ImVec2 ts1 = ImGui::CalcTextSize(lbl1);
                dl->AddText(ImVec2(pos.x + hm_w - ts1.x - 6, pos.y + 9),
                    IM_COL32(255, 255, 255, 220), lbl1);

                ImGui::Dummy(ImVec2(hm_w, field_h));
            } else {
                // Output probabilities — width proportional to probability
                float field_h = 38.0f;
                float w0 = hm_w * p0;

                dl->AddRectFilled(pos,
                    ImVec2(pos.x + w0, pos.y + field_h),
                    IM_COL32(c0r, c0g, c0b, 255));
                dl->AddRectFilled(ImVec2(pos.x + w0, pos.y),
                    ImVec2(pos.x + hm_w, pos.y + field_h),
                    IM_COL32(c1r, c1g, c1b, 255));

                dl->AddLine(ImVec2(pos.x + w0, pos.y),
                    ImVec2(pos.x + w0, pos.y + field_h),
                    IM_COL32(255, 255, 255, 100), 1.5f);

                char lbl0[48], lbl1[48];
                std::snprintf(lbl0, sizeof(lbl0), "Normal %.1f%%", p0 * 100);
                std::snprintf(lbl1, sizeof(lbl1), "PE %.1f%%", p1 * 100);
                dl->AddText(ImVec2(pos.x + 6, pos.y + 12),
                    IM_COL32(255, 255, 255, 230), lbl0);
                ImVec2 ts1 = ImGui::CalcTextSize(lbl1);
                dl->AddText(ImVec2(pos.x + hm_w - ts1.x - 6, pos.y + 12),
                    IM_COL32(255, 255, 255, 230), lbl1);

                ImGui::Dummy(ImVec2(hm_w, field_h));
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PopID();
            continue;
        }

        ImVec4 hdr_col = {0.6f, 0.8f, 1.0f, 1.0f};
        ImGui::TextColored(hdr_col, "%s", kLayerLabels[i]);
        ImGui::SameLine();
        if (la.valid)
            ImGui::TextDisabled("shape: %s  |  mu=%.4f  sigma=%.4f  [%.3f, %.3f]",
                la.shape.c_str(), la.mean, la.stddev, la.min_val, la.max_val);

        if (s.detail_texs[i]) {
            int rows = (int)t2d.size(0);
            float hm_h = (rows <= 2) ? 32.0f : std::min(100.0f, (float)rows * 4.0f);
            ImGui::Image((ImTextureID)(intptr_t)s.detail_texs[i],
                         ImVec2(hm_w, hm_h));
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PopID();
    }

    ImGui::EndChild();
}

// ============================================================================
// WEIGHT VISUALIZATION VIEW
// ============================================================================

void RepNetDemoApplet::draw_weight_view() {
    auto& s = *s_;

    if (!s.weights_extracted || s.weight_entries.empty()) {
        ImGui::TextDisabled("Load a model to inspect its weights.");
        return;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("##weight_scroll", avail, false);

    float content_w = ImGui::GetContentRegionAvail().x;
    float hm_w = std::max(200.0f, content_w - 24.0f);

    for (int i = 0; i < (int)s.weight_entries.size(); i++) {
        auto& w = s.weight_entries[i];
        ImGui::PushID(i);

        auto t = w.tensor.contiguous().to(torch::kFloat);
        if (t.dim() == 1) t = t.unsqueeze(0);
        int rows = (int)t.size(0);
        int cols = t.dim() >= 2 ? (int)t.size(1) : 1;
        int64_t numel = t.numel();

        float wmin = t.min().item<float>();
        float wmax = t.max().item<float>();
        float wmean = t.mean().item<float>();
        float wstd = t.std().item<float>();

        if (!w.tex)
            w.tex = heatmap::upload_texture(t, true);

        ImGui::TextColored({0.85f, 0.75f, 1.0f, 1.0f}, "%s", w.label.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%dx%d (%lld params)  mu=%.4f  sigma=%.4f  [%.3f, %.3f]",
            rows, cols, (long long)numel, wmean, wstd, wmin, wmax);

        if (w.tex) {
            float hm_h;
            if (rows <= 2)
                hm_h = 32.0f;
            else if (rows <= 16)
                hm_h = std::max(60.0f, (float)rows * 5.0f);
            else if (rows <= 64)
                hm_h = std::max(80.0f, (float)rows * 2.5f);
            else
                hm_h = std::min(200.0f, (float)rows * 1.5f);

            ImGui::Image((ImTextureID)(intptr_t)w.tex, ImVec2(hm_w, hm_h));

            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 lp = ImGui::GetCursorScreenPos();
            float leg_w = std::min(200.0f, hm_w * 0.4f);
            float leg_h = 10.0f;
            for (int p = 0; p < (int)leg_w; p++) {
                float ft = (float)p / leg_w;
                uint8_t cr, cg, cb;
                heatmap::colormap(ft, cr, cg, cb, true);
                dl->AddRectFilled(
                    ImVec2(lp.x + p, lp.y),
                    ImVec2(lp.x + p + 1, lp.y + leg_h),
                    IM_COL32(cr, cg, cb, 255));
            }
            char lmin[32], lmax[32];
            std::snprintf(lmin, sizeof(lmin), "%.3f", wmin);
            std::snprintf(lmax, sizeof(lmax), "%.3f", wmax);
            dl->AddText(ImVec2(lp.x, lp.y + leg_h + 1),
                        IM_COL32(180, 190, 200, 200), lmin);
            ImVec2 ts = ImGui::CalcTextSize(lmax);
            dl->AddText(ImVec2(lp.x + leg_w - ts.x, lp.y + leg_h + 1),
                        IM_COL32(180, 190, 200, 200), lmax);
            ImGui::Dummy(ImVec2(leg_w, leg_h + 18));
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PopID();
    }

    ImGui::EndChild();
}

// ============================================================================
// STATISTICS TAB
// ============================================================================

void RepNetDemoApplet::draw_statistics_tab() {
    auto& s = *s_;

    if (!s.model_loaded) {
        ImGui::TextDisabled("Load a model to compute statistics.");
        return;
    }
    if (s.samples.empty()) {
        ImGui::TextDisabled("Open a dataset to compute statistics.");
        return;
    }

    // Count how many samples are processed
    int n_ready = 0;
    for (auto& samp : s.samples)
        if (samp.processed_valid) n_ready++;

    if (n_ready == 0) {
        ImGui::TextDisabled("Waiting for samples to load...");
        return;
    }

    // Run batch button
    if (s.batch_stale) {
        char btn_label[64];
        std::snprintf(btn_label, sizeof(btn_label), "Run Inference on %d Samples", n_ready);
        if (ImGui::Button(btn_label)) {
            s.batch_results.clear();
            s.batch_results.reserve(n_ready);

            torch::NoGradGuard no_grad;
            for (auto& samp : s.samples) {
                if (!samp.processed_valid) continue;

                State::SampleResult r;
                r.id = samp.file_id;

                if (!samp.label.empty()) {
                    r.true_class = (samp.label.find("Normal") == std::string::npos) ? 1 : 0;
                }

                try {
                    const int nl = (int)samp.processed.size();
                    const int n = samp.num_samples;
                    auto input = torch::zeros({1, nl, n});
                    auto acc = input.accessor<float, 3>();
                    for (int l = 0; l < nl; l++)
                        for (int i = 0; i < n; i++)
                            acc[0][l][i] = samp.processed[l][i];

                    auto x = input.reshape({nl, 1, n});
                    x = s.model.attr("backbone").toModule().forward({x}).toTensor();
                    x = s.model.attr("pool").toModule().forward({x}).toTensor().squeeze(-1);
                    // x: (12, 48) — concat to (1, 576)
                    auto gap = x.reshape({1, nl * x.size(1)});

                    auto gap_acc = gap.accessor<float, 2>();
                    int feat_dim = std::min(576, (int)gap.size(1));
                    for (int fi = 0; fi < feat_dim; fi++)
                        r.gap_feat[fi] = gap_acc[0][fi];

                    x = s.model.attr("head_drop").toModule().forward({gap}).toTensor();
                    x = s.model.attr("fc").toModule().forward({x}).toTensor();
                    auto probs = torch::softmax(x, 1);
                    auto pa = probs.accessor<float, 2>();
                    r.prob_pe = pa[0][1];
                    r.pred_class = pa[0][1] > pa[0][0] ? 1 : 0;
                } catch (...) {
                    r.pred_class = -1;
                }

                s.batch_results.push_back(r);
            }

            // ── PCA on GAP features ──
            {
                int n_valid = 0;
                for (auto& r : s.batch_results)
                    if (r.pred_class >= 0) n_valid++;

                if (n_valid >= 2) {
                    auto mat = torch::zeros({n_valid, 576});
                    auto ma = mat.accessor<float, 2>();
                    int ri = 0;
                    for (auto& r : s.batch_results) {
                        if (r.pred_class < 0) continue;
                        for (int f = 0; f < 576; f++)
                            ma[ri][f] = r.gap_feat[f];
                        ri++;
                    }

                    auto mean = mat.mean(0, true);
                    auto centered = mat - mean;
                    auto cov = centered.t().mm(centered) / (float)(n_valid - 1);
                    auto eig = torch::linalg_eigh(cov);
                    auto eigvecs = std::get<1>(eig);
                    auto pc = eigvecs.index({torch::indexing::Slice(),
                        torch::indexing::Slice(-2, torch::indexing::None)}).flip(1);
                    auto proj = centered.mm(pc);
                    auto pa2 = proj.accessor<float, 2>();

                    s.pca_x.resize(n_valid);
                    s.pca_y.resize(n_valid);
                    for (int i = 0; i < n_valid; i++) {
                        s.pca_x[i] = pa2[i][0];
                        s.pca_y[i] = pa2[i][1];
                    }
                }
            }

            s.batch_stale = false;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(%d samples ready)", n_ready);
        return;
    }

    if (s.batch_results.empty()) return;

    // ── Compute statistics ──
    int tp = 0, fp = 0, tn = 0, fn = 0;
    int pred_pos = 0, pred_neg = 0;
    int true_pos_count = 0, true_neg_count = 0;
    int labeled = 0;
    std::vector<float> probs_all, probs_pos, probs_neg;

    for (auto& r : s.batch_results) {
        if (r.pred_class < 0) continue;
        probs_all.push_back(r.prob_pe);

        if (r.pred_class == 1) pred_pos++;
        else pred_neg++;

        if (r.true_class >= 0) {
            labeled++;
            if (r.true_class == 1) {
                true_pos_count++;
                probs_pos.push_back(r.prob_pe);
                if (r.pred_class == 1) tp++; else fn++;
            } else {
                true_neg_count++;
                probs_neg.push_back(r.prob_pe);
                if (r.pred_class == 0) tn++; else fp++;
            }
        }
    }

    int total = (int)s.batch_results.size();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("##stats_scroll", avail, false);

    float content_w = ImGui::GetContentRegionAvail().x;

    // ── Summary ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "Dataset Summary");
    ImGui::Separator();
    ImGui::Text("Total samples: %d", total);
    ImGui::Text("Predictions:   %d PE  /  %d Normal", pred_pos, pred_neg);
    if (labeled > 0) {
        ImGui::Text("Ground truth:  %d PE  /  %d Normal  (%d labeled)",
            true_pos_count, true_neg_count, labeled);
    } else {
        ImGui::TextDisabled("No ground truth labels available");
    }
    ImGui::Spacing();

    // ── Classification Report ──
    if (labeled > 0) {
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "Classification Report");
        ImGui::Separator();

        float accuracy = (labeled > 0) ? (float)(tp + tn) / labeled : 0;
        float precision_pe = (tp + fp > 0) ? (float)tp / (tp + fp) : 0;
        float recall_pe    = (tp + fn > 0) ? (float)tp / (tp + fn) : 0;
        float f1_pe = (precision_pe + recall_pe > 0)
            ? 2 * precision_pe * recall_pe / (precision_pe + recall_pe) : 0;
        float precision_n = (tn + fn > 0) ? (float)tn / (tn + fn) : 0;
        float recall_n    = (tn + fp > 0) ? (float)tn / (tn + fp) : 0;
        float f1_n = (precision_n + recall_n > 0)
            ? 2 * precision_n * recall_n / (precision_n + recall_n) : 0;

        ImGui::Spacing();
        if (ImGui::BeginTable("##cls_report", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Precision", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Recall", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("F1-Score", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Support", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableHeadersRow();

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Normal");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", precision_n);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", recall_n);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", f1_n);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%d", true_neg_count);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("PE");
            ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", precision_pe);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%.3f", recall_pe);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%.3f", f1_pe);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%d", true_pos_count);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::TextColored({0.8f, 0.9f, 1.0f, 1.0f}, "Accuracy");
            ImGui::TableSetColumnIndex(3); ImGui::TextColored({0.8f, 0.9f, 1.0f, 1.0f}, "%.3f", accuracy);
            ImGui::TableSetColumnIndex(4); ImGui::TextColored({0.8f, 0.9f, 1.0f, 1.0f}, "%d", labeled);

            ImGui::EndTable();
        }

        ImGui::Spacing();

        // ── Confusion Matrix ──
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "Confusion Matrix");
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::BeginTable("##conf_mat", 3,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Pred Normal", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Pred PE", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableHeadersRow();

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("True Normal");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f}, "%d", tn);
            ImGui::TableSetColumnIndex(2);
            if (fp > 0) ImGui::TextColored({1.0f, 0.6f, 0.3f, 1.0f}, "%d", fp);
            else ImGui::Text("%d", fp);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("True PE");
            ImGui::TableSetColumnIndex(1);
            if (fn > 0) ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "%d", fn);
            else ImGui::Text("%d", fn);
            ImGui::TableSetColumnIndex(2);
            ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f}, "%d", tp);

            ImGui::EndTable();
        }

        ImGui::Spacing();
    }

    // ── Probability Distribution ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "PE Probability Distribution");
    ImGui::Separator();
    ImGui::Spacing();

    float plot_w = std::max(300.0f, content_w - 24.0f);
    float plot_h = 200.0f;

    if (labeled > 0 && (!probs_pos.empty() || !probs_neg.empty())) {
        if (ImPlot::BeginPlot("##prob_dist", ImVec2(plot_w, plot_h))) {
            ImPlot::SetupAxes("P(PE)", "Count");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImGuiCond_Always);

            if (!probs_neg.empty()) {
                ImPlot::PlotHistogram("Normal", probs_neg.data(),
                    (int)probs_neg.size(), 20, 1.0, ImPlotRange(0, 1),
                    ImPlotSpec(ImPlotProp_FillColor, ImVec4(0.3f, 0.7f, 1.0f, 1.0f),
                               ImPlotProp_FillAlpha, 0.5f));
            }
            if (!probs_pos.empty()) {
                ImPlot::PlotHistogram("PE", probs_pos.data(),
                    (int)probs_pos.size(), 20, 1.0, ImPlotRange(0, 1),
                    ImPlotSpec(ImPlotProp_FillColor, ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               ImPlotProp_FillAlpha, 0.5f));
            }
            ImPlot::EndPlot();
        }
    } else {
        if (ImPlot::BeginPlot("##prob_dist_all", ImVec2(plot_w, plot_h))) {
            ImPlot::SetupAxes("P(PE)", "Count");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImGuiCond_Always);
            ImPlot::PlotHistogram("All", probs_all.data(),
                (int)probs_all.size(), 20, 1.0, ImPlotRange(0, 1),
                ImPlotSpec(ImPlotProp_FillColor, ImVec4(0.5f, 0.7f, 1.0f, 1.0f),
                           ImPlotProp_FillAlpha, 0.6f));
            ImPlot::EndPlot();
        }
    }

    ImGui::Spacing();

    // ── PCA Scatter ──
    if ((int)s.pca_x.size() >= 2) {
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f},
            "PCA — 576-dim Feature Space (GAP Layer)");
        ImGui::Separator();
        ImGui::Spacing();

        std::vector<float> px_n, py_n, px_p, py_p, px_u, py_u;
        int vi = 0;
        for (auto& r : s.batch_results) {
            if (r.pred_class < 0) continue;
            if (r.true_class == 0) {
                px_n.push_back(s.pca_x[vi]); py_n.push_back(s.pca_y[vi]);
            } else if (r.true_class == 1) {
                px_p.push_back(s.pca_x[vi]); py_p.push_back(s.pca_y[vi]);
            } else {
                px_u.push_back(s.pca_x[vi]); py_u.push_back(s.pca_y[vi]);
            }
            vi++;
        }

        if (ImPlot::BeginPlot("##pca_scatter", ImVec2(plot_w, 300.0f))) {
            ImPlot::SetupAxes("PC1", "PC2");

            if (!px_n.empty())
                ImPlot::PlotScatter("Normal", px_n.data(), py_n.data(), (int)px_n.size(),
                    ImPlotSpec(ImPlotProp_Marker, (double)ImPlotMarker_Circle,
                               ImPlotProp_MarkerSize, 5.0,
                               ImPlotProp_MarkerFillColor, ImVec4(0.3f, 0.7f, 1.0f, 0.8f)));
            if (!px_p.empty())
                ImPlot::PlotScatter("PE", px_p.data(), py_p.data(), (int)px_p.size(),
                    ImPlotSpec(ImPlotProp_Marker, (double)ImPlotMarker_Circle,
                               ImPlotProp_MarkerSize, 5.0,
                               ImPlotProp_MarkerFillColor, ImVec4(1.0f, 0.3f, 0.3f, 0.8f)));
            if (!px_u.empty())
                ImPlot::PlotScatter("Unknown", px_u.data(), py_u.data(), (int)px_u.size(),
                    ImPlotSpec(ImPlotProp_Marker, (double)ImPlotMarker_Circle,
                               ImPlotProp_MarkerSize, 4.0,
                               ImPlotProp_MarkerFillColor, ImVec4(0.6f, 0.6f, 0.6f, 0.6f)));

            ImPlot::EndPlot();
        }
        ImGui::Spacing();
    }

    // ── ROC Curve ──
    if (labeled > 0 && !probs_pos.empty() && !probs_neg.empty()) {
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "ROC Curve");
        ImGui::Separator();
        ImGui::Spacing();

        std::vector<float> thresholds;
        for (int t = 0; t <= 100; t++)
            thresholds.push_back((float)t / 100.0f);

        std::vector<float> fpr_v, tpr_v;
        for (float th : thresholds) {
            int tp_t = 0, fp_t = 0, fn_t = 0, tn_t = 0;
            for (auto& r : s.batch_results) {
                if (r.pred_class < 0 || r.true_class < 0) continue;
                int pred = r.prob_pe >= th ? 1 : 0;
                if (r.true_class == 1 && pred == 1) tp_t++;
                else if (r.true_class == 0 && pred == 1) fp_t++;
                else if (r.true_class == 1 && pred == 0) fn_t++;
                else tn_t++;
            }
            float fpr_val = (fp_t + tn_t > 0) ? (float)fp_t / (fp_t + tn_t) : 0;
            float tpr_val = (tp_t + fn_t > 0) ? (float)tp_t / (tp_t + fn_t) : 0;
            fpr_v.push_back(fpr_val);
            tpr_v.push_back(tpr_val);
        }

        // AUC via trapezoidal rule
        float auc = 0;
        for (int i = 1; i < (int)fpr_v.size(); i++)
            auc += 0.5f * std::abs(fpr_v[i-1] - fpr_v[i]) * (tpr_v[i-1] + tpr_v[i]);

        if (ImPlot::BeginPlot("##roc", ImVec2(plot_w, 300.0f))) {
            ImPlot::SetupAxes("False Positive Rate", "True Positive Rate");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImGuiCond_Always);

            float diag_x[] = {0, 1}, diag_y[] = {0, 1};
            ImPlot::PlotLine("Random", diag_x, diag_y, 2,
                ImPlotSpec(ImPlotProp_LineColor, ImVec4(0.5f, 0.5f, 0.5f, 0.5f)));

            char roc_label[64];
            std::snprintf(roc_label, sizeof(roc_label), "ROC (AUC=%.3f)", auc);
            ImPlot::PlotLine(roc_label, fpr_v.data(), tpr_v.data(), (int)fpr_v.size(),
                ImPlotSpec(ImPlotProp_LineColor, ImVec4(0.3f, 0.8f, 1.0f, 1.0f),
                           ImPlotProp_LineWeight, 2.0f));

            ImPlot::EndPlot();
        }
        ImGui::Spacing();
    }

    // ── Per-sample results table ──
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "Per-Sample Results");
    ImGui::Separator();
    ImGui::Spacing();

    int n_cols = labeled > 0 ? 5 : 3;
    if (ImGui::BeginTable("##sample_results", n_cols,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY
            | ImGuiTableFlags_Sortable,
            ImVec2(0, std::min(400.0f, (float)(total + 1) * 24.0f)))) {
        ImGui::TableSetupColumn("Sample", ImGuiTableColumnFlags_WidthStretch);
        if (labeled > 0)
            ImGui::TableSetupColumn("True", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Predicted", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("P(PE)", ImGuiTableColumnFlags_WidthFixed, 70);
        if (labeled > 0)
            ImGui::TableSetupColumn("Correct", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableHeadersRow();

        for (auto& r : s.batch_results) {
            if (r.pred_class < 0) continue;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", r.id.c_str());

            int col = 1;
            if (labeled > 0) {
                ImGui::TableSetColumnIndex(col++);
                if (r.true_class == 1)
                    ImGui::TextColored({1.0f, 0.5f, 0.5f, 1.0f}, "PE");
                else if (r.true_class == 0)
                    ImGui::TextColored({0.5f, 1.0f, 0.7f, 1.0f}, "Normal");
                else
                    ImGui::TextDisabled("?");
            }

            ImGui::TableSetColumnIndex(col++);
            if (r.pred_class == 1)
                ImGui::TextColored({1.0f, 0.5f, 0.5f, 1.0f}, "PE");
            else
                ImGui::TextColored({0.5f, 1.0f, 0.7f, 1.0f}, "Normal");

            ImGui::TableSetColumnIndex(col++);
            ImGui::Text("%.3f", r.prob_pe);

            if (labeled > 0) {
                ImGui::TableSetColumnIndex(col++);
                if (r.true_class >= 0) {
                    bool correct = (r.true_class == r.pred_class);
                    if (correct)
                        ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f}, "Yes");
                    else
                        ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "No");
                }
            }
        }
        ImGui::EndTable();
    }

    // ── Refresh button ──
    ImGui::Spacing();
    if (ImGui::Button("Re-run Inference"))
        s.batch_stale = true;

    ImGui::EndChild();
}

