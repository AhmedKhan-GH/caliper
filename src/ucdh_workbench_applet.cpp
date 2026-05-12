#define _USE_MATH_DEFINES
#include "ucdh_workbench_applet.h"
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

void process(ECGSample& sample, const ProcessingParams& params) {
    sample.processed.resize(NUM_LEADS);
    sample.stats.resize(NUM_LEADS);

    for (int lead = 0; lead < NUM_LEADS; lead++) {
        sample.processed[lead] = sample.raw[lead];
        auto& sig = sample.processed[lead];

        if (params.baseline_wander_correction && params.baseline_cutoff_hz > 0 && sample.sampling_rate > 0) {
            butterworth_highpass(sig, params.baseline_cutoff_hz, sample.sampling_rate);
        }

        if (params.zscore) {
            zscore(sig);
        }

        compute_stats(sig, sample.stats[lead]);
    }
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
// BACKGROUND PROCESSOR
// ============================================================================

class BackgroundProcessor {
public:
    BackgroundProcessor() : stop_(false) {
        worker_ = std::thread(&BackgroundProcessor::run, this);
    }

    ~BackgroundProcessor() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
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
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                if (queue_.empty()) continue;

                idx = queue_.front();
                queue_.pop_front();
                params = params_;
                sample = &(*samples_)[idx];
                loader = loader_;
            }

            if (!sample->loaded && loader) {
                loader->load(*sample);
            }

            if (sample->loaded) {
                dsp::process(*sample, params);
            }

            processed_count_.fetch_add(1);
        }
    }

    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
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
    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const auto p = entry.path().string();
        const FormatInfo* fmt = match_format(p);
        if (!fmt) continue;
        DiscoveredFile f;
        f.path = p;
        f.display_name = entry.path().filename().string();
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

struct UCDHWorkbenchApplet::State {
    // ── ECG dataset ──
    std::vector<ECGSample> samples;
    int selected = -1;
    std::unique_ptr<IDatasetLoader> loader;
    std::string current_dir;
    DatasetFormat current_format = DatasetFormat::Auto;

    enum class ScanStatus { Idle, Scanning, Ready, Failed };
    std::atomic<ScanStatus> scan_status{ScanStatus::Idle};
    std::thread scan_thread;
    std::mutex scan_result_mtx;
    std::unique_ptr<IDatasetLoader> pending_loader;
    std::vector<ECGSample> pending_samples;
    DatasetFormat pending_format = DatasetFormat::Auto;
    std::string scan_dir;
    std::string scan_error;

    DatasetFormat pending_fmt_override = DatasetFormat::Auto;

    ProcessingParams params;
    std::unique_ptr<BackgroundProcessor> bg;

    bool lead_visible[NUM_LEADS] = {true,true,true,true,true,true,true,true,true,true,true,true};
    int last_plot_sample = -1;
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
    std::string model_output;

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

UCDHWorkbenchApplet::UCDHWorkbenchApplet() = default;
UCDHWorkbenchApplet::~UCDHWorkbenchApplet() = default;

bool UCDHWorkbenchApplet::initialize() {
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
            open_dataset(last_dir, static_cast<int>(DatasetFormat::Auto));
        }
    }

    return true;
}

void UCDHWorkbenchApplet::cleanup() {
    if (!s_) return;
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

void UCDHWorkbenchApplet::draw_ui(int /*win_w*/, int /*win_h*/) {
    if (!s_) return;

    // ── Commit async scan results ──
    if (s_->scan_status.load() == State::ScanStatus::Ready) {
        if (s_->scan_thread.joinable()) s_->scan_thread.join();
        s_->bg.reset();
        {
            std::lock_guard<std::mutex> lk(s_->scan_result_mtx);
            s_->loader = std::move(s_->pending_loader);
            s_->samples = std::move(s_->pending_samples);
            s_->current_format = s_->pending_format;
            s_->current_dir = s_->scan_dir;
        }
        s_->selected = -1;
        s_->bg = std::make_unique<BackgroundProcessor>();
        s_->scan_status.store(State::ScanStatus::Idle);

        std::cout << "[workbench] Opened " << s_->current_dir << " ("
                  << format_display_name(s_->current_format) << "): "
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
    float panel_w = 240.0f;
    float plot_w = avail_w - panel_w - sp;

    // -- Left panel --
    ImGui::BeginChild("##Panel", ImVec2(panel_w, avail_h), true);
    draw_panel();
    ImGui::EndChild();

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

void UCDHWorkbenchApplet::open_dataset(const std::string& dir, int fmt_override_int) {
    DatasetFormat fmt_override = static_cast<DatasetFormat>(fmt_override_int);
    if (s_->scan_thread.joinable()) s_->scan_thread.join();
    s_->scan_status.store(State::ScanStatus::Scanning);
    s_->scan_error.clear();
    s_->scan_dir = dir;

    auto* sp = s_.get();
    s_->scan_thread = std::thread([sp, dir, fmt_override]() {
        DatasetFormat fmt = (fmt_override == DatasetFormat::Auto)
            ? detect_format(dir) : fmt_override;

        auto loader = make_dataset_loader(fmt);
        if (!loader) {
            std::lock_guard<std::mutex> lk(sp->scan_result_mtx);
            sp->scan_error = "Could not create loader for format";
            sp->scan_status.store(UCDHWorkbenchApplet::State::ScanStatus::Failed);
            return;
        }

        std::vector<ECGSample> samples;
        bool ok = loader->scan(dir, samples);
        if (!ok) {
            std::lock_guard<std::mutex> lk(sp->scan_result_mtx);
            sp->scan_error = std::string("No matching files in: ") + dir;
            sp->scan_status.store(UCDHWorkbenchApplet::State::ScanStatus::Failed);
            return;
        }

        std::lock_guard<std::mutex> lk(sp->scan_result_mtx);
        sp->pending_loader = std::move(loader);
        sp->pending_samples = std::move(samples);
        sp->pending_format = fmt;
        sp->scan_status.store(UCDHWorkbenchApplet::State::ScanStatus::Ready);
    });
}

void UCDHWorkbenchApplet::select_sample(int idx) {
    auto& s = *s_;
    if (idx < 0 || idx >= (int)s.samples.size()) return;
    s.selected = idx;
    if (!s.bg || !s.loader) return;

    auto& samp = s.samples[idx];

    if (samp.loaded && samp.processed_valid) return;

    if (samp.loaded) {
        dsp::process(samp, s.params);
        return;
    }

    // Need to load from disk — prioritize without clearing existing queue
    std::vector<int> to_load;
    to_load.push_back(idx);
    for (int i : outward_indices(idx, 20, (int)s.samples.size())) {
        if (!s.samples[i].loaded) to_load.push_back(i);
    }
    s.bg->prioritize(to_load);
}

void UCDHWorkbenchApplet::on_params_changed() {
    auto& s = *s_;
    s.params.version++;
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

void UCDHWorkbenchApplet::ensure_vcg_cached() {
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

void UCDHWorkbenchApplet::draw_panel() {
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
        ImGui::TextColored({0.7f, 0.85f, 1.0f, 1.0f}, "Format: %s",
            format_display_name(s.current_format));
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Open as:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    const DatasetFormat fmts[] = {
        DatasetFormat::Auto,
        DatasetFormat::SingleFilePerSample,
        DatasetFormat::LeadFilesRowPerSample,
    };
    if (ImGui::BeginCombo("##fmt_override", format_display_name(s.pending_fmt_override))) {
        for (DatasetFormat f : fmts) {
            bool sel = (f == s.pending_fmt_override);
            if (ImGui::Selectable(format_display_name(f), sel)) s.pending_fmt_override = f;
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
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
            open_dataset(dir, static_cast<int>(s.pending_fmt_override));
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
            std::string label = si.file_id;
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

    if (ImGui::Button("<< Prev", ImVec2(110, 0)) && s.selected > 0)
        select_sample(s.selected - 1);
    ImGui::SameLine();
    if (ImGui::Button("Next >>", ImVec2(110, 0)) && s.selected < (int)s.samples.size() - 1)
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
        if (ImGui::DragFloat("Cutoff (Hz)", &s.params.baseline_cutoff_hz, 0.01f, 0.0f, 125.0f, "%.2f Hz"))
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

void UCDHWorkbenchApplet::draw_leads() {
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
        if (s.bg) {
            int done = s.bg->processed_count();
            int total = s.bg->total_queued();
            float frac = total > 0 ? (float)done / (float)total : 0.0f;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d/%d", done, total);
            ImGui::ProgressBar(frac, ImVec2(-1, 18), buf);
        }
        return;
    }

    float avail_h = ImGui::GetContentRegionAvail().y;
    float avail_w = ImGui::GetContentRegionAvail().x;

    int visible_count = 0;
    for (int i = 0; i < NUM_LEADS; i++) if (s.lead_visible[i]) visible_count++;
    if (visible_count == 0) { ImGui::Text("No leads selected."); return; }

    float sp = ImGui::GetStyle().ItemSpacing.y;
    float plot_h = (avail_h - sp * (visible_count - 1)) / (float)visible_count;
    plot_h = std::max(plot_h, 60.0f);

    float duration = samp.num_samples / std::max(1.0f, samp.sampling_rate);

    if ((int)s.time_axis.size() != samp.num_samples) {
        s.time_axis.resize(samp.num_samples);
        for (int i = 0; i < samp.num_samples; i++)
            s.time_axis[i] = (float)i / samp.sampling_rate;
    }

    // Reset axis limits when the displayed sample changes
    bool sample_changed = (s.selected != s.last_plot_sample);
    s.last_plot_sample = s.selected;

    ImPlot::GetInputMap().ZoomRate = 0.15f;

    for (int lead = 0; lead < NUM_LEADS; lead++) {
        if (!s.lead_visible[lead]) continue;
        auto& sig = samp.processed[lead];
        if (sig.empty()) continue;

        char plot_id[64];
        snprintf(plot_id, sizeof(plot_id), "##lead_%d", lead);

        if (ImPlot::BeginPlot(plot_id, ImVec2(avail_w, plot_h),
                ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("Time (s)", LEAD_NAMES[lead],
                ImPlotAxisFlags_NoLabel, ImPlotAxisFlags_NoLabel);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Once);
            auto& st = samp.stats[lead];
            float margin = (st.max_val - st.min_val) * 0.1f;
            ImPlot::SetupAxisLimits(ImAxis_Y1, st.min_val - margin, st.max_val + margin,
                sample_changed ? ImGuiCond_Always : ImGuiCond_Once);

            ImPlot::Annotation(0.0, samp.stats[lead].max_val, LEAD_COLORS[lead],
                ImVec2(5, 5), false, "%s", LEAD_NAMES[lead]);

            ImPlot::PlotLine("##sig", s.time_axis.data(), sig.data(), samp.num_samples,
                ImPlotSpec(ImPlotProp_LineColor, LEAD_COLORS[lead], ImPlotProp_LineWeight, 1.2f));

            ImPlot::EndPlot();
        }
    }
}

// ============================================================================
// VCG 3D TAB
// ============================================================================

void UCDHWorkbenchApplet::draw_vcg_3d() {
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

void UCDHWorkbenchApplet::draw_raw_browser() {
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
// MODEL TAB (scaffold)
// ============================================================================

void UCDHWorkbenchApplet::draw_model_tab() {
    auto& s = *s_;
    ImGui::Spacing();
    ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "MODEL INFERENCE");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button("Load Model...", ImVec2(160, 28))) {
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
            s.model_output.clear();
            try {
                s.model = torch::jit::load(path);
                s.model.eval();
                s.model_loaded = true;
                s.model_path = path;
            } catch (const c10::Error& e) {
                s.model_loaded = false;
                s.model_error = e.what();
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Spacing();

    if (!s.model_loaded) {
        if (!s.model_error.empty()) {
            ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "Load failed:");
            ImGui::TextWrapped("%s", s.model_error.c_str());
        } else {
            ImGui::TextDisabled("No model loaded.");
            ImGui::Spacing();
            ImGui::TextWrapped(
                "Load a TorchScript (.pt) model trained on the UCDH dataset. "
                "Use torch.jit.script() or torch.jit.trace() in Python to "
                "export your trained model.");
        }
        return;
    }

    ImGui::TextColored({0.4f, 1.0f, 0.6f, 1.0f}, "Model loaded");
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", fs::path(s.model_path).filename().string().c_str());

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    bool has_sample = s.selected >= 0 && s.selected < (int)s.samples.size()
                      && s.samples[s.selected].processed_valid;

    if (!has_sample) {
        ImGui::TextDisabled("Select a processed ECG sample to run inference.");
        return;
    }

    if (ImGui::Button("Run Inference", ImVec2(160, 28))) {
        auto& samp = s.samples[s.selected];
        s.model_output.clear();
        try {
            const int nl = (int)samp.processed.size();
            const int n = samp.num_samples;
            auto input = torch::zeros({1, nl, n});
            auto acc = input.accessor<float, 3>();
            for (int l = 0; l < nl; l++)
                for (int i = 0; i < n; i++)
                    acc[0][l][i] = samp.processed[l][i];
            auto out = s.model.forward({input}).toTensor();
            std::ostringstream oss;
            oss << out.sizes() << "\n" << out.slice(1, 0, std::min((int64_t)10, out.size(1)));
            s.model_output = oss.str();
        } catch (const c10::Error& e) {
            s.model_output = std::string("Inference error: ") + e.what();
        }
    }

    if (!s.model_output.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", s.model_output.c_str());
    }
}

