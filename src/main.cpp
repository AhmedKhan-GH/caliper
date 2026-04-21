#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <chrono>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <functional>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>
#include <ImGuiFileDialog.h>

#include "intro_screen.h"
#include "dataset.h"

namespace fs = std::filesystem;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// NUM_LEADS and LEAD_NAMES are defined in dataset.h/.cpp

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

// ECGSample is defined in dataset.h

struct ProcessingParams {
    bool zscore = true;
    bool baseline_wander_correction = false;
    float baseline_cutoff_hz = 0.0f;    // high-pass cutoff in Hz, 0 = off
    uint32_t version = 1;               // bumped on any param change
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

    // Z-score normalization: (x - mean) / std
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

    // 4th-order zero-phase Butterworth high-pass (equivalent to scipy sosfiltfilt)
    void butterworth_highpass(std::vector<float>& data, float cutoff_hz, float sample_rate) {
        if (data.empty() || cutoff_hz <= 0 || sample_rate <= 0) return;
        if (cutoff_hz >= sample_rate * 0.5f) return; // above Nyquist
        int n = (int)data.size();

        // Pre-warp cutoff for bilinear transform
        double wc = std::tan(M_PI * (double)cutoff_hz / (double)sample_rate);
        double wc2 = wc * wc;

        // 4th-order Butterworth Q factors for two cascaded biquad sections
        // Poles at angles pi*(2k+5)/8 for k=0..3; paired into conjugate pairs
        // Q1 = 1/(2*cos(pi/8)) = 0.54120, Q2 = 1/(2*cos(3*pi/8)) = 1.30656
        const double Q[2] = { 0.54119610014620, 1.30656296487638 };

        // Second-order section coefficients (high-pass via bilinear transform)
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

        // Reflect-pad signal at both ends (like scipy's sosfiltfilt)
        // Pad length = 3 * number of sections * section_order = 3 * 2 * 2 = 12
        int pad = std::min(12, n - 1);
        int pn = n + 2 * pad;
        std::vector<double> buf(pn);

        // Left reflection: 2*x[0] - x[pad], ..., 2*x[0] - x[1]
        for (int i = 0; i < pad; i++)
            buf[i] = 2.0 * data[0] - data[pad - i];
        // Original signal
        for (int i = 0; i < n; i++)
            buf[pad + i] = data[i];
        // Right reflection: 2*x[n-1] - x[n-2], ..., 2*x[n-1] - x[n-1-pad]
        for (int i = 0; i < pad; i++)
            buf[pad + n + i] = 2.0 * data[n - 1] - data[n - 2 - i];

        // Apply each SOS as filtfilt (forward + reverse)
        // For high-pass, DC gain = 0, so initial conditions are zero.
        // The reflection padding handles edge transients.
        for (int s = 0; s < 2; s++) {
            auto& bq = sos[s];

            // Forward pass (transposed direct form II)
            double w1 = 0, w2 = 0;
            for (int i = 0; i < pn; i++) {
                double xi = buf[i];
                double yi = bq.b0 * xi + w1;
                w1 = bq.b1 * xi - bq.a1 * yi + w2;
                w2 = bq.b2 * xi - bq.a2 * yi;
                buf[i] = yi;
            }

            // Reverse pass
            w1 = 0; w2 = 0;
            for (int i = pn - 1; i >= 0; i--) {
                double xi = buf[i];
                double yi = bq.b0 * xi + w1;
                w1 = bq.b1 * xi - bq.a1 * yi + w2;
                w2 = bq.b2 * xi - bq.a2 * yi;
                buf[i] = yi;
            }
        }

        // Extract the original-length portion
        for (int i = 0; i < n; i++)
            data[i] = (float)buf[pad + i];
    }

    // Apply full processing pipeline to a sample
    void process(ECGSample& sample, const ProcessingParams& params) {
        sample.processed.resize(NUM_LEADS);
        sample.stats.resize(NUM_LEADS);

        for (int lead = 0; lead < NUM_LEADS; lead++) {
            // Start from raw
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

} // namespace dsp

// Dataset loading lives in dataset.{h,cpp}. See IDatasetLoader.

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

    // Queue a batch of sample indices for processing. `loader` is borrowed —
    // caller must keep it alive until the queue drains (enqueue again with a
    // new loader to hand off cleanly).
    void enqueue(std::vector<ECGSample>* samples, IDatasetLoader* loader,
                 const ProcessingParams& params, const std::vector<int>& indices) {
        std::lock_guard<std::mutex> lk(mtx_);
        std::queue<int>().swap(queue_);
        samples_ = samples;
        loader_ = loader;
        params_ = params;
        for (int idx : indices) queue_.push(idx);
        processed_count_.store(0);
        total_queued_.store((int)indices.size());
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
                queue_.pop();
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
    std::queue<int> queue_;
    std::vector<ECGSample>* samples_ = nullptr;
    IDatasetLoader* loader_ = nullptr;
    ProcessingParams params_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> total_queued_{0};
};

// ============================================================================
// (Intro/landing screen lives in intro_screen.{h,cpp})
// ============================================================================


// ============================================================================
// APPLICATION
// ============================================================================

enum class AppPage {
    Landing,
    ECGApp
};

class CaliperApp {
public:
    CaliperApp() = default;

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "GLFW init failed" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        int ax, ay, aw, ah;
        glfwGetMonitorWorkarea(monitor, &ax, &ay, &aw, &ah);
        float sx = 1.0f, sy = 1.0f;
        glfwGetMonitorContentScale(monitor, &sx, &sy);
        int ww = (int)((aw / sx) * 0.95f);
        int wh = (int)((ah / sy) * 0.95f);

        window_ = glfwCreateWindow(ww, wh, "Caliper", nullptr, nullptr);
        if (!window_) { glfwTerminate(); return false; }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) { glfwTerminate(); return false; }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();
        style_ui();

        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        // Initialize intro screen (3D scene + overlay)
        if (!intro_.initialize()) {
            std::cerr << "Intro screen init failed" << std::endl;
            return false;
        }

        // Dataset is opened via the UI ("Open Dataset..." button). No
        // hardcoded defaults — paths are user-supplied at runtime.
        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);
            glViewport(0, 0, dw, dh);
            glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Render 3D scene behind ImGui on the intro screen
            if (page_ == AppPage::Landing) {
                intro_.update(window_);
                intro_.render_3d(dw, dh);
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            commit_scan_if_ready();

            if (page_ == AppPage::Landing) {
                intro_.draw_ui(dw, dh);
                if (intro_.should_launch()) {
                    intro_.reset_launch_flag();
                    AppletKind k = intro_.selected_applet();
                    if (k == AppletKind::ECGExplorer) {
                        page_ = AppPage::ECGApp;
                        params_.baseline_wander_correction = true;
                        if (params_.baseline_cutoff_hz <= 0) params_.baseline_cutoff_hz = 0.5f;
                        glfwSetWindowTitle(window_, "Caliper - ECG Explorer");
                    }
                }
            } else {
                draw_ecg_ui();
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }
    }

    void cleanup() {
        bg_.reset(); // join background load thread
        if (scan_thread_.joinable()) scan_thread_.join();
        intro_.cleanup();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:

    // ── Selection & Processing ──

    // Build indices expanding outward from center: center-1, center+1, center-2, center+2, ...
    std::vector<int> outward_indices(int center, int count) {
        std::vector<int> out;
        out.reserve(count);
        for (int d = 1; out.size() < (size_t)count; d++) {
            bool added = false;
            int lo = center - d, hi = center + d;
            if (lo >= 0 && lo < (int)samples_.size()) { out.push_back(lo); added = true; }
            if (hi >= 0 && hi < (int)samples_.size()) { out.push_back(hi); added = true; }
            if (!added) break;
        }
        return out;
    }

    // Non-blocking selection. Two cases:
    //   - sample already loaded  → main thread processes it (no race; bg
    //                              only gets outward neighbors). draws this frame.
    //   - sample not yet loaded  → bg handles load+process with selected at
    //                              the front of the queue. draw_plots shows
    //                              a "loading..." placeholder until ready.
    //
    // Main thread never touches a sample the bg could be writing to.
    void select_sample(int idx) {
        if (idx < 0 || idx >= (int)samples_.size()) return;
        selected_ = idx;
        if (!bg_ || !loader_) return;

        auto& s = samples_[selected_];
        auto neighbors = outward_indices(idx, (int)samples_.size() - 1);

        if (s.loaded) {
            if (!s.processed_valid) dsp::process(s, params_);
            // BG handles neighbors only; current is already done.
            bg_->enqueue(&samples_, loader_.get(), params_, neighbors);
        } else {
            // BG handles current AND neighbors; current first.
            neighbors.insert(neighbors.begin(), idx);
            bg_->enqueue(&samples_, loader_.get(), params_, neighbors);
        }
    }

    void on_params_changed() {
        params_.version++;

        // Invalidate processed state for all samples.
        for (auto& s : samples_) s.processed_valid = false;

        if (!bg_ || !loader_ || samples_.empty()) return;

        // Main thread reprocesses current (if loaded); bg re-processes the rest.
        int center = std::max(0, selected_);
        if (selected_ >= 0 && selected_ < (int)samples_.size()
            && samples_[selected_].loaded) {
            dsp::process(samples_[selected_], params_);
            auto neighbors = outward_indices(center, (int)samples_.size() - 1);
            bg_->enqueue(&samples_, loader_.get(), params_, neighbors);
        } else {
            auto idxs = outward_indices(center, (int)samples_.size() - 1);
            idxs.insert(idxs.begin(), center);
            bg_->enqueue(&samples_, loader_.get(), params_, idxs);
        }
    }

    // ── Async dataset open ──
    //
    // open_dataset() kicks off a background scan. Main thread polls
    // commit_scan_if_ready() each frame; when the scan completes, the results
    // are swapped in under a mutex and the load worker is (re)created. The UI
    // shows a "Scanning..." spinner while the scan is in flight.

    enum class ScanStatus { Idle, Scanning, Ready, Failed };

    void open_dataset(const std::string& dir, DatasetFormat fmt_override) {
        // Join any previous scan thread before starting a new one.
        if (scan_thread_.joinable()) scan_thread_.join();

        scan_status_.store(ScanStatus::Scanning);
        scan_error_.clear();
        scan_dir_ = dir;

        scan_thread_ = std::thread([this, dir, fmt_override]() {
            DatasetFormat fmt = (fmt_override == DatasetFormat::Auto)
                ? detect_format(dir) : fmt_override;

            auto loader = make_dataset_loader(fmt);
            if (!loader) {
                std::lock_guard<std::mutex> lk(scan_result_mtx_);
                scan_error_ = "Could not create loader for format";
                scan_status_.store(ScanStatus::Failed);
                return;
            }

            std::vector<ECGSample> samples;
            bool ok = loader->scan(dir, samples);
            if (!ok) {
                std::lock_guard<std::mutex> lk(scan_result_mtx_);
                scan_error_ = std::string("No matching files in: ") + dir;
                scan_status_.store(ScanStatus::Failed);
                return;
            }

            std::lock_guard<std::mutex> lk(scan_result_mtx_);
            pending_loader_ = std::move(loader);
            pending_samples_ = std::move(samples);
            pending_format_ = fmt;
            scan_status_.store(ScanStatus::Ready);
        });
    }

    // Called each frame on the main thread. If the scan worker has staged new
    // results, swap them in and kick off background loading. Safe to call
    // every frame — no-op when nothing is ready.
    void commit_scan_if_ready() {
        ScanStatus st = scan_status_.load();
        if (st != ScanStatus::Ready) return;

        // Join the scan thread — it has already staged results.
        if (scan_thread_.joinable()) scan_thread_.join();

        // Tear down the load worker before swapping its inputs.
        bg_.reset();

        {
            std::lock_guard<std::mutex> lk(scan_result_mtx_);
            loader_ = std::move(pending_loader_);
            samples_ = std::move(pending_samples_);
            current_format_ = pending_format_;
            current_dir_ = scan_dir_;
        }

        selected_ = -1;
        bg_ = std::make_unique<BackgroundProcessor>();
        scan_status_.store(ScanStatus::Idle);

        std::cout << "[dataset] Opened " << current_dir_ << " ("
                  << format_display_name(current_format_) << "): "
                  << samples_.size() << " samples" << std::endl;

        if (!samples_.empty()) select_sample(0);
    }

    // ── UI ──

    void style_ui() {
        ImGuiStyle& st = ImGui::GetStyle();
        st.WindowRounding = 6.0f;
        st.FrameRounding = 4.0f;
        st.GrabRounding = 3.0f;
        st.ScrollbarRounding = 4.0f;
        st.ItemSpacing = ImVec2(8, 5);

        auto* c = st.Colors;
        c[ImGuiCol_WindowBg]       = {0.09f, 0.09f, 0.12f, 0.97f};
        c[ImGuiCol_ChildBg]        = {0.11f, 0.11f, 0.15f, 1.00f};
        c[ImGuiCol_Header]         = {0.18f, 0.22f, 0.32f, 1.00f};
        c[ImGuiCol_HeaderHovered]   = {0.26f, 0.30f, 0.42f, 1.00f};
        c[ImGuiCol_HeaderActive]    = {0.22f, 0.26f, 0.38f, 1.00f};
        c[ImGuiCol_Button]          = {0.18f, 0.22f, 0.32f, 1.00f};
        c[ImGuiCol_ButtonHovered]   = {0.28f, 0.32f, 0.44f, 1.00f};
        c[ImGuiCol_ButtonActive]    = {0.14f, 0.18f, 0.28f, 1.00f};
        c[ImGuiCol_FrameBg]         = {0.14f, 0.14f, 0.20f, 1.00f};
        c[ImGuiCol_FrameBgHovered]  = {0.20f, 0.20f, 0.28f, 1.00f};
        c[ImGuiCol_SliderGrab]      = {0.40f, 0.55f, 0.80f, 1.00f};
        c[ImGuiCol_SliderGrabActive]= {0.50f, 0.65f, 0.90f, 1.00f};
        c[ImGuiCol_ScrollbarBg]     = {0.08f, 0.08f, 0.10f, 1.00f};
        c[ImGuiCol_ScrollbarGrab]   = {0.25f, 0.25f, 0.35f, 1.00f};
    }

    // ── ECG App UI ──

    void draw_ecg_ui() {
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("##Root", nullptr,
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

        // -- Right: plots --
        ImGui::BeginChild("##Plots", ImVec2(plot_w, avail_h), false);
        draw_plots();
        ImGui::EndChild();

        ImGui::End();
    }

    void draw_panel() {
        // ── Back button ──
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.14f, 0.14f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.20f, 0.20f, 1.0f));
        if (ImGui::Button("<< Back to Menu", ImVec2(-1, 28))) {
            page_ = AppPage::Landing;
            glfwSetWindowTitle(window_, "Caliper");
        }
        ImGui::PopStyleColor(2);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ── Dataset ──
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "DATASET");
        ImGui::Separator();

        // Current path + format
        if (current_dir_.empty()) {
            ImGui::TextColored({0.8f, 0.8f, 0.8f, 1.0f}, "(none — click Open Dataset...)");
        } else {
            ImGui::TextWrapped("%s", current_dir_.c_str());
            ImGui::TextColored({0.7f, 0.85f, 1.0f, 1.0f}, "Format: %s",
                format_display_name(current_format_));
        }

        // Format override combo (applies to the NEXT open; current format is
        // shown above and reflects what's loaded).
        ImGui::Spacing();
        ImGui::TextUnformatted("Open as:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        const DatasetFormat fmts[] = {
            DatasetFormat::Auto,
            DatasetFormat::SingleFilePerSample,
            DatasetFormat::LeadFilesRowPerSample,
        };
        if (ImGui::BeginCombo("##fmt_override", format_display_name(pending_fmt_override_))) {
            for (DatasetFormat f : fmts) {
                bool sel = (f == pending_fmt_override_);
                if (ImGui::Selectable(format_display_name(f), sel)) pending_fmt_override_ = f;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Spacing();

        ScanStatus st = scan_status_.load();
        bool scanning = (st == ScanStatus::Scanning);

        if (scanning) ImGui::BeginDisabled();
        if (ImGui::Button("Open Dataset...", ImVec2(-1, 28))) {
            IGFD::FileDialogConfig cfg;
            cfg.path = current_dir_.empty() ? "." : current_dir_;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            // Empty filter → directory selection mode in ImGuiFileDialog.
            ImGuiFileDialog::Instance()->OpenDialog(
                "OpenDatasetDlg", "Choose dataset directory", nullptr, cfg);
        }
        if (scanning) ImGui::EndDisabled();

        if (scanning) {
            ImGui::TextColored({1.0f, 0.85f, 0.3f, 1.0f}, "Scanning...");
        } else if (st == ScanStatus::Failed) {
            std::string err;
            { std::lock_guard<std::mutex> lk(scan_result_mtx_); err = scan_error_; }
            ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "Scan failed:");
            ImGui::TextWrapped("%s", err.c_str());
        }

        // Render the file dialog itself (modal). Min size keeps it usable.
        ImVec2 min_sz(600, 400);
        ImVec2 max_sz(FLT_MAX, FLT_MAX);
        if (ImGuiFileDialog::Instance()->Display("OpenDatasetDlg",
                ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
                open_dataset(dir, pending_fmt_override_);
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (samples_.empty()) {
            if (!scanning && st != ScanStatus::Failed) {
                ImGui::TextColored({1.0f, 0.7f, 0.3f, 1.0f}, "No dataset loaded.");
                ImGui::TextWrapped("Click Open Dataset... and pick the directory containing your CSVs.");
            }
            return;
        }

        // ── Sample picker ──
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "SAMPLE PICKER");
        ImGui::Separator();

        ImGui::Text("Samples: %d", (int)samples_.size());

        // Filter box
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputTextWithHint("##filter", "Filter by ID...", filter_buf_, sizeof(filter_buf_))) {
            // filter changed
        }

        // Sample list
        float list_h = std::min(200.0f, ImGui::GetContentRegionAvail().y * 0.35f);
        if (ImGui::BeginListBox("##samples", ImVec2(-1, list_h))) {
            std::string filter(filter_buf_);
            for (int i = 0; i < (int)samples_.size(); i++) {
                if (!filter.empty() && samples_[i].file_id.find(filter) == std::string::npos)
                    continue;

                bool is_selected = (i == selected_);
                std::string label = samples_[i].file_id;
                if (samples_[i].loaded) {
                    label += " (" + std::to_string(samples_[i].num_samples) + ")";
                }
                if (!samples_[i].processed_valid && samples_[i].loaded) {
                    label += " *";
                }

                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    select_sample(i);
                }
            }
            ImGui::EndListBox();
        }

        // Navigation
        if (ImGui::Button("<< Prev", ImVec2(110, 0)) && selected_ > 0) {
            select_sample(selected_ - 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("Next >>", ImVec2(110, 0)) && selected_ < (int)samples_.size() - 1) {
            select_sample(selected_ + 1);
        }

        // ID scroller
        int sel = selected_;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##id_scroll", &sel, 0, (int)samples_.size() - 1)) {
            select_sample(sel);
        }

        ImGui::Spacing();
        ImGui::Separator();

        // ── Processing controls ──
        ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "PROCESSING");
        ImGui::Separator();

        bool changed = false;

        if (ImGui::Checkbox("Z-Score Normalize", &params_.zscore)) changed = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Standardize each lead to zero mean, unit variance");

        if (ImGui::Checkbox("Baseline Wander Correction", &params_.baseline_wander_correction))
            changed = true;

        if (params_.baseline_wander_correction) {
            ImGui::Indent(12);
            ImGui::SetNextItemWidth(-12);
            if (ImGui::DragFloat("Cutoff (Hz)", &params_.baseline_cutoff_hz, 0.01f, 0.0f, 125.0f, "%.2f Hz"))
                changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("High-pass cutoff frequency");
            ImGui::Unindent(12);
        }

        if (changed) on_params_changed();

        // Background progress
        if (bg_ && bg_->busy()) {
            ImGui::Spacing();
            int done = bg_->processed_count();
            int total = bg_->total_queued();
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

        if (selected_ >= 0 && selected_ < (int)samples_.size()) {
            auto& s = samples_[selected_];
            ImGui::Text("ID: %s", s.file_id.c_str());
            ImGui::Text("Samples: %d", s.num_samples);
            ImGui::Text("Rate: %.0f Hz", s.sampling_rate);
            ImGui::Text("Duration: %.1f sec", s.num_samples / std::max(1.0f, s.sampling_rate));

            if (s.processed_valid && !s.stats.empty()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored({0.6f, 0.8f, 1.0f, 1.0f}, "LEAD STATS");
                ImGui::Separator();

                // Lead visibility toggles
                for (int i = 0; i < NUM_LEADS; i++) {
                    ImGui::PushStyleColor(ImGuiCol_Text, LEAD_COLORS[i]);
                    ImGui::Checkbox(LEAD_NAMES[i], &lead_visible_[i]);
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
                        if (!lead_visible_[i]) continue;
                        auto& st = s.stats[i];
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

    void draw_plots() {
        if (selected_ < 0 || selected_ >= (int)samples_.size()) {
            ImGui::TextColored({0.7f, 0.7f, 0.8f, 1.0f}, "Select a sample from the panel.");
            return;
        }
        auto& s = samples_[selected_];
        if (!s.loaded || !s.processed_valid) {
            ImGui::Spacing();
            ImGui::TextColored({1.0f, 0.85f, 0.3f, 1.0f},
                "Loading sample %s ...", s.file_id.c_str());
            if (bg_) {
                int done = bg_->processed_count();
                int total = bg_->total_queued();
                float frac = total > 0 ? (float)done / (float)total : 0.0f;
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%d/%d", done, total);
                ImGui::ProgressBar(frac, ImVec2(-1, 18), buf);
            }
            return;
        }

        float avail_h = ImGui::GetContentRegionAvail().y;
        float avail_w = ImGui::GetContentRegionAvail().x;

        // Count visible leads
        int visible_count = 0;
        for (int i = 0; i < NUM_LEADS; i++) if (lead_visible_[i]) visible_count++;
        if (visible_count == 0) {
            ImGui::Text("No leads selected.");
            return;
        }

        float sp = ImGui::GetStyle().ItemSpacing.y;
        float plot_h = (avail_h - sp * (visible_count - 1)) / (float)visible_count;
        plot_h = std::max(plot_h, 60.0f);

        float duration = s.num_samples / std::max(1.0f, s.sampling_rate);

        // Generate time axis once
        if ((int)time_axis_.size() != s.num_samples) {
            time_axis_.resize(s.num_samples);
            for (int i = 0; i < s.num_samples; i++) {
                time_axis_[i] = (float)i / s.sampling_rate;
            }
        }

        // Link x-axes so panning/zooming is synchronized across leads
        ImPlot::GetInputMap().ZoomRate = 0.15f;

        for (int lead = 0; lead < NUM_LEADS; lead++) {
            if (!lead_visible_[lead]) continue;

            auto& sig = s.processed[lead];
            if (sig.empty()) continue;

            char plot_id[64];
            snprintf(plot_id, sizeof(plot_id), "##lead_%d", lead);

            if (ImPlot::BeginPlot(plot_id, ImVec2(avail_w, plot_h),
                    ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoInputs)) {

                ImPlot::SetupAxes("Time (s)", LEAD_NAMES[lead],
                    ImPlotAxisFlags_NoLabel,
                    ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Once);

                // Lead label in top-left of plot
                ImPlot::Annotation(0.0, s.stats[lead].max_val, LEAD_COLORS[lead],
                    ImVec2(5, 5), false, "%s", LEAD_NAMES[lead]);

                ImPlot::PlotLine("##sig", time_axis_.data(), sig.data(), s.num_samples,
                    ImPlotSpec(ImPlotProp_LineColor, LEAD_COLORS[lead], ImPlotProp_LineWeight, 1.2f));

                ImPlot::EndPlot();
            }
        }
    }

    // ── Members ──
    GLFWwindow* window_ = nullptr;
    AppPage page_ = AppPage::Landing;
    IntroScreen intro_;

    std::vector<ECGSample> samples_;
    int selected_ = -1;
    std::unique_ptr<IDatasetLoader> loader_;
    std::string current_dir_;
    DatasetFormat current_format_ = DatasetFormat::Auto;

    // Async scan state. The scan runs on `scan_thread_`; results are staged
    // in `pending_*` under `scan_result_mtx_` and committed on the main
    // thread in commit_scan_if_ready(). Main thread polls `scan_status_`.
    std::atomic<ScanStatus> scan_status_{ScanStatus::Idle};
    std::thread scan_thread_;
    std::mutex scan_result_mtx_;
    std::unique_ptr<IDatasetLoader> pending_loader_;
    std::vector<ECGSample> pending_samples_;
    DatasetFormat pending_format_ = DatasetFormat::Auto;
    std::string scan_dir_;
    std::string scan_error_;

    // UI state for the format-override combo in the Open dialog.
    DatasetFormat pending_fmt_override_ = DatasetFormat::Auto;

    ProcessingParams params_;
    std::unique_ptr<BackgroundProcessor> bg_;

    bool lead_visible_[NUM_LEADS] = {true,true,true,true,true,true,true,true,true,true,true,true};
    char filter_buf_[128] = {};
    std::vector<float> time_axis_;
};

// ============================================================================
// ENTRY POINT
// ============================================================================

int main() {
    std::cout << "=== Caliper ===" << std::endl;

    CaliperApp app;
    if (!app.initialize()) {
        std::cerr << "Initialization failed" << std::endl;
        return 1;
    }

    app.run();
    app.cleanup();
    return 0;
}
