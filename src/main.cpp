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

namespace fs = std::filesystem;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

static constexpr int NUM_LEADS = 12;
static const char* LEAD_NAMES[NUM_LEADS] = {
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

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

struct ECGSample {
    std::string file_id;                        // e.g. "1014507"
    std::string filepath;
    std::vector<std::vector<float>> raw;        // [lead][sample]
    std::vector<std::vector<float>> processed;  // [lead][sample] after transforms
    float sampling_rate = 0.0f;
    int num_samples = 0;
    bool loaded = false;
    bool processed_valid = false;               // true when processed matches current params

    // Per-lead stats (computed on processed data)
    struct LeadStats {
        float mean = 0, stddev = 0, min_val = 0, max_val = 0;
    };
    std::vector<LeadStats> stats; // [lead]
};

struct ProcessingParams {
    bool zscore = true;
    bool baseline_wander_correction = false;
    float baseline_strength = 0.0f;     // 0 = none, 1 = max correction
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

    // Baseline wander removal via moving-average subtraction
    void remove_baseline_wander(std::vector<float>& data, int window) {
        if (data.empty() || window < 1) return;
        int n = (int)data.size();
        int half = window / 2;

        // Compute moving average
        std::vector<float> baseline(n);
        double running = 0;
        int count = 0;
        // Fill initial window
        for (int i = 0; i < std::min(half + 1, n); i++) { running += data[i]; count++; }
        baseline[0] = (float)(running / count);

        for (int i = 1; i < n; i++) {
            int add_idx = i + half;
            int rem_idx = i - half - 1;
            if (add_idx < n) { running += data[add_idx]; count++; }
            if (rem_idx >= 0) { running -= data[rem_idx]; count--; }
            baseline[i] = (float)(running / count);
        }

        for (int i = 0; i < n; i++) data[i] -= baseline[i];
    }

    // Apply full processing pipeline to a sample
    void process(ECGSample& sample, const ProcessingParams& params) {
        sample.processed.resize(NUM_LEADS);
        sample.stats.resize(NUM_LEADS);

        for (int lead = 0; lead < NUM_LEADS; lead++) {
            // Start from raw
            sample.processed[lead] = sample.raw[lead];
            auto& sig = sample.processed[lead];

            if (params.baseline_wander_correction && params.baseline_strength > 0 && sample.sampling_rate > 0) {
                // strength 0→none, 1→max correction
                // map: small strength → large window (gentle), large strength → small window (aggressive)
                float window_sec = 2.0f * (1.0f - params.baseline_strength) + 0.01f;
                int window = std::max(1, (int)(window_sec * sample.sampling_rate));
                remove_baseline_wander(sig, window);
            }

            if (params.zscore) {
                zscore(sig);
            }

            compute_stats(sig, sample.stats[lead]);
        }
        sample.processed_valid = true;
    }

} // namespace dsp

// ============================================================================
// DATA LOADING
// ============================================================================

namespace loader {

    bool load_csv(ECGSample& sample) {
        std::ifstream file(sample.filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open: " << sample.filepath << std::endl;
            return false;
        }

        std::string line;
        // Skip header
        if (!std::getline(file, line)) return false;

        sample.raw.resize(NUM_LEADS);
        for (auto& lead : sample.raw) lead.clear();

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            int col = 0;
            while (std::getline(ss, val, ',') && col < NUM_LEADS) {
                // Trim whitespace
                size_t start = val.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) { col++; continue; }
                val = val.substr(start);
                try {
                    sample.raw[col].push_back(std::stof(val));
                } catch (...) {}
                col++;
            }
        }

        if (sample.raw[0].empty()) return false;

        sample.num_samples = (int)sample.raw[0].size();
        // Infer sampling rate from sample count: 2500 -> 250Hz, 5000 -> 500Hz
        if (sample.num_samples <= 2500) sample.sampling_rate = 250.0f;
        else sample.sampling_rate = 500.0f;

        sample.loaded = true;
        sample.processed_valid = false;
        return true;
    }

    std::vector<ECGSample> scan_directory(const std::string& dir) {
        std::vector<ECGSample> samples;
        if (!fs::exists(dir)) {
            std::cerr << "Directory not found: " << dir << std::endl;
            return samples;
        }

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".csv") {
                ECGSample s;
                s.filepath = entry.path().string();
                s.file_id = entry.path().stem().string();
                samples.push_back(std::move(s));
            }
        }

        // Sort by file ID
        std::sort(samples.begin(), samples.end(),
            [](const ECGSample& a, const ECGSample& b) { return a.file_id < b.file_id; });

        return samples;
    }

} // namespace loader

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

    // Queue a batch of sample indices for processing
    void enqueue(std::vector<ECGSample>* samples, const ProcessingParams& params,
                 const std::vector<int>& indices) {
        std::lock_guard<std::mutex> lk(mtx_);
        // Clear previous work - new params invalidate old queue
        std::queue<int>().swap(queue_);
        samples_ = samples;
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

            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                if (queue_.empty()) continue;

                idx = queue_.front();
                queue_.pop();
                params = params_;
                sample = &(*samples_)[idx];
            }

            // Load if needed
            if (!sample->loaded) {
                loader::load_csv(*sample);
            }

            // Process
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
    ProcessingParams params_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> total_queued_{0};
};

// ============================================================================
// APPLICATION
// ============================================================================

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

        window_ = glfwCreateWindow(ww, wh, "Caliper - ECG Explorer", nullptr, nullptr);
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

        // Scan dataset
        std::string data_dir = "../data/ekg_data";
        samples_ = loader::scan_directory(data_dir);
        if (samples_.empty()) {
            std::cerr << "No CSV files found in " << data_dir << std::endl;
            return false;
        }
        std::cout << "Found " << samples_.size() << " ECG samples" << std::endl;

        // Load and process first sample immediately
        select_sample(0);

        // Start background processor
        bg_ = std::make_unique<BackgroundProcessor>();

        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            draw_ui();

            ImGui::Render();
            int dw, dh;
            glfwGetFramebufferSize(window_, &dw, &dh);
            glViewport(0, 0, dw, dh);
            glClearColor(0.06f, 0.06f, 0.09f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }
    }

    void cleanup() {
        bg_.reset(); // join background thread
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:

    // ── Selection & Processing ──

    void select_sample(int idx) {
        if (idx < 0 || idx >= (int)samples_.size()) return;
        selected_ = idx;
        auto& s = samples_[selected_];

        if (!s.loaded) loader::load_csv(s);
        if (s.loaded && !s.processed_valid) {
            dsp::process(s, params_);
        }
    }

    void on_params_changed() {
        params_.version++;

        // Invalidate all
        for (auto& s : samples_) s.processed_valid = false;

        // Reprocess current immediately
        auto& cur = samples_[selected_];
        if (cur.loaded) dsp::process(cur, params_);

        // Queue the rest in background
        std::vector<int> others;
        others.reserve(samples_.size());
        for (int i = 0; i < (int)samples_.size(); i++) {
            if (i != selected_) others.push_back(i);
        }
        bg_->enqueue(&samples_, params_, others);
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

    void draw_ui() {
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

        // ── Left panel ──
        ImGui::BeginChild("##Panel", ImVec2(panel_w, avail_h), true);
        draw_panel();
        ImGui::EndChild();

        ImGui::SameLine();

        // ── Right: plots ──
        ImGui::BeginChild("##Plots", ImVec2(plot_w, avail_h), false);
        draw_plots();
        ImGui::EndChild();

        ImGui::End();
    }

    void draw_panel() {
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
            if (ImGui::DragFloat("Strength", &params_.baseline_strength, 0.001f, 0.0f, 1.0f, "%.3f"))
                changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = no correction, 1 = maximum correction");
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
        if (selected_ < 0 || selected_ >= (int)samples_.size()) return;
        auto& s = samples_[selected_];
        if (!s.loaded || !s.processed_valid) return;

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
    std::vector<ECGSample> samples_;
    int selected_ = 0;

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
    std::cout << "=== Caliper - ECG Explorer ===" << std::endl;

    CaliperApp app;
    if (!app.initialize()) {
        std::cerr << "Initialization failed" << std::endl;
        return 1;
    }

    app.run();
    app.cleanup();
    return 0;
}
