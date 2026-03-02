#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

// ============================================================================
// EEG DATASET LOADER - Nightingale Dataset
// ============================================================================

struct EEGRecord {
    std::vector<std::vector<float>> leads; // 12 leads: I, II, III, aVR, aVL, aVF, V1-V6
    float sampling_rate;
    std::string record_name;
    int num_channels;

    static const std::vector<std::string> lead_names;
};

const std::vector<std::string> EEGRecord::lead_names = {
    "Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

class EEGDataLoader {
public:
    explicit EEGDataLoader(const std::string& data_path)
        : data_path_(data_path) {}

    bool load_nightingale_data() {
        std::cout << "Loading Nightingale dataset from: " << data_path_ << std::endl;

        // Load all 12 leads
        std::vector<std::string> lead_files = {
            "/MDC_ECG_LEAD_I.csv",
            "/MDC_ECG_LEAD_II.csv",
            "/MDC_ECG_LEAD_III.csv",
            "/MDC_ECG_LEAD_AVR.csv",
            "/MDC_ECG_LEAD_AVL.csv",
            "/MDC_ECG_LEAD_AVF.csv",
            "/MDC_ECG_LEAD_V1.csv",
            "/MDC_ECG_LEAD_V2.csv",
            "/MDC_ECG_LEAD_V3.csv",
            "/MDC_ECG_LEAD_V4.csv",
            "/MDC_ECG_LEAD_V5.csv",
            "/MDC_ECG_LEAD_V6.csv"
        };

        std::vector<std::vector<std::vector<float>>> all_leads_data;
        size_t num_records = 0;

        // Load each lead file
        for (size_t lead_idx = 0; lead_idx < lead_files.size(); lead_idx++) {
            std::string lead_file = data_path_ + lead_files[lead_idx];
            std::vector<std::vector<float>> lead_signals = load_csv_file(lead_file);

            if (lead_signals.empty()) {
                std::cerr << "Failed to load data from: " << lead_file << std::endl;
                return false;
            }

            if (lead_idx == 0) {
                num_records = lead_signals.size();
                all_leads_data.resize(lead_files.size());
            } else if (lead_signals.size() != num_records) {
                std::cerr << "Mismatch in number of records across leads!" << std::endl;
                return false;
            }

            all_leads_data[lead_idx] = std::move(lead_signals);
            std::cout << "Loaded " << EEGRecord::lead_names[lead_idx] << std::endl;
        }

        std::cout << "Found " << num_records << " patient records" << std::endl;

        // Initialize records (each record has all 12 leads)
        records_.reserve(num_records);
        for (size_t i = 0; i < num_records; i++) {
            EEGRecord record;
            record.record_name = "Patient_" + std::to_string(i + 1);
            record.sampling_rate = 500.0f; // Nightingale sampling rate
            record.num_channels = 12; // All 12 leads
            record.leads.resize(12);

            for (size_t lead_idx = 0; lead_idx < 12; lead_idx++) {
                record.leads[lead_idx] = std::move(all_leads_data[lead_idx][i]);
            }

            records_.push_back(record);
        }

        std::cout << "Loaded " << num_records << " records with "
                  << records_[0].leads[0].size() << " samples each" << std::endl;
        return true;
    }

    const std::vector<EEGRecord>& get_records() const { return records_; }

private:
    std::vector<std::vector<float>> load_csv_file(const std::string& filepath) {
        std::vector<std::vector<float>> data;

        std::cout << "Attempting to open: " << filepath << std::endl;
        std::ifstream file(filepath);

        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file: " << filepath << std::endl;
            std::cerr << "Make sure the file exists and path is correct." << std::endl;
            return data;
        }

        std::cout << "File opened successfully!" << std::endl;

        std::string line;
        bool skip_header = true;

        while (std::getline(file, line)) {
            if (skip_header) {
                skip_header = false;
                continue; // Skip the first row (sample indices)
            }

            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                try {
                    row.push_back(std::stof(value));
                } catch (...) {
                    // Skip invalid values
                }
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        file.close();
        return data;
    }

    std::string data_path_;
    std::vector<EEGRecord> records_;
};

// ============================================================================
// EEG WAVEFORM ANALYZER
// ============================================================================

class EEGAnalyzer {
public:
    EEGAnalyzer() : current_record_idx_(0) {
        // Initialize all leads as visible by default
        for (int i = 0; i < 12; i++) {
            lead_visible_[i] = true;
        }
    }

    void load_data(const std::string& data_path) {
        loader_ = std::make_unique<EEGDataLoader>(data_path);
        loader_->load_nightingale_data();
        records_ = loader_->get_records();

        // Analyze first record after loading
        if (!records_.empty()) {
            analyze_waveform();
        }
    }

    void analyze_waveform() {
        if (records_.empty() || current_record_idx_ >= records_.size()) {
            return;
        }

        const auto& record = records_[current_record_idx_];

        // Calculate statistics for first visible lead
        for (size_t lead_idx = 0; lead_idx < 12; lead_idx++) {
            if (lead_visible_[lead_idx] && !record.leads[lead_idx].empty()) {
                const auto& signal = record.leads[lead_idx];
                float sum = std::accumulate(signal.begin(), signal.end(), 0.0f);
                mean_ = sum / signal.size();

                float sq_sum = std::inner_product(signal.begin(), signal.end(),
                                                 signal.begin(), 0.0f);
                std_dev_ = std::sqrt(sq_sum / signal.size() - mean_ * mean_);

                min_val_ = *std::min_element(signal.begin(), signal.end());
                max_val_ = *std::max_element(signal.begin(), signal.end());
                break; // Just calculate for first visible lead
            }
        }
    }

    const std::vector<EEGRecord>& get_records() const { return records_; }
    size_t get_current_index() const { return current_record_idx_; }
    void set_current_index(size_t idx) { current_record_idx_ = idx; analyze_waveform(); }

    float get_mean() const { return mean_; }
    float get_std_dev() const { return std_dev_; }
    float get_min() const { return min_val_; }
    float get_max() const { return max_val_; }

    bool* get_lead_visibility() { return lead_visible_; }

private:
    std::unique_ptr<EEGDataLoader> loader_;
    std::vector<EEGRecord> records_;
    size_t current_record_idx_;
    bool lead_visible_[12]; // Per-user visibility for each lead

    // Analysis results
    float mean_ = 0.0f;
    float std_dev_ = 0.0f;
    float min_val_ = 0.0f;
    float max_val_ = 0.0f;
};

// ============================================================================
// MAIN APPLICATION
// ============================================================================

class EEGVisualizerApp {
public:
    EEGVisualizerApp() : window_(nullptr), analyzer_() {
        detect_compute_device();
    }

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        window_ = glfwCreateWindow(1280, 720, "Caliper - EEG Waveform Analysis", nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        // Setup ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        // Load data - use absolute path
        analyzer_.load_data("/Users/ahmed/CLionProjects/caliper/data/Nightingale Dataset");

        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            render_ui();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }
    }

    void cleanup() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();

        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

private:
    void detect_compute_device() {
        std::cout << "=== CUDA Diagnostics ===" << std::endl;
        std::cout << "CUDA compiled: " << (torch::cuda::is_available() ? "YES" : "NO") << std::endl;

        if (torch::cuda::is_available()) {
            std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;

            // Try to create a tensor on CUDA
            try {
                auto test_tensor = torch::randn({10, 10}, torch::device(torch::kCUDA));
                std::cout << "Successfully created tensor on CUDA device: " << test_tensor.device() << std::endl;
                device_name_ = "CUDA GPU";
                device_color_ = ImVec4(0.2f, 1.0f, 0.2f, 1.0f); // Green
            } catch (const std::exception& e) {
                std::cout << "Failed to create CUDA tensor: " << e.what() << std::endl;
                device_name_ = "CPU (CUDA init failed)";
                device_color_ = ImVec4(1.0f, 1.0f, 0.2f, 1.0f); // Yellow
            }
        } else if (torch::mps::is_available()) {
            device_name_ = "MPS (Apple Silicon)";
            device_color_ = ImVec4(0.2f, 0.8f, 1.0f, 1.0f); // Blue
        } else {
            std::cout << "CUDA not available - using CPU" << std::endl;
            device_name_ = "CPU";
            device_color_ = ImVec4(1.0f, 1.0f, 0.2f, 1.0f); // Yellow
        }
        std::cout << "Final device: " << device_name_ << std::endl;
        std::cout << "======================" << std::endl;
    }

    void render_ui() {
        ImGui::Begin("EEG Waveform Analysis", nullptr, ImGuiWindowFlags_NoCollapse);

        // Show compute device
        ImGui::TextColored(device_color_, "Compute Device: %s", device_name_.c_str());
        ImGui::Separator();

        const auto& records = analyzer_.get_records();

        if (records.empty()) {
            ImGui::Text("No data loaded. Place Nightingale dataset in ./data directory");
        } else {
            // Record selection
            int current_idx = static_cast<int>(analyzer_.get_current_index());
            if (ImGui::SliderInt("Record", &current_idx, 0, records.size() - 1)) {
                analyzer_.set_current_index(current_idx);
            }

            const auto& record = records[current_idx];
            ImGui::Text("Record: %s", record.record_name.c_str());
            ImGui::Text("Sampling Rate: %.1f Hz", record.sampling_rate);
            ImGui::Text("Channels: %d", record.num_channels);
            ImGui::Text("Samples: %zu", record.leads.empty() ? 0 : record.leads[0].size());

            ImGui::Separator();

            // Lead selection (per-user)
            ImGui::Text("Select Leads to Display:");
            bool* lead_visibility = analyzer_.get_lead_visibility();

            // Create 3 columns for the checkboxes
            ImGui::Columns(3, "lead_columns", false);
            for (size_t i = 0; i < 12; i++) {
                ImGui::Checkbox(EEGRecord::lead_names[i].c_str(), &lead_visibility[i]);
                if ((i + 1) % 4 == 0) {
                    ImGui::NextColumn();
                }
            }
            ImGui::Columns(1);

            ImGui::Separator();

            // Statistics
            ImGui::Text("Statistics (first visible lead):");
            ImGui::Text("  Mean: %.4f", analyzer_.get_mean());
            ImGui::Text("  Std Dev: %.4f", analyzer_.get_std_dev());
            ImGui::Text("  Min: %.4f", analyzer_.get_min());
            ImGui::Text("  Max: %.4f", analyzer_.get_max());

            ImGui::Separator();

            // Waveform plot - render all selected leads
            if (!record.leads.empty() && !record.leads[0].empty()) {
                if (ImPlot::BeginPlot("ECG Waveform - All Leads", ImVec2(-1, 450))) {
                    ImPlot::SetupAxes("Time (samples)", "Amplitude (mV)",
                                     ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, record.leads[0].size());

                    // Plot each visible lead
                    for (size_t lead_idx = 0; lead_idx < 12; lead_idx++) {
                        if (lead_visibility[lead_idx]) {
                            ImPlot::PlotLine(EEGRecord::lead_names[lead_idx].c_str(),
                                           record.leads[lead_idx].data(),
                                           record.leads[lead_idx].size());
                        }
                    }

                    ImPlot::EndPlot();
                }
            }
        }

        ImGui::End();
    }

    GLFWwindow* window_;
    EEGAnalyzer analyzer_;
    std::string device_name_;
    ImVec4 device_color_;
};

// ============================================================================
// ENTRY POINT
// ============================================================================

int main() {
    EEGVisualizerApp app;

    if (!app.initialize()) {
        return -1;
    }

    app.run();
    app.cleanup();

    return 0;
}
