#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <chrono>
#include <random>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

// ============================================================================
// CUDA-Accelerated ECG Autoencoder for Anomaly Detection
// ============================================================================

class ECGAutoencoderImpl : public torch::nn::Module {
public:
    ECGAutoencoderImpl(int input_size = 5000, int latent_dim = 64) {
        this->input_size = input_size;
        this->latent_dim = latent_dim;

        // Simple fully connected autoencoder
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size, 512),
            torch::nn::ReLU(),
            torch::nn::Linear(512, 256),
            torch::nn::ReLU(),
            torch::nn::Linear(256, latent_dim)
        ));

        decoder = register_module("decoder", torch::nn::Sequential(
            torch::nn::Linear(latent_dim, 256),
            torch::nn::ReLU(),
            torch::nn::Linear(256, 512),
            torch::nn::ReLU(),
            torch::nn::Linear(512, input_size)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batch, 1, signal_length]
        auto batch_size = x.size(0);
        auto seq_len = x.size(2);

        // Flatten to [batch, signal_length]
        auto flat_input = x.view({batch_size, seq_len});

        // Encode and decode
        auto latent = encoder->forward(flat_input);
        auto reconstructed = decoder->forward(latent);

        // Reshape back to [batch, 1, signal_length]
        return reconstructed.view({batch_size, 1, seq_len});
    }

    torch::Tensor encode(torch::Tensor x) {
        auto batch_size = x.size(0);
        auto seq_len = x.size(2);
        auto flat_input = x.view({batch_size, seq_len});
        return encoder->forward(flat_input);
    }

private:
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential decoder{nullptr};
    int input_size;
    int latent_dim;
};

TORCH_MODULE(ECGAutoencoder);

// ============================================================================
// ECG DATASET LOADER - Nightingale Dataset
// ============================================================================

struct EEGRecord {
    std::vector<std::vector<float>> leads; // 12 leads: I, II, III, aVR, aVL, aVF, V1-V6
    float sampling_rate;
    std::string record_name;
    int num_channels;

    static const std::vector<std::string> lead_names;
    static const std::vector<ImVec4> lead_colors;
};

const std::vector<std::string> EEGRecord::lead_names = {
    "Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
};

const std::vector<ImVec4> EEGRecord::lead_colors = {
    ImVec4(1.0f, 0.2f, 0.2f, 1.0f),  // Red
    ImVec4(0.2f, 1.0f, 0.2f, 1.0f),  // Green
    ImVec4(0.2f, 0.5f, 1.0f, 1.0f),  // Blue
    ImVec4(1.0f, 1.0f, 0.2f, 1.0f),  // Yellow
    ImVec4(1.0f, 0.5f, 0.2f, 1.0f),  // Orange
    ImVec4(0.8f, 0.2f, 1.0f, 1.0f),  // Purple
    ImVec4(0.2f, 1.0f, 1.0f, 1.0f),  // Cyan
    ImVec4(1.0f, 0.2f, 0.6f, 1.0f),  // Pink
    ImVec4(0.5f, 1.0f, 0.2f, 1.0f),  // Lime
    ImVec4(1.0f, 0.7f, 0.2f, 1.0f),  // Gold
    ImVec4(0.6f, 0.3f, 1.0f, 1.0f),  // Violet
    ImVec4(0.2f, 0.8f, 0.6f, 1.0f)   // Teal
};

class EEGDataLoader {
public:
    explicit EEGDataLoader(const std::string& data_path)
        : data_path_(data_path) {}

    bool load_nightingale_data() {
        // Try to load from cache first
        std::string cache_file = data_path_ + "/nightingale_cache.bin";
        if (load_from_cache(cache_file)) {
            std::cout << "Loaded from cache: " << records_.size() << " records" << std::endl;
            return true;
        }

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
            std::cout << "Loaded " << EEGRecord::lead_names[lead_idx]
                     << " (" << all_leads_data[lead_idx].size() << " records)" << std::endl;
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

        // Save to cache for next time
        save_to_cache(cache_file);
        std::cout << "Saved cache to: " << cache_file << std::endl;

        return true;
    }

    const std::vector<EEGRecord>& get_records() const { return records_; }

private:
    bool load_from_cache(const std::string& cache_file) {
        std::ifstream file(cache_file, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        try {
            // Read number of records
            size_t num_records;
            file.read(reinterpret_cast<char*>(&num_records), sizeof(num_records));

            records_.clear();
            records_.reserve(num_records);

            for (size_t i = 0; i < num_records; i++) {
                EEGRecord record;

                // Read sampling rate
                file.read(reinterpret_cast<char*>(&record.sampling_rate), sizeof(record.sampling_rate));

                // Read record name length and data
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                record.record_name.resize(name_len);
                file.read(&record.record_name[0], name_len);

                // Read num channels
                file.read(reinterpret_cast<char*>(&record.num_channels), sizeof(record.num_channels));

                // Read all 12 leads
                record.leads.resize(12);
                for (int lead_idx = 0; lead_idx < 12; lead_idx++) {
                    size_t lead_size;
                    file.read(reinterpret_cast<char*>(&lead_size), sizeof(lead_size));
                    record.leads[lead_idx].resize(lead_size);
                    file.read(reinterpret_cast<char*>(record.leads[lead_idx].data()),
                             lead_size * sizeof(float));
                }

                records_.push_back(std::move(record));
            }

            file.close();
            return true;
        } catch (...) {
            std::cerr << "Error reading cache file" << std::endl;
            records_.clear();
            return false;
        }
    }

    void save_to_cache(const std::string& cache_file) {
        std::ofstream file(cache_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not create cache file" << std::endl;
            return;
        }

        // Write number of records
        size_t num_records = records_.size();
        file.write(reinterpret_cast<const char*>(&num_records), sizeof(num_records));

        for (const auto& record : records_) {
            // Write sampling rate
            file.write(reinterpret_cast<const char*>(&record.sampling_rate), sizeof(record.sampling_rate));

            // Write record name
            size_t name_len = record.record_name.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(record.record_name.data(), name_len);

            // Write num channels
            file.write(reinterpret_cast<const char*>(&record.num_channels), sizeof(record.num_channels));

            // Write all 12 leads
            for (int lead_idx = 0; lead_idx < 12; lead_idx++) {
                size_t lead_size = record.leads[lead_idx].size();
                file.write(reinterpret_cast<const char*>(&lead_size), sizeof(lead_size));
                file.write(reinterpret_cast<const char*>(record.leads[lead_idx].data()),
                          lead_size * sizeof(float));
            }
        }

        file.close();
    }

private:
    std::vector<std::vector<float>> load_csv_file(const std::string& filepath) {
        std::vector<std::vector<float>> data;

        std::ifstream file(filepath);

        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file: " << filepath << std::endl;
            return data;
        }

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
// CLINICAL ECG FEATURE EXTRACTION
// ============================================================================

struct ECGFeatures {
    std::vector<int> r_peaks;           // R-peak locations (QRS complex)
    std::vector<float> rr_intervals;    // RR intervals in ms
    float heart_rate_bpm = 0.0f;        // Average heart rate
    float hrv_sdnn = 0.0f;              // Heart rate variability (SDNN)
    float qt_interval_avg = 0.0f;       // Average QT interval
    int num_beats = 0;
};

class ECGAnalyzer {
public:
    // Simple Pan-Tompkins inspired QRS detection
    static ECGFeatures extract_features(const std::vector<float>& signal, float sampling_rate) {
        ECGFeatures features;

        if (signal.size() < 100) return features;

        // Step 1: Differentiation (approximate derivative)
        std::vector<float> diff_signal(signal.size() - 1);
        for (size_t i = 0; i < signal.size() - 1; i++) {
            diff_signal[i] = signal[i + 1] - signal[i];
        }

        // Step 2: Squaring
        std::vector<float> squared_signal(diff_signal.size());
        for (size_t i = 0; i < diff_signal.size(); i++) {
            squared_signal[i] = diff_signal[i] * diff_signal[i];
        }

        // Step 3: Moving average filter
        int window_size = static_cast<int>(sampling_rate * 0.15f); // 150ms window
        std::vector<float> filtered_signal(squared_signal.size());
        for (size_t i = 0; i < squared_signal.size(); i++) {
            float sum = 0.0f;
            int count = 0;
            for (int j = std::max(0, static_cast<int>(i) - window_size / 2);
                 j < std::min(static_cast<int>(squared_signal.size()), static_cast<int>(i) + window_size / 2);
                 j++) {
                sum += squared_signal[j];
                count++;
            }
            filtered_signal[i] = sum / count;
        }

        // Step 4: Adaptive thresholding for peak detection
        float threshold = 0.0f;
        for (float val : filtered_signal) {
            threshold += val;
        }
        threshold = (threshold / filtered_signal.size()) * 1.5f; // 1.5x mean

        // Step 5: Find peaks above threshold
        int min_distance = static_cast<int>(sampling_rate * 0.2f); // Min 200ms between peaks
        int last_peak = -min_distance;

        for (size_t i = 1; i < filtered_signal.size() - 1; i++) {
            if (filtered_signal[i] > threshold &&
                filtered_signal[i] > filtered_signal[i - 1] &&
                filtered_signal[i] > filtered_signal[i + 1] &&
                static_cast<int>(i) - last_peak > min_distance) {

                features.r_peaks.push_back(static_cast<int>(i));
                last_peak = static_cast<int>(i);
            }
        }

        features.num_beats = features.r_peaks.size();

        // Calculate RR intervals and heart rate
        if (features.r_peaks.size() > 1) {
            for (size_t i = 1; i < features.r_peaks.size(); i++) {
                float rr_ms = (features.r_peaks[i] - features.r_peaks[i - 1]) * 1000.0f / sampling_rate;
                features.rr_intervals.push_back(rr_ms);
            }

            // Heart rate (BPM)
            float avg_rr_ms = std::accumulate(features.rr_intervals.begin(),
                                             features.rr_intervals.end(), 0.0f) / features.rr_intervals.size();
            features.heart_rate_bpm = 60000.0f / avg_rr_ms;

            // HRV - SDNN (standard deviation of RR intervals)
            float mean_rr = avg_rr_ms;
            float variance = 0.0f;
            for (float rr : features.rr_intervals) {
                variance += (rr - mean_rr) * (rr - mean_rr);
            }
            features.hrv_sdnn = std::sqrt(variance / features.rr_intervals.size());

            // Estimate QT interval (simplified: ~40% of RR interval)
            features.qt_interval_avg = avg_rr_ms * 0.4f;
        }

        return features;
    }
};

// ============================================================================
// REAL-TIME ML ENGINE
// ============================================================================

class MLEngine {
public:
    MLEngine(torch::Device device) : device_(device) {
        model_ = ECGAutoencoder(5000, 64);
        model_->to(device_);

        std::cout << "Neural Network initialized on " << device_ << std::endl;
        std::cout << "   Architecture: Fully Connected Autoencoder" << std::endl;
        std::cout << "   Input: 5000 samples -> Latent: 64 dims" << std::endl;
    }

    void train_on_batch(const std::vector<std::vector<float>>& batch) {
        if (batch.empty()) return;

        auto start = std::chrono::high_resolution_clock::now();

        // Convert to tensor
        std::vector<float> flat_batch;
        for (const auto& signal : batch) {
            flat_batch.insert(flat_batch.end(), signal.begin(), signal.end());
        }

        auto input = torch::from_blob(flat_batch.data(),
                                     {static_cast<long>(batch.size()), 1,
                                      static_cast<long>(batch[0].size())},
                                     torch::kFloat32).to(device_);

        // Forward pass
        model_->train();
        auto output = model_->forward(input);

        // Compute reconstruction loss
        auto loss = torch::mse_loss(output, input);

        // Backward pass
        optimizer_->zero_grad();
        loss.backward();
        optimizer_->step();

        auto end = std::chrono::high_resolution_clock::now();
        last_training_time_ = std::chrono::duration<float, std::milli>(end - start).count();
        last_loss_ = loss.item<float>();
        total_batches_++;
    }

    std::pair<std::vector<float>, float> infer(const std::vector<float>& signal) {
        if (signal.empty()) return {{}, 0.0f};

        auto start = std::chrono::high_resolution_clock::now();

        // Convert to tensor
        auto input = torch::from_blob(const_cast<float*>(signal.data()),
                                     {1, 1, static_cast<long>(signal.size())},
                                     torch::kFloat32).to(device_);

        // Inference
        model_->eval();
        torch::NoGradGuard no_grad;
        auto output = model_->forward(input);

        // Compute anomaly score (reconstruction error)
        auto mse = torch::mse_loss(output, input);
        float anomaly_score = mse.item<float>();

        // Get reconstruction
        auto output_cpu = output.to(torch::kCPU);
        std::vector<float> reconstruction(signal.size());
        std::memcpy(reconstruction.data(), output_cpu.data_ptr<float>(),
                   signal.size() * sizeof(float));

        auto end = std::chrono::high_resolution_clock::now();
        last_inference_time_ = std::chrono::duration<float, std::milli>(end - start).count();

        return {reconstruction, anomaly_score};
    }

    std::vector<float> get_latent_representation(const std::vector<float>& signal) {
        if (signal.empty()) return {};

        auto input = torch::from_blob(const_cast<float*>(signal.data()),
                                     {1, 1, static_cast<long>(signal.size())},
                                     torch::kFloat32).to(device_);

        model_->eval();
        torch::NoGradGuard no_grad;
        auto latent = model_->encode(input);

        auto latent_cpu = latent.to(torch::kCPU);
        std::vector<float> latent_vec(latent_cpu.size(1));
        std::memcpy(latent_vec.data(), latent_cpu.data_ptr<float>(),
                   latent_vec.size() * sizeof(float));

        return latent_vec;
    }

    void initialize_training(float learning_rate = 0.001f) {
        optimizer_ = std::make_shared<torch::optim::Adam>(
            model_->parameters(), torch::optim::AdamOptions(learning_rate));
        std::cout << "Optimizer initialized (Adam, lr=" << learning_rate << ")" << std::endl;
    }

    float get_last_loss() const { return last_loss_; }
    float get_last_training_time() const { return last_training_time_; }
    float get_last_inference_time() const { return last_inference_time_; }
    int get_total_batches() const { return total_batches_; }

private:
    ECGAutoencoder model_{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_;
    torch::Device device_;

    float last_loss_ = 0.0f;
    float last_training_time_ = 0.0f;
    float last_inference_time_ = 0.0f;
    int total_batches_ = 0;
};

// ============================================================================
// MAIN APPLICATION
// ============================================================================

class ECGMLDemo {
public:
    ECGMLDemo() : window_(nullptr), current_record_idx_(0),
                  show_reconstruction_(false), training_enabled_(false),
                  ui_scale_(1.0f) {
        detect_device();
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

        window_ = glfwCreateWindow(1920, 1080,
                                   "Caliper - CUDA-Accelerated ECG ML Demo",
                                   nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);

        // Initialize GLEW
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if (err != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
            glfwTerminate();
            return false;
        }

        // Setup ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();
        customize_style();

        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        // Set initial UI scale
        apply_ui_scale();

        // Load data
        std::string data_path = "../data/Nightingale Dataset";
        loader_ = std::make_unique<EEGDataLoader>(data_path);
        if (!loader_->load_nightingale_data()) {
            return false;
        }
        records_ = loader_->get_records();

        // Initialize ML engine
        ml_engine_ = std::make_unique<MLEngine>(device_);
        ml_engine_->initialize_training(0.001f);

        // Run initial inference to show results immediately
        update_inference();

        std::cout << "Demo initialized successfully!" << std::endl;
        return true;
    }

    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            // Update simulation
            update();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            render_ui();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
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
    void detect_device() {
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            device_name_ = "CUDA GPU";
            device_color_ = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);
            std::cout << "CUDA device detected!" << std::endl;
            std::cout << "   GPU: " << torch::cuda::device_count() << " device(s) available" << std::endl;
        } else if (torch::mps::is_available()) {
            device_ = torch::kMPS;
            device_name_ = "MPS (Apple Silicon)";
            device_color_ = ImVec4(0.2f, 0.8f, 1.0f, 1.0f);
        } else {
            device_ = torch::kCPU;
            device_name_ = "CPU";
            device_color_ = ImVec4(1.0f, 1.0f, 0.2f, 1.0f);
        }
    }

    void customize_style() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 8.0f;
        style.FrameRounding = 4.0f;
        style.GrabRounding = 4.0f;
        style.ScrollbarRounding = 4.0f;

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.14f, 0.95f);
        colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.35f, 1.00f);
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.35f, 0.45f, 1.00f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.30f, 0.40f, 1.00f);
        colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.35f, 1.00f);
        colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.35f, 0.45f, 1.00f);
        colors[ImGuiCol_ButtonActive] = ImVec4(0.15f, 0.20f, 0.30f, 1.00f);
    }

    void apply_ui_scale() {
        ImGuiIO& io = ImGui::GetIO();
        io.FontGlobalScale = ui_scale_;
    }

    void update() {
        if (!records_.empty() && training_enabled_) {
            // Perform ML inference on first lead
            update_inference();
        }
    }

    void update_inference() {
        if (records_.empty()) return;

        const auto& record = records_[current_record_idx_];

        // Use Lead II (index 1) for ML analysis
        auto& lead_data = record.leads[1];

        // Extract clinical features
        current_features_ = ECGAnalyzer::extract_features(lead_data, record.sampling_rate);

        // Pad or truncate to model input size
        std::vector<float> segment = lead_data;
        segment.resize(5000, 0.0f);

        auto [recon, anomaly] = ml_engine_->infer(segment);
        current_reconstruction_ = recon;
        current_anomaly_score_ = anomaly;
        current_latent_ = ml_engine_->get_latent_representation(segment);

        // Training step
        if (training_enabled_ && ml_engine_->get_total_batches() < 1000) {
            ml_engine_->train_on_batch({segment});
        }
    }

    void render_ui() {
        // Main control panel
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
        ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::TextColored(device_color_, "Compute: %s", device_name_.c_str());
        ImGui::Separator();

        // Patient Navigation
        ImGui::Text("Patient Navigation");
        int idx = current_record_idx_;

        if (ImGui::Button("<<< Previous")) {
            if (current_record_idx_ > 0) {
                current_record_idx_--;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Next >>>")) {
            if (current_record_idx_ < records_.size() - 1) {
                current_record_idx_++;
            }
        }

        if (ImGui::SliderInt("Patient", &idx, 0, records_.size() - 1)) {
            current_record_idx_ = idx;
        }

        ImGui::Text("Viewing: Patient %d of %zu", current_record_idx_ + 1, records_.size());

        ImGui::Separator();
        ImGui::Text("UI Scale");
        if (ImGui::Button("Zoom In (+)", ImVec2(85, 0))) {
            ui_scale_ = std::min(ui_scale_ + 0.25f, 3.0f);
            apply_ui_scale();
        }
        ImGui::SameLine();
        if (ImGui::Button("Zoom Out (-)", ImVec2(85, 0))) {
            ui_scale_ = std::max(ui_scale_ - 0.25f, 0.5f);
            apply_ui_scale();
        }
        ImGui::Text("Current Scale: %.2fx", ui_scale_);

        ImGui::Separator();
        ImGui::Text("ML Controls");

        if (ImGui::Button("Run Inference", ImVec2(180, 30))) {
            update_inference();
        }

        if (ImGui::Button("Train Model (1 batch)", ImVec2(180, 30))) {
            const auto& record = records_[current_record_idx_];
            std::vector<float> segment = record.leads[1];
            segment.resize(5000, 0.0f);
            ml_engine_->train_on_batch({segment});
            update_inference();
        }

        if (ImGui::Button("Reset Model", ImVec2(180, 30))) {
            ml_engine_ = std::make_unique<MLEngine>(device_);
            ml_engine_->initialize_training(0.001f);
            update_inference();
        }

        ImGui::Checkbox("Show Reconstruction", &show_reconstruction_);
        if (ImGui::Checkbox("Auto-train on view", &training_enabled_)) {
            if (training_enabled_) {
                update_inference();
            }
        }

        ImGui::Separator();
        ImGui::Text("What You're Seeing:");
        ImGui::TextWrapped("Top: Original ECG vs ML reconstruction");
        ImGui::TextWrapped("Middle: Error map - spikes show where ML struggles (anomalies)");
        ImGui::TextWrapped("Bottom: Neural network activations");

        ImGui::Separator();
        ImGui::Text("ML Stats");
        ImGui::Text("Anomaly Score: %.6f", current_anomaly_score_);
        ImGui::Text("Training: %.2f ms", ml_engine_->get_last_training_time());
        ImGui::Text("Inference: %.2f ms", ml_engine_->get_last_inference_time());
        ImGui::Text("Loss: %.6f", ml_engine_->get_last_loss());
        ImGui::Text("Batches: %d", ml_engine_->get_total_batches());

        ImGui::End();


        // Main waveform display - Lead II only
        ImGui::SetNextWindowPos(ImVec2(420, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(1490, 350), ImGuiCond_FirstUseEver);
        ImGui::Begin("1. ECG Waveform - Lead II", nullptr, ImGuiWindowFlags_NoCollapse);

        if (!records_.empty()) {
            const auto& record = records_[current_record_idx_];
            auto& lead_data = record.leads[1]; // Lead II
            int total_samples = lead_data.size();

            ImGui::Text("Patient: %s | Samples: %d | Duration: %.1f sec",
                       record.record_name.c_str(), total_samples,
                       total_samples / record.sampling_rate);
            ImGui::SameLine();
            ImGui::TextColored(current_anomaly_score_ > 0.001f ? ImVec4(1,0,0,1) : ImVec4(0,1,0,1),
                             "Anomaly Score: %.6f %s", current_anomaly_score_,
                             current_anomaly_score_ > 0.001f ? "(HIGH)" : "(LOW)");

            if (ImPlot::BeginPlot("##lead2", ImVec2(-1, 280))) {
                ImPlot::SetupAxes("Time (samples)", "Amplitude (mV)");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, total_samples, ImGuiCond_Always);

                float min_val = *std::min_element(lead_data.begin(), lead_data.end());
                float max_val = *std::max_element(lead_data.begin(), lead_data.end());
                float margin = (max_val - min_val) * 0.1f;
                ImPlot::SetupAxisLimits(ImAxis_Y1, min_val - margin, max_val + margin, ImGuiCond_Always);

                ImPlot::PlotLine("Original Signal", lead_data.data(), total_samples);

                // Show reconstruction if available
                if (show_reconstruction_ && !current_reconstruction_.empty()) {
                    int recon_size = std::min(static_cast<int>(current_reconstruction_.size()), total_samples);
                    ImPlot::PlotLine("ML Reconstruction", current_reconstruction_.data(), recon_size);
                }

                // Mark R-peaks (QRS complexes)
                if (!current_features_.r_peaks.empty()) {
                    std::vector<double> peak_x(current_features_.r_peaks.size());
                    std::vector<double> peak_y(current_features_.r_peaks.size());
                    for (size_t i = 0; i < current_features_.r_peaks.size(); i++) {
                        peak_x[i] = current_features_.r_peaks[i];
                        peak_y[i] = lead_data[current_features_.r_peaks[i]];
                    }
                    ImPlot::PlotScatter("R-peaks (QRS)", peak_x.data(), peak_y.data(),
                                       static_cast<int>(peak_x.size()));
                }

                ImPlot::EndPlot();
            }
        }

        ImGui::End();

        // ML Reconstruction Error Heatmap
        ImGui::SetNextWindowPos(ImVec2(420, 370), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(1490, 350), ImGuiCond_FirstUseEver);
        ImGui::Begin("2. ML Analysis - Anomaly Detection (Reconstruction Error)", nullptr, ImGuiWindowFlags_NoCollapse);

        if (!records_.empty() && !current_reconstruction_.empty()) {
            const auto& record = records_[current_record_idx_];
            auto& lead_data = record.leads[1];
            int display_samples = std::min(static_cast<int>(lead_data.size()),
                                          static_cast<int>(current_reconstruction_.size()));

            // Calculate point-wise reconstruction error
            std::vector<float> reconstruction_error(display_samples);
            for (int i = 0; i < display_samples; i++) {
                float diff = lead_data[i] - current_reconstruction_[i];
                reconstruction_error[i] = diff * diff; // Squared error
            }

            ImGui::Text("Point-wise Reconstruction Error - Spikes indicate where the model fails (potential anomalies)");
            ImGui::Text("Overall MSE: %.6f | Max Local Error: %.6f",
                       current_anomaly_score_,
                       *std::max_element(reconstruction_error.begin(), reconstruction_error.end()));

            if (ImPlot::BeginPlot("##error", ImVec2(-1, 280))) {
                ImPlot::SetupAxes("Time (samples)", "Squared Error");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, display_samples, ImGuiCond_Always);

                float max_error = *std::max_element(reconstruction_error.begin(), reconstruction_error.end());
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_error * 1.1f, ImGuiCond_Always);

                ImPlot::PlotLine("Reconstruction Error", reconstruction_error.data(), display_samples);
                ImPlot::PlotShaded("Error Magnitude", reconstruction_error.data(), display_samples, 0.0);

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextWrapped("Click 'Run Inference' to analyze this ECG.");
            ImGui::Separator();
            ImGui::Text("How it works:");
            ImGui::TextWrapped("- Untrained model: Random reconstruction = high error everywhere");
            ImGui::TextWrapped("- Trained model: Low error on normal patterns, high error on anomalies");
            ImGui::TextWrapped("- Click 'Train Model' multiple times to see error decrease");
        }

        ImGui::End();

        // Anomaly & Latent Space
        ImGui::SetNextWindowPos(ImVec2(420, 720), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(740, 350), ImGuiCond_FirstUseEver);
        ImGui::Begin("3. Neural Network Latent Space", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::Text("64-Dimensional Compressed Representation");
        ImGui::Text("Anomaly Score: %.6f", current_anomaly_score_);

        // Visual anomaly meter
        float normalized_score = std::min(current_anomaly_score_ * 1000.0f, 1.0f);
        ImVec4 meter_color = ImVec4(
            normalized_score,
            1.0f - normalized_score,
            0.2f, 1.0f
        );
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, meter_color);
        ImGui::ProgressBar(normalized_score, ImVec2(-1, 40), "Anomaly Level");
        ImGui::PopStyleColor();

        if (!current_latent_.empty()) {
            ImGui::Text("These 64 values represent the compressed ECG pattern");
            if (ImPlot::BeginPlot("##latent", ImVec2(-1, 220))) {
                ImPlot::SetupAxes("Feature Dimension", "Activation Value");
                ImPlot::PlotBars("Latent Features", current_latent_.data(),
                               std::min(64, static_cast<int>(current_latent_.size())));
                ImPlot::EndPlot();
            }
        }

        ImGui::End();

        // Clinical Features Panel
        ImGui::SetNextWindowPos(ImVec2(1170, 720), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(740, 350), ImGuiCond_FirstUseEver);
        ImGui::Begin("4. Clinical ECG Features", nullptr, ImGuiWindowFlags_NoCollapse);

        if (!records_.empty()) {
            const auto& record = records_[current_record_idx_];
            ImGui::Text("Patient: %s", record.record_name.c_str());
            ImGui::Text("Duration: %.1f sec | Sample Rate: %.0f Hz",
                       record.leads[0].size() / record.sampling_rate, record.sampling_rate);

            ImGui::Separator();
            ImGui::Text("CLINICAL MEASUREMENTS (Pan-Tompkins Algorithm)");
            ImGui::Separator();

            // Heart Rate
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Heart Rate:");
            if (current_features_.heart_rate_bpm > 0) {
                ImVec4 hr_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green = normal
                if (current_features_.heart_rate_bpm < 60) hr_color = ImVec4(0.2f, 0.6f, 1.0f, 1.0f); // Blue = bradycardia
                if (current_features_.heart_rate_bpm > 100) hr_color = ImVec4(1.0f, 0.2f, 0.2f, 1.0f); // Red = tachycardia

                ImGui::TextColored(hr_color, "  %.1f BPM", current_features_.heart_rate_bpm);
                if (current_features_.heart_rate_bpm < 60) ImGui::Text("  (Bradycardia - slow)");
                else if (current_features_.heart_rate_bpm > 100) ImGui::Text("  (Tachycardia - fast)");
                else ImGui::Text("  (Normal)");
            } else {
                ImGui::Text("  N/A");
            }

            ImGui::Separator();

            // QRS Complexes
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "QRS Complexes Detected:");
            ImGui::Text("  %d beats", current_features_.num_beats);

            ImGui::Separator();

            // RR Intervals
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "RR Interval:");
            if (!current_features_.rr_intervals.empty()) {
                float avg_rr = std::accumulate(current_features_.rr_intervals.begin(),
                                               current_features_.rr_intervals.end(), 0.0f)
                              / current_features_.rr_intervals.size();
                ImGui::Text("  Avg: %.1f ms", avg_rr);
            } else {
                ImGui::Text("  N/A");
            }

            ImGui::Separator();

            // Heart Rate Variability
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Heart Rate Variability (SDNN):");
            if (current_features_.hrv_sdnn > 0) {
                ImGui::Text("  %.2f ms", current_features_.hrv_sdnn);
                ImGui::TextWrapped("  (Higher = healthier autonomic function)");
            } else {
                ImGui::Text("  N/A");
            }

            ImGui::Separator();

            // QT Interval
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "QT Interval (estimated):");
            if (current_features_.qt_interval_avg > 0) {
                ImGui::Text("  %.1f ms", current_features_.qt_interval_avg);
                if (current_features_.qt_interval_avg > 440)
                    ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "  (Prolonged - risk marker)");
                else
                    ImGui::Text("  (Normal)");
            } else {
                ImGui::Text("  N/A");
            }
        }

        ImGui::End();
    }

    GLFWwindow* window_;
    std::unique_ptr<EEGDataLoader> loader_;
    std::unique_ptr<MLEngine> ml_engine_;
    std::vector<EEGRecord> records_;

    torch::Device device_ = torch::kCPU;
    std::string device_name_;
    ImVec4 device_color_;

    size_t current_record_idx_;
    bool show_reconstruction_;
    bool training_enabled_;
    float ui_scale_;

    std::vector<float> current_reconstruction_;
    std::vector<float> current_latent_;
    float current_anomaly_score_ = 0.0f;
    ECGFeatures current_features_;
};

// ============================================================================
// ENTRY POINT
// ============================================================================

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "    CALIPER - CUDA ECG ML DEMO                         " << std::endl;
    std::cout << "    Real-time Deep Learning + Visualization            " << std::endl;
    std::cout << "========================================================" << std::endl;

    ECGMLDemo app;

    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }

    app.run();
    app.cleanup();

    return 0;
}
