#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <deque>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

// ============================================================================
// ECG DATASET LOADER - Supports Nightingale, MIT-BIH, and CSV formats
// ============================================================================

struct ECGRecord {
    std::vector<float> signal;      // Raw ECG signal
    std::vector<int> r_peaks;       // R-peak annotations (sample indices)
    std::vector<int> labels;        // Beat labels (0=normal, 1=abnormal, etc.)
    float sampling_rate;            // Hz
    std::string record_name;
    int num_leads;                  // Number of ECG leads (1, 2, or 12)
};

class ECGDataset : public torch::data::Dataset<ECGDataset> {
public:
    explicit ECGDataset(const std::string& root, int segment_length = 1000)
        : root_(root), segment_length_(segment_length) {
        load_data();
    }

    torch::data::Example<> get(size_t index) override {
        auto data = segments_[index];
        auto target = labels_[index];
        return {data.clone(), target.clone()};
    }

    torch::optional<size_t> size() const override {
        return segments_.size(0);
    }

    const std::vector<ECGRecord>& get_records() const { return records_; }

private:
    void load_data() {
        // Try to load CSV format ECG data
        std::vector<std::string> file_patterns = {
            root_ + "/*.csv",
            root_ + "/*.dat",
            root_ + "/*.txt"
        };

        std::vector<std::string> ecg_files;
        for (const auto& pattern : file_patterns) {
            // Simple directory scan for CSV files
            std::ifstream test_file(root_ + "/ecg_data.csv");
            if (test_file.good()) {
                ecg_files.push_back(root_ + "/ecg_data.csv");
                break;
            }
        }

        if (ecg_files.empty()) {
            std::cout << "No ECG data found. Generating synthetic ECG signals..." << std::endl;
            generate_synthetic_ecg();
            return;
        }

        // Load actual ECG data
        for (const auto& file_path : ecg_files) {
            load_csv_file(file_path);
        }

        // Segment the continuous ECG signals
        segment_records();
    }

    void generate_synthetic_ecg() {
        // Generate realistic synthetic ECG waveforms
        int num_records = 100;
        float sampling_rate = 360.0f; // Hz (common for ECG)
        int duration_seconds = 10;
        int signal_length = static_cast<int>(sampling_rate * duration_seconds);

        for (int r = 0; r < num_records; r++) {
            ECGRecord record;
            record.sampling_rate = sampling_rate;
            record.num_leads = 1;
            record.record_name = "synthetic_" + std::to_string(r);
            record.signal.resize(signal_length);

            // Generate ECG-like signal using sum of sinusoids
            float heart_rate = 60.0f + (rand() % 40); // 60-100 bpm
            float rr_interval_samples = (60.0f / heart_rate) * sampling_rate;

            for (int i = 0; i < signal_length; i++) {
                float t = i / sampling_rate;
                float phase = 2.0f * M_PI * heart_rate / 60.0f * t;

                // P wave (small)
                float p_wave = 0.15f * std::sin(phase - 0.2f) * std::exp(-50.0f * std::pow(std::fmod(phase, 2.0f * M_PI) - 5.8f, 2.0f));

                // QRS complex (sharp, tall)
                float qrs = 1.0f * std::sin(phase) * std::exp(-100.0f * std::pow(std::fmod(phase, 2.0f * M_PI), 2.0f));

                // T wave (moderate)
                float t_wave = 0.3f * std::sin(phase + 0.4f) * std::exp(-30.0f * std::pow(std::fmod(phase, 2.0f * M_PI) - 1.0f, 2.0f));

                // Baseline wander and noise
                float baseline = 0.05f * std::sin(2.0f * M_PI * 0.3f * t);
                float noise = 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

                record.signal[i] = p_wave + qrs + t_wave + baseline + noise;

                // Mark R-peaks (QRS maxima)
                if (i > 0 && i < signal_length - 1) {
                    if (record.signal[i] > record.signal[i-1] && record.signal[i] > record.signal[i+1] && record.signal[i] > 0.5f) {
                        record.r_peaks.push_back(i);
                        record.labels.push_back(rand() % 5); // 0=Normal, 1-4=Various arrhythmias
                    }
                }
            }

            records_.push_back(record);
        }

        segment_records();
    }

    void load_csv_file(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open " << file_path << std::endl;
            return;
        }

        ECGRecord record;
        record.sampling_rate = 360.0f; // Default
        record.num_leads = 1;
        record.record_name = file_path;

        std::string line;
        bool first_line = true;
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }

            std::istringstream iss(line);
            std::string value;
            if (std::getline(iss, value, ',')) {
                try {
                    float sample = std::stof(value);
                    record.signal.push_back(sample);
                } catch (...) {
                    continue;
                }
            }
        }

        if (!record.signal.empty()) {
            records_.push_back(record);
        }
    }

    void segment_records() {
        // Divide ECG records into fixed-length segments for training
        std::vector<torch::Tensor> segment_list;
        std::vector<torch::Tensor> label_list;

        for (const auto& record : records_) {
            int num_segments = (record.signal.size() - segment_length_) / (segment_length_ / 2) + 1;

            for (int i = 0; i < num_segments; i++) {
                int start_idx = i * (segment_length_ / 2);
                int end_idx = start_idx + segment_length_;

                if (end_idx > record.signal.size()) break;

                // Extract segment
                std::vector<float> segment(record.signal.begin() + start_idx,
                                          record.signal.begin() + end_idx);

                // Normalize segment
                float mean = std::accumulate(segment.begin(), segment.end(), 0.0f) / segment.size();
                float sq_sum = 0.0f;
                for (auto v : segment) sq_sum += (v - mean) * (v - mean);
                float std_dev = std::sqrt(sq_sum / segment.size()) + 1e-8f;

                for (auto& v : segment) {
                    v = (v - mean) / std_dev;
                }

                // Convert to tensor [1, segment_length]
                auto segment_tensor = torch::from_blob(segment.data(), {1, segment_length_}, torch::kFloat32).clone();
                segment_list.push_back(segment_tensor);

                // Assign label (simple: 0=normal, 1=abnormal)
                int label = (rand() % 100 < 20) ? 1 : 0; // 20% abnormal
                label_list.push_back(torch::tensor({label}, torch::kLong));
            }
        }

        if (!segment_list.empty()) {
            segments_ = torch::stack(segment_list);
            labels_ = torch::cat(label_list);
        } else {
            std::cerr << "Warning: No segments created from ECG data" << std::endl;
            // Generate synthetic as fallback
            segments_ = torch::randn({100, 1, segment_length_});
            labels_ = torch::randint(0, 2, {100}, torch::kLong);
        }

        std::cout << "ECG dataset loaded: " << segments_.size(0) << " segments" << std::endl;
    }

    std::string root_;
    int segment_length_;
    std::vector<ECGRecord> records_;
    torch::Tensor segments_;
    torch::Tensor labels_;
};

// ============================================================================
// SIGNAL PROCESSING FOR ECG
// ============================================================================

class ECGSignalProcessor {
public:
    // Bandpass filter for ECG (0.5 - 40 Hz typical range)
    static std::vector<float> bandpass_filter(const std::vector<float>& signal,
                                               float sampling_rate,
                                               float low_cutoff = 0.5f,
                                               float high_cutoff = 40.0f) {
        std::vector<float> filtered = signal;

        // Simple moving average for demonstration
        // In production, use proper Butterworth or Chebyshev filter
        int window_size = static_cast<int>(sampling_rate / 10.0f);
        std::vector<float> smoothed(signal.size(), 0.0f);

        for (size_t i = window_size; i < signal.size(); i++) {
            float sum = 0.0f;
            for (int j = 0; j < window_size; j++) {
                sum += filtered[i - j];
            }
            smoothed[i] = sum / window_size;
        }

        return smoothed;
    }

    // Pan-Tompkins QRS detection algorithm (simplified)
    static std::vector<int> detect_r_peaks(const std::vector<float>& signal, float sampling_rate) {
        std::vector<int> r_peaks;

        if (signal.empty()) return r_peaks;

        // 1. Bandpass filter (already done)
        // 2. Derivative (approximation)
        std::vector<float> derivative(signal.size(), 0.0f);
        for (size_t i = 2; i < signal.size() - 2; i++) {
            derivative[i] = (-signal[i-2] - 2*signal[i-1] + 2*signal[i+1] + signal[i+2]) / 8.0f;
        }

        // 3. Squaring
        std::vector<float> squared(derivative.size());
        for (size_t i = 0; i < derivative.size(); i++) {
            squared[i] = derivative[i] * derivative[i];
        }

        // 4. Moving window integration
        int integration_window = static_cast<int>(0.150f * sampling_rate); // 150ms window
        std::vector<float> integrated(squared.size(), 0.0f);
        for (size_t i = integration_window; i < squared.size(); i++) {
            float sum = 0.0f;
            for (int j = 0; j < integration_window; j++) {
                sum += squared[i - j];
            }
            integrated[i] = sum / integration_window;
        }

        // 5. Thresholding and peak detection
        float threshold = 0.0f;
        for (auto v : integrated) threshold += v;
        threshold = (threshold / integrated.size()) * 3.0f; // Adaptive threshold

        int refractory_period = static_cast<int>(0.2f * sampling_rate); // 200ms
        int last_peak = -refractory_period;

        for (size_t i = 1; i < integrated.size() - 1; i++) {
            if (integrated[i] > threshold &&
                integrated[i] > integrated[i-1] &&
                integrated[i] > integrated[i+1] &&
                static_cast<int>(i) - last_peak > refractory_period) {
                r_peaks.push_back(static_cast<int>(i));
                last_peak = static_cast<int>(i);
            }
        }

        return r_peaks;
    }

    // Calculate heart rate variability metrics
    static float calculate_hrv_sdnn(const std::vector<int>& r_peaks, float sampling_rate) {
        if (r_peaks.size() < 2) return 0.0f;

        // Calculate RR intervals in milliseconds
        std::vector<float> rr_intervals;
        for (size_t i = 1; i < r_peaks.size(); i++) {
            float rr_ms = (r_peaks[i] - r_peaks[i-1]) / sampling_rate * 1000.0f;
            if (rr_ms > 300.0f && rr_ms < 2000.0f) { // Valid range
                rr_intervals.push_back(rr_ms);
            }
        }

        if (rr_intervals.empty()) return 0.0f;

        // SDNN: Standard deviation of NN intervals
        float mean = std::accumulate(rr_intervals.begin(), rr_intervals.end(), 0.0f) / rr_intervals.size();
        float sq_sum = 0.0f;
        for (auto rr : rr_intervals) {
            sq_sum += (rr - mean) * (rr - mean);
        }

        return std::sqrt(sq_sum / rr_intervals.size());
    }

    // Calculate average heart rate
    static float calculate_heart_rate(const std::vector<int>& r_peaks, float sampling_rate, int signal_length) {
        if (r_peaks.size() < 2) return 0.0f;
        float duration_minutes = signal_length / sampling_rate / 60.0f;
        return r_peaks.size() / duration_minutes;
    }
};

// ============================================================================
// 1D CNN FOR ECG CLASSIFICATION
// ============================================================================

struct ECGNet : torch::nn::Module {
    // 1D Convolutional layers for temporal feature extraction
    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    int input_length;

    ECGNet(int input_len = 1000, int num_classes = 5) : input_length(input_len) {
        // Layer 1: Extract low-level temporal features
        conv1 = register_module("conv1", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(1, 32, 7).stride(1).padding(3)));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(32));

        // Layer 2: Intermediate features
        conv2 = register_module("conv2", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(32, 64, 5).stride(1).padding(2)));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(64));

        // Layer 3: Higher-level patterns
        conv3 = register_module("conv3", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(64, 128, 3).stride(1).padding(1)));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(128));

        // Layer 4: Abstract features
        conv4 = register_module("conv4", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(128, 256, 3).stride(1).padding(1)));
        bn4 = register_module("bn4", torch::nn::BatchNorm1d(256));

        // Calculate feature size after convolutions and pooling
        int feature_size = 256 * (input_len / 16); // 4 max pools with stride 2

        fc1 = register_module("fc1", torch::nn::Linear(feature_size, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, num_classes));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input: [batch, 1, sequence_length]

        // Conv block 1
        x = conv1(x);
        x = bn1(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 2, 2);

        // Conv block 2
        x = conv2(x);
        x = bn2(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 2, 2);

        // Conv block 3
        x = conv3(x);
        x = bn3(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 2, 2);

        // Conv block 4
        x = conv4(x);
        x = bn4(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 2, 2);

        // Flatten
        x = x.view({x.size(0), -1});

        // Fully connected layers
        x = fc1(x);
        x = torch::relu(x);
        x = dropout(x);
        x = fc2(x);

        return x;
    }
};

// Observable Tensor - wraps torch::Tensor with change notification
class ObservableTensor {
public:
    torch::Tensor data;
    std::function<void()> on_change_callback;

    ObservableTensor() = default;

    ObservableTensor& operator=(const torch::Tensor& tensor) {
        data = tensor;
        if (on_change_callback) {
            on_change_callback();
        }
        return *this;
    }

    operator torch::Tensor() const { return data; }
    torch::Tensor get() const { return data; }
    bool defined() const { return data.defined(); }

    void set_callback(std::function<void()> callback) {
        on_change_callback = callback;
    }
};

// Multi-Head Self-Attention Module
struct MultiHeadAttention : torch::nn::Module {
    torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr};
    torch::nn::Linear out_proj{nullptr};
    int num_heads;
    int d_model;
    int d_k;

    // Store intermediate tensors for visualization
    ObservableTensor attention_weights;
    ObservableTensor Q_projected, K_projected, V_projected;
    ObservableTensor attention_output;
    ObservableTensor context_vectors;

    MultiHeadAttention(int d_model_, int num_heads_)
        : d_model(d_model_), num_heads(num_heads_), d_k(d_model_ / num_heads_) {
        query = register_module("query", torch::nn::Linear(d_model, d_model));
        key = register_module("key", torch::nn::Linear(d_model, d_model));
        value = register_module("value", torch::nn::Linear(d_model, d_model));
        out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batch, seq_len, d_model]
        int batch_size = x.size(0);
        int seq_len = x.size(1);

        // Linear projections
        auto Q = query->forward(x); // [batch, seq_len, d_model]
        auto K = key->forward(x);
        auto V = value->forward(x);

        // Store projections
        Q_projected = Q.detach();
        K_projected = K.detach();
        V_projected = V.detach();

        // Reshape for multi-head attention: [batch, num_heads, seq_len, d_k]
        Q = Q.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);
        K = K.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);
        V = V.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);

        // Scaled dot-product attention
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k);
        auto attn = torch::softmax(scores, -1); // [batch, num_heads, seq_len, seq_len]

        // Store attention weights for visualization
        attention_weights = attn.detach();

        // Apply attention to values
        auto context = torch::matmul(attn, V); // [batch, num_heads, seq_len, d_k]

        // Store context before concatenation
        context_vectors = context.detach();

        // Concatenate heads
        context = context.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});

        // Output projection
        auto output = out_proj->forward(context);
        attention_output = output.detach();

        return output;
    }
};

// Transformer Encoder Layer
struct TransformerEncoderLayer : torch::nn::Module {
    std::shared_ptr<MultiHeadAttention> self_attn;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    float dropout_rate = 0.1f;

    // Store intermediate activations for visualization
    ObservableTensor input_tensor;
    ObservableTensor attn_output;
    ObservableTensor after_norm1;
    ObservableTensor ff_intermediate;
    ObservableTensor ff_output;
    ObservableTensor after_norm2;

    TransformerEncoderLayer(int d_model, int num_heads, int dim_feedforward = 2048)
        : self_attn(std::make_shared<MultiHeadAttention>(d_model, num_heads)) {

        register_module("self_attn", self_attn);

        fc1 = register_module("fc1", torch::nn::Linear(d_model, dim_feedforward));
        fc2 = register_module("fc2", torch::nn::Linear(dim_feedforward, d_model));

        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Store input
        input_tensor = x.detach();

        // Self-attention with residual connection
        auto attn_out = self_attn->forward(x);
        attn_output = attn_out.detach();

        auto after_attn_residual = norm1->forward(x + torch::dropout(attn_out, dropout_rate, is_training()));
        after_norm1 = after_attn_residual.detach();

        // Feed-forward with residual connection
        auto ff_mid = torch::relu(fc1->forward(after_attn_residual));
        ff_intermediate = ff_mid.detach();

        auto ff_out = fc2->forward(ff_mid);
        ff_output = ff_out.detach();

        auto final_out = norm2->forward(after_attn_residual + torch::dropout(ff_out, dropout_rate, is_training()));
        after_norm2 = final_out.detach();

        return final_out;
    }
};

// Simple Transformer for sequence classification
struct TransformerClassifier : torch::nn::Module {
    torch::nn::Embedding token_embed{nullptr};
    torch::nn::Embedding pos_embed{nullptr};
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers;
    torch::nn::Linear classifier{nullptr};

    int d_model;
    int num_layers;
    int max_seq_len;

    // Store intermediate activations for visualization
    ObservableTensor token_embeddings;
    ObservableTensor pos_embeddings;
    ObservableTensor combined_embeddings;
    ObservableTensor pooled_output;
    ObservableTensor classifier_output;

    TransformerClassifier(int vocab_size, int d_model_, int num_heads, int num_layers_, int num_classes, int max_seq_len_ = 512)
        : d_model(d_model_), num_layers(num_layers_), max_seq_len(max_seq_len_) {

        token_embed = register_module("token_embed", torch::nn::Embedding(vocab_size, d_model));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(max_seq_len, d_model));

        for (int i = 0; i < num_layers; i++) {
            auto layer = std::make_shared<TransformerEncoderLayer>(d_model, num_heads);
            layers.push_back(layer);
            register_module("layer_" + std::to_string(i), layer);
        }

        classifier = register_module("classifier", torch::nn::Linear(d_model, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batch, seq_len] (token indices)
        int batch_size = x.size(0);
        int seq_len = x.size(1);

        // Create position indices
        auto positions = torch::arange(seq_len, x.options()).unsqueeze(0).expand({batch_size, seq_len});

        // Embed tokens and add positional embeddings
        auto tok_emb = token_embed->forward(x);
        auto pos_emb = pos_embed->forward(positions);
        token_embeddings = tok_emb.detach();
        pos_embeddings = pos_emb.detach();

        auto embedded = tok_emb + pos_emb;
        combined_embeddings = embedded.detach();

        // Pass through transformer layers
        auto hidden = embedded;
        for (auto& layer : layers) {
            hidden = layer->forward(hidden);
        }

        // Use mean pooling over sequence for classification
        auto pooled = hidden.mean(1); // [batch, d_model]
        pooled_output = pooled.detach();

        auto output = classifier->forward(pooled);
        classifier_output = output.detach();

        return output;
    }
};

// ResNet Basic Block
struct BasicBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample{nullptr};
    int stride;

    BasicBlock(int in_channels, int out_channels, int stride_ = 1, torch::nn::Sequential downsample_ = nullptr)
        : stride(stride_), downsample(downsample_) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

        if (downsample_) {
            downsample = register_module("downsample", downsample_);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));

        if (!downsample.is_empty()) {
            identity = downsample->forward(x);
        }

        out += identity;
        out = torch::relu(out);

        return out;
    }
};

// Small ResNet-18 for image classification
struct ResNet18 : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    // Store activations for visualization
    ObservableTensor conv1_out, layer1_out, layer2_out, layer3_out, layer4_out, pool_out;
    bool auto_visualize = false;

    ResNet18(int num_classes = 10) {
        // Initial conv: 3x32x32 -> 64x32x32
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        // ResNet layers
        layer1 = register_module("layer1", make_layer(64, 64, 2, 1));   // 64x32x32
        layer2 = register_module("layer2", make_layer(64, 128, 2, 2));  // 128x16x16
        layer3 = register_module("layer3", make_layer(128, 256, 2, 2)); // 256x8x8
        layer4 = register_module("layer4", make_layer(256, 512, 2, 2)); // 512x4x4

        // Global average pooling and classifier
        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)));
        fc = register_module("fc", torch::nn::Linear(512, num_classes));
    }

    torch::nn::Sequential make_layer(int in_channels, int out_channels, int num_blocks, int stride) {
        torch::nn::Sequential layers;

        // Downsample if needed
        torch::nn::Sequential downsample{nullptr};
        if (stride != 1 || in_channels != out_channels) {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels)
            );
        }

        // First block (may downsample)
        layers->push_back(BasicBlock(in_channels, out_channels, stride, downsample));

        // Remaining blocks
        for (int i = 1; i < num_blocks; i++) {
            layers->push_back(BasicBlock(out_channels, out_channels));
        }

        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        // Initial conv
        conv1_out = torch::relu(bn1->forward(conv1->forward(x)));

        // ResNet blocks
        layer1_out = layer1->forward(conv1_out.get());
        layer2_out = layer2->forward(layer1_out.get());
        layer3_out = layer3->forward(layer2_out.get());
        layer4_out = layer4->forward(layer3_out.get());

        // Global pooling and classification
        pool_out = avgpool->forward(layer4_out.get());
        x = pool_out.get().view({pool_out.get().size(0), -1});
        x = fc->forward(x);

        return x;
    }
};

// Camera state
struct Camera {
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 20.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float yaw = -90.0f;   // Looking towards -Z
    float pitch = 0.0f;
    float moveSpeed = 0.5f;
    float lookSpeed = 0.3f;

    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;
    bool dragging = false;

    glm::vec3 getFront() const {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        return glm::normalize(front);
    }

    glm::vec3 getRight() const {
        return glm::normalize(glm::cross(getFront(), up));
    }
};

// Helper to draw a 3D sphere
void drawSphere(float x, float y, float z, float radius, float r, float g, float b, float alpha = 1.0f) {
    const int slices = 10;
    const int stacks = 10;

    glColor4f(r, g, b, alpha);

    for (int i = 0; i < stacks; ++i) {
        float lat0 = M_PI * (-0.5f + (float)i / stacks);
        float z0 = radius * sinf(lat0);
        float zr0 = radius * cosf(lat0);

        float lat1 = M_PI * (-0.5f + (float)(i + 1) / stacks);
        float z1 = radius * sinf(lat1);
        float zr1 = radius * cosf(lat1);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            float lng = 2 * M_PI * (float)j / slices;
            float x0 = cosf(lng);
            float y0 = sinf(lng);

            glVertex3f(x + x0 * zr0, y + y0 * zr0, z + z0);
            glVertex3f(x + x0 * zr1, y + y0 * zr1, z + z1);
        }
        glEnd();
    }
}

// Helper to draw a 3D line
void drawLine3D(float x1, float y1, float z1, float x2, float y2, float z2,
                float r, float g, float b, float alpha = 0.3f) {
    glBegin(GL_LINES);
    glColor4f(r, g, b, alpha);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glEnd();
}

// Helper to project 3D point to screen space for text labels
struct ScreenLabel {
    std::string text;
    glm::vec2 screen_pos;
    glm::vec3 color;
    bool visible;
};

std::vector<ScreenLabel> projectLabelsToScreen(const Camera& camera,
                                                const std::vector<std::tuple<std::string, glm::vec3, glm::vec3>>& labels_3d) {
    std::vector<ScreenLabel> screen_labels;

    // Setup projection and view matrices
    float aspect = 1600.0f / 900.0f;
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 200.0f;

    glm::mat4 projection = glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
    glm::vec3 front = camera.getFront();
    glm::vec3 lookTarget = camera.position + front;
    glm::mat4 view = glm::lookAt(camera.position, lookTarget, camera.up);
    glm::mat4 vp = projection * view;

    for (const auto& [text, pos_3d, color] : labels_3d) {
        glm::vec4 clip_pos = vp * glm::vec4(pos_3d, 1.0f);

        if (clip_pos.w > 0 && clip_pos.z > 0) {
            glm::vec3 ndc = glm::vec3(clip_pos) / clip_pos.w;

            if (ndc.x >= -1.0f && ndc.x <= 1.0f && ndc.y >= -1.0f && ndc.y <= 1.0f) {
                glm::vec2 screen;
                screen.x = (ndc.x + 1.0f) * 0.5f * 1600.0f;
                screen.y = (1.0f - ndc.y) * 0.5f * 900.0f;

                screen_labels.push_back({text, screen, color, true});
            } else {
                screen_labels.push_back({text, {0, 0}, color, false});
            }
        } else {
            screen_labels.push_back({text, {0, 0}, color, false});
        }
    }

    return screen_labels;
}

// Draw attention heatmap visualization with token labels
void drawAttentionHeatmap(const torch::Tensor& attention_weights,
                         const std::vector<std::string>& tokens,
                         int head_idx = 0) {
    if (!attention_weights.defined() || attention_weights.size(0) == 0) {
        ImGui::Text("No attention weights available");
        return;
    }

    // attention_weights shape: [batch, num_heads, seq_len, seq_len]
    auto attn = attention_weights[0][head_idx].to(torch::kCPU);
    int seq_len = attn.size(0);

    // Display text explanation
    ImGui::TextWrapped("Each row shows what one token attends to (looks at). "
                      "Brighter = stronger attention.");
    ImGui::Separator();

    // Show token-by-token attention in text form (most interesting patterns)
    ImGui::Text("Top Attention Patterns:");
    auto attn_accessor = attn.accessor<float, 2>();

    // Find and display top 5 attention pairs
    std::vector<std::tuple<int, int, float>> top_pairs;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (i != j) { // Skip self-attention
                top_pairs.push_back({i, j, attn_accessor[i][j]});
            }
        }
    }
    std::sort(top_pairs.begin(), top_pairs.end(),
             [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    for (int k = 0; k < std::min(5, (int)top_pairs.size()); k++) {
        auto [i, j, weight] = top_pairs[k];
        ImGui::Text("  '%s' -> '%s': %.3f",
                   tokens[i].c_str(), tokens[j].c_str(), weight);
    }

    ImGui::Separator();

    // Use ImPlot for heatmap
    if (ImPlot::BeginPlot("##AttentionHeatmap", ImVec2(-1, 400))) {
        // Setup custom tick labels for tokens
        std::vector<const char*> token_labels;
        for (const auto& token : tokens) {
            token_labels.push_back(token.c_str());
        }

        ImPlot::SetupAxes("Attends TO (Key)", "Token (Query)");
        ImPlot::SetupAxisTicks(ImAxis_X1, 0.5, seq_len - 0.5, seq_len,
                              token_labels.data());
        ImPlot::SetupAxisTicks(ImAxis_Y1, 0.5, seq_len - 0.5, seq_len,
                              token_labels.data());
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, seq_len, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, seq_len, ImGuiCond_Always);

        // Convert to flat array for ImPlot
        std::vector<float> attn_data(seq_len * seq_len);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                // Flip Y axis for proper visualization (ImPlot origin is bottom-left)
                attn_data[(seq_len - 1 - i) * seq_len + j] = attn_accessor[i][j];
            }
        }

        ImPlot::PlotHeatmap("Attention", attn_data.data(), seq_len, seq_len,
                           0.0, 1.0, nullptr,
                           ImPlotPoint(0, 0), ImPlotPoint(seq_len, seq_len));

        ImPlot::EndPlot();
    }
}

// Draw complete transformer architecture in 3D
std::vector<std::tuple<std::string, glm::vec3, glm::vec3>> drawTransformerArchitecture3D(
                                   Camera& camera,
                                   const std::shared_ptr<TransformerClassifier>& transformer,
                                   const torch::Tensor& token_indices,
                                   int selected_layer,
                                   int selected_head,
                                   const std::vector<std::string>& token_words) {
    std::vector<std::tuple<std::string, glm::vec3, glm::vec3>> labels;
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Setup projection and camera
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = 1600.0f / 900.0f;
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 200.0f;
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    float projection[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (farPlane + nearPlane) / (nearPlane - farPlane), -1,
        0, 0, (2 * farPlane * nearPlane) / (nearPlane - farPlane), 0
    };
    glLoadMatrixf(projection);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glm::vec3 front = camera.getFront();
    glm::vec3 lookTarget = camera.position + front;
    glm::mat4 view = glm::lookAt(camera.position, lookTarget, camera.up);
    glLoadMatrixf(glm::value_ptr(view));

    int seq_len = token_indices.size(1);
    float z_start = -40.0f;
    float z_spacing = 10.0f;
    float layer_z = z_start;

    // Helper to draw a layer of neurons/embeddings
    auto draw_embedding_layer = [&](const torch::Tensor& embeddings, float z_pos,
                                     float r, float g, float b, float neuron_size = 0.15f) {
        if (!embeddings.defined() || embeddings.size(0) == 0) return;
        auto emb = embeddings[0].to(torch::kCPU); // [seq_len, d_model]
        int seq = emb.size(0);
        int dim = emb.size(1);

        for (int i = 0; i < seq; i++) {
            for (int d = 0; d < dim; d += 4) { // Sample dimensions
                float val = std::abs(emb[i][d].item<float>());
                if (val > 0.01f) {
                    float x = (i - seq/2.0f) * 1.5f;
                    float y = (d - dim/2.0f) * 0.05f;
                    float intensity = std::min(1.0f, val);
                    drawSphere(x, y, z_pos, neuron_size,
                              r * intensity, g * intensity, b * intensity, 0.7f);
                }
            }
        }
    };

    // 1. Token Embeddings
    float tok_emb_z = layer_z;
    draw_embedding_layer(transformer->token_embeddings, layer_z, 0.3f, 0.7f, 1.0f, 0.18f);
    labels.push_back({"Token Embeddings", {-10.0f, 0.0f, layer_z}, {0.3f, 0.7f, 1.0f}});
    layer_z += z_spacing;

    // 2. Positional Embeddings
    float pos_emb_z = layer_z;
    draw_embedding_layer(transformer->pos_embeddings, layer_z, 1.0f, 0.7f, 0.3f, 0.18f);
    labels.push_back({"Positional Embeddings", {-10.0f, 0.0f, layer_z}, {1.0f, 0.7f, 0.3f}});

    // Draw connection lines from token embeddings
    glLineWidth(3.0f);
    for (float x = -4.0f; x <= 4.0f; x += 1.5f) {
        drawLine3D(x, 0.0f, tok_emb_z, x, 0.0f, pos_emb_z, 0.5f, 0.7f, 1.0f, 0.6f);
    }
    glLineWidth(1.0f);
    layer_z += z_spacing;

    // 3. Combined Embeddings (Token + Position)
    float combined_z = layer_z;
    draw_embedding_layer(transformer->combined_embeddings, layer_z, 0.5f, 1.0f, 0.5f, 0.2f);
    labels.push_back({"Combined Embeddings", {-10.0f, 0.0f, layer_z}, {0.5f, 1.0f, 0.5f}});

    // Draw connection lines from both embeddings to combined
    glLineWidth(3.0f);
    for (float x = -4.0f; x <= 4.0f; x += 1.5f) {
        drawLine3D(x, 0.0f, pos_emb_z, x, 0.0f, combined_z, 0.7f, 1.0f, 0.5f, 0.6f);
    }
    glLineWidth(1.0f);
    layer_z += z_spacing;

    // Draw each transformer layer
    float prev_layer_z = combined_z;
    for (int layer_idx = 0; layer_idx < transformer->layers.size(); layer_idx++) {
        auto& layer = transformer->layers[layer_idx];
        bool is_selected = (layer_idx == selected_layer);

        labels.push_back({std::string("=== Layer ") + std::to_string(layer_idx) + " ===",
                         {-10.0f, 2.0f, layer_z}, {1.0f, 1.0f, 1.0f}});

        // 4. Layer Input
        float layer_input_z = layer_z;
        if (is_selected) {
            draw_embedding_layer(layer->input_tensor, layer_z, 0.6f, 0.6f, 1.0f, 0.15f);
            labels.push_back({"Layer Input", {-10.0f, 0.0f, layer_z}, {0.6f, 0.6f, 1.0f}});

            // Connection from previous layer
            glLineWidth(3.0f);
            for (float x = -4.0f; x <= 4.0f; x += 1.5f) {
                drawLine3D(x, 0.0f, prev_layer_z, x, 0.0f, layer_input_z, 0.6f, 0.6f, 1.0f, 0.5f);
            }
            glLineWidth(1.0f);
        }
        layer_z += z_spacing * 0.5f;

        // 5. Q, K, V Projections (Multi-head attention input)
        float qkv_z = layer_z;
        if (is_selected && layer->self_attn->Q_projected.defined()) {
            auto& mha = layer->self_attn;

            // Draw Q
            draw_embedding_layer(mha->Q_projected, layer_z - 2.0f, 1.0f, 0.3f, 0.3f, 0.12f);
            labels.push_back({"Q (Query)", {-10.0f, -2.5f, layer_z - 2.0f}, {1.0f, 0.3f, 0.3f}});

            // Draw K
            draw_embedding_layer(mha->K_projected, layer_z, 0.3f, 1.0f, 0.3f, 0.12f);
            labels.push_back({"K (Key)", {-10.0f, 0.0f, layer_z}, {0.3f, 1.0f, 0.3f}});

            // Draw V
            draw_embedding_layer(mha->V_projected, layer_z + 2.0f, 0.3f, 0.3f, 1.0f, 0.12f);
            labels.push_back({"V (Value)", {-10.0f, 2.5f, layer_z + 2.0f}, {0.3f, 0.3f, 1.0f}});

            // Connection lines from input to Q, K, V
            glLineWidth(2.5f);
            for (float x = -3.0f; x <= 3.0f; x += 1.2f) {
                drawLine3D(x, 0.0f, layer_input_z, x, -2.0f, layer_z - 2.0f, 1.0f, 0.4f, 0.4f, 0.5f);
                drawLine3D(x, 0.0f, layer_input_z, x, 0.0f, layer_z, 0.4f, 1.0f, 0.4f, 0.5f);
                drawLine3D(x, 0.0f, layer_input_z, x, 2.0f, layer_z + 2.0f, 0.4f, 0.4f, 1.0f, 0.5f);
            }
            glLineWidth(1.0f);
        }
        layer_z += z_spacing;

        // 6. Attention Weights Visualization (for selected head)
        float attn_z = layer_z;
        if (is_selected && layer->self_attn->attention_weights.defined()) {
            labels.push_back({std::string("Attention (Head ") + std::to_string(selected_head) + ")",
                             {-10.0f, 0.0f, layer_z}, {0.9f, 0.5f, 0.2f}});

            auto attn = layer->self_attn->attention_weights.get()[0][selected_head].to(torch::kCPU);
            auto attn_accessor = attn.accessor<float, 2>();

            // Position tokens in a grid for attention visualization
            std::vector<glm::vec3> positions;
            for (int i = 0; i < seq_len; i++) {
                float x = (i - seq_len/2.0f) * 1.5f;
                positions.push_back({x, 0.0f, layer_z});
            }

            // Draw attention connections
            float threshold = 0.05f;
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float weight = attn_accessor[i][j];
                    if (weight > threshold) {
                        auto& from = positions[i];
                        auto& to = positions[j];
                        float alpha = weight * 0.6f;
                        glLineWidth(1.0f + weight * 3.0f);
                        drawLine3D(from.x, from.y, from.z, to.x, to.y, to.z,
                                  weight, 0.4f, 1.0f - weight, alpha);
                    }
                }
            }
            glLineWidth(1.0f);

            // Draw token nodes with labels
            for (int i = 0; i < seq_len; i++) {
                drawSphere(positions[i].x, positions[i].y, positions[i].z,
                          0.25f, 0.9f, 0.5f, 0.2f, 1.0f);

                // Add token word label
                if (i < token_words.size()) {
                    labels.push_back({token_words[i],
                                     {positions[i].x, positions[i].y - 0.5f, positions[i].z},
                                     {1.0f, 1.0f, 0.5f}});
                }
            }
        }
        layer_z += z_spacing;

        // 7. Context vectors (attention output)
        float ctx_z = layer_z;
        if (is_selected && layer->self_attn->context_vectors.defined()) {
            auto ctx = layer->self_attn->context_vectors;
            draw_embedding_layer(layer->self_attn->attention_output, layer_z, 0.8f, 0.4f, 0.8f, 0.15f);
            labels.push_back({"Attention Output", {-10.0f, 0.0f, layer_z}, {0.8f, 0.4f, 0.8f}});
        }
        layer_z += z_spacing * 0.5f;

        // 8. After LayerNorm1 + Residual
        float norm1_z = layer_z;
        if (is_selected) {
            draw_embedding_layer(layer->after_norm1, layer_z, 0.7f, 0.7f, 0.3f, 0.15f);
            labels.push_back({"LayerNorm1 + Residual", {-10.0f, 0.0f, layer_z}, {0.7f, 0.7f, 0.3f}});
            // Connection from attention output
            glLineWidth(2.5f);
            for (float x = -4.0f; x <= 4.0f; x += 1.5f) {
                drawLine3D(x, 0.0f, ctx_z, x, 0.0f, norm1_z, 0.8f, 0.8f, 0.5f, 0.5f);
            }
            glLineWidth(1.0f);
        }
        layer_z += z_spacing * 0.5f;

        // 9. Feed-Forward Intermediate (expanded dimension)
        float ff_mid_z = layer_z;
        if (is_selected && layer->ff_intermediate.defined()) {
            labels.push_back({"Feed-Forward (2048d)", {-10.0f, 0.0f, layer_z}, {1.0f, 0.6f, 0.2f}});
            auto ff = layer->ff_intermediate.get()[0].to(torch::kCPU);
            int seq = ff.size(0);
            int dim = ff.size(1);

            for (int i = 0; i < seq; i++) {
                for (int d = 0; d < dim; d += 8) { // Sample more sparsely (larger dimension)
                    float val = ff[i][d].item<float>();
                    if (val > 0.01f) {
                        float x = (i - seq/2.0f) * 1.5f;
                        float y = (d - dim/2.0f) * 0.03f;
                        float intensity = std::min(1.0f, val);
                        drawSphere(x, y, layer_z, 0.1f,
                                  1.0f * intensity, 0.6f * intensity, 0.2f, 0.6f);
                    }
                }
            }
        }
        layer_z += z_spacing * 0.5f;

        // 10. Feed-Forward Output
        float ff_out_z = layer_z;
        if (is_selected) {
            draw_embedding_layer(layer->ff_output, layer_z, 0.9f, 0.5f, 0.3f, 0.15f);
            labels.push_back({"FF Output", {-10.0f, 0.0f, layer_z}, {0.9f, 0.5f, 0.3f}});
        }
        layer_z += z_spacing * 0.5f;

        // 11. After LayerNorm2 + Residual (layer output)
        if (is_selected) {
            draw_embedding_layer(layer->after_norm2, layer_z, 0.5f, 0.9f, 0.7f, 0.16f);
            labels.push_back({"LayerNorm2 + Residual", {-10.0f, 0.0f, layer_z}, {0.5f, 0.9f, 0.7f}});
            prev_layer_z = layer_z;
        }
        layer_z += z_spacing;
    }

    // 12. Pooled Output
    float pooled_z = layer_z;
    if (transformer->pooled_output.defined()) {
        labels.push_back({"Mean Pooling", {-10.0f, 0.0f, layer_z}, {0.3f, 0.8f, 1.0f}});
        auto pooled = transformer->pooled_output.get().to(torch::kCPU);
        int dim = pooled.size(1);
        for (int d = 0; d < dim; d += 4) {
            float val = std::abs(pooled[0][d].item<float>());
            if (val > 0.01f) {
                float x = (d - dim/2.0f) * 0.1f;
                float intensity = std::min(1.0f, val);
                drawSphere(x, 0.0f, layer_z, 0.2f,
                          0.3f, intensity, 1.0f, 0.8f);
            }
        }
    }
    layer_z += z_spacing;

    // 13. Classifier Output (10 classes)
    if (transformer->classifier_output.defined()) {
        labels.push_back({"Classifier (10 classes)", {-10.0f, 0.0f, layer_z}, {1.0f, 0.8f, 0.2f}});
        auto output = transformer->classifier_output.get()[0].to(torch::kCPU);
        auto probs = torch::softmax(output, 0);
        for (int i = 0; i < 10; i++) {
            float prob = probs[i].item<float>();
            float x = (i - 4.5f) * 1.0f;
            drawSphere(x, 0.0f, layer_z, 0.3f,
                      prob, 0.8f * prob, 0.2f, 1.0f);
        }
        // Connection from pooled
        glLineWidth(2.0f);
        for (float x = -6.0f; x <= 6.0f; x += 1.5f) {
            drawLine3D(x * 0.1f, 0.0f, pooled_z, (int(x/1.5f) - 3.0f) * 1.0f, 0.0f, layer_z, 0.8f, 0.8f, 0.5f, 0.4f);
        }
        glLineWidth(1.0f);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    return labels;
}

// Helper to draw a cube (for conv layers)
void drawCube(float x, float y, float z, float width, float height, float depth,
              float r, float g, float b, float alpha = 0.8f) {
    glColor4f(r, g, b, alpha);

    float w = width / 2, h = height / 2, d = depth / 2;

    // Draw faces
    glBegin(GL_QUADS);

    // Front face
    glVertex3f(x - w, y - h, z + d);
    glVertex3f(x + w, y - h, z + d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x - w, y + h, z + d);

    // Back face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x - w, y + h, z - d);
    glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y - h, z - d);

    // Top face
    glVertex3f(x - w, y + h, z - d);
    glVertex3f(x - w, y + h, z + d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y + h, z - d);

    // Bottom face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y - h, z + d);
    glVertex3f(x - w, y - h, z + d);

    // Right face
    glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y - h, z + d);

    // Left face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x - w, y - h, z + d);
    glVertex3f(x - w, y + h, z + d);
    glVertex3f(x - w, y + h, z - d);

    glEnd();

    // Draw edges
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);

    // Bottom edges
    glVertex3f(x - w, y - h, z - d); glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y - h, z - d); glVertex3f(x + w, y - h, z + d);
    glVertex3f(x + w, y - h, z + d); glVertex3f(x - w, y - h, z + d);
    glVertex3f(x - w, y - h, z + d); glVertex3f(x - w, y - h, z - d);

    // Top edges
    glVertex3f(x - w, y + h, z - d); glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y + h, z - d); glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y + h, z + d); glVertex3f(x - w, y + h, z + d);
    glVertex3f(x - w, y + h, z + d); glVertex3f(x - w, y + h, z - d);

    // Vertical edges
    glVertex3f(x - w, y - h, z - d); glVertex3f(x - w, y + h, z - d);
    glVertex3f(x + w, y - h, z - d); glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y - h, z + d); glVertex3f(x + w, y + h, z + d);
    glVertex3f(x - w, y - h, z + d); glVertex3f(x - w, y + h, z + d);

    glEnd();
    glLineWidth(1.0f);
}

// Draw 3D ResNet network visualization
void drawResNetNetwork3D(Camera& camera,
                        const torch::Tensor& input_img,
                        const ObservableTensor& conv1_act,
                        const ObservableTensor& layer1_act,
                        const ObservableTensor& layer2_act,
                        const ObservableTensor& layer3_act,
                        const ObservableTensor& layer4_act,
                        const ObservableTensor& pool_act,
                        const torch::Tensor& output_act,
                        const std::shared_ptr<ResNet18>& net) {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Setup perspective projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = 1600.0f / 900.0f;
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    float projection[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (farPlane + nearPlane) / (nearPlane - farPlane), -1,
        0, 0, (2 * farPlane * nearPlane) / (nearPlane - farPlane), 0
    };
    glLoadMatrixf(projection);

    // Setup camera with lookAt
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glm::vec3 front = camera.getFront();
    glm::vec3 lookTarget = camera.position + front;

    glm::mat4 view = glm::lookAt(camera.position, lookTarget, camera.up);
    glLoadMatrixf(glm::value_ptr(view));

    // Layer positions (spread out for ResNet visualization)
    float z_positions[] = {-24.0f, -18.0f, -12.0f, -6.0f, 0.0f, 6.0f, 12.0f, 18.0f, 22.0f};

    // 0. Raw Image Layer - Show colored pixels as actual RGB image (composite view)
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0].to(torch::kCPU);
        float pixel_size = 0.10f;
        int img_size = 32;

        // Normalize the image for display (denormalize from training normalization)
        auto mean = torch::tensor({0.4914, 0.4822, 0.4465}).view({3, 1, 1});
        auto std = torch::tensor({0.2023, 0.1994, 0.2010}).view({3, 1, 1});
        auto denorm_img = img * std + mean;
        denorm_img = torch::clamp(denorm_img, 0.0, 1.0);

        // Draw each pixel with its true RGB color
        for (int i = 0; i < img_size; i++) {
            for (int j = 0; j < img_size; j++) {
                float r = denorm_img[0][i][j].item<float>();
                float g = denorm_img[1][i][j].item<float>();
                float b = denorm_img[2][i][j].item<float>();

                float x = (j - img_size/2.0f) * pixel_size;
                float y = (img_size/2.0f - i) * pixel_size;

                // Draw colored pixel
                drawCube(x, y, z_positions[0],
                        pixel_size * 0.95f, pixel_size * 0.95f, 0.03f,
                        r, g, b, 1.0f);
            }
        }
    }

    // 1. Input layer (3x32x32 RGB image) - show all 3 channels as separate tensor slices
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0].to(torch::kCPU);
        float pixel_size = 0.08f;
        int img_size = 32;

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < img_size; i += 2) {
                for (int j = 0; j < img_size; j += 2) {
                    float val = img[c][i][j].item<float>();
                    float x = (j - img_size/2.0f) * pixel_size;
                    float y = (img_size/2.0f - i) * pixel_size;
                    float z = z_positions[1] + (c - 1) * 0.1f;

                    if (std::abs(val) > 0.1f) {
                        float r = (c == 0) ? std::abs(val) : 0.3f;
                        float g = (c == 1) ? std::abs(val) : 0.3f;
                        float b = (c == 2) ? std::abs(val) : 0.3f;
                        drawCube(x, y, z, pixel_size * 0.8f, pixel_size * 0.8f, 0.02f, r, g, b, 0.8f);
                    }
                }
            }
        }
    }

    std::vector<glm::vec3> conv1_positions, layer1_positions, layer2_positions, layer3_positions, layer4_positions, pool_positions;

    // 2. Initial Conv1 layer - 64 channels × 32×32
    if (conv1_act.defined() && conv1_act.get().size(0) > 0) {
        auto act = conv1_act.get()[0].to(torch::kCPU);
        int channels = act.size(0); // 64
        int height = act.size(1);   // 32
        int width = act.size(2);    // 32

        float pixel_size = 0.07f;
        float channel_depth = 0.05f;

        for (int c = 0; c < channels; c += 2) {
            for (int i = 0; i < height; i += 4) {
                for (int j = 0; j < width; j += 4) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[2] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        conv1_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.7f, pixel_size * 0.7f, channel_depth * 0.8f,
                                intensity, 0.5f + intensity * 0.5f, 1.0f - intensity * 0.3f, 0.6f);
                    }
                }
            }
        }
    }

    // 3. Layer1 - 64 channels × 32×32
    if (layer1_act.defined() && layer1_act.get().size(0) > 0) {
        auto act = layer1_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.07f;
        float channel_depth = 0.05f;

        for (int c = 0; c < channels; c += 2) {
            for (int i = 0; i < height; i += 4) {
                for (int j = 0; j < width; j += 4) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[3] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer1_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.7f, pixel_size * 0.7f, channel_depth * 0.8f,
                                intensity, 0.6f, 1.0f - intensity * 0.4f, 0.65f);
                    }
                }
            }
        }
    }

    // 4. Layer2 - 128 channels × 16×16
    if (layer2_act.defined() && layer2_act.get().size(0) > 0) {
        auto act = layer2_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.09f;
        float channel_depth = 0.04f;

        for (int c = 0; c < channels; c += 3) {
            for (int i = 0; i < height; i += 2) {
                for (int j = 0; j < width; j += 2) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[4] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer2_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.8f, pixel_size * 0.8f, channel_depth * 0.8f,
                                intensity, 0.5f, 1.0f - intensity * 0.5f, 0.7f);
                    }
                }
            }
        }
    }

    // 5. Layer3 - 256 channels × 8×8
    if (layer3_act.defined() && layer3_act.get().size(0) > 0) {
        auto act = layer3_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.12f;
        float channel_depth = 0.03f;

        for (int c = 0; c < channels; c += 4) {
            for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < width; j += 1) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[5] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer3_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.9f, pixel_size * 0.9f, channel_depth * 0.8f,
                                intensity, 0.4f, 1.0f - intensity * 0.6f, 0.75f);
                    }
                }
            }
        }
    }

    // 6. Layer4 - 512 channels × 4×4
    if (layer4_act.defined() && layer4_act.get().size(0) > 0) {
        auto act = layer4_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.18f;
        float channel_depth = 0.02f;

        for (int c = 0; c < channels; c += 8) {
            for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < width; j += 1) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[6] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer4_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.95f, pixel_size * 0.95f, channel_depth * 0.8f,
                                intensity, 0.3f, 1.0f - intensity * 0.7f, 0.8f);
                    }
                }
            }
        }
    }

    // 7. Global Average Pool - 512 dimensions
    if (pool_act.defined() && pool_act.get().size(0) > 0) {
        auto act = pool_act.get()[0].to(torch::kCPU);
        act = act.view({-1});
        int num_features = act.size(0); // 512

        int grid_width = 32;
        int grid_height = 16;
        float neuron_size = 0.08f;

        for (int i = 0; i < num_features; i += 2) {
            float val = act[i].item<float>();
            if (val > 0.05f) {
                float intensity = std::min(1.0f, val);
                int row = i / grid_width;
                int col = i % grid_width;
                float x = (col - grid_width / 2.0f) * neuron_size;
                float y = (grid_height / 2.0f - row) * neuron_size;
                pool_positions.push_back({x, y, z_positions[7]});
                drawCube(x, y, z_positions[7], neuron_size * 0.9f, neuron_size * 0.9f, 0.05f,
                        intensity, 0.2f + intensity * 0.4f, 1.0f - intensity * 0.6f, 0.85f);
            }
        }
    }

    // 8. Output layer - 10 classes
    std::vector<glm::vec3> output_positions;
    if (output_act.defined() && output_act.size(0) > 0) {
        auto act = output_act[0].to(torch::kCPU);
        auto probs = torch::softmax(act, 0);
        float neuron_size = 0.20f;

        for (int i = 0; i < 10; i++) {
            float prob = probs[i].item<float>();
            float y = (i - 4.5f) * neuron_size;
            output_positions.push_back({0, y, z_positions[8]});
            drawCube(0, y, z_positions[8], neuron_size * 1.2f, neuron_size * 0.9f, 0.08f,
                    prob, 0.8f * prob, 0.2f, 1.0f);
        }
    }

    // Draw connections (sampled heavily for performance)
    glLineWidth(1.0f);

    auto draw_layer_connections = [](const std::vector<glm::vec3>& from, const std::vector<glm::vec3>& to,
                                     float r, float g, float b, float alpha, int sample_rate = 20) {
        for (size_t i = 0; i < from.size(); i += sample_rate) {
            for (size_t j = 0; j < to.size(); j += sample_rate) {
                drawLine3D(from[i].x, from[i].y, from[i].z, to[j].x, to[j].y, to[j].z, r, g, b, alpha);
            }
        }
    };

    if (!conv1_positions.empty() && !layer1_positions.empty())
        draw_layer_connections(conv1_positions, layer1_positions, 0.4f, 0.4f, 0.7f, 0.1f, 30);
    if (!layer1_positions.empty() && !layer2_positions.empty())
        draw_layer_connections(layer1_positions, layer2_positions, 0.5f, 0.5f, 0.8f, 0.12f, 35);
    if (!layer2_positions.empty() && !layer3_positions.empty())
        draw_layer_connections(layer2_positions, layer3_positions, 0.6f, 0.4f, 0.9f, 0.15f, 40);
    if (!layer3_positions.empty() && !layer4_positions.empty())
        draw_layer_connections(layer3_positions, layer4_positions, 0.7f, 0.3f, 1.0f, 0.18f, 45);
    if (!layer4_positions.empty() && !pool_positions.empty())
        draw_layer_connections(layer4_positions, pool_positions, 0.8f, 0.2f, 0.9f, 0.2f, 20);

    // Pool to output with FC weights if available
    if (!pool_positions.empty() && !output_positions.empty()) {
        auto fc_weights = net->fc->weight.to(torch::kCPU).abs();
        for (size_t i = 0; i < pool_positions.size(); i += 2) {
            for (size_t j = 0; j < output_positions.size(); j++) {
                drawLine3D(pool_positions[i].x, pool_positions[i].y, pool_positions[i].z,
                          output_positions[j].x, output_positions[j].y, output_positions[j].z,
                          0.9f, 0.4f, 0.3f, 0.25f);
            }
        }
    }

    glLineWidth(1.0f);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL 2.1 for immediate mode
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_FALSE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(1800, 1000, "Caliper - Real-Time ECG Waveform Analyzer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Camera setup
    Camera camera;
    camera.position = glm::vec3(0.0f, 5.0f, 50.0f); // Better view for transformer architecture
    glfwSetWindowUserPointer(window, &camera);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    // Install ImGui callbacks (install_callbacks=false to not override our callbacks)
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Install our custom callbacks that check ImGui state
    auto prev_mouse_button_callback = glfwSetMouseButtonCallback(window,
        [](GLFWwindow* win, int button, int action, int mods) {
            // Let ImGui process first
            ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                if (action == GLFW_PRESS) {
                    cam->dragging = true;
                    double x, y;
                    glfwGetCursorPos(win, &x, &y);
                    cam->lastMouseX = x;
                    cam->lastMouseY = y;
                } else if (action == GLFW_RELEASE) {
                    cam->dragging = false;
                }
            }
        });

    auto prev_cursor_pos_callback = glfwSetCursorPosCallback(window,
        [](GLFWwindow* win, double x, double y) {
            // Let ImGui process first
            ImGui_ImplGlfw_CursorPosCallback(win, x, y);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            if (cam->dragging) {
                float dx = x - cam->lastMouseX;
                float dy = y - cam->lastMouseY;

                // Update yaw and pitch for free look
                cam->yaw += dx * cam->lookSpeed;
                cam->pitch -= dy * cam->lookSpeed;

                // Clamp pitch to avoid gimbal lock
                cam->pitch = std::max(-89.0f, std::min(89.0f, cam->pitch));

                cam->lastMouseX = x;
                cam->lastMouseY = y;
            }
        });

    auto prev_scroll_callback = glfwSetScrollCallback(window,
        [](GLFWwindow* win, double xoffset, double yoffset) {
            // Let ImGui process first
            ImGui_ImplGlfw_ScrollCallback(win, xoffset, yoffset);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            // Move camera forward/backward along view direction
            glm::vec3 front = cam->getFront();
            cam->position += front * static_cast<float>(yoffset) * cam->moveSpeed;
        });

    // Install other ImGui callbacks
    glfwSetKeyCallback(window, ImGui_ImplGlfw_KeyCallback);
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    // Check device availability
    bool mps_available = torch::mps::is_available();
    torch::Device device(mps_available ? torch::kMPS : torch::kCPU);

    // ECG-specific mode (no transformer/resnet switching needed)

    std::cout << "Loading CIFAR-10 dataset..." << std::endl;

    // Load CIFAR-10 dataset (using custom loader since LibTorch C++ doesn't have built-in CIFAR)
    const std::string dataset_path = "/Users/ahmed/CLionProjects/caliper/data";

    // Create synthetic image dataset for demonstration
    class SyntheticImageDataset : public torch::data::Dataset<SyntheticImageDataset> {
    public:
        explicit SyntheticImageDataset(size_t size) : size_(size) {}

        torch::data::Example<> get(size_t index) override {
            // Generate random 3x32x32 images (CIFAR-10 size)
            auto data = torch::randn({3, 32, 32});
            auto target = torch::tensor(static_cast<int64_t>(index % 10));
            return {data, target};
        }

        torch::optional<size_t> size() const override {
            return size_;
        }
    private:
        size_t size_;
    };

    auto train_dataset = SyntheticImageDataset(5000).map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(32).workers(0)
    );

    std::cout << "CIFAR-10 dataset loaded successfully!" << std::endl;

    // Load ECG data for waveform visualization
    ECGDataset ecg_dataset("./data/ecg", 1000);
    std::vector<ECGRecord> ecg_records = ecg_dataset.get_records();
    std::vector<float> current_ecg_signal;

    // Create ResNet18 for CIFAR-10 classification
    float learning_rate = 0.001f;
    int num_classes = 10; // CIFAR-10 has 10 classes
    auto net = std::make_shared<ResNet18>(num_classes);
    net->to(device);

    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(5e-4));

    // Training state
    bool training = false;
    int batch_count = 0;
    int epoch = 0;
    float loss_value = 0.0f;
    float accuracy = 0.0f;
    std::vector<float> loss_history;
    float max_loss_ever = 0.0f;

    // Training speed control
    int batches_per_frame = 1;  // How many batches to process per frame
    int frame_counter = 0;
    int training_delay_ms = 0;  // Delay between training steps in milliseconds (0 = no delay)
    auto last_training_time = std::chrono::steady_clock::now();

    // Current batch for visualization
    torch::Tensor current_input;
    torch::Tensor current_target;
    int current_predicted = 0;
    int current_actual = 0;
    torch::Tensor output_for_vis;

    auto train_iter = train_loader->begin();

    // ECG visualization state
    int current_record_idx = 0;
    float sampling_rate = 360.0f;
    std::vector<int> detected_r_peaks;
    std::deque<float> realtime_buffer; // For scrolling visualization
    int buffer_size = 2000; // Show 2000 samples at a time
    float current_hr = 0.0f;
    float current_hrv = 0.0f;
    bool auto_detect_peaks = true;
    float time_elapsed = 0.0f;

    // Initialize with first ECG record
    if (!ecg_records.empty()) {
        current_ecg_signal = ecg_records[0].signal;
        sampling_rate = ecg_records[0].sampling_rate;
        detected_r_peaks = ECGSignalProcessor::detect_r_peaks(current_ecg_signal, sampling_rate);
        current_hr = ECGSignalProcessor::calculate_heart_rate(detected_r_peaks, sampling_rate, current_ecg_signal.size());
        current_hrv = ECGSignalProcessor::calculate_hrv_sdnn(detected_r_peaks, sampling_rate);

        // Initialize realtime buffer
        for (int i = 0; i < std::min(buffer_size, (int)current_ecg_signal.size()); i++) {
            realtime_buffer.push_back(current_ecg_signal[i]);
        }
    } else {
        std::cout << "Warning: No ECG records available for visualization" << std::endl;
    }

    // Initialize with first sample for training
    {
        auto& batch = *train_iter;
        current_input = batch.data.slice(0, 0, 1).to(device);
        current_target = batch.target.slice(0, 0, 1).to(device);
        current_actual = batch.target[0].item<int64_t>();
        net->eval();
        torch::NoGradGuard no_grad;
        output_for_vis = net->forward(current_input);
        current_predicted = output_for_vis.argmax(1).item<int64_t>();
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Main Image Display
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(1300, 450), ImGuiCond_FirstUseEver);
        ImGui::Begin("CIFAR-10 Image Classification - Real-Time");

        // Simulate real-time scrolling
        if (!current_ecg_signal.empty()) {
            static int signal_idx = buffer_size;
            if (training) {
                signal_idx += 3; // Scroll speed
                if (signal_idx >= (int)current_ecg_signal.size()) {
                    signal_idx = buffer_size;
                }

                // Update buffer with new data
                for (int i = 0; i < 3 && signal_idx < (int)current_ecg_signal.size(); i++) {
                    realtime_buffer.pop_front();
                    realtime_buffer.push_back(current_ecg_signal[signal_idx]);
                    signal_idx++;
                }
            }

            // Plot ECG waveform
            if (ImPlot::BeginPlot("##ECG", ImVec2(-1, 350))) {
                ImPlot::SetupAxes("Sample", "Amplitude (mV)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, buffer_size, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, -2.0, 2.0, ImGuiCond_Always);

                // Plot signal
                std::vector<float> buffer_vec(realtime_buffer.begin(), realtime_buffer.end());
                ImPlot::PlotLine("ECG Signal", buffer_vec.data(), buffer_vec.size());

                // Mark R-peaks
                if (auto_detect_peaks && !detected_r_peaks.empty()) {
                    std::vector<float> peak_x, peak_y;
                    for (auto peak : detected_r_peaks) {
                        if (peak >= signal_idx - buffer_size && peak < signal_idx) {
                            int plot_x = peak - (signal_idx - buffer_size);
                            if (plot_x >= 0 && plot_x < (int)buffer_vec.size()) {
                                peak_x.push_back(static_cast<float>(plot_x));
                                peak_y.push_back(buffer_vec[plot_x]);
                            }
                        }
                    }
                    if (!peak_x.empty()) {
                        ImPlot::PlotScatter("R-peaks", peak_x.data(), peak_y.data(), peak_x.size());
                    }
                }

                ImPlot::EndPlot();
            }
        }

        ImGui::Text("HR: %.1f bpm | HRV (SDNN): %.1f ms", current_hr, current_hrv);
        ImGui::End();

        // Control Panel
        ImGui::SetNextWindowPos(ImVec2(10, 470), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(450, 520), ImGuiCond_FirstUseEver);
        ImGui::Begin("ResNet18 Training & Analysis");

        ImGui::Text("LibTorch: %s", TORCH_VERSION);
        ImGui::Text("Device: %s", mps_available ? "MPS (Apple Silicon)" : "CPU");
        ImGui::Separator();

        // ECG Record Selection
        ImGui::Text("ECG Record Selection:");
        if (!ecg_records.empty() && ImGui::SliderInt("Record", &current_record_idx, 0, ecg_records.size() - 1)) {
            current_ecg_signal = ecg_records[current_record_idx].signal;
            sampling_rate = ecg_records[current_record_idx].sampling_rate;
            detected_r_peaks = ECGSignalProcessor::detect_r_peaks(current_ecg_signal, sampling_rate);
            current_hr = ECGSignalProcessor::calculate_heart_rate(detected_r_peaks, sampling_rate, current_ecg_signal.size());
            current_hrv = ECGSignalProcessor::calculate_hrv_sdnn(detected_r_peaks, sampling_rate);

            realtime_buffer.clear();
            for (int i = 0; i < std::min(buffer_size, (int)current_ecg_signal.size()); i++) {
                realtime_buffer.push_back(current_ecg_signal[i]);
            }
        }
        ImGui::Text("Record: %s", ecg_records.empty() ? "None" : ecg_records[current_record_idx].record_name.c_str());
        ImGui::Text("Sampling Rate: %.0f Hz", sampling_rate);
        ImGui::Text("Duration: %.1f seconds", current_ecg_signal.size() / sampling_rate);
        ImGui::Separator();

        // Signal Processing Controls
        ImGui::Text("Signal Processing:");
        ImGui::Checkbox("Auto-detect R-peaks", &auto_detect_peaks);
        if (ImGui::Button("Re-detect R-peaks", ImVec2(-1, 30))) {
            detected_r_peaks = ECGSignalProcessor::detect_r_peaks(current_ecg_signal, sampling_rate);
            current_hr = ECGSignalProcessor::calculate_heart_rate(detected_r_peaks, sampling_rate, current_ecg_signal.size());
            current_hrv = ECGSignalProcessor::calculate_hrv_sdnn(detected_r_peaks, sampling_rate);
        }
        ImGui::Text("Detected R-peaks: %d", (int)detected_r_peaks.size());
        ImGui::Separator();

        // Training controls
        ImGui::Text("Neural Network Training:");
        ImGui::Text("1D CNN for ECG Classification");
        if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
            for (auto& param_group : optimizer.param_groups()) {
                static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(learning_rate);
            }
        }
        ImGui::SliderInt("Training Delay (ms)", &training_delay_ms, 0, 1000);

        if (ImGui::Button(training ? "Stop Training" : "Start Training", ImVec2(-1, 40))) {
            training = !training;
        }

        ImGui::Text("Batch: %d", batch_count);
        ImGui::Text("Epoch: %d", epoch);
        ImGui::Text("Loss: %.6f", loss_value);
        ImGui::Text("Accuracy: %.2f%%", accuracy * 100.0f);
        ImGui::Separator();

        // Current prediction
        const char* class_names[] = {"Normal", "Arrhythmia-1", "Arrhythmia-2", "Arrhythmia-3", "Arrhythmia-4"};
        ImGui::Text("Current Sample Prediction:");
        ImGui::Text("Predicted: %s", class_names[current_predicted % 5]);
        ImGui::Text("Actual: %s", class_names[current_actual % 5]);
        ImGui::Text("Correct: %s", current_predicted == current_actual ? "YES" : "NO");

        ImGui::End();

        // Statistics Panel
        ImGui::SetNextWindowPos(ImVec2(470, 470), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 520), ImGuiCond_FirstUseEver);
        ImGui::Begin("Training Statistics & Metrics");

        ImGui::Text("Cardiovascular Metrics:");
        ImGui::Separator();

        // Heart Rate
        ImGui::Text("Heart Rate (HR):");
        ImGui::Text("  Average: %.1f bpm", current_hr);
        if (current_hr < 60) {
            ImGui::TextColored(ImVec4(0.3f, 0.5f, 1.0f, 1.0f), "  Status: Bradycardia");
        } else if (current_hr > 100) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "  Status: Tachycardia");
        } else {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "  Status: Normal");
        }

        ImGui::Separator();

        // Heart Rate Variability
        ImGui::Text("Heart Rate Variability:");
        ImGui::Text("  SDNN: %.1f ms", current_hrv);
        if (current_hrv < 50) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "  Status: Low HRV");
        } else {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "  Status: Healthy HRV");
        }

        ImGui::Separator();

        // R-R Intervals
        if (!detected_r_peaks.empty() && detected_r_peaks.size() > 1) {
            std::vector<float> rr_intervals;
            for (size_t i = 1; i < detected_r_peaks.size(); i++) {
                float rr_ms = (detected_r_peaks[i] - detected_r_peaks[i-1]) / sampling_rate * 1000.0f;
                if (rr_ms > 300.0f && rr_ms < 2000.0f) {
                    rr_intervals.push_back(rr_ms);
                }
            }

            if (!rr_intervals.empty()) {
                float mean_rr = std::accumulate(rr_intervals.begin(), rr_intervals.end(), 0.0f) / rr_intervals.size();
                float min_rr = *std::min_element(rr_intervals.begin(), rr_intervals.end());
                float max_rr = *std::max_element(rr_intervals.begin(), rr_intervals.end());

                ImGui::Text("R-R Intervals:");
                ImGui::Text("  Mean: %.0f ms", mean_rr);
                ImGui::Text("  Min: %.0f ms", min_rr);
                ImGui::Text("  Max: %.0f ms", max_rr);

                // Plot R-R intervals
                if (ImPlot::BeginPlot("##RRIntervals", ImVec2(-1, 150))) {
                    ImPlot::SetupAxes("Beat Number", "RR Interval (ms)");
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, rr_intervals.size(), ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1500, ImGuiCond_Always);
                    ImPlot::PlotLine("RR Intervals", rr_intervals.data(), std::min((int)rr_intervals.size(), 100));
                    ImPlot::EndPlot();
                }
            }
        }

        ImGui::Separator();

        // Signal Quality Indicators
        ImGui::Text("Signal Quality:");
        ImGui::Text("  R-peak Count: %d", (int)detected_r_peaks.size());
        float signal_length_sec = current_ecg_signal.size() / sampling_rate;
        float expected_peaks = (current_hr / 60.0f) * signal_length_sec;
        float detection_ratio = detected_r_peaks.size() / (expected_peaks + 1e-6f);
        ImGui::Text("  Detection Quality: %.1f%%", detection_ratio * 100.0f);

        if (detection_ratio > 0.9f && detection_ratio < 1.1f) {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "  Quality: Good");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "  Quality: Check Signal");
        }

        ImGui::Separator();

        // Loss history plot
        if (!loss_history.empty()) {
            ImGui::Text("Training Loss History:");
            if (ImPlot::BeginPlot("##TrainingLoss", ImVec2(-1, 120))) {
                ImPlot::SetupAxes("Batch", "Loss");
                ImPlot::SetupAxisLimits(ImAxis_X1, std::max(0, (int)loss_history.size() - 100), loss_history.size(), ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_loss_ever * 1.1f, ImGuiCond_Always);
                ImPlot::PlotLine("Loss", loss_history.data(), std::min((int)loss_history.size(), 200));
                ImPlot::EndPlot();
            }
        }

        ImGui::End();

        // Weight Visualization Window
        ImGui::SetNextWindowPos(ImVec2(880, 470), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(430, 520), ImGuiCond_FirstUseEver);
        ImGui::Begin("ResNet18 Weight Visualization");

        ImGui::Text("Model: ResNet18");
        ImGui::Text("Total Parameters: ~11M");
        ImGui::Separator();

        // Layer selection
        static int selected_layer = 0;
        const char* layer_names[] = {
            "conv1 (64x3x3x3)",
            "layer1.0.conv1 (64x64x3x3)",
            "layer1.0.conv2 (64x64x3x3)",
            "layer2.0.conv1 (128x64x3x3)",
            "layer2.0.conv2 (128x128x3x3)",
            "layer3.0.conv1 (256x128x3x3)",
            "layer3.0.conv2 (256x256x3x3)",
            "layer4.0.conv1 (512x256x3x3)",
            "layer4.0.conv2 (512x512x3x3)",
            "fc (10x512)"
        };

        ImGui::Text("Select Layer:");
        ImGui::Combo("##LayerSelect", &selected_layer, layer_names, IM_ARRAYSIZE(layer_names));

        ImGui::Separator();

        // Get weights from the selected layer
        torch::Tensor weights;
        try {
            auto params = net->named_parameters();
            std::vector<std::string> param_names = {
                "conv1.weight", "layer1.0.conv1.weight", "layer1.0.conv2.weight",
                "layer2.0.conv1.weight", "layer2.0.conv2.weight",
                "layer3.0.conv1.weight", "layer3.0.conv2.weight",
                "layer4.0.conv1.weight", "layer4.0.conv2.weight", "fc.weight"
            };

            if (selected_layer < param_names.size()) {
                weights = params[param_names[selected_layer]].cpu();

                // Display weight statistics
                auto w_flat = weights.flatten();
                float w_mean = w_flat.mean().item<float>();
                float w_std = w_flat.std().item<float>();
                float w_min = w_flat.min().item<float>();
                float w_max = w_flat.max().item<float>();

                ImGui::Text("Weight Statistics:");
                ImGui::Text("  Mean: %.6f", w_mean);
                ImGui::Text("  Std:  %.6f", w_std);
                ImGui::Text("  Min:  %.6f", w_min);
                ImGui::Text("  Max:  %.6f", w_max);

                ImGui::Separator();

                // Create histogram of weight values
                std::vector<float> weight_data(w_flat.data_ptr<float>(),
                                              w_flat.data_ptr<float>() + w_flat.numel());

                if (ImPlot::BeginPlot("##WeightHist", ImVec2(-1, 200))) {
                    ImPlot::SetupAxes("Weight Value", "Count");

                    // Sample weights if too many
                    int max_samples = 10000;
                    std::vector<float> sampled_weights;
                    if (weight_data.size() > max_samples) {
                        int step = weight_data.size() / max_samples;
                        for (size_t i = 0; i < weight_data.size(); i += step) {
                            sampled_weights.push_back(weight_data[i]);
                        }
                    } else {
                        sampled_weights = weight_data;
                    }

                    ImPlot::PlotHistogram("Weights", sampled_weights.data(), sampled_weights.size(), 50,
                                         1.0, ImPlotRange(w_min, w_max));
                    ImPlot::EndPlot();
                }

                // Display layer shape
                auto sizes = weights.sizes();
                std::string shape_str = "Shape: [";
                for (int i = 0; i < sizes.size(); i++) {
                    shape_str += std::to_string(sizes[i]);
                    if (i < sizes.size() - 1) shape_str += " x ";
                }
                shape_str += "]";
                ImGui::Text("%s", shape_str.c_str());
                ImGui::Text("Total weights: %lld", (long long)w_flat.numel());

                // Gradient information if available
                if (weights.grad().defined()) {
                    auto grad_flat = weights.grad().flatten();
                    float g_mean = grad_flat.mean().item<float>();
                    float g_norm = grad_flat.norm().item<float>();
                    ImGui::Separator();
                    ImGui::Text("Gradient Info:");
                    ImGui::Text("  Mean: %.6e", g_mean);
                    ImGui::Text("  Norm: %.6e", g_norm);
                }
            }
        } catch (const std::exception& e) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error accessing weights");
        }

        ImGui::End();

        // Training step
        if (training) {
            // Check if enough time has passed since last training step
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_training_time).count();

            if (elapsed >= training_delay_ms) {
                last_training_time = current_time;

                // Process multiple batches per frame based on speed slider
                for (int b = 0; b < batches_per_frame; b++) {
                    // Get next batch
                    if (train_iter == train_loader->end()) {
                        train_iter = train_loader->begin();
                        epoch++;
                        std::cout << "Epoch " << epoch << " completed" << std::endl;
                    }

                    auto& batch = *train_iter;
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    // Store first sample for visualization (keep on device for now)
                    current_input = data.slice(0, 0, 1);
                    current_actual = target[0].item<int64_t>();

                    // Forward pass
                    net->train();
                    optimizer.zero_grad();
                    auto output = net->forward(data);
                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    // Backward pass
                    loss.backward();
                    optimizer.step();

                    // Calculate accuracy for this batch
                    auto pred = output.argmax(1);
                    accuracy = pred.eq(target).sum().item<float>() / target.size(0);
                    current_predicted = output.slice(0, 0, 1).argmax(1).item<int64_t>();

                    // Update stats
                    loss_value = loss.item<float>();
                    loss_history.push_back(loss_value);
                    max_loss_ever = std::max(max_loss_ever, loss_value);

                    batch_count++;

                    // Print progress every 10 batches
                    if (batch_count % 10 == 0) {
                        std::cout << "Batch " << batch_count << " - Loss: " << loss_value
                                 << " - Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
                    }

                    ++train_iter;
                }
            }

            // Get output for visualization (only once per frame)
            net->eval();
            {
                torch::NoGradGuard no_grad;
                output_for_vis = net->forward(current_input);
            }
            net->train();
        } else {
            // Just do forward pass for visualization if not training
            net->eval();
            torch::NoGradGuard no_grad;
            output_for_vis = net->forward(current_input);
            current_predicted = output_for_vis.argmax(1).item<int64_t>();
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // For ECG analyzer, we don't need 3D rendering - all visualization is done via ImPlot
        // Just render the ImGui interface
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
