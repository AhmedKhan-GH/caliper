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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
// 3D LANDING PAGE VISUALIZATION
// ============================================================================

static float hashf(int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)(seed & 0x7FFFFFFF) / 2147483647.0f;
}

class LandingPage {
public:
    bool initialize() {
        // -- Compile shaders --
        const char* vertSrc = R"(
            #version 330 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec4 aColorPhase;

            uniform mat4 uMVP;
            uniform float uTime;
            uniform float uPointScale;
            uniform float uOscAmp;

            out vec4 vColor;

            void main() {
                float phase = aColorPhase.a;
                vec3 pos = aPos;
                float t = uTime * 0.7 + phase * 6.2832;
                pos.x += uOscAmp * sin(t);
                pos.y += uOscAmp * cos(t * 1.3);
                pos.z += uOscAmp * sin(t * 0.7);

                gl_Position = uMVP * vec4(pos, 1.0);
                gl_PointSize = uPointScale * (1.0 + 0.3 * sin(uTime * 2.0 + phase * 6.2832));

                float pulse = 0.75 + 0.25 * sin(uTime * 1.5 + phase * 6.2832);
                vColor = vec4(aColorPhase.rgb * pulse, 1.0);
            }
        )";

        const char* fragSrc = R"(
            #version 330 core
            in vec4 vColor;
            out vec4 FragColor;
            uniform int uMode;

            void main() {
                if (uMode == 1) {
                    vec2 c = gl_PointCoord - vec2(0.5);
                    float d = length(c);
                    float glow = exp(-d * d * 10.0);
                    FragColor = vec4(vColor.rgb, glow);
                } else {
                    FragColor = vec4(vColor.rgb, 0.22);
                }
            }
        )";

        GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
        if (!vs || !fs) return false;

        prog_ = glCreateProgram();
        glAttachShader(prog_, vs);
        glAttachShader(prog_, fs);
        glLinkProgram(prog_);

        int ok;
        glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetProgramInfoLog(prog_, 512, nullptr, log);
            std::cerr << "Shader link error: " << log << std::endl;
            return false;
        }
        glDeleteShader(vs);
        glDeleteShader(fs);

        locMVP_ = glGetUniformLocation(prog_, "uMVP");
        locTime_ = glGetUniformLocation(prog_, "uTime");
        locPointScale_ = glGetUniformLocation(prog_, "uPointScale");
        locOscAmp_ = glGetUniformLocation(prog_, "uOscAmp");
        locMode_ = glGetUniformLocation(prog_, "uMode");

        // -- Generate neural network geometry --
        struct Layer { int count; float radius; float z; glm::vec3 color; };
        Layer layers[] = {
            { 8,  1.0f, -3.0f, {0.2f, 0.85f, 1.0f}},
            {12,  1.6f, -1.5f, {0.3f, 0.65f, 1.0f}},
            {16,  2.0f,  0.0f, {0.5f, 0.45f, 1.0f}},
            {12,  1.6f,  1.5f, {0.7f, 0.35f, 0.9f}},
            { 6,  0.8f,  3.0f, {1.0f, 0.35f, 0.7f}},
        };
        int numLayers = 5;

        struct Vert { float x, y, z, r, g, b, phase; };
        std::vector<Vert> nodeVerts;
        std::vector<Vert> edgeVerts;

        // Track node indices per layer for edge generation
        std::vector<std::vector<int>> layerNodeIndices(numLayers);

        int nodeIdx = 0;
        for (int li = 0; li < numLayers; li++) {
            auto& L = layers[li];
            for (int i = 0; i < L.count; i++) {
                float angle = 2.0f * (float)M_PI * (float)i / (float)L.count;
                // Add slight vertical scatter for organic look
                float yoff = 0.15f * std::sin(angle * 3.0f + (float)li);
                Vert v;
                v.x = L.radius * std::cos(angle);
                v.y = L.radius * std::sin(angle) + yoff;
                v.z = L.z;
                v.r = L.color.r;
                v.g = L.color.g;
                v.b = L.color.b;
                v.phase = hashf(nodeIdx * 7 + 13);
                nodeVerts.push_back(v);
                layerNodeIndices[li].push_back(nodeIdx);
                nodeIdx++;
            }
        }
        numNodes_ = (int)nodeVerts.size();

        // Generate edges between adjacent layers
        int globalOffset = 0;
        for (int li = 0; li < numLayers - 1; li++) {
            auto& curLayer = layers[li];
            auto& nextLayer = layers[li + 1];
            auto& curIndices = layerNodeIndices[li];
            auto& nextIndices = layerNodeIndices[li + 1];

            for (int i = 0; i < curLayer.count; i++) {
                // Connect to 2-3 nodes in next layer
                int connections = 2 + (int)(hashf(i * 31 + li * 97) * 2.0f);
                float baseAngle = 2.0f * (float)M_PI * (float)i / (float)curLayer.count;

                for (int c = 0; c < connections; c++) {
                    // Pick target in next layer by angle proximity + offset
                    float targetAngle = baseAngle + (hashf(i * 17 + c * 53 + li * 71) - 0.5f) * 1.5f;
                    // Find nearest node in next layer
                    int bestJ = 0;
                    float bestDist = 999.0f;
                    for (int j = 0; j < nextLayer.count; j++) {
                        float a = 2.0f * (float)M_PI * (float)j / (float)nextLayer.count;
                        float d = std::abs(a - targetAngle);
                        if (d > (float)M_PI) d = 2.0f * (float)M_PI - d;
                        if (d < bestDist) { bestDist = d; bestJ = j; }
                    }

                    auto& n1 = nodeVerts[curIndices[i]];
                    auto& n2 = nodeVerts[nextIndices[bestJ]];

                    // Blend colors for the edge
                    float blend = 0.5f;
                    Vert e1, e2;
                    e1.x = n1.x; e1.y = n1.y; e1.z = n1.z;
                    e1.r = n1.r * blend + n2.r * (1 - blend);
                    e1.g = n1.g * blend + n2.g * (1 - blend);
                    e1.b = n1.b * blend + n2.b * (1 - blend);
                    e1.phase = n1.phase;

                    e2.x = n2.x; e2.y = n2.y; e2.z = n2.z;
                    e2.r = e1.r; e2.g = e1.g; e2.b = e1.b;
                    e2.phase = n2.phase;

                    edgeVerts.push_back(e1);
                    edgeVerts.push_back(e2);
                }
            }
        }
        numEdgeVerts_ = (int)edgeVerts.size();

        // -- Generate floating particles --
        std::vector<Vert> particleVerts;
        for (int i = 0; i < 180; i++) {
            float theta = hashf(i * 3 + 1000) * 2.0f * (float)M_PI;
            float phi = std::acos(2.0f * hashf(i * 3 + 2000) - 1.0f);
            float r = 3.8f * std::cbrt(hashf(i * 3 + 3000));

            Vert v;
            v.x = r * std::sin(phi) * std::cos(theta);
            v.y = r * std::sin(phi) * std::sin(theta);
            v.z = r * std::cos(phi);

            float brightness = 0.25f + 0.35f * hashf(i * 3 + 4000);
            float hue = hashf(i * 3 + 5000);
            v.r = brightness * (0.4f + 0.6f * hue);
            v.g = brightness * (0.5f + 0.5f * (1.0f - hue));
            v.b = brightness * (0.8f + 0.2f * hue);
            v.phase = hashf(i * 3 + 6000);
            particleVerts.push_back(v);
        }
        numParticles_ = (int)particleVerts.size();

        // -- Upload node geometry --
        glGenVertexArrays(1, &nodeVAO_);
        glGenBuffers(1, &nodeVBO_);
        glBindVertexArray(nodeVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, nodeVBO_);
        glBufferData(GL_ARRAY_BUFFER, nodeVerts.size() * sizeof(Vert), nodeVerts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // -- Upload edge geometry --
        glGenVertexArrays(1, &edgeVAO_);
        glGenBuffers(1, &edgeVBO_);
        glBindVertexArray(edgeVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, edgeVBO_);
        glBufferData(GL_ARRAY_BUFFER, edgeVerts.size() * sizeof(Vert), edgeVerts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // -- Upload particle geometry --
        glGenVertexArrays(1, &particleVAO_);
        glGenBuffers(1, &particleVBO_);
        glBindVertexArray(particleVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO_);
        glBufferData(GL_ARRAY_BUFFER, particleVerts.size() * sizeof(Vert), particleVerts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        startTime_ = (float)glfwGetTime();
        return true;
    }

    void render(int fb_w, int fb_h) {
        float time = (float)glfwGetTime() - startTime_;
        float aspect = (float)fb_w / std::max(1.0f, (float)fb_h);

        // Build MVP: perspective + camera + rotation
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 50.0f);
        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 10.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, time * 0.3f, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, 0.3f * std::sin(time * 0.15f), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 mvp = proj * view * model;

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glUseProgram(prog_);
        glUniformMatrix4fv(locMVP_, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1f(locTime_, time);

        // Draw edges (lines)
        glUniform1i(locMode_, 0);
        glUniform1f(locPointScale_, 1.0f);
        glUniform1f(locOscAmp_, 0.06f);
        glBindVertexArray(edgeVAO_);
        glDrawArrays(GL_LINES, 0, numEdgeVerts_);

        // Draw nodes (points)
        glUniform1i(locMode_, 1);
        glUniform1f(locPointScale_, 8.0f);
        glUniform1f(locOscAmp_, 0.06f);
        glBindVertexArray(nodeVAO_);
        glDrawArrays(GL_POINTS, 0, numNodes_);

        // Draw particles (small points)
        glUniform1f(locPointScale_, 2.5f);
        glUniform1f(locOscAmp_, 0.25f);
        glBindVertexArray(particleVAO_);
        glDrawArrays(GL_POINTS, 0, numParticles_);

        glBindVertexArray(0);
        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }

    void cleanup() {
        if (prog_) glDeleteProgram(prog_);
        if (nodeVAO_) glDeleteVertexArrays(1, &nodeVAO_);
        if (nodeVBO_) glDeleteBuffers(1, &nodeVBO_);
        if (edgeVAO_) glDeleteVertexArrays(1, &edgeVAO_);
        if (edgeVBO_) glDeleteBuffers(1, &edgeVBO_);
        if (particleVAO_) glDeleteVertexArrays(1, &particleVAO_);
        if (particleVBO_) glDeleteBuffers(1, &particleVBO_);
    }

private:
    GLuint compileShader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader compile error: " << log << std::endl;
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    GLuint prog_ = 0;
    GLuint nodeVAO_ = 0, nodeVBO_ = 0;
    GLuint edgeVAO_ = 0, edgeVBO_ = 0;
    GLuint particleVAO_ = 0, particleVBO_ = 0;
    int numNodes_ = 0, numEdgeVerts_ = 0, numParticles_ = 0;
    GLint locMVP_ = -1, locTime_ = -1, locPointScale_ = -1, locOscAmp_ = -1, locMode_ = -1;
    float startTime_ = 0;
};

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

        // Initialize 3D landing page
        if (!landing_.initialize()) {
            std::cerr << "Landing page 3D init failed" << std::endl;
            return false;
        }

        // Scan dataset (non-fatal if empty)
        std::string data_dir = "../data/seniordesign_upload_balanced/ekg_data";
        samples_ = loader::scan_directory(data_dir);
        if (!samples_.empty()) {
            std::cout << "Found " << samples_.size() << " ECG samples" << std::endl;
            select_sample(0);
            bg_ = std::make_unique<BackgroundProcessor>();
        } else {
            std::cout << "No ECG data found in " << data_dir << " (load data to use the ECG explorer)" << std::endl;
        }

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

            // Render 3D scene behind ImGui on landing page
            if (page_ == AppPage::Landing) {
                landing_.render(dw, dh);
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (page_ == AppPage::Landing) {
                draw_landing_ui();
            } else {
                draw_ecg_ui();
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window_);
        }
    }

    void cleanup() {
        bg_.reset(); // join background thread
        landing_.cleanup();
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

    void select_sample(int idx) {
        if (idx < 0 || idx >= (int)samples_.size()) return;
        selected_ = idx;
        auto& s = samples_[selected_];

        if (!s.loaded) loader::load_csv(s);
        if (s.loaded && !s.processed_valid) {
            dsp::process(s, params_);
        }

        // Prefetch neighbors outward from current selection
        if (bg_) {
            auto neighbors = outward_indices(idx, (int)samples_.size() - 1);
            bg_->enqueue(&samples_, params_, neighbors);
        }
    }

    void on_params_changed() {
        params_.version++;

        // Invalidate all
        for (auto& s : samples_) s.processed_valid = false;

        // Reprocess current immediately
        if (selected_ >= 0 && selected_ < (int)samples_.size()) {
            auto& cur = samples_[selected_];
            if (cur.loaded) dsp::process(cur, params_);
        }

        // Queue the rest in background, expanding outward from current selection
        if (bg_) {
            auto others = outward_indices(selected_, (int)samples_.size() - 1);
            bg_->enqueue(&samples_, params_, others);
        }
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

    // ── Landing Page UI (ImGui overlay on 3D scene) ──

    void draw_landing_ui() {
        ImGuiViewport* vp = ImGui::GetMainViewport();
        float w = vp->WorkSize.x;
        float h = vp->WorkSize.y;

        // Full-screen transparent window
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
        ImGui::Begin("##Landing", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleColor();

        // -- Title --
        {
            const char* title = "C A L I P E R";
            ImGui::SetWindowFontScale(3.0f);
            float tw = ImGui::CalcTextSize(title).x;
            ImGui::SetCursorPosX((w - tw) * 0.5f);
            ImGui::SetCursorPosY(h * 0.10f);
            ImGui::TextColored(ImVec4(0.75f, 0.88f, 1.0f, 1.0f), "%s", title);
            ImGui::SetWindowFontScale(1.0f);
        }

        // -- Subtitle --
        {
            const char* sub = "Machine Learning Signal Processing";
            ImGui::SetWindowFontScale(1.3f);
            float sw = ImGui::CalcTextSize(sub).x;
            ImGui::SetCursorPosX((w - sw) * 0.5f);
            ImGui::SetCursorPosY(h * 0.10f + 60.0f);
            ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.75f, 0.85f), "%s", sub);
            ImGui::SetWindowFontScale(1.0f);
        }

        // -- Applet section --
        float card_w = std::min(520.0f, w * 0.55f);
        float card_x = (w - card_w) * 0.5f;

        // Section header
        {
            const char* hdr = "Tools";
            ImGui::SetWindowFontScale(1.1f);
            ImGui::SetCursorPosX(card_x);
            ImGui::SetCursorPosY(h * 0.32f);
            ImGui::TextColored(ImVec4(0.5f, 0.6f, 0.8f, 0.7f), "%s", hdr);
            ImGui::SetWindowFontScale(1.0f);
        }

        // Thin separator
        ImGui::SetCursorPosX(card_x);
        ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.3f, 0.4f, 0.6f, 0.4f));
        ImGui::Separator();
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // -- Applet Card: Baseline Wander Correction --
        ImGui::SetCursorPosX(card_x);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f, 0.09f, 0.16f, 0.88f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.25f, 0.35f, 0.55f, 0.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 16));

        ImGui::BeginChild("##applet_bwc", ImVec2(card_w, 150), ImGuiChildFlags_Borders);

        ImGui::SetWindowFontScale(1.35f);
        ImGui::TextColored(ImVec4(0.4f, 0.78f, 1.0f, 1.0f), "Baseline Wander Correction");
        ImGui::SetWindowFontScale(1.0f);

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.65f, 0.75f, 0.9f));
        ImGui::TextWrapped(
            "Remove low-frequency baseline drift from 12-lead ECG signals "
            "using adaptive 4th-order Butterworth high-pass filtering with "
            "zero-phase distortion. Real-time visualization and parameter tuning.");
        ImGui::PopStyleColor();

        ImGui::Spacing();

        // Launch button - right aligned
        float btnW = 110.0f;
        float avail = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - btnW);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.14f, 0.32f, 0.58f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.42f, 0.70f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.25f, 0.48f, 1.0f));
        if (ImGui::Button("Launch  >>", ImVec2(btnW, 32))) {
            page_ = AppPage::ECGApp;
            params_.baseline_wander_correction = true;
            if (params_.baseline_cutoff_hz <= 0) params_.baseline_cutoff_hz = 0.5f;
            if (!samples_.empty()) on_params_changed();
            glfwSetWindowTitle(window_, "Caliper - Baseline Wander Correction");
        }
        ImGui::PopStyleColor(3);

        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);

        // -- Footer --
        {
            const char* footer = "OpenGL 3.3  |  ImGui  |  ImPlot  |  LibTorch";
            ImGui::SetWindowFontScale(0.85f);
            float fw = ImGui::CalcTextSize(footer).x;
            ImGui::SetCursorPosX((w - fw) * 0.5f);
            ImGui::SetCursorPosY(h - 40.0f);
            ImGui::TextColored(ImVec4(0.35f, 0.40f, 0.50f, 0.5f), "%s", footer);
            ImGui::SetWindowFontScale(1.0f);
        }

        ImGui::End();
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

        if (samples_.empty()) {
            ImGui::TextColored({1.0f, 0.7f, 0.3f, 1.0f}, "No ECG data loaded.");
            ImGui::TextWrapped("Place .csv files in the data directory and restart.");
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
    AppPage page_ = AppPage::Landing;
    LandingPage landing_;

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
