#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include "ollama_models.h"

struct llama_model;
struct llama_context;

struct LayerActivation {
    int layer_index;
    float mean;
    float norm;
    float max_val;
    float cosine_prev;  // cosine similarity with previous layer
    std::vector<float> values;
    int rows;
    int cols;
    std::string name;   // "attn_out" or "l_out"
};

struct TokenLogitInfo {
    std::string token_text;
    float probability;
    float entropy;
    std::vector<std::pair<std::string, float>> top_k;
};

class OpenGllamaApplet {
public:
    OpenGllamaApplet();
    ~OpenGllamaApplet();

    bool initialize();
    void draw_ui(int width, int height);
    void cleanup();

    bool load_model(const std::string& path);
    void unload_model();

    bool is_model_loaded() const { return model_loaded_; }
    std::string model_path() const { return model_path_; }

    static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data);

private:
    void draw_ollama_models();
    void draw_inference_view();
    void update_activation_textures();

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

    std::string model_path_;
    bool model_loaded_ = false;

    int n_gpu_layers_ = 99;
    int context_size_ = 2048;

    // Inference hyperparameters
    int max_tokens_ = 256;
    float temperature_ = 0.8f;
    int top_k_ = 40;
    float top_p_ = 0.95f;
    float min_p_ = 0.05f;
    float repeat_penalty_ = 1.1f;
    int repeat_last_n_ = 64;
    uint32_t seed_ = 0;  // 0 = random

    // Playback control
    enum class InferenceMode { Continuous, Paused };
    std::atomic<InferenceMode> inference_mode_{InferenceMode::Continuous};
    std::atomic<bool> step_requested_{false};
    int token_delay_ms_ = 0;  // ms delay between tokens (0 = full speed)

    // Inference state
    std::string prompt_buf_;
    std::string output_text_;
    std::atomic<bool> inference_running_{false};
    std::atomic<int> tokens_generated_{0};
    std::atomic<bool> inference_finished_{false};
    std::thread inference_thread_;
    std::mutex output_mutex_;

    void run_inference_async(const std::string& prompt);

    // Async model loading
    std::thread load_thread_;
    std::atomic<float> load_progress_{0.0f};
    std::atomic<bool> load_finished_{false};
    std::atomic<bool> load_success_{false};
    std::string loading_model_name_;
    std::string load_error_msg_;

    void load_model_async(const std::string& path, const std::string& display_name);

    // Activations captured from eval callback (protected by output_mutex_)
    std::vector<LayerActivation> activations_;
    std::vector<LayerActivation> pending_activations_;
    std::vector<TokenLogitInfo> token_logits_;
    std::vector<unsigned int> layer_textures_;
    std::atomic<bool> textures_dirty_{false};

    // Context activation map: [n_layers][n_context_tokens] = activation norm
    std::vector<std::vector<float>> context_map_;
    std::vector<std::string> context_tokens_;
    unsigned int context_map_texture_ = 0;
    bool context_map_dirty_ = false;
    void update_context_map_texture();

    OllamaModelStore ollama_store_;
};
