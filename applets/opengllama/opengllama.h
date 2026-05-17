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
    std::vector<float> values;
    int rows;
    int cols;
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

    // Activations (protected by output_mutex_ during inference)
    std::vector<LayerActivation> activations_;
    std::vector<unsigned int> layer_textures_;
    std::atomic<bool> textures_dirty_{false};

    OllamaModelStore ollama_store_;
};
