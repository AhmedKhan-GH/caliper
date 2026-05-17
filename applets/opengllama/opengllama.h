#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

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

private:
    void draw_model_loader();
    void draw_inference_panel();
    void draw_activation_viewer();

    bool load_model(const std::string& path);
    void unload_model();
    bool run_inference(const std::string& prompt);

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

    std::string model_path_;
    std::string prompt_text_;
    std::string output_text_;
    bool model_loaded_ = false;
    bool inference_running_ = false;

    int n_gpu_layers_ = 99;
    int context_size_ = 2048;

    std::vector<LayerActivation> activations_;
    int selected_layer_ = 0;

    unsigned int activation_texture_ = 0;
    bool texture_needs_update_ = false;
};
