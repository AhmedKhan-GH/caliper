#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

#include <imterm/terminal.hpp>
#include <imterm/terminal_helpers.hpp>

#include "ollama_models.h"

struct llama_model;
struct llama_context;

class OpenGllamaApplet;

struct LlamaTerminalValue {
    OpenGllamaApplet* applet = nullptr;
};

class LlamaTerminalHelper : public ImTerm::basic_terminal_helper<LlamaTerminalHelper, LlamaTerminalValue> {
public:
    LlamaTerminalHelper();

    static std::vector<std::string> no_completion(argument_type&) { return {}; }

    static void cmd_infer(argument_type& arg);
    static void cmd_clear(argument_type& arg);
    static void cmd_status(argument_type& arg);
    static void cmd_help(argument_type& arg);
    static void cmd_load(argument_type& arg);
    static void cmd_unload(argument_type& arg);
    static void cmd_set(argument_type& arg);
    static std::vector<std::string> set_completion(argument_type& arg);
};

using LlamaTerminal = ImTerm::terminal<LlamaTerminalHelper>;

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
    std::string run_inference(const std::string& prompt);

    bool is_model_loaded() const { return model_loaded_; }
    std::string model_path() const { return model_path_; }
    int gpu_layers() const { return n_gpu_layers_; }
    int context_size() const { return context_size_; }
    void set_gpu_layers(int n) { n_gpu_layers_ = n; }
    void set_context_size(int n) { context_size_ = n; }
    void start_inference(const std::string& prompt) { run_inference_async(prompt); }

private:
    void draw_ollama_models();
    void draw_inference_view();
    void draw_terminal();

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

    LlamaTerminalValue term_value_;
    std::shared_ptr<LlamaTerminalHelper> term_helper_;
    std::unique_ptr<LlamaTerminal> terminal_;
};
