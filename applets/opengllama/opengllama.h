#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>

#include "ollama_models.h"
#include "ollama_server.h"

struct llama_model;
struct llama_context;

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

    std::string format_chat_prompt(const std::string& user_input) const;
    void run_inference_async(const std::string& prompt);
    void run_inference_blocking(const std::string& prompt,
                                const std::function<bool(const std::string&)>& token_cb);
    friend class OllamaServer;

    // Async model loading
    std::thread load_thread_;
    std::atomic<float> load_progress_{0.0f};
    std::atomic<bool> load_finished_{false};
    std::atomic<bool> load_success_{false};
    std::string loading_model_name_;
    std::string load_error_msg_;

    void load_model_async(const std::string& path, const std::string& display_name);

    // Context activation map: [n_layers][n_context_tokens] = activation norm
    std::vector<std::vector<float>> context_map_;
    std::vector<std::string> context_tokens_;
    std::atomic<bool> context_map_dirty_{false};

    // Text heatmap: GL texture baked to match paragraph layout
    unsigned int ctx_text_heatmap_tex_ = 0;
    int ctx_text_heatmap_tex_w_ = 0;
    int ctx_text_heatmap_tex_h_ = 0;
    int ctx_text_heatmap_n_ctx_ = 0;
    int ctx_text_heatmap_n_gen_ = 0;
    float ctx_text_heatmap_last_width_ = 0.0f;
    struct TokenLayout { int token_idx; float x, y, w; std::string text; };
    std::vector<TokenLayout> ctx_text_layout_;
    float ctx_text_total_h_ = 0.0f;
    std::vector<unsigned char> ctx_text_heatmap_pixels_;  // reusable pixel buffer
    enum TextHeatmapMode { THM_NONE = 0, THM_EMA, THM_MAX, THM_RECENT, THM_FINAL_LAYER };
    int ctx_text_heatmap_mode_ = THM_EMA;
    int ctx_text_heatmap_prev_mode_ = -1;

    // Per-layer attention weights: [n_layers] = head-averaged attention over KV for latest token
    std::vector<std::vector<float>> pending_attn_weights_;  // built during eval callback
    // Only the latest token's attention is kept (previous entries discarded)
    struct TokenAttn {
        std::vector<std::vector<float>> layer_attn;  // [n_layers][n_kv_at_that_point]
    };
    TokenAttn attn_latest_;
    bool attn_latest_valid_ = false;

    // Incremental attention aggregates, updated in inference thread, O(n_ctx) each
    std::vector<float> attn_agg_ema_;        // EMA (α=0.3) across all layers
    std::vector<float> attn_agg_max_;        // max across all layers and steps
    std::vector<float> attn_agg_final_ema_;  // EMA over final layer only
    static constexpr int kAttnRecentWindow = 8;
    std::vector<std::vector<float>> attn_recent_ring_;  // ring buffer of last N layer-averaged vectors
    int attn_recent_ring_idx_ = 0;
    int attn_agg_gen_count_ = 0;             // how many generation steps contributed
    void update_attn_aggregates(const std::vector<std::vector<float>>& layer_attn);
    std::atomic<bool> attn_map_dirty_{false};
    TokenAttn cached_attn_;
    bool cached_attn_valid_ = false;
    int cached_attn_n_lay_ = 0;
    int cached_attn_n_kv_ = 0;

    // Attention focus timeline: [n_layers][n_gen_tokens] = max attention weight per layer per step
    std::vector<std::vector<float>> attn_focus_timeline_;
    std::atomic<bool> attn_focus_dirty_{false};
    std::vector<std::vector<float>> cached_attn_focus_;
    int cached_attn_focus_n_lay_ = 0;
    int cached_attn_focus_n_gen_ = 0;

    void draw_attn_tape(const char* imgui_id, const char* title, const char* description,
                        const std::vector<std::vector<float>>& layer_data,
                        int n_layers, int n_kv, bool auto_scroll,
                        bool relative_scale = false);

    OllamaModelStore ollama_store_;

    // Embedded Ollama-compatible API server
    OllamaServer ollama_server_{this};
    int server_port_ = 11435;
};
