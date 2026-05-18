#include "opengllama.h"

#include <imgui.h>
#include <implot.h>
#include <ImGuiFileDialog.h>
#include <GL/glew.h>
#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

// ============================================================================
// Eval callback — intercepts layer output tensors during graph evaluation
// ============================================================================

bool OpenGllamaApplet::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* self = static_cast<OpenGllamaApplet*>(user_data);
    const char* name = t->name;

    bool is_layer_out = (strncmp(name, "l_out-", 6) == 0);
    bool is_attn_out  = (strncmp(name, "attn_out-", 9) == 0);
    bool is_kq_soft   = (strncmp(name, "kq_soft_max-", 12) == 0);
    bool want = is_layer_out || is_attn_out || is_kq_soft;

    if (ask) {
        static int ask_log_count = 0;
        if (ask_log_count < 200) {
            std::fprintf(stderr, "[eval_cb] ask tensor: '%s' want=%d\n", name, (int)want);
            ++ask_log_count;
        }
        return want;
    }
    if (!want) return true;

    int layer = atoi(name + (is_layer_out ? 6 : (is_kq_soft ? 12 : 9)));

    // Attention weights: kq_soft_max shape is [n_kv, n_tokens_q, n_heads] or similar
    // Average across heads, take last query token row → per-KV-position attention
    if (is_kq_soft) {
        int64_t n_elem = ggml_nelements(t);
        int n_kv    = (int)t->ne[0];
        int n_q     = (int)t->ne[1];
        int n_heads = (int)t->ne[2];

        std::vector<float> buf(n_elem);
        ggml_backend_tensor_get(t, buf.data(), 0, n_elem * sizeof(float));

        while ((int)self->pending_attn_weights_.size() <= layer)
            self->pending_attn_weights_.push_back({});

        // Head-averaged attention for the last query token
        std::vector<float> avg(n_kv, 0.0f);
        int last_q = n_q - 1;
        for (int h = 0; h < n_heads; ++h) {
            int head_offset = h * n_q * n_kv;
            for (int k = 0; k < n_kv; ++k)
                avg[k] += buf[head_offset + last_q * n_kv + k];
        }
        float inv_h = 1.0f / (float)n_heads;
        for (int k = 0; k < n_kv; ++k) avg[k] *= inv_h;

        self->pending_attn_weights_[layer] = std::move(avg);
        if (layer == 0)
            std::fprintf(stderr, "[attn] captured kq_soft_max: n_heads=%d n_kv=%d n_q=%d layers_so_far=%d\n",
                n_heads, n_kv, n_q, (int)self->pending_attn_weights_.size());
        return true;
    }

    int64_t n_elem = ggml_nelements(t);
    int rows = (int)t->ne[1];
    int cols = (int)t->ne[0];
    if (rows < 1) rows = 1;

    int vis_cols = std::min(cols, 256);
    int vis_rows = std::min(rows, 1);  // single-token: just first row
    int stride = std::max(1, cols / vis_cols);

    std::vector<float> buf(n_elem);
    ggml_backend_tensor_get(t, buf.data(), 0, n_elem * sizeof(float));

    LayerActivation act;
    act.layer_index = layer;
    act.name = is_layer_out ? "l_out" : "attn_out";
    act.rows = vis_rows;
    act.cols = vis_cols;
    act.values.resize(vis_rows * vis_cols);

    float sum = 0.0f, sq_sum = 0.0f, mx = -1e30f;
    for (int r = 0; r < vis_rows; ++r) {
        for (int c = 0; c < vis_cols; ++c) {
            float v = buf[r * cols + c * stride];
            act.values[r * vis_cols + c] = v;
            sum += std::abs(v);
            sq_sum += v * v;
            mx = std::max(mx, std::abs(v));
        }
    }
    int n = vis_rows * vis_cols;
    act.mean = sum / (float)n;
    act.norm = std::sqrt(sq_sum / (float)n);
    act.max_val = mx;

    // Capture full hidden state for logit lens (l_out only, last token row)
    act.cosine_final = 0.0f;
    if (is_layer_out) {
        int last_row = rows - 1;
        act.full_hidden.resize(cols);
        for (int c = 0; c < cols; ++c)
            act.full_hidden[c] = buf[last_row * cols + c];
    }

    // Cosine similarity with previous l_out
    act.cosine_prev = 0.0f;
    if (is_layer_out && !self->pending_activations_.empty()) {
        for (int k = (int)self->pending_activations_.size() - 1; k >= 0; --k) {
            if (self->pending_activations_[k].name == "l_out") {
                auto& prev = self->pending_activations_[k].values;
                int len = std::min((int)prev.size(), (int)act.values.size());
                float dot = 0, na = 0, nb = 0;
                for (int i = 0; i < len; ++i) {
                    dot += prev[i] * act.values[i];
                    na += prev[i] * prev[i];
                    nb += act.values[i] * act.values[i];
                }
                float denom = std::sqrt(na) * std::sqrt(nb);
                act.cosine_prev = denom > 1e-8f ? dot / denom : 0.0f;
                break;
            }
        }
    }

    self->pending_activations_.push_back(std::move(act));

    // Context map: per-token norm at this layer (only l_out)
    if (is_layer_out) {
        // Ensure context_map_ has enough layers
        while ((int)self->context_map_.size() <= layer)
            self->context_map_.push_back({});

        // Compute L2 norm for each token position (row)
        for (int r = 0; r < rows; ++r) {
            float sq = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float v = buf[r * cols + c];
                sq += v * v;
            }
            self->context_map_[layer].push_back(std::sqrt(sq / (float)cols));
        }
    }

    return true;
}

// ============================================================================
// Applet lifecycle
// ============================================================================

OpenGllamaApplet::OpenGllamaApplet() = default;

OpenGllamaApplet::~OpenGllamaApplet() { cleanup(); }

bool OpenGllamaApplet::initialize() {
    llama_backend_init();
    return true;
}

void OpenGllamaApplet::cleanup() {
    inference_running_ = false;
    if (inference_thread_.joinable()) inference_thread_.join();
    if (load_thread_.joinable()) load_thread_.join();
    unload_model();

    for (auto tex : layer_textures_)
        if (tex) glDeleteTextures(1, &tex);
    layer_textures_.clear();
    if (context_map_texture_) glDeleteTextures(1, &context_map_texture_);
    context_map_texture_ = 0;
    if (attn_map_texture_) glDeleteTextures(1, &attn_map_texture_);
    attn_map_texture_ = 0;

    llama_backend_free();
}

// ============================================================================
// Main UI
// ============================================================================

void OpenGllamaApplet::draw_ui(int /*width*/, int /*height*/) {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##OpenGllamaRoot", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    if (!loading_model_name_.empty() && !load_finished_) {
        float progress = load_progress_.load();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
            "Loading %s ...", loading_model_name_.c_str());
        ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0),
            (std::to_string((int)(progress * 100)) + "%%").c_str());
        ImGui::TextDisabled("Mapping model to GPU memory...");
    } else if (load_finished_) {
        bool ok = load_success_.load();
        if (load_thread_.joinable()) load_thread_.join();
        load_finished_ = false;
        if (ok) {
            model_loaded_ = true;
            load_error_msg_.clear();
        } else {
            load_error_msg_ = "Failed to load " + loading_model_name_ +
                " — architecture may not be supported by llama.cpp";
        }
        loading_model_name_.clear();
    } else if (!model_loaded_) {
        draw_ollama_models();
    } else {
        draw_inference_view();
    }

    ImVec2 min_sz(600, 400);
    ImVec2 max_sz(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display("ChooseGGUF",
            ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            load_model(path);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::End();
}

// ============================================================================
// Model Selection
// ============================================================================

void OpenGllamaApplet::draw_ollama_models() {
    ImGui::SeparatorText("Ollama Models");

    static char path_buf[512];
    static bool path_buf_init = false;
    if (!path_buf_init) {
        std::strncpy(path_buf, ollama_store_.ollama_path().c_str(), sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
        path_buf_init = true;
    }

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 160.0f);
    if (ImGui::InputText("##ollama_path", path_buf, sizeof(path_buf),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        ollama_store_.set_ollama_path(path_buf);
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply & Scan", ImVec2(150, 0))) {
        ollama_store_.set_ollama_path(path_buf);
    }

    ImGui::Spacing();

    if (!load_error_msg_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", load_error_msg_.c_str());
        ImGui::Spacing();
    }

    const auto& models = ollama_store_.models();

    if (models.empty()) {
        ImGui::TextDisabled("No Ollama models found.");
        ImGui::Spacing();
        if (ImGui::Button("Refresh")) ollama_store_.refresh();
    } else {
        if (ImGui::Button("Refresh")) ollama_store_.refresh();
        ImGui::SameLine();
        ImGui::TextDisabled("(%d models)", (int)models.size());
        ImGui::Spacing();

        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            double gb = (double)m.size_bytes / (1024.0 * 1024.0 * 1024.0);

            ImGui::PushID((int)i);

            if (!loading_model_name_.empty()) {
                ImGui::BeginDisabled();
                ImGui::Button("Loading...");
                ImGui::EndDisabled();
            } else if (ImGui::Button("Load")) {
                load_error_msg_.clear();
                load_model_async(m.blob_path, m.name + ":" + m.tag);
            }
            ImGui::SameLine();
            ImGui::Text("%s:%s  (%.1f GB)", m.name.c_str(), m.tag.c_str(), gb);

            ImGui::PopID();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    IGFD::FileDialogConfig cfg;
    cfg.path = ".";
    cfg.flags = ImGuiFileDialogFlags_Modal;
    if (ImGui::Button("Browse GGUF...")) {
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseGGUF", "Select GGUF Model", ".gguf", cfg);
    }
}

// ============================================================================
// Inference View
// ============================================================================

void OpenGllamaApplet::draw_inference_view() {
    // ── Top bar ──
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Model:");
    ImGui::SameLine();
    ImGui::Text("%s", model_path_.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Unload")) {
        inference_running_ = false;
        if (inference_thread_.joinable()) inference_thread_.join();
        unload_model();
        return;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("| %d layers | %d ctx",
        model_ ? llama_model_n_layer(model_) : 0, context_size_);

    ImGui::Separator();

    // ── Prompt ──
    bool paused = inference_running_ && (inference_mode_ == InferenceMode::Paused);
    float btn_w = inference_running_ ? (paused ? 210.0f : 155.0f) : 80.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - btn_w - 8.0f);

    static char prompt_input[2048] = {};
    bool enter = ImGui::InputText("##prompt", prompt_input, sizeof(prompt_input),
        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    if (inference_running_) {
        if (ImGui::Button("Stop", ImVec2(70, 0)))
            inference_running_ = false;

        ImGui::SameLine();
        bool paused = (inference_mode_ == InferenceMode::Paused);
        if (paused) {
            if (ImGui::Button("Resume", ImVec2(70, 0)))
                inference_mode_ = InferenceMode::Continuous;
            ImGui::SameLine();
            if (ImGui::Button("Step", ImVec2(50, 0)))
                step_requested_ = true;
        } else {
            if (ImGui::Button("Pause", ImVec2(70, 0)))
                inference_mode_ = InferenceMode::Paused;
        }
    } else {
        bool run = ImGui::Button("Run", ImVec2(70, 0));
        if ((enter || run) && prompt_input[0] != '\0') {
            prompt_buf_ = prompt_input;
            inference_mode_ = InferenceMode::Continuous;
            run_inference_async(prompt_buf_);
        }
    }

    if (inference_finished_) {
        if (inference_thread_.joinable()) inference_thread_.join();
        inference_finished_ = false;
    }

    // ── Hyperparameters (collapsible) ──
    if (ImGui::CollapsingHeader("Sampling Parameters")) {
        ImGui::SliderInt("Max Tokens", &max_tokens_, 16, 2048);
        ImGui::SliderFloat("Temperature", &temperature_, 0.0f, 2.0f, "%.2f");
        ImGui::SliderInt("Top-K", &top_k_, 1, 200);
        ImGui::SliderFloat("Top-P", &top_p_, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Min-P", &min_p_, 0.0f, 0.5f, "%.3f");
        ImGui::SliderFloat("Repeat Penalty", &repeat_penalty_, 1.0f, 2.0f, "%.2f");
        ImGui::SliderInt("Repeat Window", &repeat_last_n_, 0, 256);
        int seed_i = (int)seed_;
        if (ImGui::SliderInt("Seed (0=random)", &seed_i, 0, 9999))
            seed_ = (uint32_t)seed_i;
        ImGui::Spacing();
        ImGui::SliderInt("Token Delay (ms)", &token_delay_ms_, 0, 2000);
        ImGui::SameLine();
        ImGui::TextDisabled("(0=full speed)");
    }

    ImGui::Separator();

    // ── Debug: attention capture status ──
    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        int ah = (int)attn_history_.size();
        int pa = (int)pending_attn_weights_.size();
        int cm = (int)context_map_.size();
        if (ah > 0) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f),
                "[attn OK] history=%d layers_last=%d ctx_map=%d",
                ah, (int)attn_history_.back().layer_attn.size(), cm);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                "[attn EMPTY] history=0 pending=%d ctx_map=%d — kq_soft_max not captured?", pa, cm);
        }
    }

    // ── Scrollable content ──
    ImGui::BeginChild("##content", ImVec2(0, 0), ImGuiChildFlags_None,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);
    {
        std::string text_snap;
        std::vector<LayerActivation> act_snap;
        std::vector<TokenLogitInfo> logit_snap;
        int toks;
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            text_snap = output_text_;
            act_snap = activations_;
            logit_snap = token_logits_;
        }
        toks = tokens_generated_.load();

        // ── Output text ──
        if (!text_snap.empty() || inference_running_) {
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "Generating... (%d tokens)", toks);
            } else if (toks > 0) {
                ImGui::TextDisabled("Complete — %d tokens", toks);
            }
            ImGui::TextWrapped("%s", text_snap.c_str());
            if (inference_running_) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "|");
            }
        }

        // ── Token Confidence (ImPlot) ──
        if (!logit_snap.empty()) {
            ImGui::Spacing();
            ImGui::SeparatorText("Token Confidence");

            int n = (int)logit_snap.size();
            float plot_w = ImGui::GetContentRegionAvail().x;

            if (ImPlot::BeginPlot("##token_conf", ImVec2(plot_w, 120.0f),
                    ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
                ImPlot::SetupAxes("Token", "P", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.05, ImPlotCond_Always);

                for (int i = 0; i < n; ++i) {
                    float prob = logit_snap[i].probability;
                    float ent = std::clamp(logit_snap[i].entropy / 6.0f, 0.0f, 1.0f);
                    float r, g, b;
                    if (prob > 0.5f) {
                        r = 0.15f * (1.0f - prob);
                        g = 0.7f + 0.3f * prob;
                        b = 0.23f;
                    } else {
                        r = 0.78f + 0.22f * ent;
                        g = 0.78f * prob + 0.31f * (1.0f - ent);
                        b = 0.15f;
                    }
                    ImU32 fill = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.85f));
                    ImU32 edge = ImGui::ColorConvertFloat4ToU32(ImVec4(r * 0.7f, g * 0.7f, b * 0.7f, 1.0f));
                    double x = i, y = prob;
                    char id[16];
                    snprintf(id, sizeof(id), "##b%d", i);
                    ImPlot::PlotBars(id, &x, &y, 1, 0.8,
                        ImPlotSpec(ImPlotProp_FillColor, fill, ImPlotProp_LineColor, edge));
                }

                if (ImPlot::IsPlotHovered()) {
                    ImPlotPoint mp = ImPlot::GetPlotMousePos();
                    int idx = std::clamp((int)std::round(mp.x), 0, n - 1);
                    ImGui::BeginTooltip();
                    ImGui::Text("Token %d: \"%s\"", idx, logit_snap[idx].token_text.c_str());
                    ImGui::Text("Probability: %.1f%%", logit_snap[idx].probability * 100.0f);
                    ImGui::Text("Entropy: %.2f bits", logit_snap[idx].entropy);
                    if (!logit_snap[idx].top_k.empty()) {
                        ImGui::Separator();
                        for (auto& [tok, p] : logit_snap[idx].top_k)
                            ImGui::Text("  %s: %.1f%%", tok.c_str(), p * 100.0f);
                    }
                    ImGui::EndTooltip();
                }

                ImPlot::EndPlot();
            }
        }

        // ── Context Activation Map ──
        {
            std::vector<std::vector<float>> cmap_snap;
            std::vector<std::string> ctok_snap;
            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                cmap_snap = context_map_;
                ctok_snap = context_tokens_;
            }

            int n_layers = (int)cmap_snap.size();
            int n_ctx = 0;
            for (auto& row : cmap_snap)
                n_ctx = std::max(n_ctx, (int)row.size());

            if (n_layers > 0 && n_ctx > 0) {
                ImGui::Spacing();
                ImGui::SeparatorText("Context Activation Map");
                ImGui::TextDisabled(
                    "Layers (top=0, bottom=%d) vs context tokens — bright = high activation",
                    n_layers - 1);

                update_context_map_texture(cmap_snap);

                if (context_map_texture_) {
                    float map_w = ImGui::GetContentRegionAvail().x;
                    float aspect = (float)n_layers / (float)n_ctx;
                    float map_h = std::clamp(map_w * aspect, 60.0f, 300.0f);

                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    ImGui::Image((ImTextureID)(intptr_t)context_map_texture_,
                                 ImVec2(map_w, map_h));

                    // Hover: show token + layer info
                    if (ImGui::IsItemHovered()) {
                        ImVec2 mouse = ImGui::GetMousePos();
                        int tok_idx = (int)((mouse.x - img_pos.x) / map_w * n_ctx);
                        int lay_idx = (int)((mouse.y - img_pos.y) / map_h * n_layers);
                        tok_idx = std::clamp(tok_idx, 0, n_ctx - 1);
                        lay_idx = std::clamp(lay_idx, 0, n_layers - 1);

                        ImGui::BeginTooltip();
                        if (tok_idx < (int)ctok_snap.size())
                            ImGui::Text("Token %d: \"%s\"", tok_idx, ctok_snap[tok_idx].c_str());
                        ImGui::Text("Layer %d", lay_idx);
                        if (lay_idx < (int)cmap_snap.size() && tok_idx < (int)cmap_snap[lay_idx].size())
                            ImGui::Text("Activation norm: %.4f", cmap_snap[lay_idx][tok_idx]);
                        ImGui::EndTooltip();
                    }

                    // Token labels along bottom (sampled to fit)
                    if (!ctok_snap.empty()) {
                        float cell_w = map_w / (float)n_ctx;
                        int label_skip = std::max(1, (int)(12.0f / cell_w));
                        for (int i = 0; i < n_ctx && i < (int)ctok_snap.size(); i += label_skip) {
                            ImGui::SameLine(i * cell_w);
                            ImGui::TextDisabled("%s", ctok_snap[i].c_str());
                        }
                        ImGui::NewLine();
                    }
                }
            }
        }

        // ── Decision Crystallization (Logit Lens) ──
        {
            std::vector<LayerActivation> lens_snap;
            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                lens_snap = activations_;
            }

            std::vector<const LayerActivation*> l_outs_lens;
            for (auto& a : lens_snap)
                if (a.name == "l_out" && !a.full_hidden.empty())
                    l_outs_lens.push_back(&a);

            if (l_outs_lens.size() > 1) {
                const auto& final_hidden = l_outs_lens.back()->full_hidden;
                float final_norm_sq = 0.0f;
                for (float v : final_hidden) final_norm_sq += v * v;
                float final_norm = std::sqrt(final_norm_sq);

                int nl = (int)l_outs_lens.size();
                std::vector<double> xs(nl), ys(nl);

                for (int l = 0; l < nl; ++l) {
                    auto& h = l_outs_lens[l]->full_hidden;
                    int len = std::min((int)h.size(), (int)final_hidden.size());
                    float dot = 0, na = 0;
                    for (int j = 0; j < len; ++j) {
                        dot += h[j] * final_hidden[j];
                        na += h[j] * h[j];
                    }
                    float denom = std::sqrt(na) * final_norm;
                    float cos = denom > 1e-8f ? dot / denom : 0.0f;
                    xs[l] = l;
                    ys[l] = cos;
                }

                ImGui::Spacing();
                ImGui::SeparatorText("Decision Crystallization (Logit Lens)");
                ImGui::TextDisabled("Cosine similarity to final layer — "
                    "1.0 = prediction locked in, low = still exploring");

                float plot_w = ImGui::GetContentRegionAvail().x;
                if (ImPlot::BeginPlot("##logit_lens", ImVec2(plot_w, 100.0f),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
                    ImPlot::SetupAxes("Layer", "cos(final)",
                        ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -0.05, 1.05, ImPlotCond_Always);

                    for (int l = 0; l < nl; ++l) {
                        float cos = (float)ys[l];
                        float r, g, b;
                        if (cos < 0.3f) {
                            r = 0.2f; g = 0.4f; b = 0.9f;
                        } else if (cos < 0.6f) {
                            float t = (cos - 0.3f) / 0.3f;
                            r = 0.2f * (1 - t); g = 0.4f + 0.5f * t; b = 0.9f - 0.3f * t;
                        } else if (cos < 0.85f) {
                            float t = (cos - 0.6f) / 0.25f;
                            r = 0.1f + 0.3f * t; g = 0.9f; b = 0.6f - 0.4f * t;
                        } else {
                            float t = (cos - 0.85f) / 0.15f;
                            r = 0.4f + 0.6f * t; g = 0.9f; b = 0.2f * (1 - t);
                        }
                        ImU32 fill = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.85f));
                        ImU32 edge = ImGui::ColorConvertFloat4ToU32(ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 1.0f));
                        ImPlot::PlotBars("##ll", &xs[l], &ys[l], 1, 0.8,
                            ImPlotSpec(ImPlotProp_FillColor, fill, ImPlotProp_LineColor, edge));
                    }

                    if (ImPlot::IsPlotHovered()) {
                        ImPlotPoint mp = ImPlot::GetPlotMousePos();
                        int idx = std::clamp((int)std::round(mp.x), 0, nl - 1);
                        float cos = (float)ys[idx];
                        const char* label =
                            cos > 0.9f ? "Prediction locked in" :
                            cos > 0.7f ? "Converging" :
                            cos > 0.4f ? "Forming" : "Still exploring";
                        ImGui::BeginTooltip();
                        ImGui::Text("Layer %d", l_outs_lens[idx]->layer_index);
                        ImGui::Text("Cosine to final: %.4f", cos);
                        ImGui::Text("Status: %s", label);
                        ImGui::EndTooltip();
                    }

                    ImPlot::EndPlot();
                }
            }
        }

        // ── Live Attention Timeline ──
        draw_live_attention_timeline();

        // ── Context Attention (Last Token) ──
        {
            std::vector<TokenAttn> attn_snap;
            std::vector<std::string> ctok2;
            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                attn_snap = attn_history_;
                ctok2 = context_tokens_;
            }

            ImGui::Spacing();
            ImGui::SeparatorText("Context Attention (Last Token)");

            if (attn_snap.empty() || attn_snap.back().layer_attn.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                    "Waiting for kq_soft_max data... (attn_history=%d)",
                    (int)attn_snap.size());
            } else {

                auto& latest = attn_snap.back();
                int n_lay = (int)latest.layer_attn.size();
                int n_kv = 0;
                for (auto& row : latest.layer_attn)
                    n_kv = std::max(n_kv, (int)row.size());

                ImGui::TextDisabled(
                    "Layers (top=0, bottom=%d) vs context tokens — "
                    "bright = high attention from token \"%s\"",
                    n_lay - 1,
                    ctok2.empty() ? "?" : ctok2.back().c_str());

                update_attn_map_texture(attn_snap.back().layer_attn);

                if (attn_map_texture_ && n_kv > 0) {
                    float map_w = ImGui::GetContentRegionAvail().x;
                    float aspect = (float)n_lay / (float)n_kv;
                    float map_h = std::clamp(map_w * aspect, 60.0f, 300.0f);

                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    ImGui::Image((ImTextureID)(intptr_t)attn_map_texture_,
                                 ImVec2(map_w, map_h));

                    if (ImGui::IsItemHovered()) {
                        ImVec2 mouse = ImGui::GetMousePos();
                        int tok_idx = (int)((mouse.x - img_pos.x) / map_w * n_kv);
                        int lay_idx = (int)((mouse.y - img_pos.y) / map_h * n_lay);
                        tok_idx = std::clamp(tok_idx, 0, n_kv - 1);
                        lay_idx = std::clamp(lay_idx, 0, n_lay - 1);

                        ImGui::BeginTooltip();
                        if (tok_idx < (int)ctok2.size())
                            ImGui::Text("Attending to token %d: \"%s\"",
                                tok_idx, ctok2[tok_idx].c_str());
                        ImGui::Text("Layer %d", lay_idx);
                        if (lay_idx < n_lay && tok_idx < (int)latest.layer_attn[lay_idx].size())
                            ImGui::Text("Attention: %.4f", latest.layer_attn[lay_idx][tok_idx]);
                        ImGui::EndTooltip();
                    }

                    // Token labels along bottom
                    if (!ctok2.empty() && n_kv > 0) {
                        float cell_w = map_w / (float)n_kv;
                        int label_skip = std::max(1, (int)(12.0f / cell_w));
                        for (int i = 0; i < n_kv && i < (int)ctok2.size(); i += label_skip) {
                            ImGui::SameLine(i * cell_w);
                            ImGui::TextDisabled("%s", ctok2[i].c_str());
                        }
                        ImGui::NewLine();
                    }
                }
            }
        }

        // ── Layer activation heatmaps ──
        if (!act_snap.empty()) {
            ImGui::Spacing();
            ImGui::SeparatorText("Embedding Flow Through Layers");
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "%d tensors captured (live)", (int)act_snap.size());
            }
            ImGui::Spacing();

            // ── Summary: norm bar + cosine similarity bar ──
            // Filter to just l_out for the summary
            std::vector<const LayerActivation*> l_outs;
            for (auto& a : act_snap)
                if (a.name == "l_out") l_outs.push_back(&a);

            if (!l_outs.empty()) {
                float bar_w = ImGui::GetContentRegionAvail().x;
                ImDrawList* dl = ImGui::GetWindowDrawList();

                // Norm bar
                {
                    float bar_h = 20.0f;
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    float max_norm = 0.0f;
                    for (auto* a : l_outs) max_norm = std::max(max_norm, a->norm);
                    if (max_norm < 1e-7f) max_norm = 1.0f;

                    float cell_w = bar_w / (float)l_outs.size();
                    for (int l = 0; l < (int)l_outs.size(); ++l) {
                        float t = l_outs[l]->norm / max_norm;
                        unsigned char r = (unsigned char)(30 + 225 * t);
                        unsigned char g = (unsigned char)(120 + 80 * (1.0f - t));
                        unsigned char b = (unsigned char)(200 * (1.0f - t));

                        float x = origin.x + l * cell_w;
                        dl->AddRectFilled(
                            ImVec2(x, origin.y + bar_h * (1.0f - t)),
                            ImVec2(x + cell_w - 1.0f, origin.y + bar_h),
                            IM_COL32(r, g, b, 220));

                        if (ImGui::IsMouseHoveringRect(
                                ImVec2(x, origin.y), ImVec2(x + cell_w, origin.y + bar_h))) {
                            ImGui::BeginTooltip();
                            ImGui::Text("Layer %d", l_outs[l]->layer_index);
                            ImGui::Text("RMS norm: %.4f", l_outs[l]->norm);
                            ImGui::Text("Mean |act|: %.4f", l_outs[l]->mean);
                            ImGui::EndTooltip();
                        }
                    }
                    dl->AddRect(origin, ImVec2(origin.x + bar_w, origin.y + bar_h),
                        IM_COL32(80, 80, 80, 120));
                    ImGui::Dummy(ImVec2(bar_w, bar_h + 2.0f));
                }
                ImGui::TextDisabled("Activation norms per layer (brighter = stronger activation)");
                ImGui::Spacing();

                // Cosine similarity bar — shows semantic drift between layers
                {
                    float bar_h = 20.0f;
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    float cell_w = bar_w / (float)l_outs.size();

                    for (int l = 0; l < (int)l_outs.size(); ++l) {
                        float cos = l_outs[l]->cosine_prev;
                        // 1.0 = no change (green), 0.0 = orthogonal (red)
                        float drift = 1.0f - std::clamp(cos, 0.0f, 1.0f);
                        unsigned char r = (unsigned char)(40 + 215 * drift);
                        unsigned char g = (unsigned char)(200 * (1.0f - drift));
                        unsigned char b = 60;

                        float x = origin.x + l * cell_w;
                        float h = bar_h * std::max(drift, 0.05f);
                        dl->AddRectFilled(
                            ImVec2(x, origin.y + bar_h - h),
                            ImVec2(x + cell_w - 1.0f, origin.y + bar_h),
                            IM_COL32(r, g, b, 220));

                        if (ImGui::IsMouseHoveringRect(
                                ImVec2(x, origin.y), ImVec2(x + cell_w, origin.y + bar_h))) {
                            ImGui::BeginTooltip();
                            ImGui::Text("Layer %d -> %d", l_outs[l]->layer_index - 1,
                                l_outs[l]->layer_index);
                            ImGui::Text("Cosine similarity: %.4f", cos);
                            ImGui::Text("Semantic drift: %.1f%%", drift * 100.0f);
                            ImGui::EndTooltip();
                        }
                    }
                    dl->AddRect(origin, ImVec2(origin.x + bar_w, origin.y + bar_h),
                        IM_COL32(80, 80, 80, 120));
                    ImGui::Dummy(ImVec2(bar_w, bar_h + 2.0f));
                }
                ImGui::TextDisabled("Semantic drift between layers (red = meaning changes most — emotion/sentiment processing)");
            }

            ImGui::Spacing();

            update_activation_textures();

            float tile_w = ImGui::GetContentRegionAvail().x;
            ImDrawList* dl = ImGui::GetWindowDrawList();

            for (int l = 0; l < (int)act_snap.size(); ++l) {
                const auto& act = act_snap[l];
                if (act.values.empty()) continue;

                ImGui::PushID(l);

                bool is_attn = (act.name == "attn_out");
                ImVec4 color = is_attn ? ImVec4(1.0f, 0.7f, 0.3f, 1.0f)
                                       : ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                ImGui::TextColored(color, "%s %d", is_attn ? "Attn" : "Layer", act.layer_index);
                ImGui::SameLine();
                if (act.cosine_prev > 0.0f && !is_attn) {
                    ImGui::TextDisabled("cos=%.3f  norm=%.3f",
                        act.cosine_prev, act.norm);
                } else {
                    ImGui::TextDisabled("norm=%.3f  max=%.3f", act.norm, act.max_val);
                }

                if (l < (int)layer_textures_.size() && layer_textures_[l]) {
                    float hm_h = std::clamp((float)act.rows * 2.0f, 16.0f, 80.0f);
                    ImGui::Image((ImTextureID)(intptr_t)layer_textures_[l],
                                 ImVec2(tile_w, hm_h));
                }

                if (l < (int)act_snap.size() - 1) {
                    ImVec2 p = ImGui::GetCursorScreenPos();
                    float cx = p.x + tile_w * 0.5f;
                    dl->AddLine(ImVec2(cx, p.y), ImVec2(cx, p.y + 10.0f),
                                IM_COL32(80, 180, 255, 140), 1.5f);
                    dl->AddTriangleFilled(
                        ImVec2(cx, p.y + 14.0f),
                        ImVec2(cx - 3, p.y + 9.0f),
                        ImVec2(cx + 3, p.y + 9.0f),
                        IM_COL32(80, 180, 255, 140));
                    ImGui::Dummy(ImVec2(tile_w, 16.0f));
                }

                ImGui::PopID();
            }
        } else if (!inference_running_) {
            ImGui::Spacing();
            ImGui::TextDisabled("Enter a prompt and press Run to see activation flow.");
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Live Attention Timeline
// ============================================================================

void OpenGllamaApplet::draw_live_attention_timeline() {
    std::vector<TokenAttn> attn_snap;
    std::vector<std::string> ctok_snap;
    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        attn_snap = attn_history_;
        ctok_snap = context_tokens_;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Live Attention");

    if (attn_snap.empty()) {
        ImGui::TextDisabled("Waiting for attention data...");
        return;
    }

    int n_tokens = (int)attn_snap.size();
    int n_layers = 0;
    for (auto& ta : attn_snap)
        n_layers = std::max(n_layers, (int)ta.layer_attn.size());

    if (n_layers == 0) {
        ImGui::TextDisabled("No layer data captured yet.");
        return;
    }

    // Build timeline: for each token, for each layer, store max attention value
    live_attn_timeline_.resize(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        auto& ta = attn_snap[t];
        live_attn_timeline_[t].resize(n_layers, 0.0f);
        for (int l = 0; l < (int)ta.layer_attn.size(); ++l) {
            float mx = 0.0f;
            for (float v : ta.layer_attn[l])
                mx = std::max(mx, v);
            live_attn_timeline_[t][l] = mx;
        }
    }

    // Flash timer: pulse when new tokens arrive
    if (n_tokens != live_attn_last_count_) {
        live_attn_last_count_ = n_tokens;
        live_attn_flash_timer_ = 1.0f;
    } else {
        live_attn_flash_timer_ = std::max(0.0f, live_attn_flash_timer_ - ImGui::GetIO().DeltaTime * 3.0f);
    }

    ImGui::TextDisabled("Rows=layers (0..%d), columns=tokens (growing right) — bright = peaked attention",
        n_layers - 1);

    float avail_w = ImGui::GetContentRegionAvail().x;
    float cell_w = std::max(3.0f, std::min(12.0f, avail_w / (float)n_tokens));
    float cell_h = std::max(3.0f, std::min(10.0f, 200.0f / (float)n_layers));
    float total_w = cell_w * n_tokens;
    float total_h = cell_h * n_layers;

    // Scrollable region if it grows beyond available width
    ImGui::BeginChild("##live_attn_scroll", ImVec2(avail_w, total_h + 30.0f),
        ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);

    // Auto-scroll to rightmost column
    if (inference_running_)
        ImGui::SetScrollX(std::max(0.0f, total_w - avail_w));

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (int t = 0; t < n_tokens; ++t) {
        bool is_newest = (t == n_tokens - 1);
        float flash = is_newest ? live_attn_flash_timer_ : 0.0f;

        for (int l = 0; l < n_layers; ++l) {
            float val = (t < (int)live_attn_timeline_.size() && l < (int)live_attn_timeline_[t].size())
                ? live_attn_timeline_[t][l] : 0.0f;

            // Color: dark blue → cyan → yellow → white based on attention intensity
            float norm = std::clamp(val, 0.0f, 1.0f);
            float r, g, b;
            if (norm < 0.25f) {
                float s = norm / 0.25f;
                r = 0.05f + 0.05f * s;
                g = 0.05f + 0.15f * s;
                b = 0.2f + 0.4f * s;
            } else if (norm < 0.5f) {
                float s = (norm - 0.25f) / 0.25f;
                r = 0.1f * (1.0f - s);
                g = 0.2f + 0.6f * s;
                b = 0.6f + 0.2f * s;
            } else if (norm < 0.75f) {
                float s = (norm - 0.5f) / 0.25f;
                r = 0.8f * s;
                g = 0.8f + 0.1f * s;
                b = 0.8f - 0.6f * s;
            } else {
                float s = (norm - 0.75f) / 0.25f;
                r = 0.8f + 0.2f * s;
                g = 0.9f + 0.1f * s;
                b = 0.2f + 0.8f * s;
            }

            // Flash boost on newest column
            if (flash > 0.0f) {
                r = std::min(1.0f, r + 0.3f * flash);
                g = std::min(1.0f, g + 0.3f * flash);
                b = std::min(1.0f, b + 0.3f * flash);
            }

            float x = origin.x + t * cell_w;
            float y = origin.y + l * cell_h;
            ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.95f));
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cell_w - 0.5f, y + cell_h - 0.5f), col);
        }
    }

    // Flash border on newest column
    if (live_attn_flash_timer_ > 0.0f && n_tokens > 0) {
        float x = origin.x + (n_tokens - 1) * cell_w;
        ImU32 flash_col = ImGui::ColorConvertFloat4ToU32(
            ImVec4(1.0f, 1.0f, 1.0f, live_attn_flash_timer_ * 0.8f));
        dl->AddRect(ImVec2(x, origin.y),
                    ImVec2(x + cell_w, origin.y + total_h), flash_col, 0.0f, 0, 2.0f);
    }

    // Reserve space for the drawing
    ImGui::Dummy(ImVec2(total_w, total_h));

    // Tooltip on hover
    if (ImGui::IsItemHovered()) {
        ImVec2 mouse = ImGui::GetMousePos();
        int tok_idx = (int)((mouse.x - origin.x) / cell_w);
        int lay_idx = (int)((mouse.y - origin.y) / cell_h);
        tok_idx = std::clamp(tok_idx, 0, n_tokens - 1);
        lay_idx = std::clamp(lay_idx, 0, n_layers - 1);

        float val = (tok_idx < (int)live_attn_timeline_.size() &&
                     lay_idx < (int)live_attn_timeline_[tok_idx].size())
            ? live_attn_timeline_[tok_idx][lay_idx] : 0.0f;

        ImGui::BeginTooltip();
        // Token label: offset by prompt tokens (attn_history starts after prompt decode)
        int prompt_len = (int)ctok_snap.size() - n_tokens;
        int ctx_idx = prompt_len + tok_idx;
        if (ctx_idx >= 0 && ctx_idx < (int)ctok_snap.size())
            ImGui::Text("Token %d: \"%s\"", tok_idx, ctok_snap[ctx_idx].c_str());
        else
            ImGui::Text("Token %d", tok_idx);
        ImGui::Text("Layer %d", lay_idx);
        ImGui::Text("Max attention: %.4f", val);
        ImGui::EndTooltip();
    }

    ImGui::EndChild();
}

// ============================================================================
// Activation Texture Upload
// ============================================================================

void OpenGllamaApplet::update_activation_textures() {
    if (!textures_dirty_) return;

    for (auto tex : layer_textures_)
        if (tex) glDeleteTextures(1, &tex);
    layer_textures_.clear();
    layer_textures_.resize(activations_.size(), 0);

    for (int l = 0; l < (int)activations_.size(); ++l) {
        const auto& act = activations_[l];
        if (act.values.empty()) continue;

        float vmin = *std::min_element(act.values.begin(), act.values.end());
        float vmax = *std::max_element(act.values.begin(), act.values.end());
        float range = (vmax - vmin) > 1e-7f ? (vmax - vmin) : 1.0f;

        std::vector<unsigned char> pixels(act.values.size() * 3);
        for (size_t i = 0; i < act.values.size(); ++i) {
            float norm = (act.values[i] - vmin) / range;
            unsigned char r, g, b;
            if (norm < 0.33f) {
                float t = norm / 0.33f;
                r = (unsigned char)(10 + 50 * t);
                g = (unsigned char)(20 + 180 * t);
                b = (unsigned char)(120 + 135 * t);
            } else if (norm < 0.66f) {
                float t = (norm - 0.33f) / 0.33f;
                r = (unsigned char)(60 + 160 * t);
                g = (unsigned char)(200 + 55 * t);
                b = (unsigned char)(255 - 180 * t);
            } else {
                float t = (norm - 0.66f) / 0.34f;
                r = (unsigned char)(220 + 35 * t);
                g = (unsigned char)(255 - 130 * t);
                b = (unsigned char)(75 - 60 * t);
            }
            pixels[i * 3 + 0] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }

        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, act.cols, act.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        layer_textures_[l] = tex;
    }

    textures_dirty_ = false;
}

void OpenGllamaApplet::update_context_map_texture(const std::vector<std::vector<float>>& cmap) {
    int n_layers = (int)cmap.size();
    if (n_layers == 0) return;
    int n_ctx = 0;
    for (auto& row : cmap)
        n_ctx = std::max(n_ctx, (int)row.size());
    if (n_ctx == 0) return;

    // BOS exclusion: skip token 0 in normalization
    // Percentile normalization (2nd–98th) on non-BOS tokens
    std::vector<float> all_vals;
    for (auto& row : cmap)
        for (int c = 1; c < (int)row.size(); ++c)
            all_vals.push_back(row[c]);

    float pmin = 0.0f, pmax = 1.0f;
    if (!all_vals.empty()) {
        std::sort(all_vals.begin(), all_vals.end());
        int lo = (int)(all_vals.size() * 0.02f);
        int hi = (int)(all_vals.size() * 0.98f);
        hi = std::min(hi, (int)all_vals.size() - 1);
        pmin = all_vals[lo];
        pmax = all_vals[hi];
    }
    float range = (pmax - pmin) > 1e-7f ? (pmax - pmin) : 1.0f;

    // Build RGB pixels [n_layers rows × n_ctx cols], skip BOS visually (col 0 = black)
    std::vector<unsigned char> pixels(n_layers * n_ctx * 3, 0);
    for (int l = 0; l < n_layers; ++l) {
        for (int c = 0; c < (int)cmap[l].size(); ++c) {
            if (c == 0) continue;  // BOS excluded
            float norm = std::clamp((cmap[l][c] - pmin) / range, 0.0f, 1.0f);
            unsigned char r, g, b;
            if (norm < 0.5f) {
                float t = norm / 0.5f;
                r = (unsigned char)(10 * t);
                g = (unsigned char)(20 + 100 * t);
                b = (unsigned char)(80 + 175 * t);
            } else {
                float t = (norm - 0.5f) / 0.5f;
                r = (unsigned char)(10 + 245 * t);
                g = (unsigned char)(120 + 135 * t);
                b = (unsigned char)(255 - 200 * t);
            }
            int idx = (l * n_ctx + c) * 3;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    if (context_map_texture_)
        glDeleteTextures(1, &context_map_texture_);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, n_ctx, n_layers, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    context_map_texture_ = tex;
}

void OpenGllamaApplet::update_attn_map_texture(const std::vector<std::vector<float>>& layer_attn) {
    int n_layers = (int)layer_attn.size();
    if (n_layers == 0) return;
    int n_kv = 0;
    for (auto& row : layer_attn)
        n_kv = std::max(n_kv, (int)row.size());
    if (n_kv == 0) return;

    // Per-row normalization: each layer's max maps to 1.0, BOS excluded
    std::vector<unsigned char> pixels(n_layers * n_kv * 3, 0);
    for (int l = 0; l < n_layers; ++l) {
        auto& row = layer_attn[l];
        float row_max = 0.0f;
        for (int c = 1; c < (int)row.size(); ++c)
            row_max = std::max(row_max, row[c]);
        if (row_max < 1e-8f) row_max = 1.0f;

        for (int c = 0; c < (int)row.size(); ++c) {
            if (c == 0) continue;  // BOS excluded
            float norm = std::clamp(row[c] / row_max, 0.0f, 1.0f);
            // Color ramp: dark purple → blue → cyan → yellow → white
            unsigned char r, g, b;
            if (norm < 0.25f) {
                float t = norm / 0.25f;
                r = (unsigned char)(30 + 10 * t);
                g = (unsigned char)(10 + 20 * t);
                b = (unsigned char)(60 + 140 * t);
            } else if (norm < 0.5f) {
                float t = (norm - 0.25f) / 0.25f;
                r = (unsigned char)(40 * (1 - t));
                g = (unsigned char)(30 + 200 * t);
                b = (unsigned char)(200 + 55 * t);
            } else if (norm < 0.75f) {
                float t = (norm - 0.5f) / 0.25f;
                r = (unsigned char)(220 * t);
                g = (unsigned char)(230 + 25 * t);
                b = (unsigned char)(255 - 200 * t);
            } else {
                float t = (norm - 0.75f) / 0.25f;
                r = (unsigned char)(220 + 35 * t);
                g = 255;
                b = (unsigned char)(55 + 200 * t);
            }
            int idx = (l * n_kv + c) * 3;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    if (attn_map_texture_)
        glDeleteTextures(1, &attn_map_texture_);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, n_kv, n_layers, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    attn_map_texture_ = tex;
}

// ============================================================================
// Logit Lens Computation
// ============================================================================

void OpenGllamaApplet::compute_logit_lens() {
    layer_predictions_.clear();
    std::vector<const LayerActivation*> l_outs;
    for (auto& a : activations_)
        if (a.name == "l_out" && !a.full_hidden.empty())
            l_outs.push_back(&a);
    if (l_outs.size() < 2) return;

    const auto& final_h = l_outs.back()->full_hidden;
    float fn2 = 0.0f;
    for (float v : final_h) fn2 += v * v;
    float fn = std::sqrt(fn2);

    for (auto* la : l_outs) {
        auto& h = la->full_hidden;
        int len = std::min((int)h.size(), (int)final_h.size());
        float dot = 0, n2 = 0;
        for (int i = 0; i < len; ++i) {
            dot += h[i] * final_h[i];
            n2 += h[i] * h[i];
        }
        float denom = std::sqrt(n2) * fn;
        LayerPrediction lp;
        lp.layer = la->layer_index;
        lp.cosine_to_final = denom > 1e-8f ? dot / denom : 0.0f;
        layer_predictions_.push_back(lp);
    }
}

// ============================================================================
// Model Loading
// ============================================================================

void OpenGllamaApplet::load_model_async(const std::string& path, const std::string& display_name) {
    if (load_thread_.joinable()) load_thread_.join();

    load_progress_ = 0.0f;
    load_finished_ = false;
    load_success_ = false;
    loading_model_name_ = display_name;

    load_thread_ = std::thread([this, path]() {
        unload_model();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers_;
        model_params.progress_callback = [](float progress, void* user_data) -> bool {
            auto* self = static_cast<OpenGllamaApplet*>(user_data);
            self->load_progress_.store(progress);
            return true;
        };
        model_params.progress_callback_user_data = this;

        model_ = llama_model_load_from_file(path.c_str(), model_params);
        if (!model_) {
            load_success_ = false;
            load_finished_ = true;
            return;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = context_size_;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.cb_eval = eval_callback;
        ctx_params.cb_eval_user_data = this;

        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            llama_model_free(model_);
            model_ = nullptr;
            load_success_ = false;
            load_finished_ = true;
            return;
        }

        model_path_ = path;
        load_success_ = true;
        load_finished_ = true;
    });
}

bool OpenGllamaApplet::load_model(const std::string& path) {
    unload_model();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers_;

    model_ = llama_model_load_from_file(path.c_str(), model_params);
    if (!model_) return false;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size_;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.cb_eval = eval_callback;
    ctx_params.cb_eval_user_data = this;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    model_loaded_ = true;
    model_path_ = path;
    return true;
}

void OpenGllamaApplet::unload_model() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    model_loaded_ = false;
    activations_.clear();
    pending_activations_.clear();
    pending_attn_weights_.clear();
    token_logits_.clear();
    attn_history_.clear();
    layer_predictions_.clear();
    output_text_.clear();
    tokens_generated_ = 0;
}

// ============================================================================
// Inference (async, streaming with real activations)
// ============================================================================

void OpenGllamaApplet::run_inference_async(const std::string& prompt) {
    if (inference_thread_.joinable()) inference_thread_.join();

    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        output_text_.clear();
        activations_.clear();
        token_logits_.clear();
        attn_history_.clear();
    }
    tokens_generated_ = 0;
    inference_running_ = true;
    inference_finished_ = false;

    inference_thread_ = std::thread([this, prompt]() {
        if (!model_ || !ctx_) {
            std::lock_guard<std::mutex> lk(output_mutex_);
            output_text_ = "ERROR: no model loaded";
            inference_running_ = false;
            inference_finished_ = true;
            return;
        }

        llama_memory_clear(llama_get_memory(ctx_), true);

        const llama_vocab* vocab = llama_model_get_vocab(model_);
        int n_layers = llama_model_n_layer(model_);

        // Clear context map for new inference
        context_map_.clear();
        context_map_.resize(n_layers);
        context_tokens_.clear();

        std::vector<llama_token> tokens(prompt.size() + 8);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        }
        tokens.resize(n_tokens);

        // Store prompt token texts for context map labels
        for (int i = 0; i < n_tokens; ++i) {
            char piece[64] = {};
            llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, false);
            context_tokens_.push_back(piece);
        }

        llama_batch batch = llama_batch_init(std::max(n_tokens, 1), 0, 1);
        batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }

        pending_activations_.clear();
        pending_attn_weights_.clear();

        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            std::lock_guard<std::mutex> lk(output_mutex_);
            output_text_ = "ERROR: decode failed on prompt";
            inference_running_ = false;
            inference_finished_ = true;
            return;
        }

        // Push prompt activations + attention
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            activations_ = pending_activations_;
            if (!pending_attn_weights_.empty()) {
                TokenAttn ta;
                ta.layer_attn = pending_attn_weights_;
                attn_history_.push_back(std::move(ta));
            }
            textures_dirty_ = true;
        }

        int n_vocab = llama_vocab_n_tokens(vocab);
        int n_gen = max_tokens_;

        // Build sampler chain with user hyperparams
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler* smpl = llama_sampler_chain_init(sparams);

        if (repeat_penalty_ > 1.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_penalties(repeat_last_n_, repeat_penalty_, 0.0f, 0.0f));
        }
        if (top_k_ > 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k_));
        if (top_p_ < 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p_, 1));
        if (min_p_ > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p_, 1));
        if (temperature_ > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature_));
        else
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        uint32_t s = seed_ == 0 ? (uint32_t)time(nullptr) : seed_;
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(s));

        for (int i = 0; i < n_gen; ++i) {
            if (!inference_running_) break;

            // Playback control: pause/step
            while (inference_running_ &&
                   inference_mode_ == InferenceMode::Paused &&
                   !step_requested_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            step_requested_ = false;
            if (!inference_running_) break;

            // Speed control delay
            if (token_delay_ms_ > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));

            float* logits = llama_get_logits_ith(ctx_, -1);

            // Compute softmax for visualization (before sampling modifies anything)
            float max_logit = *std::max_element(logits, logits + n_vocab);
            double sum_exp = 0.0;
            for (int t = 0; t < n_vocab; ++t)
                sum_exp += std::exp((double)(logits[t] - max_logit));

            // Sample using the chain
            llama_token best = llama_sampler_sample(smpl, ctx_, -1);
            float best_prob = (float)(std::exp((double)(logits[best] - max_logit)) / sum_exp);

            // Entropy
            float entropy = 0.0f;
            for (int t = 0; t < n_vocab; ++t) {
                float p = (float)(std::exp((double)(logits[t] - max_logit)) / sum_exp);
                if (p > 1e-10f) entropy -= p * std::log2(p);
            }

            // Top-5 for tooltip
            std::vector<std::pair<float, int>> scored(n_vocab);
            for (int t = 0; t < n_vocab; ++t)
                scored[t] = {logits[t], t};
            std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                [](auto& a, auto& b) { return a.first > b.first; });

            if (llama_vocab_is_eog(vocab, best)) break;

            char piece[64] = {};
            llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);

            TokenLogitInfo tli;
            tli.token_text = piece;
            tli.probability = best_prob;
            tli.entropy = entropy;
            for (int k = 0; k < 5 && k < n_vocab; ++k) {
                char kpiece[64] = {};
                llama_token_to_piece(vocab, scored[k].second, kpiece, sizeof(kpiece), 0, false);
                float kprob = (float)(std::exp((double)(scored[k].first - max_logit)) / sum_exp);
                tli.top_k.push_back({kpiece, kprob});
            }

            llama_sampler_accept(smpl, best);

            pending_activations_.clear();
            pending_attn_weights_.clear();

            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = n_tokens + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx_, batch) != 0) break;

            std::fprintf(stderr, "[inference] token %d: pending_attn=%d pending_act=%d\n",
                i, (int)pending_attn_weights_.size(), (int)pending_activations_.size());

            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                output_text_ += piece;
                tokens_generated_.store(i + 1);
                token_logits_.push_back(tli);
                context_tokens_.push_back(piece);
                activations_ = pending_activations_;
                if (!pending_attn_weights_.empty()) {
                    TokenAttn ta;
                    ta.layer_attn = pending_attn_weights_;
                    attn_history_.push_back(std::move(ta));
                }
                textures_dirty_ = true;
            }
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        inference_running_ = false;
        inference_finished_ = true;
    });
}
