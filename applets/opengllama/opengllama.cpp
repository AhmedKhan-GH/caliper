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
// Vertical text helper — renders text rotated 90° CW, reading bottom to top
// ============================================================================

static void DrawTextVertical(ImDrawList* dl, ImVec2 pos, ImU32 col, const char* text) {
    ImFont* font = ImGui::GetFont();
    float font_size = ImGui::GetFontSize();
    ImFontBaked* baked = font->GetFontBaked(font_size);
    float scale = font_size / baked->Size;

    dl->PushTexture(font->OwnerAtlas->TexRef);

    const char* s = text;
    while (*s) {
        unsigned int c = (unsigned char)*s++;
        ImFontGlyph* glyph = baked->FindGlyph((ImWchar)c);
        if (!glyph) continue;

        if (glyph->Visible) {
            dl->PrimReserve(6, 4);
            dl->PrimQuadUV(
                ImVec2(pos.x + glyph->Y0 * scale, pos.y - glyph->X0 * scale),
                ImVec2(pos.x + glyph->Y1 * scale, pos.y - glyph->X0 * scale),
                ImVec2(pos.x + glyph->Y1 * scale, pos.y - glyph->X1 * scale),
                ImVec2(pos.x + glyph->Y0 * scale, pos.y - glyph->X1 * scale),
                ImVec2(glyph->U0, glyph->V0),
                ImVec2(glyph->U0, glyph->V1),
                ImVec2(glyph->U1, glyph->V1),
                ImVec2(glyph->U1, glyph->V0),
                col);
        }

        pos.y -= glyph->AdvanceX * scale;
    }

    dl->PopTexture();
}

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
            std::lock_guard<std::mutex> lk(output_mutex_);
            if (load_error_msg_.empty())
                load_error_msg_ = "Failed to load " + loading_model_name_ + " — check terminal for details";
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
        int ag = attn_agg_gen_count_;
        int pa = (int)pending_attn_weights_.size();
        int cm = (int)context_map_.size();
        if (ag > 0) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f),
                "[attn OK] gen_steps=%d latest_layers=%d ctx_map=%d",
                ag, (int)attn_latest_.layer_attn.size(), cm);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                "[attn EMPTY] gen_steps=0 pending=%d ctx_map=%d — kq_soft_max not captured?", pa, cm);
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
            float text_h = std::min(ImGui::GetContentRegionAvail().y * 0.4f, 200.0f);
            ImGui::BeginChild("##text_output", ImVec2(0, text_h), ImGuiChildFlags_Borders,
                ImGuiWindowFlags_AlwaysVerticalScrollbar);
            ImGui::TextWrapped("%s", text_snap.c_str());
            if (inference_running_) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "|");
                if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f)
                    ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();
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
            if (context_map_dirty_.exchange(false)) {
                std::lock_guard<std::mutex> lk(output_mutex_);
                cached_cmap_ = context_map_;
                cached_cmap_n_layers_ = (int)context_map_.size();
                cached_cmap_n_ctx_ = 0;
                for (auto& row : context_map_)
                    cached_cmap_n_ctx_ = std::max(cached_cmap_n_ctx_, (int)row.size());
                update_context_map_texture(cached_cmap_);
            }

            int n_layers = cached_cmap_n_layers_;
            int n_ctx = cached_cmap_n_ctx_;

            if (n_layers > 0 && n_ctx > 0) {
                ImGui::Spacing();
                ImGui::SeparatorText("Context Activation Map");
                ImGui::TextDisabled(
                    "Layers (top=0, bottom=%d) vs context tokens — bright = high activation",
                    n_layers - 1);

                float map_w = ImGui::GetContentRegionAvail().x;
                float aspect = (float)n_layers / (float)n_ctx;
                float map_h = std::clamp(map_w * aspect, 60.0f, 300.0f);

                ImGui::BeginChild("##ctx_actmap_port", ImVec2(0, map_h + 8.0f), ImGuiChildFlags_Borders);

                if (context_map_texture_) {
                    ImVec2 img_pos = ImGui::GetCursorScreenPos();
                    float draw_w = ImGui::GetContentRegionAvail().x;
                    float draw_h = std::clamp(draw_w * aspect, 60.0f, 300.0f);
                    ImGui::Image((ImTextureID)(intptr_t)context_map_texture_,
                                 ImVec2(draw_w, draw_h));

                    if (ImGui::IsItemHovered()) {
                        ImVec2 mouse = ImGui::GetMousePos();
                        int tok_idx = (int)((mouse.x - img_pos.x) / draw_w * n_ctx);
                        int lay_idx = (int)((mouse.y - img_pos.y) / draw_h * n_layers);
                        tok_idx = std::clamp(tok_idx, 0, n_ctx - 1);
                        lay_idx = std::clamp(lay_idx, 0, n_layers - 1);

                        ImGui::BeginTooltip();
                        {
                            std::lock_guard<std::mutex> lk(output_mutex_);
                            if (tok_idx < (int)context_tokens_.size())
                                ImGui::Text("Token %d: \"%s\"", tok_idx, context_tokens_[tok_idx].c_str());
                            ImGui::Text("Layer %d", lay_idx);
                            if (lay_idx < (int)context_map_.size() && tok_idx < (int)context_map_[lay_idx].size())
                                ImGui::Text("Activation norm: %.4f", context_map_[lay_idx][tok_idx]);
                        }
                        ImGui::EndTooltip();
                    }
                }
                ImGui::EndChild();
            }

            // ── Context Text Heatmap (pre-computed attention aggregates) ──
            {
                std::vector<float> agg_snap;
                std::vector<std::string> ctok_th;
                int n_gen;
                {
                    std::lock_guard<std::mutex> lk(output_mutex_);
                    n_gen = attn_agg_gen_count_;
                    ctok_th = context_tokens_;
                    // Snapshot the selected aggregate — O(n_ctx) copy
                    if (ctx_text_heatmap_mode_ == THM_EMA)
                        agg_snap = attn_agg_ema_;
                    else if (ctx_text_heatmap_mode_ == THM_MAX)
                        agg_snap = attn_agg_max_;
                    else if (ctx_text_heatmap_mode_ == THM_FINAL_LAYER)
                        agg_snap = attn_agg_final_ema_;
                    else if (ctx_text_heatmap_mode_ == THM_RECENT) {
                        // Compute mean from ring buffer
                        int filled = std::min(attn_recent_ring_idx_, kAttnRecentWindow);
                        if (filled > 0) {
                            int max_k = 0;
                            for (int r = 0; r < filled; ++r)
                                max_k = std::max(max_k, (int)attn_recent_ring_[r].size());
                            agg_snap.resize(max_k, 0.0f);
                            for (int r = 0; r < filled; ++r)
                                for (int k = 0; k < (int)attn_recent_ring_[r].size(); ++k)
                                    agg_snap[k] += attn_recent_ring_[r][k];
                            for (int k = 0; k < max_k; ++k)
                                agg_snap[k] /= (float)filled;
                        }
                    }
                }

                int n_tok = (int)ctok_th.size();

                if (n_gen > 0 && n_tok > 1 && !agg_snap.empty()) {
                    ImGui::Spacing();

                    static const char* thm_labels[] = {
                        "EMA (decay)", "Max", "Recent (last 8)", "Final Layer" };
                    ImGui::SetNextItemWidth(160.0f);
                    ImGui::Combo("##thm_mode", &ctx_text_heatmap_mode_, thm_labels, 4);

                    float wrap_width = ImGui::GetContentRegionAvail().x;
                    float line_h = ImGui::GetFontSize();

                    bool mode_changed = (ctx_text_heatmap_mode_ != ctx_text_heatmap_prev_mode_);
                    bool need_rebuild = mode_changed
                        || (n_tok != ctx_text_heatmap_n_ctx_)
                        || (n_gen != ctx_text_heatmap_n_gen_)
                        || (std::abs(wrap_width - ctx_text_heatmap_last_width_) > 1.0f);

                    if (need_rebuild) {
                        ctx_text_heatmap_n_ctx_ = n_tok;
                        ctx_text_heatmap_n_gen_ = n_gen;
                        ctx_text_heatmap_last_width_ = wrap_width;
                        ctx_text_heatmap_prev_mode_ = ctx_text_heatmap_mode_;

                        int n_agg = (int)agg_snap.size();

                        // Percentile normalization, skip BOS
                        std::vector<float> sort_vals;
                        sort_vals.reserve(n_agg);
                        for (int c = 1; c < n_agg; ++c)
                            sort_vals.push_back(agg_snap[c]);
                        float pmin_t = 0.0f, pmax_t = 1.0f;
                        if (!sort_vals.empty()) {
                            std::sort(sort_vals.begin(), sort_vals.end());
                            int lo = (int)(sort_vals.size() * 0.02f);
                            int hi = std::min((int)(sort_vals.size() * 0.98f), (int)sort_vals.size() - 1);
                            pmin_t = sort_vals[lo];
                            pmax_t = sort_vals[hi];
                        }
                        float range_t = (pmax_t - pmin_t) > 1e-7f ? (pmax_t - pmin_t) : 1.0f;

                        std::vector<float> tok_norm(n_tok, 0.0f);
                        for (int c = 1; c < n_tok && c < n_agg; ++c)
                            tok_norm[c] = std::clamp((agg_snap[c] - pmin_t) / range_t, 0.0f, 1.0f);

                        // Paragraph layout honoring newlines
                        ctx_text_layout_.clear();
                        float cx = 0.0f, cy = 0.0f;
                        for (int i = 1; i < n_tok && i < (int)ctok_th.size(); ++i) {
                            const std::string& tok = ctok_th[i];

                            if (tok.find('\n') != std::string::npos) {
                                std::string part;
                                for (char ch : tok) {
                                    if (ch == '\n') {
                                        if (!part.empty()) {
                                            float tw = ImGui::CalcTextSize(part.c_str()).x;
                                            if (cx + tw > wrap_width && cx > 0.0f) {
                                                cx = 0.0f;
                                                cy += line_h;
                                            }
                                            ctx_text_layout_.push_back({i, cx, cy, tw, part});
                                            cx += tw;
                                        }
                                        cx = 0.0f;
                                        cy += line_h;
                                        part.clear();
                                    } else {
                                        part += ch;
                                    }
                                }
                                if (!part.empty()) {
                                    float tw = ImGui::CalcTextSize(part.c_str()).x;
                                    ctx_text_layout_.push_back({i, cx, cy, tw, part});
                                    cx += tw;
                                }
                                continue;
                            }

                            float tw = ImGui::CalcTextSize(tok.c_str()).x;
                            if (cx + tw > wrap_width && cx > 0.0f) {
                                cx = 0.0f;
                                cy += line_h;
                            }
                            ctx_text_layout_.push_back({i, cx, cy, tw, tok});
                            cx += tw;
                        }
                        ctx_text_total_h_ = cy + line_h;

                        // Bake heatmap texture — reuse pixel buffer
                        int tex_w = std::max(1, (int)wrap_width);
                        int tex_h = std::max(1, (int)ctx_text_total_h_);
                        size_t needed = (size_t)tex_w * tex_h * 4;
                        ctx_text_heatmap_pixels_.resize(needed);
                        std::memset(ctx_text_heatmap_pixels_.data(), 0, needed);

                        auto& pixels = ctx_text_heatmap_pixels_;
                        for (auto& tl : ctx_text_layout_) {
                            float norm = tok_norm[tl.token_idx];

                            float rf, gf, bf;
                            if (norm < 0.5f) {
                                float s = norm / 0.5f;
                                rf = 0.06f + 0.44f * s;
                                gf = 0.06f + 0.10f * s;
                                bf = 0.30f + 0.10f * s;
                            } else {
                                float s = (norm - 0.5f) / 0.5f;
                                rf = 0.50f + 0.45f * s;
                                gf = 0.16f + 0.24f * s;
                                bf = 0.40f - 0.35f * s;
                            }
                            unsigned char pr = (unsigned char)(rf * 255);
                            unsigned char pg = (unsigned char)(gf * 255);
                            unsigned char pb = (unsigned char)(bf * 255);

                            int x0 = std::max(0, (int)tl.x);
                            int x1 = std::min(tex_w, (int)(tl.x + tl.w));
                            int y0 = std::max(0, (int)tl.y);
                            int y1 = std::min(tex_h, (int)(tl.y + line_h));
                            for (int py = y0; py < y1; ++py) {
                                for (int px = x0; px < x1; ++px) {
                                    int idx = (py * tex_w + px) * 4;
                                    pixels[idx + 0] = pr;
                                    pixels[idx + 1] = pg;
                                    pixels[idx + 2] = pb;
                                    pixels[idx + 3] = 216;
                                }
                            }
                        }

                        if (ctx_text_heatmap_tex_)
                            glDeleteTextures(1, &ctx_text_heatmap_tex_);
                        GLuint tex;
                        glGenTextures(1, &tex);
                        glBindTexture(GL_TEXTURE_2D, tex);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_w, tex_h, 0,
                                     GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                        glBindTexture(GL_TEXTURE_2D, 0);
                        ctx_text_heatmap_tex_ = tex;
                        ctx_text_heatmap_tex_w_ = tex_w;
                        ctx_text_heatmap_tex_h_ = tex_h;
                    }

                    if (ctx_text_heatmap_tex_ && !ctx_text_layout_.empty()) {
                        ImVec2 origin = ImGui::GetCursorScreenPos();
                        ImGui::Image((ImTextureID)(intptr_t)ctx_text_heatmap_tex_,
                                     ImVec2((float)ctx_text_heatmap_tex_w_, (float)ctx_text_heatmap_tex_h_));

                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        ImU32 txt_col = ImGui::GetColorU32(ImGuiCol_Text);
                        for (auto& tl : ctx_text_layout_) {
                            dl->AddText(
                                ImVec2(origin.x + tl.x, origin.y + tl.y),
                                txt_col, tl.text.c_str());
                        }
                    }
                }
            }
        }

        // ── Decision Crystallization (Logit Lens) ──
        {
            std::vector<LogitLensEntry> lens_snap;
            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                lens_snap = logit_lens_entries_;
            }

            int nl = (int)lens_snap.size();
            if (nl > 1) {
                std::vector<double> xs(nl), ys(nl);
                for (int l = 0; l < nl; ++l) {
                    xs[l] = l;
                    ys[l] = lens_snap[l].cosine_to_final;
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
                        ImGui::Text("Layer %d", lens_snap[idx].layer);
                        ImGui::Text("Cosine to final: %.4f", cos);
                        ImGui::Text("Status: %s", label);
                        ImGui::EndTooltip();
                    }

                    ImPlot::EndPlot();
                }
            }
        }

        // ── Live Attention — where the model is looking right now ──
        {
            if (attn_map_dirty_.exchange(false)) {
                std::lock_guard<std::mutex> lk(output_mutex_);
                cached_attn_ = attn_latest_;
                cached_attn_valid_ = attn_latest_valid_;
                cached_attn_n_lay_ = (int)attn_latest_.layer_attn.size();
                cached_attn_n_kv_ = 0;
                for (auto& row : attn_latest_.layer_attn)
                    cached_attn_n_kv_ = std::max(cached_attn_n_kv_, (int)row.size());
            }

            if (!cached_attn_valid_ || cached_attn_.layer_attn.empty()) {
                ImGui::Spacing();
                ImGui::SeparatorText("Live Attention");
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                    "Waiting for kq_soft_max data...");
            } else {
                draw_attn_tape("live_attn", "Live Attention",
                    "Where the current token attends — bright = high attention weight",
                    cached_attn_.layer_attn, cached_attn_n_lay_, cached_attn_n_kv_, true);
            }
        }

        // ── Attention Focus — per-layer certainty across generated tokens ──
        {
            if (attn_focus_dirty_.exchange(false)) {
                std::lock_guard<std::mutex> lk(output_mutex_);
                cached_attn_focus_ = attn_focus_timeline_;
                cached_attn_focus_n_lay_ = (int)attn_focus_timeline_.size();
                cached_attn_focus_n_gen_ = 0;
                for (auto& row : attn_focus_timeline_)
                    cached_attn_focus_n_gen_ = std::max(cached_attn_focus_n_gen_, (int)row.size());
            }

            if (cached_attn_focus_n_lay_ > 0 && cached_attn_focus_n_gen_ > 0) {
                draw_attn_tape("attn_focus", "Attention Focus",
                    "Max attention weight per layer per token — bright = focused, dark = diffuse",
                    cached_attn_focus_, cached_attn_focus_n_lay_, cached_attn_focus_n_gen_, true);
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
// Reusable Attention Tape (layers x KV-positions heatmap)
// ============================================================================

void OpenGllamaApplet::draw_attn_tape(const char* imgui_id, const char* title,
                                       const char* description,
                                       const std::vector<std::vector<float>>& layer_data,
                                       int n_layers, int n_kv, bool auto_scroll) {
    ImGui::Spacing();
    ImGui::SeparatorText(title);
    ImGui::TextDisabled("%s", description);

    if (n_layers == 0 || n_kv == 0) {
        ImGui::TextDisabled("Waiting for attention data...");
        return;
    }

    float avail_w = ImGui::GetContentRegionAvail().x;
    float label_margin = 30.0f;
    float chart_w = avail_w - label_margin;
    float cell_w = 6.0f;
    float cell_h = std::max(3.0f, std::min(10.0f, 200.0f / (float)n_layers));
    float total_w = cell_w * n_kv;
    float total_h = cell_h * n_layers;

    // Build child window ID strings from imgui_id
    char port_id[64];
    char scroll_id[64];
    snprintf(port_id, sizeof(port_id), "##%s_port", imgui_id);
    snprintf(scroll_id, sizeof(scroll_id), "##%s_scroll", imgui_id);

    float port_h = std::min(total_h + 24.0f, 280.0f);
    ImGui::BeginChild(port_id, ImVec2(0, port_h), ImGuiChildFlags_Borders,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);

    // Layer labels on the left
    ImVec2 label_origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl_labels = ImGui::GetWindowDrawList();
    int label_skip = std::max(1, n_layers / 8);
    for (int l = 0; l < n_layers; l += label_skip) {
        float y = label_origin.y + l * cell_h + cell_h * 0.5f - 5.0f;
        char lbl[16];
        snprintf(lbl, sizeof(lbl), "%d", l);
        dl_labels->AddText(ImVec2(label_origin.x, y), IM_COL32(160, 160, 160, 200), lbl);
    }

    // Scrollable chart area (offset by label margin)
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + label_margin);
    ImGui::BeginChild(scroll_id, ImVec2(chart_w - 16.0f, total_h + 20.0f),
        ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);

    if (auto_scroll && inference_running_)
        ImGui::SetScrollX(std::max(0.0f, total_w - chart_w));

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Per-row normalization: find max per layer (skip BOS at index 0)
    std::vector<float> row_max(n_layers, 0.0f);
    for (int l = 0; l < n_layers; ++l) {
        if (l >= (int)layer_data.size()) continue;
        const auto& row = layer_data[l];
        for (int k = 1; k < n_kv && k < (int)row.size(); ++k) {
            if (row[k] > row_max[l]) row_max[l] = row[k];
        }
    }

    for (int k = 0; k < n_kv; ++k) {
        for (int l = 0; l < n_layers; ++l) {
            float val = (l < (int)layer_data.size() && k < (int)layer_data[l].size())
                ? layer_data[l][k] : 0.0f;

            float norm = (row_max[l] > 1e-9f) ? std::clamp(val / row_max[l], 0.0f, 1.0f) : 0.0f;

            // 4-segment color ramp: dark blue -> cyan -> yellow-green -> white
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

            float x = origin.x + k * cell_w;
            float y = origin.y + l * cell_h;
            ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.95f));
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cell_w - 0.5f, y + cell_h - 0.5f), col);
        }
    }

    ImGui::Dummy(ImVec2(total_w, total_h));

    if (ImGui::IsItemHovered()) {
        ImVec2 mouse = ImGui::GetMousePos();
        int tok_idx = (int)((mouse.x - origin.x) / cell_w);
        int lay_idx = (int)((mouse.y - origin.y) / cell_h);
        tok_idx = std::clamp(tok_idx, 0, n_kv - 1);
        lay_idx = std::clamp(lay_idx, 0, n_layers - 1);

        float raw_val = (lay_idx < (int)layer_data.size() &&
                         tok_idx < (int)layer_data[lay_idx].size())
            ? layer_data[lay_idx][tok_idx] : 0.0f;

        ImGui::BeginTooltip();
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            if (tok_idx < (int)context_tokens_.size())
                ImGui::Text("KV %d: \"%s\"", tok_idx, context_tokens_[tok_idx].c_str());
            else
                ImGui::Text("KV %d", tok_idx);
        }
        ImGui::Text("Layer %d", lay_idx);
        ImGui::Text("Attention: %.6f", raw_val);
        ImGui::EndTooltip();
    }

    ImGui::EndChild();
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

void OpenGllamaApplet::update_attn_aggregates(const std::vector<std::vector<float>>& layer_attn) {
    int n_layers = (int)layer_attn.size();
    if (n_layers == 0) return;

    int n_kv = 0;
    for (auto& lv : layer_attn)
        n_kv = std::max(n_kv, (int)lv.size());
    if (n_kv == 0) return;

    // Layer-averaged attention for this step
    std::vector<float> step_avg(n_kv, 0.0f);
    for (auto& lv : layer_attn)
        for (int k = 0; k < (int)lv.size(); ++k)
            step_avg[k] += lv[k];
    for (int k = 0; k < n_kv; ++k)
        step_avg[k] /= (float)n_layers;

    // Grow aggregates if context expanded
    if ((int)attn_agg_ema_.size() < n_kv) attn_agg_ema_.resize(n_kv, 0.0f);
    if ((int)attn_agg_max_.size() < n_kv) attn_agg_max_.resize(n_kv, 0.0f);
    if ((int)attn_agg_final_ema_.size() < n_kv) attn_agg_final_ema_.resize(n_kv, 0.0f);

    // EMA across all layers
    float alpha = 0.3f;
    bool first = (attn_agg_gen_count_ == 0);
    for (int k = 0; k < n_kv; ++k) {
        if (first)
            attn_agg_ema_[k] = step_avg[k];
        else
            attn_agg_ema_[k] = alpha * step_avg[k] + (1.0f - alpha) * attn_agg_ema_[k];
    }

    // Max
    for (auto& lv : layer_attn)
        for (int k = 0; k < (int)lv.size(); ++k)
            attn_agg_max_[k] = std::max(attn_agg_max_[k], lv[k]);

    // Final layer EMA
    auto& final_lv = layer_attn.back();
    for (int k = 0; k < (int)final_lv.size(); ++k) {
        if (first)
            attn_agg_final_ema_[k] = final_lv[k];
        else
            attn_agg_final_ema_[k] = alpha * final_lv[k] + (1.0f - alpha) * attn_agg_final_ema_[k];
    }

    // Recent ring buffer
    if ((int)attn_recent_ring_.size() < kAttnRecentWindow)
        attn_recent_ring_.resize(kAttnRecentWindow);
    attn_recent_ring_[attn_recent_ring_idx_ % kAttnRecentWindow] = std::move(step_avg);
    ++attn_recent_ring_idx_;
    ++attn_agg_gen_count_;
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
    size_t needed = (size_t)n_layers * n_ctx * 3;
    ctx_map_pixels_.resize(needed);
    std::memset(ctx_map_pixels_.data(), 0, needed);
    auto& pixels = ctx_map_pixels_;
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
        try {
            unload_model();

            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = n_gpu_layers_;
            model_params.progress_callback = [](float progress, void* user_data) -> bool {
                auto* self = static_cast<OpenGllamaApplet*>(user_data);
                self->load_progress_.store(progress);
                return true;
            };
            model_params.progress_callback_user_data = this;

            std::fprintf(stderr, "[opengllama] loading model from: %s\n", path.c_str());
            model_ = llama_model_load_from_file(path.c_str(), model_params);
            if (!model_) {
                std::fprintf(stderr, "[opengllama] llama_model_load_from_file failed\n");
                std::lock_guard<std::mutex> lk(output_mutex_);
                load_error_msg_ = "Failed to load GGUF file — check stderr for llama.cpp errors";
                load_success_ = false;
                load_finished_ = true;
                return;
            }

            char model_desc[128] = {};
            llama_model_desc(model_, model_desc, sizeof(model_desc));
            int n_layers = llama_model_n_layer(model_);
            int n_embd = llama_model_n_embd(model_);
            std::fprintf(stderr, "[opengllama] model loaded: desc='%s' layers=%d embd=%d\n",
                model_desc, n_layers, n_embd);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = context_size_;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            ctx_params.cb_eval = eval_callback;
            ctx_params.cb_eval_user_data = this;

            std::fprintf(stderr, "[opengllama] creating context: n_ctx=%d flash_attn=disabled\n", context_size_);
            ctx_ = llama_init_from_model(model_, ctx_params);
            if (!ctx_) {
                std::fprintf(stderr, "[opengllama] llama_init_from_model failed (context creation)\n");
                std::lock_guard<std::mutex> lk(output_mutex_);
                load_error_msg_ = std::string("Context creation failed for '") +
                    model_desc +
                    "' — check stderr for details (may need more VRAM or unsupported op)";
                llama_model_free(model_);
                model_ = nullptr;
                load_success_ = false;
                load_finished_ = true;
                return;
            }

            std::fprintf(stderr, "[opengllama] model + context ready\n");
            model_path_ = path;
            load_success_ = true;
            load_finished_ = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[opengllama] exception during load: %s\n", e.what());
            std::lock_guard<std::mutex> lk(output_mutex_);
            load_error_msg_ = std::string("Exception: ") + e.what();
            if (model_) { llama_model_free(model_); model_ = nullptr; }
            load_success_ = false;
            load_finished_ = true;
        }
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
    attn_latest_.layer_attn.clear();
    attn_latest_valid_ = false;
    attn_agg_ema_.clear();
    attn_agg_max_.clear();
    attn_agg_final_ema_.clear();
    attn_recent_ring_.clear();
    attn_recent_ring_idx_ = 0;
    attn_agg_gen_count_ = 0;
    logit_lens_entries_.clear();
    cached_cmap_.clear();
    cached_cmap_n_layers_ = 0;
    cached_cmap_n_ctx_ = 0;
    cached_attn_.layer_attn.clear();
    cached_attn_valid_ = false;
    cached_attn_n_lay_ = 0;
    cached_attn_n_kv_ = 0;
    context_map_dirty_ = false;
    attn_map_dirty_ = false;
    attn_focus_timeline_.clear();
    attn_focus_dirty_ = false;
    cached_attn_focus_.clear();
    cached_attn_focus_n_lay_ = 0;
    cached_attn_focus_n_gen_ = 0;
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
        logit_lens_entries_.clear();
        attn_latest_.layer_attn.clear();
        attn_latest_valid_ = false;
        attn_agg_ema_.clear();
        attn_agg_max_.clear();
        attn_agg_final_ema_.clear();
        attn_recent_ring_.clear();
        attn_recent_ring_idx_ = 0;
        attn_agg_gen_count_ = 0;
        cached_cmap_.clear();
        cached_cmap_n_layers_ = 0;
        cached_cmap_n_ctx_ = 0;
        cached_attn_.layer_attn.clear();
        cached_attn_valid_ = false;
        cached_attn_n_lay_ = 0;
        cached_attn_n_kv_ = 0;
        context_map_dirty_ = false;
        attn_map_dirty_ = false;
        attn_focus_timeline_.clear();
        attn_focus_dirty_ = false;
        cached_attn_focus_.clear();
        cached_attn_focus_n_lay_ = 0;
        cached_attn_focus_n_gen_ = 0;
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

        // Push prompt activations + attention + timeline
        {
            // Compute logit lens before stripping full_hidden
            std::vector<LogitLensEntry> lens;
            {
                std::vector<const LayerActivation*> l_outs;
                for (auto& a : pending_activations_)
                    if (a.name == "l_out" && !a.full_hidden.empty())
                        l_outs.push_back(&a);
                if (l_outs.size() > 1) {
                    auto& fh = l_outs.back()->full_hidden;
                    float fn2 = 0; for (float v : fh) fn2 += v * v;
                    float fn = std::sqrt(fn2);
                    for (auto* la : l_outs) {
                        auto& h = la->full_hidden;
                        int len = std::min((int)h.size(), (int)fh.size());
                        float dot = 0, n2 = 0;
                        for (int j = 0; j < len; ++j) { dot += h[j] * fh[j]; n2 += h[j] * h[j]; }
                        float denom = std::sqrt(n2) * fn;
                        lens.push_back({la->layer_index, denom > 1e-8f ? dot / denom : 0.0f});
                    }
                }
            }
            for (auto& a : pending_activations_)
                a.full_hidden.clear();

            std::lock_guard<std::mutex> lk(output_mutex_);
            activations_ = std::move(pending_activations_);
            logit_lens_entries_ = std::move(lens);
            if (!pending_attn_weights_.empty()) {
                update_attn_aggregates(pending_attn_weights_);
                attn_latest_.layer_attn = pending_attn_weights_;
                attn_latest_valid_ = true;
                attn_map_dirty_ = true;
                {
                    int nl = (int)pending_attn_weights_.size();
                    while ((int)attn_focus_timeline_.size() < nl)
                        attn_focus_timeline_.push_back({});
                    for (int l = 0; l < nl; ++l) {
                        float mx = 0.0f;
                        for (float v : pending_attn_weights_[l])
                            mx = std::max(mx, v);
                        attn_focus_timeline_[l].push_back(mx);
                    }
                }
                attn_focus_dirty_ = true;
            }
            context_map_dirty_ = true;
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
                // Compute logit lens before stripping full_hidden
                std::vector<LogitLensEntry> lens;
                {
                    std::vector<const LayerActivation*> l_outs;
                    for (auto& a : pending_activations_)
                        if (a.name == "l_out" && !a.full_hidden.empty())
                            l_outs.push_back(&a);
                    if (l_outs.size() > 1) {
                        auto& fh = l_outs.back()->full_hidden;
                        float fn2 = 0; for (float v : fh) fn2 += v * v;
                        float fn = std::sqrt(fn2);
                        for (auto* la : l_outs) {
                            auto& h = la->full_hidden;
                            int len = std::min((int)h.size(), (int)fh.size());
                            float dot = 0, n2 = 0;
                            for (int j = 0; j < len; ++j) { dot += h[j] * fh[j]; n2 += h[j] * h[j]; }
                            float denom = std::sqrt(n2) * fn;
                            lens.push_back({la->layer_index, denom > 1e-8f ? dot / denom : 0.0f});
                        }
                    }
                }
                for (auto& a : pending_activations_)
                    a.full_hidden.clear();

                std::lock_guard<std::mutex> lk(output_mutex_);
                output_text_ += piece;
                tokens_generated_.store(i + 1);
                token_logits_.push_back(tli);
                context_tokens_.push_back(piece);
                activations_ = std::move(pending_activations_);
                logit_lens_entries_ = std::move(lens);
                if (!pending_attn_weights_.empty()) {
                    update_attn_aggregates(pending_attn_weights_);
                    attn_latest_.layer_attn = pending_attn_weights_;
                    attn_latest_valid_ = true;
                    attn_map_dirty_ = true;
                    {
                        int nl = (int)pending_attn_weights_.size();
                        while ((int)attn_focus_timeline_.size() < nl)
                            attn_focus_timeline_.push_back({});
                        for (int l = 0; l < nl; ++l) {
                            float mx = 0.0f;
                            for (float v : pending_attn_weights_[l])
                                mx = std::max(mx, v);
                            attn_focus_timeline_[l].push_back(mx);
                        }
                    }
                    attn_focus_dirty_ = true;
                }
                context_map_dirty_ = true;
                textures_dirty_ = true;
            }
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        inference_running_ = false;
        inference_finished_ = true;
    });
}
