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
    bool is_kq_soft   = (strncmp(name, "kq_soft_max-", 12) == 0);
    bool want = is_layer_out || is_kq_soft;

    if (ask) return want;
    if (!want) return true;

    int layer = atoi(name + (is_layer_out ? 6 : 12));

    if (is_kq_soft) {
        int64_t n_elem = ggml_nelements(t);
        int n_kv    = (int)t->ne[0];
        int n_q     = (int)t->ne[1];
        int n_heads = (int)t->ne[2];

        std::vector<float> buf(n_elem);
        ggml_backend_tensor_get(t, buf.data(), 0, n_elem * sizeof(float));

        while ((int)self->pending_attn_weights_.size() <= layer)
            self->pending_attn_weights_.push_back({});

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
        return true;
    }

    // l_out: context map (per-token activation norm per layer)
    int rows = (int)t->ne[1];
    int cols = (int)t->ne[0];
    if (rows < 1) rows = 1;

    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));

    while ((int)self->context_map_.size() <= layer)
        self->context_map_.push_back({});

    for (int r = 0; r < rows; ++r) {
        float sq = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float v = buf[r * cols + c];
            sq += v * v;
        }
        self->context_map_[layer].push_back(std::sqrt(sq / (float)cols));
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
    ollama_server_.stop();
    inference_running_ = false;
    if (inference_thread_.joinable()) inference_thread_.join();
    if (load_thread_.joinable()) load_thread_.join();
    unload_model();

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
    {
        const char* display_name = model_path_.c_str();
        size_t slash = model_path_.find_last_of('/');
        if (slash != std::string::npos) display_name = model_path_.c_str() + slash + 1;

        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Model:");
        ImGui::SameLine();
        ImGui::Text("%s", display_name);
        ImGui::SameLine();
        if (ImGui::SmallButton("Unload")) {
            inference_running_ = false;
            if (inference_thread_.joinable()) inference_thread_.join();
            unload_model();
            return;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%d layers | %d ctx",
            model_ ? llama_model_n_layer(model_) : 0, context_size_);
        ImGui::SameLine();
        if (ollama_server_.is_running()) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.6f, 1.0f), "API :%d", ollama_server_.port());
            ImGui::SameLine();
            if (ImGui::SmallButton("Stop Server")) ollama_server_.stop();
        } else {
            ImGui::SetNextItemWidth(50);
            ImGui::InputInt("##port", &server_port_, 0, 0);
            ImGui::SameLine();
            if (ImGui::SmallButton("Serve")) ollama_server_.start(server_port_);
        }
    }

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
            prompt_buf_ = format_chat_prompt(prompt_input);
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
        ImGui::SliderInt("Max Tokens", &max_tokens_, 0, 4096,
                         max_tokens_ == 0 ? "Unlimited" : "%d");
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
        int toks;
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            text_snap = output_text_;
        }
        toks = tokens_generated_.load();

        // ── Text output with optional attention heatmap ──
        {
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "Generating... (%d tokens)", toks);
            } else if (toks > 0) {
                ImGui::TextDisabled("Complete — %d tokens", toks);
            } else {
                ImGui::TextDisabled("Output");
            }

            char recent_label[32];
            snprintf(recent_label, sizeof(recent_label), "Recent (last %d)", attn_recent_window_);
            const char* thm_labels[] = {
                "None", "EMA (decay)", "Max", recent_label, "Final Layer" };
            ImGui::SetNextItemWidth(160.0f);
            ImGui::Combo("##thm_mode", &ctx_text_heatmap_mode_, thm_labels, 5);
            if (ctx_text_heatmap_mode_ == THM_EMA) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                ImGui::SliderFloat("Alpha", &attn_ema_alpha_, 0.01f, 1.0f, "%.2f");
            } else if (ctx_text_heatmap_mode_ == THM_MAX) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                ImGui::DragFloat("Contrast", &ctx_text_heatmap_contrast_, 0.1f, 1.0f, 0.0f, "%.1f");
                if (ctx_text_heatmap_contrast_ < 1.0f) ctx_text_heatmap_contrast_ = 1.0f;
            } else if (ctx_text_heatmap_mode_ == THM_RECENT) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                ImGui::SliderInt("Window", &attn_recent_window_, 1, kAttnRecentRingMax);
            }

            std::vector<float> agg_snap;
            std::vector<std::string> ctok_th;
            int n_gen;
            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                n_gen = attn_agg_gen_count_;
                ctok_th = context_tokens_;
                if (ctx_text_heatmap_mode_ == THM_EMA)
                    agg_snap = attn_agg_ema_;
                else if (ctx_text_heatmap_mode_ == THM_MAX)
                    agg_snap = attn_agg_max_;
                else if (ctx_text_heatmap_mode_ == THM_FINAL_LAYER)
                    agg_snap = attn_agg_final_ema_;
                else if (ctx_text_heatmap_mode_ == THM_RECENT) {
                    int filled = std::min(attn_recent_ring_idx_, attn_recent_window_);
                    if (filled > 0) {
                        int max_k = 0;
                        for (int r = 0; r < filled; ++r) {
                            int idx = ((attn_recent_ring_idx_ - 1 - r) % kAttnRecentRingMax + kAttnRecentRingMax) % kAttnRecentRingMax;
                            max_k = std::max(max_k, (int)attn_recent_ring_[idx].size());
                        }
                        agg_snap.resize(max_k, 0.0f);
                        for (int r = 0; r < filled; ++r) {
                            int idx = ((attn_recent_ring_idx_ - 1 - r) % kAttnRecentRingMax + kAttnRecentRingMax) % kAttnRecentRingMax;
                            for (int k = 0; k < (int)attn_recent_ring_[idx].size(); ++k)
                                agg_snap[k] += attn_recent_ring_[idx][k];
                        }
                        for (int k = 0; k < max_k; ++k)
                            agg_snap[k] /= (float)filled;
                    }
                }
            }

            int n_tok = (int)ctok_th.size();
            bool have_heatmap = ctx_text_heatmap_mode_ != THM_NONE
                && n_gen > 0 && n_tok > 1 && !agg_snap.empty();

            float text_h = std::min(ImGui::GetContentRegionAvail().y * 0.4f, 200.0f);
            ImGui::BeginChild("##text_output", ImVec2(0, text_h), ImGuiChildFlags_Borders,
                ImGuiWindowFlags_AlwaysVerticalScrollbar);

            if (have_heatmap) {
                float wrap_width = ImGui::GetContentRegionAvail().x;
                float line_h = ImGui::GetFontSize();

                bool mode_changed = (ctx_text_heatmap_mode_ != ctx_text_heatmap_prev_mode_);
                bool contrast_changed = (ctx_text_heatmap_contrast_ != ctx_text_heatmap_prev_contrast_);
                bool need_rebuild = mode_changed || contrast_changed
                    || (n_tok != ctx_text_heatmap_n_ctx_)
                    || (n_gen != ctx_text_heatmap_n_gen_)
                    || (std::abs(wrap_width - ctx_text_heatmap_last_width_) > 1.0f);

                if (need_rebuild) {
                    ctx_text_heatmap_n_ctx_ = n_tok;
                    ctx_text_heatmap_n_gen_ = n_gen;
                    ctx_text_heatmap_last_width_ = wrap_width;
                    ctx_text_heatmap_prev_mode_ = ctx_text_heatmap_mode_;
                    ctx_text_heatmap_prev_contrast_ = ctx_text_heatmap_contrast_;

                    int n_agg = (int)agg_snap.size();

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
                    for (int c = 1; c < n_tok && c < n_agg; ++c) {
                        float linear = std::clamp((agg_snap[c] - pmin_t) / range_t, 0.0f, 1.0f);
                        if (ctx_text_heatmap_mode_ == THM_MAX)
                            tok_norm[c] = std::pow(linear, ctx_text_heatmap_contrast_);
                        else
                            tok_norm[c] = std::log1p(linear * 49.0f) / std::log1p(49.0f);
                    }

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

                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                        ImGui::OpenPopup("##copy_heatmap_text");
                    if (ImGui::BeginPopup("##copy_heatmap_text")) {
                        if (ImGui::MenuItem("Copy text"))
                            ImGui::SetClipboardText(text_snap.c_str());
                        ImGui::EndPopup();
                    }

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    ImU32 txt_col = ImGui::GetColorU32(ImGuiCol_Text);
                    for (auto& tl : ctx_text_layout_) {
                        dl->AddText(
                            ImVec2(origin.x + tl.x, origin.y + tl.y),
                            txt_col, tl.text.c_str());
                    }
                }
            } else {
                ImGui::TextWrapped("%s", text_snap.c_str());
                if (inference_running_) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "|");
                }
            }

            if (inference_running_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f)
                ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
        }

        // Cache live attention data (always runs, independent of UI collapse state)
        if (attn_map_dirty_.exchange(false)) {
            std::lock_guard<std::mutex> lk(output_mutex_);
            cached_attn_ = attn_latest_;
            cached_attn_valid_ = attn_latest_valid_;
            cached_attn_n_lay_ = (int)attn_latest_.layer_attn.size();
            cached_attn_n_kv_ = 0;
            for (auto& row : attn_latest_.layer_attn)
                cached_attn_n_kv_ = std::max(cached_attn_n_kv_, (int)row.size());

            bool has_swa = model_ && llama_model_n_swa(model_) > 0;
            cached_live_swa_.clear();
            cached_live_full_.clear();
            cached_live_swa_n_lay_ = 0;
            cached_live_swa_n_kv_ = 0;
            cached_live_full_n_lay_ = 0;
            cached_live_full_n_kv_ = 0;
            if (has_swa && cached_attn_valid_) {
                for (int l = 0; l < cached_attn_n_lay_; ++l) {
                    auto& dest = (l % 2 == 0) ? cached_live_swa_ : cached_live_full_;
                    dest.push_back(cached_attn_.layer_attn[l]);
                }
                cached_live_swa_n_lay_ = (int)cached_live_swa_.size();
                for (auto& row : cached_live_swa_)
                    cached_live_swa_n_kv_ = std::max(cached_live_swa_n_kv_, (int)row.size());
                cached_live_full_n_lay_ = (int)cached_live_full_.size();
                for (auto& row : cached_live_full_)
                    cached_live_full_n_kv_ = std::max(cached_live_full_n_kv_, (int)row.size());
            }

            if (!has_recurrent_layers_ && cached_attn_valid_) {
                for (auto& row : cached_attn_.layer_attn) {
                    if (row.empty()) { has_recurrent_layers_ = true; break; }
                }
                if (has_recurrent_layers_) {
                    attn_layer_mask_.resize(cached_attn_n_lay_);
                    for (int l = 0; l < cached_attn_n_lay_; ++l)
                        attn_layer_mask_[l] = !cached_attn_.layer_attn[l].empty();
                }
            }
            cached_live_attn_only_.clear();
            cached_live_attn_only_n_lay_ = 0;
            cached_live_attn_only_n_kv_ = 0;
            if (has_recurrent_layers_ && cached_attn_valid_) {
                for (int l = 0; l < cached_attn_n_lay_; ++l) {
                    if (l < (int)attn_layer_mask_.size() && attn_layer_mask_[l]) {
                        cached_live_attn_only_.push_back(cached_attn_.layer_attn[l]);
                        cached_live_attn_only_n_kv_ = std::max(cached_live_attn_only_n_kv_, (int)cached_attn_.layer_attn[l].size());
                    }
                }
                cached_live_attn_only_n_lay_ = (int)cached_live_attn_only_.size();
            }
        }

        // Cache focus timeline data (always runs)
        if (attn_focus_dirty_.exchange(false)) {
            std::lock_guard<std::mutex> lk(output_mutex_);
            cached_attn_focus_ = attn_focus_timeline_;
            cached_attn_focus_n_lay_ = (int)attn_focus_timeline_.size();
            cached_attn_focus_n_gen_ = 0;
            for (auto& row : attn_focus_timeline_)
                cached_attn_focus_n_gen_ = std::max(cached_attn_focus_n_gen_, (int)row.size());

            cached_focus_swa_ = attn_focus_swa_;
            cached_focus_swa_n_lay_ = (int)attn_focus_swa_.size();
            cached_focus_swa_n_gen_ = 0;
            for (auto& row : attn_focus_swa_)
                cached_focus_swa_n_gen_ = std::max(cached_focus_swa_n_gen_, (int)row.size());

            cached_focus_full_ = attn_focus_full_;
            cached_focus_full_n_lay_ = (int)attn_focus_full_.size();
            cached_focus_full_n_gen_ = 0;
            for (auto& row : attn_focus_full_)
                cached_focus_full_n_gen_ = std::max(cached_focus_full_n_gen_, (int)row.size());

            cached_focus_attn_only_.clear();
            cached_focus_attn_only_n_lay_ = 0;
            cached_focus_attn_only_n_gen_ = 0;
            if (has_recurrent_layers_) {
                int n = std::min((int)cached_attn_focus_.size(), (int)attn_layer_mask_.size());
                for (int l = 0; l < n; ++l) {
                    if (attn_layer_mask_[l]) {
                        cached_focus_attn_only_.push_back(cached_attn_focus_[l]);
                        cached_focus_attn_only_n_gen_ = std::max(cached_focus_attn_only_n_gen_, (int)cached_attn_focus_[l].size());
                    }
                }
                cached_focus_attn_only_n_lay_ = (int)cached_focus_attn_only_.size();
            }
        }

        // ── Live Attention ──
        {
            bool has_swa = cached_live_swa_n_lay_ > 0;
            ImGui::Spacing();
            ImGui::SeparatorText("Live Attention");
            if (has_swa) {
                const char* live_items[] = { "All Layers", "Sliding Window (even)", "Full Attention (odd)" };
                ImGui::SetNextItemWidth(220);
                ImGui::Combo("##live_view", &live_attn_view_, live_items, 3);
            } else if (has_recurrent_layers_) {
                const char* live_items[] = { "Attention Only", "All Layers" };
                ImGui::SetNextItemWidth(220);
                ImGui::Combo("##live_view", &live_attn_view_, live_items, 2);
            }

            if (live_attn_view_ == 1 && has_swa) {
                draw_attn_tape("live_swa", "Sliding Window Layers",
                    "SWA layers only — local attention pattern",
                    cached_live_swa_, cached_live_swa_n_lay_, cached_live_swa_n_kv_, true, true);
            } else if (live_attn_view_ == 2 && has_swa) {
                draw_attn_tape("live_full", "Full Attention Layers",
                    "Full-context layers only — global attention pattern",
                    cached_live_full_, cached_live_full_n_lay_, cached_live_full_n_kv_, true, true);
            } else if (live_attn_view_ == 0 && has_recurrent_layers_ && cached_live_attn_only_n_lay_ > 0) {
                draw_attn_tape("live_attn", "Attention Layers",
                    "Attention layers only — recurrent layers hidden",
                    cached_live_attn_only_, cached_live_attn_only_n_lay_, cached_live_attn_only_n_kv_, true, true);
            } else {
                if (cached_attn_valid_ && !cached_attn_.layer_attn.empty()) {
                    draw_attn_tape("live_attn", "All Layers",
                        "Where the current token attends — bright = high attention weight",
                        cached_attn_.layer_attn, cached_attn_n_lay_, cached_attn_n_kv_, true, true);
                } else {
                    static const std::vector<std::vector<float>> empty_data;
                    draw_attn_tape("live_attn", "All Layers",
                        "Where the current token attends — bright = high attention weight",
                        empty_data, 0, 0, false, true);
                }
            }
        }

        // ── Attention Focus ──
        {
            bool has_swa = cached_focus_swa_n_lay_ > 0;
            ImGui::Spacing();
            ImGui::SeparatorText("Attention Focus");
            if (has_swa) {
                const char* focus_items[] = { "All Layers", "Sliding Window (even)", "Full Attention (odd)" };
                ImGui::SetNextItemWidth(220);
                ImGui::Combo("##focus_view", &focus_attn_view_, focus_items, 3);
            } else if (has_recurrent_layers_) {
                const char* focus_items[] = { "Attention Only", "All Layers" };
                ImGui::SetNextItemWidth(220);
                ImGui::Combo("##focus_view", &focus_attn_view_, focus_items, 2);
            }

            if (focus_attn_view_ == 1 && has_swa) {
                draw_attn_tape("focus_swa", "Sliding Window Layers",
                    "SWA layers only — local attention with limited context window",
                    cached_focus_swa_, cached_focus_swa_n_lay_, cached_focus_swa_n_gen_, true);
            } else if (focus_attn_view_ == 2 && has_swa) {
                draw_attn_tape("focus_full", "Full Attention Layers",
                    "Full-context attention layers — global dependency tracking",
                    cached_focus_full_, cached_focus_full_n_lay_, cached_focus_full_n_gen_, true);
            } else if (focus_attn_view_ == 0 && has_recurrent_layers_ && cached_focus_attn_only_n_lay_ > 0) {
                draw_attn_tape("attn_focus", "Attention Layers",
                    "Attention layers only — recurrent layers hidden",
                    cached_focus_attn_only_, cached_focus_attn_only_n_lay_, cached_focus_attn_only_n_gen_, true);
            } else {
                if (cached_attn_focus_n_lay_ > 0 && cached_attn_focus_n_gen_ > 0) {
                    draw_attn_tape("attn_focus", "All Layers",
                        "Max attention weight per layer per token — bright = focused, dark = diffuse",
                        cached_attn_focus_, cached_attn_focus_n_lay_, cached_attn_focus_n_gen_, true);
                } else {
                    static const std::vector<std::vector<float>> empty_data;
                    draw_attn_tape("attn_focus", "All Layers",
                        "Max attention weight per layer per token — bright = focused, dark = diffuse",
                        empty_data, 0, 0, false);
                }
            }
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
                                       int n_layers, int n_kv, bool auto_scroll,
                                       bool relative_scale) {
    ImGui::Spacing();
    ImGui::SeparatorText(title);
    ImGui::TextDisabled("%s", description);

    if (n_layers == 0 || n_kv == 0) {
        int model_layers = model_ ? llama_model_n_layer(model_) : 16;
        float cell_h_e = std::max(3.0f, std::min(10.0f, 200.0f / (float)model_layers));
        float total_h_e = cell_h_e * model_layers;
        float scrollbar_h_e = ImGui::GetStyle().ScrollbarSize;
        float port_h_e = total_h_e + scrollbar_h_e + 8.0f;
        char port_id_e[64];
        snprintf(port_id_e, sizeof(port_id_e), "##%s_port", imgui_id);
        ImGui::BeginChild(port_id_e, ImVec2(0, port_h_e), ImGuiChildFlags_Borders,
            ImGuiWindowFlags_NoScrollbar);
        ImGui::EndChild();
        return;
    }

    float avail_w = ImGui::GetContentRegionAvail().x;
    float label_margin = 30.0f;
    float chart_w = avail_w - label_margin;
    float cell_w = 6.0f;
    float cell_h = std::max(3.0f, std::min(10.0f, 200.0f / (float)n_layers));
    int k_start = (n_kv > 1) ? 1 : 0;
    float total_w = cell_w * (n_kv - k_start);
    float total_h = cell_h * n_layers;

    // Build child window ID strings from imgui_id
    char port_id[64];
    char scroll_id[64];
    snprintf(port_id, sizeof(port_id), "##%s_port", imgui_id);
    snprintf(scroll_id, sizeof(scroll_id), "##%s_scroll", imgui_id);

    float scrollbar_h = ImGui::GetStyle().ScrollbarSize;
    float port_h = total_h + scrollbar_h + 8.0f;
    ImGui::BeginChild(port_id, ImVec2(0, port_h), ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

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
    ImGui::BeginChild(scroll_id, ImVec2(chart_w, total_h + scrollbar_h),
        ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);

    if (auto_scroll && inference_running_)
        ImGui::SetScrollX(std::max(0.0f, total_w - chart_w));

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    std::vector<float> row_max;
    if (relative_scale) {
        row_max.resize(n_layers, 0.0f);
        for (int l = 0; l < n_layers; ++l) {
            if (l >= (int)layer_data.size()) continue;
            const auto& row = layer_data[l];
            for (int k = 1; k < n_kv && k < (int)row.size(); ++k)
                if (row[k] > row_max[l]) row_max[l] = row[k];
        }
    }

    for (int k = k_start; k < n_kv; ++k) {
        for (int l = 0; l < n_layers; ++l) {
            float val = (l < (int)layer_data.size() && k < (int)layer_data[l].size())
                ? layer_data[l][k] : 0.0f;

            float norm;
            if (relative_scale) {
                norm = (row_max[l] > 1e-9f) ? std::clamp(val / row_max[l], 0.0f, 1.0f) : 0.0f;
            } else {
                float clamped = std::clamp(val, 0.0f, 1.0f);
                norm = std::log1p(clamped * 49.0f) / std::log1p(49.0f);
            }

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

            float x = origin.x + (k - k_start) * cell_w;
            float y = origin.y + l * cell_h;
            ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 0.95f));
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + cell_w - 0.5f, y + cell_h - 0.5f), col);
        }
    }

    ImGui::Dummy(ImVec2(total_w, total_h));

    if (ImGui::IsItemHovered()) {
        ImVec2 mouse = ImGui::GetMousePos();
        int tok_idx = k_start + (int)((mouse.x - origin.x) / cell_w);
        int lay_idx = (int)((mouse.y - origin.y) / cell_h);
        tok_idx = std::clamp(tok_idx, k_start, n_kv - 1);
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
    float alpha = attn_ema_alpha_;
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
    if ((int)attn_recent_ring_.size() < kAttnRecentRingMax)
        attn_recent_ring_.resize(kAttnRecentRingMax);
    attn_recent_ring_[attn_recent_ring_idx_ % kAttnRecentRingMax] = std::move(step_avg);
    ++attn_recent_ring_idx_;
    ++attn_agg_gen_count_;
}

void OpenGllamaApplet::append_focus_timelines(const std::vector<std::vector<float>>& layer_attn) {
    int nl = (int)layer_attn.size();
    while ((int)attn_focus_timeline_.size() < nl)
        attn_focus_timeline_.push_back({});
    for (int l = 0; l < nl; ++l) {
        float mx = 0.0f;
        for (float v : layer_attn[l])
            mx = std::max(mx, v);
        attn_focus_timeline_[l].push_back(mx);
    }

    bool has_swa = model_ && llama_model_n_swa(model_) > 0;
    if (has_swa) {
        for (int l = 0; l < nl; ++l) {
            float mx = 0.0f;
            for (float v : layer_attn[l])
                mx = std::max(mx, v);
            auto& dest = (l % 2 == 0) ? attn_focus_swa_ : attn_focus_full_;
            int row = l / 2;
            while ((int)dest.size() <= row) dest.push_back({});
            dest[row].push_back(mx);
        }
    }
    attn_focus_dirty_ = true;
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

            active_profile_ = detect_model_profile(model_);
            std::fprintf(stderr, "[opengllama] detected profile: %s (ctx=%d thinking=%s)\n",
                active_profile_.display_name.c_str(),
                active_profile_.context_size,
                active_profile_.supports_thinking ? "yes" : "no");

            context_size_ = active_profile_.context_size;
            temperature_  = active_profile_.temperature;
            top_k_        = active_profile_.top_k;
            top_p_        = active_profile_.top_p;
            min_p_        = active_profile_.min_p;
            repeat_penalty_ = active_profile_.repeat_penalty;
            repeat_last_n_  = active_profile_.repeat_last_n;
            thinking_enabled_ = active_profile_.supports_thinking;

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

    active_profile_ = detect_model_profile(model_);
    context_size_    = active_profile_.context_size;
    temperature_     = active_profile_.temperature;
    top_k_           = active_profile_.top_k;
    top_p_           = active_profile_.top_p;
    min_p_           = active_profile_.min_p;
    repeat_penalty_  = active_profile_.repeat_penalty;
    repeat_last_n_   = active_profile_.repeat_last_n;
    thinking_enabled_ = active_profile_.supports_thinking;

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
    pending_attn_weights_.clear();
    attn_latest_.layer_attn.clear();
    attn_latest_valid_ = false;
    attn_agg_ema_.clear();
    attn_agg_max_.clear();
    attn_agg_final_ema_.clear();
    attn_recent_ring_.clear();
    attn_recent_ring_idx_ = 0;
    attn_agg_gen_count_ = 0;

    cached_attn_.layer_attn.clear();
    cached_attn_valid_ = false;
    cached_attn_n_lay_ = 0;
    cached_attn_n_kv_ = 0;
    context_map_dirty_ = false;
    attn_map_dirty_ = false;
    attn_focus_timeline_.clear();
    attn_focus_swa_.clear();
    attn_focus_full_.clear();
    attn_focus_dirty_ = false;
    cached_attn_focus_.clear();
    cached_attn_focus_n_lay_ = 0;
    cached_attn_focus_n_gen_ = 0;
    cached_focus_swa_.clear();
    cached_focus_full_.clear();
    cached_focus_swa_n_lay_ = 0;
    cached_focus_swa_n_gen_ = 0;
    cached_focus_full_n_lay_ = 0;
    cached_focus_full_n_gen_ = 0;
    has_recurrent_layers_ = false;
    attn_layer_mask_.clear();
    cached_live_attn_only_.clear();
    cached_live_attn_only_n_lay_ = 0;
    cached_live_attn_only_n_kv_ = 0;
    cached_focus_attn_only_.clear();
    cached_focus_attn_only_n_lay_ = 0;
    cached_focus_attn_only_n_gen_ = 0;
    live_attn_view_ = 0;
    focus_attn_view_ = 0;
    output_text_.clear();
    tokens_generated_ = 0;
}

// ============================================================================
// Chat template formatting
// ============================================================================

std::string OpenGllamaApplet::format_chat_prompt(const std::string& user_input) const {
    if (!model_) return user_input;

    const char* tmpl = llama_model_chat_template(model_, nullptr);
    std::string tmpl_fallback;
    if (!tmpl) {
        char desc[256];
        llama_model_desc(model_, desc, sizeof(desc));
        std::string d(desc);
        if (d.find("gptoss") != std::string::npos || d.find("gpt-oss") != std::string::npos)
            tmpl_fallback = "gptoss";
        else
            tmpl_fallback = "chatml";
        tmpl = tmpl_fallback.c_str();
    }

    llama_chat_message sys_msg  = {"system", "You are a helpful assistant."};
    llama_chat_message user_msg = {"user", user_input.c_str()};
    const llama_chat_message msgs[] = { sys_msg, user_msg };

    std::vector<char> buf(user_input.size() * 2 + 512);
    int32_t n = llama_chat_apply_template(tmpl, msgs, 2, true,
                                          buf.data(), (int32_t)buf.size());
    if (n < 0) return user_input;
    if ((size_t)n > buf.size()) {
        buf.resize(n + 1);
        n = llama_chat_apply_template(tmpl, msgs, 2, true,
                                      buf.data(), (int32_t)buf.size());
    }
    return std::string(buf.data(), n);
}

// ============================================================================
// Inference (async, streaming with real activations)
// ============================================================================

void OpenGllamaApplet::run_inference_async(const std::string& prompt) {
    if (inference_thread_.joinable()) inference_thread_.join();

    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        output_text_.clear();
    
        attn_latest_.layer_attn.clear();
        attn_latest_valid_ = false;
        attn_agg_ema_.clear();
        attn_agg_max_.clear();
        attn_agg_final_ema_.clear();
        attn_recent_ring_.clear();
        attn_recent_ring_idx_ = 0;
        attn_agg_gen_count_ = 0;
        cached_attn_.layer_attn.clear();
        cached_attn_valid_ = false;
        cached_attn_n_lay_ = 0;
        cached_attn_n_kv_ = 0;
        context_map_dirty_ = false;
        attn_map_dirty_ = false;
        attn_focus_timeline_.clear();
        attn_focus_swa_.clear();
        attn_focus_full_.clear();
        attn_focus_dirty_ = false;
        cached_attn_focus_.clear();
        cached_attn_focus_n_lay_ = 0;
        cached_attn_focus_n_gen_ = 0;
        cached_focus_swa_.clear();
        cached_focus_full_.clear();
        cached_focus_swa_n_lay_ = 0;
        cached_focus_swa_n_gen_ = 0;
        cached_focus_full_n_lay_ = 0;
        cached_focus_full_n_gen_ = 0;
        cached_live_attn_only_.clear();
        cached_live_attn_only_n_lay_ = 0;
        cached_live_attn_only_n_kv_ = 0;
        cached_focus_attn_only_.clear();
        cached_focus_attn_only_n_lay_ = 0;
        cached_focus_attn_only_n_gen_ = 0;
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
                                      tokens.data(), (int)tokens.size(), true, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, true);
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
            std::lock_guard<std::mutex> lk(output_mutex_);
            if (!pending_attn_weights_.empty()) {
                int actual_ctx = (int)context_tokens_.size();
                for (auto& aw : pending_attn_weights_)
                    if ((int)aw.size() > actual_ctx) aw.resize(actual_ctx);
                update_attn_aggregates(pending_attn_weights_);
                attn_latest_.layer_attn = pending_attn_weights_;
                attn_latest_valid_ = true;
                attn_map_dirty_ = true;
                append_focus_timelines(pending_attn_weights_);
            }
            context_map_dirty_ = true;
        }

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

        int n_ctx = (int)llama_n_ctx(ctx_);
        int n_keep = std::min(n_tokens, n_ctx / 4);
        int pos = n_tokens;
        int gen_count = 0;

        while (inference_running_) {
            if (max_tokens_ > 0 && gen_count >= max_tokens_) break;

            while (inference_running_ &&
                   inference_mode_ == InferenceMode::Paused &&
                   !step_requested_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            step_requested_ = false;
            if (!inference_running_) break;

            if (token_delay_ms_ > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));

            llama_token best = llama_sampler_sample(smpl, ctx_, -1);

            if (llama_vocab_is_eog(vocab, best)) break;

            char piece[64] = {};
            llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);

            llama_sampler_accept(smpl, best);

            pending_attn_weights_.clear();

            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;

            int rc = llama_decode(ctx_, batch);
            if (rc == 1) {
                int n_discard = (pos - n_keep) / 2;
                if (n_discard < 1) break;
                llama_memory_t mem = llama_get_memory(ctx_);
                llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
                llama_memory_seq_add(mem, 0, n_keep + n_discard, -1, -n_discard);
                pos -= n_discard;
                batch.pos[0] = pos;
                rc = llama_decode(ctx_, batch);
            }
            if (rc != 0) break;

            pos++;
            gen_count++;

            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                output_text_ += piece;
                tokens_generated_.store(gen_count);
                context_tokens_.push_back(piece);
                if (!pending_attn_weights_.empty()) {
                    int actual_ctx = (int)context_tokens_.size();
                    for (auto& aw : pending_attn_weights_)
                        if ((int)aw.size() > actual_ctx) aw.resize(actual_ctx);
                    update_attn_aggregates(pending_attn_weights_);
                    attn_latest_.layer_attn = pending_attn_weights_;
                    attn_latest_valid_ = true;
                    attn_map_dirty_ = true;
                    append_focus_timelines(pending_attn_weights_);
                }
                context_map_dirty_ = true;
            }
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        inference_running_ = false;
        inference_finished_ = true;
    });
}

// ============================================================================
// Blocking inference — called from HTTP server thread
// ============================================================================

void OpenGllamaApplet::run_inference_blocking(
        const std::string& prompt,
        const std::function<bool(const std::string&)>& token_cb) {

    if (inference_thread_.joinable()) inference_thread_.join();

    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        output_text_.clear();
        attn_latest_.layer_attn.clear();
        attn_latest_valid_ = false;
        attn_agg_ema_.clear();
        attn_agg_max_.clear();
        attn_agg_final_ema_.clear();
        attn_recent_ring_.clear();
        attn_recent_ring_idx_ = 0;
        attn_agg_gen_count_ = 0;
        cached_attn_.layer_attn.clear();
        cached_attn_valid_ = false;
        cached_attn_n_lay_ = 0;
        cached_attn_n_kv_ = 0;
        context_map_dirty_ = false;
        attn_map_dirty_ = false;
        attn_focus_timeline_.clear();
        attn_focus_swa_.clear();
        attn_focus_full_.clear();
        attn_focus_dirty_ = false;
        cached_attn_focus_.clear();
        cached_attn_focus_n_lay_ = 0;
        cached_attn_focus_n_gen_ = 0;
        cached_focus_swa_.clear();
        cached_focus_full_.clear();
        cached_focus_swa_n_lay_ = 0;
        cached_focus_swa_n_gen_ = 0;
        cached_focus_full_n_lay_ = 0;
        cached_focus_full_n_gen_ = 0;
        cached_live_attn_only_.clear();
        cached_live_attn_only_n_lay_ = 0;
        cached_live_attn_only_n_kv_ = 0;
        cached_focus_attn_only_.clear();
        cached_focus_attn_only_n_lay_ = 0;
        cached_focus_attn_only_n_gen_ = 0;
    }
    tokens_generated_ = 0;
    inference_running_ = true;
    inference_finished_ = false;
    inference_mode_ = InferenceMode::Continuous;

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

    context_map_.clear();
    context_map_.resize(n_layers);
    context_tokens_.clear();

    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                  tokens.data(), (int)tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                  tokens.data(), (int)tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

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

    pending_attn_weights_.clear();

    if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        std::lock_guard<std::mutex> lk(output_mutex_);
        output_text_ = "ERROR: decode failed on prompt";
        inference_running_ = false;
        inference_finished_ = true;
        return;
    }

    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        if (!pending_attn_weights_.empty()) {
            int actual_ctx = (int)context_tokens_.size();
            for (auto& aw : pending_attn_weights_)
                if ((int)aw.size() > actual_ctx) aw.resize(actual_ctx);
            update_attn_aggregates(pending_attn_weights_);
            attn_latest_.layer_attn = pending_attn_weights_;
            attn_latest_valid_ = true;
            attn_map_dirty_ = true;
            append_focus_timelines(pending_attn_weights_);
        }
        context_map_dirty_ = true;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);

    if (repeat_penalty_ > 1.0f)
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(repeat_last_n_, repeat_penalty_, 0.0f, 0.0f));
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

    int n_ctx = (int)llama_n_ctx(ctx_);
    int n_keep = std::min(n_tokens, n_ctx / 4);
    int pos = n_tokens;
    int gen_count = 0;

    while (inference_running_) {
        if (max_tokens_ > 0 && gen_count >= max_tokens_) break;

        llama_token best = llama_sampler_sample(smpl, ctx_, -1);
        if (llama_vocab_is_eog(vocab, best)) break;

        char piece[64] = {};
        llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);

        llama_sampler_accept(smpl, best);

        pending_attn_weights_.clear();

        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        int rc = llama_decode(ctx_, batch);
        if (rc == 1) {
            int n_discard = (pos - n_keep) / 2;
            if (n_discard < 1) break;
            llama_memory_t mem = llama_get_memory(ctx_);
            llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(mem, 0, n_keep + n_discard, -1, -n_discard);
            pos -= n_discard;
            batch.pos[0] = pos;
            rc = llama_decode(ctx_, batch);
        }
        if (rc != 0) break;

        pos++;
        gen_count++;

        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            std::string piece_str(piece);
            output_text_ += piece_str;
            tokens_generated_.store(gen_count);
            context_tokens_.push_back(piece_str);
            if (!pending_attn_weights_.empty()) {
                int actual_ctx = (int)context_tokens_.size();
                for (auto& aw : pending_attn_weights_)
                    if ((int)aw.size() > actual_ctx) aw.resize(actual_ctx);
                update_attn_aggregates(pending_attn_weights_);
                attn_latest_.layer_attn = pending_attn_weights_;
                attn_latest_valid_ = true;
                attn_map_dirty_ = true;
                append_focus_timelines(pending_attn_weights_);
            }
            context_map_dirty_ = true;
        }

        if (!token_cb(std::string(piece))) break;
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    inference_running_ = false;
    inference_finished_ = true;
}
