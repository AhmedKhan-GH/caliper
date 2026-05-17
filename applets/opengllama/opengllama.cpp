#include "opengllama.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <GL/glew.h>
#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

// ============================================================================
// Eval callback — intercepts layer output tensors during graph evaluation
// ============================================================================

bool OpenGllamaApplet::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* self = static_cast<OpenGllamaApplet*>(user_data);
    const char* name = t->name;

    // We want the output of each transformer layer's feed-forward block.
    // In llama.cpp these are named "l_out-N" (layer output after residual add).
    bool is_layer_out = (strncmp(name, "l_out-", 6) == 0);

    if (ask) return is_layer_out;

    if (!is_layer_out) return true;

    int layer = atoi(name + 6);

    int64_t n_elem = ggml_nelements(t);
    int rows = (int)t->ne[1];
    int cols = (int)t->ne[0];
    if (rows < 1) rows = 1;

    // Cap the data we copy — take a representative slice
    int vis_cols = std::min(cols, 256);
    int vis_rows = std::min(rows, 64);
    int stride = std::max(1, cols / vis_cols);

    std::vector<float> buf(n_elem);
    ggml_backend_tensor_get(t, buf.data(), 0, n_elem * sizeof(float));

    LayerActivation act;
    act.layer_index = layer;
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

    self->pending_activations_.push_back(std::move(act));
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
    float btn_w = inference_running_ ? 160.0f : 80.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - btn_w - 8.0f);

    static char prompt_input[2048] = {};
    bool enter = ImGui::InputText("##prompt", prompt_input, sizeof(prompt_input),
        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    if (inference_running_) {
        if (ImGui::Button("Stop", ImVec2(btn_w, 0)))
            inference_running_ = false;
    } else {
        bool run = ImGui::Button("Run", ImVec2(btn_w, 0));
        if ((enter || run) && prompt_input[0] != '\0') {
            prompt_buf_ = prompt_input;
            run_inference_async(prompt_buf_);
        }
    }

    if (inference_finished_) {
        if (inference_thread_.joinable()) inference_thread_.join();
        inference_finished_ = false;
    }

    // ── Hyperparameters (collapsible) ──
    if (ImGui::CollapsingHeader("Sampling Parameters")) {
        ImGui::Columns(4, nullptr, false);
        ImGui::SetColumnWidth(0, 180);
        ImGui::SetColumnWidth(1, 180);
        ImGui::SetColumnWidth(2, 180);
        ImGui::SetColumnWidth(3, 180);

        ImGui::SliderInt("Max Tokens", &max_tokens_, 16, 2048);
        ImGui::NextColumn();
        ImGui::SliderFloat("Temperature", &temperature_, 0.0f, 2.0f, "%.2f");
        ImGui::NextColumn();
        ImGui::SliderInt("Top-K", &top_k_, 1, 200);
        ImGui::NextColumn();
        ImGui::SliderFloat("Top-P", &top_p_, 0.0f, 1.0f, "%.2f");
        ImGui::NextColumn();

        ImGui::SliderFloat("Min-P", &min_p_, 0.0f, 0.5f, "%.3f");
        ImGui::NextColumn();
        ImGui::SliderFloat("Repeat Penalty", &repeat_penalty_, 1.0f, 2.0f, "%.2f");
        ImGui::NextColumn();
        ImGui::SliderInt("Repeat Window", &repeat_last_n_, 0, 256);
        ImGui::NextColumn();
        int seed_i = (int)seed_;
        ImGui::SliderInt("Seed", &seed_i, 0, 9999);
        seed_ = (uint32_t)seed_i;
        ImGui::SameLine();
        ImGui::TextDisabled("(0=random)");
        ImGui::NextColumn();

        ImGui::Columns(1);
    }

    ImGui::Separator();

    // ── Scrollable content ──
    ImGui::BeginChild("##content", ImVec2(0, 0), ImGuiChildFlags_None,
        ImGuiWindowFlags_HorizontalScrollbar);
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

        // ── Token confidence ribbon ──
        if (!logit_snap.empty()) {
            ImGui::Spacing();
            ImGui::SeparatorText("Token Confidence");

            float ribbon_w = ImGui::GetContentRegionAvail().x;
            float cell_w = std::max(4.0f, ribbon_w / (float)logit_snap.size());
            ImVec2 origin = ImGui::GetCursorScreenPos();
            float ribbon_h = 40.0f;
            ImDrawList* dl = ImGui::GetWindowDrawList();

            for (int i = 0; i < (int)logit_snap.size(); ++i) {
                float prob = logit_snap[i].probability;
                float ent = std::clamp(logit_snap[i].entropy / 6.0f, 0.0f, 1.0f);

                // Green = confident, Yellow = uncertain, Red = high entropy
                unsigned char r, g, b;
                if (prob > 0.5f) {
                    r = (unsigned char)(40 * (1.0f - prob));
                    g = (unsigned char)(180 + 75 * prob);
                    b = 60;
                } else {
                    r = (unsigned char)(200 + 55 * ent);
                    g = (unsigned char)(200 * prob + 80 * (1.0f - ent));
                    b = 40;
                }

                float x = origin.x + i * cell_w;
                float h = ribbon_h * std::clamp(prob, 0.05f, 1.0f);
                dl->AddRectFilled(
                    ImVec2(x, origin.y + ribbon_h - h),
                    ImVec2(x + cell_w - 1.0f, origin.y + ribbon_h),
                    IM_COL32(r, g, b, 220));

                // Tooltip on hover
                if (ImGui::IsMouseHoveringRect(
                        ImVec2(x, origin.y), ImVec2(x + cell_w, origin.y + ribbon_h))) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Token %d: \"%s\"", i, logit_snap[i].token_text.c_str());
                    ImGui::Text("Probability: %.1f%%", prob * 100.0f);
                    ImGui::Text("Entropy: %.2f", logit_snap[i].entropy);
                    if (!logit_snap[i].top_k.empty()) {
                        ImGui::Separator();
                        ImGui::Text("Top alternatives:");
                        for (auto& [tok, p] : logit_snap[i].top_k)
                            ImGui::Text("  %s: %.1f%%", tok.c_str(), p * 100.0f);
                    }
                    ImGui::EndTooltip();
                }
            }
            ImGui::Dummy(ImVec2(ribbon_w, ribbon_h + 4.0f));
        }

        // ── Layer activation heatmaps ──
        if (!act_snap.empty()) {
            ImGui::Spacing();
            ImGui::SeparatorText("Layer Activations");
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "%d layers (live)", (int)act_snap.size());
            }
            ImGui::Spacing();

            // ── Summary bar: per-layer activation norm ──
            {
                float bar_w = ImGui::GetContentRegionAvail().x;
                float bar_h = 24.0f;
                ImVec2 bar_origin = ImGui::GetCursorScreenPos();
                ImDrawList* dl = ImGui::GetWindowDrawList();

                float max_norm = 0.0f;
                for (auto& a : act_snap) max_norm = std::max(max_norm, a.norm);
                if (max_norm < 1e-7f) max_norm = 1.0f;

                float cell_w = bar_w / (float)act_snap.size();
                for (int l = 0; l < (int)act_snap.size(); ++l) {
                    float t = act_snap[l].norm / max_norm;
                    unsigned char r = (unsigned char)(30 + 225 * t);
                    unsigned char g = (unsigned char)(120 + 80 * (1.0f - t));
                    unsigned char b = (unsigned char)(200 * (1.0f - t));

                    float x = bar_origin.x + l * cell_w;
                    float h = bar_h * t;
                    dl->AddRectFilled(
                        ImVec2(x, bar_origin.y + bar_h - h),
                        ImVec2(x + cell_w - 1.0f, bar_origin.y + bar_h),
                        IM_COL32(r, g, b, 220));

                    if (ImGui::IsMouseHoveringRect(
                            ImVec2(x, bar_origin.y),
                            ImVec2(x + cell_w, bar_origin.y + bar_h))) {
                        ImGui::BeginTooltip();
                        ImGui::Text("Layer %d", l);
                        ImGui::Text("Mean |act|: %.4f", act_snap[l].mean);
                        ImGui::Text("RMS norm: %.4f", act_snap[l].norm);
                        ImGui::Text("Max |act|: %.4f", act_snap[l].max_val);
                        ImGui::EndTooltip();
                    }
                }
                dl->AddRect(bar_origin,
                    ImVec2(bar_origin.x + bar_w, bar_origin.y + bar_h),
                    IM_COL32(80, 80, 80, 120));
                ImGui::Dummy(ImVec2(bar_w, bar_h + 4.0f));
                ImGui::TextDisabled("Layer activation norms (hover for details)");
            }

            ImGui::Spacing();

            update_activation_textures();

            float tile_w = ImGui::GetContentRegionAvail().x;
            ImDrawList* dl = ImGui::GetWindowDrawList();

            for (int l = 0; l < (int)act_snap.size(); ++l) {
                const auto& act = act_snap[l];
                if (act.values.empty()) continue;

                ImGui::PushID(l);

                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Layer %d", l);
                ImGui::SameLine();
                ImGui::TextDisabled("mean=%.3f  norm=%.3f  max=%.3f",
                    act.mean, act.norm, act.max_val);

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
    token_logits_.clear();
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

        std::vector<llama_token> tokens(prompt.size() + 8);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        }
        tokens.resize(n_tokens);

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

        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            std::lock_guard<std::mutex> lk(output_mutex_);
            output_text_ = "ERROR: decode failed on prompt";
            inference_running_ = false;
            inference_finished_ = true;
            return;
        }

        // Push prompt activations
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            activations_ = pending_activations_;
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

            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = n_tokens + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx_, batch) != 0) break;

            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                output_text_ += piece;
                tokens_generated_.store(i + 1);
                token_logits_.push_back(tli);
                activations_ = pending_activations_;
                textures_dirty_ = true;
            }
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        inference_running_ = false;
        inference_finished_ = true;
    });
}
