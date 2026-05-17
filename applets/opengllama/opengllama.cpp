#include "opengllama.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <GL/glew.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>

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

void OpenGllamaApplet::draw_ui(int width, int height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)width, (float)height));
    ImGui::Begin("OpenGllama", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoBackground);

    // Async model loading — progress bar
    if (!loading_model_name_.empty() && !load_finished_) {
        float progress = load_progress_.load();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
            "Loading %s ...", loading_model_name_.c_str());
        ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0),
            (std::to_string((int)(progress * 100)) + "%%").c_str());
        ImGui::TextDisabled("Mapping model to Metal GPU memory...");
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

    // File dialog
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
// Inference View — prompt on top, output + activations below (vertical)
// ============================================================================

void OpenGllamaApplet::draw_inference_view() {
    // ── Top bar: model info ──
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

    // ── Prompt input ──
    float btn_w = inference_running_ ? 160.0f : 80.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - btn_w - 8.0f);

    static char prompt_input[2048] = {};
    bool enter = ImGui::InputText("##prompt", prompt_input, sizeof(prompt_input),
        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    if (inference_running_) {
        if (ImGui::Button("Stop", ImVec2(btn_w, 0))) {
            inference_running_ = false;
        }
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

    ImGui::Separator();

    // ── Content: output text then activation flow, all vertical ──
    ImGui::BeginChild("##content", ImVec2(0, 0), false);
    {
        // Snapshot shared state
        std::string text_snap;
        std::vector<LayerActivation> act_snap;
        int toks;
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            text_snap = output_text_;
            act_snap = activations_;
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
            ImGui::Spacing();
            ImGui::Separator();
        }

        // ── Activation flow (vertical, full width) ──
        if (act_snap.empty() && !inference_running_) {
            ImGui::TextDisabled("Enter a prompt and press Run to see activation flow.");
        } else if (!act_snap.empty()) {
            ImGui::Spacing();
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "Activation Flow — %d layers (live, token %d)", (int)act_snap.size(), toks);
            } else {
                ImGui::Text("Activation Flow — %d layers", (int)act_snap.size());
            }
            ImGui::Spacing();

            update_activation_textures();

            float tile_w = ImGui::GetContentRegionAvail().x - 16.0f;
            ImDrawList* dl = ImGui::GetWindowDrawList();

            for (int l = 0; l < (int)act_snap.size(); ++l) {
                const auto& act = act_snap[l];
                if (act.values.empty()) continue;

                ImGui::PushID(l);

                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Layer %d", l);
                ImGui::SameLine();
                float vmin = *std::min_element(act.values.begin(), act.values.end());
                float vmax = *std::max_element(act.values.begin(), act.values.end());
                ImGui::TextDisabled("[%.3f, %.3f]", vmin, vmax);

                if (l < (int)layer_textures_.size() && layer_textures_[l]) {
                    float hm_h = std::clamp((float)act.rows * 2.0f, 16.0f, 80.0f);
                    ImGui::Image((ImTextureID)(intptr_t)layer_textures_[l],
                                 ImVec2(tile_w, hm_h));
                }

                if (l < (int)act_snap.size() - 1) {
                    ImVec2 p = ImGui::GetCursorScreenPos();
                    float cx = p.x + tile_w * 0.5f;
                    dl->AddLine(ImVec2(cx, p.y), ImVec2(cx, p.y + 12.0f),
                                IM_COL32(80, 180, 255, 180), 2.0f);
                    dl->AddTriangleFilled(
                        ImVec2(cx, p.y + 16.0f),
                        ImVec2(cx - 4, p.y + 10.0f),
                        ImVec2(cx + 4, p.y + 10.0f),
                        IM_COL32(80, 180, 255, 180));
                    ImGui::Dummy(ImVec2(tile_w, 18.0f));
                }

                ImGui::PopID();
            }
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
// Model Loading (async)
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
    output_text_.clear();
    tokens_generated_ = 0;
}

// ============================================================================
// Inference (async, streaming)
// ============================================================================

void OpenGllamaApplet::run_inference_async(const std::string& prompt) {
    if (inference_thread_.joinable()) inference_thread_.join();

    {
        std::lock_guard<std::mutex> lk(output_mutex_);
        output_text_.clear();
        activations_.clear();
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

        std::vector<llama_token> tokens(prompt.size() + 8);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        }
        tokens.resize(n_tokens);

        int n_layers = llama_model_n_layer(model_);

        llama_batch batch = llama_batch_init(std::max(n_tokens, 1), 0, 1);
        batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }

        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            std::lock_guard<std::mutex> lk(output_mutex_);
            output_text_ = "ERROR: decode failed on prompt";
            inference_running_ = false;
            inference_finished_ = true;
            return;
        }

        int n_vocab = llama_vocab_n_tokens(vocab);
        const int n_gen = 512;

        for (int i = 0; i < n_gen; ++i) {
            if (!inference_running_) break;

            float* logits = llama_get_logits_ith(ctx_, -1);

            llama_token best = 0;
            float best_logit = logits[0];
            for (int t = 1; t < n_vocab; ++t) {
                if (logits[t] > best_logit) {
                    best_logit = logits[t];
                    best = t;
                }
            }

            if (llama_vocab_is_eog(vocab, best)) break;

            char piece[64] = {};
            llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);

            {
                std::lock_guard<std::mutex> lk(output_mutex_);
                output_text_ += piece;
                tokens_generated_.store(i + 1);

                activations_.resize(n_layers);
                for (int l = 0; l < n_layers; ++l) {
                    activations_[l].layer_index = l;
                    activations_[l].rows = 32;
                    activations_[l].cols = 64;
                    activations_[l].values.resize(32 * 64);
                    for (auto& v : activations_[l].values)
                        v = (float)(rand() % 1000) / 1000.0f;
                }
                textures_dirty_ = true;
            }

            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = n_tokens + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx_, batch) != 0) break;
        }

        llama_batch_free(batch);
        inference_running_ = false;
        inference_finished_ = true;
    });
}
