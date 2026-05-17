#include "opengllama.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <GL/glew.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>

OpenGllamaApplet::OpenGllamaApplet() = default;

OpenGllamaApplet::~OpenGllamaApplet() {
    cleanup();
}

bool OpenGllamaApplet::initialize() {
    llama_backend_init();
    return true;
}

void OpenGllamaApplet::cleanup() {
    unload_model();

    if (activation_texture_) {
        glDeleteTextures(1, &activation_texture_);
        activation_texture_ = 0;
    }

    llama_backend_free();
}

void OpenGllamaApplet::draw_ui(int width, int height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)width, (float)height));
    ImGui::Begin("OpenGllama", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    if (ImGui::BeginTabBar("MainTabs")) {
        if (ImGui::BeginTabItem("Model")) {
            draw_model_loader();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Inference")) {
            draw_inference_panel();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Activations")) {
            draw_activation_viewer();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void OpenGllamaApplet::draw_model_loader() {
    ImGui::SeparatorText("Load GGUF Model");

    ImGui::Text("Model Path:");
    ImGui::SameLine();

    char buf[512];
    std::strncpy(buf, model_path_.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80.0f);
    if (ImGui::InputText("##model_path", buf, sizeof(buf))) {
        model_path_ = buf;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseGGUF", "Select GGUF Model", ".gguf", config);
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseGGUF")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            model_path_ = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Separator();
    ImGui::Text("Inference Parameters:");
    ImGui::SliderInt("GPU Layers", &n_gpu_layers_, 0, 128);
    ImGui::SliderInt("Context Size", &context_size_, 512, 8192);

    ImGui::Separator();

    if (!model_loaded_) {
        bool can_load = !model_path_.empty();
        if (!can_load) ImGui::BeginDisabled();
        if (ImGui::Button("Load Model", ImVec2(200, 40))) {
            load_model(model_path_);
        }
        if (!can_load) ImGui::EndDisabled();
    } else {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Model loaded successfully");
        if (ImGui::Button("Unload Model", ImVec2(200, 40))) {
            unload_model();
        }
    }

    ImGui::Separator();
    ImGui::SeparatorText("Supported Formats");
    ImGui::BulletText("GGUF — GPT-Generated Unified Format");
    ImGui::BulletText("Quantizations: Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16, F32");
    ImGui::BulletText("Source: HuggingFace repos with -GGUF suffix");
    ImGui::BulletText("Example: TheBloke/Llama-2-7B-GGUF");
}

void OpenGllamaApplet::draw_inference_panel() {
    if (!model_loaded_) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Load a model first from the Model tab.");
        return;
    }

    ImGui::SeparatorText("Prompt");

    char prompt_buf[4096];
    std::strncpy(prompt_buf, prompt_text_.c_str(), sizeof(prompt_buf) - 1);
    prompt_buf[sizeof(prompt_buf) - 1] = '\0';
    ImGui::InputTextMultiline("##prompt", prompt_buf, sizeof(prompt_buf),
                              ImVec2(-1, 120));
    prompt_text_ = prompt_buf;

    if (inference_running_) ImGui::BeginDisabled();
    if (ImGui::Button("Run Inference", ImVec2(200, 35))) {
        run_inference(prompt_text_);
    }
    if (inference_running_) ImGui::EndDisabled();

    ImGui::SeparatorText("Output");
    ImGui::TextWrapped("%s", output_text_.c_str());
}

void OpenGllamaApplet::draw_activation_viewer() {
    if (activations_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Run inference to capture layer activations.");
        return;
    }

    ImGui::SeparatorText("Layer Activations (OpenGL Render)");

    int n_layers = (int)activations_.size();
    ImGui::SliderInt("Layer", &selected_layer_, 0, n_layers - 1);

    const auto& act = activations_[selected_layer_];
    ImGui::Text("Layer %d: %d x %d values (min/max range visualized as heatmap)",
                act.layer_index, act.rows, act.cols);

    if (texture_needs_update_ && !act.values.empty()) {
        if (!activation_texture_) {
            glGenTextures(1, &activation_texture_);
        }

        float vmin = *std::min_element(act.values.begin(), act.values.end());
        float vmax = *std::max_element(act.values.begin(), act.values.end());
        float range = (vmax - vmin) > 1e-7f ? (vmax - vmin) : 1.0f;

        std::vector<unsigned char> pixels(act.values.size() * 3);
        for (size_t i = 0; i < act.values.size(); ++i) {
            float norm = (act.values[i] - vmin) / range;
            unsigned char r = (unsigned char)(norm * 255);
            unsigned char b = (unsigned char)((1.0f - norm) * 255);
            pixels[i * 3 + 0] = r;
            pixels[i * 3 + 1] = 0;
            pixels[i * 3 + 2] = b;
        }

        glBindTexture(GL_TEXTURE_2D, activation_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, act.cols, act.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        texture_needs_update_ = false;
    }

    if (activation_texture_) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect = (float)act.cols / (float)std::max(act.rows, 1);
        float w = avail.x;
        float h = w / aspect;
        if (h > avail.y) { h = avail.y; w = h * aspect; }
        ImGui::Image((ImTextureID)(intptr_t)activation_texture_, ImVec2(w, h));
    }
}

bool OpenGllamaApplet::load_model(const std::string& path) {
    unload_model();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers_;

    model_ = llama_model_load_from_file(path.c_str(), model_params);
    if (!model_) {
        output_text_ = "ERROR: Failed to load model from: " + path;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size_;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        output_text_ = "ERROR: Failed to create context";
        return false;
    }

    model_loaded_ = true;
    model_path_ = path;
    output_text_ = "Model loaded: " + path;
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
}

bool OpenGllamaApplet::run_inference(const std::string& prompt) {
    if (!model_ || !ctx_) return false;

    inference_running_ = true;
    output_text_.clear();
    activations_.clear();

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

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    if (llama_decode(ctx_, batch) != 0) {
        output_text_ = "ERROR: decode failed";
        llama_batch_free(batch);
        inference_running_ = false;
        return false;
    }

    const int n_gen = 128;
    for (int i = 0; i < n_gen; ++i) {
        float* logits = llama_get_logits_ith(ctx_, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);

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
        output_text_ += piece;

        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = n_tokens + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx_, batch) != 0) break;
    }

    llama_batch_free(batch);

    // Capture a placeholder activation for visualization
    // (real hook into ggml graph will come in a follow-up)
    int n_layers = llama_model_n_layer(model_);
    activations_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        activations_[l].layer_index = l;
        activations_[l].rows = 32;
        activations_[l].cols = 64;
        activations_[l].values.resize(32 * 64);
        for (auto& v : activations_[l].values) {
            v = (float)(rand() % 1000) / 1000.0f;
        }
    }
    texture_needs_update_ = true;
    selected_layer_ = 0;

    inference_running_ = false;
    return true;
}
