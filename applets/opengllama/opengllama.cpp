#include "opengllama.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <GL/glew.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

// ============================================================================
// Terminal Helper
// ============================================================================

LlamaTerminalHelper::LlamaTerminalHelper() {
    add_command_({"clear ", "clear the terminal", cmd_clear, no_completion});
    add_command_({"help ", "show available commands", cmd_help, no_completion});
    add_command_({"infer ", "run inference: infer <prompt text>", cmd_infer, no_completion});
    add_command_({"load ", "open file dialog to load a model", cmd_load, no_completion});
    add_command_({"set ", "set parameter: set gpu_layers|context_size <value>", cmd_set, set_completion});
    add_command_({"status ", "show model status", cmd_status, no_completion});
    add_command_({"unload ", "unload the current model", cmd_unload, no_completion});
}

void LlamaTerminalHelper::cmd_clear(argument_type& arg) { arg.term.clear(); }

void LlamaTerminalHelper::cmd_help(argument_type& arg) {
    arg.term.add_text("Commands:");
    arg.term.add_text("  infer <prompt>  - run inference on the loaded model");
    arg.term.add_text("  load [path]     - load model (opens file dialog if no path)");
    arg.term.add_text("  unload          - unload the current model");
    arg.term.add_text("  status          - show model and parameter info");
    arg.term.add_text("  set <key> <val> - set gpu_layers or context_size");
    arg.term.add_text("  clear           - clear terminal output");
}

void LlamaTerminalHelper::cmd_infer(argument_type& arg) {
    auto* app = arg.val.applet;
    if (!app || !app->is_model_loaded()) {
        arg.term.add_text_err("No model loaded. Use 'load' first.");
        return;
    }
    if (arg.command_line.size() < 2) {
        arg.term.add_text_err("Usage: infer <prompt text>");
        return;
    }
    std::string prompt;
    for (size_t i = 1; i < arg.command_line.size(); ++i) {
        if (i > 1) prompt += ' ';
        prompt += arg.command_line[i];
    }
    arg.term.add_text("> " + prompt);
    app->start_inference(prompt);
    arg.term.add_text("(generating...)");
}

void LlamaTerminalHelper::cmd_status(argument_type& arg) {
    auto* app = arg.val.applet;
    if (!app) return;
    if (app->is_model_loaded()) {
        arg.term.add_text("Model: " + app->model_path());
        arg.term.add_text("GPU Layers: " + std::to_string(app->gpu_layers()));
        arg.term.add_text("Context Size: " + std::to_string(app->context_size()));
    } else {
        arg.term.add_text("No model loaded.");
    }
}

void LlamaTerminalHelper::cmd_load(argument_type& arg) {
    auto* app = arg.val.applet;
    if (!app) return;
    if (arg.command_line.size() >= 2) {
        std::string path;
        for (size_t i = 1; i < arg.command_line.size(); ++i) {
            if (i > 1) path += ' ';
            path += arg.command_line[i];
        }
        if (app->load_model(path))
            arg.term.add_text("Model loaded: " + path);
        else
            arg.term.add_text_err("Failed to load model: " + path);
    } else {
        IGFD::FileDialogConfig cfg;
        cfg.path = ".";
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseGGUF", "Select GGUF Model", ".gguf", cfg);
        arg.term.add_text("Opening file dialog...");
    }
}

void LlamaTerminalHelper::cmd_unload(argument_type& arg) {
    auto* app = arg.val.applet;
    if (!app) return;
    if (!app->is_model_loaded()) {
        arg.term.add_text_err("No model is loaded.");
        return;
    }
    app->unload_model();
    arg.term.add_text("Model unloaded.");
}

void LlamaTerminalHelper::cmd_set(argument_type& arg) {
    auto* app = arg.val.applet;
    if (!app) return;
    if (arg.command_line.size() < 3) {
        arg.term.add_text_err("Usage: set <gpu_layers|context_size> <value>");
        return;
    }
    const auto& key = arg.command_line[1];
    int val = 0;
    try { val = std::stoi(arg.command_line[2]); }
    catch (...) {
        arg.term.add_text_err("Invalid integer: " + arg.command_line[2]);
        return;
    }
    if (key == "gpu_layers") {
        app->set_gpu_layers(val);
        arg.term.add_text("gpu_layers = " + std::to_string(val));
    } else if (key == "context_size") {
        app->set_context_size(val);
        arg.term.add_text("context_size = " + std::to_string(val));
    } else {
        arg.term.add_text_err("Unknown parameter: " + key);
    }
}

std::vector<std::string> LlamaTerminalHelper::set_completion(argument_type& arg) {
    if (arg.command_line.size() <= 2) return {"gpu_layers", "context_size"};
    return {};
}

// ============================================================================
// Applet
// ============================================================================

OpenGllamaApplet::OpenGllamaApplet() = default;

OpenGllamaApplet::~OpenGllamaApplet() { cleanup(); }

bool OpenGllamaApplet::initialize() {
    llama_backend_init();

    term_value_.applet = this;
    term_helper_ = std::make_shared<LlamaTerminalHelper>();
    terminal_ = std::make_unique<LlamaTerminal>(
        term_value_, "llama>", 900, 400, term_helper_);
    terminal_->set_autocomplete_pos(ImTerm::position::up);
    terminal_->add_text("OpenGllama — type 'help' for commands");

    return true;
}

void OpenGllamaApplet::cleanup() {
    inference_running_ = false;
    if (inference_thread_.joinable()) inference_thread_.join();
    if (load_thread_.joinable()) load_thread_.join();
    terminal_.reset();
    unload_model();

    for (auto tex : layer_textures_)
        if (tex) glDeleteTextures(1, &tex);
    layer_textures_.clear();

    llama_backend_free();
}

// ============================================================================
// Main UI — streamlined single-page layout
// ============================================================================

void OpenGllamaApplet::draw_ui(int width, int height) {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)width, (float)height));
    ImGui::Begin("OpenGllama", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    // Async model loading — show progress bar while loading
    if (inference_running_ && !load_finished_) {
        float progress = load_progress_.load();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
            "Loading %s ...", loading_model_name_.c_str());
        ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0),
            (std::to_string((int)(progress * 100)) + "%").c_str());
        ImGui::TextDisabled("Mapping model to Metal GPU memory...");
    } else if (load_finished_) {
        bool ok = load_success_.load();
        if (load_thread_.joinable()) load_thread_.join();
        inference_running_ = false;
        load_finished_ = false;
        if (ok) {
            model_loaded_ = true;
            load_error_msg_.clear();
            if (terminal_) terminal_->add_text("Loaded: " + loading_model_name_);
        } else {
            load_error_msg_ = "Failed to load " + loading_model_name_ +
                " — architecture may not be supported by llama.cpp";
            if (terminal_) terminal_->add_text_err(load_error_msg_);
        }
        loading_model_name_.clear();
    } else if (!model_loaded_) {
        draw_ollama_models();
    } else {
        draw_inference_view();
    }

    // File dialog (always active)
    ImVec2 min_sz(600, 400);
    ImVec2 max_sz(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display("ChooseGGUF",
            ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            if (load_model(path)) {
                if (terminal_) terminal_->add_text("Model loaded: " + path);
            } else {
                if (terminal_) terminal_->add_text_err("Failed to load: " + path);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::End();
}

// ============================================================================
// Model Selection (shown when no model loaded)
// ============================================================================

void OpenGllamaApplet::draw_ollama_models() {
    ImGui::SeparatorText("Ollama Path");

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

    // ── Inference params ──
    ImGui::Spacing();
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderInt("GPU Layers", &n_gpu_layers_, 0, 128);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderInt("Context Size", &context_size_, 512, 8192);
    ImGui::SameLine();
    if (ImGui::Button("Browse GGUF...")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = ".";
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseGGUF", "Select GGUF Model", ".gguf", cfg);
    }

    // ── Model list ──
    ImGui::SeparatorText("Select a Model to Load");

    const auto& models = ollama_store_.models();

    if (models.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "No Ollama models found. Use 'Browse GGUF...' or check path.");
        ImGui::Spacing();
        if (ImGui::Button("Refresh")) ollama_store_.refresh();
    } else {
        if (ImGui::Button("Refresh")) ollama_store_.refresh();
        ImGui::SameLine();
        ImGui::TextDisabled("(%d models)", (int)models.size());

        if (!load_error_msg_.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", load_error_msg_.c_str());
        }
        ImGui::Spacing();

        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            double gb = (double)m.size_bytes / (1024.0 * 1024.0 * 1024.0);

            ImGui::PushID((int)i);

            if (inference_running_) {
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
}

// ============================================================================
// Inference View (shown when model is loaded)
// ============================================================================

void OpenGllamaApplet::draw_inference_view() {
    // ── Top bar: model info + unload ──
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Model:");
    ImGui::SameLine();
    ImGui::Text("%s", model_path_.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Unload")) {
        unload_model();
        return;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("| %d layers | %d ctx",
        model_ ? llama_model_n_layer(model_) : 0, context_size_);

    ImGui::Separator();

    // ── Prompt input + Run ──
    float run_btn_w = 80.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - run_btn_w - 10.0f);

    static char prompt_input[2048] = {};
    bool enter_pressed = ImGui::InputText("##prompt", prompt_input, sizeof(prompt_input),
        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    bool run_clicked = ImGui::Button("Run", ImVec2(run_btn_w, 0));

    if (inference_running_) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Stop")) {
            inference_running_ = false;
        }
    }

    if ((enter_pressed || run_clicked) && prompt_input[0] != '\0' && !inference_running_) {
        prompt_buf_ = prompt_input;
        run_inference_async(prompt_buf_);
    }

    if (inference_finished_) {
        if (inference_thread_.joinable()) inference_thread_.join();
        inference_finished_ = false;
    }

    ImGui::Separator();

    // ── Split: output left, activations right ──
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;
    float left_w = avail_w * 0.35f;
    float right_w = avail_w - left_w - 8.0f;

    // Left panel: output text + terminal
    ImGui::BeginChild("##left_panel", ImVec2(left_w, avail_h), true);
    {
        std::string snapshot;
        int toks;
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            snapshot = output_text_;
        }
        toks = tokens_generated_.load();

        if (!snapshot.empty() || inference_running_) {
            ImGui::SeparatorText("Output");
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "Generating... (%d tokens)", toks);
            } else if (toks > 0) {
                ImGui::TextDisabled("%d tokens", toks);
            }
            ImGui::TextWrapped("%s", snapshot.c_str());
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "|");
            }
            ImGui::Spacing();
            ImGui::Separator();
        }

        ImGui::SeparatorText("Console");
        if (terminal_) terminal_->show();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel: activation flow
    ImGui::BeginChild("##right_panel", ImVec2(right_w, avail_h), true);
    {
        // Snapshot activations under lock
        std::vector<LayerActivation> act_snap;
        {
            std::lock_guard<std::mutex> lk(output_mutex_);
            act_snap = activations_;
        }

        if (act_snap.empty() && !inference_running_) {
            ImGui::TextDisabled("Enter a prompt and press Run to see activation flow.");
        } else {
            ImGui::SeparatorText("Activation Flow");
            int toks = tokens_generated_.load();
            if (inference_running_) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.4f, 1.0f),
                    "%d layers | token %d (live)", (int)act_snap.size(), toks);
            } else {
                ImGui::Text("%d layers | %d tokens", (int)act_snap.size(), toks);
            }
            ImGui::Spacing();

            update_activation_textures();

            float tile_w = right_w - 30.0f;
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

void OpenGllamaApplet::draw_terminal() {
    if (terminal_) terminal_->show();
}

// ============================================================================
// Activation Texture Upload
// ============================================================================

void OpenGllamaApplet::update_activation_textures() {
    if (!textures_dirty_) return;

    // Clean old
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
            // Blue -> Cyan -> Yellow -> Red
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
// Model Management
// ============================================================================

void OpenGllamaApplet::load_model_async(const std::string& path, const std::string& display_name) {
    if (load_thread_.joinable()) load_thread_.join();

    inference_running_ = true;
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

            // Update activations per token (placeholder until ggml hooks)
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
                    for (auto& v : activations_[l].values) {
                        v = (float)(rand() % 1000) / 1000.0f;
                    }
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
