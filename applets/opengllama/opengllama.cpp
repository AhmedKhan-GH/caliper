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
    std::string result = app->run_inference(prompt);
    arg.term.add_text(result);
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

    // Deferred model load: wait 2 frames so "Loading..." renders
    if (!pending_load_path_.empty()) {
        load_frame_delay_++;
        if (load_frame_delay_ >= 2) {
            std::string path = pending_load_path_;
            std::string name = pending_load_name_;
            pending_load_path_.clear();
            pending_load_name_.clear();
            load_frame_delay_ = 0;
            inference_running_ = false;

            if (load_model(path)) {
                if (terminal_) terminal_->add_text("Loaded: " + name);
            } else {
                if (terminal_) terminal_->add_text_err("Failed to load: " + name);
            }
        } else {
            // Show loading indicator
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                "Loading %s ...", pending_load_name_.c_str());
            ImGui::TextDisabled("This may take a moment for large models.");
        }
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

        if (ImGui::BeginTable("OllamaModels", 4,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_SizingFixedFit,
                ImVec2(0, ImGui::GetContentRegionAvail().y))) {

            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Tag", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < models.size(); ++i) {
                const auto& m = models[i];
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%s", m.name.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%s", m.tag.c_str());

                ImGui::TableNextColumn();
                double gb = (double)m.size_bytes / (1024.0 * 1024.0 * 1024.0);
                ImGui::Text("%.1f GB", gb);

                ImGui::TableNextColumn();
                ImGui::PushID((int)i);
                if (inference_running_) {
                    ImGui::TextDisabled("...");
                } else if (ImGui::Button("Load")) {
                    inference_running_ = true;
                    pending_load_path_ = m.blob_path;
                    pending_load_name_ = m.name + ":" + m.tag;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
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

    if ((enter_pressed || run_clicked) && prompt_input[0] != '\0' && !inference_running_) {
        prompt_buf_ = prompt_input;
        output_text_ = run_inference(prompt_buf_);
        tokens_generated_ = (int)output_text_.size();
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
        if (!output_text_.empty()) {
            ImGui::SeparatorText("Output");
            ImGui::TextWrapped("%s", output_text_.c_str());
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
        if (activations_.empty()) {
            ImGui::TextDisabled("Enter a prompt and press Run to see activation flow.");
        } else {
            ImGui::SeparatorText("Activation Flow");
            ImGui::Text("%d layers | %d tokens generated",
                        (int)activations_.size(), tokens_generated_);
            ImGui::Spacing();

            update_activation_textures();

            float tile_w = right_w - 30.0f;
            ImDrawList* dl = ImGui::GetWindowDrawList();

            for (int l = 0; l < (int)activations_.size(); ++l) {
                const auto& act = activations_[l];
                if (act.values.empty()) continue;

                ImGui::PushID(l);

                // Layer label
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Layer %d", l);
                ImGui::SameLine();
                float vmin = *std::min_element(act.values.begin(), act.values.end());
                float vmax = *std::max_element(act.values.begin(), act.values.end());
                ImGui::TextDisabled("[%.3f, %.3f]", vmin, vmax);

                // Heatmap tile
                if (l < (int)layer_textures_.size() && layer_textures_[l]) {
                    float hm_h = std::clamp((float)act.rows * 2.0f, 16.0f, 80.0f);
                    ImGui::Image((ImTextureID)(intptr_t)layer_textures_[l],
                                 ImVec2(tile_w, hm_h));
                }

                // Flow arrow to next layer
                if (l < (int)activations_.size() - 1) {
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

std::string OpenGllamaApplet::run_inference(const std::string& prompt) {
    if (!model_ || !ctx_) return "ERROR: no model loaded";

    inference_running_ = true;
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
        inference_running_ = false;
        return "ERROR: decode failed on prompt";
    }

    std::string output;
    const int n_gen = 256;
    int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_gen; ++i) {
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
        output += piece;

        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = n_tokens + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx_, batch) != 0) break;
    }

    llama_batch_free(batch);
    tokens_generated_ = (int)output.size();

    // Capture per-layer activations (placeholder — real ggml hook next)
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
    textures_dirty_ = true;

    inference_running_ = false;
    return output;
}
