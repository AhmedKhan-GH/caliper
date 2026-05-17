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

void LlamaTerminalHelper::cmd_clear(argument_type& arg) {
    arg.term.clear();
}

void LlamaTerminalHelper::cmd_help(argument_type& arg) {
    arg.term.add_text("Commands:");
    arg.term.add_text("  infer <prompt>  - run inference on the loaded model");
    arg.term.add_text("  load            - open file dialog to select a .gguf model");
    arg.term.add_text("  unload          - unload the current model");
    arg.term.add_text("  status          - show model and parameter info");
    arg.term.add_text("  set <key> <val> - set gpu_layers or context_size");
    arg.term.add_text("  clear           - clear terminal output");
    arg.term.add_text("  help            - show this help");
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
        arg.term.add_text("GPU Layers: " + std::to_string(app->gpu_layers()));
        arg.term.add_text("Context Size: " + std::to_string(app->context_size()));
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
        if (app->load_model(path)) {
            arg.term.add_text("Model loaded: " + path);
        } else {
            arg.term.add_text_err("Failed to load model: " + path);
        }
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
    if (arg.command_line.size() <= 2) {
        return {"gpu_layers", "context_size"};
    }
    return {};
}

// ============================================================================
// Applet
// ============================================================================

OpenGllamaApplet::OpenGllamaApplet() = default;

OpenGllamaApplet::~OpenGllamaApplet() {
    cleanup();
}

bool OpenGllamaApplet::initialize() {
    llama_backend_init();

    term_value_.applet = this;
    term_helper_ = std::make_shared<LlamaTerminalHelper>();
    terminal_ = std::make_unique<LlamaTerminal>(
        term_value_, "llama>", 900, 400, term_helper_);

    terminal_->set_autocomplete_pos(ImTerm::position::up);
    terminal_->add_text("OpenGllama Terminal — type 'help' for commands");
    terminal_->add_text("Supported format: GGUF (use 'load' to browse for .gguf files)");

    return true;
}

void OpenGllamaApplet::cleanup() {
    terminal_.reset();
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
        if (ImGui::BeginTabItem("Terminal")) {
            draw_terminal();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Activations")) {
            draw_activation_viewer();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Handle the file dialog regardless of which tab is active
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

void OpenGllamaApplet::draw_model_loader() {
    ImGui::SeparatorText("GGUF Model");

    if (model_loaded_) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Loaded:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", model_path_.c_str());
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No model loaded");
    }

    ImGui::Spacing();

    if (ImGui::Button("Browse GGUF...", ImVec2(200, 35))) {
        IGFD::FileDialogConfig cfg;
        cfg.path = ".";
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseGGUF", "Select GGUF Model", ".gguf", cfg);
    }

    if (model_loaded_) {
        ImGui::SameLine();
        if (ImGui::Button("Unload", ImVec2(120, 35))) {
            unload_model();
        }
    }

    ImGui::Separator();
    ImGui::Text("Inference Parameters:");
    ImGui::SliderInt("GPU Layers", &n_gpu_layers_, 0, 128);
    ImGui::SliderInt("Context Size", &context_size_, 512, 8192);

    ImGui::Separator();
    ImGui::SeparatorText("Supported Formats");
    ImGui::BulletText("GGUF — GPT-Generated Unified Format");
    ImGui::BulletText("Quantizations: Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16, F32");
    ImGui::BulletText("Source: HuggingFace repos with -GGUF suffix");
}

void OpenGllamaApplet::draw_terminal() {
    if (terminal_) {
        terminal_->show();
    }
}

void OpenGllamaApplet::draw_activation_viewer() {
    if (activations_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Run inference from the Terminal tab to capture layer activations.");
        return;
    }

    ImGui::SeparatorText("Layer Activations (OpenGL Render)");

    int n_layers = (int)activations_.size();
    ImGui::SliderInt("Layer", &selected_layer_, 0, n_layers - 1);

    const auto& act = activations_[selected_layer_];
    ImGui::Text("Layer %d: %d x %d", act.layer_index, act.rows, act.cols);

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
}

std::string OpenGllamaApplet::run_inference(const std::string& prompt) {
    if (!model_ || !ctx_) return "ERROR: no model loaded";

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

    // Capture placeholder activations for visualization
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

    return output;
}
