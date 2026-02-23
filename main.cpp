#include <iostream>
#include <vector>
#include <cmath>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

// Simple neural network
struct Net : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Net(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

// Helper to draw a circle using OpenGL immediate mode
void drawCircle(float x, float y, float radius, float r, float g, float b, float alpha = 1.0f) {
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(r, g, b, alpha);
    glVertex2f(x, y);
    for (int i = 0; i <= 20; i++) {
        float angle = 2.0f * 3.14159f * float(i) / 20.0f;
        float dx = radius * cosf(angle);
        float dy = radius * sinf(angle);
        glVertex2f(x + dx, y + dy);
    }
    glEnd();
}

// Helper to draw a line
void drawLine(float x1, float y1, float x2, float y2, float r, float g, float b, float alpha = 0.3f) {
    glBegin(GL_LINES);
    glColor4f(r, g, b, alpha);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

// Draw neural network visualization
void drawNeuralNetwork(int input_size, int hidden_size, int output_size,
                      const std::vector<float>& input_activations,
                      const std::vector<float>& hidden1_activations,
                      const std::vector<float>& hidden2_activations,
                      const std::vector<float>& output_activations,
                      int window_width, int window_height) {

    // Setup orthographic projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, window_height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.5f);

    float margin = 100.0f;
    float layer_spacing = (window_width - 2 * margin) / 4.0f;

    auto drawLayer = [&](const std::vector<float>& activations, float x_pos, float node_radius) {
        int num_nodes = activations.size();
        float total_height = window_height - 2 * margin;
        float node_spacing = (num_nodes > 1) ? total_height / (num_nodes - 1) : 0;

        std::vector<glm::vec2> positions;
        for (int i = 0; i < num_nodes; i++) {
            float y_pos = margin + (num_nodes > 1 ? i * node_spacing : total_height / 2);
            positions.push_back({x_pos, y_pos});

            float activation = std::tanh(activations[i]); // Normalize to [-1, 1]
            float intensity = (activation + 1.0f) / 2.0f; // Map to [0, 1]

            // Color based on activation (blue = negative, red = positive)
            float r = intensity;
            float g = 0.3f;
            float b = 1.0f - intensity;

            drawCircle(x_pos, y_pos, node_radius, r, g, b, 0.8f);
        }
        return positions;
    };

    // Draw connections first (so they're behind nodes)
    float x_positions[] = {
        margin,
        margin + layer_spacing,
        margin + 2 * layer_spacing,
        margin + 3 * layer_spacing
    };

    // Draw connections between layers
    auto drawConnections = [&](const std::vector<glm::vec2>& from, const std::vector<glm::vec2>& to) {
        for (const auto& p1 : from) {
            for (const auto& p2 : to) {
                drawLine(p1.x, p1.y, p2.x, p2.y, 0.5f, 0.5f, 0.5f, 0.1f);
            }
        }
    };

    // Calculate positions
    float node_radius = 8.0f;

    // Limit visualization to reasonable sizes
    int vis_input_size = std::min(input_size, 20);
    int vis_hidden_size = std::min(hidden_size, 30);
    int vis_output_size = std::min(output_size, 20);

    std::vector<float> vis_input(input_activations.begin(),
                                  input_activations.begin() + vis_input_size);
    std::vector<float> vis_hidden1(hidden1_activations.begin(),
                                   hidden1_activations.begin() + vis_hidden_size);
    std::vector<float> vis_hidden2(hidden2_activations.begin(),
                                   hidden2_activations.begin() + vis_hidden_size);
    std::vector<float> vis_output(output_activations.begin(),
                                  output_activations.begin() + vis_output_size);

    auto input_pos = drawLayer(vis_input, x_positions[0], node_radius);
    auto hidden1_pos = drawLayer(vis_hidden1, x_positions[1], node_radius);
    auto hidden2_pos = drawLayer(vis_hidden2, x_positions[2], node_radius);
    auto output_pos = drawLayer(vis_output, x_positions[3], node_radius);

    // Draw connections
    drawConnections(input_pos, hidden1_pos);
    drawConnections(hidden1_pos, hidden2_pos);
    drawConnections(hidden2_pos, output_pos);

    // Redraw nodes on top
    drawLayer(vis_input, x_positions[0], node_radius);
    drawLayer(vis_hidden1, x_positions[1], node_radius);
    drawLayer(vis_hidden2, x_positions[2], node_radius);
    drawLayer(vis_output, x_positions[3], node_radius);

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL 2.1 for immediate mode
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_FALSE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Caliper - Neural Network Visualizer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Check MPS availability
    bool mps_available = torch::mps::is_available();
    torch::Device device(mps_available ? torch::kMPS : torch::kCPU);

    // Neural network parameters
    int input_size = 10;
    int hidden_size = 64;
    int output_size = 5;
    float learning_rate = 0.001f;

    // Create neural network
    auto net = std::make_shared<Net>(input_size, hidden_size, output_size);
    net->to(device);

    torch::optim::Adam optimizer(net->parameters(), learning_rate);

    // Training state
    bool training = false;
    int epoch = 0;
    float loss_value = 0.0f;
    std::vector<float> loss_history;

    // Current activations for visualization
    std::vector<float> input_activations(input_size, 0.0f);
    std::vector<float> hidden1_activations(hidden_size, 0.0f);
    std::vector<float> hidden2_activations(hidden_size, 0.0f);
    std::vector<float> output_activations(output_size, 0.0f);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Control panel
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);
        ImGui::Begin("Neural Network Controls");

        ImGui::Text("LibTorch version: %s", TORCH_VERSION);
        ImGui::Text("Device: %s", mps_available ? "MPS (Apple Silicon)" : "CPU");
        ImGui::Separator();

        ImGui::Text("Network Architecture");
        if (ImGui::SliderInt("Input Size", &input_size, 1, 100)) {
            net = std::make_shared<Net>(input_size, hidden_size, output_size);
            net->to(device);
            optimizer = torch::optim::Adam(net->parameters(), learning_rate);
            loss_history.clear();
            epoch = 0;
            input_activations.resize(input_size, 0.0f);
        }
        if (ImGui::SliderInt("Hidden Size", &hidden_size, 8, 512)) {
            net = std::make_shared<Net>(input_size, hidden_size, output_size);
            net->to(device);
            optimizer = torch::optim::Adam(net->parameters(), learning_rate);
            loss_history.clear();
            epoch = 0;
            hidden1_activations.resize(hidden_size, 0.0f);
            hidden2_activations.resize(hidden_size, 0.0f);
        }
        if (ImGui::SliderInt("Output Size", &output_size, 1, 20)) {
            net = std::make_shared<Net>(input_size, hidden_size, output_size);
            net->to(device);
            optimizer = torch::optim::Adam(net->parameters(), learning_rate);
            loss_history.clear();
            epoch = 0;
            output_activations.resize(output_size, 0.0f);
        }

        ImGui::Separator();
        ImGui::Text("Training Parameters");
        if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
            optimizer = torch::optim::Adam(net->parameters(), learning_rate);
        }

        ImGui::Separator();
        if (ImGui::Button(training ? "Stop Training" : "Start Training")) {
            training = !training;
        }

        ImGui::Text("Epoch: %d", epoch);
        ImGui::Text("Loss: %.6f", loss_value);

        // Plot loss history
        if (!loss_history.empty()) {
            ImGui::PlotLines("Loss History", loss_history.data(), loss_history.size(),
                           0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 120));
        }

        ImGui::Separator();
        ImGui::Text("Network Statistics");
        auto fc1_weight = net->fc1->weight.to(torch::kCPU);
        float mean = fc1_weight.mean().item<float>();
        float std = fc1_weight.std().item<float>();
        ImGui::Text("FC1 Weight Mean: %.4f", mean);
        ImGui::Text("FC1 Weight Std: %.4f", std);

        ImGui::End();

        // Training step and forward pass for visualization
        if (training || epoch == 0) {
            // Generate random training data
            auto input = torch::randn({1, input_size}, device);
            auto target = torch::randn({1, output_size}, device);

            // Forward pass with intermediate activations
            auto h1 = torch::relu(net->fc1->forward(input));
            auto h2 = torch::relu(net->fc2->forward(h1));
            auto output = net->fc3->forward(h2);

            // Copy activations to CPU for visualization
            auto input_cpu = input.to(torch::kCPU).squeeze(0);
            auto h1_cpu = h1.to(torch::kCPU).squeeze(0);
            auto h2_cpu = h2.to(torch::kCPU).squeeze(0);
            auto output_cpu = output.to(torch::kCPU).squeeze(0);

            for (int i = 0; i < input_size; i++)
                input_activations[i] = input_cpu[i].item<float>();
            for (int i = 0; i < hidden_size; i++)
                hidden1_activations[i] = h1_cpu[i].item<float>();
            for (int i = 0; i < hidden_size; i++)
                hidden2_activations[i] = h2_cpu[i].item<float>();
            for (int i = 0; i < output_size; i++)
                output_activations[i] = output_cpu[i].item<float>();

            if (training) {
                optimizer.zero_grad();
                auto loss = torch::mse_loss(output, target);
                loss.backward();
                optimizer.step();

                loss_value = loss.item<float>();
                loss_history.push_back(loss_value);

                // Keep only last 100 values
                if (loss_history.size() > 100) {
                    loss_history.erase(loss_history.begin());
                }

                epoch++;
            }
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw neural network visualization
        drawNeuralNetwork(input_size, hidden_size, output_size,
                         input_activations, hidden1_activations,
                         hidden2_activations, output_activations,
                         display_w, display_h);

        // Render ImGui on top
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
