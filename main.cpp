#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

// MNIST CNN Architecture
struct MNISTNet : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    // Store activations for visualization
    torch::Tensor conv1_out, conv2_out, fc1_out, fc2_out;

    MNISTNet() {
        // Conv layers: 1x28x28 -> 32x24x24 -> 64x8x8
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5)));

        // FC layers: 64*4*4=1024 -> 128 -> 64 -> 10
        fc1 = register_module("fc1", torch::nn::Linear(64 * 4 * 4, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Conv1 -> ReLU -> MaxPool
        conv1_out = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(conv1_out, 2);

        // Conv2 -> ReLU -> MaxPool
        conv2_out = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(conv2_out, 2);

        // Flatten
        x = x.view({x.size(0), -1});

        // FC layers
        fc1_out = torch::relu(fc1->forward(x));
        fc2_out = torch::relu(fc2->forward(fc1_out));
        x = fc3->forward(fc2_out);

        return torch::log_softmax(x, 1);
    }
};

// Camera state
struct Camera {
    float angleX = 30.0f;
    float angleY = 45.0f;
    float distance = 15.0f;
    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;
    bool dragging = false;
};

// Helper to draw a 3D sphere
void drawSphere(float x, float y, float z, float radius, float r, float g, float b, float alpha = 1.0f) {
    const int slices = 10;
    const int stacks = 10;

    glColor4f(r, g, b, alpha);

    for (int i = 0; i < stacks; ++i) {
        float lat0 = M_PI * (-0.5f + (float)i / stacks);
        float z0 = radius * sinf(lat0);
        float zr0 = radius * cosf(lat0);

        float lat1 = M_PI * (-0.5f + (float)(i + 1) / stacks);
        float z1 = radius * sinf(lat1);
        float zr1 = radius * cosf(lat1);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            float lng = 2 * M_PI * (float)j / slices;
            float x0 = cosf(lng);
            float y0 = sinf(lng);

            glVertex3f(x + x0 * zr0, y + y0 * zr0, z + z0);
            glVertex3f(x + x0 * zr1, y + y0 * zr1, z + z1);
        }
        glEnd();
    }
}

// Helper to draw a 3D line
void drawLine3D(float x1, float y1, float z1, float x2, float y2, float z2,
                float r, float g, float b, float alpha = 0.3f) {
    glBegin(GL_LINES);
    glColor4f(r, g, b, alpha);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glEnd();
}

// Helper to draw a cube (for conv layers)
void drawCube(float x, float y, float z, float width, float height, float depth,
              float r, float g, float b, float alpha = 0.8f) {
    glColor4f(r, g, b, alpha);

    float w = width / 2, h = height / 2, d = depth / 2;

    // Draw faces
    glBegin(GL_QUADS);

    // Front face
    glVertex3f(x - w, y - h, z + d);
    glVertex3f(x + w, y - h, z + d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x - w, y + h, z + d);

    // Back face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x - w, y + h, z - d);
    glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y - h, z - d);

    // Top face
    glVertex3f(x - w, y + h, z - d);
    glVertex3f(x - w, y + h, z + d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y + h, z - d);

    // Bottom face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y - h, z + d);
    glVertex3f(x - w, y - h, z + d);

    // Right face
    glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y - h, z + d);

    // Left face
    glVertex3f(x - w, y - h, z - d);
    glVertex3f(x - w, y - h, z + d);
    glVertex3f(x - w, y + h, z + d);
    glVertex3f(x - w, y + h, z - d);

    glEnd();

    // Draw edges
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);

    // Bottom edges
    glVertex3f(x - w, y - h, z - d); glVertex3f(x + w, y - h, z - d);
    glVertex3f(x + w, y - h, z - d); glVertex3f(x + w, y - h, z + d);
    glVertex3f(x + w, y - h, z + d); glVertex3f(x - w, y - h, z + d);
    glVertex3f(x - w, y - h, z + d); glVertex3f(x - w, y - h, z - d);

    // Top edges
    glVertex3f(x - w, y + h, z - d); glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y + h, z - d); glVertex3f(x + w, y + h, z + d);
    glVertex3f(x + w, y + h, z + d); glVertex3f(x - w, y + h, z + d);
    glVertex3f(x - w, y + h, z + d); glVertex3f(x - w, y + h, z - d);

    // Vertical edges
    glVertex3f(x - w, y - h, z - d); glVertex3f(x - w, y + h, z - d);
    glVertex3f(x + w, y - h, z - d); glVertex3f(x + w, y + h, z - d);
    glVertex3f(x + w, y - h, z + d); glVertex3f(x + w, y + h, z + d);
    glVertex3f(x - w, y - h, z + d); glVertex3f(x - w, y + h, z + d);

    glEnd();
    glLineWidth(1.0f);
}

// Draw 3D MNIST network visualization
void drawMNISTNetwork3D(Camera& camera,
                       const torch::Tensor& input_img,
                       const torch::Tensor& conv1_act,
                       const torch::Tensor& conv2_act,
                       const torch::Tensor& fc1_act,
                       const torch::Tensor& fc2_act,
                       const torch::Tensor& output_act) {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Setup perspective projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = 1600.0f / 900.0f;
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    float projection[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (farPlane + nearPlane) / (nearPlane - farPlane), -1,
        0, 0, (2 * farPlane * nearPlane) / (nearPlane - farPlane), 0
    };
    glLoadMatrixf(projection);

    // Setup camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -camera.distance);
    glRotatef(camera.angleX, 1, 0, 0);
    glRotatef(camera.angleY, 0, 1, 0);

    // Layer positions
    float z_positions[] = {-8.0f, -4.0f, 0.0f, 4.0f, 6.0f, 8.0f};

    // 1. Input layer (28x28 image as 2D plane)
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0][0].to(torch::kCPU);
        float scale = 0.15f;
        for (int i = 0; i < 28; i += 2) {
            for (int j = 0; j < 28; j += 2) {
                float val = img[i][j].item<float>();
                float x = (j - 14) * scale;
                float y = (14 - i) * scale;
                if (val > 0.1f) {
                    drawSphere(x, y, z_positions[0], 0.08f, val, val, val, 0.9f);
                }
            }
        }
    }

    // 2. Conv1 layer (32 feature maps, show as cube)
    drawCube(0, 0, z_positions[1], 2.5f, 2.5f, 0.8f, 0.3f, 0.6f, 0.9f, 0.7f);

    // Show some activations
    if (conv1_act.defined() && conv1_act.size(0) > 0) {
        auto act = conv1_act[0].to(torch::kCPU);
        for (int ch = 0; ch < std::min(8, (int)act.size(0)); ch++) {
            float avg = act[ch].mean().item<float>();
            float intensity = std::min(1.0f, std::max(0.0f, avg));
            float angle = (ch / 8.0f) * 2 * M_PI;
            float radius = 1.5f;
            float x = radius * cosf(angle);
            float y = radius * sinf(angle);
            drawSphere(x, y, z_positions[1], 0.12f, intensity, 0.5f, 1.0f - intensity, 0.8f);
        }
    }

    // 3. Conv2 layer (64 feature maps, show as larger cube)
    drawCube(0, 0, z_positions[2], 3.0f, 3.0f, 1.2f, 0.2f, 0.7f, 0.8f, 0.7f);

    if (conv2_act.defined() && conv2_act.size(0) > 0) {
        auto act = conv2_act[0].to(torch::kCPU);
        for (int ch = 0; ch < std::min(12, (int)act.size(0)); ch++) {
            float avg = act[ch].mean().item<float>();
            float intensity = std::min(1.0f, std::max(0.0f, avg));
            float angle = (ch / 12.0f) * 2 * M_PI;
            float radius = 1.8f;
            float x = radius * cosf(angle);
            float y = radius * sinf(angle);
            drawSphere(x, y, z_positions[2], 0.1f, intensity, 0.4f, 1.0f - intensity, 0.8f);
        }
    }

    // 4. FC1 layer (128 neurons in a grid)
    if (fc1_act.defined() && fc1_act.size(0) > 0) {
        auto act = fc1_act[0].to(torch::kCPU);
        int grid_size = 8; // 8x8 grid for 64 neurons
        float spacing = 0.3f;
        for (int i = 0; i < grid_size && i * grid_size < std::min(64, (int)act.size(0)); i++) {
            for (int j = 0; j < grid_size && i * grid_size + j < std::min(64, (int)act.size(0)); j++) {
                float val = act[i * grid_size + j].item<float>();
                float intensity = std::min(1.0f, std::max(0.0f, val));
                float x = (j - grid_size / 2.0f) * spacing;
                float y = (i - grid_size / 2.0f) * spacing;
                drawSphere(x, y, z_positions[3], 0.1f, intensity, 0.3f, 1.0f - intensity, 0.8f);
            }
        }
    }

    // 5. FC2 layer (64 neurons in a circle)
    if (fc2_act.defined() && fc2_act.size(0) > 0) {
        auto act = fc2_act[0].to(torch::kCPU);
        int num_show = std::min(32, (int)act.size(0));
        for (int i = 0; i < num_show; i++) {
            float val = act[i].item<float>();
            float intensity = std::min(1.0f, std::max(0.0f, val));
            float angle = (i / (float)num_show) * 2 * M_PI;
            float radius = 1.2f;
            float x = radius * cosf(angle);
            float y = radius * sinf(angle);
            drawSphere(x, y, z_positions[4], 0.12f, intensity, 0.3f, 1.0f - intensity, 0.9f);
        }
    }

    // 6. Output layer (10 classes in a line)
    if (output_act.defined() && output_act.size(0) > 0) {
        auto act = output_act[0].to(torch::kCPU);
        auto probs = torch::softmax(act, 0);
        for (int i = 0; i < 10; i++) {
            float prob = probs[i].item<float>();
            float y = (i - 4.5f) * 0.3f;
            drawSphere(0, y, z_positions[5], 0.15f, prob, 0.8f * prob, 0.2f, 1.0f);
        }
    }

    // Draw connecting lines between layers
    glColor4f(0.3f, 0.3f, 0.4f, 0.2f);
    for (int i = 0; i < 5; i++) {
        drawLine3D(0, 0, z_positions[i], 0, 0, z_positions[i + 1], 0.3f, 0.3f, 0.4f, 0.3f);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
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
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Caliper - MNIST 3D Visualizer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Camera setup
    Camera camera;
    glfwSetWindowUserPointer(window, &camera);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    // Install ImGui callbacks (install_callbacks=false to not override our callbacks)
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Install our custom callbacks that check ImGui state
    auto prev_mouse_button_callback = glfwSetMouseButtonCallback(window,
        [](GLFWwindow* win, int button, int action, int mods) {
            // Let ImGui process first
            ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                if (action == GLFW_PRESS) {
                    cam->dragging = true;
                    double x, y;
                    glfwGetCursorPos(win, &x, &y);
                    cam->lastMouseX = x;
                    cam->lastMouseY = y;
                } else if (action == GLFW_RELEASE) {
                    cam->dragging = false;
                }
            }
        });

    auto prev_cursor_pos_callback = glfwSetCursorPosCallback(window,
        [](GLFWwindow* win, double x, double y) {
            // Let ImGui process first
            ImGui_ImplGlfw_CursorPosCallback(win, x, y);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            if (cam->dragging) {
                float dx = x - cam->lastMouseX;
                float dy = y - cam->lastMouseY;
                cam->angleY += dx * 0.5f;
                cam->angleX += dy * 0.5f;
                cam->lastMouseX = x;
                cam->lastMouseY = y;
            }
        });

    auto prev_scroll_callback = glfwSetScrollCallback(window,
        [](GLFWwindow* win, double xoffset, double yoffset) {
            // Let ImGui process first
            ImGui_ImplGlfw_ScrollCallback(win, xoffset, yoffset);

            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse) return; // ImGui is using the mouse

            Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(win));
            cam->distance -= yoffset * 0.5f;
            cam->distance = std::max(5.0f, std::min(30.0f, cam->distance));
        });

    // Install other ImGui callbacks
    glfwSetKeyCallback(window, ImGui_ImplGlfw_KeyCallback);
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    // Check device availability
    bool mps_available = torch::mps::is_available();
    torch::Device device(mps_available ? torch::kMPS : torch::kCPU);

    std::cout << "Loading MNIST dataset..." << std::endl;

    // Load MNIST dataset
    const std::string dataset_path = "/Users/ahmed/CLionProjects/caliper/data";
    auto train_dataset = torch::data::datasets::MNIST(dataset_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2)
    );

    std::cout << "MNIST dataset loaded successfully!" << std::endl;

    // Create MNIST network
    float learning_rate = 0.01f;
    auto net = std::make_shared<MNISTNet>();
    net->to(device);

    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9));

    // Training state
    bool training = false;
    int batch_count = 0;
    int epoch = 0;
    float loss_value = 0.0f;
    float accuracy = 0.0f;
    std::vector<float> loss_history;
    float max_loss_ever = 0.0f;

    // Current batch for visualization
    torch::Tensor current_input;
    torch::Tensor current_target;
    int current_predicted = 0;
    int current_actual = 0;

    auto train_iter = train_loader->begin();

    // Initialize with first sample
    {
        auto& batch = *train_iter;
        current_input = batch.data.slice(0, 0, 1).to(device);
        current_target = batch.target.slice(0, 0, 1).to(device);
        current_actual = batch.target[0].item<int64_t>();
        net->eval();
        torch::NoGradGuard no_grad;
        auto output = net->forward(current_input);
        current_predicted = output.argmax(1).item<int64_t>();
    }

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
        ImGui::SetNextWindowSize(ImVec2(400, 650), ImGuiCond_FirstUseEver);
        ImGui::Begin("MNIST Training Monitor");

        ImGui::Text("LibTorch: %s", TORCH_VERSION);
        ImGui::Text("Device: %s", mps_available ? "MPS (Apple Silicon)" : "CPU");
        ImGui::Separator();

        ImGui::Text("Network Architecture:");
        ImGui::BulletText("Input: 1x28x28 (784)");
        ImGui::BulletText("Conv1: 32 filters, 5x5");
        ImGui::BulletText("Conv2: 64 filters, 5x5");
        ImGui::BulletText("FC1: 1024 -> 128");
        ImGui::BulletText("FC2: 128 -> 64");
        ImGui::BulletText("Output: 64 -> 10");

        ImGui::Separator();
        ImGui::Text("Training Parameters");
        if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
            for (auto& param_group : optimizer.param_groups()) {
                static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(learning_rate);
            }
        }

        ImGui::Separator();
        if (ImGui::Button(training ? "Stop Training" : "Start Training", ImVec2(-1, 40))) {
            training = !training;
        }

        ImGui::Text("Batch: %d", batch_count);
        ImGui::Text("Epoch: %d", epoch);
        ImGui::Text("Loss: %.6f", loss_value);
        ImGui::Text("Accuracy: %.2f%%", accuracy * 100.0f);

        ImGui::Separator();
        ImGui::Text("Current Sample:");
        ImGui::Text("Predicted: %d", current_predicted);
        ImGui::Text("Actual: %d", current_actual);
        ImGui::Text("Correct: %s", current_predicted == current_actual ? "YES" : "NO");

        // Plot loss history
        if (!loss_history.empty()) {
            ImGui::Separator();
            ImGui::Text("Loss History (All %d batches)", (int)loss_history.size());

            // Plot from 0 to max_loss_ever, fitting ALL data horizontally
            ImGui::PlotLines("##Loss", loss_history.data(), loss_history.size(),
                           0, nullptr, 0.0f, max_loss_ever, ImVec2(-1, 120));

            ImGui::Text("Current: %.4f | Max: %.4f", loss_value, max_loss_ever);
        }

        ImGui::Separator();
        ImGui::Text("Camera Controls:");
        ImGui::BulletText("Left Mouse: Rotate");
        ImGui::BulletText("Scroll: Zoom");
        if (ImGui::Button("Reset Camera")) {
            camera.angleX = 30.0f;
            camera.angleY = 45.0f;
            camera.distance = 15.0f;
        }

        ImGui::End();

        // Training step
        torch::Tensor output_for_vis;
        if (training) {
            // Get next batch
            if (train_iter == train_loader->end()) {
                train_iter = train_loader->begin();
                epoch++;
                std::cout << "Epoch " << epoch << " completed" << std::endl;
            }

            auto& batch = *train_iter;
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Store first sample for visualization (keep on device for now)
            current_input = data.slice(0, 0, 1);
            current_actual = target[0].item<int64_t>();

            // Forward pass
            net->train();
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto loss = torch::nll_loss(output, target);

            // Backward pass
            loss.backward();
            optimizer.step();

            // Calculate accuracy for this batch
            auto pred = output.argmax(1);
            accuracy = pred.eq(target).sum().item<float>() / target.size(0);
            current_predicted = output.slice(0, 0, 1).argmax(1).item<int64_t>();

            // Update stats
            loss_value = loss.item<float>();
            loss_history.push_back(loss_value);
            max_loss_ever = std::max(max_loss_ever, loss_value);

            // Don't limit history size - keep everything!

            batch_count++;

            // Print progress every 10 batches
            if (batch_count % 10 == 0) {
                std::cout << "Batch " << batch_count << " - Loss: " << loss_value
                         << " - Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
            }

            ++train_iter;

            // Get output for visualization
            net->eval();
            {
                torch::NoGradGuard no_grad;
                output_for_vis = net->forward(current_input);
            }
            net->train();
        } else {
            // Just do forward pass for visualization if not training
            net->eval();
            torch::NoGradGuard no_grad;
            output_for_vis = net->forward(current_input);
            current_predicted = output_for_vis.argmax(1).item<int64_t>();
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw 3D MNIST network (use stored activations from forward pass)
        drawMNISTNetwork3D(camera,
                          current_input,
                          net->conv1_out,
                          net->conv2_out,
                          net->fc1_out,
                          net->fc2_out,
                          output_for_vis);

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
