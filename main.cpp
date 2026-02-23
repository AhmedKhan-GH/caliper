#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

// Observable Tensor - wraps torch::Tensor with change notification
class ObservableTensor {
public:
    torch::Tensor data;
    std::function<void()> on_change_callback;

    ObservableTensor() = default;

    ObservableTensor& operator=(const torch::Tensor& tensor) {
        data = tensor;
        if (on_change_callback) {
            on_change_callback();
        }
        return *this;
    }

    operator torch::Tensor() const { return data; }
    torch::Tensor get() const { return data; }
    bool defined() const { return data.defined(); }

    void set_callback(std::function<void()> callback) {
        on_change_callback = callback;
    }
};

// MNIST CNN Architecture
struct MNISTNet : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    // Store activations for visualization (now observable)
    ObservableTensor conv1_out, conv2_out, fc1_out, fc2_out;
    bool auto_visualize = false;

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
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 20.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float yaw = -90.0f;   // Looking towards -Z
    float pitch = 0.0f;
    float moveSpeed = 0.5f;
    float lookSpeed = 0.3f;

    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;
    bool dragging = false;

    glm::vec3 getFront() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        return glm::normalize(front);
    }

    glm::vec3 getRight() {
        return glm::normalize(glm::cross(getFront(), up));
    }
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
                       const ObservableTensor& conv1_act,
                       const ObservableTensor& conv2_act,
                       const ObservableTensor& fc1_act,
                       const ObservableTensor& fc2_act,
                       const torch::Tensor& output_act,
                       const std::shared_ptr<MNISTNet>& net) {

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

    // Setup camera with lookAt
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glm::vec3 front = camera.getFront();
    glm::vec3 lookTarget = camera.position + front;

    glm::mat4 view = glm::lookAt(camera.position, lookTarget, camera.up);
    glLoadMatrixf(glm::value_ptr(view));

    // Layer positions (spread out more for clarity)
    float z_positions[] = {-12.0f, -7.0f, -2.0f, 4.0f, 8.0f, 12.0f};

    // 1. Input layer (28x28 image as 2D plane) - 1 channel
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0][0].to(torch::kCPU);
        float pixel_size = 0.12f;  // Size of each pixel cube
        for (int i = 0; i < 28; i += 2) {
            for (int j = 0; j < 28; j += 2) {
                float val = img[i][j].item<float>();
                float x = (j - 14) * pixel_size;
                float y = (14 - i) * pixel_size;
                if (val > 0.1f) {
                    drawCube(x, y, z_positions[0],
                            pixel_size * 0.8f, pixel_size * 0.8f, 0.02f,
                            val, val, val, 0.9f);
                }
            }
        }
    }

    // Store positions for drawing connections
    std::vector<glm::vec3> conv1_positions;
    std::vector<glm::vec3> conv2_positions;
    std::vector<glm::vec3> fc1_positions;
    std::vector<glm::vec3> fc2_positions;

    // 2. Conv1 layer - 32 channels × 24×24 spatial
    // Render as 3D volume: width×height = spatial, depth = channels
    if (conv1_act.defined() && conv1_act.get().size(0) > 0) {
        auto act = conv1_act.get()[0].to(torch::kCPU);
        int channels = act.size(0); // 32
        int height = act.size(1);   // 24
        int width = act.size(2);    // 24

        float pixel_size = 0.10f;  // Smaller than input (24 vs 28)
        float channel_depth = 0.08f; // Depth per channel

        // Sample spatial positions and show all channels as depth
        for (int c = 0; c < channels; c += 1) {
            for (int i = 0; i < height; i += 3) {  // Sample spatial
                for (int j = 0; j < width; j += 3) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[1] + (c - channels/2.0f) * channel_depth;

                        float intensity = std::min(1.0f, std::max(0.0f, val));
                        conv1_positions.push_back({x, y, z});

                        drawCube(x, y, z,
                                pixel_size * 0.7f, pixel_size * 0.7f, channel_depth * 0.8f,
                                intensity, 0.5f + intensity * 0.5f, 1.0f - intensity * 0.5f, 0.7f);
                    }
                }
            }
        }
    }

    // 3. Conv2 layer - 64 channels × 8×8 spatial
    // Render as 3D volume: width×height = spatial (smaller), depth = channels (more)
    if (conv2_act.defined() && conv2_act.get().size(0) > 0) {
        auto act = conv2_act.get()[0].to(torch::kCPU);
        int channels = act.size(0); // 64
        int height = act.size(1);   // 8
        int width = act.size(2);    // 8

        float pixel_size = 0.15f;  // Larger pixels for smaller spatial dim
        float channel_depth = 0.06f; // Thinner per channel (more channels)

        // Sample spatial positions and show all channels as depth
        for (int c = 0; c < channels; c += 1) {
            for (int i = 0; i < height; i += 1) {  // Show all spatial (8x8 is small)
                for (int j = 0; j < width; j += 1) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[2] + (c - channels/2.0f) * channel_depth;

                        float intensity = std::min(1.0f, std::max(0.0f, val));
                        conv2_positions.push_back({x, y, z});

                        drawCube(x, y, z,
                                pixel_size * 0.8f, pixel_size * 0.8f, channel_depth * 0.8f,
                                intensity, 0.4f + intensity * 0.5f, 1.0f - intensity * 0.6f, 0.7f);
                    }
                }
            }
        }
    }

    // 4. FC1 layer - 1024 flattened -> 128 neurons
    // Render as a grid that collapses from wide (1024 = 32x32 grid) to narrow (128 = 16x8 grid)
    if (fc1_act.defined() && fc1_act.get().size(0) > 0) {
        auto act = fc1_act.get()[0].to(torch::kCPU);
        int num_neurons = act.size(0); // 128

        // Arrange as a 2D grid to show dimensionality (16 wide × 8 tall)
        int grid_width = 16;
        int grid_height = 8;
        float neuron_size = 0.10f;

        for (int i = 0; i < num_neurons; i++) {
            float val = act[i].item<float>();
            if (val > 0.05f) {
                float intensity = std::min(1.0f, std::max(0.0f, val));

                int row = i / grid_width;
                int col = i % grid_width;
                float x = (col - grid_width / 2.0f) * neuron_size;
                float y = (grid_height / 2.0f - row) * neuron_size;

                fc1_positions.push_back({x, y, z_positions[3]});

                drawCube(x, y, z_positions[3],
                        neuron_size * 0.8f, neuron_size * 0.8f, 0.05f,
                        intensity, 0.3f + intensity * 0.4f, 1.0f - intensity * 0.5f, 0.85f);
            }
        }
    }

    // 5. FC2 layer - 128 -> 64 neurons
    // Render as smaller 2D grid (8 wide × 8 tall)
    if (fc2_act.defined() && fc2_act.get().size(0) > 0) {
        auto act = fc2_act.get()[0].to(torch::kCPU);
        int num_neurons = act.size(0); // 64

        // Arrange as a 2D grid (8×8)
        int grid_width = 8;
        int grid_height = 8;
        float neuron_size = 0.12f;

        for (int i = 0; i < num_neurons; i++) {
            float val = act[i].item<float>();
            if (val > 0.05f) {
                float intensity = std::min(1.0f, std::max(0.0f, val));

                int row = i / grid_width;
                int col = i % grid_width;
                float x = (col - grid_width / 2.0f) * neuron_size;
                float y = (grid_height / 2.0f - row) * neuron_size;

                fc2_positions.push_back({x, y, z_positions[4]});

                drawCube(x, y, z_positions[4],
                        neuron_size * 0.9f, neuron_size * 0.9f, 0.05f,
                        intensity, 0.3f + intensity * 0.4f, 1.0f - intensity * 0.6f, 0.9f);
            }
        }
    }

    // 6. Output layer - 64 -> 10 classes
    // Render as very small vertical strip (10×1)
    std::vector<glm::vec3> output_positions;
    if (output_act.defined() && output_act.size(0) > 0) {
        auto act = output_act[0].to(torch::kCPU);
        auto probs = torch::softmax(act, 0);

        float neuron_size = 0.20f; // Large for visibility

        for (int i = 0; i < 10; i++) {
            float prob = probs[i].item<float>();
            float y = (i - 4.5f) * neuron_size;
            output_positions.push_back({0, y, z_positions[5]});

            drawCube(0, y, z_positions[5],
                    neuron_size * 1.2f, neuron_size * 0.9f, 0.08f,
                    prob, 0.8f * prob, 0.2f, 1.0f);
        }
    }

    // Draw connecting lines between layers (reduced for clarity)
    glLineWidth(1.5f);

    // Get weights from network for visualization
    auto fc2_weights = net->fc2->weight.to(torch::kCPU).abs(); // Shape: [64, 128]
    auto fc3_weights = net->fc3->weight.to(torch::kCPU).abs(); // Shape: [10, 64]

    // Normalize weights
    float fc2_mean = fc2_weights.mean().item<float>();
    float fc3_mean = fc3_weights.mean().item<float>();
    float fc2_std = fc2_weights.std().item<float>();
    float fc3_std = fc3_weights.std().item<float>();

    // Conv1 to Conv2 connections (heavily sampled - show shrinking spatial dimensions)
    if (!conv1_positions.empty() && !conv2_positions.empty()) {
        for (size_t i = 0; i < std::min(conv1_positions.size(), size_t(50)); i += 10) {
            for (size_t j = 0; j < std::min(conv2_positions.size(), size_t(50)); j += 10) {
                drawLine3D(conv1_positions[i].x, conv1_positions[i].y, conv1_positions[i].z,
                          conv2_positions[j].x, conv2_positions[j].y, conv2_positions[j].z,
                          0.4f, 0.4f, 0.7f, 0.15f);
            }
        }
    }

    // Conv2 to FC1 connections (show flattening - 3D volume to 2D grid)
    if (!conv2_positions.empty() && !fc1_positions.empty()) {
        for (size_t i = 0; i < std::min(conv2_positions.size(), size_t(64)); i += 8) {
            for (size_t j = 0; j < std::min(fc1_positions.size(), size_t(32)); j += 4) {
                drawLine3D(conv2_positions[i].x, conv2_positions[i].y, conv2_positions[i].z,
                          fc1_positions[j].x, fc1_positions[j].y, fc1_positions[j].z,
                          0.5f, 0.3f, 0.6f, 0.2f);
            }
        }
    }

    // FC1 to FC2 connections (show dimensional collapse: 128 -> 64)
    if (!fc1_positions.empty() && !fc2_positions.empty()) {
        for (size_t i = 0; i < fc1_positions.size(); i += 2) {
            for (size_t j = 0; j < fc2_positions.size(); j += 1) {
                if (j < fc2_weights.size(0) && i < fc2_weights.size(1)) {
                    float weight = fc2_weights[j][i].item<float>();
                    float normalized = (weight - fc2_mean) / (fc2_std + 0.0001f);
                    normalized = std::min(1.0f, std::max(0.0f, (normalized + 2.0f) / 4.0f));

                    float alpha = 0.15f + normalized * 0.25f;
                    drawLine3D(fc1_positions[i].x, fc1_positions[i].y, fc1_positions[i].z,
                              fc2_positions[j].x, fc2_positions[j].y, fc2_positions[j].z,
                              0.6f + normalized * 0.4f, 0.4f, 1.0f - normalized * 0.3f, alpha);
                }
            }
        }
    }

    // FC2 to Output connections (show final collapse: 64 -> 10)
    if (!fc2_positions.empty() && !output_positions.empty()) {
        for (size_t i = 0; i < fc2_positions.size(); i++) {
            for (size_t j = 0; j < output_positions.size(); j++) {
                float weight = fc3_weights[j][i].item<float>();
                float normalized = (weight - fc3_mean) / (fc3_std + 0.0001f);
                normalized = std::min(1.0f, std::max(0.0f, (normalized + 2.0f) / 4.0f));

                float r = 0.2f + normalized * 0.8f;
                float g = 0.3f + normalized * 0.4f;
                float b = 0.8f - normalized * 0.6f;
                float alpha = 0.2f + normalized * 0.5f;

                drawLine3D(fc2_positions[i].x, fc2_positions[i].y, fc2_positions[i].z,
                          output_positions[j].x, output_positions[j].y, output_positions[j].z,
                          r, g, b, alpha);
            }
        }
    }

    glLineWidth(1.0f);

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
    ImPlot::CreateContext();
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

                // Update yaw and pitch for free look
                cam->yaw += dx * cam->lookSpeed;
                cam->pitch -= dy * cam->lookSpeed;

                // Clamp pitch to avoid gimbal lock
                cam->pitch = std::max(-89.0f, std::min(89.0f, cam->pitch));

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
            // Move camera forward/backward along view direction
            glm::vec3 front = cam->getFront();
            cam->position += front * static_cast<float>(yoffset) * cam->moveSpeed;
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

    // Training speed control
    int batches_per_frame = 1;  // How many batches to process per frame
    int frame_counter = 0;
    int training_delay_ms = 0;  // Delay between training steps in milliseconds (0 = no delay)
    auto last_training_time = std::chrono::steady_clock::now();

    // Current batch for visualization
    torch::Tensor current_input;
    torch::Tensor current_target;
    int current_predicted = 0;
    int current_actual = 0;
    torch::Tensor output_for_vis;

    auto train_iter = train_loader->begin();

    // Flag to trigger visualization update
    bool needs_visualization_update = false;

    // Setup callbacks on observable tensors to trigger visualization
    net->conv1_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->conv2_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->fc1_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->fc2_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });

    // Initialize with first sample
    {
        auto& batch = *train_iter;
        current_input = batch.data.slice(0, 0, 1).to(device);
        current_target = batch.target.slice(0, 0, 1).to(device);
        current_actual = batch.target[0].item<int64_t>();
        net->eval();
        net->auto_visualize = true; // Enable auto-visualization
        torch::NoGradGuard no_grad;
        output_for_vis = net->forward(current_input);
        current_predicted = output_for_vis.argmax(1).item<int64_t>();
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Arrow key movement
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard) {
            glm::vec3 front = camera.getFront();
            glm::vec3 right = camera.getRight();

            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                camera.position += front * camera.moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                camera.position -= front * camera.moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
                camera.position -= right * camera.moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                camera.position += right * camera.moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
                camera.position += camera.up * camera.moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                camera.position -= camera.up * camera.moveSpeed;
        }

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

        ImGui::SliderInt("Training Delay (ms)", &training_delay_ms, 0, 2000);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Delay between training steps in milliseconds (higher = slower training)");
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

        // Plot loss history with ImPlot
        if (!loss_history.empty()) {
            ImGui::Separator();
            ImGui::Text("Loss History (%d batches)", (int)loss_history.size());

            if (ImPlot::BeginPlot("##Loss", ImVec2(-1, 180))) {
                ImPlot::SetupAxes("Batch", "Loss");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, loss_history.size(), ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_loss_ever * 1.1, ImGuiCond_Always);

                ImPlot::PlotLine("Training Loss", loss_history.data(), loss_history.size());
                ImPlot::EndPlot();
            }

            ImGui::Text("Current: %.4f | Max: %.4f", loss_value, max_loss_ever);
        }

        ImGui::Separator();
        ImGui::Checkbox("Auto-Update Visualization", &net->auto_visualize);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Automatically update visualization when tensors change");
        }

        ImGui::Separator();
        ImGui::Text("Camera Controls:");
        ImGui::BulletText("Arrow Keys: Move Forward/Left/Back/Right");
        ImGui::BulletText("Space/Shift: Move Up/Down");
        ImGui::BulletText("Left Mouse Drag: Look Around");
        ImGui::BulletText("Scroll: Move Forward/Backward");
        if (ImGui::Button("Reset Camera")) {
            camera.position = glm::vec3(0.0f, 0.0f, 20.0f);
            camera.yaw = -90.0f;
            camera.pitch = 0.0f;
        }

        ImGui::End();

        // Training step
        if (training) {
            // Check if enough time has passed since last training step
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_training_time).count();

            if (elapsed >= training_delay_ms) {
                last_training_time = current_time;

                // Process multiple batches per frame based on speed slider
                for (int b = 0; b < batches_per_frame; b++) {
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

                    batch_count++;

                    // Print progress every 10 batches
                    if (batch_count % 10 == 0) {
                        std::cout << "Batch " << batch_count << " - Loss: " << loss_value
                                 << " - Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
                    }

                    ++train_iter;
                }
            }

            // Get output for visualization (only once per frame)
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
                          output_for_vis,
                          net);

        // Render ImGui on top
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
