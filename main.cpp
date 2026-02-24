#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <fstream>
#include <GLFW/glfw3.h>
#include <torch/torch.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>

// Custom CIFAR-10 Dataset
class CIFAR10Dataset : public torch::data::Dataset<CIFAR10Dataset> {
public:
    explicit CIFAR10Dataset(const std::string& root, bool train = true)
        : root_(root), train_(train) {
        load_data();
    }

    torch::data::Example<> get(size_t index) override {
        auto data = images_[index];
        auto target = labels_[index];
        return {data.clone(), target.clone()};
    }

    torch::optional<size_t> size() const override {
        return images_.size(0);
    }

private:
    void load_data() {
        std::vector<std::string> file_names;

        if (train_) {
            // Training files: data_batch_1.bin through data_batch_5.bin
            for (int i = 1; i <= 5; i++) {
                file_names.push_back(root_ + "/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin");
            }
        } else {
            // Test file
            file_names.push_back(root_ + "/cifar-10-batches-bin/test_batch.bin");
        }

        int total_images = train_ ? 50000 : 10000;

        // Pre-allocate tensors for better performance
        images_ = torch::zeros({total_images, 3, 32, 32}, torch::kFloat32);
        labels_ = torch::zeros({total_images}, torch::kLong);

        int image_idx = 0;

        for (const auto& file_name : file_names) {
            std::ifstream file(file_name, std::ios::binary);

            if (!file.is_open()) {
                std::cerr << "Warning: Could not open " << file_name << std::endl;
                std::cerr << "Please download CIFAR-10 binary version from:" << std::endl;
                std::cerr << "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" << std::endl;
                std::cerr << "Extract to: " << root_ << std::endl;
                std::cerr << "Generating synthetic data instead..." << std::endl;

                // Generate synthetic data as fallback
                images_ = torch::randn({total_images, 3, 32, 32});
                labels_ = torch::randint(0, 10, {total_images}, torch::kLong);
                return;
            }

            // Each CIFAR-10 binary file contains 10,000 images
            // Format: [1 byte label][3072 bytes image (1024 R, 1024 G, 1024 B)]
            const int num_images_per_file = 10000;
            const int image_size = 3072; // 32x32x3
            const int record_size = 1 + image_size; // label + image

            // Read entire file at once for speed
            std::vector<uint8_t> file_buffer(record_size * num_images_per_file);
            file.read(reinterpret_cast<char*>(file_buffer.data()), file_buffer.size());

            if (!file) {
                std::cerr << "Error reading from " << file_name << std::endl;
                file.close();
                continue;
            }

            // Parse the buffer and fill tensors
            auto images_accessor = images_.accessor<float, 4>();
            auto labels_accessor = labels_.accessor<int64_t, 1>();

            for (int i = 0; i < num_images_per_file && image_idx < total_images; i++, image_idx++) {
                int offset = i * record_size;

                // First byte is the label
                labels_accessor[image_idx] = static_cast<int64_t>(file_buffer[offset]);

                // Next 3072 bytes are the image: R (1024), G (1024), B (1024)
                for (int c = 0; c < 3; c++) {
                    for (int h = 0; h < 32; h++) {
                        for (int w = 0; w < 32; w++) {
                            int idx = offset + 1 + c * 1024 + h * 32 + w;
                            images_accessor[image_idx][c][h][w] = static_cast<float>(file_buffer[idx]) / 255.0f;
                        }
                    }
                }
            }

            file.close();
            std::cout << "Loaded " << num_images_per_file << " images from " << file_name << std::endl;
        }

        if (image_idx == 0) {
            std::cerr << "No images loaded! Using synthetic data." << std::endl;
            images_ = torch::randn({total_images, 3, 32, 32});
            labels_ = torch::randint(0, 10, {total_images}, torch::kLong);
            return;
        }

        std::cout << "CIFAR-10 dataset loaded: " << image_idx << " images" << std::endl;
    }

    std::string root_;
    bool train_;
    torch::Tensor images_;
    torch::Tensor labels_;
};

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

// ResNet Basic Block
struct BasicBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample{nullptr};
    int stride;

    BasicBlock(int in_channels, int out_channels, int stride_ = 1, torch::nn::Sequential downsample_ = nullptr)
        : stride(stride_), downsample(downsample_) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

        if (downsample_) {
            downsample = register_module("downsample", downsample_);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = torch::relu(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));

        if (!downsample.is_empty()) {
            identity = downsample->forward(x);
        }

        out += identity;
        out = torch::relu(out);

        return out;
    }
};

// Small ResNet-18 for image classification
struct ResNet18 : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    // Store activations for visualization
    ObservableTensor conv1_out, layer1_out, layer2_out, layer3_out, layer4_out, pool_out;
    bool auto_visualize = false;

    ResNet18(int num_classes = 10) {
        // Initial conv: 3x32x32 -> 64x32x32
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        // ResNet layers
        layer1 = register_module("layer1", make_layer(64, 64, 2, 1));   // 64x32x32
        layer2 = register_module("layer2", make_layer(64, 128, 2, 2));  // 128x16x16
        layer3 = register_module("layer3", make_layer(128, 256, 2, 2)); // 256x8x8
        layer4 = register_module("layer4", make_layer(256, 512, 2, 2)); // 512x4x4

        // Global average pooling and classifier
        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)));
        fc = register_module("fc", torch::nn::Linear(512, num_classes));
    }

    torch::nn::Sequential make_layer(int in_channels, int out_channels, int num_blocks, int stride) {
        torch::nn::Sequential layers;

        // Downsample if needed
        torch::nn::Sequential downsample{nullptr};
        if (stride != 1 || in_channels != out_channels) {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels)
            );
        }

        // First block (may downsample)
        layers->push_back(BasicBlock(in_channels, out_channels, stride, downsample));

        // Remaining blocks
        for (int i = 1; i < num_blocks; i++) {
            layers->push_back(BasicBlock(out_channels, out_channels));
        }

        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        // Initial conv
        conv1_out = torch::relu(bn1->forward(conv1->forward(x)));

        // ResNet blocks
        layer1_out = layer1->forward(conv1_out.get());
        layer2_out = layer2->forward(layer1_out.get());
        layer3_out = layer3->forward(layer2_out.get());
        layer4_out = layer4->forward(layer3_out.get());

        // Global pooling and classification
        pool_out = avgpool->forward(layer4_out.get());
        x = pool_out.get().view({pool_out.get().size(0), -1});
        x = fc->forward(x);

        return x;
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

// Draw 3D ResNet network visualization
void drawResNetNetwork3D(Camera& camera,
                        const torch::Tensor& input_img,
                        const ObservableTensor& conv1_act,
                        const ObservableTensor& layer1_act,
                        const ObservableTensor& layer2_act,
                        const ObservableTensor& layer3_act,
                        const ObservableTensor& layer4_act,
                        const ObservableTensor& pool_act,
                        const torch::Tensor& output_act,
                        const std::shared_ptr<ResNet18>& net) {

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

    // Layer positions (spread out for ResNet visualization)
    float z_positions[] = {-24.0f, -18.0f, -12.0f, -6.0f, 0.0f, 6.0f, 12.0f, 18.0f, 22.0f};

    // 0. Raw Image Layer - Show colored pixels as actual RGB image (composite view)
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0].to(torch::kCPU);
        float pixel_size = 0.10f;
        int img_size = 32;

        // Normalize the image for display (denormalize from training normalization)
        auto mean = torch::tensor({0.4914, 0.4822, 0.4465}).view({3, 1, 1});
        auto std = torch::tensor({0.2023, 0.1994, 0.2010}).view({3, 1, 1});
        auto denorm_img = img * std + mean;
        denorm_img = torch::clamp(denorm_img, 0.0, 1.0);

        // Draw each pixel with its true RGB color
        for (int i = 0; i < img_size; i++) {
            for (int j = 0; j < img_size; j++) {
                float r = denorm_img[0][i][j].item<float>();
                float g = denorm_img[1][i][j].item<float>();
                float b = denorm_img[2][i][j].item<float>();

                float x = (j - img_size/2.0f) * pixel_size;
                float y = (img_size/2.0f - i) * pixel_size;

                // Draw colored pixel
                drawCube(x, y, z_positions[0],
                        pixel_size * 0.95f, pixel_size * 0.95f, 0.03f,
                        r, g, b, 1.0f);
            }
        }
    }

    // 1. Input layer (3x32x32 RGB image) - show all 3 channels as separate tensor slices
    if (input_img.defined() && input_img.size(0) > 0) {
        auto img = input_img[0].to(torch::kCPU);
        float pixel_size = 0.08f;
        int img_size = 32;

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < img_size; i += 2) {
                for (int j = 0; j < img_size; j += 2) {
                    float val = img[c][i][j].item<float>();
                    float x = (j - img_size/2.0f) * pixel_size;
                    float y = (img_size/2.0f - i) * pixel_size;
                    float z = z_positions[1] + (c - 1) * 0.1f;

                    if (std::abs(val) > 0.1f) {
                        float r = (c == 0) ? std::abs(val) : 0.3f;
                        float g = (c == 1) ? std::abs(val) : 0.3f;
                        float b = (c == 2) ? std::abs(val) : 0.3f;
                        drawCube(x, y, z, pixel_size * 0.8f, pixel_size * 0.8f, 0.02f, r, g, b, 0.8f);
                    }
                }
            }
        }
    }

    std::vector<glm::vec3> conv1_positions, layer1_positions, layer2_positions, layer3_positions, layer4_positions, pool_positions;

    // 2. Initial Conv1 layer - 64 channels × 32×32
    if (conv1_act.defined() && conv1_act.get().size(0) > 0) {
        auto act = conv1_act.get()[0].to(torch::kCPU);
        int channels = act.size(0); // 64
        int height = act.size(1);   // 32
        int width = act.size(2);    // 32

        float pixel_size = 0.07f;
        float channel_depth = 0.05f;

        for (int c = 0; c < channels; c += 2) {
            for (int i = 0; i < height; i += 4) {
                for (int j = 0; j < width; j += 4) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[2] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        conv1_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.7f, pixel_size * 0.7f, channel_depth * 0.8f,
                                intensity, 0.5f + intensity * 0.5f, 1.0f - intensity * 0.3f, 0.6f);
                    }
                }
            }
        }
    }

    // 3. Layer1 - 64 channels × 32×32
    if (layer1_act.defined() && layer1_act.get().size(0) > 0) {
        auto act = layer1_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.07f;
        float channel_depth = 0.05f;

        for (int c = 0; c < channels; c += 2) {
            for (int i = 0; i < height; i += 4) {
                for (int j = 0; j < width; j += 4) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[3] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer1_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.7f, pixel_size * 0.7f, channel_depth * 0.8f,
                                intensity, 0.6f, 1.0f - intensity * 0.4f, 0.65f);
                    }
                }
            }
        }
    }

    // 4. Layer2 - 128 channels × 16×16
    if (layer2_act.defined() && layer2_act.get().size(0) > 0) {
        auto act = layer2_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.09f;
        float channel_depth = 0.04f;

        for (int c = 0; c < channels; c += 3) {
            for (int i = 0; i < height; i += 2) {
                for (int j = 0; j < width; j += 2) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[4] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer2_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.8f, pixel_size * 0.8f, channel_depth * 0.8f,
                                intensity, 0.5f, 1.0f - intensity * 0.5f, 0.7f);
                    }
                }
            }
        }
    }

    // 5. Layer3 - 256 channels × 8×8
    if (layer3_act.defined() && layer3_act.get().size(0) > 0) {
        auto act = layer3_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.12f;
        float channel_depth = 0.03f;

        for (int c = 0; c < channels; c += 4) {
            for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < width; j += 1) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[5] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer3_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.9f, pixel_size * 0.9f, channel_depth * 0.8f,
                                intensity, 0.4f, 1.0f - intensity * 0.6f, 0.75f);
                    }
                }
            }
        }
    }

    // 6. Layer4 - 512 channels × 4×4
    if (layer4_act.defined() && layer4_act.get().size(0) > 0) {
        auto act = layer4_act.get()[0].to(torch::kCPU);
        int channels = act.size(0);
        int height = act.size(1);
        int width = act.size(2);

        float pixel_size = 0.18f;
        float channel_depth = 0.02f;

        for (int c = 0; c < channels; c += 8) {
            for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < width; j += 1) {
                    float val = act[c][i][j].item<float>();
                    if (val > 0.05f) {
                        float x = (j - width/2.0f) * pixel_size;
                        float y = (height/2.0f - i) * pixel_size;
                        float z = z_positions[6] + (c - channels/2.0f) * channel_depth;
                        float intensity = std::min(1.0f, val);
                        layer4_positions.push_back({x, y, z});
                        drawCube(x, y, z, pixel_size * 0.95f, pixel_size * 0.95f, channel_depth * 0.8f,
                                intensity, 0.3f, 1.0f - intensity * 0.7f, 0.8f);
                    }
                }
            }
        }
    }

    // 7. Global Average Pool - 512 dimensions
    if (pool_act.defined() && pool_act.get().size(0) > 0) {
        auto act = pool_act.get()[0].to(torch::kCPU);
        act = act.view({-1});
        int num_features = act.size(0); // 512

        int grid_width = 32;
        int grid_height = 16;
        float neuron_size = 0.08f;

        for (int i = 0; i < num_features; i += 2) {
            float val = act[i].item<float>();
            if (val > 0.05f) {
                float intensity = std::min(1.0f, val);
                int row = i / grid_width;
                int col = i % grid_width;
                float x = (col - grid_width / 2.0f) * neuron_size;
                float y = (grid_height / 2.0f - row) * neuron_size;
                pool_positions.push_back({x, y, z_positions[7]});
                drawCube(x, y, z_positions[7], neuron_size * 0.9f, neuron_size * 0.9f, 0.05f,
                        intensity, 0.2f + intensity * 0.4f, 1.0f - intensity * 0.6f, 0.85f);
            }
        }
    }

    // 8. Output layer - 10 classes
    std::vector<glm::vec3> output_positions;
    if (output_act.defined() && output_act.size(0) > 0) {
        auto act = output_act[0].to(torch::kCPU);
        auto probs = torch::softmax(act, 0);
        float neuron_size = 0.20f;

        for (int i = 0; i < 10; i++) {
            float prob = probs[i].item<float>();
            float y = (i - 4.5f) * neuron_size;
            output_positions.push_back({0, y, z_positions[8]});
            drawCube(0, y, z_positions[8], neuron_size * 1.2f, neuron_size * 0.9f, 0.08f,
                    prob, 0.8f * prob, 0.2f, 1.0f);
        }
    }

    // Draw connections (sampled heavily for performance)
    glLineWidth(1.0f);

    auto draw_layer_connections = [](const std::vector<glm::vec3>& from, const std::vector<glm::vec3>& to,
                                     float r, float g, float b, float alpha, int sample_rate = 20) {
        for (size_t i = 0; i < from.size(); i += sample_rate) {
            for (size_t j = 0; j < to.size(); j += sample_rate) {
                drawLine3D(from[i].x, from[i].y, from[i].z, to[j].x, to[j].y, to[j].z, r, g, b, alpha);
            }
        }
    };

    if (!conv1_positions.empty() && !layer1_positions.empty())
        draw_layer_connections(conv1_positions, layer1_positions, 0.4f, 0.4f, 0.7f, 0.1f, 30);
    if (!layer1_positions.empty() && !layer2_positions.empty())
        draw_layer_connections(layer1_positions, layer2_positions, 0.5f, 0.5f, 0.8f, 0.12f, 35);
    if (!layer2_positions.empty() && !layer3_positions.empty())
        draw_layer_connections(layer2_positions, layer3_positions, 0.6f, 0.4f, 0.9f, 0.15f, 40);
    if (!layer3_positions.empty() && !layer4_positions.empty())
        draw_layer_connections(layer3_positions, layer4_positions, 0.7f, 0.3f, 1.0f, 0.18f, 45);
    if (!layer4_positions.empty() && !pool_positions.empty())
        draw_layer_connections(layer4_positions, pool_positions, 0.8f, 0.2f, 0.9f, 0.2f, 20);

    // Pool to output with FC weights if available
    if (!pool_positions.empty() && !output_positions.empty()) {
        auto fc_weights = net->fc->weight.to(torch::kCPU).abs();
        for (size_t i = 0; i < pool_positions.size(); i += 2) {
            for (size_t j = 0; j < output_positions.size(); j++) {
                drawLine3D(pool_positions[i].x, pool_positions[i].y, pool_positions[i].z,
                          output_positions[j].x, output_positions[j].y, output_positions[j].z,
                          0.9f, 0.4f, 0.3f, 0.25f);
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
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Caliper - ResNet-18 CIFAR-10 Visualizer", nullptr, nullptr);
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

    std::cout << "Loading CIFAR-10 dataset..." << std::endl;

    // Load CIFAR-10 dataset
    const std::string dataset_path = "/Users/ahmed/CLionProjects/caliper/data";
    auto train_dataset = CIFAR10Dataset(dataset_path, true)
        .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(64).workers(0)
    );

    std::cout << "CIFAR-10 dataset loaded successfully!" << std::endl;

    // Create ResNet-18 network
    float learning_rate = 0.01f;
    auto net = std::make_shared<ResNet18>(10);
    net->to(device);

    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(5e-4));

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
    net->layer1_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->layer2_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->layer3_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->layer4_out.set_callback([&needs_visualization_update, &net]() {
        if (net->auto_visualize) needs_visualization_update = true;
    });
    net->pool_out.set_callback([&needs_visualization_update, &net]() {
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
        ImGui::SetNextWindowSize(ImVec2(450, 720), ImGuiCond_FirstUseEver);
        ImGui::Begin("ResNet-18 CIFAR-10 Training Monitor");

        ImGui::Text("LibTorch: %s", TORCH_VERSION);
        ImGui::Text("Device: %s", mps_available ? "MPS (Apple Silicon)" : "CPU");
        ImGui::Separator();

        ImGui::Text("ResNet-18 Architecture:");
        ImGui::BulletText("Input: 3x32x32 RGB (3,072)");
        ImGui::BulletText("Conv1: 64 filters, 3x3");
        ImGui::BulletText("Layer1: 64 channels (2 blocks)");
        ImGui::BulletText("Layer2: 128 channels (2 blocks)");
        ImGui::BulletText("Layer3: 256 channels (2 blocks)");
        ImGui::BulletText("Layer4: 512 channels (2 blocks)");
        ImGui::BulletText("Global Avg Pool + FC: 512 -> 10");
        ImGui::Text("Total Parameters: ~11.2M");

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
                    auto loss = torch::nn::functional::cross_entropy(output, target);

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

        // Draw 3D ResNet network (use stored activations from forward pass)
        drawResNetNetwork3D(camera,
                           current_input,
                           net->conv1_out,
                           net->layer1_out,
                           net->layer2_out,
                           net->layer3_out,
                           net->layer4_out,
                           net->pool_out,
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
