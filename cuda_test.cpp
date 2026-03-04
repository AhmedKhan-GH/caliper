#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch CUDA Test\n";
    std::cout << "==================\n\n";

    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "YES" : "NO") << "\n";

    if (torch::cuda::is_available()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << "\n";

        try {
            auto tensor = torch::randn({3, 3}, torch::device(torch::kCUDA));
            std::cout << "Successfully created CUDA tensor:\n" << tensor << "\n";
        } catch (const std::exception& e) {
            std::cout << "Error creating CUDA tensor: " << e.what() << "\n";
        }
    }

    return 0;
}
