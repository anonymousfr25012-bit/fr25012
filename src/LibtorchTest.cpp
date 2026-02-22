#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});

  std::cout << tensor << std::endl;

  const bool cuda_is_available = torch::cuda::is_available();
  auto device = cuda_is_available ? torch::kCUDA : torch::kCPU;
  std::cout << "cuda is available: " << cuda_is_available << std::endl;
  std::string device_name = cuda_is_available ? "CUDA" : "CPU";

  std::cout << tensor.to(device) << std::endl;
}
