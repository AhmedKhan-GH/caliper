# Torch objects on each platform

A training tensor reaches the screen without a copy through the CPU. These are the six steps on that path, in order,
and what each one is on macOS and on Windows. Applet code is the same on both; what differs is underneath.

| Step | macOS · Metal | Windows · Vulkan + CUDA |
|---|---|---|
| `torch::Device` | `torch::kMPS` | `torch::kCUDA` |
| `at::Tensor` | storage in an `id<MTLBuffer>` | storage in CUDA device memory |
| `adapters::to_tensor` | `CaliperTensor` on `CALIPER_DEV_METAL` | `CaliperTensor` on `CALIPER_DEV_CUDA` |
| `adapters::stream_to_tensor` | ordered after the producer's MPS command queue | ordered after the producer's CUDA stream |
| `ExportablePool::to_bridge` | the tensor's `MTLBuffer`, imported once | a `cuMemCreate` block, imported into Vulkan |
| `Bridge::texture_from_tensor_mapped` | a Metal texture | a Vulkan image |

## What each step does

1. **`torch::Device`: where the tensor is created.** Training has to run on the GPU the app draws with, or there is
   nothing on the GPU to share. [`caliper::Device::query`](services/device-v1.md) reports which one that is.
2. **`at::Tensor`: the memory that holds the data.** On macOS a Metal buffer in memory the CPU and GPU share; on
   Windows CUDA memory on the graphics card. Everything after this depends on which.
3. **`adapters::to_tensor`: how Caliper refers to that memory without copying it.** It wraps the existing buffer in a
   [`CaliperTensor`](tensor.md): where the data lives, its type and its shape.
4. **`adapters::stream_to_tensor`: when it is safe to read.** The GPU runs torch's work asynchronously, so a read too
   early draws a half-written tensor. This waits on torch's own MPS command queue or CUDA stream, not on the whole
   GPU. See [Adapters (torch)](adapters.md).
5. **`ExportablePool::to_bridge`: how the memory crosses from torch to the graphics side.** On macOS, Metal uses
   torch's buffer directly. On Windows the memory is allocated in a shareable form and imported into Vulkan once.
6. **`Bridge::texture_from_tensor_mapped`: what is drawn.** The tensor, colormapped on the GPU into a Metal texture or
   a Vulkan image. See [caliper.tensor_bridge.v1](services/tensor-bridge-v1.md).

Most applets call only steps 1, 3 and 6 directly; the SDK does 4 and 5. The walkthrough on the
[Caliper site](https://ahmedkhan-gh.github.io/caliper/#develop) shows the three in `examples/sine_scope`.
