# Torch per Platform

A training tensor reaches the screen without a copy through the CPU. Six steps make up that path, and the table shows
what each step is on macOS and on Windows. Applet code is the same on both platforms. Only the layers underneath differ.

| Step | macOS · Metal | Windows · Vulkan + CUDA |
|---|---|---|
| `torch::Device` | `torch::kMPS` | `torch::kCUDA` |
| `at::Tensor` | storage in an `id<MTLBuffer>` | storage in CUDA device memory |
| `adapters::to_tensor` | `CaliperTensor` on `CALIPER_DEV_METAL` | `CaliperTensor` on `CALIPER_DEV_CUDA` |
| `adapters::stream_to_tensor` | ordered after the producer's MPS command queue | ordered after the producer's CUDA stream |
| `ExportablePool::to_bridge` | the tensor's `MTLBuffer`, imported once | a `cuMemCreate` block, imported into Vulkan |
| `Bridge::texture_from_tensor_mapped` | a Metal texture | a Vulkan image |

## Step Details

1. **`torch::Device`: where the tensor is created.** Training must run on the GPU that the app draws with. Otherwise
   nothing on the GPU is available to share. [`caliper::Device::query`](services/caliper-device-v1.md) reports which GPU
   that is.
2. **`at::Tensor`: the memory that holds the data.** On macOS this is a Metal buffer in memory that the CPU and GPU
   share. On Windows it is CUDA memory on the graphics card.
3. **`adapters::to_tensor`: how Caliper refers to that memory without a copy.** It wraps the existing buffer in a
   [`CaliperTensor`](tensor-and-arrow-headers.md#tensor-header) that holds the location, type and shape of the data.
4. **`adapters::stream_to_tensor`: when it is safe to read.** The GPU runs torch work asynchronously, so an early read
   draws a half-written tensor. This step waits on torch's own MPS command queue or CUDA stream, not on the whole GPU.
   See [Adapters (torch)](adapters.md).
5. **`ExportablePool::to_bridge`: how the memory crosses from torch to the graphics side.** On macOS, Metal uses the
   buffer of torch directly. On Windows, the memory is allocated in a shareable form and imported into Vulkan once.
6. **`Bridge::texture_from_tensor_mapped`: what is drawn.** The GPU colormaps the tensor into a Metal texture or a
   Vulkan image. See [caliper.tensor_bridge.v1](services/caliper-tensor-bridge-v1.md).

Most applets call steps 1, 3 and 6 directly, and the SDK does steps 4 and 5. The `sine_scope` example shows the three
calls in the [Caliper site](https://ahmedkhan-gh.github.io/caliper/#develop) walkthrough.

## Windows Memory Sharing

An applet chooses one of two routes from a CUDA tensor to a Vulkan image for each tensor. Both routes use step 5.

- **An arbitrary tensor.** Any contiguous CUDA tensor from the default allocator of torch goes through
  `Bridge::texture_from_tensor_mapped` and `update_texture`. The host loads a CUDA kernel into the context of torch.
  The kernel reads the tensor in place and writes the Vulkan image, so nothing is copied and the applet does not change.
  If the kernel path refuses the tensor, the host copies it inside VRAM. If that also fails, the host stages it on the
  CPU. See [Vulkan and CUDA on Windows](../development/vulkan-and-cuda.md).
- **The exportable pool.** Tensors from an `ExportablePool` live in `cuMemCreate` blocks with shareable handles. The
  host imports a block once. Later updates and geometry draws address the tensor as an allocation id and a byte offset.
  [`caliper.geometry.v1`](services/caliper-geometry-v1.md) draws from this pool because geometry reads imported memory
  in place and has no tensor to read through a kernel.

On macOS, neither route needs an allocator change because the storage of an MPS tensor is already an `MTLBuffer`.
`ExportablePool` on MPS imports the own buffer of the tensor in the same process.

`alloc_shared` also differs between the platforms. On Windows it returns a device-backed tensor, so CUDA kernels write
the memory of the texture in place. On macOS it returns a CPU vector in unified memory. CPU writers use it without a
copy, but a device writer goes through the normal update path. See
[caliper.tensor_bridge.v1](services/caliper-tensor-bridge-v1.md#alloc_shared).

## `ExportablePool` Opt-In

An applet opts in after the host grants `CALIPER_BRIDGE_CAP_IMPORT_ALLOC`. Follow these rules to keep the opt-in safe:

1. **Check `ok()`.** Construction fails quietly when CUDA, the driver or a driver symbol is missing, and `ok()` is then
   `false`. The class compiles on every platform. Where the pool cannot be backed, `use()` does nothing and `to_bridge`
   always declines. Construct the pool once, on the worker. Construction also initializes the CUDA allocator of torch,
   which must exist before a pool scope begins.
2. **Allocate once, inside `pool.use()`, on the worker.** `use()` returns a scope, and tensors allocated while it is
   alive land in the pool. Allocate persistent slot tensors in one scope and write into them in place afterward. Then
   the host imports each block once and sees a stable address.
3. **Call `to_bridge` on the frame thread only.** It imports the block of the tensor on first use and caches the import.
   It also releases imports that frees on other threads queued. The destructor of the pool does the same release, so
   destroy the pool on the frame thread.
4. **Keep offsets 512-byte aligned.** Tensors from the allocator of torch already are. The f32 update path needs an
   offset that is a multiple of the storage-buffer offset alignment of the device. The u8 path needs a multiple of
   four. A misaligned offset returns `false`.
5. **Fall back on any miss.** `to_bridge` returns `std::nullopt` for a tensor that is not pool-backed, for a freed
   block, and for an import that the host declined. It remembers a declined block. The update calls return `false` for an
   out-of-bounds window or a released allocation. In each case, use the ordinary `update_texture` or the CPU fallback.
6. **Keep the bytes still.** The host reads an imported tensor in place. Do not rewrite it until the next update of its
   texture or the next draw that uses it. Drain the device before you publish a slot. Write the slot that is neither
   published nor displayed.
7. **On macOS, keep the tensor alive.** The MPS variant requires a contiguous MPS tensor, and the tensor must outlive
   its import. Otherwise the allocator of torch can give the same buffer to an unrelated tensor while the host still
   draws from it. Persistent slot tensors satisfy this rule.
