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
   nothing on the GPU to share. [`caliper::Device::query`](service-device-v1.md) reports which one that is.
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
   a Vulkan image. See [caliper.tensor_bridge.v1](service-tensor-bridge-v1.md).

Most applets call only steps 1, 3 and 6 directly; the SDK does 4 and 5. The walkthrough on the
[Caliper site](https://ahmedkhan-gh.github.io/caliper/#develop) shows the three in `examples/sine_scope`.

## Two ways to share memory on Windows

Step 5 above is one of two routes from a CUDA tensor to a Vulkan image, and an applet chooses per tensor.

- **An arbitrary tensor.** Any contiguous CUDA tensor from torch's default allocator goes through
  `Bridge::texture_from_tensor_mapped` and `update_texture`. The host loads a CUDA kernel into torch's own context, and
  the kernel reads the tensor where it lies and writes the Vulkan image. Nothing is copied and nothing in the applet
  changes. If the kernel path refuses the tensor, the host copies it inside VRAM first, and if that fails it stages on
  the CPU. See [Vulkan and CUDA on Windows](../explanations/vulkan-cuda.md).
- **The exportable pool.** Tensors allocated from an `ExportablePool` live in `cuMemCreate` blocks with shareable
  handles. The host imports a block once, and later updates and geometry draws address the tensor as an allocation id
  and a byte offset. The pool is what [`caliper.geometry.v1`](service-geometry-v1.md) draws from, because geometry
  reads imported memory in place and has no tensor to read through a kernel.

On macOS neither route needs an allocator change: an MPS tensor's storage is already an `MTLBuffer`. `ExportablePool`
on MPS imports the tensor's own buffer in-process.

`alloc_shared` also differs. On Windows it returns a device-backed tensor, so CUDA kernels write the texture's memory
in place. On macOS it returns a CPU vector in unified memory: that is zero-copy for CPU writers, but it is not
device-backed, so a device writer goes through the normal update path. See
[caliper.tensor_bridge.v1](service-tensor-bridge-v1.md#alloc_shared).

## Opting in to ExportablePool

An applet opts in after the host grants `CALIPER_BRIDGE_CAP_IMPORT_ALLOC`. These rules keep the opt-in safe, and each
is checked against `sdk/include/caliper/adapters/exportable_pool.hpp`.

1. **Check `ok()`.** Construction fails quietly when there is no CUDA, no driver or a missing driver symbol, and
   `ok()` is then `false`. The class compiles on every platform, and where the pool cannot be backed `use()` does
   nothing and `to_bridge` always declines. Construct the pool once, on the worker. Construction also initializes
   torch's CUDA allocator, which has to exist before a pool scope begins.
2. **Allocate once, inside `pool.use()`, on the worker.** `use()` returns a scope, and tensors allocated while it is
   alive land in the pool. Allocate persistent slot tensors in one scope and write into them in place afterwards, so
   each block is imported once and the host sees a stable address.
3. **Call `to_bridge` on the frame thread only.** It imports the tensor's block into the bridge on first use and caches
   the import, and the bridge is frame-thread-only. It also releases imports queued by frees that happened on other
   threads. The pool's destructor does the same release work, so destroy it on the frame thread too.
4. **Keep offsets 512-byte aligned.** Tensors placed by torch's allocator already are. The f32 update path needs an
   offset that is a multiple of the device's storage-buffer offset alignment, the u8 path a multiple of four, and a
   misaligned offset returns `false`.
5. **Fall back on any miss.** `to_bridge` returns `std::nullopt` for a tensor that is not pool-backed, for a block that
   was freed, and for an import the host declined, and it remembers a declined block. The update calls return `false`
   for an out-of-bounds window or a released allocation. In each case, use the ordinary `update_texture` or the CPU
   fallback.
6. **Keep the bytes still.** The host reads an imported tensor in place. Do not rewrite it until the next update of
   its texture or the next draw that uses it. Drain the device before publishing a slot, and write the slot that is
   neither being published nor displayed.
7. **On macOS, keep the tensor alive.** The MPS variant requires a contiguous MPS tensor, and the tensor has to outlive
   its import. Otherwise torch's allocator may hand the same buffer to an unrelated tensor while the host still
   draws from it. Persistent slot tensors satisfy this.
