#pragma once
/* caliper/adapters/torch.hpp — header-only bridge from a torch::Tensor to the
 * frozen CaliperTensor (PLATFORM.md §7.2). It compiles against the APPLET's
 * libtorch and is NEVER included by the host: torch stays out of the ABI (D11).
 * The host links no torch; it consumes only the CaliperTensor this produces.
 *
 * Exemplar teaching points baked into the contract:
 *   - The adapter NEVER silently copies. A non-contiguous tensor is REJECTED
 *     (std::nullopt); the caller makes the copy visible in applet code with an
 *     explicit `.contiguous()` before calling again — the cost is in the applet,
 *     not hidden in the bridge.
 *   - MPS tensors additionally REQUIRE storage_offset() == 0. The frozen
 *     CaliperTensor has no storage-offset channel, and the bridge casts
 *     storage().mutable_data() straight to an id<MTLBuffer>; a view with a
 *     nonzero offset would silently address the wrong texels. Reject it — the
 *     caller clones the view (`.clone()` / `.contiguous()` of a materialized
 *     tensor) so the buffer starts at offset 0.
 *   - v1 device story: `stream == NULL`. Correctness on device textures comes
 *     from torch::mps::synchronize() (see synced_to_tensor), i.e. sync-then-
 *     update. That sync is a FULL device barrier — it blocks the calling CPU
 *     thread until every MPS stream drains — so pay it once at the handoff, not
 *     needlessly per frame. stream_to_tensor (M2/D24) supersedes this when the
 *     host's bridge-v1.1 caps() grants stream-ordered handoff.
 */
#include <optional>

#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <torch/torch.h>
#if defined(__APPLE__)
#include <objc/message.h>   // producer-queue lookup without an ObjC++ TU
#endif
// !__APPLE__: mac libtorch ships the c10/cuda headers but no CUDA toolkit
// headers, so presence-of-header alone is not usability — and Apple torch
// builds never have CUDA anyway.
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)
#include <c10/cuda/CUDAStream.h>
#endif

namespace caliper::adapters {

namespace detail {

// v1 dtype map. Anything not named here (f64, bool, complex, quantized, fp8…)
// is rejected — the bridge only understands these six.
inline std::optional<CaliperDType> map_dtype(at::ScalarType st) {
    switch (st) {
        case at::kFloat:    return CALIPER_DT_F32;
        case at::kHalf:     return CALIPER_DT_F16;
        case at::kBFloat16: return CALIPER_DT_BF16;
        case at::kLong:     return CALIPER_DT_I64;
        case at::kInt:      return CALIPER_DT_I32;
        case at::kByte:     return CALIPER_DT_U8;
        default:            return std::nullopt;
    }
}

#if defined(__APPLE__)
// [cb commandQueue] via the C ObjC runtime, so this header stays compilable
// as plain C++ (applet TUs are .cpp, not .mm). The queue is torch's global
// MPS command queue — process-lifetime, safe to hand across the ABI.
inline void* mtl_command_queue_of(void* command_buffer) {
    using Send = void* (*)(void*, SEL);
    return command_buffer
        ? ((Send)objc_msgSend)(command_buffer, sel_registerName("commandQueue"))
        : nullptr;
}
#endif

}  // namespace detail

// Build a CaliperTensor that aliases `t`'s memory (zero-copy). Returns nullopt
// — and copies nothing — when `t` cannot be represented as a v1 CaliperTensor.
// Rejection reasons: unsupported dtype, ndim > 8, non-contiguous, an MPS view
// with a nonzero storage offset, or a device other than CPU/MPS. The caller
// logs / repairs (e.g. `t.contiguous()`); the adapter never hides a copy.
inline std::optional<CaliperTensor> to_tensor(const at::Tensor& t) {
    const auto dt = detail::map_dtype(t.scalar_type());
    if (!dt) return std::nullopt;

    const int64_t nd = t.dim();
    if (nd < 0 || nd > 8) return std::nullopt;   // frozen shape/strides are [8]

    // Contiguity is mandatory on every device: the bridge assumes row-major with
    // no gaps. Reject rather than copy so the cost is visible in applet code.
    if (!t.is_contiguous()) return std::nullopt;

    CaliperTensor out{};
    out.struct_size = sizeof(CaliperTensor);
    out.dtype = *dt;
    out.ndim = static_cast<int32_t>(nd);
    for (int64_t i = 0; i < nd; ++i) {
        out.shape[i]   = t.size(i);      // elements
        out.strides[i] = t.stride(i);    // elements
    }
    out.stream = nullptr;                // v1: no stream channel (sync explicitly)

    if (t.is_cpu()) {
        out.data = t.data_ptr();
        out.device = CALIPER_DEV_CPU;
        out.device_index = 0;
        return out;
    }
    if (t.is_mps()) {
        // No offset channel in CaliperTensor and the bridge casts this pointer
        // straight to id<MTLBuffer>; a nonzero offset would mis-address texels.
        if (t.storage_offset() != 0) return std::nullopt;
        out.data = t.storage().mutable_data();   // the MTLBuffer bridge pointer
        out.device = CALIPER_DEV_METAL;
        out.device_index = static_cast<int32_t>(t.device().index());
        return out;
    }
    if (t.is_cuda()) {
        // Unlike MPS there is no buffer-object cast: data_ptr() IS the device
        // address (storage offset already applied), and the Vulkan backend
        // copies from it in-VRAM (ZEROCOPY.md). Contiguity was enforced above;
        // views with a nonzero storage offset are therefore fine here.
        out.data = t.data_ptr();
        out.device = CALIPER_DEV_CUDA;
        out.device_index = static_cast<int32_t>(t.device().index());
        return out;
    }
    // Any other device is not a v1 target for this adapter.
    return std::nullopt;
}

// Same as to_tensor, but first drains the MPS device so the texture the host
// uploads THIS frame reflects every kernel the applet has enqueued. This is the
// v1 correctness story for device textures (sync-then-update). Cost, honestly:
// torch::mps::synchronize() is a full device barrier that blocks the CPU until
// all MPS streams complete — the price of a stream-free v1 ABI. CPU tensors
// need no sync, so the call is skipped for them.
inline std::optional<CaliperTensor> synced_to_tensor(const at::Tensor& t) {
    if (t.is_mps()) torch::mps::synchronize();
    if (t.is_cuda()) torch::cuda::synchronize();   // same contract, CUDA form
    return to_tensor(t);
}

// M2 (docs/metal-pipelining.md §4, D24): hand over ORDER instead of a drained
// device. When the host's bridge-v1.1 caps carry
// CALIPER_BRIDGE_CAP_STREAM_ORDERED, populate t.stream with the producer's
// stream/queue and SKIP the full-device drain; the renderer GPU-orders its
// update after the producer's already-enqueued work. Without the bit (v1
// host, GL fallback, headless) this is exactly synced_to_tensor — the adapter
// never skips a drain the host didn't promise to replace. Thread-safety story
// unchanged from v1: the caller still hands over a tensor it owns at a
// quiescent point in its own logic (spec §4, last paragraph).
inline std::optional<CaliperTensor> stream_to_tensor(const at::Tensor& t,
                                                     uint32_t bridge_caps) {
    if (!(bridge_caps & CALIPER_BRIDGE_CAP_STREAM_ORDERED))
        return synced_to_tensor(t);
#if defined(__APPLE__)
    if (t.is_mps()) {
        auto out = to_tensor(t);
        if (!out) return out;
        // Queue BEFORE commit: torch may release the command-buffer object
        // once the GPU retires it; the queue lives for the process.
        void* queue = detail::mtl_command_queue_of(torch::mps::get_command_buffer());
        if (queue == nullptr) { torch::mps::synchronize(); return out; }
        // Enqueue, not drain: pending torch kernels become committed GPU work
        // the renderer's producer-queue signal is ordered after (M2b).
        torch::mps::commit();
        out->stream = queue;
        return out;
    }
#endif
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)
    if (t.is_cuda()) {
        auto out = to_tensor(t);
        if (!out) return out;
        // Stream order puts the renderer's DtoD copy after the producer's
        // kernels — torch::cuda::synchronize() elided entirely (M2a).
        out->stream = (void*)at::cuda::getCurrentCUDAStream(
                          t.device().index()).stream();
        return out;
    }
#endif
    return synced_to_tensor(t);   // CPU: no sync needed; unknown devices drain
}

}  // namespace caliper::adapters
