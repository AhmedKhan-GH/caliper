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
 *     needlessly per frame.
 */
#include <optional>

#include <caliper/tensor.h>
#include <torch/torch.h>
#if defined(__APPLE__)
#include <dispatch/dispatch.h>  // dispatch_sync_f: serialize on torch's MPS stream queue
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
// torch::mps::synchronize() is NOT internally serialized in this libtorch
// (verified by disassembly: deviceSynchronize tail-calls MPSStream::synchronize
// — straight-line objc_msgSends, no dispatch_sync), while every torch-internal
// kernel encode runs as a block on get_dispatch_queue() (its documented purpose
// is exactly this synchronization). Draining from the frame thread while a
// training thread encodes therefore corrupts the MPSCommandBuffer/encoder
// state — MPS aborts with 'commit an already committed command buffer' (the
// EmbedScope SIGABRT). Run the drain as ONE block on torch's stream dispatch
// queue, atomic with worker encodes. The same disassembly makes nesting
// synchronize() inside the block deadlock-free: it never dispatches. Plain C
// dispatch API — no ObjC blocks — keeps the header .cpp-compilable.
inline void mps_sync_block(void*) { torch::mps::synchronize(); }
inline void mps_synchronize_serialized() {
    if (void* dq = torch::mps::get_dispatch_queue())
        dispatch_sync_f(static_cast<dispatch_queue_t>(dq), nullptr,
                        &mps_sync_block);
    else
        torch::mps::synchronize();
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
#if defined(__APPLE__)
    // Serialized on torch's MPS stream dispatch queue — a bare synchronize()
    // races concurrent training-thread encodes (see the detail helper).
    if (t.is_mps()) detail::mps_synchronize_serialized();
#else
    if (t.is_mps()) torch::mps::synchronize();
#endif
    if (t.is_cuda()) torch::cuda::synchronize();   // same contract, CUDA form
    return to_tensor(t);
}

}  // namespace caliper::adapters
