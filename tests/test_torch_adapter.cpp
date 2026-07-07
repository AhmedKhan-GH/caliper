// Unit tests for the applet-side torch adapter (sdk/include/caliper/adapters/
// torch.hpp). This target links libtorch, so it lives in its OWN binary
// (caliper_torch_tests, ctest label "torch") — never in caliper_tests, to keep
// the fast unit suite free of the torch link cost.
//
// CPU assertions run everywhere. The MPS branch only asserts under
// torch::mps::is_available(); on a machine without an MPS device those cases
// report a MESSAGE and pass, so the label stays green rather than red.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <caliper/adapters/torch.hpp>
#include <caliper/adapters/exportable_pool.hpp>

#include <torch/torch.h>

// Mirrors the adapter's guard (adapters/torch.hpp): the CUDA cases drive the
// handoff from a non-default pool stream, which needs the c10 stream API.
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <atomic>
#include <thread>

using caliper::adapters::to_tensor;
using caliper::adapters::synced_to_tensor;
using caliper::adapters::stream_to_tensor;

TEST_CASE("cpu f32 2D round-trips field-by-field, zero-copy") {
    // A contiguous (3,4) f32 CPU tensor with known values.
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4}).contiguous();

    auto ct = to_tensor(t);
    REQUIRE(ct.has_value());
    CHECK(ct->struct_size == sizeof(CaliperTensor));
    CHECK(ct->data == t.data_ptr());          // aliases, no copy
    CHECK(ct->dtype == CALIPER_DT_F32);
    CHECK(ct->ndim == 2);
    CHECK(ct->shape[0] == 3);
    CHECK(ct->shape[1] == 4);
    CHECK(ct->strides[0] == 4);               // elements, row-major
    CHECK(ct->strides[1] == 1);
    CHECK(ct->device == CALIPER_DEV_CPU);
    CHECK(ct->device_index == 0);
    CHECK(ct->stream == nullptr);
}

TEST_CASE("non-contiguous cpu tensor is rejected (adapter never copies)") {
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4});
    torch::Tensor view = t.t();               // transpose -> non-contiguous view
    REQUIRE_FALSE(view.is_contiguous());
    CHECK_FALSE(to_tensor(view).has_value());

    // The exemplar repair: caller makes the copy visible with .contiguous().
    auto fixed = to_tensor(view.contiguous());
    REQUIRE(fixed.has_value());
    CHECK(fixed->shape[0] == 4);
    CHECK(fixed->shape[1] == 3);
}

TEST_CASE("dtype map matrix: the six v1 dtypes map, others reject") {
    struct Case { torch::ScalarType st; CaliperDType dt; };
    const Case ok[] = {
        {torch::kFloat,    CALIPER_DT_F32},
        {torch::kHalf,     CALIPER_DT_F16},
        {torch::kBFloat16, CALIPER_DT_BF16},
        {torch::kLong,     CALIPER_DT_I64},
        {torch::kInt,      CALIPER_DT_I32},
        {torch::kByte,     CALIPER_DT_U8},
    };
    for (const auto& c : ok) {
        auto ct = to_tensor(torch::zeros({2, 2}, torch::dtype(c.st)));
        REQUIRE(ct.has_value());
        CHECK(ct->dtype == c.dt);
    }
    // Unsupported dtypes are rejected, not coerced.
    CHECK_FALSE(to_tensor(torch::zeros({2, 2}, torch::kDouble)).has_value());
    CHECK_FALSE(to_tensor(torch::zeros({2, 2}, torch::kBool)).has_value());
}

TEST_CASE("ndim > 8 is rejected") {
    auto t9 = torch::zeros({1, 1, 1, 1, 1, 1, 1, 1, 1}, torch::kFloat);  // 9 dims
    REQUIRE(t9.dim() == 9);
    CHECK_FALSE(to_tensor(t9).has_value());

    auto t8 = torch::zeros({1, 1, 1, 1, 1, 1, 1, 2}, torch::kFloat);     // 8 dims
    REQUIRE(t8.dim() == 8);
    CHECK(to_tensor(t8).has_value());
}

TEST_CASE("mps tensor -> METAL device, storage().data pointer, matches C5 shape") {
    if (!torch::mps::is_available()) {
        MESSAGE("no MPS device — skipping MPS adapter cases");
        return;
    }
    // Contiguous (H,W) f32 on MPS — the exact shape C5 builds a METAL-device
    // CaliperTensor from (ndim=2, shape={h,w}, strides={w,1}).
    torch::Tensor t = torch::arange(20, torch::TensorOptions().dtype(torch::kFloat)
                                    .device(torch::kMPS))
                          .reshape({4, 5}).contiguous();
    auto ct = to_tensor(t);
    REQUIRE(ct.has_value());
    CHECK(ct->device == CALIPER_DEV_METAL);
    CHECK(ct->data == t.storage().mutable_data());   // the MTLBuffer bridge ptr
    CHECK(ct->dtype == CALIPER_DT_F32);
    CHECK(ct->ndim == 2);
    CHECK(ct->shape[0] == 4);
    CHECK(ct->shape[1] == 5);
    CHECK(ct->strides[0] == 5);
    CHECK(ct->strides[1] == 1);
    CHECK(ct->stream == nullptr);
}

TEST_CASE("mps view with nonzero storage_offset is rejected") {
    if (!torch::mps::is_available()) {
        MESSAGE("no MPS device — skipping MPS adapter cases");
        return;
    }
    torch::Tensor base = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat)
                                       .device(torch::kMPS)).reshape({3, 4});
    // Row 1 onward: a contiguous slice, but its storage starts at a nonzero
    // element offset — no offset channel in CaliperTensor, so reject.
    torch::Tensor sliced = base.slice(/*dim=*/0, /*start=*/1, /*end=*/3);
    REQUIRE(sliced.is_contiguous());
    REQUIRE(sliced.storage_offset() != 0);
    CHECK_FALSE(to_tensor(sliced).has_value());

    // Clone materializes a fresh offset-0 buffer -> accepted (the repair path).
    auto fixed = to_tensor(sliced.clone());
    REQUIRE(fixed.has_value());
    CHECK(fixed->device == CALIPER_DEV_METAL);
}

TEST_CASE("synced_to_tensor matches to_tensor for a cpu tensor (no sync needed)") {
    torch::Tensor t = torch::ones({2, 3}, torch::kFloat).contiguous();
    auto a = synced_to_tensor(t);
    auto b = to_tensor(t);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    CHECK(a->data == b->data);
    CHECK(a->device == CALIPER_DEV_CPU);
}

TEST_CASE("stream_to_tensor: no caps bit -> exactly the v1 drained handoff (stream NULL)") {
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4}).contiguous();
    auto ct = stream_to_tensor(t, 0);
    REQUIRE(ct.has_value());
    CHECK(ct->stream == nullptr);
    CHECK(ct->device == CALIPER_DEV_CPU);
    CHECK(ct->data == t.data_ptr());
}

TEST_CASE("stream_to_tensor: cpu tensor never carries a stream, even when honored") {
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4}).contiguous();
    auto ct = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    REQUIRE(ct.has_value());
    CHECK(ct->stream == nullptr);
}

TEST_CASE("stream_to_tensor: mps tensor carries the producer queue when honored; drains when not") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device — skipping"); return; }
    torch::Tensor t = torch::ones({4, 4},
        torch::TensorOptions().device(torch::kMPS)) * 2.0f;

    auto honored = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    REQUIRE(honored.has_value());
    CHECK(honored->device == CALIPER_DEV_METAL);
    CHECK(honored->stream != nullptr);           // the MTLCommandQueue*

    auto v1 = stream_to_tensor(t, 0);            // negotiation pin, other direction
    REQUIRE(v1.has_value());
    CHECK(v1->stream == nullptr);
}

TEST_CASE("stream_to_tensor: cuda tensor carries the producer stream when honored; drains when not") {
    if (!torch::cuda::is_available()) { MESSAGE("no CUDA device — skipping"); return; }
#if defined(__APPLE__)
    // Unreachable: Apple torch never reports CUDA. The early return above
    // keeps this TU compiling without the c10 CUDA headers.
#elif !__has_include(<c10/cuda/CUDAStream.h>)
    // The finding-1 tripwire must fire LOUDLY, never skip: if this header
    // vanished from the include path, the adapter's guard compiled its CUDA
    // branch out too — the drain is silently back and the speedup fictional.
    REQUIRE_MESSAGE(false, "c10/cuda/CUDAStream.h not found — the adapter's "
                           "CUDA branch is compiled out on a CUDA machine");
#else
    torch::Tensor t = torch::ones({4, 4},
        torch::TensorOptions().device(torch::kCUDA)) * 2.0f;

    // On the DEFAULT current stream the handoff is honored but the handle it
    // carries is NULL — CUDAStream::stream() returns nullptr for the legacy
    // default stream by CUDA semantics (unlike MPS, whose queue pointer is
    // never NULL). That is still a correct, drain-elided handoff: the
    // renderer's NULL rung enqueues on the same legacy default stream the
    // producer used, so the copy stays stream-ordered after its kernels.
    auto honored_default = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    REQUIRE(honored_default.has_value());
    REQUIRE(honored_default->device == CALIPER_DEV_CUDA);

    // NULL therefore can't distinguish "default stream" from "branch compiled
    // out", so the tripwire pins a NON-default pool stream: a compiled-in
    // branch must hand back exactly that stream's handle, which is never NULL.
    {
        auto pool = c10::cuda::getStreamFromPool(false, t.device().index());
        c10::cuda::CUDAStreamGuard guard(pool);
        auto honored = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
        REQUIRE(honored.has_value());
        REQUIRE(honored->device == CALIPER_DEV_CUDA);
        CHECK_FALSE(honored->stream == nullptr);   // FAILS if the guard compiled the branch out
        REQUIRE(honored->stream == (void*)pool.stream());
    }

    auto v1 = stream_to_tensor(t, 0);          // negotiation pin, other direction
    REQUIRE(v1.has_value());
    CHECK_FALSE(v1->stream != nullptr);
#endif
}

TEST_CASE("stream_to_tensor: handoff survives a concurrently-encoding training thread") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device — skipping"); return; }
    // Regression for the EmbedScope SIGABRT (MPSCore MPSPredicate.mm 'command
    // buffer already committed. State: 4'): the frame thread performs stream
    // handoffs while a worker thread — the training loop — continuously
    // enqueues MPS kernels. torch::mps::get_command_buffer() is a bare,
    // UNSERIALIZED accessor of MPSStream::_commandBuffer, so calling it here
    // raced the worker's encode/commitAndContinue blocks and MPS aborted
    // within a few hundred handoffs. The fix snapshots the producer queue
    // inside a dispatch_sync block on torch's stream dispatch queue — the
    // documented synchronization point every torch-internal encode also uses.
    std::atomic<bool> stop{false};
    std::thread worker([&] {
        torch::Tensor a = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kMPS));
        torch::Tensor b = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kMPS));
        while (!stop.load(std::memory_order_relaxed))
            a = torch::mm(a, b).tanh();          // endless kernel stream
    });
    torch::Tensor t = torch::ones({8, 8}, torch::TensorOptions().device(torch::kMPS));
    for (int i = 0; i < 500; ++i) {
        auto ct = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
        REQUIRE(ct.has_value());
        CHECK_FALSE(ct->stream == nullptr);
    }
    stop.store(true, std::memory_order_relaxed);
    worker.join();
    torch::mps::synchronize();                   // leave the device quiet for other cases
}

TEST_CASE("synced_to_tensor: drain survives a concurrently-encoding training thread") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device — skipping"); return; }
    // The drain-path twin of the stress case above (the fix backported to main
    // as 8b0a010, kept through the metal-pipelining merge): the frame thread
    // drains via synced_to_tensor — the no-caps / GL-fallback route — while a
    // worker thread continuously enqueues MPS kernels.
    // torch::mps::synchronize() is just as unserialized as get_command_buffer()
    // (deviceSynchronize tail-calls MPSStream::synchronize — straight-line
    // objc_msgSends), so a bare drain corrupts the MPSCommandBuffer state the
    // same way: SIGABRT on either thread, or an AGX encoder-coalescing SIGSEGV.
    std::atomic<bool> stop{false};
    std::thread worker([&] {
        torch::Tensor a = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kMPS));
        torch::Tensor b = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kMPS));
        while (!stop.load(std::memory_order_relaxed))
            a = torch::mm(a, b).tanh();          // endless kernel stream
    });
    torch::Tensor t = torch::ones({8, 8}, torch::TensorOptions().device(torch::kMPS));
    for (int i = 0; i < 500; ++i) {
        auto ct = synced_to_tensor(t);
        REQUIRE(ct.has_value());
        CHECK(ct->stream == nullptr);            // drained handoff carries no stream
    }
    stop.store(true, std::memory_order_relaxed);
    worker.join();
    torch::mps::synchronize();                   // leave the device quiet for other cases
}

TEST_CASE("exportable pool: allocations are pool-backed, registry-resolvable, "
          "and export a shareable handle" * doctest::skip(!torch::cuda::is_available())) {
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<c10/cuda/CUDACachingAllocator.h>)
    caliper::adapters::ExportablePool pool(0);
    REQUIRE_MESSAGE(pool.ok(), "cuMemCreate-backed pool failed on a CUDA machine "
                               "— VMM or export unsupported by this driver?");
    at::Tensor t;
    {
        auto scope = pool.use();
        t = torch::rand({17, 9}, torch::TensorOptions()
                                     .device(torch::kCUDA).dtype(torch::kFloat32));
        // a DERIVED tensor inside the scope must also land in the pool:
        t = t.square().contiguous();
    }
    auto hit = pool.registry().find(t.data_ptr(), t.numel() * sizeof(float));
    REQUIRE_MESSAGE(hit.has_value(),
        "pool-scoped tensor not resolvable in the AllocRegistry — "
        "MemPool routing broke (torch 2.5.1 API drift?)");
    CHECK(hit->os_handle != nullptr);
    CHECK(hit->size >= t.numel() * sizeof(float));
#else
    REQUIRE_MESSAGE(!torch::cuda::is_available(),
        "CUDA machine but the exportable-pool branch is compiled out");
#endif
}

TEST_CASE("mps exportable pool: storage_ref extracts (buffer, size, offset) "
          "from tensor storage" * doctest::skip(!torch::mps::is_available())) {
#if defined(__APPLE__)
    caliper::adapters::ExportablePool pool(0);
    REQUIRE(pool.ok());

    auto t = torch::rand({17, 9}, torch::TensorOptions()
                                      .device(torch::kMPS).dtype(torch::kFloat32));
    auto ref = caliper::adapters::ExportablePool::storage_ref(t);
    REQUIRE(ref.has_value());
    CHECK(ref->buffer == t.storage().mutable_data());   // the id<MTLBuffer> bridge pointer
    CHECK(ref->size >= (uint64_t)t.numel() * sizeof(float));
    CHECK(ref->offset == 0);

    // a slice that shares storage carries its byte offset — this is exactly
    // what the (alloc, offset) addressing fixes vs the offset-rejecting v1 path
    auto view = t.reshape({17 * 9}).slice(0, 9, 18);    // storage_offset 9 elements
    auto vref = caliper::adapters::ExportablePool::storage_ref(view);
    REQUIRE(vref.has_value());
    CHECK(vref->buffer == ref->buffer);
    CHECK(vref->offset == 9 * sizeof(float));

    // rejections: CPU tensor, non-contiguous
    CHECK_FALSE(caliper::adapters::ExportablePool::storage_ref(
        torch::rand({4, 4})).has_value());
    CHECK_FALSE(caliper::adapters::ExportablePool::storage_ref(
        t.transpose(0, 1)).has_value());

    // to_bridge against a null (default) Bridge declines without crashing
    caliper::Bridge nobridge{};
    CHECK_FALSE(pool.to_bridge(nobridge, t).has_value());
#endif
}

TEST_CASE("stream_to_tensor: cuda handoff survives a concurrently-training thread") {
    if (!torch::cuda::is_available()) { MESSAGE("no CUDA device — skipping"); return; }
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)
    // CUDA twin of the MPS stress case above (docs/m2a-windows-verification.md
    // T4, finding 2). On MPS the equivalent race SIGABRTed within a few hundred
    // handoffs because torch's public stream calls are not internally
    // serialized. The CUDA handoff is only a getCurrentCUDAStream() handle
    // read and CUDA driver calls are thread-safe by API contract, so no
    // serialization SHOULD be needed — this case tests that assumption
    // empirically instead of trusting it. The handoff thread runs on a
    // non-default pool stream so the carried handle is assertable (non-NULL,
    // see the tripwire case above) while the worker encodes on its own
    // thread-local current stream.
    std::atomic<bool> stop{false};
    std::thread worker([&] {
        torch::Tensor a = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kCUDA));
        torch::Tensor b = torch::randn({64, 64},
            torch::TensorOptions().device(torch::kCUDA));
        while (!stop.load(std::memory_order_relaxed))
            a = torch::mm(a, b).tanh();          // endless kernel stream
    });
    torch::Tensor t = torch::ones({8, 8}, torch::TensorOptions().device(torch::kCUDA));
    auto pool = c10::cuda::getStreamFromPool(false, t.device().index());
    c10::cuda::CUDAStreamGuard guard(pool);
    for (int i = 0; i < 500; ++i) {
        auto ct = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
        REQUIRE(ct.has_value());
        CHECK_FALSE(ct->stream == nullptr);
    }
    stop.store(true, std::memory_order_relaxed);
    worker.join();
    torch::cuda::synchronize();                  // leave the device quiet for other cases
#endif
}
