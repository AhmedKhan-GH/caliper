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

#include <torch/torch.h>

#include <atomic>
#include <thread>

using caliper::adapters::to_tensor;
using caliper::adapters::synced_to_tensor;

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

TEST_CASE("synced_to_tensor: handoff survives a concurrently-encoding training thread") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device — skipping"); return; }
    // Regression for the EmbedScope SIGABRT ('commit an already committed
    // command buffer', MPSCore MPSPredicate.mm State: 4): the frame thread
    // drains via synced_to_tensor while a worker thread — the training loop —
    // continuously enqueues MPS kernels. torch::mps::synchronize() is NOT
    // internally serialized on torch's MPS stream dispatch queue (verified by
    // disassembly: deviceSynchronize tail-calls MPSStream::synchronize —
    // straight-line objc_msgSends), while every torch-internal encode runs as
    // a block on get_dispatch_queue(). Racing them corrupts the
    // MPSCommandBuffer state and MPS aborts within a few hundred handoffs.
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
    }
    stop.store(true, std::memory_order_relaxed);
    worker.join();
    torch::mps::synchronize();                   // leave the device quiet for other cases
}
