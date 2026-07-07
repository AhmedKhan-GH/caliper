#pragma once
/* caliper/adapters/exportable_pool.hpp — a torch MemPool whose blocks are
 * cuMemCreate'd with a shareable handle type, so the bridge can IMPORT them
 * (tensor_bridge.v1_2) and update textures with zero copies of the data.
 * Applet-side only (D11). Every failure degrades to "not pool-backed": the
 * caller falls back to the v1/v1.1 path (stream_to_tensor + update_texture).
 *
 * The file has three clearly-marked sections:
 *   §1  Self-contained CUDA-driver VMM mini-loader (nvcuda.dll / libcuda.so.1).
 *       Resolves ONLY the nine cuMem* symbols the pool needs — no toolkit link,
 *       no host-internal headers (src/host/*), no cuInit (torch owns the ctx).
 *   §2  ExportablePool — a CUDAPluggableAllocator whose alloc fn cuMemCreate's
 *       shareable blocks + registers them; a c10::cuda::MemPool over it; a
 *       Scope RAII that routes in-scope tensor allocations into the pool.
 *   §3  to_bridge() glue — resolves a tensor's device ptr in the AllocRegistry,
 *       imports its block into the bridge once (cached by base; negative-cached
 *       on import failure), returns BridgeRef{alloc, offset}.
 *
 * Torch 2.5.1 API surface this is written against (verified in the local
 * libtorch, file:line):
 *   - torch/csrc/cuda/CUDAPluggableAllocator.h:51  createCustomAllocator(
 *         std::function<void*(size_t,int,cudaStream_t)>,
 *         std::function<void(void*,size_t,int,cudaStream_t)>)
 *   - c10/cuda/CUDACachingAllocator.h:461  c10::cuda::MemPool(CUDAAllocator*,bool)
 *   - c10/cuda/CUDACachingAllocator.h:466  MemPool::id() -> MempoolId_t
 *   - c10/cuda/CUDACachingAllocator.h:479  MemPoolContext(MemPool*) RAII
 *   - c10/cuda/CUDACachingAllocator.h:372  beginAllocateToPool(DeviceIndex,
 *         MempoolId_t, std::function<bool(cudaStream_t)>)
 *   - c10/cuda/CUDACachingAllocator.h:379  endAllocateToPool(DeviceIndex,MempoolId_t)
 *   - c10/cuda/CUDACachingAllocator.h:417  releasePool(DeviceIndex,MempoolId_t)
 * The custom-allocator-per-MemPool mechanism itself is documented at
 * CUDACachingAllocator.h:455-493 ("route allocations to a user provided
 * allocator ... active_pool->allocator()->raw_alloc(size)"). Runtime routing is
 * validated on hardware in Task 6; this header is compiled out on macOS.
 */
#include <caliper/adapters/alloc_registry.hpp>
#include <caliper/caliper.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

// Same guard discipline as adapters/torch.hpp:37 — mac libtorch ships the
// c10/cuda headers but no CUDA toolkit, and Apple torch never has CUDA, so the
// whole interior is compiled out here; the fallback class below keeps applet
// code that mentions ExportablePool compiling on every platform.
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<c10/cuda/CUDACachingAllocator.h>)
#define CALIPER_EXPORTABLE_POOL_CUDA 1
#else
#define CALIPER_EXPORTABLE_POOL_CUDA 0
#endif

/* Windows compile fix (first build of this TU): these includes lived inside
 * namespace caliper::adapters, which nested c10/torch into
 * caliper::adapters::c10 and broke every torch declaration. Includes must
 * precede the namespace. */
#if CALIPER_EXPORTABLE_POOL_CUDA
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <map>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#endif  // CALIPER_EXPORTABLE_POOL_CUDA

namespace caliper::adapters {

// A tensor located inside a pool block, expressed the way the v1.2 bridge wants
// it: an imported-allocation id plus the tensor's byte offset within it.
struct BridgeRef {
    CaliperAllocId alloc;
    uint64_t       offset;
};

#if CALIPER_EXPORTABLE_POOL_CUDA
// ===========================================================================
// §1  Self-contained CUDA driver VMM mini-loader
// ===========================================================================
// Structs/enums mirror the stable driver ABI in cuda.h (VMM APIs, CUDA 10.2+),
// same reserved-padding style as src/host/cuda_driver.h but self-contained: the
// SDK must not include host-internal headers. Only the layouts the pool uses.
// (Includes for this section are hoisted above the namespace — see top.)

namespace detail {

using CUresult                     = int;                 // CUDA_SUCCESS == 0
using CUdeviceptr                  = unsigned long long;  // cuda.h
using CUmemGenericAllocationHandle = unsigned long long;  // cuda.h
constexpr CUresult kCudaSuccess = 0;

// CUmemAllocationType (cuda.h).
enum CUmemAllocationType : unsigned int {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED  = 0x1,
};
// CUmemAllocationHandleType (cuda.h): POSIX fd == 1, opaque Win32 == 2.
enum CUmemAllocationHandleType : unsigned int {
    CU_MEM_HANDLE_TYPE_NONE                  = 0x0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
    CU_MEM_HANDLE_TYPE_WIN32                 = 0x2,
};
// CUmemLocationType (cuda.h).
enum CUmemLocationType : unsigned int {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE  = 0x1,
};
// CUmemAllocationGranularity_flags (cuda.h).
enum CUmemAllocationGranularity_flags : unsigned int {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM     = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1,
};
// CUmemAccess_flags (cuda.h): READWRITE == 0x3.
enum CUmemAccess_flags : unsigned int {
    CU_MEM_ACCESS_FLAGS_PROT_NONE      = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ      = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
};

// CUmemLocation (cuda.h).
struct CUmemLocation {
    CUmemLocationType type;
    int               id;
};
// CUmemAllocationProp (cuda.h, v1 layout). The trailing allocFlags block keeps
// the reserved bytes so the driver reads the whole struct.
struct CUmemAllocationProp {
    CUmemAllocationType       type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation             location;
    void*                     win32HandleMetaData;
    struct {
        unsigned char  compressionType;
        unsigned char  gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char  reserved[4];
    } allocFlags;
};
// CUmemAccessDesc (cuda.h).
struct CUmemAccessDesc {
    CUmemLocation     location;
    CUmemAccess_flags flags;
};

// The nine driver entry points — and ONLY these nine.
struct CuVmmApi {
    CUresult (*cuMemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*,
                                              CUmemAllocationGranularity_flags);
    CUresult (*cuMemCreate)(CUmemGenericAllocationHandle*, size_t,
                            const CUmemAllocationProp*, unsigned long long);
    CUresult (*cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr,
                                    unsigned long long);
    CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t,
                         CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult (*cuMemExportToShareableHandle)(void*, CUmemGenericAllocationHandle,
                                             CUmemAllocationHandleType,
                                             unsigned long long);
    CUresult (*cuMemUnmap)(CUdeviceptr, size_t);
    CUresult (*cuMemRelease)(CUmemGenericAllocationHandle);
    CUresult (*cuMemAddressFree)(CUdeviceptr, size_t);
};

inline void* cu_lib_open() {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#else
    return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}
inline void* cu_lib_sym(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

// Resolves the table once for the process. Any missing symbol → nullptr, which
// the pool reports as ok() == false so callers fall back — never a null call.
inline const CuVmmApi* load_cu_vmm() {
    static const CuVmmApi* cached = [] () -> const CuVmmApi* {
        void* lib = cu_lib_open();
        if (!lib) return nullptr;
        static CuVmmApi a{};
        struct Entry { void** slot; const char* name; };
        const Entry entries[] = {
            {reinterpret_cast<void**>(&a.cuMemGetAllocationGranularity),
                 "cuMemGetAllocationGranularity"},
            {reinterpret_cast<void**>(&a.cuMemCreate),                  "cuMemCreate"},
            {reinterpret_cast<void**>(&a.cuMemAddressReserve),          "cuMemAddressReserve"},
            {reinterpret_cast<void**>(&a.cuMemMap),                     "cuMemMap"},
            {reinterpret_cast<void**>(&a.cuMemSetAccess),               "cuMemSetAccess"},
            {reinterpret_cast<void**>(&a.cuMemExportToShareableHandle), "cuMemExportToShareableHandle"},
            {reinterpret_cast<void**>(&a.cuMemUnmap),                   "cuMemUnmap"},
            {reinterpret_cast<void**>(&a.cuMemRelease),                 "cuMemRelease"},
            {reinterpret_cast<void**>(&a.cuMemAddressFree),             "cuMemAddressFree"},
        };
        for (const Entry& e : entries) {
            *e.slot = cu_lib_sym(lib, e.name);
            if (*e.slot == nullptr) return nullptr;   // hole → whole load fails
        }
        return &a;
    }();
    return cached;
}

}  // namespace detail

// ===========================================================================
// §2  ExportablePool — shareable-handle MemPool
// ===========================================================================
class ExportablePool {
public:
    // RAII: tensors allocated while a Scope is alive land in the pool. Returned
    // by value from use() (guaranteed elision, C++17); non-copyable/movable so
    // the begin/end pool pair is balanced exactly once.
    class Scope {
    public:
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&)                 = delete;
        Scope& operator=(Scope&&)      = delete;
        ~Scope() {
            if (owner_) {
                c10::cuda::CUDACachingAllocator::endAllocateToPool(
                    static_cast<c10::DeviceIndex>(owner_->device_),
                    owner_->pool_->id());
            }
            // ctx_ destructor pops the MemPoolContext (restores the prior pool).
        }
    private:
        friend class ExportablePool;
        explicit Scope(ExportablePool* owner) {
            if (!owner || !owner->ok_) return;
            owner_ = owner;
            ctx_ = std::make_unique<c10::cuda::MemPoolContext>(owner->pool_.get());
            c10::cuda::CUDACachingAllocator::beginAllocateToPool(
                static_cast<c10::DeviceIndex>(owner->device_), owner->pool_->id(),
                [](cudaStream_t) { return true; });   // route every stream
        }
        ExportablePool*                                 owner_ = nullptr;
        std::unique_ptr<c10::cuda::MemPoolContext>      ctx_;
    };

    explicit ExportablePool(int device_index) : device_(device_index) {
        cu_ = detail::load_cu_vmm();
        if (!cu_) return;                     // no driver / missing symbol
        // Hardware finding (cold start): torch's CUDA caching allocator
        // initializes lazily on the first CUDA op; beginAllocateToPool on the
        // uninitialized per-device table terminates the process when the pool
        // scope is the process's FIRST CUDA use. Force init here.
        if (!torch::cuda::is_available()) return;   // ok_ stays false -> fallback
        at::globalContext().lazyInitCUDA();
        allocator_ = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
            [this](size_t size, int device, cudaStream_t stream) {
                return alloc_block(size, device, stream);
            },
            [this](void* ptr, size_t, int, cudaStream_t) { free_block(ptr); });
        if (!allocator_) return;
        pool_ = std::make_unique<c10::cuda::MemPool>(allocator_.get(),
                                                     /*is_user_created=*/true);
        ok_ = true;
    }

    ~ExportablePool() {
        if (ok_ && pool_) {
            // Drop torch's bookkeeping for this private pool; then reclaim any
            // block the allocator still holds (frees VA + shareable handle).
            c10::cuda::CUDACachingAllocator::releasePool(
                static_cast<c10::DeviceIndex>(device_), pool_->id());
        }
        std::lock_guard<std::mutex> lock(mu_);
        // The destructor is a frame-thread call site, so bridge releases are
        // legal here: drain whatever the allocator callbacks queued, then
        // release the import of every block torch still holds before
        // reclaiming it. Whether releasePool freed cached segments
        // synchronously (through free_block) is a torch internal this code
        // deliberately does not depend on — both orders end fully released.
        if (import_bridge_) {
            for (CaliperAllocId id : pending_releases_)
                import_bridge_->release_allocation(id);
        }
        pending_releases_.clear();
        for (auto& [base, blk] : blocks_) {
            if (auto ci = import_cache_.find(base); ci != import_cache_.end() &&
                ci->second != 0 && import_bridge_)
                import_bridge_->release_allocation(ci->second);
            reclaim(blk);
        }
        blocks_.clear();
        import_cache_.clear();
    }

    ExportablePool(const ExportablePool&)            = delete;
    ExportablePool& operator=(const ExportablePool&) = delete;

    bool                 ok() const       { return ok_; }
    const AllocRegistry& registry() const { return registry_; }
    Scope                use()            { return Scope(this); }

    // §3 to_bridge — see below the class.
    std::optional<BridgeRef> to_bridge(caliper::Bridge& bridge, const at::Tensor& t);

private:
    // Handle type selected by platform, once at both the driver-export and the
    // bridge-import call so they always agree.
#ifdef _WIN32
    static constexpr detail::CUmemAllocationHandleType kHandleType =
        detail::CU_MEM_HANDLE_TYPE_WIN32;
    static constexpr uint32_t kBridgeHandleType = CALIPER_ALLOC_HANDLE_OPAQUE_WIN32;
#else
    static constexpr detail::CUmemAllocationHandleType kHandleType =
        detail::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    static constexpr uint32_t kBridgeHandleType = CALIPER_ALLOC_HANDLE_OPAQUE_FD;
#endif

    struct Block {
        detail::CUdeviceptr                  va;
        size_t                               size;   // granularity-padded
        detail::CUmemGenericAllocationHandle mem;
        void*                                os_handle;
    };

    static void close_os_handle(void* h) {
#ifdef _WIN32
        if (h) CloseHandle(reinterpret_cast<HANDLE>(h));
#else
        const int fd = static_cast<int>(reinterpret_cast<intptr_t>(h));
        if (fd >= 0) ::close(fd);
#endif
    }

    // Full teardown of one block (caller holds mu_): unmap, release, free VA,
    // close the shareable handle. Order mirrors the driver's ownership graph.
    void reclaim(const Block& b) {
        cu_->cuMemUnmap(b.va, b.size);
        cu_->cuMemRelease(b.mem);
        cu_->cuMemAddressFree(b.va, b.size);
        close_os_handle(b.os_handle);
    }

    // Pluggable-allocator alloc fn: cuMemCreate a shareable block, map it, grant
    // device RW, export the handle, register (base -> {size, handle}). Any
    // driver failure unwinds and returns nullptr (torch treats it as OOM).
    void* alloc_block(size_t size, int device, cudaStream_t /*stream*/) {
        detail::CUmemAllocationProp prop{};
        prop.type                 = detail::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = kHandleType;
        prop.location.type        = detail::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = device;
#ifdef _WIN32
        // Hardware finding (RTX 500 Ada, driver 596.47): cuMemCreate REJECTS a
        // WIN32-shareable request with null win32HandleMetaData
        // (CUDA_ERROR_INVALID_VALUE). An exportable NT handle needs
        // SECURITY_ATTRIBUTES (the CUDA memMapIPCDrv sample's precedent); the
        // default descriptor is enough for same-user DuplicateHandle import.
        static SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE};
        prop.win32HandleMetaData = &sa;
#endif

        size_t gran = 0;
        if (cu_->cuMemGetAllocationGranularity(
                &gran, &prop, detail::CU_MEM_ALLOC_GRANULARITY_MINIMUM)
                != detail::kCudaSuccess || gran == 0)
            return nullptr;
        const size_t padded = ((size + gran - 1) / gran) * gran;

        detail::CUmemGenericAllocationHandle mem{};
        if (cu_->cuMemCreate(&mem, padded, &prop, 0) != detail::kCudaSuccess)
            return nullptr;

        detail::CUdeviceptr va = 0;
        if (cu_->cuMemAddressReserve(&va, padded, 0, 0, 0) != detail::kCudaSuccess) {
            cu_->cuMemRelease(mem);
            return nullptr;
        }
        if (cu_->cuMemMap(va, padded, 0, mem, 0) != detail::kCudaSuccess) {
            cu_->cuMemAddressFree(va, padded);
            cu_->cuMemRelease(mem);
            return nullptr;
        }
        detail::CUmemAccessDesc acc{};
        acc.location.type = detail::CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id   = device;
        acc.flags         = detail::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cu_->cuMemSetAccess(va, padded, &acc, 1) != detail::kCudaSuccess) {
            cu_->cuMemUnmap(va, padded);
            cu_->cuMemAddressFree(va, padded);
            cu_->cuMemRelease(mem);
            return nullptr;
        }

        // Export the shareable handle: POSIX writes an int fd, Win32 a HANDLE.
        void* os_handle = nullptr;
#ifdef _WIN32
        if (cu_->cuMemExportToShareableHandle(&os_handle, mem, kHandleType, 0)
                != detail::kCudaSuccess) {
            cu_->cuMemUnmap(va, padded);
            cu_->cuMemAddressFree(va, padded);
            cu_->cuMemRelease(mem);
            return nullptr;
        }
#else
        int fd = -1;
        if (cu_->cuMemExportToShareableHandle(&fd, mem, kHandleType, 0)
                != detail::kCudaSuccess) {
            cu_->cuMemUnmap(va, padded);
            cu_->cuMemAddressFree(va, padded);
            cu_->cuMemRelease(mem);
            return nullptr;
        }
        os_handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif

        void* ptr = reinterpret_cast<void*>(va);
        const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
        {
            std::lock_guard<std::mutex> lock(mu_);
            blocks_[base] = Block{va, padded, mem, os_handle};
        }
        registry_.add(base, padded, os_handle);
        return ptr;
    }

    // Pluggable-allocator free fn: reverse of alloc_block. Deregisters from the
    // AllocRegistry, QUEUES the host-import release, then reclaims the driver
    // resources and closes the shareable handle. Queued, not called: torch may
    // invoke this callback from any thread, and the Bridge is frame-thread-only
    // by contract — the queue drains in to_bridge()/~ExportablePool, both
    // applet frame-thread call sites.
    void free_block(void* ptr) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = blocks_.find(base);
        if (it == blocks_.end()) return;          // not one of ours
        const Block b = it->second;
        blocks_.erase(it);
        registry_.remove(base);
        if (auto ci = import_cache_.find(base); ci != import_cache_.end()) {
            if (ci->second != 0)
                pending_releases_.push_back(ci->second);
            import_cache_.erase(ci);              // clear the per-base cache
        }
        reclaim(b);
    }

    const detail::CuVmmApi* cu_      = nullptr;
    int                     device_  = 0;
    bool                    ok_      = false;

    AllocRegistry registry_;
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator_;
    std::unique_ptr<c10::cuda::MemPool>                             pool_;

    std::mutex                         mu_;      // guards the maps below
    std::map<uintptr_t, Block>         blocks_;
    std::map<uintptr_t, CaliperAllocId> import_cache_;  // base -> id (0 = negative)
    std::vector<CaliperAllocId>        pending_releases_;  // freed off-thread
    caliper::Bridge*                   import_bridge_ = nullptr;
};

// ===========================================================================
// §3  to_bridge glue
// ===========================================================================
inline std::optional<BridgeRef>
ExportablePool::to_bridge(caliper::Bridge& bridge, const at::Tensor& t) {
    if (!ok_) return std::nullopt;
    const uint64_t extent =
        static_cast<uint64_t>(t.numel()) * static_cast<uint64_t>(t.element_size());
    auto hit = registry_.find(t.data_ptr(), extent);
    if (!hit) return std::nullopt;               // tensor is not pool-backed

    std::lock_guard<std::mutex> lock(mu_);
    import_bridge_ = &bridge;                     // remembered for the drains
    // Frame-thread call site: drain releases queued by off-thread frees.
    for (CaliperAllocId id : pending_releases_) bridge.release_allocation(id);
    pending_releases_.clear();
    // Re-validate under mu_: a free_block between registry_.find and here
    // means the handle is gone — and a FUTURE block may reuse this base, so
    // caching a negative for it would silently disable zero-copy forever.
    if (blocks_.find(hit->base) == blocks_.end()) return std::nullopt;
    auto it = import_cache_.find(hit->base);
    if (it == import_cache_.end()) {
        // Import once per block. 0 = host declined; cache it as a permanent
        // negative for this block so we never re-ask (caller stays on fallback).
        const CaliperAllocId id =
            bridge.import_allocation(hit->os_handle, hit->size, kBridgeHandleType);
        it = import_cache_.emplace(hit->base, id).first;
    }
    if (it->second == 0) return std::nullopt;     // negative-cached
    return BridgeRef{it->second, hit->offset};
}

#else   // !CALIPER_EXPORTABLE_POOL_CUDA
// ===========================================================================
// Fallback: the class exists everywhere so applet code compiles, but the pool
// is never backed — ok() == false, use() is a no-op, to_bridge always declines.
// ===========================================================================
class ExportablePool {
public:
    class Scope { public: Scope() = default; };

    explicit ExportablePool(int /*device_index*/) {}

    bool                 ok() const       { return false; }
    const AllocRegistry& registry() const { return registry_; }
    Scope                use()            { return Scope{}; }
    std::optional<BridgeRef> to_bridge(caliper::Bridge& /*bridge*/,
                                       const at::Tensor& /*t*/) {
        return std::nullopt;
    }

private:
    AllocRegistry registry_;
};
#endif  // CALIPER_EXPORTABLE_POOL_CUDA

}  // namespace caliper::adapters
