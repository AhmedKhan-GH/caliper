#pragma once
// Minimal CUDA *driver API* surface, loaded from nvcuda.dll / libcuda.so at
// runtime (PLATFORM.md D11: the host links no ML framework and no CUDA
// toolkit — the driver library ships with the GPU driver itself, so this adds
// zero build- or run-time dependencies on machines without NVIDIA hardware).
//
// Consumers: device_query_cuda.cpp (detection) and vulkan_renderer.cpp
// (external-memory import + the one in-VRAM copy of the zero-copy path).
//
// The typedefs below mirror cuda.h's stable driver ABI (v1 structs, _v2 entry
// points where those are the current exported symbols). Only what we use.

#include <cstdint>

namespace caliper_host {
namespace cudadrv {

using CUresult   = int;            // CUDA_SUCCESS == 0
using CUdevice   = int;
using CUdeviceptr = unsigned long long;
using CUcontext  = void*;
using CUstream   = void*;          // NULL = the legacy default stream
using CUexternalMemory = void*;
using CUexternalSemaphore = void*;

constexpr CUresult CUDA_SUCCESS = 0;

struct CUuuid {
    char bytes[16];
};

// CUexternalMemoryHandleType (cuda.h): opaque Win32 NT handle == 2.
constexpr unsigned kExtMemHandleTypeOpaqueWin32 = 2;
// CUexternalSemaphoreHandleType (cuda.h): a Vulkan timeline semaphore exported
// as an opaque Win32 NT handle == 10 (V4 semaphore pipelining).
constexpr unsigned kExtSemHandleTypeTimelineOpaqueWin32 = 10;

// CUDA_EXTERNAL_MEMORY_HANDLE_DESC (v1 layout, cuda.h).
struct ExternalMemoryHandleDesc {
    unsigned int type;                 // CUexternalMemoryHandleType
    union {
        int fd;
        struct {
            void*       handle;        // NT HANDLE from vkGetMemoryWin32HandleKHR
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;           // byte size of the allocation
    unsigned int flags;
    unsigned int reserved[16];
};

// CUDA_EXTERNAL_MEMORY_BUFFER_DESC (v1 layout, cuda.h).
struct ExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

// CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC (v1 layout, cuda.h).
struct ExternalSemaphoreHandleDesc {
    unsigned int type;                 // CUexternalSemaphoreHandleType
    union {
        int fd;
        struct {
            void*       handle;        // NT HANDLE from vkGetSemaphoreWin32HandleKHR
            const void* name;
        } win32;
        const void* nvSciSyncObj;
    } handle;
    unsigned int flags;
    unsigned int reserved[16];
};

// CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS (v1 layout, cuda.h). For a timeline
// semaphore, params.fence.value is the value to signal.
struct ExternalSemaphoreSignalParams {
    struct {
        struct { unsigned long long value; } fence;
        union { void* fence; unsigned long long reserved; } nvSciSync;
        struct { unsigned long long key; } keyedMutex;
        unsigned int reserved[12];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

// Loaded function table. All pointers null until load() succeeds.
struct Api {
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuDeviceGetCount)(int* count);
    CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
    CUresult (*cuDeviceGetUuid)(CUuuid* uuid, CUdevice dev);
    CUresult (*cuDeviceTotalMem)(size_t* bytes, CUdevice dev);        // _v2
    CUresult (*cuMemGetInfo)(size_t* free, size_t* total);            // _v2
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext* ctx, CUdevice dev);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice dev);              // _v2
    CUresult (*cuCtxSetCurrent)(CUcontext ctx);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuMemcpyDtoD)(CUdeviceptr dst, CUdeviceptr src, size_t bytes); // _v2
    // Owning allocation's [base, base+size) for a device pointer — lets the
    // renderer bound a tensor's byte extent against real memory before the
    // copy (finding-#1 parity with Metal's src.length check).
    CUresult (*cuMemGetAddressRange)(CUdeviceptr* base, size_t* size,
                                     CUdeviceptr dptr);                       // _v2
    // Plain device alloc / host->device copy / free. Used only by the renderer's
    // hardware self-test (CALIPER_VULKAN_SELFTEST) to stage a known source
    // tensor; the production interop path never allocates device memory itself.
    CUresult (*cuMemAlloc)(CUdeviceptr* dptr, size_t bytes);                  // _v2
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dst, const void* src, size_t bytes); // _v2
    CUresult (*cuMemFree)(CUdeviceptr dptr);                                  // _v2
    CUresult (*cuImportExternalMemory)(CUexternalMemory* out,
                                       const ExternalMemoryHandleDesc* desc);
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr* out,
                                                CUexternalMemory mem,
                                                const ExternalMemoryBufferDesc* desc);
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory mem);
    // V4 semaphore pipelining: stream-ordered copy + GPU-side signal so the
    // handoff needs no CPU synchronize (vulkan_renderer.cpp).
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr dst, CUdeviceptr src,
                                  size_t bytes, CUstream stream);             // _v2
    CUresult (*cuImportExternalSemaphore)(CUexternalSemaphore* out,
                                          const ExternalSemaphoreHandleDesc* desc);
    CUresult (*cuSignalExternalSemaphoresAsync)(
        const CUexternalSemaphore* sems, const ExternalSemaphoreSignalParams* params,
        unsigned int count, CUstream stream);
    CUresult (*cuDestroyExternalSemaphore)(CUexternalSemaphore sem);
    CUresult (*cuGetErrorName)(CUresult err, const char** str);
};

// Loads the driver library and resolves the table on first call; cached for
// the process lifetime. Returns nullptr when no NVIDIA driver is installed —
// callers treat that as "no CUDA device", never an error.
const Api* api();

// ---------------------------------------------------------------------------
// OPTIONAL VMM surface (bridge v1.2 tests): the cuMem* virtual-memory APIs
// (CUDA 10.2+) used to build applet-shaped exportable allocations. A SEPARATE
// table so the core Api above keeps its all-or-nothing rule on older drivers:
// vmm_api() returns nullptr when the driver lacks ANY of these nine symbols
// (or api() itself failed), and callers skip — never an error.
// Struct layouts mirror cuda.h's stable v1 ABI, field-for-field identical to
// the applet-side transcriptions in sdk/.../adapters/exportable_pool.hpp
// (which the host must not include).
// ---------------------------------------------------------------------------

using CUmemGenericAllocationHandle = unsigned long long;

// CUmemAllocationType / CUmemAllocationHandleType / CUmemLocationType /
// CUmemAllocationGranularity_flags / CUmemAccess_flags values (cuda.h).
constexpr unsigned kMemAllocationTypePinned      = 1;  // CU_MEM_ALLOCATION_TYPE_PINNED
constexpr unsigned kMemHandleTypePosixFd         = 1;  // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
constexpr unsigned kMemHandleTypeWin32           = 2;  // CU_MEM_HANDLE_TYPE_WIN32
constexpr unsigned kMemLocationTypeDevice        = 1;  // CU_MEM_LOCATION_TYPE_DEVICE
constexpr unsigned kMemAllocGranularityMinimum   = 0;  // CU_MEM_ALLOC_GRANULARITY_MINIMUM
constexpr unsigned kMemAccessFlagsProtReadWrite  = 3;  // CU_MEM_ACCESS_FLAGS_PROT_READWRITE

// CUmemLocation (cuda.h).
struct MemLocation {
    unsigned int type;                 // CUmemLocationType
    int          id;
};
// CUmemAllocationProp (cuda.h, v1 layout). win32HandleMetaData must be a valid
// LPSECURITYATTRIBUTES when requestedHandleTypes is WIN32 (hardware-verified:
// null is CUDA_ERROR_INVALID_VALUE on 596.47).
struct MemAllocationProp {
    unsigned int type;                 // CUmemAllocationType
    unsigned int requestedHandleTypes; // CUmemAllocationHandleType
    MemLocation  location;
    void*        win32HandleMetaData;
    struct {
        unsigned char  compressionType;
        unsigned char  gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char  reserved[4];
    } allocFlags;
};
// CUmemAccessDesc (cuda.h).
struct MemAccessDesc {
    MemLocation  location;
    unsigned int flags;                // CUmemAccess_flags
};

// The nine VMM entry points (none have _v2 exports). All null until load.
struct VmmApi {
    CUresult (*cuMemGetAllocationGranularity)(size_t* granularity,
                                              const MemAllocationProp* prop,
                                              unsigned int option);
    CUresult (*cuMemCreate)(CUmemGenericAllocationHandle* handle, size_t size,
                            const MemAllocationProp* prop, unsigned long long flags);
    CUresult (*cuMemAddressReserve)(CUdeviceptr* ptr, size_t size, size_t alignment,
                                    CUdeviceptr addr, unsigned long long flags);
    CUresult (*cuMemMap)(CUdeviceptr ptr, size_t size, size_t offset,
                         CUmemGenericAllocationHandle handle, unsigned long long flags);
    CUresult (*cuMemSetAccess)(CUdeviceptr ptr, size_t size,
                               const MemAccessDesc* desc, size_t count);
    CUresult (*cuMemExportToShareableHandle)(void* shareableHandle,
                                             CUmemGenericAllocationHandle handle,
                                             unsigned int handleType,
                                             unsigned long long flags);
    CUresult (*cuMemUnmap)(CUdeviceptr ptr, size_t size);
    CUresult (*cuMemRelease)(CUmemGenericAllocationHandle handle);
    CUresult (*cuMemAddressFree)(CUdeviceptr ptr, size_t size);
};

// Resolves the optional table on first call; cached. nullptr when the driver
// is absent, the core api() load failed, or ANY symbol is missing.
const VmmApi* vmm_api();

}  // namespace cudadrv
}  // namespace caliper_host
