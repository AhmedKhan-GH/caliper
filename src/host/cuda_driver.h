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
using CUexternalMemory = void*;

constexpr CUresult CUDA_SUCCESS = 0;

struct CUuuid {
    char bytes[16];
};

// CUexternalMemoryHandleType (cuda.h): opaque Win32 NT handle == 2.
constexpr unsigned kExtMemHandleTypeOpaqueWin32 = 2;

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
    CUresult (*cuImportExternalMemory)(CUexternalMemory* out,
                                       const ExternalMemoryHandleDesc* desc);
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr* out,
                                                CUexternalMemory mem,
                                                const ExternalMemoryBufferDesc* desc);
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory mem);
    CUresult (*cuGetErrorName)(CUresult err, const char** str);
};

// Loads the driver library and resolves the table on first call; cached for
// the process lifetime. Returns nullptr when no NVIDIA driver is installed —
// callers treat that as "no CUDA device", never an error.
const Api* api();

}  // namespace cudadrv
}  // namespace caliper_host
