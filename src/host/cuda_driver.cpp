// Runtime loader for the CUDA driver API (see cuda_driver.h for why the host
// loads nvcuda.dll instead of linking the CUDA toolkit).
#include "cuda_driver.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace caliper_host {
namespace cudadrv {
namespace {

void* lib_open() {
#ifdef _WIN32
    return (void*)LoadLibraryA("nvcuda.dll");
#else
    return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* lib_sym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// Resolves the table once. Any missing symbol fails the whole load — a table
// with holes would turn into a null call somewhere much harder to diagnose.
const Api* load() {
    void* lib = lib_open();
    if (!lib) return nullptr;

    static Api a{};
    struct Entry { void** slot; const char* name; };
    const Entry entries[] = {
        {(void**)&a.cuInit,                         "cuInit"},
        {(void**)&a.cuDeviceGetCount,               "cuDeviceGetCount"},
        {(void**)&a.cuDeviceGetName,                "cuDeviceGetName"},
        {(void**)&a.cuDeviceGetUuid,                "cuDeviceGetUuid"},
        {(void**)&a.cuDeviceTotalMem,               "cuDeviceTotalMem_v2"},
        {(void**)&a.cuMemGetInfo,                   "cuMemGetInfo_v2"},
        {(void**)&a.cuDevicePrimaryCtxRetain,       "cuDevicePrimaryCtxRetain"},
        {(void**)&a.cuDevicePrimaryCtxRelease,      "cuDevicePrimaryCtxRelease_v2"},
        {(void**)&a.cuCtxSetCurrent,                "cuCtxSetCurrent"},
        {(void**)&a.cuCtxSynchronize,               "cuCtxSynchronize"},
        {(void**)&a.cuMemcpyDtoD,                   "cuMemcpyDtoD_v2"},
        {(void**)&a.cuMemGetAddressRange,           "cuMemGetAddressRange_v2"},
        {(void**)&a.cuMemAlloc,                     "cuMemAlloc_v2"},
        {(void**)&a.cuMemcpyHtoD,                   "cuMemcpyHtoD_v2"},
        {(void**)&a.cuMemFree,                      "cuMemFree_v2"},
        {(void**)&a.cuImportExternalMemory,         "cuImportExternalMemory"},
        {(void**)&a.cuExternalMemoryGetMappedBuffer,"cuExternalMemoryGetMappedBuffer"},
        {(void**)&a.cuDestroyExternalMemory,        "cuDestroyExternalMemory"},
        {(void**)&a.cuMemcpyDtoDAsync,              "cuMemcpyDtoDAsync_v2"},
        {(void**)&a.cuImportExternalSemaphore,      "cuImportExternalSemaphore"},
        {(void**)&a.cuSignalExternalSemaphoresAsync,"cuSignalExternalSemaphoresAsync"},
        {(void**)&a.cuDestroyExternalSemaphore,     "cuDestroyExternalSemaphore"},
        {(void**)&a.cuGetErrorName,                 "cuGetErrorName"},
    };
    for (const Entry& e : entries) {
        *e.slot = lib_sym(lib, e.name);
        if (*e.slot == nullptr) return nullptr;
    }
    if (a.cuInit(0) != CUDA_SUCCESS) return nullptr;
    return &a;
}

// Optional VMM table (bridge v1.2). Separate from load() so a driver without
// these nine symbols still yields a full core Api; a hole here nulls ONLY the
// VMM surface. Requires api() to have succeeded (cuInit done) so callers can
// use the table immediately.
const VmmApi* load_vmm() {
    if (api() == nullptr) return nullptr;
    void* lib = lib_open();
    if (!lib) return nullptr;

    static VmmApi v{};
    struct Entry { void** slot; const char* name; };
    const Entry entries[] = {
        {(void**)&v.cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity"},
        {(void**)&v.cuMemCreate,                   "cuMemCreate"},
        {(void**)&v.cuMemAddressReserve,           "cuMemAddressReserve"},
        {(void**)&v.cuMemMap,                      "cuMemMap"},
        {(void**)&v.cuMemSetAccess,                "cuMemSetAccess"},
        {(void**)&v.cuMemExportToShareableHandle,  "cuMemExportToShareableHandle"},
        {(void**)&v.cuMemUnmap,                    "cuMemUnmap"},
        {(void**)&v.cuMemRelease,                  "cuMemRelease"},
        {(void**)&v.cuMemAddressFree,              "cuMemAddressFree"},
    };
    for (const Entry& e : entries) {
        *e.slot = lib_sym(lib, e.name);
        if (*e.slot == nullptr) return nullptr;
    }
    return &v;
}

}  // namespace

const Api* api() {
    static const Api* a = load();   // magic-static: thread-safe, once
    return a;
}

const VmmApi* vmm_api() {
    static const VmmApi* v = load_vmm();
    return v;
}

}  // namespace cudadrv
}  // namespace caliper_host
