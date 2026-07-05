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
        {(void**)&a.cuGetErrorName,                 "cuGetErrorName"},
    };
    for (const Entry& e : entries) {
        *e.slot = lib_sym(lib, e.name);
        if (*e.slot == nullptr) return nullptr;
    }
    if (a.cuInit(0) != CUDA_SUCCESS) return nullptr;
    return &a;
}

}  // namespace

const Api* api() {
    static const Api* a = load();   // magic-static: thread-safe, once
    return a;
}

}  // namespace cudadrv
}  // namespace caliper_host
