// Vulkan backend for the HostRenderer seam (PLATFORM.md §5.4, Phase 4).
// Sibling to metal_renderer.mm / gl_renderer.cpp; make_vulkan_renderer()
// constructs it and main.cpp selects it as the Windows default, falling back
// to the GL backend if init() fails (no Vulkan driver, remote desktop, ...).
//
// Zero-copy story (ZEROCOPY.md, Windows/NVIDIA): discrete GPUs have no
// unified memory, so "zero-copy" means the tensor bytes never leave VRAM.
// The bridge hands us a CaliperTensor whose data is a CUDA device pointer
// (torch's own allocation). We keep a per-texture VkBuffer allocated with
// VK_KHR_external_memory_win32 export, imported into CUDA once via
// cuImportExternalMemory — the SAME physical VRAM visible to both APIs. Each
// update is then:
//   1. one cuMemcpyDtoD  (torch tensor -> shared buffer, in-VRAM)
//   2. one Vulkan pass    (shared buffer -> VkImage, colormap on-GPU for f32)
// = 0 host copies + 1 in-VRAM copy, exactly the budget ZEROCOPY.md's table
// gives this path. Note the direction deviates from the doc's sketch (CUDA
// exports / Vulkan imports): torch's caching allocator does not produce
// exportable allocations, so the renderer exports and CUDA imports. Same
// residency guarantee, same crossing count.
//
// Synchronization (V4, spec D21 upgraded): the handoff is GPU-ordered via a
// per-texture SHARED TIMELINE SEMAPHORE (VK_KHR_timeline_semaphore exported
// opaque-Win32, cuImportExternalSemaphore). Per update: CUDA's stream-ordered
// copy signals base+1, the Vulkan pass GPU-waits base+1 and signals base+2,
// and the frame submission GPU-waits base+2 before sampling — no
// cuCtxSynchronize, no fence wait, the CPU never blocks on this update. The
// only host wait is retire(): back-pressure if the PREVIOUS update of the same
// texture hasn't finished (re-record + buffer-overwrite safety; instant in
// steady state). Where timeline semaphores are unavailable, the v1 synchronous
// model (copy + cuCtxSynchronize + fenced submit, Metal's waitUntilCompleted
// shape) remains as the per-texture fallback. The adapter-side
// torch::cuda::synchronize() at the handoff is the v1 ABI contract and stays;
// elided when the adapter populates CaliperTensor.stream under bridge-v1.1
// caps (M2a, D24) — the copy+signal then ride the producer's stream.
//
// The frame loop follows imgui's example_glfw_vulkan (the ImGui_ImplVulkanH_
// helpers); the CLEAR lives in the render pass load-op with the same color as
// the GL backend's top-of-frame clear (C1 contract).

#include <volk.h>                 // also provides <vulkan/vulkan.h> types
                                  // (implementation: volk::volk static lib)

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>           // sees VK_VERSION_1_0 -> vulkan support on

#include "host_renderer.h"
#include "../cuda_driver.h"
#include <caliper/services/tensor_bridge_v1_2.h>   // CALIPER_ALLOC_HANDLE_OPAQUE_WIN32

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
// volk shims the Win32 types vulkan_win32.h needs instead of pulling in
// <windows.h> (deliberate, see volk.h) — so declare the one kernel32 call we
// use the same way rather than dragging the whole header in after the shims.
extern "C" __declspec(dllimport) int __stdcall CloseHandle(void* hObject);
// import_external_allocation dups the applet's shareable handle so the host
// owns a lifetime-independent copy (closed at release). Same declare-not-
// include discipline as CloseHandle above.
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall DuplicateHandle(
    void* hSourceProcessHandle, void* hSourceHandle,
    void* hTargetProcessHandle, void** lpTargetHandle,
    unsigned long dwDesiredAccess, int bInheritHandle, unsigned long dwOptions);
#endif

// Build-time SPIR-V of shaders/colormap.comp (glslang -V --vn kColormapSpv),
// byte-identical index math to the CPU reference (§16).
#include <colormap_spv.h>

namespace caliper_host {

// Resolve a colormap id to its 256-entry RGBA8 LUT (tensor_bridge.cpp, present
// in every exe/test link scope that pulls this backend). Unlike
// tex_update_from_device — where the bridge hands the renderer a resolved
// lut256 — the v1.2 imported path receives the raw colormap id (the texture's
// pinned mapping), so the renderer resolves it here.
const uint32_t* colormap_lut(int32_t colormap);

namespace {

// Byte extent a tensor addresses (same derivation as metal_renderer.mm).
// Unlike Metal — where the incoming MTLBuffer knows its own length — a CUDA
// device pointer carries no size, so this extent (already sanity-bounded by
// the bridge) is also what we size the shared buffer with.
uint64_t tensor_extent_bytes(const CaliperTensor& t, uint64_t elem_size) {
    uint64_t maxidx = 0;
    for (int i = 0; i < t.ndim; ++i)
        maxidx += (uint64_t)(t.shape[i] - 1) * (uint64_t)t.strides[i];
    return (maxidx + 1) * elem_size;
}

struct CmapPush {
    uint32_t w, h, sx, sy;
    float    vmin, vmax;
};

class VulkanRenderer final : public HostRenderer {
public:
    const char* name() const override { return "vulkan"; }
    const char* last_device_path() const override { return last_device_path_; }

    void window_hints() override {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // no GL context
    }

    bool init(GLFWwindow* window) override {
        window_ = window;
        if (volkInitialize() != VK_SUCCESS) return false;
        if (!glfwVulkanSupported()) return false;
        if (!create_instance()) return false;
        if (!pick_device_and_queue()) return false;
        if (!create_descriptor_pool()) return false;

        if (glfwCreateWindowSurface(instance_, window, nullptr, &surface_) != VK_SUCCESS)
            return false;
        VkBool32 wsi = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_, queue_family_, surface_, &wsi);
        if (wsi != VK_TRUE) return false;

        // Swapchain / render pass / framebuffers via the imgui helpers.
        wd_.Surface = surface_;
        const VkFormat fmts[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_FORMAT_B8G8R8_UNORM,   VK_FORMAT_R8G8B8_UNORM};
        wd_.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
            physical_, surface_, fmts, 4, VK_COLORSPACE_SRGB_NONLINEAR_KHR);
        const VkPresentModeKHR modes[] = {VK_PRESENT_MODE_FIFO_KHR};
        wd_.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(physical_, surface_, modes, 1);
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        ImGui_ImplVulkanH_CreateOrResizeWindow(instance_, physical_, device_, &wd_,
                                               queue_family_, nullptr, w, h,
                                               kMinImageCount, 0);
        // C1 contract: same background as the GL backend's clear.
        wd_.ClearValue.color.float32[0] = 0.05f;
        wd_.ClearValue.color.float32[1] = 0.05f;
        wd_.ClearValue.color.float32[2] = 0.08f;
        wd_.ClearValue.color.float32[3] = 1.0f;

        if (!ImGui_ImplGlfw_InitForVulkan(window, true)) return false;
        ImGui_ImplVulkan_InitInfo ii = {};
        ii.ApiVersion = VK_API_VERSION_1_1;
        ii.Instance = instance_;
        ii.PhysicalDevice = physical_;
        ii.Device = device_;
        ii.QueueFamily = queue_family_;
        ii.Queue = queue_;
        ii.DescriptorPool = imgui_pool_;
        ii.MinImageCount = kMinImageCount;
        ii.ImageCount = wd_.ImageCount;
        ii.PipelineInfoMain.RenderPass = wd_.RenderPass;
        ii.PipelineInfoMain.Subpass = 0;
        ii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        if (!ImGui_ImplVulkan_Init(&ii)) {
            ImGui_ImplGlfw_Shutdown();
            return false;
        }

        if (!create_oneshot_pool()) return false;

        // Resolve the CUDA interop pairing NOW, at session start (spec §3.1),
        // not lazily: the bridge reads interop_device() when it is constructed
        // (before the first device upload), so the pairing must be settled by
        // the time init() returns. Requires both the Vulkan external-memory
        // extensions AND a CUDA device whose UUID matches this one.
        interop_ok_ = external_memory_ok_ && ensure_cuda();

        // V4 semaphore pipelining: available when the device also exports
        // timeline semaphores CUDA can import. Per-texture creation can still
        // fall back to the synchronous path if a runtime step fails.
        const cudadrv::Api* cu = cudadrv::api();
        pipelined_ok_ = interop_ok_ && timeline_ok_ &&
                        cu && cu->cuImportExternalSemaphore != nullptr;
        if (interop_ok_)
            dev_note(pipelined_ok_
                         ? "sync mode: pipelined (shared timeline semaphores)"
                         : "sync mode: synchronous (timeline semaphores unavailable)");
        return true;
    }

    // Vulkan imports CUDA VRAM only when a UUID-matched CUDA device is paired
    // (spec §3.4/D20). Otherwise the backend still renders, but via CPU staging,
    // so it advertises CPU and the bridge won't accept CUDA tensors.
    CaliperDeviceKind interop_device() const override {
        return interop_ok_ ? CALIPER_DEV_CUDA : CALIPER_DEV_CPU;
    }

    // M2a (D24): only the pipelined path GPU-orders after t.stream; the
    // synchronous fallback must keep the v1 drained contract, so advertise
    // stream handoff only when pipelining is actually live.
    bool honors_stream_ordered_handoff() const override { return pipelined_ok_; }

    void new_frame() override {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        if (w > 0 && h > 0 &&
            (rebuild_swapchain_ || wd_.Width != w || wd_.Height != h)) {
            ImGui_ImplVulkan_SetMinImageCount(kMinImageCount);
            ImGui_ImplVulkanH_CreateOrResizeWindow(instance_, physical_, device_, &wd_,
                                                   queue_family_, nullptr, w, h,
                                                   kMinImageCount, 0);
            wd_.FrameIndex = 0;
            rebuild_swapchain_ = false;
        }
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void render(int /*fb_w*/, int /*fb_h*/) override {
        ImGui::Render();
        ImDrawData* dd = ImGui::GetDrawData();
        const bool minimized = (dd->DisplaySize.x <= 0.0f || dd->DisplaySize.y <= 0.0f);
        if (minimized || rebuild_swapchain_) return;
        frame_render(dd);
        frame_present();
    }

    void shutdown() override {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);
        for (auto& kv : textures_) destroy_tex(kv.second);
        textures_.clear();
        // Any imported allocations the bridge did not release (device is idle).
        for (auto& kv : imported_) {
            ImportedAlloc& a = kv.second;
            if (a.buf) vkDestroyBuffer(device_, a.buf, nullptr);
            if (a.memory) vkFreeMemory(device_, a.memory, nullptr);
#ifdef _WIN32
            if (a.handle_dup) CloseHandle((HANDLE)a.handle_dup);
#endif
        }
        imported_.clear();
        destroy_compute();
        destroy_buffer(staging_);
        destroy_buffer(lut_buf_);
        if (oneshot_pool_) { vkDestroyCommandPool(device_, oneshot_pool_, nullptr); oneshot_pool_ = VK_NULL_HANDLE; }
        if (oneshot_fence_) { vkDestroyFence(device_, oneshot_fence_, nullptr); oneshot_fence_ = VK_NULL_HANDLE; }
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &wd_, nullptr);
        surface_ = VK_NULL_HANDLE;   // destroyed by DestroyWindow
        if (imgui_pool_) { vkDestroyDescriptorPool(device_, imgui_pool_, nullptr); imgui_pool_ = VK_NULL_HANDLE; }
        release_cuda();
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    // ---- Texture ops ----
    uint64_t tex_create_rgba8(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == VK_NULL_HANDLE) return 0;
        Tex t{};
        t.w = w; t.h = h;

        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {(uint32_t)w, (uint32_t)h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device_, &ici, nullptr, &t.image) != VK_SUCCESS) return 0;

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, t.image, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        if (!find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &mai.memoryTypeIndex) ||
            vkAllocateMemory(device_, &mai, nullptr, &t.memory) != VK_SUCCESS) {
            destroy_tex(t); return 0;
        }
        vkBindImageMemory(device_, t.image, t.memory, 0);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = t.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_, &vci, nullptr, &t.view) != VK_SUCCESS) {
            destroy_tex(t); return 0;
        }

        // The ImGui-facing handle: a descriptor set in SHADER_READ_ONLY layout.
        t.descset = ImGui_ImplVulkan_AddTexture(t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (t.descset == VK_NULL_HANDLE) { destroy_tex(t); return 0; }

        // Images start UNDEFINED; move to SHADER_READ_ONLY so drawing an
        // un-uploaded texture is defined (black), matching GL/Metal behavior.
        submit_once([&](VkCommandBuffer cb) {
            barrier(cb, t, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        uint64_t id = next_id_++;
        textures_[id] = t;
        return id;
    }

    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override {
        auto it = textures_.find(tex);
        if (it == textures_.end() || data == nullptr || w <= 0 || h <= 0) return false;
        Tex& t = it->second;
        if (!retire(t)) return false;   // a pipelined update may still be writing
        const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;
        if (!ensure_buffer(staging_, bytes,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           /*external=*/false))
            return false;
        std::memcpy(staging_.mapped, data, (size_t)bytes);

        submit_once([&](VkCommandBuffer cb) {
            barrier(cb, t, t.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            vkCmdCopyBufferToImage(cb, staging_.buf, t.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            barrier(cb, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return true;
    }

    void tex_release(uint64_t tex) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return;
        vkQueueWaitIdle(queue_);   // the texture may be in the in-flight frame
        destroy_tex(it->second);
        textures_.erase(it);
    }

    uint64_t tex_imtexture_id(uint64_t tex) override {
        auto it = textures_.find(tex);
        return it == textures_.end() ? 0 : (uint64_t)it->second.descset;
    }

    // Device-resident update: t.data is a CUDA device pointer (torch CUDA
    // tensor). One in-VRAM DtoD into the exported shared buffer, then one
    // Vulkan pass into the texture. Returns false -> caller CPU-stages.
    bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                const uint32_t* lut256,
                                float vmin, float vmax) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return dev_bail("no such texture");
        Tex& dst = it->second;
        if (t.device != CALIPER_DEV_CUDA || t.data == nullptr)
            return dev_bail("tensor not CUDA-resident");
        if (!interop_ok_) return dev_bail("external-memory interop unavailable");

        const uint64_t elem = (t.dtype == CALIPER_DT_F32) ? 4 : 1;
        if (t.dtype != CALIPER_DT_F32 && t.dtype != CALIPER_DT_U8)
            return dev_bail("dtype not f32/u8");
        const uint64_t bytes = tensor_extent_bytes(t, elem);

        if (!ensure_cuda()) return dev_bail("CUDA context init failed");
        const cudadrv::Api* cu = cudadrv::api();
        cu->cuCtxSetCurrent(cuda_ctx_);

        // Bounds check before any copy (spec §3.3.2 — the CUDA analog of Metal's
        // src.length check): the tensor's byte extent must fit inside the owning
        // CUDA allocation, or the DtoD below would read past it. The bridge
        // already bounded the extent in elements; this re-bounds it against the
        // real allocation. Query failure -> treat as unbounded -> reject.
        const cudadrv::CUdeviceptr src = (cudadrv::CUdeviceptr)(uintptr_t)t.data;
        cudadrv::CUdeviceptr base = 0; size_t alloc = 0;
        if (cu->cuMemGetAddressRange(&base, &alloc, src) != cudadrv::CUDA_SUCCESS)
            return dev_bail("cuMemGetAddressRange failed");
        if (src < base || (src - base) + bytes > alloc)
            return dev_bail("tensor extent exceeds CUDA allocation");

        if (!ensure_shared_buffer(dst, bytes)) return dev_bail("shared-buffer import failed");

        // When the tensor's data IS our shared buffer (the alloc_shared
        // literal-zero-copy path, spec §3.5), there is nothing to copy — the
        // applet's kernels already wrote it in place.
        const bool shared_in_place = (src == dst.interop.cuda_ptr);

        bool ok = false;
        if (pipelined_ok_ && ensure_pipeline_objects(dst)) {
            // V4: GPU-ordered handoff via the texture's timeline semaphore —
            // stream-ordered copy, GPU signal/wait, no CPU synchronize.
            ok = update_pipelined(dst, t, src, bytes, shared_in_place,
                                  lut256, vmin, vmax);
        } else {
            // Synchronous fallback (the v1 model, Metal's waitUntilCompleted
            // shape): in-VRAM copy, drain, fence-waited Vulkan pass.
            if (!shared_in_place &&
                cu->cuMemcpyDtoD(dst.interop.cuda_ptr, src, (size_t)bytes)
                    != cudadrv::CUDA_SUCCESS)
                return dev_bail("cuMemcpyDtoD failed");
            if (cu->cuCtxSynchronize() != cudadrv::CUDA_SUCCESS)
                return dev_bail("cuCtxSynchronize failed");
            if (t.dtype == CALIPER_DT_F32 && lut256 != nullptr)
                ok = colormap_compute(dst, t, lut256, vmin, vmax);
            else if (t.dtype == CALIPER_DT_U8)
                ok = blit_u8(dst, t);
        }
        // Honesty rule (spec §3.5): "zero-copy" is the alloc_shared in-place
        // path (no D2D); an arbitrary torch tensor costs one D2D copy, so it's
        // "GPU-resident", not zero-copy.
        if (ok) dev_note(shared_in_place ? "CUDA interop OK — zero-copy (alloc_shared in place)"
                                         : "CUDA interop OK — GPU-resident (no CPU staging)");
        return ok;
    }

    // Literal zero-copy alloc_shared (spec §3.5): create/size texture `tex`'s
    // interop buffer and hand back a CUDA device pointer into it. The applet
    // wraps this with torch::from_blob and writes it; update_texture then runs
    // the buffer->image pass with no copy (shared_in_place above).
    // Test-only readback: the CaliperTextureId is the descriptor set, so find
    // the Tex whose descset matches and copy its image to host bytes.
    std::vector<uint8_t> debug_readback_rgba8(uint64_t id, int, int) override {
        for (auto& kv : textures_)
            if ((uint64_t)kv.second.descset == id) return readback_rgba8(kv.second);
        return {};
    }

    bool alloc_device_shared(uint64_t tex, uint64_t bytes, void** out) override {
        if (!interop_ok_ || out == nullptr) return false;
        auto it = textures_.find(tex);
        if (it == textures_.end()) return false;
        if (!ensure_cuda()) return false;
        cudadrv::api()->cuCtxSetCurrent(cuda_ctx_);
        if (!ensure_shared_buffer(it->second, bytes)) return false;
        *out = (void*)(uintptr_t)it->second.interop.cuda_ptr;
        return true;
    }

    // ---- Bridge v1.2: import applet-exported allocations (zero data copies) --
    // supports_external_import gates the bridge's CALIPER_BRIDGE_CAP_IMPORT_ALLOC:
    // available only when the Vulkan side exports external memory AND a
    // UUID-matched CUDA device is paired (interop live).
    bool supports_external_import() const override {
#ifdef _WIN32
        return interop_ok_ && external_memory_ok_;
#else
        return false;
#endif
    }

    // Import an applet-exported CUDA allocation as one VkBuffer (its import
    // twin of ensure_buffer's export). Returns a renderer-internal id (0 on
    // failure -> bridge falls back). Any failure frees everything it created.
    uint64_t import_external_allocation(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type) override {
#ifdef _WIN32
        if (!interop_ok_ || !external_memory_ok_) return 0;
        if (handle_type != CALIPER_ALLOC_HANDLE_OPAQUE_WIN32) return 0;
        if (os_handle == nullptr || size_bytes == 0) return 0;

        // Own our own copy of the shareable handle (DUPLICATE_SAME_ACCESS);
        // the applet/adapter keeps and closes theirs independently.
        HANDLE dup = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), (HANDLE)os_handle,
                             GetCurrentProcess(), &dup, 0, /*bInheritHandle=*/0,
                             /*DUPLICATE_SAME_ACCESS=*/0x00000002u) ||
            dup == nullptr)
            return 0;

        ImportedAlloc a{};
        a.handle_dup = dup;
        a.size = (VkDeviceSize)size_bytes;

        VkExternalMemoryBufferCreateInfo emci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        emci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.pNext = &emci;
        ci.size = size_bytes;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &ci, nullptr, &a.buf) != VK_SUCCESS) {
            CloseHandle(dup);
            return 0;
        }

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, a.buf, &mr);
        VkImportMemoryWin32HandleInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        imp.handle = dup;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &imp;
        mai.allocationSize = size_bytes;   // the applet's padded allocation size
        if (!find_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &mai.memoryTypeIndex) ||
            vkAllocateMemory(device_, &mai, nullptr, &a.memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, a.buf, nullptr);
            CloseHandle(dup);
            return 0;
        }
        if (vkBindBufferMemory(device_, a.buf, a.memory, 0) != VK_SUCCESS) {
            vkFreeMemory(device_, a.memory, nullptr);
            vkDestroyBuffer(device_, a.buf, nullptr);
            CloseHandle(dup);
            return 0;
        }

        const uint64_t id = next_import_id_++;   // 0 invalid; never reused
        imported_[id] = a;
        return id;
#else
        (void)os_handle; (void)size_bytes; (void)handle_type;
        return 0;
#endif
    }

    // Synchronous release (the tex_release precedent): drain the queue, then
    // destroy buffer -> free memory -> close the dup handle -> erase. A texture
    // still referencing a released alloc simply fails its next update -> the
    // bridge CPU-stages.
    void release_external_allocation(uint64_t id) override {
        auto it = imported_.find(id);
        if (it == imported_.end()) return;   // invalid id / double release: no-op
        vkQueueWaitIdle(queue_);
        ImportedAlloc& a = it->second;
        if (a.buf) vkDestroyBuffer(device_, a.buf, nullptr);
        if (a.memory) vkFreeMemory(device_, a.memory, nullptr);
#ifdef _WIN32
        if (a.handle_dup) CloseHandle((HANDLE)a.handle_dup);
#endif
        imported_.erase(it);
    }

    // Colormap/blit a texture FROM an imported allocation at a byte offset, with
    // NO data copy (the applet's kernels already wrote the bytes). Guards mirror
    // tex_update_from_device; the byte offset rides the descriptor (f32) or the
    // BufferImageCopy (u8). Returns false -> bridge CPU-stages.
    bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t offset_bytes,
                                  const CaliperTensor& desc, int32_t colormap,
                                  float vmin, float vmax) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return dev_bail("import-update: no such texture");
        Tex& dst = it->second;
        auto ia = imported_.find(alloc);
        if (ia == imported_.end()) return dev_bail("import-update: no such alloc");
        ImportedAlloc& src = ia->second;
        if (!interop_ok_) return dev_bail("import-update: interop unavailable");

        if (desc.dtype != CALIPER_DT_F32 && desc.dtype != CALIPER_DT_U8)
            return dev_bail("import-update: dtype not f32/u8");
        const uint64_t elem = (desc.dtype == CALIPER_DT_F32) ? 4 : 1;
        const uint64_t bytes = tensor_extent_bytes(desc, elem);
        // offset + bytes <= size, phrased to never overflow.
        if (bytes > (uint64_t)src.size || offset_bytes > (uint64_t)src.size - bytes)
            return dev_bail("import-update: extent exceeds imported allocation");

        const bool is_cmap = (desc.dtype == CALIPER_DT_F32);
        const uint32_t* lut256 = is_cmap ? colormap_lut(colormap) : nullptr;
        if (is_cmap && lut256 == nullptr)
            return dev_bail("import-update: bad colormap");

        // f32: VkDescriptorBufferInfo.offset must meet minStorageBufferOffsetAlignment.
        if (is_cmap && (offset_bytes % (uint64_t)storage_buffer_alignment_) != 0)
            return dev_bail("import-update: f32 offset misaligned");
        // u8: VkBufferImageCopy.bufferOffset must be a multiple of 4 (non-depth
        // format rule; the RGBA8 texel size 4 also divides it).
        if (!is_cmap && (offset_bytes % 4u) != 0)
            return dev_bail("import-update: u8 offset not 4-aligned");

        if (!ensure_cuda()) return dev_bail("import-update: CUDA context init failed");
        cudadrv::api()->cuCtxSetCurrent(cuda_ctx_);

        // stream != NULL => GPU-order after the producer via the texture's
        // timeline semaphore (no copy). stream == NULL => adapter drained (v1
        // rung contract) => a plain fenced submit is correct and simpler.
        bool ok;
        if (desc.stream != nullptr && pipelined_ok_ && ensure_pipeline_objects(dst))
            ok = update_imported_pipelined(dst, src, offset_bytes, bytes, is_cmap,
                                           lut256, desc, vmin, vmax);
        else
            ok = update_imported_sync(dst, src, offset_bytes, bytes, is_cmap,
                                      lut256, desc, vmin, vmax);
        if (ok) dev_note("import path OK — zero-copy (imported allocation in place)");
        return ok;
    }

    // One-time-per-message stderr breadcrumbs so a run makes it obvious whether
    // the zero-copy interop path fired or why it fell back to CPU staging. A
    // bail here is not an error (the bridge CPU-stages) — just diagnostic.
    void dev_note(const char* msg) {
        if (dev_seen_.insert(msg).second)
            std::fprintf(stderr, "[vulkan] device path: %s\n", msg);
    }
    bool dev_bail(const char* why) { dev_note(why); return false; }

private:
    static constexpr uint32_t kMinImageCount = 2;

    struct Buffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void* mapped = nullptr;   // host-visible buffers only
    };
    struct Interop {
        Buffer shared;                       // exported device-local buffer
        void* win32_handle = nullptr;        // NT handle from Vulkan
        cudadrv::CUexternalMemory ext = nullptr;
        cudadrv::CUdeviceptr cuda_ptr = 0;

        // V4 pipelining (per texture): a shared timeline semaphore orders
        // CUDA-copy -> Vulkan-pass -> frame-draw entirely on the GPU. The
        // monotonic timeline_value is the last value the chain will signal;
        // a host wait on it retires the whole prior chain.
        VkSemaphore timeline = VK_NULL_HANDLE;
        void* sem_handle = nullptr;          // NT handle exported to CUDA
        cudadrv::CUexternalSemaphore cuda_sem = nullptr;
        uint64_t timeline_value = 0;
        VkCommandBuffer cb = VK_NULL_HANDLE; // re-recorded per update (retired first)
        VkDescriptorSet set = VK_NULL_HANDLE;// per-texture cmap descriptor set
        Buffer lut;                          // per-texture LUT (256 RGBA8)
        bool pipelined = false;              // sync objects created OK
    };
    struct Tex {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet descset = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        int w = 0, h = 0;
        Interop interop;
    };

    // Bridge v1.2 imported external allocation: one VkBuffer per applet-exported
    // CUDA allocation, imported (not exported) via VkImportMemoryWin32HandleInfoKHR.
    // Textures reference it at descriptor/copy byte offsets — no per-texture
    // buffer, no data copy. handle_dup is our DuplicateHandle copy (CloseHandle
    // at release; the applet/adapter keeps and closes its own).
    struct ImportedAlloc {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void* handle_dup = nullptr;
    };

    // ---- init helpers ----
    bool create_instance() {
        uint32_t n = 0;
        const char** glfw_ext = glfwGetRequiredInstanceExtensions(&n);
        if (!glfw_ext) return false;
        std::vector<const char*> exts(glfw_ext, glfw_ext + n);

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "caliper";
        app.apiVersion = VK_API_VERSION_1_1;   // properties2 + external-mem core
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();
        if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) return false;
        volkLoadInstance(instance_);
        return true;
    }

    static bool has_ext(const std::vector<VkExtensionProperties>& v, const char* name) {
        for (const auto& e : v)
            if (std::strcmp(e.extensionName, name) == 0) return true;
        return false;
    }

    bool pick_device_and_queue() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (n == 0) return false;
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance_, &n, devs.data());

        int best_score = -1;
        for (VkPhysicalDevice d : devs) {
            uint32_t en = 0;
            vkEnumerateDeviceExtensionProperties(d, nullptr, &en, nullptr);
            std::vector<VkExtensionProperties> eprops(en);
            vkEnumerateDeviceExtensionProperties(d, nullptr, &en, eprops.data());
            if (!has_ext(eprops, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;

            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            p2.pNext = &idp;
            vkGetPhysicalDeviceProperties2(d, &p2);

            const bool ext_mem =
                has_ext(eprops, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) &&
                has_ext(eprops, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            // V4 pipelining wants timeline semaphores exportable to CUDA.
            const bool ext_sem =
                has_ext(eprops, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) &&
                has_ext(eprops, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
            int score = 0;
            if (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 4;
            if (ext_mem) score += 2;
            if (score > best_score) {
                best_score = score;
                physical_ = d;
                external_memory_ok_ = ext_mem;
                timeline_ok_ = ext_sem;
                std::memcpy(device_uuid_, idp.deviceUUID, 16);
            }
        }
        if (physical_ == VK_NULL_HANDLE) return false;

        // The extension alone isn't enough — the timelineSemaphore feature must
        // be supported and explicitly enabled at device creation.
        if (timeline_ok_) {
            VkPhysicalDeviceTimelineSemaphoreFeaturesKHR tlf{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &tlf;
            vkGetPhysicalDeviceFeatures2(physical_, &f2);
            timeline_ok_ = (tlf.timelineSemaphore == VK_TRUE);
        }

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qn, qf.data());
        queue_family_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; ++i) {
            const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if ((qf[i].queueFlags & need) == need) { queue_family_ = i; break; }
        }
        if (queue_family_ == UINT32_MAX) return false;

        std::vector<const char*> dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        if (external_memory_ok_) {
            dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        }
        VkPhysicalDeviceTimelineSemaphoreFeaturesKHR tl_feat{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR};
        tl_feat.timelineSemaphore = VK_TRUE;
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (timeline_ok_) {
            dev_exts.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            dev_exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
            dci.pNext = &tl_feat;
        }
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.data();
        if (vkCreateDevice(physical_, &dci, nullptr, &device_) != VK_SUCCESS) return false;
        volkLoadDevice(device_);
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem_props_);
        // The v1.2 imported f32 path binds the allocation at a byte offset as a
        // storage buffer; VkDescriptorBufferInfo.offset must be a multiple of
        // this. Cache it once (torch sub-allocations are 512-aligned, so real
        // offsets clear it; a violation falls back — see tex_update_from_imported).
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_, &props);
        storage_buffer_alignment_ = props.limits.minStorageBufferOffsetAlignment;
        return true;
    }

    bool create_descriptor_pool() {
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + 256},
            {VK_DESCRIPTOR_TYPE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE},
        };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets = 0;
        for (const auto& s : sizes) pi.maxSets += s.descriptorCount;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        return vkCreateDescriptorPool(device_, &pi, nullptr, &imgui_pool_) == VK_SUCCESS;
    }

    bool create_oneshot_pool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queue_family_;
        if (vkCreateCommandPool(device_, &ci, nullptr, &oneshot_pool_) != VK_SUCCESS)
            return false;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        return vkCreateFence(device_, &fi, nullptr, &oneshot_fence_) == VK_SUCCESS;
    }

    // ---- frame loop (structure of imgui's example_glfw_vulkan) ----
    void frame_render(ImDrawData* dd) {
        VkSemaphore img_acq = wd_.FrameSemaphores[wd_.SemaphoreIndex].ImageAcquiredSemaphore;
        VkSemaphore rend_done = wd_.FrameSemaphores[wd_.SemaphoreIndex].RenderCompleteSemaphore;
        VkResult err = vkAcquireNextImageKHR(device_, wd_.Swapchain, UINT64_MAX,
                                             img_acq, VK_NULL_HANDLE, &wd_.FrameIndex);
        if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
            rebuild_swapchain_ = true;
        if (err == VK_ERROR_OUT_OF_DATE_KHR) return;

        ImGui_ImplVulkanH_Frame* fd = &wd_.Frames[wd_.FrameIndex];
        vkWaitForFences(device_, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &fd->Fence);
        vkResetCommandPool(device_, fd->CommandPool, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fd->CommandBuffer, &bi);
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = wd_.RenderPass;
        rp.framebuffer = fd->Framebuffer;
        rp.renderArea.extent = {(uint32_t)wd_.Width, (uint32_t)wd_.Height};
        rp.clearValueCount = 1;
        rp.pClearValues = &wd_.ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(dd, fd->CommandBuffer);
        vkCmdEndRenderPass(fd->CommandBuffer);
        vkEndCommandBuffer(fd->CommandBuffer);

        // V4: pipelined device updates queued this frame must complete before
        // ImGui samples their textures — GPU-side timeline waits, not fences.
        std::vector<VkSemaphore> wait_sems{img_acq};
        std::vector<uint64_t> wait_vals{0};                 // binary: ignored
        std::vector<VkPipelineStageFlags> wait_stages{
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        for (const auto& pw : pending_frame_waits_) {
            wait_sems.push_back(pw.first);
            wait_vals.push_back(pw.second);
            wait_stages.push_back(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
        uint64_t sig_val = 0;                               // binary: ignored
        VkTimelineSemaphoreSubmitInfoKHR tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
        tsi.waitSemaphoreValueCount = (uint32_t)wait_vals.size();
        tsi.pWaitSemaphoreValues = wait_vals.data();
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &sig_val;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        if (!pending_frame_waits_.empty()) si.pNext = &tsi;
        si.waitSemaphoreCount = (uint32_t)wait_sems.size();
        si.pWaitSemaphores = wait_sems.data();
        si.pWaitDstStageMask = wait_stages.data();
        si.commandBufferCount = 1;
        si.pCommandBuffers = &fd->CommandBuffer;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &rend_done;
        if (vkQueueSubmit(queue_, 1, &si, fd->Fence) == VK_SUCCESS)
            pending_frame_waits_.clear();
    }

    void frame_present() {
        if (rebuild_swapchain_) return;
        VkSemaphore rend_done = wd_.FrameSemaphores[wd_.SemaphoreIndex].RenderCompleteSemaphore;
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &rend_done;
        pi.swapchainCount = 1;
        pi.pSwapchains = &wd_.Swapchain;
        pi.pImageIndices = &wd_.FrameIndex;
        VkResult err = vkQueuePresentKHR(queue_, &pi);
        if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
            rebuild_swapchain_ = true;
        wd_.SemaphoreIndex = (wd_.SemaphoreIndex + 1) % wd_.SemaphoreCount;
    }

    // ---- small helpers ----
    bool find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags props, uint32_t* out) {
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i)
            if ((type_bits & (1u << i)) &&
                (mem_props_.memoryTypes[i].propertyFlags & props) == props) {
                *out = i;
                return true;
            }
        return false;
    }

    template <typename F>
    bool submit_once(F&& record) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = oneshot_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        if (vkAllocateCommandBuffers(device_, &ai, &cb) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        record(cb);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        bool ok = vkQueueSubmit(queue_, 1, &si, oneshot_fence_) == VK_SUCCESS;
        if (ok) {
            vkWaitForFences(device_, 1, &oneshot_fence_, VK_TRUE, UINT64_MAX);
            vkResetFences(device_, 1, &oneshot_fence_);
        }
        vkFreeCommandBuffers(device_, oneshot_pool_, 1, &cb);
        return ok;
    }

    void barrier(VkCommandBuffer cb, Tex& t, VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = t.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &b);
    }

    bool ensure_buffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, bool external) {
        if (b.buf != VK_NULL_HANDLE && b.size >= size) return true;
        destroy_buffer(b);

        VkExternalMemoryBufferCreateInfo emci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        emci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        if (external) ci.pNext = &emci;
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &ci, nullptr, &b.buf) != VK_SUCCESS) return false;

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, b.buf, &mr);
        VkExportMemoryAllocateInfo xmai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        xmai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        if (external) mai.pNext = &xmai;
        mai.allocationSize = mr.size;
        if (!find_mem_type(mr.memoryTypeBits, props, &mai.memoryTypeIndex) ||
            vkAllocateMemory(device_, &mai, nullptr, &b.memory) != VK_SUCCESS) {
            destroy_buffer(b);
            return false;
        }
        vkBindBufferMemory(device_, b.buf, b.memory, 0);
        b.size = mr.size;
        if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            if (vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped) != VK_SUCCESS) {
                destroy_buffer(b);
                return false;
            }
        }
        return true;
    }

    void destroy_buffer(Buffer& b) {
        if (b.mapped) { vkUnmapMemory(device_, b.memory); b.mapped = nullptr; }
        if (b.buf) { vkDestroyBuffer(device_, b.buf, nullptr); b.buf = VK_NULL_HANDLE; }
        if (b.memory) { vkFreeMemory(device_, b.memory, nullptr); b.memory = VK_NULL_HANDLE; }
        b.size = 0;
    }

    // V4 sync-object teardown (subset of destroy_interop; also the cleanup for
    // a partially-constructed ensure_pipeline_objects). Caller guarantees the
    // GPU is done with them (retire/queue-idle/device-idle).
    void destroy_sync_objects(Interop& io) {
        const cudadrv::Api* cu = cudadrv::api();
        if (io.timeline != VK_NULL_HANDLE) {
            // A destroyed semaphore must never linger in the frame-wait list.
            for (size_t i = pending_frame_waits_.size(); i-- > 0;)
                if (pending_frame_waits_[i].first == io.timeline)
                    pending_frame_waits_.erase(pending_frame_waits_.begin() + (long)i);
        }
        if (io.cuda_sem && cu) {
            cu->cuCtxSetCurrent(cuda_ctx_);
            cu->cuDestroyExternalSemaphore(io.cuda_sem);
            io.cuda_sem = nullptr;
        }
        if (io.sem_handle) {
#ifdef _WIN32
            CloseHandle((HANDLE)io.sem_handle);
#endif
            io.sem_handle = nullptr;
        }
        if (io.timeline) { vkDestroySemaphore(device_, io.timeline, nullptr); io.timeline = VK_NULL_HANDLE; }
        if (io.cb) { vkFreeCommandBuffers(device_, oneshot_pool_, 1, &io.cb); io.cb = VK_NULL_HANDLE; }
        if (io.set) { vkFreeDescriptorSets(device_, cmap_pool_, 1, &io.set); io.set = VK_NULL_HANDLE; }
        destroy_buffer(io.lut);
        io.timeline_value = 0;
        io.pipelined = false;
    }

    void destroy_interop(Interop& io) {
        destroy_sync_objects(io);
        const cudadrv::Api* cu = cudadrv::api();
        if (io.ext && cu) {
            cu->cuCtxSetCurrent(cuda_ctx_);
            cu->cuDestroyExternalMemory(io.ext);
            io.ext = nullptr;
        }
        io.cuda_ptr = 0;
        if (io.win32_handle) {
#ifdef _WIN32
            CloseHandle((HANDLE)io.win32_handle);
#endif
            io.win32_handle = nullptr;
        }
        destroy_buffer(io.shared);
    }

    // ---- V4 semaphore pipelining -------------------------------------------
    // Host wait on a timeline value: retires a chain. 2 s timeout so a hung
    // GPU degrades to a staged frame instead of deadlocking the UI thread.
    bool wait_timeline(VkSemaphore sem, uint64_t value) {
        VkSemaphoreWaitInfoKHR wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &sem;
        wi.pValues = &value;
        return vkWaitSemaphoresKHR(device_, &wi, 2000000000ull) == VK_SUCCESS;
    }

    // Retire any in-flight pipelined chain on this texture. Needed before
    // re-recording its command buffer, overwriting its shared buffer, mixing
    // in a staged upload, or reading it back. Instant in steady state (the
    // chain from the previous training step finished long ago).
    bool retire(Tex& t) {
        if (!t.interop.pipelined || t.interop.timeline_value == 0) return true;
        return wait_timeline(t.interop.timeline, t.interop.timeline_value);
    }

    // Per-texture V4 sync objects, created lazily on the first pipelined
    // update: an exported timeline semaphore CUDA imports, plus this texture's
    // own command buffer, descriptor set, and LUT buffer (an in-flight update
    // of one texture must never race another's resources — the sync path's
    // shared globals are safe only because it fence-waits every dispatch).
    bool ensure_pipeline_objects(Tex& t) {
        Interop& io = t.interop;
        if (io.pipelined) return true;
        if (!pipelined_ok_ || !ensure_compute()) return false;

        VkSemaphoreTypeCreateInfoKHR sti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR};
        sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
        sti.initialValue = 0;
        VkExportSemaphoreCreateInfo esi{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        esi.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        sti.pNext = &esi;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &sti;
        if (vkCreateSemaphore(device_, &sci, nullptr, &io.timeline) != VK_SUCCESS)
            return false;

#ifdef _WIN32
        VkSemaphoreGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        gi.semaphore = io.timeline;
        gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        HANDLE h = nullptr;
        if (vkGetSemaphoreWin32HandleKHR(device_, &gi, &h) != VK_SUCCESS) {
            destroy_sync_objects(io);
            return false;
        }
        io.sem_handle = h;

        const cudadrv::Api* cu = cudadrv::api();
        cu->cuCtxSetCurrent(cuda_ctx_);
        cudadrv::ExternalSemaphoreHandleDesc sd{};
        sd.type = cudadrv::kExtSemHandleTypeTimelineOpaqueWin32;
        sd.handle.win32.handle = h;
        if (cu->cuImportExternalSemaphore(&io.cuda_sem, &sd) != cudadrv::CUDA_SUCCESS) {
            destroy_sync_objects(io);
            return false;
        }
#else
        destroy_sync_objects(io);
        return false;   // non-Win32 handle export is a Linux-port task
#endif

        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = oneshot_pool_;   // RESET flag -> Begin re-records
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &ai, &io.cb) != VK_SUCCESS) {
            destroy_sync_objects(io);
            return false;
        }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = cmap_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &cmap_layout_;
        if (vkAllocateDescriptorSets(device_, &dai, &io.set) != VK_SUCCESS) {
            destroy_sync_objects(io);
            return false;
        }

        io.pipelined = true;
        return true;
    }

    void destroy_tex(Tex& t) {
        destroy_interop(t.interop);
        if (t.descset) { ImGui_ImplVulkan_RemoveTexture(t.descset); t.descset = VK_NULL_HANDLE; }
        if (t.view) { vkDestroyImageView(device_, t.view, nullptr); t.view = VK_NULL_HANDLE; }
        if (t.image) { vkDestroyImage(device_, t.image, nullptr); t.image = VK_NULL_HANDLE; }
        if (t.memory) { vkFreeMemory(device_, t.memory, nullptr); t.memory = VK_NULL_HANDLE; }
    }

    // ---- CUDA interop ----
    // Retain the primary context of the CUDA device whose UUID matches the
    // Vulkan physical device — the same context torch's runtime uses, so
    // torch device pointers are directly usable by our DtoD copy.
    bool ensure_cuda() {
        if (cuda_ctx_) return true;
        const cudadrv::Api* cu = cudadrv::api();
        if (!cu) return false;
        int count = 0;
        if (cu->cuDeviceGetCount(&count) != cudadrv::CUDA_SUCCESS || count <= 0)
            return false;
        int matched = -1;
        for (int i = 0; i < count; ++i) {
            cudadrv::CUuuid u{};
            if (cu->cuDeviceGetUuid(&u, i) == cudadrv::CUDA_SUCCESS &&
                std::memcmp(u.bytes, device_uuid_, 16) == 0) {
                matched = i;
                break;
            }
        }
        // D20: no CUDA device matches the Vulkan device's UUID (hybrid laptop,
        // or the Vulkan device isn't the NVIDIA one) -> interop OFF. Never fall
        // back to device 0, which would pair with the wrong GPU.
        if (matched < 0) return false;
        cuda_dev_ = matched;
        return cu->cuDevicePrimaryCtxRetain(&cuda_ctx_, cuda_dev_) == cudadrv::CUDA_SUCCESS;
    }

    void release_cuda() {
        const cudadrv::Api* cu = cudadrv::api();
        if (cuda_ctx_ && cu) {
            cu->cuDevicePrimaryCtxRelease(cuda_dev_);
            cuda_ctx_ = nullptr;
        }
    }

    // Copy a texture's pixels back to host RGBA8 (requires TRANSFER_SRC usage,
    // set at create). Drives debug_readback_rgba8 for the gfx determinism suite.
    // Restores the image's prior layout.
    std::vector<uint8_t> readback_rgba8(Tex& t) {
        if (!retire(t)) return {};      // drain any in-flight pipelined chain
        const size_t n = (size_t)t.w * (size_t)t.h * 4;
        Buffer host{};
        if (!ensure_buffer(host, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, /*external=*/false))
            return {};
        const VkImageLayout prev = t.layout;
        const bool ok = submit_once([&](VkCommandBuffer cb) {
            barrier(cb, t, prev, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {(uint32_t)t.w, (uint32_t)t.h, 1};
            vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   host.buf, 1, &r);
            barrier(cb, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, prev);
        });
        std::vector<uint8_t> out;
        if (ok && host.mapped) { out.resize(n); std::memcpy(out.data(), host.mapped, n); }
        destroy_buffer(host);
        return out;
    }

    bool ensure_shared_buffer(Tex& t, uint64_t bytes) {
        Interop& io = t.interop;
        if (io.shared.buf != VK_NULL_HANDLE && io.shared.size >= bytes && io.cuda_ptr)
            return true;
        destroy_interop(io);

        if (!ensure_buffer(io.shared, bytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           /*external=*/true))
            return false;

#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        gi.memory = io.shared.memory;
        gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        HANDLE h = nullptr;
        if (vkGetMemoryWin32HandleKHR(device_, &gi, &h) != VK_SUCCESS) {
            destroy_interop(io);
            return false;
        }
        io.win32_handle = h;

        const cudadrv::Api* cu = cudadrv::api();
        cu->cuCtxSetCurrent(cuda_ctx_);
        cudadrv::ExternalMemoryHandleDesc hd{};
        hd.type = cudadrv::kExtMemHandleTypeOpaqueWin32;
        hd.handle.win32.handle = h;
        hd.size = io.shared.size;
        if (cu->cuImportExternalMemory(&io.ext, &hd) != cudadrv::CUDA_SUCCESS) {
            destroy_interop(io);
            return false;
        }
        cudadrv::ExternalMemoryBufferDesc bd{};
        bd.offset = 0;
        bd.size = io.shared.size;
        if (cu->cuExternalMemoryGetMappedBuffer(&io.cuda_ptr, io.ext, &bd) !=
            cudadrv::CUDA_SUCCESS) {
            destroy_interop(io);
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    // ---- compute colormap (f32 + LUT -> RGBA8 storage image) ----
    bool ensure_compute() {
        if (cmap_pipeline_) return true;

        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 3;
        li.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &li, nullptr, &cmap_layout_) != VK_SUCCESS)
            return false;

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CmapPush)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &cmap_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device_, &pli, nullptr, &cmap_pipe_layout_) != VK_SUCCESS)
            return false;

        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = sizeof(kColormapSpv);
        smi.pCode = kColormapSpv;
        VkShaderModule sm;
        if (vkCreateShaderModule(device_, &smi, nullptr, &sm) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        ci.layout = cmap_pipe_layout_;
        const bool ok = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci,
                                                 nullptr, &cmap_pipeline_) == VK_SUCCESS;
        vkDestroyShaderModule(device_, sm, nullptr);
        if (!ok) return false;

        // 1 global set (sync path) + up to 256 per-texture sets (pipelined
        // path — each texture needs its own set because its dispatch may still
        // be in flight when another texture's update records).
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 257},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 257},
        };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets = 257;
        pi.poolSizeCount = 2;
        pi.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(device_, &pi, nullptr, &cmap_pool_) != VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = cmap_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &cmap_layout_;
        return vkAllocateDescriptorSets(device_, &ai, &cmap_set_) == VK_SUCCESS;
    }

    void destroy_compute() {
        if (cmap_pool_) { vkDestroyDescriptorPool(device_, cmap_pool_, nullptr); cmap_pool_ = VK_NULL_HANDLE; }
        if (cmap_pipeline_) { vkDestroyPipeline(device_, cmap_pipeline_, nullptr); cmap_pipeline_ = VK_NULL_HANDLE; }
        if (cmap_pipe_layout_) { vkDestroyPipelineLayout(device_, cmap_pipe_layout_, nullptr); cmap_pipe_layout_ = VK_NULL_HANDLE; }
        if (cmap_layout_) { vkDestroyDescriptorSetLayout(device_, cmap_layout_, nullptr); cmap_layout_ = VK_NULL_HANDLE; }
    }

    // Write a cmap descriptor set from an explicit source buffer subrange
    // (binding 0) + LUT (binding 1) + storage image (binding 2). The v1.2
    // imported path binds imported.buf at a byte offset here; the byte offset
    // rides the descriptor so colormap.comp is unchanged.
    void write_cmap_set_src(VkDescriptorSet set, Tex& dst, const Buffer& lut,
                            VkBuffer src, VkDeviceSize src_off, VkDeviceSize src_range) {
        VkDescriptorBufferInfo src_info{src, src_off, src_range};
        VkDescriptorBufferInfo lut_info{lut.buf, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo img_info{VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[3]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &src_info, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lut_info, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img_info, nullptr, nullptr};
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    }

    // Write a cmap descriptor set: the per-texture shared buffer + LUT + image.
    void write_cmap_set(VkDescriptorSet set, Tex& dst, const Buffer& lut) {
        write_cmap_set_src(set, dst, lut, dst.interop.shared.buf, 0, VK_WHOLE_SIZE);
    }

    // Command-buffer bodies shared by the sync path (fence-waited, global set)
    // and the V4 pipelined path (semaphore-ordered, per-texture set).
    void record_cmap_body(VkCommandBuffer cb, Tex& dst, const CmapPush& p,
                          VkDescriptorSet set) {
        // Make the CUDA writes to the shared buffer visible to the shader.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
        barrier(cb, dst, dst.layout, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cmap_pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                cmap_pipe_layout_, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, cmap_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(p), &p);
        vkCmdDispatch(cb, (p.w + 15) / 16, (p.h + 15) / 16, 1);
        barrier(cb, dst, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // Blit body from an explicit source buffer + byte offset. The v1.2 imported
    // path copies imported.buf at offset; the sync/pipelined device paths pass
    // the per-texture shared buffer at 0.
    void record_blit_body_src(VkCommandBuffer cb, Tex& dst, uint32_t w, uint32_t h,
                              VkBuffer src, VkDeviceSize src_off) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
        barrier(cb, dst, dst.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.bufferOffset = src_off;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cb, src, dst.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    void record_blit_body(VkCommandBuffer cb, Tex& dst, uint32_t w, uint32_t h) {
        record_blit_body_src(cb, dst, w, h, dst.interop.shared.buf, 0);
    }

    CmapPush make_push(Tex& dst, const CaliperTensor& t, float vmin, float vmax) {
        CmapPush p{};
        p.w = (uint32_t)dst.w;
        p.h = (uint32_t)dst.h;
        p.sx = (t.ndim >= 1) ? (uint32_t)t.strides[t.ndim - 1] : 1u;
        p.sy = (t.ndim >= 2) ? (uint32_t)t.strides[t.ndim - 2] : p.w;
        p.vmin = vmin;
        p.vmax = vmax;
        return p;
    }

    bool colormap_compute(Tex& dst, const CaliperTensor& t,
                          const uint32_t* lut256, float vmin, float vmax) {
        if (!ensure_compute()) return false;
        if (!ensure_buffer(lut_buf_, 256 * sizeof(uint32_t),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           /*external=*/false))
            return false;
        std::memcpy(lut_buf_.mapped, lut256, 256 * sizeof(uint32_t));

        const CmapPush p = make_push(dst, t, vmin, vmax);
        // The global set is not in flight (this path fence-waits every
        // dispatch), so one reusable descriptor set updated in place is safe.
        write_cmap_set(cmap_set_, dst, lut_buf_);
        const bool ok = submit_once([&](VkCommandBuffer cb) {
            record_cmap_body(cb, dst, p, cmap_set_);
        });
        if (!ok) return false;
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "compute";
        return true;
    }

    // u8 HWC (RGBA8) -> copy from the shared buffer into the texture.
    bool blit_u8(Tex& dst, const CaliperTensor& t) {
        if (t.ndim < 2) return false;
        const uint32_t h = (uint32_t)t.shape[0];
        const uint32_t w = (uint32_t)t.shape[1];
        const uint32_t c = (t.ndim >= 3) ? (uint32_t)t.shape[2] : 1;
        if (c != 4 || (int)w != dst.w || (int)h != dst.h) return false;   // RGBA8 only

        const bool ok = submit_once([&](VkCommandBuffer cb) {
            record_blit_body(cb, dst, w, h);
        });
        if (!ok) return false;
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "blit";
        return true;
    }

    // ---- V4 pipelined update: GPU-ordered CUDA->Vulkan handoff --------------
    // No CPU sync anywhere on the hot path. The texture's timeline semaphore
    // carries the chain: CUDA signals base+1 after its stream-ordered copy,
    // this texture's Vulkan pass waits base+1 on the GPU and signals base+2,
    // and the frame submission waits base+2 before sampling (frame_render).
    // The only host wait is retire() — back-pressure if the PREVIOUS update of
    // this same texture hasn't finished (rare; command-buffer re-record and
    // shared-buffer overwrite safety demand it).
    bool update_pipelined(Tex& dst, const CaliperTensor& t,
                          cudadrv::CUdeviceptr src, uint64_t bytes,
                          bool shared_in_place, const uint32_t* lut256,
                          float vmin, float vmax) {
        Interop& io = dst.interop;
        if (!retire(dst)) return dev_bail("pipelined: retire timed out");

        // Validate + prepare descriptors BEFORE enqueuing anything on CUDA,
        // so a rejected tensor leaves no half-signaled timeline behind.
        const bool is_cmap = (t.dtype == CALIPER_DT_F32 && lut256 != nullptr);
        uint32_t blit_w = 0, blit_h = 0;
        VkPipelineStageFlags wait_stage;
        if (is_cmap) {
            if (!ensure_buffer(io.lut, 256 * sizeof(uint32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               /*external=*/false))
                return dev_bail("pipelined: lut alloc failed");
            std::memcpy(io.lut.mapped, lut256, 256 * sizeof(uint32_t));
            write_cmap_set(io.set, dst, io.lut);
            wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else if (t.dtype == CALIPER_DT_U8) {
            if (t.ndim < 2) return false;
            blit_h = (uint32_t)t.shape[0];
            blit_w = (uint32_t)t.shape[1];
            const uint32_t c = (t.ndim >= 3) ? (uint32_t)t.shape[2] : 1;
            if (c != 4 || (int)blit_w != dst.w || (int)blit_h != dst.h) return false;
            wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else {
            return false;
        }

        // CUDA side, ordered on the PRODUCER's stream when the adapter
        // supplied one (M2a, D24) — stream order puts the copy after the
        // producer's kernels, so the adapter's torch::cuda::synchronize() is
        // elided. NULL keeps the legacy default stream (v1 drained handoff).
        const cudadrv::Api* cu = cudadrv::api();
        cudadrv::CUstream stream = (cudadrv::CUstream)t.stream;
        const uint64_t base = io.timeline_value;
        if (!shared_in_place &&
            cu->cuMemcpyDtoDAsync(io.cuda_ptr, src, (size_t)bytes, stream)
                != cudadrv::CUDA_SUCCESS)
            return dev_bail("pipelined: cuMemcpyDtoDAsync failed");
        cudadrv::ExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = base + 1;
        if (cu->cuSignalExternalSemaphoresAsync(&io.cuda_sem, &sp, 1, stream)
                != cudadrv::CUDA_SUCCESS)
            return dev_bail("pipelined: semaphore signal failed");

        // Vulkan side: re-record this texture's command buffer (retired above)
        // and submit waiting base+1 / signaling base+2. No fence.
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(io.cb, &bi);
        if (is_cmap)
            record_cmap_body(io.cb, dst, make_push(dst, t, vmin, vmax), io.set);
        else
            record_blit_body(io.cb, dst, blit_w, blit_h);
        vkEndCommandBuffer(io.cb);

        uint64_t wait_val = base + 1, signal_val = base + 2;
        VkTimelineSemaphoreSubmitInfoKHR tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
        tsi.waitSemaphoreValueCount = 1;
        tsi.pWaitSemaphoreValues = &wait_val;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &signal_val;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &tsi;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &io.timeline;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &io.cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &io.timeline;
        if (vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            return dev_bail("pipelined: submit failed");

        io.timeline_value = base + 2;
        pending_frame_waits_.emplace_back(io.timeline, io.timeline_value);
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;   // CB's end state
        last_device_path_ = is_cmap ? "compute" : "blit";
        return true;
    }

    // ---- V4 pipelined imported update: update_pipelined minus the D2D copy ----
    // The applet already wrote the imported allocation, so CUDA only SIGNALS
    // base+1 on the producer's stream (no cuMemcpyDtoDAsync); the Vulkan pass
    // GPU-waits base+1 / signals base+2, exactly as update_pipelined does. Reads
    // src.buf at src_off in place. Reuses the texture's own sync objects
    // (ensure_pipeline_objects, which does not touch the interop shared buffer).
    bool update_imported_pipelined(Tex& dst, ImportedAlloc& src, uint64_t offset,
                                   uint64_t bytes, bool is_cmap,
                                   const uint32_t* lut256, const CaliperTensor& desc,
                                   float vmin, float vmax) {
        Interop& io = dst.interop;
        if (!retire(dst)) return dev_bail("import-pipelined: retire timed out");

        uint32_t blit_w = 0, blit_h = 0;
        VkPipelineStageFlags wait_stage;
        if (is_cmap) {
            if (!ensure_buffer(io.lut, 256 * sizeof(uint32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               /*external=*/false))
                return dev_bail("import-pipelined: lut alloc failed");
            std::memcpy(io.lut.mapped, lut256, 256 * sizeof(uint32_t));
            write_cmap_set_src(io.set, dst, io.lut, src.buf, offset, bytes);
            wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        } else {
            if (desc.ndim < 2) return false;
            blit_h = (uint32_t)desc.shape[0];
            blit_w = (uint32_t)desc.shape[1];
            const uint32_t c = (desc.ndim >= 3) ? (uint32_t)desc.shape[2] : 1;
            if (c != 4 || (int)blit_w != dst.w || (int)blit_h != dst.h) return false;
            wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        const cudadrv::Api* cu = cudadrv::api();
        cudadrv::CUstream stream = (cudadrv::CUstream)desc.stream;
        const uint64_t base = io.timeline_value;
        cudadrv::ExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = base + 1;
        if (cu->cuSignalExternalSemaphoresAsync(&io.cuda_sem, &sp, 1, stream)
                != cudadrv::CUDA_SUCCESS)
            return dev_bail("import-pipelined: semaphore signal failed");

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(io.cb, &bi);
        if (is_cmap)
            record_cmap_body(io.cb, dst, make_push(dst, desc, vmin, vmax), io.set);
        else
            record_blit_body_src(io.cb, dst, blit_w, blit_h, src.buf, offset);
        vkEndCommandBuffer(io.cb);

        uint64_t wait_val = base + 1, signal_val = base + 2;
        VkTimelineSemaphoreSubmitInfoKHR tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
        tsi.waitSemaphoreValueCount = 1;
        tsi.pWaitSemaphoreValues = &wait_val;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &signal_val;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &tsi;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &io.timeline;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &io.cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &io.timeline;
        if (vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            return dev_bail("import-pipelined: submit failed");

        io.timeline_value = base + 2;
        pending_frame_waits_.emplace_back(io.timeline, io.timeline_value);
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = is_cmap ? "compute-imported" : "blit-imported";
        return true;
    }

    // Synchronous imported update (stream == NULL: adapter drained per the v1
    // rung contract): the sync device shape (:tex_update_from_device fallback)
    // without any copy — fence-waited submit reading src.buf at src_off. retire()
    // first so a still-in-flight pipelined chain on this texture can't race the
    // shared global cmap_set_ / oneshot fence.
    bool update_imported_sync(Tex& dst, ImportedAlloc& src, uint64_t offset,
                              uint64_t bytes, bool is_cmap,
                              const uint32_t* lut256, const CaliperTensor& desc,
                              float vmin, float vmax) {
        if (!retire(dst)) return dev_bail("import-sync: retire timed out");
        if (is_cmap) {
            if (!ensure_compute()) return false;
            if (!ensure_buffer(lut_buf_, 256 * sizeof(uint32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               /*external=*/false))
                return false;
            std::memcpy(lut_buf_.mapped, lut256, 256 * sizeof(uint32_t));
            const CmapPush p = make_push(dst, desc, vmin, vmax);
            write_cmap_set_src(cmap_set_, dst, lut_buf_, src.buf, offset, bytes);
            const bool ok = submit_once([&](VkCommandBuffer cb) {
                record_cmap_body(cb, dst, p, cmap_set_);
            });
            if (!ok) return false;
            dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            last_device_path_ = "compute-imported";
            return true;
        }
        if (desc.ndim < 2) return false;
        const uint32_t h = (uint32_t)desc.shape[0];
        const uint32_t w = (uint32_t)desc.shape[1];
        const uint32_t c = (desc.ndim >= 3) ? (uint32_t)desc.shape[2] : 1;
        if (c != 4 || (int)w != dst.w || (int)h != dst.h) return false;
        const bool ok = submit_once([&](VkCommandBuffer cb) {
            record_blit_body_src(cb, dst, w, h, src.buf, offset);
        });
        if (!ok) return false;
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "blit-imported";
        return true;
    }

    // ---- members ----
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props_{};
    VkDeviceSize storage_buffer_alignment_ = 256;   // minStorageBufferOffsetAlignment
    ImGui_ImplVulkanH_Window wd_{};
    bool rebuild_swapchain_ = false;

    VkCommandPool oneshot_pool_ = VK_NULL_HANDLE;
    VkFence oneshot_fence_ = VK_NULL_HANDLE;
    Buffer staging_;    // CPU upload path
    Buffer lut_buf_;    // 256-entry colormap LUT

    VkDescriptorSetLayout cmap_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout cmap_pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline cmap_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool cmap_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet cmap_set_ = VK_NULL_HANDLE;

    // CUDA interop (driver API, loaded at runtime; see cuda_driver.h).
    bool external_memory_ok_ = false;   // Vulkan side exports external memory
    bool interop_ok_ = false;           // …AND a UUID-matched CUDA device paired
    bool timeline_ok_ = false;          // device exports timeline semaphores
    bool pipelined_ok_ = false;         // interop + timeline: V4 pipelining live
    uint8_t device_uuid_[16] = {};
    cudadrv::CUcontext cuda_ctx_ = nullptr;
    cudadrv::CUdevice cuda_dev_ = 0;

    // Timeline waits the next frame submission must honor before sampling
    // (semaphore, value) — filled by pipelined updates, consumed by
    // frame_render. Bridge + frame loop are UI-thread-only, so no lock.
    std::vector<std::pair<VkSemaphore, uint64_t>> pending_frame_waits_;

    std::unordered_map<uint64_t, Tex> textures_;
    uint64_t next_id_ = 1;   // 0 is the invalid id

    // Bridge v1.2 imported allocations. Ids are renderer-internal (the bridge
    // maps its own CaliperAllocId -> this); never reused, 0 invalid.
    std::unordered_map<uint64_t, ImportedAlloc> imported_;
    uint64_t next_import_id_ = 1;
    const char* last_device_path_ = "";
    std::set<std::string> dev_seen_;   // messages already logged by dev_note()
};

}  // namespace

std::unique_ptr<HostRenderer> make_vulkan_renderer() {
    return std::make_unique<VulkanRenderer>();
}

}  // namespace caliper_host
