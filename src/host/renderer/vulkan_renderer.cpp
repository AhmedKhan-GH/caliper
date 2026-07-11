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
#include <caliper/services/geometry_v1_1.h>        // CALIPER_GEOM_* topology/mode/flag ids
#include <caliper/services/geometry_v1_2.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <cmath>
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
// caliper.geometry.v1 instanced-point shaders (same build-time discipline).
#include <points_vert_spv.h>
#include <points_frag_spv.h>
// caliper.geometry.v1_1 general-primitive shaders (GLSL twins of Metal's
// kGeomShaderSrc; std140 params block byte-matches PrimParams above).
#include <geom_vert_spv.h>
#include <geom_frag_spv.h>
#include <geom_tex_frag_spv.h>

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

// Push block of shaders/points.vert: proj*view premultiplied host-side so the
// whole block (88 B) fits Vulkan's guaranteed 128-byte push budget.
struct GeomPush {
    float    mvp[16];
    uint32_t pos_base;    // float-element bases: byte offsets / 4
    uint32_t attr_base;
    uint32_t use_attr;
    float    vmin, vmax;
    float    size_px;
};
static_assert(sizeof(GeomPush) == 88, "points.vert push block layout");

// caliper.geometry.v1_1 per-draw params. Byte-identical to Metal's PrimParams
// (metal_renderer.mm PrimParams / kGeomShaderSrc) and to the std140 Params
// block of geom.vert — same member order, same 192-byte size — so both backends
// and the shader agree on every offset. Delivered via a dynamic UBO (exceeds
// Vulkan's 128-B push budget), not push constants. The v1.3 instance tail
// (use_instance/inst_base/use_instance_attr/inst_attr_base) grew the struct in
// textual lockstep on this box; the Vulkan instance-PULL logic that consumes it
// is T4 — only the layout grew here to keep the cross-backend seam aligned.
struct PrimParams {
    float    mvp[16];
    float    nmat0[4];
    float    nmat1[4];
    float    nmat2[4];
    uint32_t pos_base, idx_base, nrm_base, attr_base;
    uint32_t use_index, vertex_count, color_mode, shade_mode;
    uint32_t flat_rgba;
    float    vmin, vmax, size_px;
    uint32_t uv_base, use_instance, inst_base, use_instance_attr;
    uint32_t inst_attr_base, pad0, pad1, pad2;
};
static_assert(sizeof(PrimParams) == 192, "geom.vert std140 params layout");

// Column-major 4x4 multiply out = a*b. Transcribed VERBATIM from
// metal_renderer.mm mat4_mul_cm so both backends premultiply mvp with an
// identical order of float operations (byte-parity of the geometry rows).
inline void mat4_mul_cm(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k)
                acc += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = acc;
        }
}

// Columns of transpose(inverse(upper3x3(view*model))) — the normal matrix
// (§4.4), computed in double precision then truncated. Transcribed VERBATIM
// from metal_renderer.mm normal_matrix_columns; a singular view-model falls
// back to identity so a degenerate transform still yields a defined image.
inline void normal_matrix_columns(const float* view_model,
                                  float* c0, float* c1, float* c2) {
    const double a00 = view_model[0], a01 = view_model[4], a02 = view_model[8];
    const double a10 = view_model[1], a11 = view_model[5], a12 = view_model[9];
    const double a20 = view_model[2], a21 = view_model[6], a22 = view_model[10];
    const double det = a00 * (a11 * a22 - a12 * a21)
                     - a01 * (a10 * a22 - a12 * a20)
                     + a02 * (a10 * a21 - a11 * a20);
    if (std::abs(det) < 1e-12) {
        c0[0] = 1.f; c0[1] = 0.f; c0[2] = 0.f; c0[3] = 0.f;
        c1[0] = 0.f; c1[1] = 1.f; c1[2] = 0.f; c1[3] = 0.f;
        c2[0] = 0.f; c2[1] = 0.f; c2[2] = 1.f; c2[3] = 0.f;
        return;
    }
    const double inv_det = 1.0 / det;
    const double inv00 =  (a11 * a22 - a12 * a21) * inv_det;
    const double inv01 = -(a01 * a22 - a02 * a21) * inv_det;
    const double inv02 =  (a01 * a12 - a02 * a11) * inv_det;
    const double inv10 = -(a10 * a22 - a12 * a20) * inv_det;
    const double inv11 =  (a00 * a22 - a02 * a20) * inv_det;
    const double inv12 = -(a00 * a12 - a02 * a10) * inv_det;
    const double inv20 =  (a10 * a21 - a11 * a20) * inv_det;
    const double inv21 = -(a00 * a21 - a01 * a20) * inv_det;
    const double inv22 =  (a00 * a11 - a01 * a10) * inv_det;
    // Columns of transpose(inverse(A)) are the rows of inverse(A).
    c0[0] = (float)inv00; c0[1] = (float)inv01; c0[2] = (float)inv02; c0[3] = 0.f;
    c1[0] = (float)inv10; c1[1] = (float)inv11; c1[2] = (float)inv12; c1[3] = 0.f;
    c2[0] = (float)inv20; c2[1] = (float)inv21; c2[2] = (float)inv22; c2[3] = 0.f;
}

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
        destroy_geom_prim();
        destroy_geom();
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

    // ---- caliper.geometry.v1: instanced points from imported allocations ----
    // Same gate as the imported-texture path: point data lives in v1.2
    // imported blocks, which only exist when interop + external memory are up.
    bool supports_geometry() const override { return supports_external_import(); }

    // An offscreen render target that is ALSO an ordinary sampled texture:
    // it lives in textures_, so tex_imtexture_id / debug_readback / tex_release
    // work unchanged; the extra COLOR_ATTACHMENT usage + framebuffer make it
    // drawable by the point pass. Cleared to opaque black at create so ImGui
    // sampling before the first draw is defined.
    uint64_t geom_create_view(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == VK_NULL_HANDLE) return 0;
        if (!ensure_geom_objects()) return 0;
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
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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

        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = geom_pass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &t.view;
        fci.width = (uint32_t)w;
        fci.height = (uint32_t)h;
        fci.layers = 1;
        if (vkCreateFramebuffer(device_, &fci, nullptr, &t.fb) != VK_SUCCESS) {
            destroy_tex(t); return 0;
        }

        t.descset = ImGui_ImplVulkan_AddTexture(t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (t.descset == VK_NULL_HANDLE) { destroy_tex(t); return 0; }

        const bool ok = submit_once([&](VkCommandBuffer cb) {
            barrier(cb, t, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue black{};
            black.float32[3] = 1.0f;
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &black, 1, &r);
            barrier(cb, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        if (!ok) { destroy_tex(t); return 0; }
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        uint64_t id = next_id_++;
        textures_[id] = t;
        return id;
    }

    // One view frame, atomically: clear + draw `count` vertex-pulled points.
    // Positions/attr are element bases into the imported blocks (bound whole),
    // so the offset gate is 4-byte alignment — deliberately looser than the
    // descriptor-offset path's minStorageBufferOffsetAlignment. Additive
    // blend, no depth. GL-style NDC (+y up) via the negative-viewport trick
    // (core in Vulkan 1.1). Fenced submit, like the v1.2 sync fallback.
    bool geom_draw_points(uint64_t view_tex, const float* view16,
                          const float* proj16,
                          uint64_t pos_alloc, uint64_t pos_offset,
                          uint64_t count,
                          uint64_t attr_alloc, uint64_t attr_offset,
                          const uint32_t* lut256, float vmin, float vmax,
                          float size_px, uint32_t clear_rgba) override {
        if (!supports_geometry() || !ensure_geom_objects()) return false;
        auto vt = textures_.find(view_tex);
        if (vt == textures_.end() || vt->second.fb == VK_NULL_HANDLE) return false;
        Tex& t = vt->second;

        const ImportedAlloc* pos = nullptr;
        const ImportedAlloc* attr = nullptr;
        if (count > 0) {
            auto pit = imported_.find(pos_alloc);
            if (pit == imported_.end()) return false;
            pos = &pit->second;
            // Renderer re-check of the host-side gates against the REAL
            // allocation (same both-layers discipline as the texture path).
            if (pos_offset % 4 != 0 || count > UINT64_MAX / 12u) return false;
            const uint64_t pos_bytes = count * 12u;
            if (pos_offset > pos->size || pos_bytes > pos->size - pos_offset)
                return false;
            // Element bases ride a 32-bit push constant.
            if (pos_offset / 4 > UINT32_MAX || count > UINT32_MAX) return false;
            if (attr_alloc != 0) {
                auto ait = imported_.find(attr_alloc);
                if (ait == imported_.end() || lut256 == nullptr) return false;
                attr = &ait->second;
                if (attr_offset % 4 != 0) return false;
                const uint64_t attr_bytes = count * 4u;
                if (attr_offset > attr->size || attr_bytes > attr->size - attr_offset)
                    return false;
                if (attr_offset / 4 > UINT32_MAX) return false;
            }
        }

        // Stage the LUT for this draw (fenced submits serialize all draws, so
        // one shared host-visible buffer is race-free).
        if (attr) std::memcpy(geom_lut_.mapped, lut256, 256 * sizeof(uint32_t));

        // Descriptor set: positions, attr (positions again when flat — a valid
        // binding the shader never reads), LUT.
        VkDescriptorBufferInfo bi[3] = {};
        bi[0].buffer = pos ? pos->buf : geom_lut_.buf;   // count==0: never read
        bi[0].range  = VK_WHOLE_SIZE;
        bi[1].buffer = attr ? attr->buf : bi[0].buffer;
        bi[1].range  = VK_WHOLE_SIZE;
        bi[2].buffer = geom_lut_.buf;
        bi[2].range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet wr[3] = {};
        for (int i = 0; i < 3; ++i) {
            wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = geom_set_;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device_, 3, wr, 0, nullptr);

        GeomPush push{};
        // mvp = proj * view, column-major (GLSL default).
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {
                float s = 0.f;
                for (int k = 0; k < 4; ++k)
                    s += proj16[k * 4 + r] * view16[c * 4 + k];
                push.mvp[c * 4 + r] = s;
            }
        push.pos_base  = (uint32_t)(pos_offset / 4);
        push.attr_base = (uint32_t)(attr_offset / 4);
        push.use_attr  = attr ? 1u : 0u;
        push.vmin = vmin;
        push.vmax = vmax;
        push.size_px = size_px < 1.f ? 1.f
                     : (size_px > point_size_max_ ? point_size_max_ : size_px);

        VkClearValue clear{};
        clear.color.float32[0] = (float)((clear_rgba)       & 0xFFu) / 255.f;
        clear.color.float32[1] = (float)((clear_rgba >> 8)  & 0xFFu) / 255.f;
        clear.color.float32[2] = (float)((clear_rgba >> 16) & 0xFFu) / 255.f;
        clear.color.float32[3] = (float)((clear_rgba >> 24) & 0xFFu) / 255.f;

        const bool ok = submit_once([&](VkCommandBuffer cb) {
            // Make the applet's CUDA writes to the imported blocks visible to
            // the vertex stage (external-memory coherence, as on the imported
            // texture path).
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);

            VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rp.renderPass = geom_pass_;
            rp.framebuffer = t.fb;
            rp.renderArea = {{0, 0}, {(uint32_t)t.w, (uint32_t)t.h}};
            rp.clearValueCount = 1;
            rp.pClearValues = &clear;
            vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);

            if (count > 0) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, geom_pipeline_);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        geom_pipe_layout_, 0, 1, &geom_set_, 0, nullptr);
                // Negative viewport height = GL-style NDC (+y up), core 1.1.
                VkViewport vp{};
                vp.x = 0.f;
                vp.y = (float)t.h;
                vp.width  = (float)t.w;
                vp.height = -(float)t.h;
                vp.minDepth = 0.f;
                vp.maxDepth = 1.f;
                vkCmdSetViewport(cb, 0, 1, &vp);
                VkRect2D sc{{0, 0}, {(uint32_t)t.w, (uint32_t)t.h}};
                vkCmdSetScissor(cb, 0, 1, &sc);
                vkCmdPushConstants(cb, geom_pipe_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof(GeomPush), &push);
                vkCmdDraw(cb, (uint32_t)count, 1, 0, 0);
            }
            vkCmdEndRenderPass(cb);   // finalLayout: SHADER_READ_ONLY_OPTIMAL
        });
        if (!ok) return false;
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "points-imported";
        dev_note("geometry path OK — points drawn from imported allocation in place");
        return true;
    }

    // ---- caliper.geometry.v1_1: general primitives from imported allocations ----
    // Same prerequisite as points: the imported-alloc machinery must be live
    // (interop + external memory), exactly mirroring supports_geometry().
    bool supports_geometry_primitives() const override {
        return supports_external_import();
    }
    bool supports_geometry_textured() const override {
        return supports_external_import();
    }

    // Like geom_create_view, plus flags. flags==0 is byte-for-byte the color-only
    // view geom_create_view builds (usable by draw_primitives without depth).
    // CALIPER_GEOM_VIEW_DEPTH additionally attaches a device-local D32-float
    // depth image and binds the framebuffer against geom_pass_depth_. Unknown
    // flag bits are refused (return 0) — never silently ignored.
    uint64_t geom_create_view_ex(int w, int h, uint32_t flags) override {
        if ((flags & ~CALIPER_GEOM_VIEW_DEPTH) != 0u) return 0;   // refuse unknown bits
        if (w <= 0 || h <= 0 || device_ == VK_NULL_HANDLE) return 0;
        if (!ensure_geom_objects()) return 0;                     // geom_pass_ (color)
        const bool want_depth = (flags & CALIPER_GEOM_VIEW_DEPTH) != 0u;
        if (want_depth && !ensure_geom_prim_objects()) return 0;  // geom_pass_depth_

        Tex t{};
        t.w = w; t.h = h;

        // Color image/view: identical to geom_create_view (SAMPLED + COLOR
        // ATTACHMENT, device-local, cleared to opaque black at create).
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {(uint32_t)w, (uint32_t)h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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

        // Depth image/view: device-local, DEPTH_STENCIL_ATTACHMENT only (never
        // sampled, never transferred). Cleared each frame via the render pass
        // loadOp — no clear at create.
        if (want_depth) {
            VkImageCreateInfo dici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            dici.imageType = VK_IMAGE_TYPE_2D;
            dici.format = VK_FORMAT_D32_SFLOAT;
            dici.extent = {(uint32_t)w, (uint32_t)h, 1};
            dici.mipLevels = 1;
            dici.arrayLayers = 1;
            dici.samples = VK_SAMPLE_COUNT_1_BIT;
            dici.tiling = VK_IMAGE_TILING_OPTIMAL;
            dici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            dici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (vkCreateImage(device_, &dici, nullptr, &t.depth_image) != VK_SUCCESS) {
                destroy_tex(t); return 0;
            }
            VkMemoryRequirements dmr;
            vkGetImageMemoryRequirements(device_, t.depth_image, &dmr);
            VkMemoryAllocateInfo dmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            dmai.allocationSize = dmr.size;
            if (!find_mem_type(dmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &dmai.memoryTypeIndex) ||
                vkAllocateMemory(device_, &dmai, nullptr, &t.depth_memory) != VK_SUCCESS) {
                destroy_tex(t); return 0;
            }
            vkBindImageMemory(device_, t.depth_image, t.depth_memory, 0);

            VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            dvci.image = t.depth_image;
            dvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            dvci.format = VK_FORMAT_D32_SFLOAT;
            dvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(device_, &dvci, nullptr, &t.depth_view) != VK_SUCCESS) {
                destroy_tex(t); return 0;
            }
            t.has_depth = true;
        }

        // Framebuffer: color-only against geom_pass_, or color+depth against
        // geom_pass_depth_ (attachment order matches the pass: color 0, depth 1).
        VkImageView atts[2] = {t.view, t.depth_view};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = want_depth ? geom_pass_depth_ : geom_pass_;
        fci.attachmentCount = want_depth ? 2u : 1u;
        fci.pAttachments = atts;
        fci.width = (uint32_t)w;
        fci.height = (uint32_t)h;
        fci.layers = 1;
        if (vkCreateFramebuffer(device_, &fci, nullptr, &t.fb) != VK_SUCCESS) {
            destroy_tex(t); return 0;
        }

        t.descset = ImGui_ImplVulkan_AddTexture(t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (t.descset == VK_NULL_HANDLE) { destroy_tex(t); return 0; }

        const bool ok = submit_once([&](VkCommandBuffer cb) {
            barrier(cb, t, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue black{};
            black.float32[3] = 1.0f;
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &black, 1, &r);
            barrier(cb, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        if (!ok) { destroy_tex(t); return 0; }
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        uint64_t id = next_id_++;
        textures_[id] = t;
        return id;
    }

    // One view frame, atomically (GEOMETRY.md §5.4). Gate EVERY draw against the
    // renderer's OWN imported_ table first (defense in depth, §2.3) — nothing is
    // encoded or cleared unless all draws pass, so a refusal leaves the view's
    // pixels exactly as they were. Then barrier the imported buffers, clear, and
    // encode the draws in array order into a single pass. Fenced submit, same
    // drain semantics as geom_draw_points; indexed draws are issued NON-indexed
    // (the shader pulls + clamps indices, §4.3).
    bool geom_draw_primitives(uint64_t view_tex,
                              const float* view16, const float* proj16,
                              const HostGeomDraw* draws, uint32_t count,
                              uint32_t clear_rgba) override {
        if (!supports_geometry_primitives() || !ensure_geom_objects() ||
            !ensure_geom_prim_objects())
            return dev_bail("primitives: backend objects unavailable");

        // Gate a: view live and a real geometry view (has a framebuffer); camera
        // matrices present (the bridge guarantees cam, but re-gate — §5.5.a).
        auto vt = textures_.find(view_tex);
        if (vt == textures_.end() || vt->second.fb == VK_NULL_HANDLE)
            return dev_bail("primitives: unknown or non-view texture");
        Tex& t = vt->second;
        if (view16 == nullptr || proj16 == nullptr)
            return dev_bail("primitives: null camera matrices");
        if (count > 0 && draws == nullptr)
            return dev_bail("primitives: null draw array");

        // Per-draw record built during gating; encoding only starts once EVERY
        // draw has passed. lut_slot indexes the per-frame LUT ring (colormap
        // draws only; -1 otherwise).
        struct Enc {
            const ImportedAlloc* pos = nullptr;
            const ImportedAlloc* idx = nullptr;
            const ImportedAlloc* nrm = nullptr;
            const ImportedAlloc* attr = nullptr;
            const ImportedAlloc* uv = nullptr;
            const Tex* texture = nullptr;
            uint32_t consumed = 0;
            VkPipeline pipe = VK_NULL_HANDLE;
            int lut_slot = -1;
            const uint32_t* lut256 = nullptr;
            PrimParams params{};
        };
        std::vector<Enc> encs;
        encs.reserve(count);
        uint32_t n_lut = 0;

        for (uint32_t i = 0; i < count; ++i) {
            const HostGeomDraw& d = draws[i];
            Enc e;

            // Gate b: enum ranges (the bridge already range-checked, but re-gate
            // — the renderer must never index its caches out of range).
            if (d.topology > CALIPER_GEOM_TOPO_TRIANGLE_STRIP)
                return dev_bail("primitives: topology out of range");
            if (d.color_mode > CALIPER_GEOM_COLOR_TEXTURE)
                return dev_bail("primitives: color_mode out of range");
            if (d.shade_mode > CALIPER_GEOM_SHADE_LAMBERT)
                return dev_bail("primitives: shade_mode out of range");
            if (d.blend_mode > CALIPER_GEOM_BLEND_ADDITIVE)
                return dev_bail("primitives: blend_mode out of range");

            // Positions: live, 4-aligned, non-empty, overflow-safe bounds. Note
            // pos->size is the IMPORTED (granularity-padded) allocation size.
            auto pit = imported_.find(d.pos_alloc);
            if (pit == imported_.end()) return dev_bail("primitives: unknown pos alloc");
            e.pos = &pit->second;
            if (d.vertex_count == 0) return dev_bail("primitives: zero vertex_count");
            if (d.pos_offset % 4 != 0) return dev_bail("primitives: pos offset misaligned");
            if (d.vertex_count > UINT64_MAX / 12u) return dev_bail("primitives: position byte overflow");
            if (d.pos_offset > e.pos->size || d.vertex_count * 12u > e.pos->size - d.pos_offset)
                return dev_bail("primitives: positions out of bounds");
            // Element bases ride uint32 params fields.
            if (d.pos_offset / 4 > UINT32_MAX || d.vertex_count > UINT32_MAX)
                return dev_bail("primitives: position base exceeds 32 bits");

            // Indexed: consumed = index_count; else vertex_count.
            uint64_t consumed = d.vertex_count;
            if (d.index_alloc != 0) {
                auto iit = imported_.find(d.index_alloc);
                if (iit == imported_.end()) return dev_bail("primitives: unknown index alloc");
                e.idx = &iit->second;
                if (d.index_offset % 4 != 0) return dev_bail("primitives: index offset misaligned");
                if (d.index_count == 0) return dev_bail("primitives: zero index_count");
                if (d.index_count > UINT32_MAX || d.index_count > UINT64_MAX / 4u)
                    return dev_bail("primitives: index count overflow");
                if (d.index_offset > e.idx->size || d.index_count * 4u > e.idx->size - d.index_offset)
                    return dev_bail("primitives: indices out of bounds");
                if (d.index_offset / 4 > UINT32_MAX)
                    return dev_bail("primitives: index base exceeds 32 bits");
                consumed = d.index_count;
            }
            e.consumed = (uint32_t)consumed;

            // Topology minimum-vertex rules (consumed vertices).
            if ((d.topology == CALIPER_GEOM_TOPO_LINES ||
                 d.topology == CALIPER_GEOM_TOPO_LINE_STRIP) && consumed < 2)
                return dev_bail("primitives: line draw has too few vertices");
            if ((d.topology == CALIPER_GEOM_TOPO_TRIANGLES ||
                 d.topology == CALIPER_GEOM_TOPO_TRIANGLE_STRIP) && consumed < 3)
                return dev_bail("primitives: triangle draw has too few vertices");
            // Point size must be positive (mirrors Metal's draw_primitives gate;
            // the value is then clamped to the device range at encode).
            if (d.topology == CALIPER_GEOM_TOPO_POINTS && !(d.size_px > 0.0f))
                return dev_bail("primitives: non-positive point size");

            // Normals: required + bounds-checked for LAMBERT; if supplied for a
            // non-LAMBERT draw, still bounds-checked (mirrors Metal).
            if (d.shade_mode == CALIPER_GEOM_SHADE_LAMBERT) {
                auto nit = imported_.find(d.normal_alloc);
                if (nit == imported_.end()) return dev_bail("primitives: lambert missing normals");
                e.nrm = &nit->second;
                if (d.normal_offset % 4 != 0) return dev_bail("primitives: normal offset misaligned");
                if (d.normal_offset > e.nrm->size ||
                    d.vertex_count * 12u > e.nrm->size - d.normal_offset)
                    return dev_bail("primitives: normals out of bounds");
                if (d.normal_offset / 4 > UINT32_MAX)
                    return dev_bail("primitives: normal base exceeds 32 bits");
            } else if (d.normal_alloc != 0) {
                auto nit = imported_.find(d.normal_alloc);
                if (nit == imported_.end()) return dev_bail("primitives: unknown optional normal alloc");
                e.nrm = &nit->second;
                if (d.normal_offset % 4 != 0) return dev_bail("primitives: optional normal offset misaligned");
                if (d.normal_offset > e.nrm->size ||
                    d.vertex_count * 12u > e.nrm->size - d.normal_offset)
                    return dev_bail("primitives: optional normals out of bounds");
                if (d.normal_offset / 4 > UINT32_MAX)
                    return dev_bail("primitives: optional normal base exceeds 32 bits");
            }

            // Attributes: required + bounds-checked when color_mode != FLAT;
            // COLORMAP additionally needs a resolved LUT (the bridge fills it).
            if (d.color_mode == CALIPER_GEOM_COLOR_COLORMAP ||
                d.color_mode == CALIPER_GEOM_COLOR_VERTEX_RGBA) {
                auto ait = imported_.find(d.attr_alloc);
                if (ait == imported_.end()) return dev_bail("primitives: unknown attr alloc");
                e.attr = &ait->second;
                if (d.attr_offset % 4 != 0) return dev_bail("primitives: attr offset misaligned");
                if (d.vertex_count > UINT64_MAX / 4u) return dev_bail("primitives: attr byte overflow");
                if (d.attr_offset > e.attr->size || d.vertex_count * 4u > e.attr->size - d.attr_offset)
                    return dev_bail("primitives: attributes out of bounds");
                if (d.attr_offset / 4 > UINT32_MAX)
                    return dev_bail("primitives: attr base exceeds 32 bits");
                if (d.color_mode == CALIPER_GEOM_COLOR_COLORMAP) {
                    if (d.lut256 == nullptr) return dev_bail("primitives: colormap missing LUT");
                    e.lut256 = d.lut256;
                    e.lut_slot = (int)n_lut++;
                }
            }

            if (d.color_mode == CALIPER_GEOM_COLOR_TEXTURE) {
                auto uit = imported_.find(d.uv_alloc);
                if (uit == imported_.end())
                    return dev_bail("primitives: unknown uv alloc");
                e.uv = &uit->second;
                if (d.uv_offset % 4 != 0)
                    return dev_bail("primitives: uv offset misaligned");
                if (d.vertex_count > UINT64_MAX / 8u)
                    return dev_bail("primitives: uv byte overflow");
                if (d.uv_offset > e.uv->size ||
                    d.vertex_count * 8u > e.uv->size - d.uv_offset)
                    return dev_bail("primitives: uvs out of bounds");
                if (d.uv_offset / 4 > UINT32_MAX)
                    return dev_bail("primitives: uv base exceeds 32 bits");

                auto tit = textures_.find(d.texture);
                if (tit == textures_.end() || tit->second.fb != VK_NULL_HANDLE ||
                    d.texture == view_tex)
                    return dev_bail("primitives: bad sampled texture");
                if (tit->second.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    return dev_bail("primitives: sampled texture not shader-readable");
                e.texture = &tit->second;
            }

            // Depth flags on a depthless view are refused, never ignored (§2.3.7).
            if (d.depth_flags != 0 && !t.has_depth)
                return dev_bail("primitives: depth flags on depthless view");

            // Cached pipeline for this (topology, blend, depth_flags, pass) combo.
            e.pipe = geom_prim_pipeline(
                d.topology, d.blend_mode, d.depth_flags, t.has_depth,
                d.color_mode == CALIPER_GEOM_COLOR_TEXTURE);
            if (e.pipe == VK_NULL_HANDLE) return dev_bail("primitives: pipeline creation failed");

            // Params (§4.5): mvp = proj*view*model, plus the double-precision
            // normal matrix — both transcribed from the Metal helpers so the two
            // backends compute identical floats.
            float view_model[16], mvp[16];
            mat4_mul_cm(view16, d.model, view_model);
            mat4_mul_cm(proj16, view_model, mvp);
            std::memcpy(e.params.mvp, mvp, sizeof(mvp));
            normal_matrix_columns(view_model, e.params.nmat0, e.params.nmat1, e.params.nmat2);
            e.params.pos_base = (uint32_t)(d.pos_offset / 4);
            e.params.idx_base = (uint32_t)(d.index_offset / 4);
            e.params.nrm_base = (uint32_t)(d.normal_offset / 4);
            e.params.attr_base = (uint32_t)(d.attr_offset / 4);
            e.params.use_index = d.index_alloc != 0 ? 1u : 0u;
            e.params.vertex_count = (uint32_t)d.vertex_count;
            e.params.color_mode = d.color_mode;
            e.params.shade_mode = d.shade_mode;
            e.params.flat_rgba = d.flat_rgba;
            e.params.vmin = d.vmin;
            e.params.vmax = d.vmax;
            e.params.size_px = d.size_px < 1.f ? 1.f
                             : (d.size_px > point_size_max_ ? point_size_max_ : d.size_px);
            e.params.uv_base = (uint32_t)(d.uv_offset / 4);
            encs.push_back(e);
        }

        // All gates passed. Size the per-frame rings (grown ×2 on demand; the
        // fenced-submit model means no prior submit is still reading the old
        // buffers, so growth here is safe). Params: one 256-aligned dynamic-UBO
        // slot per draw. LUT: one 1 KB SSBO slot per colormap draw.
        if (count > 0) {
            const VkDeviceSize need = (VkDeviceSize)count * params_slot_;
            if (geom_prim_params_.buf == VK_NULL_HANDLE || geom_prim_params_.size < need) {
                VkDeviceSize grow = geom_prim_params_.size ? geom_prim_params_.size : params_slot_;
                while (grow < need) grow *= 2;
                if (!ensure_buffer(geom_prim_params_, grow,
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false))
                    return dev_bail("primitives: params ring alloc failed");
            }
        }
        if (n_lut > 0) {
            const VkDeviceSize need = (VkDeviceSize)n_lut * kLutSlot;
            if (geom_prim_lut_.buf == VK_NULL_HANDLE || geom_prim_lut_.size < need) {
                VkDeviceSize grow = geom_prim_lut_.size ? geom_prim_lut_.size : kLutSlot;
                while (grow < need) grow *= 2;
                if (!ensure_buffer(geom_prim_lut_, grow,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false))
                    return dev_bail("primitives: LUT ring alloc failed");
            }
        }

        // Write params (one slot per draw) and LUTs (one slot per colormap draw)
        // into the host-coherent rings; the queue submit makes them GPU-visible.
        for (uint32_t i = 0; i < count; ++i) {
            std::memcpy((uint8_t*)geom_prim_params_.mapped + (size_t)i * params_slot_,
                        &encs[i].params, sizeof(PrimParams));
            if (encs[i].lut_slot >= 0)
                std::memcpy((uint8_t*)geom_prim_lut_.mapped +
                            (size_t)encs[i].lut_slot * kLutSlot,
                            encs[i].lut256, 256 * sizeof(uint32_t));
        }

        // Per-frame descriptor sets: one per draw, from a pool reset each call
        // (no prior set is in flight after the previous fenced submit drained).
        std::vector<VkDescriptorSet> sets;
        if (count > 0) {
            if (!ensure_geom_prim_pool(count))
                return dev_bail("primitives: descriptor pool alloc failed");
            vkResetDescriptorPool(device_, geom_prim_pool_, 0);
            std::vector<VkDescriptorSetLayout> layouts(count, geom_prim_set_layout_);
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = geom_prim_pool_;
            dai.descriptorSetCount = count;
            dai.pSetLayouts = layouts.data();
            sets.resize(count);
            if (vkAllocateDescriptorSets(device_, &dai, sets.data()) != VK_SUCCESS)
                return dev_bail("primitives: descriptor set alloc failed");

            for (uint32_t i = 0; i < count; ++i) {
                const Enc& e = encs[i];
                // Absent source (idx/nrm/attr) → bind pos as a harmless
                // placeholder; the shader never reads it (the v1 trick).
                VkDescriptorBufferInfo bi[7] = {};
                bi[0].buffer = e.pos->buf;                    bi[0].range = VK_WHOLE_SIZE;
                bi[1].buffer = e.idx ? e.idx->buf : e.pos->buf;  bi[1].range = VK_WHOLE_SIZE;
                bi[2].buffer = e.nrm ? e.nrm->buf : e.pos->buf;  bi[2].range = VK_WHOLE_SIZE;
                bi[3].buffer = e.attr ? e.attr->buf : e.pos->buf; bi[3].range = VK_WHOLE_SIZE;
                if (e.lut_slot >= 0) {
                    bi[4].buffer = geom_prim_lut_.buf;
                    bi[4].offset = (VkDeviceSize)e.lut_slot * kLutSlot;
                    bi[4].range  = 256 * sizeof(uint32_t);
                } else {
                    bi[4].buffer = e.pos->buf;                bi[4].range = VK_WHOLE_SIZE;
                }
                bi[5].buffer = geom_prim_params_.buf;         // dynamic UBO base 0
                bi[5].offset = 0;
                bi[5].range  = sizeof(PrimParams);
                bi[6].buffer = e.uv ? e.uv->buf : e.pos->buf;
                bi[6].range = VK_WHOLE_SIZE;

                VkWriteDescriptorSet wr[8] = {};
                for (int b = 0; b < 7; ++b) {
                    wr[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    wr[b].dstSet = sets[i];
                    wr[b].dstBinding = (uint32_t)b;
                    wr[b].descriptorCount = 1;
                    wr[b].pBufferInfo = &bi[b];
                    wr[b].descriptorType = (b == 5)
                        ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                }
                uint32_t write_count = 7;
                VkDescriptorImageInfo sampled{};
                if (e.texture != nullptr) {
                    sampled.sampler = geom_prim_sampler_;
                    sampled.imageView = e.texture->view;
                    sampled.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    wr[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    wr[7].dstSet = sets[i];
                    wr[7].dstBinding = 7;
                    wr[7].descriptorCount = 1;
                    wr[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wr[7].pImageInfo = &sampled;
                    write_count = 8;
                }
                vkUpdateDescriptorSets(device_, write_count, wr, 0, nullptr);
            }
        }

        // Clear values: color unpacked LE (as v1), depth 1.0 when present.
        VkClearValue clears[2] = {};
        clears[0].color.float32[0] = (float)((clear_rgba)       & 0xFFu) / 255.f;
        clears[0].color.float32[1] = (float)((clear_rgba >> 8)  & 0xFFu) / 255.f;
        clears[0].color.float32[2] = (float)((clear_rgba >> 16) & 0xFFu) / 255.f;
        clears[0].color.float32[3] = (float)((clear_rgba >> 24) & 0xFFu) / 255.f;
        clears[1].depthStencil.depth = 1.0f;

        const bool ok = submit_once([&](VkCommandBuffer cb) {
            // Make the applet's CUDA writes to the imported blocks visible to the
            // vertex stage — one global barrier covers the union of referenced
            // imported buffers (the geom_draw_points discipline).
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);

            VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rp.renderPass = t.has_depth ? geom_pass_depth_ : geom_pass_;
            rp.framebuffer = t.fb;
            rp.renderArea = {{0, 0}, {(uint32_t)t.w, (uint32_t)t.h}};
            rp.clearValueCount = t.has_depth ? 2u : 1u;
            rp.pClearValues = clears;
            vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);

            // Negative viewport height = GL-style NDC (+y up), core 1.1 — shared
            // by every pipeline via dynamic state, so set once for the pass.
            VkViewport vp{};
            vp.x = 0.f;
            vp.y = (float)t.h;
            vp.width  = (float)t.w;
            vp.height = -(float)t.h;
            vp.minDepth = 0.f;
            vp.maxDepth = 1.f;
            vkCmdSetViewport(cb, 0, 1, &vp);
            VkRect2D sc{{0, 0}, {(uint32_t)t.w, (uint32_t)t.h}};
            vkCmdSetScissor(cb, 0, 1, &sc);

            for (uint32_t i = 0; i < count; ++i) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, encs[i].pipe);
                const uint32_t dyn = (uint32_t)((VkDeviceSize)i * params_slot_);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        geom_prim_pipe_layout_, 0, 1, &sets[i],
                                        1, &dyn);
                // Indexed draws are issued NON-indexed with index_count vertices;
                // the shader pulls idx[] and clamps to vertex_count-1 (§4.3).
                vkCmdDraw(cb, encs[i].consumed, 1, 0, 0);
            }
            vkCmdEndRenderPass(cb);   // color -> SHADER_READ_ONLY_OPTIMAL
        });
        if (!ok) return dev_bail("primitives: submit failed");
        t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "primitives-imported";
        dev_note("geometry path OK — primitives drawn from imported allocations in place");
        return true;
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
        if (mr.size > size_bytes) {
            // Binding a buffer bigger than the imported memory is invalid
            // usage. Never expected (CUDA granularity is 2 MiB multiples),
            // but if the padded-size assumption ever breaks, fail closed and
            // report both sizes — the caller falls back to the copy path.
            std::fprintf(stderr,
                         "[vulkan] import: buffer needs %llu bytes > imported "
                         "%llu — declining import\n",
                         (unsigned long long)mr.size,
                         (unsigned long long)size_bytes);
            vkDestroyBuffer(device_, a.buf, nullptr);
            CloseHandle(dup);
            return 0;
        }
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
    // One geometry.v1_1 LUT ring slot: 256 RGBA8 entries = 1 KB, a multiple of
    // minStorageBufferOffsetAlignment (256) so slot i at i*kLutSlot is bindable.
    static constexpr VkDeviceSize kLutSlot = 256 * sizeof(uint32_t);

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
        VkFramebuffer fb = VK_NULL_HANDLE;  // geometry.v1 view render target
        // geometry.v1_1 optional depth attachment (CALIPER_GEOM_VIEW_DEPTH).
        // Device-local D32-float, never sampled, never an id; the framebuffer
        // then binds against geom_pass_depth_ instead of geom_pass_.
        VkImage        depth_image  = VK_NULL_HANDLE;
        VkDeviceMemory depth_memory = VK_NULL_HANDLE;
        VkImageView    depth_view   = VK_NULL_HANDLE;
        bool           has_depth    = false;
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
        if (t.fb) { vkDestroyFramebuffer(device_, t.fb, nullptr); t.fb = VK_NULL_HANDLE; }
        if (t.descset) { ImGui_ImplVulkan_RemoveTexture(t.descset); t.descset = VK_NULL_HANDLE; }
        if (t.view) { vkDestroyImageView(device_, t.view, nullptr); t.view = VK_NULL_HANDLE; }
        if (t.image) { vkDestroyImage(device_, t.image, nullptr); t.image = VK_NULL_HANDLE; }
        if (t.memory) { vkFreeMemory(device_, t.memory, nullptr); t.memory = VK_NULL_HANDLE; }
        // geometry.v1_1 depth attachment (view created via geom_create_view_ex
        // with CALIPER_GEOM_VIEW_DEPTH); depthless views leave these null.
        if (t.depth_view) { vkDestroyImageView(device_, t.depth_view, nullptr); t.depth_view = VK_NULL_HANDLE; }
        if (t.depth_image) { vkDestroyImage(device_, t.depth_image, nullptr); t.depth_image = VK_NULL_HANDLE; }
        if (t.depth_memory) { vkFreeMemory(device_, t.depth_memory, nullptr); t.depth_memory = VK_NULL_HANDLE; }
        t.has_depth = false;
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

    // ---- geometry.v1 pipeline objects (lazy, once; destroyed at shutdown) ---
    bool ensure_geom_objects() {
        if (geom_pipeline_ != VK_NULL_HANDLE) return true;

        VkPhysicalDeviceProperties pd{};
        vkGetPhysicalDeviceProperties(physical_, &pd);
        point_size_max_ = pd.limits.pointSizeRange[1] > 1.f
                              ? pd.limits.pointSizeRange[1] : 1.f;

        // Render pass: one RGBA8 color attachment, clear-on-load, ends
        // SHADER_READ_ONLY so a drawn view is immediately sampleable.
        VkAttachmentDescription att{};
        att.format = VK_FORMAT_R8G8B8A8_UNORM;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;   // cleared every draw
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &ar;
        VkSubpassDependency deps[2] = {};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpi.attachmentCount = 1;
        rpi.pAttachments = &att;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 2;
        rpi.pDependencies = deps;
        if (vkCreateRenderPass(device_, &rpi, nullptr, &geom_pass_) != VK_SUCCESS)
            return false;

        // Set layout: three storage buffers (positions, attr, LUT), vertex stage.
        VkDescriptorSetLayoutBinding bindings[3] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 3;
        li.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &li, nullptr, &geom_layout_) != VK_SUCCESS) {
            destroy_geom(); return false;
        }

        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GeomPush)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &geom_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device_, &pli, nullptr, &geom_pipe_layout_) != VK_SUCCESS) {
            destroy_geom(); return false;
        }

        VkShaderModuleCreateInfo vmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vmi.codeSize = sizeof(kPointsVertSpv);
        vmi.pCode = kPointsVertSpv;
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &vmi, nullptr, &vs) != VK_SUCCESS) {
            destroy_geom(); return false;
        }
        VkShaderModuleCreateInfo fmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        fmi.codeSize = sizeof(kPointsFragSpv);
        fmi.pCode = kPointsFragSpv;
        if (vkCreateShaderModule(device_, &fmi, nullptr, &fs) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, vs, nullptr);
            destroy_geom(); return false;
        }

        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable = VK_TRUE;                     // additive: glow, order-free
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.alphaBlendOp = VK_BLEND_OP_ADD;
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &ba;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2;
        ds.pDynamicStates = dyn;

        VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpi.stageCount = 2;
        gpi.pStages = stages;
        gpi.pVertexInputState = &vin;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vps;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms;
        gpi.pColorBlendState = &cb;
        gpi.pDynamicState = &ds;
        gpi.layout = geom_pipe_layout_;
        gpi.renderPass = geom_pass_;
        gpi.subpass = 0;
        const VkResult pr = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1,
                                                      &gpi, nullptr, &geom_pipeline_);
        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
        if (pr != VK_SUCCESS) { destroy_geom(); return false; }

        VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &psz;
        if (vkCreateDescriptorPool(device_, &dpi, nullptr, &geom_pool_) != VK_SUCCESS) {
            destroy_geom(); return false;
        }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = geom_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &geom_layout_;
        if (vkAllocateDescriptorSets(device_, &dai, &geom_set_) != VK_SUCCESS) {
            destroy_geom(); return false;
        }

        if (!ensure_buffer(geom_lut_, 256 * sizeof(uint32_t),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           /*external=*/false)) {
            destroy_geom(); return false;
        }
        return true;
    }

    void destroy_geom() {
        destroy_buffer(geom_lut_);
        if (geom_pool_) { vkDestroyDescriptorPool(device_, geom_pool_, nullptr); geom_pool_ = VK_NULL_HANDLE; geom_set_ = VK_NULL_HANDLE; }
        if (geom_pipeline_) { vkDestroyPipeline(device_, geom_pipeline_, nullptr); geom_pipeline_ = VK_NULL_HANDLE; }
        if (geom_pipe_layout_) { vkDestroyPipelineLayout(device_, geom_pipe_layout_, nullptr); geom_pipe_layout_ = VK_NULL_HANDLE; }
        if (geom_layout_) { vkDestroyDescriptorSetLayout(device_, geom_layout_, nullptr); geom_layout_ = VK_NULL_HANDLE; }
        if (geom_pass_) { vkDestroyRenderPass(device_, geom_pass_, nullptr); geom_pass_ = VK_NULL_HANDLE; }
    }

    // caliper.geometry.v1_1 shared objects (lazy; destroyed at shutdown). The v1
    // members above are FROZEN — this builds a SEPARATE depth-capable render
    // pass, the 6-binding descriptor/pipeline layout, and computes the dynamic-
    // UBO slot stride. Pipelines themselves are cached per (topology, blend,
    // depth, pass) in geom_prim_pipeline; the rings + pool grow lazily in the
    // draw path. Returns false (with partial-state cleanup) on any failure.
    bool ensure_geom_prim_objects() {
        if (geom_prim_pipe_layout_ != VK_NULL_HANDLE) return true;

        // Dynamic-UBO slot stride: 256 (the spec's pinned value; ≤ every device's
        // minUniformBufferOffsetAlignment budget) rounded up if a device somehow
        // demands more. 256 is a multiple of any power-of-two alignment ≤ 256, so
        // slot i at i*params_slot_ is always a valid dynamic offset.
        VkPhysicalDeviceProperties pd{};
        vkGetPhysicalDeviceProperties(physical_, &pd);
        params_slot_ = 256;
        while (params_slot_ < pd.limits.minUniformBufferOffsetAlignment)
            params_slot_ *= 2;

        // Depth render pass: color identical to geom_pass_ (CLEAR -> STORE, ends
        // SHADER_READ_ONLY_OPTIMAL) plus a D32 depth attachment (CLEAR, DONT_CARE
        // store, ends DEPTH_STENCIL_ATTACHMENT_OPTIMAL — never sampled).
        VkAttachmentDescription atts[2] = {};
        atts[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        atts[1].format = VK_FORMAT_D32_SFLOAT;
        atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference car{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dar{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &car;
        sp.pDepthStencilAttachment = &dar;
        VkSubpassDependency deps[2] = {};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpi.attachmentCount = 2;
        rpi.pAttachments = atts;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 2;
        rpi.pDependencies = deps;
        if (vkCreateRenderPass(device_, &rpi, nullptr, &geom_pass_depth_) != VK_SUCCESS) {
            destroy_geom_prim(); return false;
        }

        // Set layout: bindings 0-4 storage buffers (pos/idx/nrm/attr/lut),
        // binding 5 a dynamic uniform buffer (params) — all vertex stage.
        VkDescriptorSetLayoutBinding bindings[8] = {};
        for (uint32_t i = 0; i < 7; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = (i == 5)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        }
        bindings[7].binding = 7;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 8;
        li.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &li, nullptr, &geom_prim_set_layout_) != VK_SUCCESS) {
            destroy_geom_prim(); return false;
        }

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.minLod = 0.f;
        sci.maxLod = 0.f;
        sci.maxAnisotropy = 1.f;
        if (vkCreateSampler(device_, &sci, nullptr, &geom_prim_sampler_) != VK_SUCCESS) {
            destroy_geom_prim(); return false;
        }

        // Pipeline layout: one set, NO push constants (params exceed the 128-B
        // budget — hence the dynamic UBO).
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &geom_prim_set_layout_;
        if (vkCreatePipelineLayout(device_, &pli, nullptr, &geom_prim_pipe_layout_) != VK_SUCCESS) {
            destroy_geom_prim(); return false;
        }
        return true;
    }

    // Lazy pipeline for a (topology, blend, depth_flags, has_depth_pass) combo,
    // built once from the single kGeomVertSpv/kGeomFragSpv pair. Depthless passes
    // (geom_pass_) carry no depth state; depth passes (geom_pass_depth_) use
    // LESS_OR_EQUAL when testing (else ALWAYS, so a write-without-test still
    // writes — Metal parity) and write per DEPTH_WRITE. VK_NULL_HANDLE on failure.
    VkPipeline geom_prim_pipeline(uint32_t topology, uint32_t blend,
                                  uint32_t depth_flags, bool has_depth,
                                  bool textured) {
        const uint32_t key = (topology & 0x7u) | ((blend & 0x3u) << 3) |
                             ((depth_flags & 0x3u) << 5) |
                             (has_depth ? (1u << 7) : 0u) |
                             (textured ? (1u << 8) : 0u);
        auto hit = geom_prim_pipelines_.find(key);
        if (hit != geom_prim_pipelines_.end()) return hit->second;

        VkPrimitiveTopology vt;
        switch (topology) {
            case CALIPER_GEOM_TOPO_POINTS:         vt = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
            case CALIPER_GEOM_TOPO_LINES:          vt = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
            case CALIPER_GEOM_TOPO_LINE_STRIP:     vt = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
            case CALIPER_GEOM_TOPO_TRIANGLES:      vt = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
            case CALIPER_GEOM_TOPO_TRIANGLE_STRIP: vt = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
            default: return VK_NULL_HANDLE;
        }

        VkShaderModuleCreateInfo vmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vmi.codeSize = sizeof(kGeomVertSpv);
        vmi.pCode = kGeomVertSpv;
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &vmi, nullptr, &vs) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        VkShaderModuleCreateInfo fmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        fmi.codeSize = textured ? sizeof(kGeomTexFragSpv) : sizeof(kGeomFragSpv);
        fmi.pCode = textured ? kGeomTexFragSpv : kGeomFragSpv;
        if (vkCreateShaderModule(device_, &fmi, nullptr, &fs) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, vs, nullptr);
            return VK_NULL_HANDLE;
        }

        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = vt;
        ia.primitiveRestartEnable = VK_FALSE;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;                   // two-sided (§1.2)
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.f;                                // portable thick lines are triangles
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Depth-stencil: only for depth-pass pipelines. Test enabled so a write-
        // without-test still writes (compare ALWAYS); compare LESS_OR_EQUAL when
        // DEPTH_TEST (fixed, enables coplanar overlays, §4.2); write per DEPTH_WRITE.
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dss.depthTestEnable = VK_TRUE;
        dss.depthWriteEnable = (depth_flags & CALIPER_GEOM_DEPTH_WRITE) ? VK_TRUE : VK_FALSE;
        dss.depthCompareOp = (depth_flags & CALIPER_GEOM_DEPTH_TEST)
            ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_ALWAYS;
        dss.depthBoundsTestEnable = VK_FALSE;
        dss.stencilTestEnable = VK_FALSE;

        // Blend per §4.2 (ADDITIVE byte-identical to v1 points).
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (blend == CALIPER_GEOM_BLEND_ALPHA) {
            ba.blendEnable = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.colorBlendOp = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.alphaBlendOp = VK_BLEND_OP_ADD;
        } else if (blend == CALIPER_GEOM_BLEND_ADDITIVE) {
            ba.blendEnable = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.colorBlendOp = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.alphaBlendOp = VK_BLEND_OP_ADD;
        } else {
            ba.blendEnable = VK_FALSE;                      // OPAQUE
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &ba;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2;
        ds.pDynamicStates = dyn;

        VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpi.stageCount = 2;
        gpi.pStages = stages;
        gpi.pVertexInputState = &vin;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vps;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms;
        gpi.pDepthStencilState = has_depth ? &dss : nullptr;
        gpi.pColorBlendState = &cb;
        gpi.pDynamicState = &ds;
        gpi.layout = geom_prim_pipe_layout_;
        gpi.renderPass = has_depth ? geom_pass_depth_ : geom_pass_;
        gpi.subpass = 0;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult pr = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1,
                                                      &gpi, nullptr, &pipe);
        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
        if (pr != VK_SUCCESS) return VK_NULL_HANDLE;
        geom_prim_pipelines_[key] = pipe;
        return pipe;
    }

    // Grow the per-frame descriptor pool to hold `need_sets` sets (5 storage + 1
    // dynamic-uniform each). Reset (not per-set free) each draw call, so the pool
    // only needs recreating when a frame asks for more sets than ever before.
    bool ensure_geom_prim_pool(uint32_t need_sets) {
        if (geom_prim_pool_ != VK_NULL_HANDLE && geom_prim_pool_cap_ >= need_sets)
            return true;
        if (geom_prim_pool_) {
            vkDestroyDescriptorPool(device_, geom_prim_pool_, nullptr);
            geom_prim_pool_ = VK_NULL_HANDLE;
            geom_prim_pool_cap_ = 0;
        }
        uint32_t cap = geom_prim_pool_cap_ ? geom_prim_pool_cap_ : 1u;
        while (cap < need_sets) cap *= 2;
        VkDescriptorPoolSize sizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6u * cap},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1u * cap},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u * cap},
        };
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = cap;
        dpi.poolSizeCount = 3;
        dpi.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(device_, &dpi, nullptr, &geom_prim_pool_) != VK_SUCCESS)
            return false;
        geom_prim_pool_cap_ = cap;
        return true;
    }

    void destroy_geom_prim() {
        destroy_buffer(geom_prim_params_);
        destroy_buffer(geom_prim_lut_);
        if (geom_prim_pool_) {
            vkDestroyDescriptorPool(device_, geom_prim_pool_, nullptr);
            geom_prim_pool_ = VK_NULL_HANDLE; geom_prim_pool_cap_ = 0;
        }
        for (auto& kv : geom_prim_pipelines_)
            vkDestroyPipeline(device_, kv.second, nullptr);
        geom_prim_pipelines_.clear();
        if (geom_prim_sampler_) { vkDestroySampler(device_, geom_prim_sampler_, nullptr); geom_prim_sampler_ = VK_NULL_HANDLE; }
        if (geom_prim_pipe_layout_) { vkDestroyPipelineLayout(device_, geom_prim_pipe_layout_, nullptr); geom_prim_pipe_layout_ = VK_NULL_HANDLE; }
        if (geom_prim_set_layout_) { vkDestroyDescriptorSetLayout(device_, geom_prim_set_layout_, nullptr); geom_prim_set_layout_ = VK_NULL_HANDLE; }
        if (geom_pass_depth_) { vkDestroyRenderPass(device_, geom_pass_depth_, nullptr); geom_pass_depth_ = VK_NULL_HANDLE; }
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

    // caliper.geometry.v1 point pipeline (lazy; destroy_geom at shutdown).
    VkRenderPass geom_pass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout geom_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout geom_pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline geom_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool geom_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet geom_set_ = VK_NULL_HANDLE;
    Buffer geom_lut_;                 // per-draw LUT staging (serialized draws)
    float point_size_max_ = 1.f;

    // caliper.geometry.v1_1 primitive path (lazy; destroy_geom_prim at shutdown).
    // Separate from the frozen v1 members above: its own depth render pass,
    // 6-binding layout, per-combo pipeline cache, per-frame descriptor pool, and
    // two host-coherent rings (dynamic-UBO params, one slot per draw; LUTs, one
    // 1 KB slot per colormap draw). geom_pass_ (color-only) is reused for
    // depthless views.
    VkRenderPass geom_pass_depth_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout geom_prim_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout geom_prim_pipe_layout_ = VK_NULL_HANDLE;
    VkSampler geom_prim_sampler_ = VK_NULL_HANDLE;
    std::unordered_map<uint32_t, VkPipeline> geom_prim_pipelines_;
    VkDescriptorPool geom_prim_pool_ = VK_NULL_HANDLE;
    uint32_t geom_prim_pool_cap_ = 0;
    Buffer geom_prim_params_;         // dynamic-UBO ring
    Buffer geom_prim_lut_;            // colormap LUT ring
    VkDeviceSize params_slot_ = 256;  // per-draw dynamic-UBO slot stride

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
