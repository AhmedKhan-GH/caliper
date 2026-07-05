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
// Synchronization is the v1 sync-then-update contract (§7.2), same as Metal:
// the adapter drains the producer (torch::cuda::synchronize()), our DtoD copy
// is followed by cuCtxSynchronize(), and every Vulkan submission here is
// fence-waited (Metal's waitUntilCompleted). Shared timeline semaphores are a
// Phase-4+ optimization, not a correctness requirement at v1.
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

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
// volk shims the Win32 types vulkan_win32.h needs instead of pulling in
// <windows.h> (deliberate, see volk.h) — so declare the one kernel32 call we
// use the same way rather than dragging the whole header in after the shims.
extern "C" __declspec(dllimport) int __stdcall CloseHandle(void* hObject);
#endif

// Build-time SPIR-V of shaders/colormap.comp (glslang -V --vn kColormapSpv),
// byte-identical index math to the CPU reference (§16).
#include <colormap_spv.h>

namespace caliper_host {
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
        return true;
    }

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
                    VK_IMAGE_USAGE_STORAGE_BIT;
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
        if (!external_memory_ok_) return dev_bail("external-memory interop unavailable");

        const uint64_t elem = (t.dtype == CALIPER_DT_F32) ? 4 : 1;
        if (t.dtype != CALIPER_DT_F32 && t.dtype != CALIPER_DT_U8)
            return dev_bail("dtype not f32/u8");
        const uint64_t bytes = tensor_extent_bytes(t, elem);

        if (!ensure_cuda()) return dev_bail("CUDA context init failed");
        if (!ensure_shared_buffer(dst, bytes)) return dev_bail("shared-buffer import failed");

        // In-VRAM copy: torch allocation -> shared allocation, then drain so
        // the Vulkan pass below (fence-waited) reads finished bytes.
        const cudadrv::Api* cu = cudadrv::api();
        cu->cuCtxSetCurrent(cuda_ctx_);
        if (cu->cuMemcpyDtoD(dst.interop.cuda_ptr,
                             (cudadrv::CUdeviceptr)(uintptr_t)t.data,
                             (size_t)bytes) != cudadrv::CUDA_SUCCESS)
            return dev_bail("cuMemcpyDtoD failed");
        if (cu->cuCtxSynchronize() != cudadrv::CUDA_SUCCESS)
            return dev_bail("cuCtxSynchronize failed");

        bool ok = false;
        if (t.dtype == CALIPER_DT_F32 && lut256 != nullptr)
            ok = colormap_compute(dst, t, lut256, vmin, vmax);
        else if (t.dtype == CALIPER_DT_U8)
            ok = blit_u8(dst, t);
        if (ok) dev_note("CUDA interop OK — zero-copy VRAM path");
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
            int score = 0;
            if (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 4;
            if (ext_mem) score += 2;
            if (score > best_score) {
                best_score = score;
                physical_ = d;
                external_memory_ok_ = ext_mem;
                std::memcpy(device_uuid_, idp.deviceUUID, 16);
            }
        }
        if (physical_ == VK_NULL_HANDLE) return false;

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
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.data();
        if (vkCreateDevice(physical_, &dci, nullptr, &device_) != VK_SUCCESS) return false;
        volkLoadDevice(device_);
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem_props_);
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

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &img_acq;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &fd->CommandBuffer;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &rend_done;
        vkQueueSubmit(queue_, 1, &si, fd->Fence);
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

    void destroy_interop(Interop& io) {
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
        cuda_dev_ = 0;
        for (int i = 0; i < count; ++i) {
            cudadrv::CUuuid u{};
            if (cu->cuDeviceGetUuid(&u, i) == cudadrv::CUDA_SUCCESS &&
                std::memcmp(u.bytes, device_uuid_, 16) == 0) {
                cuda_dev_ = i;
                break;
            }
        }
        return cu->cuDevicePrimaryCtxRetain(&cuda_ctx_, cuda_dev_) == cudadrv::CUDA_SUCCESS;
    }

    void release_cuda() {
        const cudadrv::Api* cu = cudadrv::api();
        if (cuda_ctx_ && cu) {
            cu->cuDevicePrimaryCtxRelease(cuda_dev_);
            cuda_ctx_ = nullptr;
        }
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

        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1;
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

        CmapPush p{};
        p.w = (uint32_t)dst.w;
        p.h = (uint32_t)dst.h;
        p.sx = (t.ndim >= 1) ? (uint32_t)t.strides[t.ndim - 1] : 1u;
        p.sy = (t.ndim >= 2) ? (uint32_t)t.strides[t.ndim - 2] : p.w;
        p.vmin = vmin;
        p.vmax = vmax;

        // The set is not in flight (every dispatch below is fence-waited), so
        // one reusable descriptor set updated in place is safe.
        VkDescriptorBufferInfo src_info{dst.interop.shared.buf, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo lut_info{lut_buf_.buf, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo img_info{VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[3]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cmap_set_, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &src_info, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cmap_set_, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lut_info, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cmap_set_, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img_info, nullptr, nullptr};
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

        const bool ok = submit_once([&](VkCommandBuffer cb) {
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
                                    cmap_pipe_layout_, 0, 1, &cmap_set_, 0, nullptr);
            vkCmdPushConstants(cb, cmap_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(p), &p);
            vkCmdDispatch(cb, (p.w + 15) / 16, (p.h + 15) / 16, 1);
            barrier(cb, dst, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
            barrier(cb, dst, dst.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(cb, dst.interop.shared.buf, dst.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        if (!ok) return false;
        dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        last_device_path_ = "blit";
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
    bool external_memory_ok_ = false;
    uint8_t device_uuid_[16] = {};
    cudadrv::CUcontext cuda_ctx_ = nullptr;
    cudadrv::CUdevice cuda_dev_ = 0;

    std::unordered_map<uint64_t, Tex> textures_;
    uint64_t next_id_ = 1;   // 0 is the invalid id
    const char* last_device_path_ = "";
    std::set<std::string> dev_seen_;   // messages already logged by dev_note()
};

}  // namespace

std::unique_ptr<HostRenderer> make_vulkan_renderer() {
    return std::make_unique<VulkanRenderer>();
}

}  // namespace caliper_host
