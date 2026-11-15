// ===========================================================================
// embed_views.cpp — the view layer of the embed core: the v1.2 built-in tensor
// views (caliper_core_view_*), the hosted core (embed_core.h) and the
// one-core-per-process slot they share with caliper_core_create.
//
// Split out of embed_core.cpp so that a host without the native core's
// lifecycle (the applet loader, the service registry, the job system, the app
// paths: the WebAssembly viewer in web/wasm) can link the views and the hosted
// core alone. Everything here needs only ImGui, the renderer seam and the
// process TensorBridge (services_bridge()); embed_core.cpp keeps create,
// shutdown, the canvas, the frame arc, events and applets, and calls back in
// through caliper_embed_views_draw / caliper_embed_views_destroy_all.
// ===========================================================================
#include "caliper/embed.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include "embed_core.h"
#include "host_services.h"   // services_bridge
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"   // TensorBridge (views), colormap_lut, bridge_last_reject

using namespace caliper_host;

// One CaliperCore per process in v0: the slot that caliper_core_create and
// caliper_core_create_hosted both claim (embed_core.cpp explains the atomic).
std::atomic<CaliperCore*>& caliper_core_slot() {
    static std::atomic<CaliperCore*> slot{nullptr};
    return slot;
}

namespace {

std::atomic<CaliperCore*>& g_live_core = caliper_core_slot();
CaliperCore* const kCoreReserved = caliper_core_slot_reserved();

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// hosted core (embed_core.h): the Caliper app's live page. The app owns the
// window, the renderer, the UI contexts and the stores, exactly as before it
// had a core; the core borrows the renderer for the view layer and nothing else.
// ---------------------------------------------------------------------------
CaliperCore* caliper_core_create_hosted(HostRenderer* renderer) {
    if (!renderer) return nullptr;
    CaliperCore* expected = nullptr;
    if (!g_live_core.compare_exchange_strong(expected, kCoreReserved,
                                             std::memory_order_acq_rel)) {
        std::fprintf(stderr, "[embed] create_hosted refused: a core is already "
                             "live (one per process in v0)\n");
        return nullptr;
    }
    auto* core = new CaliperCore();
    core->hosted = true;
    core->renderer.reset(renderer);   // borrowed: released, not deleted, at destroy
    core->canvas_ready = true;        // the host's window is the canvas
    core->canvas_mode = HostRenderer::CANVAS_WINDOW;
    core->last_frame_time = now_sec();
    g_live_core.store(core, std::memory_order_release);
    return core;
}

void caliper_core_hosted_frame(CaliperCore* core, float x, float y, float w, float h,
                               float scale) {
    if (!core || !core->hosted) return;
    // The placement rules read the canvas size in physical px and the scale;
    // the host's work area stands in for the canvas.
    core->scale = scale > 0.0f ? scale : 1.0f;
    core->area_x = x;
    core->area_y = y;
    core->canvas_w = std::max(1, (int)(w * core->scale));
    core->canvas_h = std::max(1, (int)(h * core->scale));
    caliper_embed_views_draw(core);
    caliper_live_views_frame(core);   // typed views: points/series/histogram
    caliper_live_frame(core);         // live bookkeeping (no chrome when hosted)
}

void caliper_core_destroy_hosted(CaliperCore* core) {
    if (!core || !core->hosted) return;
    // Shutdown's view discipline: the live attach first, then the typed views,
    // then the v1.2 views, all while the host keeps the renderer bound.
    caliper_live_shutdown(core);
    caliper_live_views_shutdown(core);
    caliper_embed_views_destroy_all(core);
    (void)core->renderer.release();   // the host's; it outlives this core
    CaliperCore* expected = core;
    g_live_core.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    delete core;
}


const char* caliper_core_last_error(CaliperCore* core) {
    return core ? core->last_error.c_str() : "";
}
// ===========================================================================
// v1.2 — built-in tensor views (embed.h). The core draws a host's tensor itself:
// texture through the process TensorBridge (the SAME bridge applets use, driven
// host-side so it can reach the host-only set_mapping), an ImGui window per view
// composited after the applet. Frame-thread only, like every bridge op.
//
// ZERO-COPY HONESTY. A METAL tensor goes through the v1.2 import path only
// (import_allocation + update_texture_from_alloc: the renderer's compute/blit
// reads the MTLBuffer in place at byte_offset). If that path refuses, the call
// fails — there is deliberately NO fallback that downloads the buffer and
// CPU-stages it, so info.imported/path never lie about what happened.
// ===========================================================================
using caliper_embed_views::TensorView;

namespace {

constexpr int64_t kMaxViewDim = 16384;   // Metal's 2-D texture limit

const char* view_device_name(int32_t d) {
    switch (d) {
        case CALIPER_DEV_CPU:   return "cpu";
        case CALIPER_DEV_METAL: return "metal";
        case CALIPER_DEV_CUDA:  return "cuda";
        default:                return "device?";
    }
}

const char* view_dtype_name(CaliperDType d) {
    switch (d) {
        case CALIPER_DT_F32:  return "f32";
        case CALIPER_DT_F16:  return "f16";
        case CALIPER_DT_BF16: return "bf16";
        case CALIPER_DT_I64:  return "i64";
        case CALIPER_DT_I32:  return "i32";
        case CALIPER_DT_U8:   return "u8";
    }
    return "unknown";
}

TensorView* find_view(CaliperCore* core, const char* name) {
    for (auto& v : core->views)
        if (v.name == name) return &v;
    return nullptr;
}

// Release a view's GPU resources through the bridge (no-op pieces when already
// zero). The bridge may be null only if no renderer is bound — then the
// renderer is gone and its tables went with it.
void release_view(TensorBridge* b, TensorView& v) {
    if (b) {
        if (v.tex)   b->release_texture(v.tex);
        if (v.alloc) b->release_allocation(v.alloc);
    }
    v.tex = 0;
    v.alloc = 0;
    v.alloc_buf = nullptr;
    v.alloc_bytes = 0;
}

// " (bridge: <reason>)" when the bridge logged a rejection during this call.
std::string bridge_reason() {
    const char* r = bridge_last_reject();
    return (r && r[0]) ? std::string(" (bridge: ") + r + ")" : std::string();
}

// First placement of a windowed view: image at 1 texel = 1 physical pixel,
// small tensors magnified by an integer factor up to a ~192 pt long side (the
// draw samples nearest when magnifying, so cells stay crisp), big ones capped
// to 90% of the canvas; windows tiled left-to-right, wrapping rows.
void place_view(CaliperCore* core, TensorView& v, ImVec2 img) {
    const float lw = (float)core->canvas_w / core->scale;
    const float lh = (float)core->canvas_h / core->scale;
    const ImGuiStyle& st = ImGui::GetStyle();
    const float chrome_w = st.WindowPadding.x * 2.0f;
    const float chrome_h = ImGui::GetFrameHeight() + st.WindowPadding.y * 2.0f +
                           ImGui::GetTextLineHeightWithSpacing();
    float s = 1.0f;
    const float longest = std::max(img.x, img.y);
    if (longest > 0.0f && longest < 192.0f) s = std::floor(192.0f / longest);
    const float max_w = std::max(64.0f, lw * 0.9f - chrome_w);
    const float max_h = std::max(64.0f, lh * 0.9f - chrome_h);
    s = std::min(s, std::min(max_w / img.x, max_h / img.y));
    const float ww = std::max(img.x * s + chrome_w, 240.0f);
    const float wh = img.y * s + chrome_h;

    const float gap = 8.0f;
    if (core->tile_x > gap && core->tile_x + ww > lw) {          // wrap the row
        core->tile_x = gap;
        core->tile_y += core->tile_row_h + gap;
        core->tile_row_h = 0.0f;
    }
    if (core->tile_y > gap && core->tile_y + wh > lh) {          // out of rows
        core->tile_x = gap;
        core->tile_y = gap;
        core->tile_row_h = 0.0f;
    }
    v.pos_x = core->area_x + core->tile_x;   // a hosted core tiles into the host's work area
    v.pos_y = core->area_y + core->tile_y;
    v.size_x = ww;
    v.size_y = wh;
    core->tile_x += ww + gap;
    core->tile_row_h = std::max(core->tile_row_h, wh);
    v.placed = true;
}

}  // namespace

void caliper_embed_views_destroy_all(CaliperCore* core) {
    if (core->views.empty()) return;
    TensorBridge* b = services_bridge();
    for (auto& v : core->views) release_view(b, v);
    core->views.clear();
}

// Draw every view into the current ImGui frame (between canvas_new_frame and
// canvas_render). The ImTextureID is the bridge's public CaliperTextureId: the
// bridge hands out renderer->tex_imtexture_id(tex), which on Metal is the
// id<MTLTexture> pointer imgui_impl_metal binds directly (tensor_bridge.h §5.4).
void caliper_embed_views_draw(CaliperCore* core) {
    if (core->views.empty()) return;
    const ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    const ImDrawCallback nearest = pio.DrawCallback_SetSamplerNearest;
    const ImDrawCallback linear  = pio.DrawCallback_SetSamplerLinear;

    for (auto& v : core->views) {
        if (!v.tex) continue;
        // Logical size at which one texel covers one PHYSICAL pixel.
        const ImVec2 img((float)v.w / core->scale, (float)v.h / core->scale);

        if (v.flags & CALIPER_VIEW_FLAG_RAW) {
            // Borderless, padless, backgroundless, pinned at logical (0,0): at
            // content_scale 1.0 texel (i,j) lands on canvas pixel (i,j), sampled
            // nearest — the byte-compare surface.
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(img, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 0.0f);
            const ImGuiWindowFlags f =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoBringToFrontOnFocus;
            const std::string id = "##caliper.view.raw." + v.name;
            if (ImGui::Begin(id.c_str(), nullptr, f)) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (nearest) dl->AddCallback(nearest, nullptr);
                ImGui::Image((ImTextureID)v.tex, img);
                if (linear) dl->AddCallback(linear, nullptr);
            }
            ImGui::End();
            ImGui::PopStyleVar(5);
            continue;
        }

        if (!v.placed) place_view(core, v, img);
        ImGui::SetNextWindowPos(ImVec2(v.pos_x, v.pos_y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(v.size_x, v.size_y), ImGuiCond_FirstUseEver);
        // Title = the view name; the ### id keeps it apart from applet windows.
        const std::string label = v.name + "###caliper.view." + v.name;
        if (ImGui::Begin(label.c_str(), nullptr,
                         ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                             (core->views_bare ? ImGuiWindowFlags_NoTitleBar : 0))) {
            ImGui::Text("%dx%d %s %s %s updates=%llu last=%.2fms", v.w, v.h,
                        v.mapped ? "f32" : "u8", view_device_name(v.device),
                        v.path.c_str(), (unsigned long long)v.updates,
                        v.last_update_ms);
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float s = std::min(avail.x / img.x, avail.y / img.y);
            if (s > 0.0f) {
                const ImVec2 sz(img.x * s, img.y * s);
                const float pad = (avail.x - sz.x) * 0.5f;      // center horizontally
                if (pad > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
                // Magnifying (>= 1 physical px per texel): nearest, so tensor
                // cells read as cells. Minifying: the default linear filter.
                const bool magnify = s >= 1.0f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (magnify && nearest) dl->AddCallback(nearest, nullptr);
                ImGui::Image((ImTextureID)v.tex, sz);
                if (magnify && linear) dl->AddCallback(linear, nullptr);
            }
        }
        ImGui::End();
    }
}

int caliper_core_view_tensor(CaliperCore* core, const CaliperViewDesc* desc) {
    if (!core) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (!desc || desc->struct_size < sizeof(CaliperViewDesc)) {
        core->fail("view_tensor: null/short CaliperViewDesc");
        return 0;
    }
    if (!desc->name || !desc->name[0]) {
        core->fail("view_tensor: name must be a non-empty string");
        return 0;
    }
    const std::string name = desc->name;
    auto vfail = [&](const std::string& why) {
        core->fail("view_tensor('" + name + "'): " + why);
        return 0;
    };
    if (!desc->tensor) return vfail("null tensor");
    if (!core->canvas_ready)
        return vfail("attach a canvas before showing tensors (the renderer's "
                     "device exists only after attach_canvas)");
    TensorBridge* bridge = services_bridge();
    if (!bridge || !core->renderer) return vfail("no renderer bound to the tensor bridge");

    // ---- validation (clear reasons up front; the bridge re-checks) ----------
    const CaliperTensor& src = *desc->tensor;
    bool mapped = false;
    if (src.dtype == CALIPER_DT_F32 && src.ndim == 2) {
        mapped = true;
    } else if (src.dtype == CALIPER_DT_U8 && src.ndim == 3) {
        if (src.shape[2] != 4)
            return vfail("a u8 image must be (H,W,4) RGBA; got C=" +
                         std::to_string(src.shape[2]) +
                         " (convert RGB/gray to RGBA before showing)");
    } else {
        return vfail(std::string("want a 2-D f32 (H,W) or a 3-D u8 (H,W,4) RGBA "
                                 "tensor; got ndim=") +
                     std::to_string(src.ndim) + " dtype=" + view_dtype_name(src.dtype));
    }
    const int64_t H = src.shape[0], W = src.shape[1];
    if (H < 1 || W < 1 || H > kMaxViewDim || W > kMaxViewDim)
        return vfail("dims must be in 1..16384; got " + std::to_string(H) + "x" +
                     std::to_string(W));
    {
        int64_t expect = 1;
        for (int i = src.ndim - 1; i >= 0; --i) {
            if (src.strides[i] != expect)
                return vfail("tensor must be row-major contiguous (strides in "
                             "elements); make it contiguous first");
            expect *= src.shape[i];
        }
    }
    const uint64_t extent_bytes =
        (uint64_t)H * (uint64_t)W * (mapped ? sizeof(float) : 4u);
    const bool metal = (src.device == CALIPER_DEV_METAL);
    if (src.device != CALIPER_DEV_CPU && !metal)
        return vfail(std::string("device must be CPU or METAL; got ") +
                     view_device_name(src.device));
    if (src.data == nullptr)
        return vfail(metal ? "null MTLBuffer (tensor->data)" : "null data");
    if (metal) {
        if (core->renderer->interop_device() != CALIPER_DEV_METAL)
            return vfail("METAL tensor, but the active renderer is not Metal");
        if (desc->byte_offset % 4 != 0)
            return vfail("METAL byte_offset must be a multiple of 4; got " +
                         std::to_string(desc->byte_offset));
        if (desc->byte_offset > UINT64_MAX - extent_bytes)
            return vfail("byte_offset overflows");
    }
    if (mapped) {
        if (colormap_lut(desc->colormap) == nullptr)
            return vfail("bad colormap " + std::to_string(desc->colormap) +
                         " (0 viridis, 1 magma, 2 rdbu)");
        if (!std::isfinite(desc->vmin) || !std::isfinite(desc->vmax))
            return vfail("vmin/vmax must be finite");
    }

    // What the bridge sees. METAL: data = the MTLBuffer; byte_offset rides the
    // import window (update_texture_from_alloc). CPU: bytes at data+byte_offset.
    CaliperTensor t = src;
    t.struct_size = sizeof(CaliperTensor);
    if (!metal) t.data = (uint8_t*)src.data + desc->byte_offset;
    const int32_t cm   = mapped ? desc->colormap : 0;
    const float   vmin = mapped ? desc->vmin : 0.0f;
    const float   vmax = mapped ? desc->vmax : 1.0f;

    TensorView* v = find_view(core, name.c_str());
    const bool need_create = !v || v->mapped != mapped || v->w != (int)W ||
                             v->h != (int)H;
    bridge_clear_last_reject();

    // ---- 1. METAL: import first, so the device path is proven before any
    // texture exists. Cached per view by buffer pointer (our strong retain keeps
    // that object alive, so an equal pointer IS the same buffer).
    CaliperAllocId alloc = 0;
    bool fresh_import = false;
    const uint64_t window = desc->byte_offset + extent_bytes;
    if (metal) {
        if (v && v->alloc && v->alloc_buf == src.data && v->alloc_bytes >= window) {
            alloc = v->alloc;
        } else {
            alloc = bridge->import_allocation(src.data, window,
                                              CALIPER_ALLOC_HANDLE_MTLBUFFER);
            if (!alloc)
                return vfail("MTLBuffer import refused — the buffer is shorter than "
                             "byte_offset + extent (" + std::to_string(window) +
                             " bytes), lives on another GPU, or the renderer "
                             "cannot import" + bridge_reason());
            fresh_import = true;
        }
    }
    auto drop_fresh_import = [&]() {
        if (fresh_import) bridge->release_allocation(alloc);
    };

    // ---- 2. texture: (re)create on a new view or a shape/dtype change — the new
    // one is built BEFORE the old is released, so a failed recreate keeps the old
    // view intact. A METAL create runs the bridge's direct device pass at offset
    // 0 (in bounds: window <= buffer length was just proven); the imported update
    // in step 3 overwrites it in queue order before any frame samples it. A CPU
    // create uploads the data itself (it IS this call's update).
    CaliperTextureId tex = need_create ? 0 : v->tex;
    bool cpu_uploaded = false;
    if (need_create) {
        tex = mapped ? bridge->texture_from_tensor_mapped(&t, cm, vmin, vmax, 0)
                     : bridge->texture_from_tensor(&t, 0);
        if (!tex) {
            drop_fresh_import();
            return vfail("texture creation failed" + bridge_reason());
        }
        cpu_uploaded = !metal;
    } else if (mapped && (v->colormap != cm || v->vmin != vmin || v->vmax != vmax)) {
        // Mapping change on a live view: re-pin, keep the texture.
        if (!bridge->set_mapping(tex, cm, vmin, vmax)) {
            drop_fresh_import();
            return vfail("mapping change refused" + bridge_reason());
        }
        v->colormap = cm; v->vmin = vmin; v->vmax = vmax;   // pinned now
    }

    // ---- 3. the update ------------------------------------------------------
    bool ok = true;
    bool imported = false;
    std::string path = "cpu-staged";
    if (metal) {
        ok = bridge->update_texture_from_alloc(tex, alloc, desc->byte_offset, &t);
        if (ok) {
            imported = true;
            const char* p = core->renderer->last_device_path();
            path = p ? p : "";
            if (desc->flags & CALIPER_VIEW_FLAG_WAIT_GPU)
                core->renderer->device_update_wait();
        }
    } else if (!cpu_uploaded) {
        ok = bridge->update_texture(tex, &t);
    }
    if (ok && !metal) path = bridge->last_upload_path();   // "cpu-staged", or the GPU map's path
    if (!ok) {
        if (need_create) bridge->release_texture(tex);
        drop_fresh_import();
        return vfail(metal
            ? "device update from the imported MTLBuffer was refused (METAL "
              "tensors never fall back to a host copy)" + bridge_reason()
            : "CPU-staged upload refused" + bridge_reason());
    }

    // ---- 4. commit -----------------------------------------------------------
    if (!v) {
        core->views.emplace_back();
        v = &core->views.back();
        v->name = name;
    } else if (need_create && v->tex && v->tex != tex) {
        bridge->release_texture(v->tex);
    }
    if (metal) {
        if (fresh_import) {
            if (v->alloc) bridge->release_allocation(v->alloc);   // stale import
            v->alloc = alloc;
            v->alloc_buf = src.data;
            v->alloc_bytes = window;
        }
    } else if (v->alloc) {
        // A CPU update: stop pinning the previous MTLBuffer.
        bridge->release_allocation(v->alloc);
        v->alloc = 0; v->alloc_buf = nullptr; v->alloc_bytes = 0;
    }
    if (need_create) ++v->creates;
    v->tex = tex;
    v->mapped = mapped;
    v->w = (int)W;
    v->h = (int)H;
    v->colormap = cm;
    v->vmin = vmin;
    v->vmax = vmax;
    v->flags = desc->flags;
    v->device = src.device;
    v->imported = imported;
    v->path = path;
    ++v->updates;
    v->last_update_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    core->last_error.clear();
    return 1;
}

int caliper_core_view_remove(CaliperCore* core, const char* name) {
    if (!core || !name) return 0;
    for (auto it = core->views.begin(); it != core->views.end(); ++it) {
        if (it->name == name) {
            release_view(services_bridge(), *it);
            core->views.erase(it);
            return 1;
        }
    }
    core->fail(std::string("view_remove: no view named '") + name + "'");
    return 0;
}

int caliper_core_view_count(CaliperCore* core) {
    return core ? (int)core->views.size() : 0;
}

int caliper_core_view_info(CaliperCore* core, const char* name, CaliperViewInfo* out) {
    if (!core) return 0;
    if (!name || !out || out->struct_size < sizeof(CaliperViewInfo)) {
        core->fail("view_info: null name or null/short CaliperViewInfo");
        return 0;
    }
    const TensorView* v = find_view(core, name);
    if (!v) {
        core->fail(std::string("view_info: no view named '") + name + "'");
        return 0;
    }
    out->path = v->path.c_str();
    out->width = v->w;
    out->height = v->h;
    out->device = v->device;
    out->updates = v->updates;
    out->last_update_ms = v->last_update_ms;
    out->imported = v->imported ? 1 : 0;
    out->creates = v->creates;
    return 1;
}

int caliper_core_view_readback(CaliperCore* core, const char* name, void* buf,
                               size_t buf_bytes) {
    if (!core) return 0;
    if (!name || !buf) {
        core->fail("view_readback: null name/buffer");
        return 0;
    }
    const TensorView* v = find_view(core, name);
    if (!v) {
        core->fail(std::string("view_readback: no view named '") + name + "'");
        return 0;
    }
    const size_t need = (size_t)v->w * (size_t)v->h * 4;
    if (buf_bytes < need) {
        core->fail("view_readback: buffer too small (need " + std::to_string(need) +
                   " bytes)");
        return 0;
    }
    // Same-queue readback: commit order retires every prior update first.
    std::vector<uint8_t> px = core->renderer->debug_readback_rgba8(v->tex, v->w, v->h);
    if (px.size() != need) {
        core->fail("view_readback: renderer readback failed");
        return 0;
    }
    std::memcpy(buf, px.data(), need);
    return 1;
}

int caliper_core_colormap_lut(int32_t colormap, uint32_t out256[256]) {
    const uint32_t* lut = colormap_lut(colormap);
    if (!lut || !out256) return 0;
    std::memcpy(out256, lut, 256 * sizeof(uint32_t));
    return 1;
}

