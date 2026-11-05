#!/usr/bin/env python3
"""The frame traces under the VRAM figures on the Caliper landing page (site/index.html).

    python3 scripts/site_frame_traces.py [ref]     # default ref: origin/main

One frame at 60 Hz on each backend, drawn as lanes: the threads that run code and the GPU queues they feed, with
arrows where one hands work or a signal to another. Every span names the source lines behind it, found by their
text in a published commit (git show ref:path), so the page shows the real line and links to it. The order is the
code's; the durations are schematic. Traces go between <!-- trace:NAME --> and <!-- /trace:NAME --> markers; the
page's script plays them and lights the matching rows in the code panels.
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/AhmedKhan-GH/caliper"
PAGE = Path(__file__).resolve().parents[1] / "site/index.html"

GS = "applets/gpt_scope/gpt_scope.cpp"
TORCH = "sdk/include/caliper/adapters/torch.hpp"
MR = "src/host/renderer/metal_renderer.mm"
VR = "src/host/renderer/vulkan_renderer.cpp"
COMP = "src/host/renderer/shaders/colormap.comp"

# A ref is (path, text) or (path, text, anchor): the line holding text, the first one after the anchor's line when
# an anchor is given. Without an anchor the text must occur once in the file.
STEP = "// ---- the optimizer step ----"
FORWARD = (GS, "auto logits = model->forward(xb);", STEP)
BACKWARD = (GS, "loss.backward();", STEP)
GRAD_NORMS = (GS, "auto gn = acc.sqrt().to(torch::kCPU).contiguous();")
OPT_STEP = (GS, "opt.step();", STEP)
LOSS = (GS, "const float l = loss.item<float>();")
HANDOFF = (GS, "return caliper::adapters::stream_to_tensor(dev_t, bridge.caps());")
UPDATE = (GS, "if (id != 0) { bridge.update_texture(id, &*ct); return id; }")
WIDGETS = (GS, 'ImGui::Text("selected: layer %d, head %d')
IMAGE = (GS, "ImGui::Image(caliper::Bridge::imtex(st->head_tex), ImVec2(side, side));")

# The training thread is the same code on both backends; only how long torch takes to encode or launch differs.
def training(fw, bw, gn, st, lo, fw2, bw2):
    return [
        (0, 0.0, fw, "torch", "forward", "forward: torch queues the kernels", [FORWARD]),
        (0, fw, bw, "torch", "backward", "backward: torch queues the kernels", [BACKWARD]),
        (0, bw, gn, "wait", "waits: grad norms", "blocks until the GPU is done: grad norms copied to the CPU", [GRAD_NORMS]),
        (0, gn, st, "torch", "step", "optimizer step: torch queues the kernels", [OPT_STEP]),
        (0, st, lo, "wait", "waits: loss", "blocks until the GPU is done: the loss read back", [LOSS]),
        (0, lo, fw2, "torch", "forward", "next step: forward", [FORWARD]),
        (0, fw2, bw2, "torch", "backward", "next step: backward", [BACKWARD]),
        (0, bw2, 20.0, "wait", "waits: grad norms", "next step: blocks until the GPU is done", [GRAD_NORMS]),
    ]


VSYNC = 1000 / 60

MR_NEW_FRAME = [(MR, "drawable_ = [layer_ nextDrawable];"), (MR, "frame_cmd_ = [queue_ commandBuffer];")]
MR_CCF = "bool colormap_compute_from("
MR_FRAME = "void render(int /*fb_w*/, int /*fb_h*/) override {"   # the window's render, not the canvas's
MR_RENDER = (MR, "ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),", MR_FRAME)
MR_PRESENT = (MR, "[frame_cmd_ presentDrawable:drawable_];")
MR_SIGNAL = (MR, "[sig encodeSignalEvent:ev value:v];")
MR_WAIT = (MR, "[cb encodeWaitForEvent:ev value:v];")

METAL = {
    "name": "metal",
    "scope": ["call", "metal"],
    "start": 3.45,
    # Stretched spans of the time axis, (from, to, factor): the handoff, where the calls are a fraction of a ms.
    "bands": [(2.75, 3.95, 8)],
    "label": "Metal, one frame: the frame thread hands torch's tensor to Caliper's command queue, which waits on a "
             "shared event torch's queue signals, colormaps the buffer in place, and draws the frame that the display "
             "shows at the next vsync; the training thread blocks only on its own reads",
    "lanes": [("TRAINING THREAD", "CPU · gpt_scope train_job"), ("FRAME THREAD", "CPU · Caliper host"),
              ("TORCH'S QUEUE", "GPU · MTLCommandQueue"), ("CALIPER'S QUEUE", "GPU · MTLCommandQueue"),
              ("DISPLAY", "scan-out")],
    "segs": training(2.2, 4.6, 8.9, 9.9, 12.1, 14.3, 16.7) + [
        (1, 0.8, 1.3, "caliper", "new_frame", "new_frame: the next drawable and the frame's command buffer", MR_NEW_FRAME),
        (1, 1.3, 2.8, "caliper", "applet UI", "the applet's draw_ui builds its widgets", [WIDGETS]),
        (1, 2.8, 3.0, "caliper", "tensor", "to_tensor: the MPS storage's MTLBuffer becomes the descriptor's data",
         [(TORCH, "out.data = t.storage().mutable_data();"), (TORCH, "out.device = CALIPER_DEV_METAL;"), HANDOFF]),
        (1, 3.0, 3.2, "caliper", "commit", "stream_to_tensor: commit torch's pending kernels and take its queue",
         [(TORCH, "void* queue = detail::mps_commit_and_get_queue();"), (TORCH, "torch::mps::commit();"),
          (TORCH, "out->stream = queue;"), HANDOFF]),
        (1, 3.2, 3.35, "caliper", "cast", "update_texture: the descriptor's pointer is cast to id<MTLBuffer>",
         [(MR, "id<MTLBuffer> src = (__bridge id<MTLBuffer>)t.data;", "bool tex_update_from_device("), UPDATE,
          (MR, "return colormap_compute_from(tex, dst, src, 0, t, lut256, vmin, vmax,")]),
        (1, 3.35, 3.6, "caliper", "order", "order_after_producer: signal v queued on torch's queue, a wait for v encoded",
         [MR_SIGNAL, (MR, "[sig commit];", "void order_after_producer("), MR_WAIT,
          (MR, "id<MTLCommandBuffer> sig = [producer commandBuffer];"),
          (MR, "if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);", MR_CCF)]),
        (1, 3.6, 3.9, "caliper", "encode", "the colormap pass encoded and committed, with no CPU wait",
         [(MR, "[enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];", MR_CCF),
          (MR, "[enc setBuffer:src offset:(NSUInteger)src_offset atIndex:0];", MR_CCF),
          (MR, "[enc setTexture:dst atIndex:0];", MR_CCF), (MR, "[cb commit];", MR_CCF)]),
        (1, 3.9, 5.6, "caliper", "applet UI", "the applet draws the texture with ImGui::Image", [IMAGE]),
        (1, 5.6, 6.4, "caliper", "render", "render: ImGui's draw encoded, the drawable presented, the frame committed",
         [MR_RENDER, MR_PRESENT, (MR, "[frame_cmd_ commit];", MR_FRAME)]),
        (1, 17.5, 18.0, "caliper", "new_frame", "next frame: new_frame", MR_NEW_FRAME),
        (1, 18.0, 20.0, "caliper", "applet UI", "next frame: the applet's draw_ui", [WIDGETS]),
        (2, 0.4, 5.0, "torch", "torch's kernels", "torch's kernels committed before the handoff: forward, start of backward", []),
        (2, 5.0, 5.15, "caliper", "", "signal v: runs once everything committed before it has finished", [MR_SIGNAL]),
        (2, 5.15, 8.85, "torch", "torch's kernels", "torch's kernels: rest of backward, grad norms", []),
        (2, 9.2, 11.9, "torch", "torch's kernels", "torch's kernels: optimizer step", []),
        (2, 12.4, 20.0, "torch", "torch's kernels", "torch's kernels: next step", []),
        (3, 3.9, 5.15, "wait", "waits for v", "the colormap command buffer waits for event v, on the GPU", [MR_WAIT]),
        (3, 5.15, 5.75, "caliper", "cmap_f32", "cmap_f32 reads the tensor's MTLBuffer and writes the texture",
         [(MR, "float v = src[gid.y * p.sy + gid.x * p.sx];"), (MR, "dst.write(c, gid);"),
          (MR, "kernel void cmap_f32(device const float*  src  [[buffer(0)]],")]),
        (3, 6.4, 7.5, "caliper", "ImGui draw", "ImGui's draw samples the texture into the drawable", [MR_RENDER]),
        (4, 0.0, VSYNC, "screen", "previous frame", "the previous frame on screen", []),
        (4, VSYNC, 20.0, "screen", "this frame", "this frame on screen from the vsync after its present", [MR_PRESENT]),
    ],
    # (from lane, time, to lane, time, label)
    "arrows": [
        (1, 3.1, 2, 3.1, ""),
        (1, 3.5, 2, 5.0, ""),
        (1, 3.88, 3, 3.9, ""),
        (2, 5.15, 3, 5.15, ""),
        (1, 6.4, 3, 6.4, ""),
        (3, 7.5, 4, VSYNC, "present"),
        (2, 8.85, 0, 8.9, ""),
        (2, 11.9, 0, 12.1, ""),
    ],
}

VR_PIPE = "bool update_pipelined("
VR_COPY = (VR, "cu->cuMemcpyDtoDAsync(io.cuda_ptr, src, (size_t)bytes, stream)", VR_PIPE)
VR_SIGNAL = (VR, "if (cu->cuSignalExternalSemaphoresAsync(&io.cuda_sem, &sp, 1, stream)", VR_PIPE)
VR_WAIT = (VR, "tsi.pWaitSemaphoreValues = &wait_val;", VR_PIPE)
VR_NEW_FRAME = [(VR, "ImGui_ImplVulkan_NewFrame();", "void new_frame() override {")]
VR_DRAW = (VR, "ImGui_ImplVulkan_RenderDrawData(dd, fd->CommandBuffer);")
VR_PRESENT = (VR, "VkResult err = vkQueuePresentKHR(queue_, &pi);")

NVIDIA = {
    "name": "nvidia",
    "scope": ["call", "nvidia"],
    "start": 3.45,
    "bands": [(2.75, 4.0, 13), (5.35, 6.65, 3)],
    "label": "NVIDIA, one frame: the frame thread enqueues an in-VRAM copy and a timeline signal on torch's CUDA "
             "stream, submits a colormap pass that waits for that value on Caliper's Vulkan queue, and submits the "
             "frame, which waits for the pass and is presented at the next vsync",
    "lanes": [("TRAINING THREAD", "CPU · gpt_scope train_job"), ("FRAME THREAD", "CPU · Caliper host"),
              ("TORCH'S STREAM", "GPU · CUDA stream"), ("CALIPER'S QUEUE", "GPU · VkQueue"),
              ("DISPLAY", "swapchain")],
    "segs": training(1.8, 3.9, 8.3, 9.0, 10.6, 12.4, 14.5) + [
        (1, 0.8, 1.1, "caliper", "new_frame", "new_frame: ImGui's Vulkan frame begins", VR_NEW_FRAME),
        (1, 1.1, 2.8, "caliper", "applet UI", "the applet's draw_ui builds its widgets", [WIDGETS]),
        (1, 2.78, 2.96, "caliper", "tensor", "to_tensor: the CUDA tensor's device address becomes the descriptor's data",
         [(TORCH, "out.data = t.data_ptr();", "if (t.is_cuda()) {"), (TORCH, "out.device = CALIPER_DEV_CUDA;"), HANDOFF]),
        (1, 2.96, 3.1, "caliper", "stream", "stream_to_tensor: torch's current CUDA stream goes in the descriptor",
         [(TORCH, "out->stream = (void*)at::cuda::getCurrentCUDAStream("), HANDOFF]),
        (1, 3.1, 3.25, "caliper", "retire", "update_texture: retire this texture's last update, a host wait only if it is still running",
         [(VR, 'if (!retire(dst)) return dev_bail("pipelined: retire timed out");', VR_PIPE), UPDATE]),
        (1, 3.25, 3.4, "caliper", "copy", "the in-VRAM copy enqueued on torch's stream", [VR_COPY]),
        (1, 3.4, 3.55, "caliper", "signal", "the semaphore signal, base+1, enqueued on torch's stream",
         [VR_SIGNAL, (VR, "sp.params.fence.value = base + 1;", VR_PIPE), (VR, "cudadrv::ExternalSemaphoreSignalParams sp{};", VR_PIPE)]),
        (1, 3.55, 3.75, "caliper", "record", "the colormap pass recorded into this texture's command buffer",
         [(VR, "record_cmap_body(io.cb, dst, make_push(dst, t, vmin, vmax), io.set);", VR_PIPE)]),
        (1, 3.75, 3.95, "caliper", "submit", "vkQueueSubmit: wait for base+1, signal base+2, no fence",
         [(VR, "if (vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)", VR_PIPE), VR_WAIT,
          (VR, "tsi.pSignalSemaphoreValues = &signal_val;", VR_PIPE), (VR, "uint64_t wait_val = base + 1, signal_val = base + 2;", VR_PIPE),
          (VR, "si.pWaitSemaphores = &io.timeline;", VR_PIPE), (VR, "si.pSignalSemaphores = &io.timeline;", VR_PIPE),
          (VR, "tsi.waitSemaphoreValueCount = 1;", VR_PIPE), (VR, "tsi.signalSemaphoreValueCount = 1;", VR_PIPE),
          (VR, "VkTimelineSemaphoreSubmitInfoKHR tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};", VR_PIPE),
          (VR, "VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};", VR_PIPE), (VR, "si.pNext = &tsi;", VR_PIPE),
          (VR, "si.waitSemaphoreCount = 1;", VR_PIPE), (VR, "si.pWaitDstStageMask = &wait_stage;", VR_PIPE),
          (VR, "si.commandBufferCount = 1;", VR_PIPE), (VR, "si.pCommandBuffers = &io.cb;", VR_PIPE),
          (VR, "si.signalSemaphoreCount = 1;", VR_PIPE)]),
        (1, 3.95, 5.4, "caliper", "applet UI", "the applet draws the texture with ImGui::Image", [IMAGE]),
        (1, 5.4, 5.65, "caliper", "", "frame_render: acquire a swapchain image, wait for its frame's fence",
         [(VR, "VkResult err = vkAcquireNextImageKHR(device_, wd_.Swapchain, UINT64_MAX,"),
          (VR, "vkWaitForFences(device_, 1, &fd->Fence, VK_TRUE, UINT64_MAX);", "void frame_render(")]),
        (1, 5.65, 6.25, "caliper", "record", "ImGui's draw recorded into the frame's command buffer", [VR_DRAW]),
        (1, 6.25, 6.45, "caliper", "", "vkQueueSubmit: wait for the image and for base+2",
         [(VR, "if (vkQueueSubmit(queue_, 1, &si, fd->Fence) == VK_SUCCESS)"),
          (VR, "for (const auto& pw : pending_frame_waits_) {", "void frame_render(")]),
        (1, 6.45, 6.6, "caliper", "", "vkQueuePresentKHR: queued behind render-complete", [VR_PRESENT]),
        (1, 17.5, 17.8, "caliper", "new_frame", "next frame: new_frame", VR_NEW_FRAME),
        (1, 17.8, 20.0, "caliper", "applet UI", "next frame: the applet's draw_ui", [WIDGETS]),
        (2, 0.3, 4.6, "torch", "torch's kernels", "torch's kernels launched before the handoff: forward, start of backward", []),
        (2, 4.6, 4.95, "caliper", "copy", "cuMemcpyDtoDAsync: the tensor into the shared buffer, inside VRAM", [VR_COPY]),
        (2, 4.95, 5.05, "caliper", "", "the timeline semaphore reaches base+1", [VR_SIGNAL]),
        (2, 5.05, 8.2, "torch", "torch's kernels", "torch's kernels: rest of backward, grad norms", []),
        (2, 8.5, 10.4, "torch", "torch's kernels", "torch's kernels: optimizer step", []),
        (2, 10.9, 20.0, "torch", "torch's kernels", "torch's kernels: next step", []),
        (3, 3.95, 5.05, "wait", "waits base+1", "the colormap submit waits for base+1, on the GPU", [VR_WAIT]),
        (3, 5.05, 5.6, "caliper", "colormap", "colormap.comp reads the shared buffer, writes the VkImage, signals base+2",
         [(COMP, "float v = src[gid.y * p.sy + gid.x * p.sx];"), (COMP, "imageStore(dst, ivec2(gid), c);")]),
        (3, 6.45, 7.6, "caliper", "ImGui draw", "ImGui's draw samples the VkImage into the swapchain image", [VR_DRAW]),
        (4, 0.0, VSYNC, "screen", "previous frame", "the previous frame on screen", []),
        (4, VSYNC, 20.0, "screen", "this frame", "this frame on screen from the vsync after its present", [VR_PRESENT]),
    ],
    "arrows": [
        (1, 3.33, 2, 4.6, ""),
        (1, 3.48, 2, 4.95, ""),
        (1, 3.93, 3, 3.95, ""),
        (2, 5.05, 3, 5.05, ""),
        (1, 6.4, 3, 6.45, ""),
        (3, 7.6, 4, VSYNC, "present"),
        (2, 8.2, 0, 8.3, ""),
        (2, 10.4, 0, 10.6, ""),
    ],
}

# Geometry, in the SVG's own units. The time axis is piecewise linear: inside a band, a ms is drawn `factor` times
# wider, and playback moves at a steady speed across the drawing, so the handoff plays slower too.
W, T = 1100, 20.0
X0, X1 = 212, 1084
Y0, PITCH, TRACK = 44, 76, 24


def axis(bands):
    def units(t):
        return t + sum((k - 1) * max(0.0, min(t, b) - a) for a, b, k in bands)
    total = units(T)
    return lambda t: round(X0 + (X1 - X0) * units(t) / total, 2), total


_files = {}


def source(ref, path):
    if path not in _files:
        _files[path] = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True,
                                      check=True).stdout.split("\n")
    return _files[path]


def resolve(ref, sha, r):
    path, needle, *anchor = r
    lines = source(ref, path)
    start = 0
    if anchor:
        start = next(i for i, l in enumerate(lines) if anchor[0] in l)
    hits = [i for i, l in enumerate(lines) if i >= start and needle in l]
    if not hits or (not anchor and len(hits) > 1):
        sys.exit(f"{path}: {needle!r} found {len(hits)} times")
    i = hits[0]
    return {"f": path.split("/")[-1], "p": path, "ln": i + 1, "code": lines[i].strip()}


def svg(tr, refs):
    n = tr["name"]
    lanes = tr["lanes"]
    x, _ = axis(tr["bands"])
    H = Y0 + len(lanes) * PITCH - 8
    out = [f'<svg class="trace-svg" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(tr["label"])}">',
           f'<defs><pattern id="wait-{n}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           f'<line x1="0" y1="0" x2="0" y2="6" class="hatch"/></pattern>'
           f'<marker id="arrow-{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
           f'<path d="M0,1 L10,5 L0,9 Z" class="tip"/></marker></defs>']
    # The stretched bands, the time axis and the two vsyncs.
    edges = [x(e) for a, b, _ in tr["bands"] for e in (a, b)]
    ticks = [float(ms) for ms in range(0, int(T) + 1, 2)
             if not any(a < ms < b for a, b, _ in tr["bands"]) and all(abs(x(ms) - e) > 24 for e in edges)]
    for a, b, k in tr["bands"]:
        out.append(f'<rect x="{x(a)}" y="{Y0 - 12}" width="{round(x(b) - x(a), 2)}" height="{H - Y0 + 12}" class="band"/>')
        out.append(f'<text x="{x(a) + 5}" y="{Y0 - 30}" class="band-l">×{k}</text>')
        ticks += [m / 2 for m in range(int(a * 2) + 1, int(b * 2) + 1) if a < m / 2 < b]
    for ms in sorted(ticks):
        out.append(f'<line x1="{x(ms)}" y1="{Y0 - 12}" x2="{x(ms)}" y2="{H}" class="grid"/>')
        out.append(f'<text x="{x(ms)}" y="{Y0 - 18}" class="tick" text-anchor="middle">{ms:g}{" ms" if ms == T else ""}</text>')
    for v in (0, VSYNC):
        out.append(f'<line x1="{x(v)}" y1="{Y0 - 30}" x2="{x(v)}" y2="{H}" class="vsync"/>')
    out.append(f'<text x="{x(VSYNC) + 6}" y="{Y0 - 30}" class="vsync-l">VSYNC</text>')
    for i, (name, sub) in enumerate(lanes):
        y = Y0 + i * PITCH
        out.append(f'<text x="0" y="{y + 10}" class="lane-name">{html.escape(name)}</text>')
        out.append(f'<text x="0" y="{y + 27}" class="lane-sub">{html.escape(sub)}</text>')
        out.append(f'<rect x="{X0}" y="{y}" width="{X1 - X0}" height="{TRACK}" rx="3" class="track"/>')
    # Arrows under the spans' text rows but over the tracks.
    for j, (la, ta, lb, tb, label) in enumerate(tr["arrows"]):
        ya = Y0 + la * PITCH + (TRACK - 2 if lb > la else 2)
        yb = Y0 + lb * PITCH + (2 if lb > la else TRACK - 2)
        g = [f'<g class="flow" data-j="{j}"><path d="M {x(ta)} {ya} L {x(tb)} {yb}" marker-end="url(#arrow-{n})"/>']
        if label:
            mx, my = (x(ta) + x(tb)) / 2, (ya + yb) / 2
            g.append(f'<text x="{mx + 7}" y="{my + 4}" class="al">{html.escape(label)}</text>')
        out.append("".join(g) + "</g>")
    for k, (lane, a, b, kind, short, label, rs) in enumerate(tr["segs"]):
        y = Y0 + lane * PITCH
        xa, xb = x(a), x(b)
        w = max(xb - xa, 3)
        tip = label + "".join(f'\n{r["f"]}:{r["ln"]}  {r["code"]}' for r in refs[k][:2])
        g = [f'<g class="seg k-{kind}" data-i="{k}"><title>{html.escape(tip)}</title>',
             f'<rect x="{xa}" y="{y + 3}" width="{round(w, 2)}" height="{TRACK - 6}" rx="2.5"'
             + (f' fill="url(#wait-{n})"' if kind == "wait" else "") + "/>"]
        if short and w > len(short) * 5.2 + 10:
            g.append(f'<text x="{xa + 6}" y="{y + 16}" class="seglab">{html.escape(short)}</text>')
        out.append("".join(g) + "</g>")
    # Under each track: what that lane is doing now and the line it is on, filled in by the page's script.
    for i in range(len(lanes)):
        y = Y0 + i * PITCH
        out.append(f'<text x="{X0}" y="{y + TRACK + 18}" class="now-l" data-lane="{i}"><tspan class="lbl"></tspan>'
                   f'<tspan class="where" dx="12"></tspan></text>')
        out.append(f'<a class="now-a" data-lane="{i}" target="_blank" rel="noopener"><text x="{X0}" y="{y + TRACK + 36}" '
                   f'class="now-c"></text></a>')
    out.append(f'<g class="playhead"><line x1="0" y1="{Y0 - 30}" x2="0" y2="{H}"/>'
               f'<path d="M -5 {Y0 - 32} L 5 {Y0 - 32} L 0 {Y0 - 25} Z"/></g>')
    out.append("</svg>")
    return "\n      ".join(out)


PLAY = ('<svg class="i-play" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5 L10.5 6 L3 10.5 Z"/></svg>'
        '<svg class="i-pause" viewBox="0 0 12 12" aria-hidden="true"><rect x="2.5" y="1.5" width="2.6" height="9"/>'
        '<rect x="6.9" y="1.5" width="2.6" height="9"/></svg>')


def block(tr, ref, sha):
    refs = [[resolve(ref, sha, r) for r in s[6]] for s in tr["segs"]]
    _, total = axis(tr["bands"])
    data = {
        "name": tr["name"], "base": f"{REPO}/blob/{sha}/", "T": T, "x0": X0, "x1": X1, "W": W, "start": tr["start"], "scope": tr["scope"],
        "bands": tr["bands"],
        "lanes": len(tr["lanes"]),
        "segs": [{"l": s[0], "a": s[1], "b": s[2], "label": s[5], "refs": rs} for s, rs in zip(tr["segs"], refs)],
        "arrows": [a[1] for a in tr["arrows"]],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (f'<div class="trace trace-plate" data-trace="{tr["name"]}">\n'
            f'      <div class="trace-bar"><span class="k">Frame trace · 60 Hz</span>'
            f'<button class="tb-play" type="button" aria-label="Play or pause">{PLAY}</button>'
            f'<input type="range" min="0" max="{round(total * 1000)}" step="10" aria-label="Position in the frame">'
            f'<output>0.00 ms</output>'
            f'<label class="tb-speed">Speed <select><option value="120">fast</option>'
            f'<option value="400" selected>normal</option><option value="1400">slow</option></select></label>'
            f'<span class="tb-note">Order from the code · durations schematic</span></div>\n'
            f'      <div class="trace-scroll">{svg(tr, refs)}</div>\n'
            f'      <script type="application/json" class="trace-data">{blob}</script>\n'
            f'    </div>')


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    sha = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()
    page = PAGE.read_text()
    for tr in (METAL, NVIDIA):
        name = tr["name"]
        pattern = re.compile(rf"(<!-- trace:{name} -->).*?(<!-- /trace:{name} -->)", re.S)
        if not pattern.search(page):
            sys.exit(f"no <!-- trace:{name} --> markers in {PAGE}")
        b = block(tr, ref, sha)
        page = pattern.sub(lambda m: f"{m.group(1)}\n    {b}\n    {m.group(2)}", page)
    PAGE.write_text(page)


if __name__ == "__main__":
    main()
