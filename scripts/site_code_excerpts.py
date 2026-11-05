#!/usr/bin/env python3
"""The code beside the VRAM figures on the Caliper landing page (site/index.html).

    python3 scripts/site_code_excerpts.py [ref]     # default ref: origin/main

Each panel is exact source lines read from a published commit (git show ref:path), so the page shows the real code
and links to those lines on GitHub. Panels go between <!-- code:NAME --> and <!-- /code:NAME --> markers in the
page. Re-run after the excerpted code changes on the published branch, and check the ranges still cover it.
"""
import html
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/AhmedKhan-GH/caliper"
PAGE = Path(__file__).resolve().parents[1] / "site/index.html"

# name -> panels: (title, path, [(first, last) line ranges], identifiers drawn in the accent)
PANELS = {
    "call": [
        ("The descriptor", "sdk/include/caliper/tensor.h", [(17, 27)], ["CaliperTensor", "data", "device", "stream"]),
        ("The call, from an applet", "applets/gpt_scope/gpt_scope.cpp", [(451, 451), (455, 456), (1388, 1388)],
         ["stream_to_tensor", "texture_from_tensor_mapped", "update_texture", "ImGui::Image"]),
    ],
    "metal": [
        ("Tensor to descriptor", "sdk/include/caliper/adapters/torch.hpp", [(148, 156)],
         ["storage_offset", "mutable_data", "CALIPER_DEV_METAL"]),
        ("The buffer, not a copy", "src/host/renderer/metal_renderer.mm", [(806, 821)],
         ["__bridge id<MTLBuffer>", "colormap_compute_from", "blit_u8_from"]),
        ("One pass, after torch's queue", "src/host/renderer/metal_renderer.mm", [(1655, 1660), (1692, 1704)],
         ["encodeSignalEvent", "encodeWaitForEvent", "setBuffer:src", "setTexture:dst", "dispatchThreadgroups"]),
        ("The kernel", "src/host/renderer/metal_renderer.mm", [(66, 84)], ["cmap_f32", "lut", "dst.write"]),
    ],
    "nvidia": [
        ("Tensor to descriptor", "sdk/include/caliper/adapters/torch.hpp", [(157, 166)],
         ["data_ptr", "CALIPER_DEV_CUDA"]),
        ("Vulkan allocates the image, exportable", "src/host/renderer/vulkan_renderer.cpp", [(638, 640), (645, 651)],
         ["VkExternalMemoryImageCreateInfo", "VkMemoryDedicatedAllocateInfo", "VkExportMemoryAllocateInfo",
          "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT"]),
        ("CUDA imports it as a surface", "src/host/renderer/vulkan_renderer.cpp", [(3723, 3728), (3730, 3734), (3745, 3749)],
         ["vkGetMemoryWin32HandleKHR", "cuImportExternalMemory", "cuExternalMemoryGetMappedMipmappedArray",
          "cuSurfObjectCreate"]),
        ("Each update", "src/host/renderer/vulkan_renderer.cpp", [(3827, 3843), (3853, 3853)],
         ["cuLaunchKernel", "cuSignalExternalSemaphoresAsync", "swap_images"]),
        ("The kernel", "src/host/renderer/kernels/cmap_direct.cu", [(21, 44)],
         ["caliper_cmap_f32", "lut", "surf2Dwrite"]),
    ],
}


def highlight(code, names):
    """HTML for one source line: comments (// and /* */) faint, the named identifiers, whole words only and not
    inside strings, in the accent."""
    accent = "|".join(re.escape(html.escape(n)) for n in sorted(names, key=len, reverse=True))
    out = []
    for i, part in enumerate(re.split(r'(//.*$|/\*.*?\*/|"[^"]*")', code)):
        if i % 2:
            out.append(html.escape(part) if part.startswith('"') else f'<span class="c">{html.escape(part)}</span>')
        else:
            text = html.escape(part)
            if accent:
                text = re.sub(rf"(?<![\w])({accent})(?![\w])", r'<span class="p">\1</span>', text)
            out.append(text)
    return "".join(out)


def panel(ref, sha, title, path, ranges, names):
    lines = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout.split("\n")
    rows = []
    for a, b in ranges:
        chunk = lines[a - 1:b]
        # Each chunk is dedented on its own: excerpts from different depths read flush left.
        indent = min(len(l) - len(l.lstrip()) for l in chunk if l.strip())
        if rows:
            rows.append('<span class="gap">⋯</span>')
        for i, l in enumerate(chunk):
            # Each line carries its file and number, so the page's traces can light up the lines that run.
            rows.append(f'<span class="row" data-file="{path.split("/")[-1]}" data-ln="{a + i}">'
                        f'<span class="ln">{a + i}</span>{highlight(l[indent:], names)}</span>')
    # The label lists every range; the link opens the first.
    where = ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in ranges)
    link = f"{REPO}/blob/{sha}/{path}#L{ranges[0][0]}-L{ranges[0][1]}"
    # Closed by default: the head names the step and links the lines; opening shows them.
    return (f'<details class="code-panel"><summary class="code-head"><span class="code-title">{html.escape(title)}</span>'
            f'<a class="code-file" href="{link}">{path.split("/")[-1]} · {where}</a></summary>'
            f'<pre class="code"><code>' + "".join(rows) + "</code></pre></details>")


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    sha = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()
    page = PAGE.read_text()
    for name, panels in PANELS.items():
        block = "\n".join(panel(ref, sha, *p) for p in panels)
        pattern = re.compile(rf"(<!-- code:{name} -->).*?(<!-- /code:{name} -->)", re.S)
        if not pattern.search(page):
            sys.exit(f"no <!-- code:{name} --> markers in {PAGE}")
        page = pattern.sub(lambda m: f"{m.group(1)}\n<div class=\"code-grid code-{name}\">{block}</div>\n{m.group(2)}", page)
    PAGE.write_text(page)


if __name__ == "__main__":
    main()
