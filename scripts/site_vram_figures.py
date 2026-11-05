#!/usr/bin/env python3
"""The two VRAM figures on the Caliper landing page (site/index.html), as inline SVG.

    python3 scripts/site_vram_figures.py apple|nvidia     # print one figure
    python3 scripts/site_vram_figures.py --write site/index.html   # replace both figures in the page

Where a visualised tensor's bytes physically live, which silicon touches them, and how the memory is laid out,
for Apple Silicon (Metal) and an NVIDIA GPU (Vulkan + CUDA). Facts from ZEROCOPY.md and
docs/vulkan-cuda-backend.md; the layouts are schematic, not die maps. Styled by the page's .hw rules.
"""
import sys
from pathlib import Path

W = 1100
out = []


def rect(x, y, w, h, cls, r=3):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>')


def text(x, y, s, cls, anchor="start"):
    out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')


def line(x1, y1, x2, y2, cls):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>')


def cores(x, y, cols, rows, size, pitch, cls):
    for r in range(rows):
        for c in range(cols):
            rect(x + c * pitch, y + r * pitch, size, size, cls, 1.5)


def bus(x1, x2, y, n, gap, cls="bus"):
    for i in range(n):
        line(x1, y + i * gap, x2, y + i * gap, cls)


def vbus(x, y1, y2, n, gap, cls="bus"):
    for i in range(n):
        line(x + i * gap, y1, x + i * gap, y2, cls)


# The page map: physical memory as a grid of pages, regions coloured by what they hold, each region annotated
# with who writes and who reads it.
COLS, PAGE_W, PAGE_H, PITCH_X, PITCH_Y = 8, 15, 13, 18, 16


def page_map(x, y, rows, regions, note_x):
    """regions: (first page, count, css class, title, [lines]) in page order."""
    owner = {}
    for first, count, cls, *_ in regions:
        for i in range(first, first + count):
            owner[i] = cls
    for i in range(COLS * rows):
        r, c = divmod(i, COLS)
        rect(x + c * PITCH_X, y + r * PITCH_Y, PAGE_W, PAGE_H, "page " + owner.get(i, ""), 1.5)
    for first, count, cls, title, lines in regions:
        top = y + (first // COLS) * PITCH_Y
        bottom = y + ((first + count - 1) // COLS) * PITCH_Y + PAGE_H
        mid = (top + bottom) / 2
        # A bracket from the region to its note.
        bx = x + COLS * PITCH_X + 6
        out.append(f'<path d="M {bx} {top + 2} L {bx + 6} {top + 2} L {bx + 6} {bottom - 2} L {bx} {bottom - 2}" class="bracket"/>')
        line(bx + 6, mid, note_x - 8, mid, "leader")
        ty = mid - 8 * len(lines) + 2
        text(note_x, ty, title, "t")
        for j, l in enumerate(lines):
            text(note_x, ty + 17 + j * 16, l, "n")


def stats(y, items, x0=16, width=W - 32):
    """A row of figures: label above, value below, ruled between."""
    w = width / len(items)
    line(x0, y - 18, x0 + width, y - 18, "rulefaint")
    for i, (label, value, cls) in enumerate(items):
        x = x0 + i * w
        if i:
            line(x, y - 6, x, y + 36, "rulefaint")
        text(x + (16 if i else 0), y + 4, label, "k")
        text(x + (16 if i else 0), y + 30, value, "v " + cls)


def defs():
    out.append('<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               '<rect width="5" height="5" class="hatchbg"/><line x1="0" y1="0" x2="0" y2="5" class="hatchline"/></pattern></defs>')


def apple():
    out.clear()
    defs()
    H = 584
    # The package: the SoC die beside its LPDDR, one memory for every client on the die.
    rect(16, 16, 700, 470, "pkg", 12)
    text(34, 42, "APPLE SILICON PACKAGE", "k")
    rect(36, 58, 430, 408, "die", 4)
    text(52, 82, "SOC", "k")
    rect(52, 92, 172, 128, "blk")
    text(64, 112, "CPU", "t")
    cores(64, 124, 6, 3, 18, 25, "core")
    text(64, 210, "encodes command buffers", "n")
    rect(52, 232, 172, 60, "blk dim")
    text(64, 256, "NEURAL ENGINE", "t dim")
    rect(52, 304, 172, 62, "blk hot-frame")
    text(64, 328, "DISPLAY ENGINE", "t")
    text(64, 350, "scans the frame out", "n")
    rect(240, 92, 210, 274, "blk hot")
    text(252, 112, "GPU", "t")
    cores(252, 124, 7, 7, 22, 28, "core on")
    text(252, 354, "torch's kernels, Caliper's pass", "n")
    rect(52, 378, 398, 30, "blk")
    text(251, 398, "SYSTEM LEVEL CACHE", "t", "middle")
    rect(52, 418, 398, 34, "blk dim")
    text(251, 440, "MEMORY CONTROLLERS", "t dim", "middle")
    # LPDDR on the package, wired to the memory controllers.
    for y0 in (58, 268):
        rect(496, y0, 204, 198, "mem", 4)
        for k in range(4):
            rect(512, y0 + 30 + k * 40, 172, 30, "memdie", 2)
        text(598, y0 + 22, "LPDDR", "t", "middle")
    bus(466, 496, 150, 5, 6)
    bus(466, 496, 360, 5, 6)
    # Zoom into DRAM: the pages and who touches them.
    text(740, 42, "DRAM PAGES", "k")
    out.append('<path d="M 700 268 L 736 58 M 700 466 L 736 480" class="zoom"/>')
    rect(736, 52, 348, 434, "zoomframe", 4)
    page_map(746, 66, 25, [
        (8, 40, "tensor", "TENSOR", ["torch's MTLBuffer, 64 MiB", "written by torch's kernels", "read by Caliper's pass"]),
        (56, 40, "texture", "TEXTURE", ["MTLTexture, 64 MiB", "written by Caliper's pass", "sampled by ImGui's draw"]),
        (112, 16, "frame", "FRAME", ["the drawable", "written by ImGui's draw", "read by the display engine"]),
        (160, 3, "commands", "COMMANDS", ["written by the CPU"]),
    ], 916)
    stats(548, [("EXAMPLE", "4096 × 4096 f32", "plain"), ("GPU READS", "64 MiB", "tensor"),
                ("GPU WRITES", "64 MiB", "texture"), ("COPIES", "0", "plain"), ("CPU TOUCHES", "0 bytes", "plain")])
    return H


def nvidia():
    out.clear()
    defs()
    H = 604
    # The host: CPU and system RAM across PCIe. Only commands go that way.
    rect(16, 16, 214, 470, "pkg dim", 12)
    text(34, 42, "HOST", "k")
    rect(32, 58, 182, 128, "blk dim")
    text(44, 78, "CPU", "t")
    cores(44, 90, 6, 3, 18, 25, "core")
    text(44, 176, "records and submits", "n")
    rect(32, 198, 182, 34, "blk dim")
    text(123, 220, "PCIE ROOT", "t dim", "middle")
    rect(32, 250, 182, 150, "mem dim", 4)
    text(123, 272, "SYSTEM RAM", "t", "middle")
    for k in range(3):
        rect(46, 286 + k * 36, 154, 26, "memdie", 2)
    text(44, 420, "used only by the OpenGL", "n")
    text(44, 436, "fallback, which stages here", "n")
    # PCIe: the lanes from the root complex to the GPU's edge.
    bus(214, 330, 205, 8, 3, "lane")
    text(257, 194, "PCIE", "k", "middle")
    text(257, 246, "commands", "n", "middle")
    text(257, 262, "only", "n", "middle")
    # The board: the GPU die ringed by its GDDR.
    rect(300, 16, 416, 470, "pkg", 12)
    text(318, 42, "GPU BOARD", "k")
    for i in range(3):
        rect(366 + i * 104, 56, 76, 48, "mem", 3)
        rect(366 + i * 104, 398, 76, 48, "mem", 3)
        vbus(386 + i * 104, 104, 128, 4, 10)
        vbus(386 + i * 104, 374, 398, 4, 10)
    text(508, 84, "GDDR", "t", "middle")
    text(508, 426, "GDDR", "t", "middle")
    rect(330, 128, 356, 246, "die", 4)
    text(344, 148, "GPU", "k")
    rect(344, 158, 328, 140, "blk hot")
    text(356, 178, "STREAMING MULTIPROCESSORS", "t")
    cores(356, 192, 13, 3, 18, 24.5, "core on")
    text(356, 290, "torch's kernels, the colormap pass, ImGui's draw", "n")
    rect(344, 306, 328, 28, "blk")
    text(508, 325, "L2 CACHE", "t", "middle")
    rect(344, 340, 328, 26, "blk dim")
    text(508, 358, "MEMORY CONTROLLERS", "t dim", "middle")
    # Zoom into VRAM: one allocation, two address spaces.
    text(740, 42, "VRAM PAGES", "k")
    out.append('<path d="M 716 56 L 736 58 M 716 446 L 736 480" class="zoom"/>')
    rect(736, 52, 348, 434, "zoomframe", 4)
    page_map(746, 66, 25, [
        (8, 40, "tensor", "TENSOR", ["torch's CUDA allocator", "written by torch's kernels", "copied by cuMemcpyDtoD"]),
        (56, 40, "shared", "SHARED BUFFER", ["exported by Vulkan and", "imported by CUDA: the same", "pages, in two address spaces"]),
        (104, 40, "texture", "VKIMAGE", ["written by the colormap pass", "sampled by ImGui's draw"]),
        (152, 16, "frame", "SWAPCHAIN", ["written by ImGui's draw", "then presented"]),
        (184, 1, "commands", "SEMAPHORE", ["timeline value: CUDA", "signals, Vulkan waits"]),
    ], 916)
    stats(548, [("EXAMPLE", "4096 × 4096 f32", "plain"), ("OVER PCIE", "0 bytes", "plain"),
                ("COPY IN VRAM", "64 MiB, 0 with alloc_shared", "tensor"), ("GPU WRITES", "64 MiB", "texture"),
                ("HOST-STAGED, OVER PCIE", "64 MiB each way", "faint")])
    return H


def svg(H, label):
    body = "\n      ".join(out)
    return f'<svg class="hw" viewBox="0 0 {W} {H}" role="img" aria-label="{label}">\n      {body}\n    </svg>'


APPLE_LABEL = ("Apple Silicon: the SoC's CPU, GPU, system level cache and display engine share the LPDDR on the "
               "package; in DRAM, torch's tensor buffer is read by Caliper's GPU pass, which writes a texture that "
               "ImGui's draw samples into the frame the display engine scans out; nothing is copied")
NVIDIA_LABEL = ("NVIDIA: the host CPU and system RAM sit across PCIe and only send commands; on the board, the tensor "
                "is copied once inside VRAM into a buffer Vulkan exports and CUDA imports, the streaming "
                "multiprocessors colormap it into a VkImage that ImGui draws into the swapchain, ordered by a "
                "timeline semaphore")


def figure(which):
    return svg(apple(), APPLE_LABEL) if which == "apple" else svg(nvidia(), NVIDIA_LABEL)


if __name__ == "__main__":
    if sys.argv[1] == "--write":
        import re
        page = Path(sys.argv[2])
        parts = re.split(r'(<svg class="hw".*?</svg>)', page.read_text(), flags=re.S)
        if len(parts) != 5:
            sys.exit("expected two <svg class=\"hw\"> figures in the page")
        parts[1], parts[3] = figure("apple"), figure("nvidia")
        page.write_text("".join(parts))
    else:
        print(figure(sys.argv[1]))
