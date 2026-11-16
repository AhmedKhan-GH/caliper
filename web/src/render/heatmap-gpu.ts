// The WebGPU heatmap and image renderer: the snapshot goes to an r32float
// texture (rgba8unorm for an image) and the fragment shader maps every pixel
// of the plot area to its cell and applies the colour table.
//
// A heatmap's colour index is exactly the native reference's (lutIndex), as
// in the native WebGPU renderer: WGSL allows a division 2.5 ULP of error and
// may assume no NaN or infinity, so the shader does not divide. The CPU finds
// the threshold of each index (colormapThresholds), and the shader counts the
// thresholds a value has passed with a binary search of exact comparisons;
// NaN, tested on its bits, is index 0.

import { colormapThresholds } from './colormap.js';
import type { Plot2D } from '../views/plot.js';
import { PLATE_RGBA } from '../views/theme.js';
import { canvasFormat, lutTexture } from './gpu.js';

const SHADER = /* wgsl */ `
struct U {
  area: vec4f,   // plot area in device pixels: x, y, w, h
  win: vec4f,    // visible data window: x0, x1, y0, y1 (columns, rows)
  bg: vec4f,
  size: vec2f,   // columns, rows
  range: vec2f,  // vmin, vmax
  mode: u32,     // 0 heatmap, 1 image
  cmap: u32,     // the thresholds' mode: 0 every value at index 0, 1 increasing, 2 decreasing
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var data: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> th: array<f32>;   // T[1..255]
fn index_of(v: f32) -> i32 {
  let b = bitcast<u32>(v);
  if (((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) || u.cmap == 0u) { return 0; }
  var lo = 1u;
  var hi = 256u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2u;
    let passed = select(v <= th[mid], th[mid] <= v, u.cmap == 1u);
    if (passed) { lo = mid + 1u; } else { hi = mid; }
  }
  return i32(lo - 1u);
}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let f = (pos.xy - u.area.xy) / u.area.zw;
  let d = vec2f(mix(u.win.x, u.win.y, f.x), mix(u.win.z, u.win.w, f.y));
  if (d.x < 0.0 || d.y < 0.0 || d.x >= u.size.x || d.y >= u.size.y) { return u.bg; }
  let c = vec2i(floor(d));
  let t = textureLoad(data, c, 0);
  if (u.mode == 1u) { return vec4f(t.rgb, 1.0); }
  return textureLoad(lut, vec2i(index_of(t.r), 0), 0);
}
`;

const pipelines = new WeakMap<GPUDevice, { heat: GPURenderPipeline; image: GPURenderPipeline; layoutHeat: GPUBindGroupLayout; layoutImage: GPUBindGroupLayout }>();

function pipelinesOf(device: GPUDevice) {
  let p = pipelines.get(device);
  if (p) return p;
  const module = device.createShaderModule({ code: SHADER });
  const make = (sampleType: GPUTextureSampleType) => {
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: canvasFormat() }] },
      primitive: { topology: 'triangle-list' },
    });
    return { layout, pipeline };
  };
  const h = make('unfilterable-float');
  const i = make('float');
  p = { heat: h.pipeline, image: i.pipeline, layoutHeat: h.layout, layoutImage: i.layout };
  pipelines.set(device, p);
  return p;
}

export class HeatmapGpu {
  readonly canvas = document.createElement('canvas');
  private readonly ctx: GPUCanvasContext;
  private readonly uniform: GPUBuffer;
  private readonly u = new ArrayBuffer(80);
  private tex: GPUTexture | null = null;
  private lut: GPUTexture | null = null;
  private lutData: Uint8Array | null = null;
  private bind: GPUBindGroup | null = null;
  private image = false;
  private cols = 0;
  private rows = 0;
  private lo = 0;
  private hi = 1;
  /** The colour thresholds of [thLo, thHi] (colormapThresholds) and their mode. */
  private readonly th: GPUBuffer;
  private thMode = 0;
  private thLo = NaN;
  private thHi = NaN;

  constructor(private readonly device: GPUDevice) {
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) throw new Error('no webgpu context');
    this.ctx = ctx;
    this.ctx.configure({ device, format: canvasFormat(), alphaMode: 'opaque' });
    this.uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.th = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  upload(data: Float32Array | Uint8Array, rows: number, cols: number, image: boolean, lo: number, hi: number, lut: Uint8Array): void {
    const d = this.device;
    const format: GPUTextureFormat = image ? 'rgba8unorm' : 'r32float';
    if (!this.tex || this.rows !== rows || this.cols !== cols || this.image !== image) {
      this.tex?.destroy();
      this.tex = d.createTexture({ size: [cols, rows], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.rows = rows;
      this.cols = cols;
      this.image = image;
      this.bind = null;
    }
    d.queue.writeTexture({ texture: this.tex }, data as Uint8Array<ArrayBuffer>, { bytesPerRow: cols * 4, rowsPerImage: rows }, [cols, rows]);
    if (this.lutData !== lut) {
      this.lut = lutTexture(d, lut, this.lut ?? undefined);
      this.lutData = lut;
    }
    this.lo = lo;
    this.hi = hi;
    if (!image && !(Object.is(lo, this.thLo) && Object.is(hi, this.thHi))) {
      const t = colormapThresholds(lo, hi);
      d.queue.writeBuffer(this.th, 0, t.T as Float32Array<ArrayBuffer>);
      this.thMode = t.mode;
      this.thLo = lo;
      this.thHi = hi;
    }
  }

  draw(plot: Plot2D, dpr: number): void {
    const d = this.device;
    if (!this.tex || !this.lut || this.canvas.width < 1) return;
    const p = pipelinesOf(d);
    if (!this.bind) {
      this.bind = d.createBindGroup({
        layout: this.image ? p.layoutImage : p.layoutHeat,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: this.tex.createView() },
          { binding: 2, resource: this.lut.createView() },
          { binding: 3, resource: { buffer: this.th } },
        ],
      });
    }
    const a = plot.area;
    const f = new Float32Array(this.u);
    f.set([a.x * dpr, a.y * dpr, a.w * dpr, a.h * dpr, plot.x.lo, plot.x.hi, plot.y.lo, plot.y.hi, ...PLATE_RGBA, this.cols, this.rows, this.lo, this.hi]);
    new Uint32Array(this.u)[16] = this.image ? 1 : 0;
    new Uint32Array(this.u)[17] = this.thMode;
    d.queue.writeBuffer(this.uniform, 0, this.u);
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: PLATE_RGBA[0], g: PLATE_RGBA[1], b: PLATE_RGBA[2], a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    const W = this.canvas.width, H = this.canvas.height;
    const x0 = Math.max(0, Math.floor(a.x * dpr)), y0 = Math.max(0, Math.floor(a.y * dpr));
    const x1 = Math.min(W, Math.ceil((a.x + a.w) * dpr)), y1 = Math.min(H, Math.ceil((a.y + a.h) * dpr));
    if (x1 > x0 && y1 > y0) {
      pass.setScissorRect(x0, y0, x1 - x0, y1 - y0);
      pass.setPipeline(this.image ? p.image : p.heat);
      pass.setBindGroup(0, this.bind);
      pass.draw(3);
    }
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.tex?.destroy();
    this.lut?.destroy();
    this.uniform.destroy();
    this.th.destroy();
    try {
      this.ctx.unconfigure();
    } catch {
      /* the device may be gone */
    }
  }
}
