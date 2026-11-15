// The WebGPU heatmap and image renderer: the snapshot goes to an r32float
// texture (rgba8unorm for an image) and the fragment shader maps every pixel
// of the plot area to its cell and applies the colour table.

import type { Plot2D } from '../views/plot.js';
import { PLATE_RGBA } from '../views/theme.js';
import { canvasFormat, lutTexture, WGSL_LUT } from './gpu.js';

const SHADER = /* wgsl */ `
struct U {
  area: vec4f,   // plot area in device pixels: x, y, w, h
  win: vec4f,    // visible data window: x0, x1, y0, y1 (columns, rows)
  bg: vec4f,
  size: vec2f,   // columns, rows
  range: vec2f,  // vmin, vmax
  mode: u32,     // 0 heatmap, 1 image
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var data: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;
${WGSL_LUT}
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
  return textureLoad(lut, vec2i(lut_index(t.r, u.range.x, u.range.y), 0), 0);
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

  constructor(private readonly device: GPUDevice) {
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) throw new Error('no webgpu context');
    this.ctx = ctx;
    this.ctx.configure({ device, format: canvasFormat(), alphaMode: 'opaque' });
    this.uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
        ],
      });
    }
    const a = plot.area;
    const f = new Float32Array(this.u);
    f.set([a.x * dpr, a.y * dpr, a.w * dpr, a.h * dpr, plot.x.lo, plot.x.hi, plot.y.lo, plot.y.hi, ...PLATE_RGBA, this.cols, this.rows, this.lo, this.hi]);
    new Uint32Array(this.u)[16] = this.image ? 1 : 0;
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
    try {
      this.ctx.unconfigure();
    } catch {
      /* the device may be gone */
    }
  }
}
