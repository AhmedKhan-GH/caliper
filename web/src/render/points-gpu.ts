// The WebGPU points renderer: one instanced quad per point, a round sprite
// cut in the fragment shader, depth-tested in 3-D so categorical colours stay
// exact under overlap.

import { PLATE_RGBA } from '../views/theme.js';
import { canvasFormat } from './gpu.js';

const SHADER = /* wgsl */ `
struct U {
  mvp: mat4x4f,
  viewport: vec2f,   // device pixels of the viewport
  radius: f32,       // device pixels
  _pad: f32,
};
@group(0) @binding(0) var<uniform> u: U;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) color: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) p: vec3f, @location(1) color: vec4f) -> V {
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  var c = u.mvp * vec4f(p, 1.0);
  c = vec4f(c.xy + corner * (2.0 * u.radius / u.viewport) * c.w, c.zw);
  var o: V;
  o.pos = c;
  o.uv = corner;
  o.color = color;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  if (dot(v.uv, v.uv) > 1.0) { discard; }
  return vec4f(v.color.rgb, 1.0);
}
`;

const cache = new WeakMap<GPUDevice, { depth: GPURenderPipeline; flat: GPURenderPipeline; layout: GPUBindGroupLayout }>();

function pipelines(device: GPUDevice) {
  let p = cache.get(device);
  if (p) return p;
  const module = device.createShaderModule({ code: SHADER });
  const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
  const make = (depth: boolean) => device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: {
      module, entryPoint: 'vs',
      buffers: [
        { arrayStride: 12, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
        { arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 1, offset: 0, format: 'unorm8x4' }] },
      ],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format: canvasFormat() }] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: depth, depthCompare: depth ? 'less' : 'always' },
  });
  p = { depth: make(true), flat: make(false), layout };
  cache.set(device, p);
  return p;
}

export class PointsGpu {
  readonly canvas = document.createElement('canvas');
  private readonly ctx: GPUCanvasContext;
  private readonly uniform: GPUBuffer;
  private readonly bind: GPUBindGroup;
  private pos: GPUBuffer | null = null;
  private col: GPUBuffer | null = null;
  private cap = 0;
  private n = 0;
  private depthTex: GPUTexture | null = null;

  constructor(private readonly device: GPUDevice) {
    const ctx = this.canvas.getContext('webgpu');
    if (!ctx) throw new Error('no webgpu context');
    this.ctx = ctx;
    ctx.configure({ device, format: canvasFormat(), alphaMode: 'opaque' });
    this.uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bind = device.createBindGroup({ layout: pipelines(device).layout, entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
  }

  private ensure(n: number): void {
    if (n <= this.cap && this.pos && this.col) return;
    this.pos?.destroy();
    this.col?.destroy();
    this.cap = Math.max(n, 1024);
    this.pos = this.device.createBuffer({ size: this.cap * 12, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.col = this.device.createBuffer({ size: this.cap * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  }

  /** positions: (n, 3) float32; colors: n packed RGBA (r in the low byte). */
  upload(positions: Float32Array, colors: Uint32Array, n: number): void {
    this.ensure(n);
    this.n = n;
    if (n === 0) return;
    this.device.queue.writeBuffer(this.pos!, 0, positions as Float32Array<ArrayBuffer>, 0, n * 3);
    this.device.queue.writeBuffer(this.col!, 0, colors as Uint32Array<ArrayBuffer>, 0, n);
  }

  /** Rewrites the colours only (a selection changed). */
  recolor(colors: Uint32Array): void {
    if (this.col && this.n) this.device.queue.writeBuffer(this.col, 0, colors as Uint32Array<ArrayBuffer>, 0, this.n);
  }

  /** viewport is the drawing rectangle in device pixels. */
  draw(mvp: Float32Array, viewport: { x: number; y: number; w: number; h: number }, radius: number, depth: boolean): void {
    const d = this.device;
    const W = this.canvas.width, H = this.canvas.height;
    if (!this.depthTex || this.depthTex.width !== W || this.depthTex.height !== H) {
      this.depthTex?.destroy();
      this.depthTex = d.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const u = new Float32Array(20);
    u.set(mvp, 0);
    u.set([viewport.w, viewport.h, radius, 0], 16);
    d.queue.writeBuffer(this.uniform, 0, u);
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: PLATE_RGBA[0], g: PLATE_RGBA[1], b: PLATE_RGBA[2], a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    const x0 = Math.max(0, Math.floor(viewport.x)), y0 = Math.max(0, Math.floor(viewport.y));
    const x1 = Math.min(W, Math.ceil(viewport.x + viewport.w)), y1 = Math.min(H, Math.ceil(viewport.y + viewport.h));
    const vx = Math.min(Math.max(0, viewport.x), W - 1), vy = Math.min(Math.max(0, viewport.y), H - 1);
    const vw = Math.min(viewport.w, W - vx), vh = Math.min(viewport.h, H - vy);
    if (this.n > 0 && x1 > x0 && y1 > y0 && vw > 0 && vh > 0) {
      const p = pipelines(d);
      pass.setViewport(vx, vy, vw, vh, 0, 1);
      pass.setScissorRect(x0, y0, x1 - x0, y1 - y0);
      pass.setPipeline(depth ? p.depth : p.flat);
      pass.setBindGroup(0, this.bind);
      pass.setVertexBuffer(0, this.pos!);
      pass.setVertexBuffer(1, this.col!);
      pass.draw(4, this.n);
    }
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.pos?.destroy();
    this.col?.destroy();
    this.depthTex?.destroy();
    this.uniform.destroy();
    try {
      this.ctx.unconfigure();
    } catch {
      /* the device may be gone */
    }
  }
}
