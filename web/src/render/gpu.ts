// One WebGPU device for the whole page, shared by every view. When WebGPU is
// missing, refused or lost, views draw with Canvas 2D instead.

let pending: Promise<GPUDevice | null> | null = null;
let lost = false;
let info: { vendor: string; architecture: string; description: string; fallback: boolean } | null = null;
const lostHandlers = new Set<() => void>();

/** The shared device, or null when the page must draw with Canvas 2D. */
export function gpuDevice(): Promise<GPUDevice | null> {
  if (lost) return Promise.resolve(null);
  if (!pending) {
    pending = (async () => {
      const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
      if (!gpu) return null;
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      const ai = adapter.info;
      info = {
        vendor: ai?.vendor ?? '',
        architecture: ai?.architecture ?? '',
        description: ai?.description ?? '',
        fallback: Boolean((ai as GPUAdapterInfo & { isFallbackAdapter?: boolean })?.isFallbackAdapter),
      };
      const device = await adapter.requestDevice();
      device.lost.then((why) => {
        if (why.reason === 'destroyed') return;
        console.warn('caliper-view: the WebGPU device was lost; drawing with Canvas 2D', why.message);
        lost = true;
        pending = null;
        for (const h of [...lostHandlers]) h();
      });
      device.onuncapturederror = (e) => console.error('caliper-view WebGPU:', e.error.message);
      return device;
    })().catch((e) => {
      console.warn('caliper-view: WebGPU unavailable', e);
      return null;
    });
  }
  return pending;
}

/** Calls fn once the shared device is lost; returns the unregister function. */
export function onDeviceLost(fn: () => void): () => void {
  lostHandlers.add(fn);
  return () => lostHandlers.delete(fn);
}

/** What the adapter reported (after gpuDevice() resolved with a device). */
export function adapterInfo() {
  return info;
}

export function canvasFormat(): GPUTextureFormat {
  return (navigator as Navigator & { gpu: GPU }).gpu.getPreferredCanvasFormat();
}

/** Uploads a 256-entry RGBA table into a 256x1 rgba8unorm texture. */
export function lutTexture(device: GPUDevice, lut: Uint8Array, tex?: GPUTexture): GPUTexture {
  const t = tex ?? device.createTexture({ size: [256, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: t }, lut as Uint8Array<ArrayBuffer>, { bytesPerRow: 256 * 4 }, [256, 1]);
  return t;
}

/** The colour table lookup shared by the shaders: NaN is index 0, as natively. */
export const WGSL_LUT = /* wgsl */ `
fn lut_index(v: f32, lo: f32, hi: f32) -> i32 {
  let b = bitcast<u32>(v);
  if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) { return 0; }
  let d = hi - lo;
  var t = 0.0;
  if (d != 0.0) { t = (v - lo) / d; }
  t = clamp(t, 0.0, 1.0);
  return clamp(i32(t * 255.0 + 0.5), 0, 255);
}
`;
