import { defineConfig } from '@playwright/test';

// Headless Chromium only. Each spec runs once per project:
//   webgpu       WebGPU on the machine's GPU (ANGLE on Metal);
//   swiftshader  WebGPU on the SwiftShader software adapter, the CI-like case;
//   canvas2d     WebGPU disabled, so the WebAssembly views do not draw.
// The WebAssembly views draw with WebGPU in the first two, and the element's
// own TypeScript views draw with Canvas 2D in all three.
// Plain headless Chromium offers a SwiftShader adapter that cannot present a
// WebGPU canvas, so the WebGPU projects choose their adapter explicitly.
const base = { browserName: 'chromium' as const, headless: true, viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 };
// A hosted CI runner has three cores and no GPU. There, every SwiftShader page
// rasterises in software on those cores, and a page of WebAssembly views draws
// on every animation frame; four such pages at once drew one or two frames a
// second and starved the rest. So in CI the swiftshader project runs one test
// at a time, and the workflow runs it apart from the canvas2d project.
const CI = !!process.env.CI;
// And in CI the swiftshader project runs only the specs that exercise WebGPU
// drawing: the WebAssembly views, the renderer choice and its fallback, the
// 3-D points, the colormap's shaders, linked selection's highlights and the
// specs that read pixels. The specs of the protocol and its round trips run
// there on canvas2d only, where they do not wait on a software rasteriser:
// codec (Node only, skipped on the other projects anyway), controls (the run
// controls and the panel; their views are drawn by views.spec, wasm.spec and
// linked.spec) and e2e (the stream page; its views' renderers are checked by
// views.spec, renderer.spec and fallback.spec). perf measures the element's
// own views, which draw with Canvas 2D everywhere, and skips itself outside
// canvas2d. Without CI every project runs every spec, as before.
const CI_SWIFTSHADER_SKIPS = ['codec.spec.ts', 'controls.spec.ts', 'e2e.spec.ts'];

export default defineConfig({
  testDir: 'test',
  testMatch: '*.spec.ts',
  timeout: 60_000,
  fullyParallel: true,
  workers: CI ? 2 : 4,
  reporter: [['list']],
  projects: [
    { name: 'webgpu', use: { ...base, launchOptions: { args: ['--enable-unsafe-webgpu', '--use-angle=metal'] } } },
    {
      name: 'swiftshader',
      workers: CI ? 1 : undefined,
      testIgnore: CI ? CI_SWIFTSHADER_SKIPS : undefined,
      use: { ...base, launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'] } },
    },
    { name: 'canvas2d', use: { ...base, launchOptions: { args: ['--disable-features=WebGPU'] } } },
  ],
});
