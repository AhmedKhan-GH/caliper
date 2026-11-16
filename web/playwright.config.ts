import { defineConfig } from '@playwright/test';

// Headless Chromium only. Each spec runs once per render path:
//   webgpu       WebGPU on the machine's GPU (ANGLE on Metal);
//   swiftshader  WebGPU on the SwiftShader software adapter, the CI-like case;
//   canvas2d     WebGPU disabled, so every view falls back to Canvas 2D.
// Plain headless Chromium offers a SwiftShader adapter that cannot present a
// WebGPU canvas, so the WebGPU projects choose their adapter explicitly.
const base = { browserName: 'chromium' as const, headless: true, viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 };
// A hosted CI runner has three cores and no GPU. There, every SwiftShader page
// rasterises in software on those cores, and a page of WebAssembly views draws
// on every animation frame; four such pages at once drew one or two frames a
// second and starved the rest. So in CI the swiftshader project runs one test
// at a time, and the workflow runs it apart from the canvas2d project.
const CI = !!process.env.CI;

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
      use: { ...base, launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'] } },
    },
    { name: 'canvas2d', use: { ...base, launchOptions: { args: ['--disable-features=WebGPU'] } } },
  ],
});
