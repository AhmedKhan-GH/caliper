import { defineConfig } from '@playwright/test';

// Headless Chromium only. Each spec runs once per render path:
//   webgpu       WebGPU on the machine's GPU (ANGLE on Metal);
//   swiftshader  WebGPU on the SwiftShader software adapter, the CI-like case;
//   canvas2d     WebGPU disabled, so every view falls back to Canvas 2D.
// Plain headless Chromium offers a SwiftShader adapter that cannot present a
// WebGPU canvas, so the WebGPU projects choose their adapter explicitly.
const base = { browserName: 'chromium' as const, headless: true, viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 };

export default defineConfig({
  testDir: 'test',
  testMatch: '*.spec.ts',
  timeout: 60_000,
  fullyParallel: true,
  workers: 4,
  reporter: [['list']],
  projects: [
    { name: 'webgpu', use: { ...base, launchOptions: { args: ['--enable-unsafe-webgpu', '--use-angle=metal'] } } },
    {
      name: 'swiftshader',
      use: { ...base, launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader'] } },
    },
    { name: 'canvas2d', use: { ...base, launchOptions: { args: ['--disable-features=WebGPU'] } } },
  ],
});
