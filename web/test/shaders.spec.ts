// The generated shader programs (src/host/gfx/shaders/generated) in headless Chromium, with raw
// WebGPU and no view code: every program compiles and every compute entry and checked pair
// becomes a pipeline with the program's explicit layouts, on a default device and on a
// compatibility-mode device; the colormap by keys matches the CPU reference byte for byte on
// both; and the points, segment and mesh programs draw a small fixture, read back as PNGs.
//
// The colormap's cases come from the C++ reference map_f32_to_rgba8 when CALIPER_COLORMAP_CASES
// names the file caliper_shader_tests writes (CALIPER_COLORMAP_CASES_OUT); otherwise from the
// TypeScript port of the same reference, lutIndex in src/render/colormap.ts, so the test also
// runs where the C++ is not built.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { lutIndex } from '../src/render/colormap.js';
import { expect, REPO, serve, SHOTS, test } from './util.js';

const GEN = path.join(REPO, 'src', 'host', 'gfx', 'shaders', 'generated');
type Prog = { name: string; test_only: boolean; pairs: [string, string][]; bindings: unknown[]; entries: { name: string; stage: string }[] };
const listing = JSON.parse(fs.readFileSync(path.join(GEN, 'programs.json'), 'utf8')) as { naga: string; programs: Prog[] };
const programs = Object.fromEntries(listing.programs.map((p) => [p.name, p]));
// The WGSL of each program, read from the raw string pieces of its generated header.
const codes = Object.fromEntries(listing.programs.map((p) => {
  const h = fs.readFileSync(path.join(GEN, `${p.name}_wgsl.h`), 'utf8');
  return [p.name, [...h.matchAll(/R"wgsl\(([\s\S]*?)\)wgsl"/g)].map((m) => m[1]).join('')];
}));

// ---- colormap cases ------------------------------------------------------------------------

type Case = { name: string; block: Buffer; inputs: Buffer; expected: Uint32Array };

function readCases(file: string): Case[] {
  const b = fs.readFileSync(file);
  if (b.toString('latin1', 0, 8) !== 'CMAPCASE') throw new Error(`${file} is not a colormap case file`);
  const n = b.readUInt32LE(8);
  const out: Case[] = [];
  let at = 12;
  for (let r = 0; r < n; r++) {
    const block = b.subarray(at, at + 2080); at += 2080;
    const count = b.readUInt32LE(at); at += 4;
    const inputs = b.subarray(at, at + 4 * count); at += 4 * count;
    const expected = new Uint32Array(b.buffer.slice(b.byteOffset + at, b.byteOffset + at + 4 * count)); at += 4 * count;
    out.push({ name: `C++ range ${r}`, block, inputs, expected });
  }
  return out;
}

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const asFloat = (b: number) => { u32[0] = b; return f32[0]; };
const asBits = (f: number) => { f32[0] = f; return u32[0]; };
const keyOf = (b: number) => ((b >>> 31) ? ~b : (b | 0x80000000)) >>> 0;
const bitsOf = (k: number) => ((k & 0x80000000) ? (k & 0x7fffffff) : ~k) >>> 0;

/**
 * The reference's index on lutIndex. Where t is NaN (a range with an infinite end), the C++
 * converts it to an int as 0 (ARM64) or INT_MIN (x86), clamped to 0, but lutIndex returns NaN;
 * this takes the C++'s 0.
 */
function refIndex(v: number, vmin: number, vmax: number): number {
  const i = lutIndex(v, vmin, vmax);
  return Number.isNaN(i) ? 0 : i;
}

/** The same table and the same derivation as tests/shaders/colormap_cases.h, on refIndex. */
function lut(): Uint32Array {
  const t = new Uint32Array(256);
  for (let k = 0; k < 256; k++) t[k] = (k | ((255 - k) << 8) | (((k * 37) & 255) << 16) | ((255 - ((k * 11) & 127)) << 24)) >>> 0;
  return t;
}

function block(vmin: number, vmax: number, table: Uint32Array): { bytes: Buffer; keys: Uint32Array; flip: number } {
  const ref = (bits: number) => refIndex(asFloat(bits), vmin, vmax);
  const kmin = keyOf(0xff800000), kmax = keyOf(0x7f800000);
  const flip = ref(0xff800000) > ref(0x7f800000) ? 0xffffffff : 0;
  const lo = flip ? (~kmax >>> 0) : kmin, hi = flip ? (~kmin >>> 0) : kmax;
  const f = (fk: number) => ref(bitsOf((fk ^ flip) >>> 0));
  const keys = new Uint32Array(260).fill(0xffffffff);
  keys[0] = 0;
  for (let k = 1; k < 256; k++) {
    if (f(hi) < k) continue;
    let a = lo, b = hi;
    while (a < b) { const mid = a + Math.floor((b - a) / 2); if (f(mid) >= k) b = mid; else a = mid + 1; }
    keys[k] = a;
  }
  const bytes = Buffer.alloc(2080);
  bytes.writeFloatLE(vmin, 0); bytes.writeFloatLE(vmax, 4); bytes.writeUInt32LE(flip >>> 0, 8);
  for (let i = 0; i < 260; i++) bytes.writeUInt32LE(keys[i], 16 + 4 * i);
  for (let i = 0; i < 256; i++) bytes.writeUInt32LE(table[i], 1056 + 4 * i);
  return { bytes, keys, flip };
}

function derivedCases(): Case[] {
  const table = lut();
  const F = 3.4028234663852886e38, inf = Infinity;
  const ranges: [string, number, number][] = [
    ['increasing', 0, 1], ['increasing, offset', -3.5, 7.25], ['decreasing', 2, -1], ['decreasing, wide', 1e30, -1e30],
    ['degenerate', 0.5, 0.5], ['infinite end above', 0, inf], ['infinite end below', -inf, 1], ['subnormal', -1e-39, 1e-39],
    ['largest floats', 0, F], ['overflowing span', -F, F]];
  let s = 0x9e3779b9;
  return ranges.map(([name, vmin, vmax]) => {
    const { bytes, keys, flip } = block(vmin, vmax, table);
    const inp = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0x7fbfffff, 0xffc00000, 0xff800001, 0x7fffffff, 0xffffffff,
                 asBits(vmin), asBits(vmax), asBits(F), asBits(-F), 0x00800000, 0x80800000, 1, 0x80000001, asBits(1), asBits(-1)];
    for (let k = 1; k < 256; k++) if (keys[k] !== 0xffffffff) for (const d of [-1, 0, 1]) inp.push(bitsOf(((keys[k] + d) ^ flip) >>> 0));
    for (let i = 0; i < 65536; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; inp.push(s >>> 0); }
    while (inp.length % 64) inp.push(0);
    const inputs = Buffer.from(new Uint32Array(inp).buffer);
    const expected = new Uint32Array(inp.map((b) => table[refIndex(asFloat(b >>> 0), vmin, vmax)]));
    return { name: `${name} (TypeScript reference)`, block: bytes, inputs, expected };
  });
}

// ---- PNG -----------------------------------------------------------------------------------

function png(rgba: Buffer, w: number, h: number): Buffer {
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(zlib.crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  const raw = Buffer.alloc(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) rgba.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr),
                        chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}

// ---- the fixture of the capture ------------------------------------------------------------

function fixture() {
  const points = { pos: [] as number[], values: [] as number[], size_px: 7 };
  for (let i = 0; i < 160; i++) {
    const t = i / 159, a = t * 5.5 * Math.PI, r = 0.12 + 0.78 * t;
    points.pos.push(r * Math.cos(a), r * Math.sin(a), 0);
    points.values.push(asBits(t));
  }
  points.pos.push(NaN, 0, 0); points.values.push(asBits(0.5));   // a NaN position: hidden
  const rgba = (r: number, g: number, b: number) => (r | (g << 8) | (b << 16) | (255 << 24)) >>> 0;
  const lines = { strip: [] as number[], strip_rgba: [] as number[], spokes: [] as number[], spoke_values: [] as number[] };
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? 0.34 : 0.82;
    lines.strip.push(r * Math.cos(a), r * Math.sin(a), 0);
    lines.strip_rgba.push(i % 2 ? rgba(240, 240, 235) : rgba(232, 112, 64));
  }
  for (let i = 0; i < 24; i++) {
    const a = i * Math.PI / 12;
    lines.spokes.push(0.86 * Math.cos(a), 0.86 * Math.sin(a), 0, 0.95 * Math.cos(a), 0.95 * Math.sin(a), 0);
    lines.spoke_values.push(asBits(i / 23), asBits(i / 23));
  }
  // A height field over [-1, 1]^2, z up, seen from above at an angle.
  const N = 28, pos: number[] = [], values: number[] = [], indices: number[] = [];
  let zmin = Infinity, zmax = -Infinity;
  const zs: number[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const x = -1 + 2 * i / (N - 1), y = -1 + 2 * j / (N - 1);
    const z = 0.35 * Math.sin(2.2 * x) * Math.cos(1.8 * y) + 0.15 * Math.exp(-4 * (x * x + y * y));
    pos.push(x, y, z); zs.push(z); zmin = Math.min(zmin, z); zmax = Math.max(zmax, z);
  }
  for (const z of zs) values.push(asBits((z - zmin) / (zmax - zmin)));
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
    const a = j * N + i;
    indices.push(a, a + 1, a + N, a + N, a + 1, a + N + 1);
  }
  const drawn = indices.length;
  indices.push(N * N, 0xffffffff, 123456);   // past the last vertex: the clamp maps them to it
  const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
  const norm = (a: number[]) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const eye = [1.9, -2.5, 2.1], target = [0, 0, -0.05], up = [0, 0, 1];
  const fwd = norm(sub(target, eye)), s = norm(cross(fwd, up)), u = cross(s, fwd);
  const view = [s[0], u[0], -fwd[0], 0, s[1], u[1], -fwd[1], 0, s[2], u[2], -fwd[2], 0, -dot(s, eye), -dot(u, eye), dot(fwd, eye), 1];
  const fy = 1 / Math.tan((38 * Math.PI / 180) / 2), near = 0.1, far = 10;
  const proj = [fy, 0, 0, 0, 0, fy, 0, 0, 0, 0, far / (near - far), -1, 0, 0, near * far / (near - far), 0];
  const mvp = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) mvp[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];
  const mesh = { pos, values, indices, drawn, vertex_count: N * N, mvp,
                 nmat0: [s[0], u[0], -fwd[0], 0], nmat1: [s[1], u[1], -fwd[1], 0], nmat2: [s[2], u[2], -fwd[2], 0] };
  return { points, lines, mesh };
}

/** A colormap block for the captures: range [0, 1] and a table from dark violet to yellow. */
function captureBlock(): Buffer {
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const t = new Uint32Array(256);
  for (let k = 0; k < 256; k++) {
    const x = (k / 255) * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
    const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
    t[k] = (c[0] | (c[1] << 8) | (c[2] << 16) | (255 << 24)) >>> 0;
  }
  return block(0, 1, t).bytes;
}

// ---- the tests -----------------------------------------------------------------------------

test.describe('generated shader programs', () => {
  let origin = '';
  let close = () => {};
  test.beforeAll(async () => { ({ origin, close } = await serve(REPO)); });
  test.afterAll(() => close());

  test.beforeEach(async ({ page }, info) => {
    test.skip(info.project.name === 'canvas2d', 'needs WebGPU');
    await page.goto(`${origin}/web/test/blank.html`);
    await page.addScriptTag({ url: `${origin}/web/test/shaders_page.js` });
  });

  for (const level of ['default', 'compatibility'] as const) {
    test(`every program compiles and every pipeline is created on a ${level} device`, async ({ page }, info) => {
      const r = await page.evaluate(async ({ level, programs, codes }) => {
        const c = (window as any).shaderCheck;
        const d = await c.device(level);
        const out: { messages: string[]; pipelines: number; expected: number } = { messages: [], pipelines: 0, expected: 0 };
        for (const p of Object.values(programs) as any[]) {
          const one = await c.compileProgram(d.dev, p, codes[p.name]);
          out.messages.push(...one.messages); out.pipelines += one.pipelines; out.expected += one.expected;
        }
        return { ...out, adapter: d.adapter, core: d.core, fallback: d.fallback };
      }, { level, programs, codes });
      console.log(`[${info.project.name}] ${level}: ${r.adapter}${r.fallback ? ' (fallback)' : ''}, core features ${r.core}: `
                  + `${Object.keys(programs).length} programs, ${r.pipelines} of ${r.expected} pipelines, ${r.messages.length} messages`);
      expect(r.core).toBe(level === 'default');
      expect(r.messages).toEqual([]);
      expect(r.pipelines).toBe(r.expected);
    });

    test(`the colormap by keys matches the CPU reference byte for byte on a ${level} device`, async ({ page }, info) => {
      test.setTimeout(240_000);
      const file = process.env.CALIPER_COLORMAP_CASES;
      const cases = file ? readCases(file) : derivedCases();
      let total = 0, bad = 0;
      const first: string[] = [];
      for (const c of cases) {
        const got64 = await page.evaluate(async ({ level, prog, code, block, inputs }) => {
          const s = (window as any).shaderCheck;
          const d = await s.device(level);
          return s.colormap(d.dev, prog, code, block, inputs);
        }, { level, prog: programs.colormap_probe, code: codes.colormap_probe, block: c.block.toString('base64'), inputs: c.inputs.toString('base64') });
        const g = Buffer.from(got64, 'base64');
        const got = new Uint32Array(g.buffer, g.byteOffset, g.length / 4);
        for (let i = 0; i < got.length; i++) {
          if (got[i] !== c.expected[i]) {
            if (bad++ < 5) first.push(`${c.name}: input 0x${c.inputs.readUInt32LE(4 * i).toString(16)} got 0x${got[i].toString(16)} want 0x${c.expected[i].toString(16)}`);
          }
        }
        total += got.length;
      }
      console.log(`[${info.project.name}] ${level}: ${total} inputs over ${cases.length} ranges from the ${file ? 'C++' : 'TypeScript'} reference, ${bad} differ`);
      expect(first).toEqual([]);
      expect(bad).toBe(0);
    });
  }

  test('the points, segment and mesh programs draw their fixture', async ({ page }, info) => {
    const fx = fixture();
    const r = await page.evaluate(async ({ programs, codes, block, fx }) => {
      const s = (window as any).shaderCheck;
      const d = await s.device('default');
      return s.capture(d.dev, programs, codes, block, fx);
    }, { programs, codes, block: captureBlock().toString('base64'), fx });
    expect(r.errors).toEqual([]);
    expect(r.clamp.same).toBe(true);
    expect(r.clamp.tail).toEqual([fx.mesh.vertex_count - 1, fx.mesh.vertex_count - 1, fx.mesh.vertex_count - 1]);
    fs.mkdirSync(SHOTS, { recursive: true });
    for (const name of ['points', 'lines', 'mesh']) {
      const px = Buffer.from(r[name], 'base64');
      // Something other than the clear colour was drawn.
      let drawn = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i] !== px[0] || px[i + 1] !== px[1] || px[i + 2] !== px[2]) drawn++;
      expect(drawn, name).toBeGreaterThan(500);
      fs.writeFileSync(path.join(SHOTS, `shaders-${name}-${info.project.name}.png`), png(px, 256, 256));
    }
  });
});
