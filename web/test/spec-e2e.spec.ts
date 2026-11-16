// The spec page end to end against the real app: `caliper --hidden --api
// --api-port 0 --workspace <tmp>` with a command script that imports a small
// Parquet file and a small safetensors file, runs their loaders, builds the
// dashboard of src/builder/spec/tests/fixtures/dashboard.spec.toml and saves
// it as a spec applet; then headless Chromium opens the spec page from the
// app's spec_url (the API file) and the test checks that the three panels drew
// and that the page's link opens nothing else.
//
// It needs the built app in CALIPER_BIN and a Python with pyarrow in
// CALIPER_PYTHON (for the Parquet file), and is skipped without them. With
// CALIPER_SPEC_E2E_DATA naming a folder that holds run42_metrics.parquet and
// attn.safetensors, it imports those instead; with CALIPER_SHOTS it saves the
// page at 2560x1440 and 1440x900 there as web-dashboard-<size>.png. The app runs
// with CALIPER_AGENT_OFFLINE=1, its own CALIPER_DATA_DIR and a hidden window,
// and is stopped by the pid the test started.

import { test, expect } from '@playwright/test';
import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { pageState, settled } from './spec-mock.js';
import { REPO, WEB } from './util.js';

const BIN = process.env.CALIPER_BIN ?? '';
const PYTHON = process.env.CALIPER_PYTHON ?? path.join(REPO, '.venv', 'bin', 'python');
const DATA = process.env.CALIPER_SPEC_E2E_DATA ?? '';
const SHOTS = process.env.CALIPER_SHOTS ?? '';

test.beforeEach(({}, info) => {
  test.skip(info.project.name !== 'canvas2d', 'the spec page draws with Canvas 2D: one project is enough');
  test.skip(!BIN || !fs.existsSync(BIN), 'needs the built app in CALIPER_BIN');
  test.skip(!DATA && !fs.existsSync(PYTHON), 'needs a Python with pyarrow in CALIPER_PYTHON');
});
test.setTimeout(120_000);

/** Writes the small data files: 200 steps of metrics as Parquet, and weights [2, 3, 8, 8] with a diagonal in every head. */
function writeData(dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
  execFileSync(PYTHON, ['-I', '-c', [
    'import math, sys, pyarrow as pa, pyarrow.parquet as pq',
    'n = 200',
    'pq.write_table(pa.table({"step": pa.array(range(n), pa.int64()),',
    '  "loss": pa.array([0.1 + 2.0 * math.exp(-i / 50.0) for i in range(n)], pa.float64()),',
    '  "accuracy": pa.array([0.1 + 0.8 * (1 - math.exp(-i / 40.0)) for i in range(n)], pa.float64())}), sys.argv[1])',
  ].join('\n'), path.join(dir, 'run42_metrics.parquet')]);
  const w = new Float32Array(2 * 3 * 8 * 8);
  for (let l = 0; l < 2; l++) for (let h = 0; h < 3; h++) for (let q = 0; q < 8; q++) w[((l * 3 + h) * 8 + q) * 8 + q] = 1 + h;
  let header = JSON.stringify({ weights: { dtype: 'F32', shape: [2, 3, 8, 8], data_offsets: [0, w.byteLength] } });
  while (header.length % 8) header += ' ';
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(path.join(dir, 'attn.safetensors'), Buffer.concat([len, Buffer.from(header), Buffer.from(w.buffer)]));
}

interface App { proc: ChildProcess; specUrl: string; root: string; log: string[] }

async function startApp(root: string): Promise<App> {
  const data = DATA || path.join(root, 'data');
  if (!DATA) writeData(data);
  const ws = path.join(root, 'workspace');
  const dataDir = path.join(root, 'app-data');
  fs.mkdirSync(dataDir, { recursive: true });
  const fixture = fs.readFileSync(path.join(REPO, 'src', 'builder', 'spec', 'tests', 'fixtures', 'dashboard.spec.toml'), 'utf8');
  const script = [
    { name: 'data.import', args: { paths: [path.join(data, 'run42_metrics.parquet'), path.join(data, 'attn.safetensors')], run: true } },
    { name: 'spec.new', args: { id: 'run42-dashboard' } },
    { name: 'spec.set_text', args: { text: fixture } },
    { name: 'applet.save', args: { overwrite: true } },
  ].map((l) => JSON.stringify(l)).join('\n') + '\n';
  fs.writeFileSync(path.join(root, 'build.jsonl'), script);
  const results = path.join(root, 'results.jsonl');
  const proc = spawn(BIN, ['--hidden', '--api', '--api-port', '0', '--workspace', ws], {
    env: {
      ...process.env, CALIPER_AGENT_OFFLINE: '1', CALIPER_DATA_DIR: dataDir, CALIPER_WINDOW_SIZE: '1280x800',
      CALIPER_COMMAND_SCRIPT: path.join(root, 'build.jsonl'), CALIPER_COMMAND_RESULTS: results,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const log: string[] = [];
  proc.stdout!.on('data', (b) => log.push(String(b)));
  proc.stderr!.on('data', (b) => log.push(String(b)));
  const apiFile = path.join(dataDir, 'api.json');
  const deadline = Date.now() + 60_000;
  let specUrl = '';
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`the app exited with ${proc.exitCode}:\n${log.join('')}`);
    const done = fs.existsSync(results) && fs.readFileSync(results, 'utf8').includes('"done"');
    if (!specUrl && fs.existsSync(apiFile)) {
      try {
        specUrl = JSON.parse(fs.readFileSync(apiFile, 'utf8')).spec_url ?? '';
      } catch { /* written meanwhile */ }
    }
    if (done && specUrl) break;
    await new Promise((ok) => setTimeout(ok, 100));
  }
  expect(specUrl, log.join('')).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v0\/spec\?token=[A-Za-z0-9_-]{22,}$/);
  const lines = fs.readFileSync(results, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  expect(lines.at(-1), JSON.stringify(lines)).toMatchObject({ done: true, failed: 0 });
  return { proc, specUrl, root, log };
}

function stopApp(app: App): Promise<void> {
  return new Promise((ok) => {
    if (app.proc.exitCode !== null) return ok();
    app.proc.once('exit', () => ok());
    app.proc.kill('SIGTERM');
  });
}

test('a saved spec applet opens in the browser from the running app', async ({ page }) => {
  const root = path.join(WEB, 'test-results', `spec-e2e-${process.pid}`);
  fs.rmSync(root, { recursive: true, force: true });
  fs.mkdirSync(root, { recursive: true });
  const app = await startApp(root);
  try {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${app.specUrl}&applet=run42-dashboard`);
    const s = await settled(page, 3);
    expect(s.status).toBe('connected');
    expect(s.title).toBe('Run 42');
    expect(s.panels.map((p) => [p.id, p.primitive, p.drawn])).toEqual([
      ['curves', 'series', true], ['attention', 'heatmap', true], ['rows', 'table', true]]);
    expect(s.panels[0].title).toBe('Loss and accuracy');
    expect(s.panels[2].title).toBe('run42_metrics');
    // The layout of the native runtime over the page's area: 1440 x 850 under the bar, over the status strip.
    expect(s.panels.map((p) => p.rect)).toEqual([
      { x: 0, y: 0, w: 720, h: 552 }, { x: 721, y: 0, w: 719, h: 552 }, { x: 0, y: 553, w: 1440, h: 297 }]);
    if (!DATA) {
      expect(s.panels[0].info).toBe('200 steps  2 lines');
      expect(s.panels[1].info).toBe('8×8  0 … 1');
      expect(s.panels[2].info).toBe('200 rows');
    }
    await expect(page.locator('#status')).toContainText('3 panels');
    // The page's token opens nothing beyond the page and its reads.
    const token = new URL(app.specUrl).searchParams.get('token')!;
    const base = new URL(app.specUrl).origin;
    const info = await page.evaluate(async ([base, token]) =>
      (await fetch(`${base}/v0/info`, { headers: { Authorization: `Bearer ${token}` } })).status, [base, token]);
    expect(info).toBe(403);
    const removed = await page.evaluate(async ([base, token]) => {
      const r = await fetch(`${base}/v0/commands/panel.remove`, { method: 'POST', body: '{"id":"rows"}',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } });
      return r.status;
    }, [base, token]);
    expect(removed).toBe(403);
    if (SHOTS) {
      fs.mkdirSync(SHOTS, { recursive: true });
      for (const [w, h] of [[2560, 1440], [1440, 900]]) {
        await page.setViewportSize({ width: w, height: h });
        await page.waitForFunction(([w, h]) => {
          const c = document.getElementById('canvas') as HTMLCanvasElement;
          return c.width === w && c.height === h - 50;
        }, [w, h]);
        await settled(page, 3);
        await page.screenshot({ path: path.join(SHOTS, `web-dashboard-${w}x${h}.png`) });
      }
    }
    expect(errors).toEqual([]);
    expect((await pageState(page)).panels.every((p) => p.drawn)).toBe(true);
  } finally {
    await stopApp(app);
  }
  expect(app.proc.exitCode, app.log.join('')).toBe(0);
  fs.rmSync(root, { recursive: true, force: true });
});
