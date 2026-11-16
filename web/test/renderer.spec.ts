// The choice between the WebAssembly views and the element's own views,
// against a stub of the WebAssembly loader served at the stream's origin.

import { expect, serve, test } from './util.js';

type Stub = { given: Record<string, unknown>; handle: { dataViews: unknown[]; destroyed: boolean; onSelect: (e: unknown) => void; onDataView: (d: unknown) => void }; hasCallbacks: boolean };

async function make(page: import('@playwright/test').Page, src: string, renderer: string | null): Promise<void> {
  await page.evaluate(({ src, renderer }) => {
    const el = document.createElement('caliper-view');
    el.id = 'v';
    el.setAttribute('src', src);
    el.setAttribute('view', 'loss');
    el.setAttribute('kind', 'series');
    el.setAttribute('height', '240');
    if (renderer) el.setAttribute('renderer', renderer);
    document.getElementById('root')!.append(el);
  }, { src, renderer });
}

const rendererOf = (page: import('@playwright/test').Page) =>
  page.waitForFunction(() => (document.getElementById('v') as HTMLElement & { renderer: string | null }).renderer).then((h) => h.jsonValue());

test('auto uses the WebAssembly views when an adapter and the loader are there, and the TypeScript views otherwise', async ({ harness, server }, info) => {
  const port = new URL(server).port;
  await make(harness, `ws://127.0.0.1:${port}/v3/stream?token=t`, null);
  const r = await rendererOf(harness);
  if (info.project.name === 'canvas2d') {
    expect(r).toBe('canvas2d');
    expect(await harness.evaluate(() => 'caliperStub' in window)).toBe(false);
    return;
  }
  expect(r).toBe('wasm');
  const stub = await harness.evaluate(() => (window as unknown as { caliperStub: Stub }).caliperStub);
  // The element keeps its header and asks for a view without a title bar of its own.
  expect(stub.given).toEqual({ src: `ws://127.0.0.1:${port}/v3/stream?token=t`, view: 'loss', kind: 'series', header: false, assetBase: `http://127.0.0.1:${port}/` });
  expect(stub.hasCallbacks).toBe(true);
  expect(await harness.evaluate(() => {
    const root = document.getElementById('v')!.shadowRoot!;
    return [(root.querySelector('.plate') as HTMLElement).hidden, root.querySelector('.name')?.textContent,
            root.querySelector('.body .wasm .stub')?.textContent];
  })).toEqual([false, 'loss', 'wasm views: loss']);
  // The data view property reaches the handle; a selection becomes the element's event, and the
  // handle's data view reply sets the element's data view without an event (the user picked nothing).
  const events = await harness.evaluate(() => {
    const el = document.getElementById('v') as HTMLElement & { dataView: unknown };
    const seen: unknown[] = [];
    el.addEventListener('caliper-select', (e) => seen.push(['select', (e as CustomEvent).detail]));
    el.addEventListener('caliper-dataview', (e) => seen.push(['dataview', (e as CustomEvent).detail]));
    el.dataView = { budget: { max_backfill: 10 } };
    const h = (window as unknown as { caliperStub: Stub }).caliperStub.handle;
    h.onSelect({ steps: [3, 9] });
    h.onDataView({ budget: { max_backfill: 10 } });
    return { seen, dataViews: h.dataViews, dataView: el.dataView };
  });
  expect(events.dataViews).toEqual([{ budget: { max_backfill: 10 } }]);
  expect(events.seen).toEqual([
    ['select', { view: 'loss', type: 'select', detail: { steps: [3, 9] } }],
  ]);
  expect(events.dataView).toEqual({ budget: { max_backfill: 10 } });
  await harness.evaluate(() => document.getElementById('v')!.remove());
  expect(await harness.evaluate(() => (window as unknown as { caliperStub: Stub }).caliperStub.handle.destroyed)).toBe(true);
});

test('renderer="ts" keeps the TypeScript views', async ({ harness, server }) => {
  await make(harness, `ws://127.0.0.1:${new URL(server).port}/v3/stream?token=t`, 'ts');
  expect(await rendererOf(harness)).toBe('canvas2d');
  expect(await harness.evaluate(() => 'caliperStub' in window)).toBe(false);
});

test('falls back to the TypeScript views when the loader throws', async ({ harness, server }) => {
  await harness.evaluate(() => Object.assign(window, { caliperStubFail: true }));
  await make(harness, `ws://127.0.0.1:${new URL(server).port}/v3/stream?token=t`, 'wasm');
  expect(await rendererOf(harness)).toBe('canvas2d');
  expect(await harness.evaluate(() => document.getElementById('v')!.shadowRoot!.querySelector('.stub'))).toBeNull();
});

test('falls back to the TypeScript views when the server has no loader', async ({ harness }) => {
  const empty = await serve('/nonexistent-root');
  try {
    await make(harness, `ws://127.0.0.1:${new URL(empty.origin).port}/v3/stream?token=t`, 'wasm');
    expect(await rendererOf(harness)).toBe('canvas2d');
  } finally {
    empty.close();
  }
});
