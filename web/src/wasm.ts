// The adapter to Caliper's own view layer compiled to WebAssembly. Its loader
// module, caliper-views-loader.js, is served by the stream's server and
// exports mountCaliperViews(container, options) and CALIPER_VIEWS_KINDS, the
// one list of the live view kinds the WebAssembly views draw. Everything the
// element knows about it is in this file; any failure here makes the element
// draw with its own TypeScript views instead.

import type { DataView } from './protocol/messages.js';
import type { ViewFacts } from './views/info.js';

export interface WasmSelect {
  view: string;
  type: 'select';
  detail: Record<string, unknown>;
}

export interface WasmHandle {
  destroy(): void;
  setDataView(dataView: DataView): boolean | void;
  /** Resolves once the first snapshot is drawn (the real loader has it; a stub may not). */
  ready?: Promise<void>;
  /** The module's counters (the real loader has them; a stub may not). */
  readonly stats?: Record<string, unknown>;
  onSelect?: ((event: WasmSelect | Record<string, unknown>) => void) | null;
  /** Sets a header toggle of the view: 'follow' or 'log' (the real loader has it; a stub may not). */
  setToggle?(name: string, on: boolean): boolean;
  onDataView?: ((dataView: DataView) => void) | null;
}

export interface WasmOptions {
  src: string;
  view: string;
  kind?: string;
  height?: number;
  /** False: the element draws the view's header, so the WebAssembly view has no title bar. */
  header?: boolean;
  /** The HTTP origin of the stream's server, with a trailing slash. */
  assetBase: string;
  /** Every selection the module's views make; the module has already sent it to the producer. */
  onSelect(event: WasmSelect | Record<string, unknown>): void;
  /** What the element's header says about the view, as it changes. */
  onInfo?(facts: ViewFacts): void;
  /** A view changed a header toggle itself (a plot stopped or started following). */
  onToggle?(name: string, on: boolean): void;
  onDataView(dataView: DataView): void;
}

interface Loader {
  mountCaliperViews(container: HTMLElement, options: WasmOptions): WasmHandle | Promise<WasmHandle>;
  /** The kinds the WebAssembly views draw; a loader without the list is taken to draw every kind. */
  CALIPER_VIEWS_KINDS?: readonly string[];
}

export const LOADER_FILE = 'caliper-views-loader.js';

/** http://host:port/ for ws://host:port/..., or null. */
export function assetBase(src: string): string | null {
  try {
    const u = new URL(src);
    if (u.protocol !== 'ws:' && u.protocol !== 'wss:') return null;
    return `${u.protocol === 'wss:' ? 'https:' : 'http:'}//${u.host}/`;
  } catch {
    return null;
  }
}

const loaders = new Map<string, Promise<Loader | null>>();

function load(base: string): Promise<Loader | null> {
  let p = loaders.get(base);
  if (!p) {
    const url = base + LOADER_FILE;
    p = import(/* @vite-ignore */ url).then(
      (m: Partial<Loader>) => (typeof m.mountCaliperViews === 'function' ? (m as Loader) : null),
      () => null,
    );
    loaders.set(base, p);
  }
  return p;
}

/**
 * The live view kinds the WebAssembly views served from base draw (the
 * loader's CALIPER_VIEWS_KINDS), or null when the loader is missing.
 */
export async function wasmKinds(base: string): Promise<readonly string[] | null> {
  const loader = await load(base);
  if (!loader) return null;
  return loader.CALIPER_VIEWS_KINDS ?? ['heatmap', 'image', 'points', 'series', 'histogram'];
}

let adapter: Promise<boolean> | null = null;

/** True when navigator.gpu gives an adapter, which the WebAssembly views need. */
export function hasAdapter(): Promise<boolean> {
  if (!adapter) {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    adapter = gpu ? gpu.requestAdapter().then((a) => a !== null, () => false) : Promise.resolve(false);
  }
  return adapter;
}

/**
 * Mounts the WebAssembly views into container, or returns null when the
 * module is missing, refuses or throws; the caller then uses its own views.
 */
export async function mountWasm(container: HTMLElement, options: WasmOptions): Promise<WasmHandle | null> {
  const loader = await load(options.assetBase);
  if (!loader) return null;
  try {
    const handle = await loader.mountCaliperViews(container, options);
    if (!handle || typeof handle.destroy !== 'function') return null;
    // The handle may take its callbacks as properties as well as options.
    if ('onSelect' in handle) handle.onSelect = options.onSelect;
    if ('onDataView' in handle) handle.onDataView = options.onDataView;
    return handle;
  } catch (e) {
    console.warn('caliper-view: the WebAssembly views failed to mount; using the TypeScript views', e);
    container.replaceChildren();
    return null;
  }
}
