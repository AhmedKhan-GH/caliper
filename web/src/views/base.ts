// What a live view (heatmap, image, points, series, histogram) gets from the
// element that hosts it, and what it must provide. The views draw with
// Canvas 2D; the element's WebGPU drawing is the WebAssembly views' (wasm.ts).

import type { SeriesMsg, SnapshotMsg } from '../protocol/messages.js';

export interface ViewHost {
  /** The view's name, for labels. */
  readonly name: string;
  /** The drawing area; a live view adds its own data canvas here. */
  readonly body: HTMLElement;
  /** A 2-D canvas above the data canvas, for axes, hover and selections. */
  readonly overlay: HTMLCanvasElement;
  /** Header slot for the view's own toggles. */
  readonly tools: HTMLElement;
  /** CSS size of the body and the device pixel ratio. */
  size(): { w: number; h: number; dpr: number };
  /** Schedules a frame: 'data' redraws the data layer too, 'overlay' only the overlay. */
  request(what: 'data' | 'overlay'): void;
  /** Shows a tooltip at a body position, or hides it. */
  tooltip(text: string | null, x?: number, y?: number): void;
  /** Reports a selection, a pick or a hover to the page and to the producer. */
  report(type: 'select' | 'pick' | 'hover', detail: Record<string, unknown>): void;
  /** Sets the faint text of the header (shape, range). */
  info(text: string): void;
}

/** The axis of a view that carries an item space (wire protocol 3.1, linked selection). */
export type LinkAxis = 'points' | 'rows' | 'columns' | 'steps';

/** Whether sorted items hold one in the source range [a, b). */
export function linkedIn(items: readonly number[], a: number, b: number): boolean {
  let lo = 0, hi = items.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (items[m] < a) lo = m + 1;
    else hi = m;
  }
  return lo < items.length && items[lo] < b;
}

export interface LiveView {
  /** Draws a linked selection: the selected items of the space the axis carries (null: none). */
  link?(axis: LinkAxis | null, items: readonly number[]): void;
  /** Takes the newest snapshot (the previous one is dropped). */
  snapshot(s: SnapshotMsg): void;
  /** Appends a series delta (series views only). */
  series?(d: SeriesMsg, curve: number): void;
  /** Draws; data is true when the data layer must be redrawn. */
  draw(data: boolean): void;
  /** The body changed size. */
  resize(): void;
  destroy(): void;
  /** Read-only state for tests and debugging. */
  debug(): Record<string, unknown>;
}

/** A small toggle button for the header's tool slot (a view's, or the element's for the WebAssembly views). */
export function toggle(host: Pick<ViewHost, 'tools'>, label: string, title: string, on: boolean, change: (on: boolean) => void) {
  const b = document.createElement('button');
  b.className = 'tool';
  b.textContent = label;
  b.title = title;
  b.setAttribute('aria-pressed', String(on));
  b.addEventListener('click', () => {
    on = !on;
    b.setAttribute('aria-pressed', String(on));
    change(on);
  });
  host.tools.append(b);
  return {
    el: b,
    set(v: boolean) {
      on = v;
      b.setAttribute('aria-pressed', String(v));
    },
  };
}
