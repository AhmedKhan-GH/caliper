// What a live view (heatmap, image, points, series, histogram) gets from the
// element that hosts it, and what it must provide.

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
  /** WebGPU device, or null to draw with Canvas 2D. */
  readonly device: GPUDevice | null;
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

export interface LiveView {
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
