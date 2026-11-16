// The anywidget module of the controls panel (caliper.controls()): a
// <caliper-controls> element fed by the stream's WebSocket directly, not by
// Jupyter's message channel, so it works while a cell runs. Traits: src and
// height.

import { defineControls } from './controls.js';

interface Model {
  get(name: string): unknown;
  on(event: string, cb: () => void): void;
  off(event?: string | null, cb?: (() => void) | null): void;
}

function render({ model, el }: { model: Model; el: HTMLElement }): () => void {
  defineControls();
  const panel = document.createElement('caliper-controls');
  const sync = () => {
    panel.setAttribute('src', String(model.get('src') ?? ''));
    const h = Number(model.get('height') ?? 0);
    if (h > 0) panel.setAttribute('height', String(h));
    else panel.removeAttribute('height');
  };
  sync();
  for (const t of ['src', 'height']) model.on(`change:${t}`, sync);
  el.append(panel);
  return () => {
    for (const t of ['src', 'height']) model.off(`change:${t}`, sync);
    panel.remove();
  };
}

export default { render };
