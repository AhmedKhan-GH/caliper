// The anywidget module of caliper.widget(): it creates a <caliper-view> fed by
// the WebSocket stream directly (not by Jupyter's message channel) and writes
// the user's selection and data view back to the traits.

import { define } from './element.js';
import type { DataView } from './protocol/messages.js';

interface Model {
  get(name: string): unknown;
  set(name: string, value: unknown): void;
  save_changes(): void;
  on(event: string, cb: () => void): void;
  off(event?: string | null, cb?: (() => void) | null): void;
}

function render({ model, el }: { model: Model; el: HTMLElement }): () => void {
  define();
  const view = document.createElement('caliper-view') as HTMLElement & { dataView: DataView | null };
  const sync = () => {
    view.setAttribute('src', String(model.get('src') ?? ''));
    view.setAttribute('view', String(model.get('view') ?? ''));
    const kind = model.get('kind');
    if (kind) view.setAttribute('kind', String(kind));
    else view.removeAttribute('kind');
    view.setAttribute('height', String(model.get('height') ?? 320));
    // A selection in one widget highlights the same items in the others (3.1).
    view.setAttribute('link', '');
  };
  sync();
  const dv = model.get('data_view') as DataView | null;
  if (dv && Object.keys(dv).length) view.dataView = dv;
  const onTraits = () => sync();
  const onDataView = () => {
    const next = model.get('data_view') as DataView | null;
    if (next && Object.keys(next).length) view.dataView = next;
  };
  for (const t of ['src', 'view', 'kind', 'height']) model.on(`change:${t}`, onTraits);
  model.on('change:data_view', onDataView);
  view.addEventListener('caliper-select', (e) => {
    model.set('selection', (e as CustomEvent).detail);
    model.save_changes();
  });
  view.addEventListener('caliper-dataview', (e) => {
    model.set('data_view', (e as CustomEvent).detail);
    model.save_changes();
  });
  el.append(view);
  return () => {
    for (const t of ['src', 'view', 'kind', 'height']) model.off(`change:${t}`, onTraits);
    model.off('change:data_view', onDataView);
    view.remove();
  };
}

export default { render };
