// The viewer page's logic: the script name and connection state from the
// stream, and one <caliper-view> plate per view in a grid. It uses the
// element only through the DOM, so the page bundle does not carry a second
// copy of it.

import type { StreamClient } from '../client/stream.js';
import { kindName, type ViewEntry } from '../protocol/messages.js';

export interface PageParts {
  grid: HTMLElement;
  script: HTMLElement;
  state: HTMLElement;
  empty: HTMLElement;
}

/** Fills the page from a client; attach() points a new element at the stream. */
export function mountViewer(client: StreamClient, parts: PageParts, attach: (el: HTMLElement) => void): () => void {
  const plates = new Map<string, HTMLElement>();
  const setState = () => {
    const s = client.state;
    const ended = s === 'ended' || client.session?.state === 'ended';
    const text = ended ? 'ended' : s === 'open' ? 'connected' : 'connecting';
    parts.state.dataset.state = text;
    const label = parts.state.querySelector('.text');
    if (label) label.textContent = text;
    parts.empty.textContent = ended ? 'The stream has ended.' : s === 'open' ? 'No views yet.' : 'Connecting to the stream.';
    parts.empty.hidden = plates.size > 0;
  };
  const setSession = () => {
    const name = client.session?.script ?? '';
    parts.script.textContent = name;
    parts.script.title = name;
    document.title = name ? `${name} · Caliper` : 'Caliper';
    setState();
  };
  const setViews = (views: readonly ViewEntry[]) => {
    const names = new Set(views.map((v) => v.name));
    for (const [name, el] of plates) {
      if (!names.has(name)) {
        el.remove();
        plates.delete(name);
      }
    }
    for (const v of views) {
      if (plates.has(v.name)) continue;
      const el = document.createElement('caliper-view');
      el.setAttribute('view', v.name);
      const kind = kindName(v.kind);
      if (kind) el.setAttribute('kind', kind);
      el.setAttribute('height', '340');
      attach(el);
      plates.set(v.name, el);
      parts.grid.append(el);
    }
    setState();
  };
  const off = client.listen({ onStatus: setState, onSession: setSession, onViews: setViews });
  client.connect();
  setSession();
  setViews(client.views);
  return off;
}
