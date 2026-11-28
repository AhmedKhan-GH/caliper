// The spec page's entry, served by Caliper API at
// /v0/spec?token=<token>&instrument=<id>: it reads the token and the
// instrument's id from its own address and draws that saved instrument.

import { SpecPage } from './page.js';

const params = new URLSearchParams(location.search);
const byId = (id: string) => document.getElementById(id) as HTMLElement;
const page = new SpecPage(
  { title: byId('title'), area: byId('area'), canvas: byId('canvas') as HTMLCanvasElement,
    overlay: byId('overlay') as HTMLCanvasElement, status: byId('status'), list: byId('list'),
    banner: byId('banner'), lock: byId('lock') as HTMLButtonElement },
  new URL(location.href), params.get('token') ?? '', params.get('instrument') ?? '',
);
(window as unknown as { caliperSpec: SpecPage }).caliperSpec = page;
page.start();
