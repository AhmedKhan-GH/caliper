// The spec page's entry, served by Caliper API at
// /v0/spec?token=<token>&applet=<id>: it reads the token and the applet's id
// from its own address and draws that saved spec applet.

import { SpecPage } from './page.js';

const params = new URLSearchParams(location.search);
const byId = (id: string) => document.getElementById(id) as HTMLElement;
const page = new SpecPage(
  { title: byId('title'), area: byId('area'), canvas: byId('canvas') as HTMLCanvasElement, status: byId('status'), list: byId('list'),
    banner: byId('banner') },
  new URL(location.href), params.get('token') ?? '', params.get('applet') ?? '',
);
(window as unknown as { caliperSpec: SpecPage }).caliperSpec = page;
page.start();
