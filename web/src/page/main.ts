// The viewer page served at http://127.0.0.1:<port>/?token=<token>: it reads
// the session token from its own URL and connects to /v3/stream on its host.

import { connect } from './caliper-view.js';
import { STREAM_PATH, TOKEN_PARAM } from '../protocol/messages.js';
import { mountViewer } from './viewer.js';

const params = new URLSearchParams(location.search);
const url = new URL(STREAM_PATH, location.href);
url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
url.search = '';
url.searchParams.set(TOKEN_PARAM, params.get(TOKEN_PARAM) ?? '');
const session = params.get('session');
if (session) url.searchParams.set('session', session);
const src = url.toString();

const byId = (id: string) => document.getElementById(id) as HTMLElement;
mountViewer(connect(src), {
  grid: byId('grid'), controls: byId('controls'), script: byId('script'), state: byId('state'), empty: byId('empty'),
}, (el) => el.setAttribute('src', src));
