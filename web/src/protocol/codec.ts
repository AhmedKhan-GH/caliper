// The binary codec of wire protocol v3 (include/caliper/live_net.h,
// python/caliper/_wire.py). One frame is a 24-byte little-endian header,
// section_count uint32 section lengths, a UTF-8 JSON metadata object, then the
// payload sections, each at an offset from the frame start that is a multiple
// of 8. This module is the only one that knows the bytes.

import {
  CLOSE, DTYPE, DTYPE_SIZE, LIMITS, MSG, PROTOCOL_MAJOR, PROTOCOL_MINOR,
  type DataView as DataViewSpec, type ErrorMsg, type Message, type MessageType, type PayloadAxis,
  type SessionIdentity, type SnapshotMsg, type UnknownMsg, type ViewEntry,
} from './messages.js';

export const MAGIC = 0x574c4143; // "CALW" read as a little-endian uint32
export const HEADER_BYTES = 24;
export const SECTION_ALIGN = 8;

const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const utf8 = new TextEncoder();
const utf8Strict = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

/** A frame that is not well formed; the receiver closes with closeCode. */
export class WireError extends Error {
  constructor(readonly reason: string, message: string, readonly closeCode: number = CLOSE.protocolError) {
    super(`${reason}: ${message}`);
  }
}

/** A well-formed frame whose message is invalid; answered with an ERROR. */
export class MessageError extends Error {
  constructor(readonly code: string, message: string, readonly messageType = 0) {
    super(`${code}: ${message}`);
  }
}

export interface Section {
  name: string;
  dtype: number;
  shape: number[];
  encoding: number;
  bytes: Uint8Array;
}

export interface Frame {
  type: number;
  flags: number;
  major: number;
  minor: number;
  meta: Record<string, unknown>;
  sections: Section[];
}

const align = (n: number) => (n + SECTION_ALIGN - 1) & ~(SECTION_ALIGN - 1);

function depth(v: unknown, d = 1): number {
  if (Array.isArray(v)) return Math.max(d, ...v.map((x) => depth(x, d + 1)));
  if (v !== null && typeof v === 'object') return Math.max(d, ...Object.values(v).map((x) => depth(x, d + 1)));
  return d - 1;
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);
const isUint = (v: unknown): v is number => isInt(v) && v >= 0;

/** Decodes one frame, applying the header's checks in their order. */
export function decodeFrame(buf: ArrayBuffer | Uint8Array): Frame {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.byteLength;
  if (n < 8) throw new WireError('truncated', `${n} bytes, fewer than 8`);
  if (dv.getUint32(0, true) !== MAGIC) throw new WireError('bad_magic', `magic 0x${dv.getUint32(0, true).toString(16)}`);
  const major = dv.getUint16(4, true);
  if (major !== PROTOCOL_MAJOR) throw new WireError('unsupported_major', `major ${major}`, CLOSE.unsupportedMajor);
  if (n < HEADER_BYTES) throw new WireError('truncated', `${n} bytes, fewer than ${HEADER_BYTES}`);
  const minor = dv.getUint16(6, true);
  const type = dv.getUint16(8, true);
  const flags = dv.getUint16(10, true);
  const frameLen = dv.getUint32(12, true);
  const metaLen = dv.getUint32(16, true);
  const count = dv.getUint16(20, true);
  if (frameLen > LIMITS.maxFrameBytes || metaLen > LIMITS.maxMetaBytes || count > LIMITS.maxSections)
    throw new WireError('too_big', `frame_len ${frameLen}, meta_len ${metaLen}, ${count} sections`, CLOSE.tooBig);
  if (n !== frameLen) throw new WireError(n < frameLen ? 'truncated' : 'malformed', `message of ${n} bytes, frame_len ${frameLen}`);
  const tableEnd = HEADER_BYTES + 4 * count;
  if (tableEnd > frameLen) throw new WireError('malformed', 'section table past the end of the frame');
  const lens: number[] = [];
  for (let i = 0; i < count; i++) lens.push(dv.getUint32(HEADER_BYTES + 4 * i, true));
  const metaEnd = tableEnd + metaLen;
  let pos = metaEnd;
  const offsets: number[] = [];
  for (const len of lens) {
    pos = align(pos);
    offsets.push(pos);
    pos += len;
  }
  if (pos !== frameLen) throw new WireError('malformed', `layout ends at ${pos}, frame_len ${frameLen}`);
  let meta: unknown;
  try {
    meta = JSON.parse(utf8Strict.decode(bytes.subarray(tableEnd, metaEnd)));
  } catch (e) {
    throw new WireError('malformed', `metadata: ${(e as Error).message}`);
  }
  if (meta === null || typeof meta !== 'object' || Array.isArray(meta)) throw new WireError('malformed', 'metadata is not a JSON object');
  if (depth(meta) > LIMITS.maxJsonDepth) throw new WireError('malformed', `metadata nested deeper than ${LIMITS.maxJsonDepth}`);
  const m = meta as Record<string, unknown>;
  const sections: Section[] = [];
  const descs = m.sections;
  if (count) {
    if (!Array.isArray(descs) || descs.length !== count) throw new WireError('malformed', `'sections' does not describe ${count} sections`);
    const names = new Set<string>();
    descs.forEach((d, i) => {
      if (d === null || typeof d !== 'object' || Array.isArray(d)) throw new WireError('malformed', 'section descriptor is not an object');
      const { name, dtype, shape } = d as Record<string, unknown>;
      const encoding = (d as Record<string, unknown>).encoding ?? 0;
      if (typeof name !== 'string' || !isInt(dtype) || !isInt(encoding) || !Array.isArray(shape) || !shape.every(isUint))
        throw new WireError('malformed', `section descriptor ${JSON.stringify(d)}`);
      if (names.has(name)) throw new WireError('malformed', `section name '${name}' repeated`);
      names.add(name);
      const size = DTYPE_SIZE[dtype];
      if (encoding === 0 && size !== undefined) {
        const want = (shape as number[]).reduce((a, b) => a * b, 1) * size;
        if (want !== lens[i]) throw new WireError('malformed', `section '${name}': ${lens[i]} bytes, shape [${shape}] needs ${want}`);
      }
      sections.push({ name, dtype, shape: shape as number[], encoding, bytes: bytes.subarray(offsets[i], offsets[i] + lens[i]) });
    });
  } else if (descs !== undefined && descs !== null && !(Array.isArray(descs) && descs.length === 0)) {
    throw new WireError('malformed', "'sections' given for a frame without sections");
  }
  return { type, flags, major, minor, meta: m, sections };
}

/** The canonical metadata text: compact, keys sorted, no NaN. */
export function canonicalJson(v: unknown): string {
  if (v === null || typeof v !== 'object') {
    if (typeof v === 'number' && !Number.isFinite(v)) throw new TypeError('NaN and the infinities are not JSON');
    return JSON.stringify(v);
  }
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`;
  const keys = Object.keys(v).filter((k) => (v as Record<string, unknown>)[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson((v as Record<string, unknown>)[k])}`).join(',')}}`;
}

/** Encodes one frame. Sections are raw arrays; meta.sections is written here. */
export function encodeFrame(type: number, meta: Record<string, unknown>, sections: Section[] = []): Uint8Array {
  const m: Record<string, unknown> = { ...meta };
  delete m.sections;
  if (sections.length) {
    m.sections = sections.map((s) => ({ name: s.name, dtype: s.dtype, shape: s.shape, ...(s.encoding ? { encoding: s.encoding } : {}) }));
  }
  const mbytes = utf8.encode(canonicalJson(m));
  if (mbytes.length > LIMITS.maxMetaBytes) throw new WireError('too_big', `metadata of ${mbytes.length} bytes`, CLOSE.tooBig);
  let pos = HEADER_BYTES + 4 * sections.length + mbytes.length;
  const offsets: number[] = [];
  for (const s of sections) {
    pos = align(pos);
    offsets.push(pos);
    pos += s.bytes.byteLength;
  }
  const out = new Uint8Array(pos);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, MAGIC, true);
  dv.setUint16(4, PROTOCOL_MAJOR, true);
  dv.setUint16(6, PROTOCOL_MINOR, true);
  dv.setUint16(8, type, true);
  dv.setUint16(10, 0, true);
  dv.setUint32(12, pos, true);
  dv.setUint32(16, mbytes.length, true);
  dv.setUint16(20, sections.length, true);
  dv.setUint16(22, 0, true);
  sections.forEach((s, i) => dv.setUint32(HEADER_BYTES + 4 * i, s.bytes.byteLength, true));
  out.set(mbytes, HEADER_BYTES + 4 * sections.length);
  sections.forEach((s, i) => out.set(s.bytes, offsets[i]));
  return out;
}

// --- typed views of sections ------------------------------------------------

function f32(s: Section): Float32Array {
  const n = s.bytes.byteLength / 4;
  if (LITTLE && s.bytes.byteOffset % 4 === 0) return new Float32Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Float32Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(4 * i, true);
  return out;
}

function u8(s: Section): Uint8Array {
  return s.bytes;
}

function f32Section(name: string, arr: Float32Array, shape: number[]): Section {
  return { name, dtype: DTYPE.f32, shape, encoding: 0, bytes: new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength) };
}

// --- message checks -----------------------------------------------------------

function bad(type: number, msg: string): never {
  throw new MessageError('bad_request', msg, type);
}

function req<T>(f: Frame, key: string, ok: (v: unknown) => boolean, what: string): T {
  if (!(key in f.meta)) bad(f.type, `missing '${key}'`);
  const v = f.meta[key];
  if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
  return v as T;
}

function opt<T>(f: Frame, key: string, ok: (v: unknown) => boolean, what: string, dflt: T): T {
  const v = f.meta[key];
  if (v === undefined || v === null) return dflt;
  if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
  return v as T;
}

const isU32 = (v: unknown) => isInt(v) && v >= 0 && v <= 0xffffffff;
const isNum = (v: unknown) => typeof v === 'number';
const isStr = (v: unknown) => typeof v === 'string';
const isObj = (v: unknown) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isStrs = (v: unknown) => Array.isArray(v) && v.every(isStr);
const isName = (v: unknown) => typeof v === 'string' && v.length > 0 && utf8.encode(v).length <= 63;
const isSessionId = (v: unknown) => typeof v === 'string' && /^[A-Za-z0-9._-]{1,63}$/.test(v);
const isIdentity = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return isSessionId(o.id) && isStr(o.script) && isUint(o.pid) && isUint(o.start_unix_ms);
};
const isViewEntry = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  if (!isUint(o.id) || !isName(o.name) || !isInt(o.kind) || !Array.isArray(o.sources)) return false;
  for (const s of o.sources as unknown[]) {
    if (!isObj(s)) return false;
    const so = s as Record<string, unknown>;
    if (!isName(so.name) || !Array.isArray(so.dims)) return false;
    for (const d of so.dims as unknown[]) {
      if (!isObj(d) || !isName((d as Record<string, unknown>).name) || !isUint((d as Record<string, unknown>).size)) return false;
    }
  }
  return o.default === undefined || o.default === null || isObj(o.default);
};

function section(f: Frame, name: string): Section | undefined {
  return f.sections.find((s) => s.name === name);
}

function decodeSnapshot(f: Frame): SnapshotMsg {
  const t = f.type;
  const kind = req<number>(f, 'kind', isInt, 'a kind id');
  const data = section(f, 'data');
  if (!data) bad(t, "a snapshot needs a 'data' section");
  const auxSec = section(f, 'aux');
  const stats = opt<(number | null)[]>(f, 'stats', (v) => Array.isArray(v) && v.length <= 64 && v.every((x) => x === null || isNum(x)), 'a list of numbers or nulls', []);
  const pointsDims = opt<number | undefined>(f, 'points_dims', (v) => v === 2 || v === 3, '2 or 3', undefined);
  const want: Record<number, [number, number]> = { 1: [DTYPE.f32, 2], 2: [DTYPE.u8, 3], 3: [DTYPE.f32, 2], 5: [DTYPE.f32, 1] };
  const w = want[kind];
  if (w && (data.dtype !== w[0] || data.shape.length !== w[1] || data.encoding !== 0))
    bad(t, `kind ${kind}: 'data' is dtype ${data.dtype} shape [${data.shape}]`);
  if (kind === 2 && data.shape[2] !== 4) bad(t, "an image's 'data' is (H, W, 4)");
  let dims: 2 | 3 | undefined;
  let aux: Float32Array | undefined;
  if (kind === 3) {
    if ((data.shape[1] !== 2 && data.shape[1] !== 3) || (pointsDims !== undefined && pointsDims !== data.shape[1]))
      bad(t, "points 'data' is (N, points_dims) with points_dims 2 or 3");
    dims = data.shape[1] as 2 | 3;
    if (auxSec) {
      if (auxSec.dtype !== DTYPE.f32 || auxSec.shape.length !== 1 || auxSec.shape[0] !== data.shape[0]) bad(t, "points 'aux' is (N,) f32");
      aux = f32(auxSec);
    }
  }
  return {
    type: 'snapshot',
    sub: req(f, 'sub', isU32, 'a uint32'),
    view: req(f, 'view', isName, 'a view name'),
    seq: req(f, 'seq', isUint, 'an unsigned integer'),
    stamp_ns: req(f, 'stamp_ns', isUint, 'an unsigned integer'),
    kind,
    data_view: req<DataViewSpec>(f, 'data_view', isObj, 'an object'),
    data: data.dtype === DTYPE.u8 ? u8(data) : data.dtype === DTYPE.f32 ? f32(data) : new Float32Array(0),
    shape: data.shape,
    aux,
    colormap: opt(f, 'colormap', isInt, 'an integer', 0),
    flags: opt(f, 'flags', isUint, 'an unsigned integer', 0),
    vmin: opt(f, 'vmin', isNum, 'a number', 0),
    vmax: opt(f, 'vmax', isNum, 'a number', 1),
    stats: stats.map((x) => (x === null ? NaN : x)),
    axes: opt<PayloadAxis[]>(f, 'axes', Array.isArray, 'a list', []),
    tile: opt(f, 'tile', isObj, 'an object', null),
    points_dims: dims,
    label_count: opt<number | undefined>(f, 'label_count', isUint, 'an unsigned integer', undefined),
  };
}

/** Decodes one message. Throws WireError or MessageError. */
export function decode(buf: ArrayBuffer | Uint8Array): Message | UnknownMsg {
  const f = decodeFrame(buf);
  const t = f.type;
  switch (t) {
    case MSG.hello:
      return {
        type: 'hello',
        role: req(f, 'role', (v) => v === 'viewer' || v === 'producer', 'viewer or producer'),
        capabilities: opt(f, 'capabilities', isStrs, 'a list of strings', []),
        software: opt(f, 'software', isObj, 'an object', undefined),
        session: opt(f, 'session', isIdentity, 'a session identity', undefined),
      };
    case MSG.welcome:
      return {
        type: 'welcome',
        role: req(f, 'role', (v) => v === 'producer' || v === 'hub', 'producer or hub'),
        capabilities: opt(f, 'capabilities', isStrs, 'a list of strings', []),
        software: opt(f, 'software', isObj, 'an object', undefined),
        session: opt<SessionIdentity | null>(f, 'session', isIdentity, 'a session identity', null),
        minor: f.minor,
      };
    case MSG.bye:
      return { type: 'bye', reason: req(f, 'reason', isStr, 'a string'), message: opt(f, 'message', isStr, 'a string', '') };
    case MSG.error:
      return {
        type: 'error',
        code: req(f, 'code', isStr, 'a string'),
        message: opt(f, 'message', isStr, 'a string', ''),
        fatal: opt(f, 'fatal', (v) => typeof v === 'boolean', 'a boolean', false),
        in_reply_to: opt(f, 'in_reply_to', isU32, 'a message type', undefined),
        request_id: opt(f, 'request_id', isU32, 'a uint32', undefined),
        sub: opt(f, 'sub', isU32, 'a uint32', undefined),
        view: opt(f, 'view', isStr, 'a string', undefined),
        session: opt(f, 'session', isStr, 'a string', undefined),
      };
    case MSG.ping:
    case MSG.pong:
      return { type: t === MSG.ping ? 'ping' : 'pong', nonce: req(f, 'nonce', isU32, 'a uint32') };
    case MSG.sessions:
      return {
        type: 'sessions',
        sessions: req(f, 'sessions', (v) => Array.isArray(v) && v.every((s) => isIdentity(s)
          && ((s as SessionIdentity).state === 'live' || (s as SessionIdentity).state === 'ended')
          && isUint((s as SessionIdentity).views)), 'a list of sessions'),
      };
    case MSG.attach:
      if (!('session' in f.meta)) bad(t, "missing 'session'");
      return { type: 'attach', session: opt<string | null>(f, 'session', isSessionId, 'a session id or null', null) };
    case MSG.views:
      return {
        type: 'views',
        views: req<ViewEntry[]>(f, 'views', (v) => Array.isArray(v) && v.every(isViewEntry), 'a list of views'),
        session: opt<string | null>(f, 'session', isSessionId, 'a session id', null),
      };
    case MSG.subscribe:
      return {
        type: 'subscribe',
        sub: req(f, 'sub', isU32, 'a uint32'),
        view: req(f, 'view', isName, 'a view name'),
        request_id: req(f, 'request_id', isU32, 'a uint32'),
        max_rate: opt(f, 'max_rate', (v) => isNum(v) && (v as number) > 0, 'a positive number', undefined),
        data_view: opt(f, 'data_view', isObj, 'an object', undefined),
      };
    case MSG.unsubscribe:
      return { type: 'unsubscribe', sub: req(f, 'sub', isU32, 'a uint32') };
    case MSG.view_request:
      return {
        type: 'view_request',
        request_id: req(f, 'request_id', isU32, 'a uint32'),
        sub: req(f, 'sub', isU32, 'a uint32'),
        view: req(f, 'view', isName, 'a view name'),
        data_view: req(f, 'data_view', isObj, 'an object'),
      };
    case MSG.view_reply: {
      const dv = opt<DataViewSpec | undefined>(f, 'data_view', isObj, 'an object', undefined);
      const err = opt<{ code: string } | undefined>(f, 'error', (v) => isObj(v) && isStr((v as Record<string, unknown>).code), '{code, message}', undefined);
      if ((dv === undefined) === (err === undefined)) bad(t, "a view_reply carries exactly one of 'data_view' and 'error'");
      return {
        type: 'view_reply',
        request_id: req(f, 'request_id', isU32, 'a uint32'),
        sub: req(f, 'sub', isU32, 'a uint32'),
        view: req(f, 'view', isName, 'a view name'),
        data_view: dv,
        error: err,
      };
    }
    case MSG.snapshot:
      return decodeSnapshot(f);
    case MSG.series: {
      const v = section(f, 'values');
      if (!v || v.dtype !== DTYPE.f32 || v.shape.length !== 1 || v.encoding !== 0) bad(t, "a series needs a (M,) f32 'values' section");
      const start = req<number>(f, 'start', isUint, 'an unsigned integer');
      const total = req<number>(f, 'total', isUint, 'an unsigned integer');
      if (v.shape[0] < 1 || v.shape[0] > 262144 || start + v.shape[0] > total)
        bad(t, `series: ${v.shape[0]} values from ${start} with total ${total}`);
      return {
        type: 'series',
        sub: req(f, 'sub', isU32, 'a uint32'),
        view: req(f, 'view', isName, 'a view name'),
        start,
        total,
        stamp_ns: req(f, 'stamp_ns', isUint, 'an unsigned integer'),
        values: f32(v),
      };
    }
    case MSG.event: {
      const detail = req<Record<string, unknown>>(f, 'detail', isObj, 'an object');
      for (const key of ['indices', 'row_ids']) {
        const lst = detail[key];
        if (lst !== undefined && lst !== null && !(Array.isArray(lst) && lst.length <= LIMITS.maxEventIndices && lst.every(isUint)))
          bad(t, `detail.${key} must be at most ${LIMITS.maxEventIndices} unsigned integers`);
      }
      return {
        type: 'event',
        view: req(f, 'view', isName, 'a view name'),
        event: req(f, 'type', isStr, 'a string'),
        detail,
        sub: opt(f, 'sub', isU32, 'a uint32', undefined),
        seq: opt(f, 'seq', isUint, 'an integer', undefined),
        data_view: opt(f, 'data_view', isObj, 'an object', undefined),
      };
    }
    default:
      return { type: 'unknown', id: t, meta: f.meta };
  }
}

const drop = (o: Record<string, unknown>) => {
  for (const k of Object.keys(o)) if (o[k] === undefined || o[k] === null) delete o[k];
  return o;
};

/** Encodes one message into the bytes of one WebSocket message. */
export function encode(msg: Message): Uint8Array {
  const id = MSG[msg.type as MessageType];
  switch (msg.type) {
    case 'hello':
    case 'welcome':
      return encodeFrame(id, drop({ role: msg.role, capabilities: msg.capabilities, software: msg.software, session: msg.session }));
    case 'bye':
      return encodeFrame(id, { reason: msg.reason, message: msg.message });
    case 'error': {
      const e: ErrorMsg = msg;
      return encodeFrame(id, drop({
        code: e.code, message: e.message, fatal: e.fatal, in_reply_to: e.in_reply_to, request_id: e.request_id,
        sub: e.sub, view: e.view, session: e.session,
      }));
    }
    case 'ping':
    case 'pong':
      return encodeFrame(id, { nonce: msg.nonce });
    case 'sessions':
      return encodeFrame(id, { sessions: msg.sessions });
    case 'attach':
      return encodeFrame(id, { session: msg.session });
    case 'views':
      return encodeFrame(id, { session: msg.session ?? null, views: msg.views });
    case 'subscribe':
      return encodeFrame(id, drop({ sub: msg.sub, view: msg.view, request_id: msg.request_id, max_rate: msg.max_rate, data_view: msg.data_view }));
    case 'unsubscribe':
      return encodeFrame(id, { sub: msg.sub });
    case 'view_request':
      return encodeFrame(id, { request_id: msg.request_id, sub: msg.sub, view: msg.view, data_view: msg.data_view });
    case 'view_reply':
      return encodeFrame(id, drop({ request_id: msg.request_id, sub: msg.sub, view: msg.view, data_view: msg.data_view, error: msg.error }));
    case 'event':
      return encodeFrame(id, drop({ view: msg.view, type: msg.event, detail: msg.detail, sub: msg.sub, seq: msg.seq, data_view: msg.data_view }));
    case 'series':
      return encodeFrame(id, { sub: msg.sub, view: msg.view, start: msg.start, total: msg.total, stamp_ns: msg.stamp_ns },
        [f32Section('values', msg.values, [msg.values.length])]);
    case 'snapshot': {
      const s = msg;
      const sections: Section[] = [{
        name: 'data', dtype: s.data instanceof Uint8Array ? DTYPE.u8 : DTYPE.f32, shape: s.shape, encoding: 0,
        bytes: new Uint8Array(s.data.buffer, s.data.byteOffset, s.data.byteLength),
      }];
      if (s.aux) sections.push(f32Section('aux', s.aux, [s.aux.length]));
      return encodeFrame(id, drop({
        sub: s.sub, view: s.view, seq: s.seq, stamp_ns: s.stamp_ns, kind: s.kind, data_view: s.data_view, colormap: s.colormap,
        flags: s.flags, vmin: s.vmin, vmax: s.vmax, stats: s.stats.map((x) => (Number.isFinite(x) ? x : null)), axes: s.axes,
        tile: s.tile, points_dims: s.points_dims, label_count: s.label_count,
      }), sections);
    }
  }
}
