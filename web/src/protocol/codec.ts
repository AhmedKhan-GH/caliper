// The binary codec of wire protocol v3 (include/caliper/live_net.h,
// python/caliper/_wire.py). One frame is a 24-byte little-endian header,
// section_count uint32 section lengths, a UTF-8 JSON metadata object, then the
// payload sections, each at an offset from the frame start that is a multiple
// of 8. This module is the only one that knows the bytes.

import {
  CLOSE, CONTROL_OPS, DTYPE, DTYPE_SIZE, LIMITS, LIMITS_31, MSG, PROTOCOL_MAJOR, PROTOCOL_MINOR, RUN_STATES,
  type ActionInfo, type ControlOp, type ControlReplyMsg, type DataView as DataViewSpec, type ErrorMsg, type Message,
  type MessageType, type ParamInfo, type ParamValue, type PayloadAxis, type RunInfo, type SelectionDetails,
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

/**
 * Encodes one frame at a minor version (this codec's, 1, by default; 0
 * re-encodes a 3.0 frame byte for byte). Sections are raw arrays;
 * meta.sections is written here.
 */
export function encodeFrame(type: number, meta: Record<string, unknown>, sections: Section[] = [], minor: number = PROTOCOL_MINOR): Uint8Array {
  return frameAt(type, meta, sections, minor);
}

function frameAt(type: number, meta: Record<string, unknown>, sections: Section[], minor: number): Uint8Array {
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
  dv.setUint16(6, minor, true);
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
const optName = (v: unknown) => v === undefined || v === null || isName(v);
const isViewEntry = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  if (!isUint(o.id) || !isName(o.name) || !isInt(o.kind) || !Array.isArray(o.sources) || !optName(o.items)) return false;
  for (const s of o.sources as unknown[]) {
    if (!isObj(s)) return false;
    const so = s as Record<string, unknown>;
    if (!isName(so.name) || !Array.isArray(so.dims)) return false;
    for (const d of so.dims as unknown[]) {
      const dd = d as Record<string, unknown>;
      if (!isObj(d) || !isName(dd.name) || !isUint(dd.size) || !optName(dd.items)) return false;
    }
  }
  return o.default === undefined || o.default === null || isObj(o.default);
};

// --- 3.1 checks (python/caliper/_wire.py has the same) -----------------------------

const isBool = (v: unknown) => typeof v === 'boolean';
const isFinite = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
const isText = (v: unknown) => typeof v === 'string' && utf8.encode(v).length <= LIMITS_31.textMax;
const INT_EXACT = 2 ** 53;
const fitsType = (p: ParamInfo, v: unknown) => {
  if (p.type === 'number') return isFinite(v);
  if (p.type === 'integer') return isInt(v) && Math.abs(v) <= INT_EXACT;
  if (p.type === 'boolean') return isBool(v);
  return typeof v === 'string' && !!p.choices?.includes(v);
};
const isParam = (v: unknown) => {
  if (!isObj(v)) return false;
  const p = v as ParamInfo;
  if (!isName(p.name) || !['number', 'integer', 'boolean', 'choice'].includes(p.type)) return false;
  if (p.type === 'number' || p.type === 'integer') {
    const bound = p.type === 'number' ? isFinite : (x: unknown) => isInt(x) && Math.abs(x) <= INT_EXACT;
    for (const x of [p.min, p.max]) if (x !== undefined && x !== null && !bound(x)) return false;
    if (p.min != null && p.max != null && p.min > p.max) return false;
    if (p.log != null && !isBool(p.log)) return false;
    if (p.log && !(p.min != null && p.min > 0 && p.max != null)) return false;
  } else if (p.type === 'choice') {
    const ch = p.choices;
    if (!Array.isArray(ch) || ch.length < 1 || ch.length > LIMITS_31.maxChoices || !ch.every(isName) || new Set(ch).size !== ch.length) return false;
  }
  if (p.help != null && !isText(p.help)) return false;
  return 'value' in p && fitsType(p, p.value) && (p.pending == null || fitsType(p, p.pending));
};
const isAction = (v: unknown) => {
  if (!isObj(v)) return false;
  const a = v as ActionInfo;
  return isName(a.name) && isBool(a.running) && isBool(a.queued) && (a.help == null || isText(a.help));
};
const isRun = (v: unknown) => {
  if (!isObj(v)) return false;
  const r = v as RunInfo;
  return RUN_STATES.includes(r.state) && (r.step == null || isUint(r.step)) && (r.steps_left == null || isUint(r.steps_left));
};
const uniqueNames = (v: { name: string }[]) => new Set(v.map((x) => x.name)).size === v.length;
const isSetValue = (v: unknown) => isFinite(v) || isBool(v) || isName(v);
const isAscending = (v: number[]) => v.every((x, i) => i === 0 || v[i - 1] < x);
const isCell = (v: unknown) => v === null || isBool(v) || isFinite(v) || (typeof v === 'string' && utf8.encode(v).length <= LIMITS_31.detailTextMax);
const isDetails = (v: unknown) => {
  if (!isObj(v)) return false;
  const d = v as SelectionDetails;
  return Array.isArray(d.fields) && d.fields.length <= LIMITS_31.maxDetailFields && d.fields.every(isName)
    && new Set(d.fields).size === d.fields.length
    && Array.isArray(d.items) && d.items.length <= LIMITS_31.maxDetailItems && d.items.every(isUint)
    && Array.isArray(d.values) && d.values.length === d.items.length
    && d.values.every((row) => Array.isArray(row) && row.length === d.fields.length && row.every(isCell));
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
        control: opt<boolean | undefined>(f, 'control', isBool, 'a boolean', undefined),
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
          && isUint((s as SessionIdentity).views)
          && ((s as SessionIdentity).capabilities == null || isStrs((s as SessionIdentity).capabilities))), 'a list of sessions'),
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
    case MSG.controls:
      return {
        type: 'controls',
        run: req(f, 'run', isRun, 'a run state'),
        params: req(f, 'params', (v) => Array.isArray(v) && v.length <= LIMITS_31.maxParams && v.every(isParam) && uniqueNames(v as ParamInfo[]),
          `at most ${LIMITS_31.maxParams} parameters with unique names`),
        actions: req(f, 'actions', (v) => Array.isArray(v) && v.length <= LIMITS_31.maxActions && v.every(isAction) && uniqueNames(v as ActionInfo[]),
          `at most ${LIMITS_31.maxActions} actions with unique names`),
      };
    case MSG.control: {
      const op = req<ControlOp>(f, 'op', (v) => CONTROL_OPS.includes(v as ControlOp), 'pause, resume, step, set or run');
      const name = opt<string | undefined>(f, 'name', isName, 'a parameter or action name', undefined);
      if ((op === 'set' || op === 'run') && name === undefined) bad(t, `'${op}' needs 'name'`);
      return {
        type: 'control',
        request_id: req(f, 'request_id', isU32, 'a uint32'),
        op,
        name: op === 'set' || op === 'run' ? name : undefined,
        value: op === 'set' ? req<ParamValue>(f, 'value', isSetValue, 'a number, a boolean or a choice') : undefined,
        count: op === 'step' ? opt<number | undefined>(f, 'count', (v) => isU32(v) && (v as number) >= 1, 'a positive uint32', undefined) : undefined,
      };
    }
    case MSG.control_reply: {
      const status = req<ControlReplyMsg['status']>(f, 'status', (v) => v === 'ok' || v === 'queued' || v === 'error', 'ok, queued or error');
      const op = req<ControlOp>(f, 'op', (v) => CONTROL_OPS.includes(v as ControlOp), 'a control op');
      const error = opt<{ code: string; message?: string } | undefined>(f, 'error', (v) => isObj(v) && isStr((v as Record<string, unknown>).code)
        && ((v as Record<string, unknown>).message == null || isText((v as Record<string, unknown>).message)), '{code, message}', undefined);
      if ((error === undefined) === (status === 'error')) bad(t, "a control_reply carries 'error' exactly when its status is 'error'");
      if (status === 'queued' && op !== 'run') bad(t, "only 'run' is queued");
      return {
        type: 'control_reply',
        request_id: req(f, 'request_id', isU32, 'a uint32'),
        op,
        status,
        name: opt(f, 'name', isName, 'a parameter or action name', undefined),
        value: opt(f, 'value', isSetValue, 'a number, a boolean or a choice', undefined),
        message: opt(f, 'message', isText, `a string of at most ${LIMITS_31.textMax} bytes`, undefined),
        error,
      };
    }
    case MSG.selection:
      return {
        type: 'selection',
        space: req(f, 'space', isName, 'an item space name'),
        items: req(f, 'items', (v) => Array.isArray(v) && v.length <= LIMITS.maxEventIndices && v.every(isUint) && isAscending(v as number[]),
          `at most ${LIMITS.maxEventIndices} ascending unsigned integers`),
        view: opt(f, 'view', isName, 'a view name', undefined),
        truncated: opt(f, 'truncated', isBool, 'a boolean', false),
        details: opt(f, 'details', isDetails, '{fields, items, values} within the limits', undefined),
        details_error: opt(f, 'details_error', isText, `a string of at most ${LIMITS_31.textMax} bytes`, undefined),
      };
    case MSG.scene_update:
      return decodeSceneUpdate(f);
    case MSG.scene_clear:
      return decodeSceneClear(f);
    default:
      return { type: 'unknown', id: t, meta: f.meta };
  }
}

const drop = (o: Record<string, unknown>) => {
  for (const k of Object.keys(o)) if (o[k] === undefined || o[k] === null) delete o[k];
  return o;
};

/**
 * Encodes one message into the bytes of one WebSocket message, at a minor
 * version (this codec's by default; a 3.0 message at minor 0 encodes to the
 * bytes a 3.0 sender writes).
 */
export function encode(msg: Message, minor: number = PROTOCOL_MINOR): Uint8Array {
  const id = MSG[msg.type as MessageType];
  const encodeFrame = (type: number, meta: Record<string, unknown>, sections: Section[] = []) => frameAt(type, meta, sections, minor);
  switch (msg.type) {
    case 'hello':
      return encodeFrame(id, drop({ role: msg.role, capabilities: msg.capabilities, software: msg.software, session: msg.session }));
    case 'welcome':
      return encodeFrame(id, drop({ role: msg.role, capabilities: msg.capabilities, software: msg.software, session: msg.session, control: msg.control }));
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
    case 'controls':
      return encodeFrame(id, { run: msg.run, params: msg.params, actions: msg.actions });
    case 'control':
      return encodeFrame(id, drop({ request_id: msg.request_id, op: msg.op, name: msg.name, value: msg.value, count: msg.count }));
    case 'control_reply':
      return encodeFrame(id, drop({
        request_id: msg.request_id, op: msg.op, status: msg.status, name: msg.name, value: msg.value, message: msg.message, error: msg.error,
      }));
    case 'selection':
      return encodeFrame(id, drop({
        space: msg.space, items: msg.items, view: msg.view, truncated: msg.truncated ? true : undefined, details: msg.details,
        details_error: msg.details_error,
      }));
    case 'scene_update':
      return encodeSceneUpdate(msg, minor);
    case 'scene_clear':
      return encodeSceneClear(msg, minor);
  }
}

// --- protocol 3.2 draft: the scenes subset (proposals: minor 2, capability "scenes", kind 6, 0x0070, 0x0072; see the 3.2 block of include/caliper/live_net.h) ---
// The same rules as the block at the end of python/caliper/_wire.py.

import {
  COLOR_PER_STRIP, COLOR_PER_VERTEX, DRAFTED_PARTS, LIMITS_32, LINES_SEGMENTS, LINES_STRIPS, PART_KEYS, PART_LINES, PART_MESH,
  PART_POINTS, PART_SECTIONS, SCENE_DATA_VIEW, SCENE_PARTS, SPACE_2D, SPACE_3D, UP_AXES_2D, UP_AXES_3D,
  type SceneArray, type SceneClearMsg, type SceneUpdateMsg, type SceneViewInfo,
} from './messages.js';

const PATH_SEGMENT = /^[A-Za-z0-9_.-]{1,63}$/;
const CLEAR_PARTS = [...SCENE_PARTS, ...DRAFTED_PARTS, 'transform'];
const MAX_STATS = 64;
const DTYPE_NAME: Record<number, string> = { 0: 'f32', 1: 'f16', 2: 'bf16', 3: 'i64', 4: 'i32', 5: 'u8' };

function i32(s: Section): Int32Array {
  const n = s.bytes.byteLength / 4;
  if (LITTLE && s.bytes.byteOffset % 4 === 0) return new Int32Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Int32Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getInt32(4 * i, true);
  return out;
}

/** An entity path: 0 to 16 segments of [A-Za-z0-9_.-]{1,63} joined by '/', at most 255 bytes, never '.' or '..'. */
export function entityPathOk(v: unknown): boolean {
  if (typeof v !== 'string') return false;
  if (v === '') return true;
  if (v.length > LIMITS_32.pathMax || !/^[\x00-\x7f]*$/.test(v)) return false;
  const segs = v.split('/');
  return segs.length <= LIMITS_32.pathSegmentsMax && segs.every((x) => PATH_SEGMENT.test(x) && x !== '.' && x !== '..');
}

const isRgba = (v: unknown) => Array.isArray(v) && v.length === 4 && v.every((x) => isInt(x) && x >= 0 && x <= 255);
const isPositive = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v > 0;
const isStatList = (v: unknown) => Array.isArray(v) && v.length <= MAX_STATS && v.every((x) => x === null || isNum(x));
const KEY_CHECKS: Record<string, [(v: unknown) => boolean, string]> = {
  color: [isRgba, '[r, g, b, a] of integers from 0 to 255'],
  radius: [isPositive, 'a positive number'],
  width: [isPositive, 'a positive number'],
  colormap: [isInt, 'an integer'],
  flags: [isUint, 'an unsigned integer'],
  vmin: [isNum, 'a number'],
  vmax: [isNum, 'a number'],
  stats: [isStatList, `at most ${MAX_STATS} numbers or nulls`],
  mode: [(v) => v === LINES_SEGMENTS || v === LINES_STRIPS || v === 'edges', 'segments or strips'],
  color_per: [(v) => v === COLOR_PER_VERTEX || v === COLOR_PER_STRIP || v === 'edge', 'vertex or strip'],
};

/** The section `name` of a scene part, or undefined; `shape` holds the rows it must have (null: any) and its rank. */
function partSection(f: Frame, name: string, dtype: number, shape: (number | null)[]): Section | undefined {
  const s = section(f, name);
  if (!s) return undefined;
  if (s.encoding !== 0 || s.dtype !== dtype || s.shape.length !== shape.length || shape.some((x, i) => x !== null && x !== s.shape[i]))
    bad(f.type, `'${name}' must be ${DTYPE_NAME[dtype]} (${shape.map((x) => x ?? '_').join(', ')}), got dtype ${s.dtype} shape [${s.shape}]`);
  return s;
}

function partPositions(f: Frame, part: string): { pos: Section; n: number } {
  if (!section(f, 'positions')) bad(f.type, `a ${part} part needs a 'positions' section`);
  const pos = partSection(f, 'positions', DTYPE.f32, [null, null])!;
  const [n, d] = pos.shape;
  if (d !== 2 && d !== 3) bad(f.type, `'positions' is (N, 2) or (N, 3), got [${pos.shape}]`);
  if (n > LIMITS_32.maxBatchPoints) bad(f.type, `${n} positions, more than ${LIMITS_32.maxBatchPoints} in a batch`);
  return { pos, n };
}

function sceneArray(s: Section): SceneArray {
  const data = s.dtype === DTYPE.u8 ? u8(s) : s.dtype === DTYPE.i32 ? i32(s) : f32(s);
  return { dtype: s.dtype, shape: s.shape, data };
}

function colours(f: Frame, rows: number, arrays: Record<string, SceneArray>): void {
  const c = partSection(f, 'colors', DTYPE.u8, [rows, 4]);
  const v = partSection(f, 'values', DTYPE.f32, [rows]);
  if (c && v) bad(f.type, "'colors' and 'values' together");
  if (c) arrays.colors = sceneArray(c);
  if (v) arrays.values = sceneArray(v);
}

function unsupported(type: number, msg: string): never {
  throw new MessageError('unsupported', msg, type);
}

function decodeScenePoints(f: Frame): Record<string, SceneArray> {
  const { pos, n } = partPositions(f, PART_POINTS);
  const arrays: Record<string, SceneArray> = { positions: sceneArray(pos) };
  colours(f, n, arrays);
  const r = partSection(f, 'radii', DTYPE.f32, [n]);
  if (r) arrays.radii = sceneArray(r);
  return arrays;
}

function decodeSceneLines(f: Frame, mode: string, colorPer: string | undefined): Record<string, SceneArray> {
  if (mode === 'edges') unsupported(f.type, "lines of mode 'edges' (index-pair edges) are not built yet");
  const { pos, n: v } = partPositions(f, PART_LINES);
  const arrays: Record<string, SceneArray> = { positions: sceneArray(pos) };
  if (section(f, 'indices')) bad(f.type, `lines of mode '${mode}' take no 'indices'`);
  let rows = v;
  if (mode === LINES_SEGMENTS) {
    if (v % 2) bad(f.type, `segments need an even number of vertices, got ${v}`);
    if (section(f, 'strip_lengths')) bad(f.type, "segments take no 'strip_lengths'");
    if (colorPer !== undefined && colorPer !== COLOR_PER_VERTEX) bad(f.type, `segments take color_per 'vertex', got '${colorPer}'`);
  } else {
    const sl = partSection(f, 'strip_lengths', DTYPE.i32, [null]);
    if (!sl) bad(f.type, "strips need a 'strip_lengths' section");
    const lens = i32(sl);
    if (lens.length > LIMITS_32.maxBatchItems) bad(f.type, `${lens.length} strips, more than ${LIMITS_32.maxBatchItems} in a batch`);
    let total = 0;
    for (const x of lens) {
      if (x < 1) bad(f.type, 'every strip length is at least 1');
      total += x;
    }
    if (total !== v) bad(f.type, `the strip lengths add up to ${total}, not to the ${v} vertices`);
    arrays.strip_lengths = sceneArray(sl);
    if (colorPer === 'edge') bad(f.type, "strips take color_per 'vertex' or 'strip'");
    if (colorPer === COLOR_PER_STRIP) rows = lens.length;
  }
  colours(f, rows, arrays);
  return arrays;
}

function decodeSceneMesh(f: Frame): Record<string, SceneArray> {
  const { pos, n: v } = partPositions(f, PART_MESH);
  const arrays: Record<string, SceneArray> = { positions: sceneArray(pos) };
  const idx = partSection(f, 'indices', DTYPE.i32, [null, 3]);
  if (idx) {
    if (idx.shape[0] > LIMITS_32.maxBatchPoints) bad(f.type, `${idx.shape[0]} triangles, more than ${LIMITS_32.maxBatchPoints} in a batch`);
    arrays.indices = sceneArray(idx);
  } else if (v % 3) {
    bad(f.type, `a mesh without 'indices' needs a multiple of 3 vertices, got ${v}`);
  }
  colours(f, v, arrays);
  return arrays;
}

function scenePartKeys(f: Frame, part: keyof typeof PART_KEYS): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of PART_KEYS[part]) {
    const v = f.meta[key];
    if (v === undefined || v === null) continue;
    const [ok, what] = KEY_CHECKS[key];
    if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
    out[key] = v;
  }
  return out;
}

export function decodeSceneUpdate(f: Frame): SceneUpdateMsg {
  const t = f.type;
  const sub = req<number>(f, 'sub', isU32, 'a uint32');
  const view = req<string>(f, 'view', isName, 'a view name');
  const seq = req<number>(f, 'seq', isUint, 'an unsigned integer');
  const stamp_ns = req<number>(f, 'stamp_ns', isUint, 'an unsigned integer');
  const path = req<string>(f, 'path', entityPathOk, 'an entity path');
  const part = req<string>(f, 'part', isStr, 'a string');
  if (DRAFTED_PARTS.includes(part)) unsupported(t, `the part '${part}' is not built yet`);
  if (!(SCENE_PARTS as readonly string[]).includes(part)) bad(t, `unknown part '${part}'`);
  if (f.meta.split !== undefined && f.meta.split !== null) unsupported(t, 'split updates are not built yet');
  const p = part as SceneUpdateMsg['part'];
  const params = scenePartKeys(f, p);
  let arrays: Record<string, SceneArray>;
  if (p === PART_POINTS) {
    arrays = decodeScenePoints(f);
  } else if (p === PART_LINES) {
    if (!('mode' in params)) bad(t, "a lines part needs 'mode'");
    arrays = decodeSceneLines(f, params.mode as string, params.color_per as string | undefined);
  } else {
    arrays = decodeSceneMesh(f);
  }
  if (params.stats) params.stats = (params.stats as (number | null)[]).map((x) => (x === null ? NaN : x));
  return {
    type: 'scene_update', sub, view, seq, stamp_ns, path, part: p, arrays,
    params: params as SceneUpdateMsg['params'],
    static: opt(f, 'static', isBool, 'a boolean', false),
    state: opt(f, 'state', isBool, 'a boolean', false),
    axes: opt<PayloadAxis[] | undefined>(f, 'axes', Array.isArray, 'a list', undefined),
    items: opt<string | undefined>(f, 'items', isName, 'an item space name', undefined),
  };
}

export function decodeSceneClear(f: Frame): SceneClearMsg {
  const c: SceneClearMsg = {
    type: 'scene_clear',
    sub: req(f, 'sub', isU32, 'a uint32'),
    view: req(f, 'view', isName, 'a view name'),
    seq: req(f, 'seq', isUint, 'an unsigned integer'),
    stamp_ns: req(f, 'stamp_ns', isUint, 'an unsigned integer'),
    path: req(f, 'path', entityPathOk, 'an entity path'),
    recursive: opt(f, 'recursive', isBool, 'a boolean', false),
    parts: opt<string[] | undefined>(f, 'parts', (v) => Array.isArray(v) && v.length > 0 && v.every((x) => CLEAR_PARTS.includes(x as string)), 'a list of parts', undefined),
    reset: opt(f, 'reset', isBool, 'a boolean', false),
    static: opt(f, 'static', isBool, 'a boolean', false),
    state: opt(f, 'state', isBool, 'a boolean', false),
  };
  if (c.reset && c.path !== '') bad(f.type, 'a reset clears the root: its \'path\' is ""');
  return c;
}

export function encodeSceneUpdate(m: SceneUpdateMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, path: m.path, part: m.part };
  for (const [key, v] of Object.entries(m.params)) {
    if (v === undefined || v === null) continue;
    meta[key] = key === 'stats' ? (v as number[]).map((x) => (Number.isFinite(x) ? x : null)) : v;
  }
  if (m.static) meta.static = true;
  if (m.state) meta.state = true;
  if (m.axes !== undefined) meta.axes = m.axes;
  if (m.items !== undefined) meta.items = m.items;
  const sections: Section[] = [];
  for (const name of PART_SECTIONS[m.part] ?? []) {
    const a = m.arrays[name];
    if (a) sections.push({ name, dtype: a.dtype, shape: a.shape, encoding: 0, bytes: new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength) });
  }
  return frameAt(MSG.scene_update, meta, sections, minor);
}

export function encodeSceneClear(m: SceneClearMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, path: m.path };
  for (const key of ['recursive', 'reset', 'static', 'state'] as const) if (m[key]) meta[key] = true;
  if (m.parts !== undefined) meta.parts = m.parts;
  return frameAt(MSG.scene_clear, meta, [], minor);
}

/** The key "scene" of a VIEWS entry of kind 6, checked, with its defaults filled in. Throws MessageError (bad_request). */
export function sceneViewInfo(entry: ViewEntry): SceneViewInfo {
  const fail = (msg: string): never => {
    throw new MessageError('bad_request', msg);
  };
  const sc = entry.scene ?? {};
  if (!isObj(sc)) fail(`'scene' must be an object, got ${JSON.stringify(sc)}`);
  const o = sc as Record<string, unknown>;
  const space = o.space ?? SPACE_3D;
  if (space !== SPACE_3D && space !== SPACE_2D) fail(`scene space ${JSON.stringify(space)}; expected '3d' or '2d'`);
  const axes = space === SPACE_3D ? UP_AXES_3D : UP_AXES_2D;
  const up = o.up ?? (space === SPACE_3D ? '+y' : '-y');
  if (typeof up !== 'string' || !axes.includes(up)) fail(`scene up ${JSON.stringify(up)}; a ${space} scene takes one of ${axes.join(', ')}`);
  const units = o.units ?? 'm';
  if (typeof units !== 'string' || utf8.encode(units).length > LIMITS_32.unitsMax)
    fail(`scene units must be a string of at most ${LIMITS_32.unitsMax} bytes, got ${JSON.stringify(units)}`);
  return { space: space as SceneViewInfo['space'], up: up as string, units: units as string };
}

/**
 * The resolved data view of a scene subscription: only the default data view is built, so a request that is absent, {},
 * or the resolved default itself. Throws MessageError (bad_data_view).
 */
export function resolveSceneDataView(view: ViewEntry, request?: DataViewSpec | null): typeof SCENE_DATA_VIEW {
  const fail = (msg: string): never => {
    throw new MessageError('bad_data_view', msg);
  };
  const isDefault = (v: unknown) => v === undefined || v === null || (isObj(v) && (Object.keys(v as object).length === 0
    || canonicalJson(v) === canonicalJson(SCENE_DATA_VIEW)));
  if (request !== undefined && request !== null && !isObj(request)) fail('a data view is a JSON object');
  if (!isDefault(view.default)) fail("a scene's own default data view is not built yet");
  if (!isDefault(request))
    fail('only the default data view of a scene is built yet: send {} or none; entity filters and budgets come in a later increment');
  return JSON.parse(JSON.stringify(SCENE_DATA_VIEW));
}
