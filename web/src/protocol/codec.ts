// The binary codec of wire protocol v3 (include/caliper/live_net.h,
// python/caliper/_wire.py). One frame is a 24-byte little-endian header,
// section_count uint32 section lengths, a UTF-8 JSON metadata object, then the
// payload sections, each at an offset from the frame start that is a multiple
// of 8. This module is the only one that knows the bytes.

import {
  CLOSE, CONTROL_OPS, DTYPE, DTYPE_SIZE, LIMITS, LIMITS_31, MSG, PROTOCOL_MAJOR, PROTOCOL_MINOR, PROTOCOL_MINOR_3_2, RUN_STATES,
  type ActionInfo, type ControlOp, type ControlReplyMsg, type ControlsMsg, type SeriesMsg, type TimelineInfo, type DataView as DataViewSpec, type ErrorMsg, type Message,
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
  return (o.default === undefined || o.default === null || isObj(o.default))
    && (o.recorded === undefined || o.recorded === null || isRecorded(o.recorded));
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
  const empty = opt(f, 'empty', isBool, 'a boolean', false);
  const data = section(f, 'data');
  if (empty) {
    if (f.sections.length) bad(t, 'an empty snapshot carries no section');
  } else if (!data) {
    bad(t, "a snapshot needs a 'data' section");
  }
  const auxSec = section(f, 'aux');
  const stats = opt<(number | null)[]>(f, 'stats', (v) => Array.isArray(v) && v.length <= 64 && v.every((x) => x === null || isNum(x)), 'a list of numbers or nulls', []);
  const sub = req<number>(f, 'sub', isU32, 'a uint32');
  const view = req<string>(f, 'view', isName, 'a view name');
  const seq = req<number>(f, 'seq', isUint, 'an unsigned integer');
  const stamp_ns = req<number>(f, 'stamp_ns', isUint, 'an unsigned integer');
  const data_view = req<DataViewSpec>(f, 'data_view', isObj, 'an object');
  const colormap = opt(f, 'colormap', isInt, 'an integer', 0);
  const flags = opt(f, 'flags', isUint, 'an unsigned integer', 0);
  const vmin = opt(f, 'vmin', isNum, 'a number', 0);
  const vmax = opt(f, 'vmax', isNum, 'a number', 1);
  const axes = opt<PayloadAxis[]>(f, 'axes', Array.isArray, 'a list', []);
  const tile = opt(f, 'tile', isObj, 'an object', null);
  const pointsDims = opt<number | undefined>(f, 'points_dims', (v) => v === 2 || v === 3, '2 or 3', undefined);
  const label_count = opt<number | undefined>(f, 'label_count', isUint, 'an unsigned integer', undefined);
  let dims: 2 | 3 | undefined;
  let aux: Float32Array | undefined;
  if (!empty) {
    const d = data!;
    const want: Record<number, [number, number]> = { 1: [DTYPE.f32, 2], 2: [DTYPE.u8, 3], 3: [DTYPE.f32, 2], 5: [DTYPE.f32, 1], 8: [DTYPE.f32, 1] };
    const w = want[kind];
    if (w) {
      // The encoding check is this viewer's own (the other codecs refuse a non-raw payload where it is read).
      if (d.dtype !== w[0] || d.shape.length !== w[1] || d.encoding !== 0) bad(t, `kind ${kind}: 'data' is dtype ${d.dtype} shape [${d.shape}]`);
      if (kind === 2 && d.shape[2] !== 4) bad(t, "an image's 'data' is (H, W, 4)");
      if (kind === 8 && !(d.shape[0] >= 1 && d.shape[0] <= MAX_BARS)) bad(t, `a bar chart has 1 to ${MAX_BARS} bars, got ${d.shape[0]}`);
      if (kind === 3) {
        if ((d.shape[1] !== 2 && d.shape[1] !== 3) || (pointsDims !== undefined && pointsDims !== d.shape[1]))
          bad(t, "points 'data' is (N, points_dims) with points_dims 2 or 3");
        dims = d.shape[1] as 2 | 3;
        if (auxSec) {
          if (auxSec.dtype !== DTYPE.f32 || auxSec.shape.length !== 1 || auxSec.shape[0] !== d.shape[0]) bad(t, "points 'aux' is (N,) f32");
          aux = f32(auxSec);
        }
      }
    }
  }
  const m: SnapshotMsg = {
    type: 'snapshot', sub, view, seq, stamp_ns, kind, data_view,
    data: data && !empty ? (data.dtype === DTYPE.u8 ? u8(data) : data.dtype === DTYPE.f32 ? f32(data) : new Float32Array(0)) : new Float32Array(0),
    shape: data && !empty ? data.shape : [],
    aux, colormap, flags, vmin, vmax,
    stats: stats.map((x) => (x === null ? NaN : x)),
    axes, tile, points_dims: dims, label_count,
  };
  if (empty) m.empty = true;
  // the 3.2 additions, checked after the 3.1 ones (_snapshot_32 of _wire.py)
  if (kind === 8 && !empty) {
    const b = data!.shape[0];
    const c = partSection(f, 'colors', DTYPE.u8, [b, 4]);
    const labels = opt<string[] | undefined>(f, 'labels', (v) => Array.isArray(v) && v.length === b && v.every((x) => typeof x === 'string' && utf8.encode(x).length <= NAME_MAX),
      `${b} labels of at most ${NAME_MAX} bytes`, undefined);
    const horizontal = opt(f, 'horizontal', isBool, 'a boolean', false);
    if (c) m.colors = u8(c);
    if (labels) m.labels = labels;
    if (horizontal) m.horizontal = true;
  }
  const isStatic = opt(f, 'static', isBool, 'a boolean', false);
  const cursor = opt<number | undefined>(f, 'cursor', isU32, 'a uint32', undefined);
  const { stamp } = stampSections(f, isStatic);
  if (isStatic) m.static = true;
  if (cursor !== undefined) m.cursor = cursor;
  if (stamp) m.stamp = i64(stamp);
  return m;
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
          && ((s as SessionIdentity).recording == null || isRecordingRef((s as SessionIdentity).recording))
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
        timelines: opt<TimelineInfo[] | undefined>(f, 'timelines', isTimelines, '2 to 8 timelines, step and time first', undefined),
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
      const sub = req<number>(f, 'sub', isU32, 'a uint32');
      const view = req<string>(f, 'view', isName, 'a view name');
      const start = req<number>(f, 'start', isUint, 'an unsigned integer');
      const total = req<number>(f, 'total', isUint, 'an unsigned integer');
      const stamp_ns = req<number>(f, 'stamp_ns', isUint, 'an unsigned integer');
      const sparse = section(f, 'indices') !== undefined;
      if (v.shape[0] < 1 || v.shape[0] > 262144 || (!sparse && start + v.shape[0] > total))
        bad(t, `series: ${v.shape[0]} values from ${start} with total ${total}`);
      const out: SeriesMsg = { type: 'series', sub, view, start, total, stamp_ns, values: f32(v) };
      // the 3.2 additions (_series_32 of _wire.py)
      const n = v.shape[0];
      const idx = partSection(f, 'indices', DTYPE.i64, [n]);
      const decimated = opt<number | undefined>(f, 'decimated', (x) => isInt(x) && x >= 1, 'an integer >= 1', undefined);
      if ((idx === undefined) !== (decimated === undefined)) bad(t, "'indices' and 'decimated' go together");
      const cursor = opt<number | undefined>(f, 'cursor', isU32, 'a uint32', undefined);
      const { stamp, stamps } = stampSections(f, false, n);
      if (idx) out.indices = i64(idx);
      if (decimated !== undefined) out.decimated = decimated;
      if (cursor !== undefined) out.cursor = cursor;
      if (stamp) out.stamp = i64(stamp);
      if (stamps) {
        out.stamps = i64(stamps);
        out.stamps_t = stamps.shape[1];
      }
      return out;
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
    case MSG.controls: {
      const c: ControlsMsg = {
        type: 'controls',
        run: req(f, 'run', isRun, 'a run state'),
        params: req(f, 'params', (v) => Array.isArray(v) && v.length <= LIMITS_31.maxParams && v.every(isParam) && uniqueNames(v as ParamInfo[]),
          `at most ${LIMITS_31.maxParams} parameters with unique names`),
        actions: req(f, 'actions', (v) => Array.isArray(v) && v.length <= LIMITS_31.maxActions && v.every(isAction) && uniqueNames(v as ActionInfo[]),
          `at most ${LIMITS_31.maxActions} actions with unique names`),
      };
      const { stamp } = stampSections(f, false);
      if (stamp) c.stamp = i64(stamp);
      return c;
    }
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
    case MSG.scene_transforms:
      return decodeSceneTransforms(f);
    case MSG.scene_clear:
      return decodeSceneClear(f);
    case MSG.scene_classes:
      return decodeSceneClasses(f);
    case MSG.text:
      return decodeText(f);
    case MSG.recordings:
      return decodeRecordings(f);
    case MSG.recording_open:
      return decodeRecordingOpen(f);
    case MSG.cursor:
      return decodeCursor(f);
    case MSG.time_state:
      return decodeTimeState(f);
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
      return encodeFrame(id, msg.timelines === undefined ? { session: msg.session ?? null, views: msg.views }
        : { session: msg.session ?? null, views: msg.views, timelines: msg.timelines });
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
    case 'series': {
      const sections: Section[] = [f32Section('values', msg.values, [msg.values.length])];
      const n = msg.values.length;
      if (msg.stamps) sections.push(stampSection('stamps', msg.stamps, [n, msg.stamps_t ?? (n ? msg.stamps.length / n : 0)]));
      else if (msg.stamp) sections.push(stampSection('stamp', msg.stamp, [msg.stamp.length]));
      if (msg.indices) sections.push(stampSection('indices', msg.indices, [msg.indices.length]));
      return encodeFrame(id, drop({ sub: msg.sub, view: msg.view, start: msg.start, total: msg.total, stamp_ns: msg.stamp_ns, decimated: msg.decimated, cursor: msg.cursor }), sections);
    }
    case 'snapshot': {
      const s = msg;
      const sections: Section[] = [];
      if (!s.empty) {
        sections.push({
          name: 'data', dtype: s.data instanceof Uint8Array ? DTYPE.u8 : DTYPE.f32, shape: s.shape, encoding: 0,
          bytes: new Uint8Array(s.data.buffer, s.data.byteOffset, s.data.byteLength),
        });
      }
      if (s.aux) sections.push(f32Section('aux', s.aux, [s.aux.length]));
      if (s.colors) sections.push({ name: 'colors', dtype: DTYPE.u8, shape: [s.colors.length / 4, 4], encoding: 0, bytes: s.colors });
      if (s.stamp) sections.push(stampSection('stamp', s.stamp, [s.stamp.length]));
      return encodeFrame(id, drop({
        sub: s.sub, view: s.view, seq: s.seq, stamp_ns: s.stamp_ns, kind: s.kind, data_view: s.data_view, colormap: s.colormap,
        flags: s.flags, vmin: s.vmin, vmax: s.vmax, stats: s.stats.map((x) => (Number.isFinite(x) ? x : null)), axes: s.axes,
        tile: s.tile, points_dims: s.points_dims, label_count: s.label_count, labels: s.labels,
        horizontal: s.horizontal ? true : undefined, static: s.static ? true : undefined, cursor: s.cursor, empty: s.empty ? true : undefined,
      }), sections);
    }
    case 'controls':
      return encodeFrame(id, { run: msg.run, params: msg.params, actions: msg.actions }, msg.stamp ? [stampSection('stamp', msg.stamp, [msg.stamp.length])] : []);
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
    case 'scene_transforms':
      return encodeSceneTransforms(msg, minor);
    case 'scene_clear':
      return encodeSceneClear(msg, minor);
    case 'scene_classes':
      return encodeSceneClasses(msg, minor);
    case 'text':
      return encodeText(msg, minor);
    case 'recordings':
      return encodeFrame(id, { recordings: msg.recordings });
    case 'recording_open':
      return encodeFrame(id, { request_id: msg.request_id, recording: msg.recording });
    case 'cursor':
      return encodeFrame(id, drop({
        request_id: msg.request_id, op: msg.op, timeline: msg.timeline, value: msg.value, count: msg.count, speed: msg.speed,
        pace: msg.pace, loop: msg.loop,
      }));
    case 'time_state':
      return encodeFrame(id, { ...drop({
        request_id: msg.request_id, session: msg.session, generation: msg.generation, state: msg.state, timeline: msg.timeline,
        ranges: msg.ranges, live: msg.live, speed: msg.speed, pace: msg.pace, loop: msg.loop,
      }), cursor: msg.cursor ?? null });
  }
}

// --- protocol 3.2 draft: the scenes subset (proposals: minor 2, capability "scenes", kind 6, 0x0070 to 0x0073, dtype u16; see the 3.2 block of include/caliper/live_net.h) ---
// The same rules, in the same order, as the block at the end of python/caliper/_wire.py.

import {
  CAMERA_REQUIRED, CLEAR_PARTS, COLOR_PER_EDGE, COLOR_PER_STRIP, COLOR_PER_VERTEX, CONVENTION_RDF, CONVENTION_RUB, FORM_COLOR,
  FORM_DEPTH, FORM_ENCODED, FORM_SEGMENTATION, FORMAT_JPEG, FORMAT_PNG, LIMITS_32, LINES_EDGES, LINES_SEGMENTS, LINES_STRIPS,
  MAX_CLASS_ID, MAX_CLASSES, MAX_ENCODED_BYTES, MAX_IMAGE_SIDE, MAX_LABELS, MAX_TIMELINES, MAX_TRANSFORMS, PART_ARROWS, PART_BOXES,
  PART_CAMERA, PART_IMAGE, PART_KEYS, PART_LINES, PART_MESH, PART_POINTS, PART_SECTIONS, SCENE_DATA_VIEW, SCENE_PARTS, SPACE_2D,
  SPACE_3D, UP_AXES_2D, UP_AXES_3D,
  CURSOR_OPS, CURSOR_PLAY, CURSOR_SEEK, CURSOR_STEP, LEVEL_ERROR, MAX_BARS, MAX_CURSOR_STEP,
  MAX_INCOMPLETE_UPDATES, MAX_PIECES, MAX_RECORDED, MAX_RECORDINGS, MAX_SPEED, MAX_UPDATE_BYTES, MIN_SPEED, PACE_EVEN, PACE_RECORDED,
  STAMP_TEXT_MAX, TEXT_LINE_MAX, TEXT_MAX_LINES, TIME_FOLLOWING, TIME_PAUSED, TIME_PLAYING, TIMELINE_SEQUENCE, TIMELINE_TIME,
  type CursorMsg, type RecordingEntry, type RecordingOpenMsg, type RecordingsMsg, type SplitKey, type TextMsg, type TimeStateMsg,
  type SceneArray, type SceneClass, type SceneClassesMsg, type SceneClearMsg, type SceneTransformsMsg, type SceneUpdateMsg,
  type SceneViewInfo, type ScenePart, type ViewCameraOrbit,
} from './messages.js';

const PATH_SEGMENT = /^[A-Za-z0-9_.-]{1,63}$/;
const MAX_STATS = 64;
const NAME_MAX = 63;
const DTYPE_NAME: Record<number, string> = { 0: 'f32', 1: 'f16', 2: 'bf16', 3: 'i64', 4: 'i32', 5: 'u8', 6: 'u16' };

function i32(s: Section): Int32Array {
  const n = s.bytes.byteLength / 4;
  if (LITTLE && s.bytes.byteOffset % 4 === 0) return new Int32Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Int32Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getInt32(4 * i, true);
  return out;
}

function u16(s: Section): Uint16Array {
  const n = s.bytes.byteLength / 2;
  if (LITTLE && s.bytes.byteOffset % 2 === 0) return new Uint16Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Uint16Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getUint16(2 * i, true);
  return out;
}

function i64(s: Section): BigInt64Array {
  const n = s.bytes.byteLength / 8;
  if (LITTLE && s.bytes.byteOffset % 8 === 0) return new BigInt64Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new BigInt64Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getBigInt64(8 * i, true);
  return out;
}

const bytesOf = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

/** An entity path: 0 to 16 segments of [A-Za-z0-9_.-]{1,63} joined by '/', at most 255 bytes, never '.' or '..'. */
export function entityPathOk(v: unknown): boolean {
  if (typeof v !== 'string') return false;
  if (v === '') return true;
  if (v.length > LIMITS_32.pathMax || !/^[\x00-\x7f]*$/.test(v)) return false;
  const segs = v.split('/');
  return segs.length <= LIMITS_32.pathSegmentsMax && segs.every((x) => PATH_SEGMENT.test(x) && x !== '.' && x !== '..');
}

const isRgba = (v: unknown) => Array.isArray(v) && v.length === 4 && v.every((x) => isInt(x) && x >= 0 && x <= 255);
const isFinite_ = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
const isPositive = (v: unknown) => isFinite_(v) && (v as number) > 0;
const isStatList = (v: unknown) => Array.isArray(v) && v.length <= MAX_STATS && v.every((x) => x === null || isNum(x));
const isLabels = (v: unknown) => Array.isArray(v) && v.length >= 1 && v.length <= MAX_LABELS
  && v.every((x) => typeof x === 'string' && utf8.encode(x).length <= NAME_MAX);
const isSide = (v: unknown) => isInt(v) && v >= 1 && v <= MAX_IMAGE_SIDE;
const KEY_CHECKS: Record<string, [(v: unknown) => boolean, string]> = {
  color: [isRgba, '[r, g, b, a] of integers from 0 to 255'],
  radius: [isPositive, 'a positive number'],
  width: [isPositive, 'a positive number'],
  labels: [isLabels, `a list of 1 to ${MAX_LABELS} strings of at most ${NAME_MAX} bytes`],
  colormap: [isInt, 'an integer'],
  flags: [isUint, 'an unsigned integer'],
  vmin: [isNum, 'a number'],
  vmax: [isNum, 'a number'],
  stats: [isStatList, `at most ${MAX_STATS} numbers or nulls`],
  mode: [(v) => v === LINES_SEGMENTS || v === LINES_STRIPS || v === LINES_EDGES, 'segments, strips or edges'],
  color_per: [(v) => v === COLOR_PER_VERTEX || v === COLOR_PER_STRIP || v === COLOR_PER_EDGE, 'vertex, strip or edge'],
  fill: [isBool, 'a boolean'],
  form: [(v) => v === FORM_COLOR || v === FORM_DEPTH || v === FORM_SEGMENTATION || v === FORM_ENCODED, 'color, depth, segmentation or encoded'],
  opacity: [(v) => isFinite_(v) && (v as number) >= 0 && (v as number) <= 1, 'a number from 0 to 1'],
  scale: [isPositive, 'a positive number'],
  cloud: [isBool, 'a boolean'],
  format: [(v) => v === FORMAT_JPEG || v === FORMAT_PNG, 'jpeg or png'],
  fx: [isPositive, 'a positive number'],
  fy: [isPositive, 'a positive number'],
  cx: [isFinite_, 'a finite number'],
  cy: [isFinite_, 'a finite number'],
  height: [isSide, `an integer from 1 to ${MAX_IMAGE_SIDE}`],
  convention: [(v) => v === CONVENTION_RDF || v === CONVENTION_RUB, 'rdf or rub'],
  plane_distance: [isPositive, 'a positive number'],
};
const CAMERA_WIDTH: [(v: unknown) => boolean, string] = [isSide, `an integer from 1 to ${MAX_IMAGE_SIDE}`];

/** The section `name` of a scene part, or undefined; `shape` holds the rows it must have (null: any) and its rank. */
function partSection(f: Frame, name: string, dtype: number, shape: (number | null)[]): Section | undefined {
  const s = section(f, name);
  if (!s) return undefined;
  if (s.encoding !== 0 || s.dtype !== dtype || s.shape.length !== shape.length || shape.some((x, i) => x !== null && x !== s.shape[i]))
    bad(f.type, `'${name}' must be ${DTYPE_NAME[dtype]} (${shape.map((x) => x ?? '_').join(', ')}), got dtype ${s.dtype} shape [${s.shape}]`);
  return s;
}

function required(f: Frame, part: string, name: string): void {
  if (!section(f, name)) bad(f.type, `a ${part} part needs a '${name}' section`);
}

/** A required f32 (N, D) section of a part, D 2 or 3 (or `dims`). */
function rowsSection(f: Frame, part: string, name: string, limit: number, what: string, dims: number | null = null): { s: Section; n: number; d: number } {
  required(f, part, name);
  const s = partSection(f, name, DTYPE.f32, [null, dims])!;
  const [n, d] = s.shape;
  if (dims === null && d !== 2 && d !== 3) bad(f.type, `'${name}' is (N, 2) or (N, 3), got [${s.shape}]`);
  if (n > limit) bad(f.type, `${n} ${what}, more than ${limit} in a batch`);
  return { s, n, d };
}

function sceneArray(s: Section): SceneArray {
  const data = s.dtype === DTYPE.u8 ? u8(s) : s.dtype === DTYPE.i32 ? i32(s) : s.dtype === DTYPE.u16 ? u16(s) : f32(s);
  return { dtype: s.dtype, shape: s.shape, data };
}

function colours(f: Frame, rows: number, arrays: Record<string, SceneArray>): void {
  const c = partSection(f, 'colors', DTYPE.u8, [rows, 4]);
  const v = partSection(f, 'values', DTYPE.f32, [rows]);
  if (c && v) bad(f.type, "'colors' and 'values' together");
  if (c) arrays.colors = sceneArray(c);
  if (v) arrays.values = sceneArray(v);
}

function labelCount(f: Frame, params: Record<string, unknown>, n: number): void {
  const labels = params.labels as string[] | undefined;
  if (labels !== undefined && labels.length !== 1 && labels.length !== n)
    bad(f.type, `${labels.length} labels for ${n} items: give one label for all or one per item`);
}

function classIds(f: Frame, n: number, arrays: Record<string, SceneArray>): void {
  const s = partSection(f, 'class_ids', DTYPE.i32, [n]);
  if (s) arrays.class_ids = sceneArray(s);
}

type Params = Record<string, unknown>;
type Arrays = Record<string, SceneArray>;

function decodeScenePoints(f: Frame, params: Params): Arrays {
  const { s: pos, n } = rowsSection(f, PART_POINTS, 'positions', LIMITS_32.maxBatchPoints, 'positions');
  const arrays: Arrays = { positions: sceneArray(pos) };
  colours(f, n, arrays);
  classIds(f, n, arrays);
  const r = partSection(f, 'radii', DTYPE.f32, [n]);
  if (r) arrays.radii = sceneArray(r);
  labelCount(f, params, n);
  return arrays;
}

function decodeSceneLines(f: Frame, params: Params): Arrays {
  const mode = params.mode as string | undefined;
  if (mode === undefined) bad(f.type, "a lines part needs 'mode'");
  const colorPer = params.color_per as string | undefined;
  const { s: pos, n: v } = rowsSection(f, PART_LINES, 'positions', LIMITS_32.maxBatchPoints, 'positions');
  const arrays: Arrays = { positions: sceneArray(pos) };
  if (mode !== LINES_EDGES && section(f, 'indices')) bad(f.type, `lines of mode '${mode}' take no 'indices'`);
  let rows = v;
  if (mode === LINES_SEGMENTS) {
    if (v % 2) bad(f.type, `segments need an even number of vertices, got ${v}`);
    if (section(f, 'strip_lengths')) bad(f.type, "segments take no 'strip_lengths'");
    if (colorPer !== undefined && colorPer !== COLOR_PER_VERTEX) bad(f.type, `segments take color_per 'vertex', got '${colorPer}'`);
  } else if (mode === LINES_STRIPS) {
    const sl = partSection(f, 'strip_lengths', DTYPE.i32, [null]);
    if (!sl) bad(f.type, "strips need a 'strip_lengths' section");
    const lens = i32(sl);
    if (lens.length > LIMITS_32.maxBatchItems) bad(f.type, `${lens.length} strips, more than ${LIMITS_32.maxBatchItems} in a batch`);
    let min = Infinity;
    let total = 0;
    for (const x of lens) {
      if (x < min) min = x;
      total += x;
    }
    if (lens.length && min < 1) bad(f.type, 'every strip length is at least 1');
    if (total !== v) bad(f.type, `the strip lengths add up to ${total}, not to the ${v} vertices`);
    arrays.strip_lengths = sceneArray(sl);
    if (colorPer === COLOR_PER_EDGE) bad(f.type, "strips take color_per 'vertex' or 'strip'");
    if (colorPer === COLOR_PER_STRIP) rows = lens.length;
  } else {
    if (section(f, 'strip_lengths')) bad(f.type, "edges take no 'strip_lengths'");
    const idx = partSection(f, 'indices', DTYPE.i32, [null, 2]);
    if (!idx) bad(f.type, "edges need an 'indices' section, (E, 2) i32");
    if (idx.shape[0] > LIMITS_32.maxBatchPoints) bad(f.type, `${idx.shape[0]} edges, more than ${LIMITS_32.maxBatchPoints} in a batch`);
    arrays.indices = sceneArray(idx);
    if (colorPer === COLOR_PER_STRIP) bad(f.type, "edges take color_per 'vertex' or 'edge'");
    if (colorPer === COLOR_PER_EDGE) rows = idx.shape[0];
  }
  colours(f, rows, arrays);
  return arrays;
}

function decodeSceneArrows(f: Frame): Arrays {
  const { s: o, n, d } = rowsSection(f, PART_ARROWS, 'origins', LIMITS_32.maxBatchItems, 'arrows');
  required(f, PART_ARROWS, 'vectors');
  const vec = partSection(f, 'vectors', DTYPE.f32, [n, d])!;
  const arrays: Arrays = { origins: sceneArray(o), vectors: sceneArray(vec) };
  colours(f, n, arrays);
  return arrays;
}

function decodeSceneBoxes(f: Frame, params: Params): Arrays {
  const { s: c, n, d } = rowsSection(f, PART_BOXES, 'centers', LIMITS_32.maxBatchItems, 'boxes');
  required(f, PART_BOXES, 'half_sizes');
  const h = partSection(f, 'half_sizes', DTYPE.f32, [n, d])!;
  const arrays: Arrays = { centers: sceneArray(c), half_sizes: sceneArray(h) };
  const r = partSection(f, 'rotations', DTYPE.f32, [n, 4]);
  if (r) {
    if (d !== 3) bad(f.type, "'rotations' are for 3-D boxes only");
    arrays.rotations = sceneArray(r);
  }
  const col = partSection(f, 'colors', DTYPE.u8, [n, 4]);
  if (col) arrays.colors = sceneArray(col);
  classIds(f, n, arrays);
  labelCount(f, params, n);
  return arrays;
}

function decodeSceneMesh(f: Frame): Arrays {
  const { s: pos, n: v } = rowsSection(f, PART_MESH, 'positions', LIMITS_32.maxBatchPoints, 'positions');
  const arrays: Arrays = { positions: sceneArray(pos) };
  const idx = partSection(f, 'indices', DTYPE.i32, [null, 3]);
  if (idx) {
    if (idx.shape[0] > LIMITS_32.maxBatchPoints) bad(f.type, `${idx.shape[0]} triangles, more than ${LIMITS_32.maxBatchPoints} in a batch`);
    arrays.indices = sceneArray(idx);
  } else if (v % 3) {
    bad(f.type, `a mesh without 'indices' needs a multiple of 3 vertices, got ${v}`);
  }
  const nrm = partSection(f, 'normals', DTYPE.f32, [v, 3]);
  if (nrm) arrays.normals = sceneArray(nrm);
  colours(f, v, arrays);
  const uv = partSection(f, 'uvs', DTYPE.f32, [v, 2]);
  const tex = partSection(f, 'texture', DTYPE.u8, [null, null, 4]);
  if (uv && ('colors' in arrays || 'values' in arrays)) bad(f.type, "'uvs' and 'colors' or 'values' together");
  if ((uv === undefined) !== (tex === undefined)) bad(f.type, "'uvs' and 'texture' go together");
  if (tex) {
    if (!(tex.shape[0] >= 1 && tex.shape[0] <= MAX_IMAGE_SIDE && tex.shape[1] >= 1 && tex.shape[1] <= MAX_IMAGE_SIDE))
      bad(f.type, `a texture is 1 to ${MAX_IMAGE_SIDE} pixels on a side, got [${tex.shape}]`);
    arrays.uvs = sceneArray(uv!);
    arrays.texture = sceneArray(tex);
  }
  return arrays;
}

/** The pixels each image form takes: [dtype, rank] pairs. */
const FORM_PIXELS: Record<string, [number, number][]> = {
  [FORM_COLOR]: [[DTYPE.u8, 3], [DTYPE.u8, 2]],
  [FORM_DEPTH]: [[DTYPE.f32, 2], [DTYPE.u16, 2]],
  [FORM_SEGMENTATION]: [[DTYPE.u16, 2], [DTYPE.u8, 2]],
  [FORM_ENCODED]: [[DTYPE.u8, 1]],
};

function decodeSceneImage(f: Frame, params: Params): Arrays {
  const form = params.form as string | undefined;
  if (form === undefined) bad(f.type, "an image part needs 'form'");
  required(f, PART_IMAGE, 'pixels');
  const p = section(f, 'pixels')!;
  if (p.encoding !== 0 || !FORM_PIXELS[form].some(([dt, rank]) => dt === p.dtype && rank === p.shape.length))
    bad(f.type, `the pixels of a ${form} image cannot be dtype ${p.dtype} shape [${p.shape}]`);
  if (form === FORM_COLOR && p.shape.length === 3 && p.shape[2] !== 4) bad(f.type, `colour pixels are (H, W, 4) or (H, W), got [${p.shape}]`);
  if (form === FORM_ENCODED) {
    if (p.shape[0] > MAX_ENCODED_BYTES) bad(f.type, `an encoded frame of ${p.shape[0]} bytes, more than ${MAX_ENCODED_BYTES}`);
    if (params.format === undefined) bad(f.type, "an encoded image needs 'format'");
  } else if (p.shape[0] > MAX_IMAGE_SIDE || p.shape[1] > MAX_IMAGE_SIDE) {
    bad(f.type, `an image is at most ${MAX_IMAGE_SIDE} pixels on a side, got [${p.shape}]`);
  }
  return { pixels: sceneArray(p) };
}

function decodeSceneCamera(f: Frame, params: Params): Arrays {
  for (const key of CAMERA_REQUIRED) if (!(key in params)) bad(f.type, `a camera part needs '${key}'`);
  return {};
}

function scenePartKeys(f: Frame, part: ScenePart): Params {
  const out: Params = {};
  for (const key of PART_KEYS[part]) {
    const v = f.meta[key];
    if (v === undefined || v === null) continue;
    const [ok, what] = part === PART_CAMERA && key === 'width' ? CAMERA_WIDTH : KEY_CHECKS[key];
    if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
    out[key] = v;
  }
  return out;
}

/** The stamp sections of a message: `stamp` i64 (T,), and where `rows` is given `stamps` i64 (rows, T), T from 1 to 8. */
function stampSections(f: Frame, isStatic: boolean, rows?: number): { stamp?: Section; stamps?: Section } {
  const s = section(f, 'stamp');
  const ss = rows !== undefined ? section(f, 'stamps') : undefined;
  if (s && ss) bad(f.type, "'stamp' and 'stamps' together");
  if (isStatic && (s || ss)) bad(f.type, 'a static item carries no stamp');
  if (s && (s.encoding !== 0 || s.dtype !== DTYPE.i64 || s.shape.length !== 1 || !(s.shape[0] >= 1 && s.shape[0] <= MAX_TIMELINES)))
    bad(f.type, `'stamp' must be i64 (T,) with T from 1 to ${MAX_TIMELINES}, got dtype ${s.dtype} shape [${s.shape}]`);
  if (ss && (ss.encoding !== 0 || ss.dtype !== DTYPE.i64 || ss.shape.length !== 2 || ss.shape[0] !== rows
      || !(ss.shape[1] >= 1 && ss.shape[1] <= MAX_TIMELINES)))
    bad(f.type, `'stamps' must be i64 (${rows}, T) with T from 1 to ${MAX_TIMELINES}, got dtype ${ss.dtype} shape [${ss.shape}]`);
  return { stamp: s, stamps: ss };
}

/** The keys every scene message starts with: sub, view, seq, stamp_ns. */
function sceneCommon(f: Frame): { sub: number; view: string; seq: number; stamp_ns: number } {
  return {
    sub: req<number>(f, 'sub', isU32, 'a uint32'),
    view: req<string>(f, 'view', isName, 'a view name'),
    seq: req<number>(f, 'seq', isUint, 'an unsigned integer'),
    stamp_ns: req<number>(f, 'stamp_ns', isUint, 'an unsigned integer'),
  };
}

/** static, state and cursor, in that order. */
function sceneFlags(f: Frame): { static: boolean; state: boolean; cursor: number | undefined } {
  return {
    static: opt(f, 'static', isBool, 'a boolean', false),
    state: opt(f, 'state', isBool, 'a boolean', false),
    cursor: opt<number | undefined>(f, 'cursor', isU32, 'a uint32', undefined),
  };
}

function writeFlags(meta: Record<string, unknown>, m: { static?: boolean; state?: boolean; cursor?: number }): void {
  if (m.static) meta.static = true;
  if (m.state) meta.state = true;
  if (m.cursor !== undefined && m.cursor !== null) meta.cursor = m.cursor;
}

function stampSection(name: string, a: BigInt64Array, shape: number[]): Section {
  return { name, dtype: DTYPE.i64, shape, encoding: 0, bytes: bytesOf(a) };
}

export function decodeSceneUpdate(f: Frame): SceneUpdateMsg {
  const t = f.type;
  const { sub, view, seq, stamp_ns } = sceneCommon(f);
  const path = req<string>(f, 'path', entityPathOk, 'an entity path');
  const part = req<string>(f, 'part', isStr, 'a string');
  if (!(SCENE_PARTS as readonly string[]).includes(part)) bad(t, `unknown part '${part}'`);
  const flags = sceneFlags(f);
  const axes = opt<PayloadAxisList | undefined>(f, 'axes', Array.isArray, 'a list', undefined);
  const items = opt<string | undefined>(f, 'items', isName, 'an item space name', undefined);
  const split = opt<SplitKey | undefined>(f, 'split', isObj, 'an object', undefined);
  const p = part as ScenePart;
  let params: Params;
  let arrays: Arrays;
  if (split !== undefined) {
    ({ params, arrays } = decodePiece(f, p, split as unknown as Record<string, unknown>));
  } else {
    params = scenePartKeys(f, p);
    switch (p) {
      case PART_POINTS: arrays = decodeScenePoints(f, params); break;
      case PART_LINES: arrays = decodeSceneLines(f, params); break;
      case PART_ARROWS: arrays = decodeSceneArrows(f); break;
      case PART_BOXES: arrays = decodeSceneBoxes(f, params); break;
      case PART_MESH: arrays = decodeSceneMesh(f); break;
      case PART_IMAGE: arrays = decodeSceneImage(f, params); break;
      default: arrays = decodeSceneCamera(f, params);
    }
  }
  const { stamp } = stampSections(f, flags.static);
  if (params.stats) params.stats = (params.stats as (number | null)[]).map((x) => (x === null ? NaN : x));
  const m: SceneUpdateMsg = {
    type: 'scene_update', sub, view, seq, stamp_ns, path, part: p, arrays,
    params: params as SceneUpdateMsg['params'],
    static: flags.static,
    state: flags.state,
    axes,
    items,
  };
  if (flags.cursor !== undefined) m.cursor = flags.cursor;
  if (split !== undefined) m.split = split;
  if (stamp) m.stamp = i64(stamp);
  return m;
}

type PayloadAxisList = NonNullable<SceneUpdateMsg['axes']>;

export function decodeSceneTransforms(f: Frame): SceneTransformsMsg {
  const { sub, view, seq, stamp_ns } = sceneCommon(f);
  const flags = sceneFlags(f);
  const paths = req<string[]>(f, 'paths', (v) => Array.isArray(v) && v.length >= 1 && v.length <= MAX_TRANSFORMS
    && v.every((p) => typeof p === 'string' && p !== '' && entityPathOk(p)),
  `a list of 1 to ${MAX_TRANSFORMS} entity paths other than the root`);
  if (new Set(paths).size !== paths.length) bad(f.type, "'paths' names an entity twice");
  if (!section(f, 'matrices')) bad(f.type, "a scene_transforms needs a 'matrices' section");
  const mat = partSection(f, 'matrices', DTYPE.f32, [paths.length, 16])!;
  const { stamp, stamps } = stampSections(f, flags.static, paths.length);
  const m: SceneTransformsMsg = {
    type: 'scene_transforms', sub, view, seq, stamp_ns, paths, matrices: f32(mat), static: flags.static, state: flags.state,
  };
  if (flags.cursor !== undefined) m.cursor = flags.cursor;
  if (stamp) m.stamp = i64(stamp);
  if (stamps) {
    m.stamps = i64(stamps);
    m.stamps_t = stamps.shape[1];
  }
  return m;
}

export function decodeSceneClear(f: Frame): SceneClearMsg {
  const { sub, view, seq, stamp_ns } = sceneCommon(f);
  const path = req<string>(f, 'path', entityPathOk, 'an entity path');
  const recursive = opt(f, 'recursive', isBool, 'a boolean', false);
  const parts = opt<string[] | undefined>(f, 'parts', (v) => Array.isArray(v) && v.length > 0 && v.every((x) => CLEAR_PARTS.includes(x as string)),
    'a list of parts', undefined);
  const reset = opt(f, 'reset', isBool, 'a boolean', false);
  const flags = sceneFlags(f);
  if (reset && path !== '') bad(f.type, 'a reset clears the root: its \'path\' is ""');
  const { stamp } = stampSections(f, flags.static);
  const c: SceneClearMsg = { type: 'scene_clear', sub, view, seq, stamp_ns, path, recursive, parts, reset, static: flags.static, state: flags.state };
  if (flags.cursor !== undefined) c.cursor = flags.cursor;
  if (stamp) c.stamp = i64(stamp);
  return c;
}

const isClassEntry = (e: unknown) => {
  if (!isObj(e)) return false;
  const o = e as Record<string, unknown>;
  return isInt(o.id) && o.id >= 0 && o.id <= MAX_CLASS_ID && typeof o.label === 'string' && utf8.encode(o.label).length <= NAME_MAX && isRgba(o.color);
};

export function decodeSceneClasses(f: Frame): SceneClassesMsg {
  const { sub, view, seq, stamp_ns } = sceneCommon(f);
  const flags = sceneFlags(f);
  const classes = req<SceneClass[]>(f, 'classes', (v) => Array.isArray(v) && v.length <= MAX_CLASSES && v.every(isClassEntry),
    `a list of at most ${MAX_CLASSES} {id, label, color} entries`);
  if (!isAscending(classes.map((e) => e.id))) bad(f.type, "'classes' must be in ascending id order without repeats");
  const { stamp } = stampSections(f, flags.static);
  const m: SceneClassesMsg = { type: 'scene_classes', sub, view, seq, stamp_ns, classes, static: flags.static, state: flags.state };
  if (flags.cursor !== undefined) m.cursor = flags.cursor;
  if (stamp) m.stamp = i64(stamp);
  return m;
}

function sceneUpdateMeta(m: SceneUpdateMsg): Record<string, unknown> {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, path: m.path, part: m.part };
  for (const [key, v] of Object.entries(m.params)) {
    if (v === undefined || v === null) continue;
    meta[key] = key === 'stats' ? (v as number[]).map((x) => (Number.isFinite(x) ? x : null)) : v;
  }
  writeFlags(meta, m);
  if (m.axes !== undefined) meta.axes = m.axes;
  if (m.items !== undefined) meta.items = m.items;
  if (m.split !== undefined) meta.split = m.split;
  return meta;
}

export function encodeSceneUpdate(m: SceneUpdateMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta = sceneUpdateMeta(m);
  const sections: Section[] = [];
  for (const name of PART_SECTIONS[m.part] ?? []) {
    const a = m.arrays[name];
    if (a) sections.push({ name, dtype: a.dtype, shape: a.shape, encoding: 0, bytes: bytesOf(a.data) });
  }
  if (m.stamp) sections.push(stampSection('stamp', m.stamp, [m.stamp.length]));
  return frameAt(MSG.scene_update, meta, sections, minor);
}

export function encodeSceneTransforms(m: SceneTransformsMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, paths: m.paths };
  writeFlags(meta, m);
  const k = m.paths.length;
  const sections: Section[] = [f32Section('matrices', m.matrices, [k, 16])];
  if (m.stamps) sections.push(stampSection('stamps', m.stamps, [k, m.stamps_t ?? (k ? m.stamps.length / k : 0)]));
  else if (m.stamp) sections.push(stampSection('stamp', m.stamp, [m.stamp.length]));
  return frameAt(MSG.scene_transforms, meta, sections, minor);
}

export function encodeSceneClear(m: SceneClearMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, path: m.path };
  for (const key of ['recursive', 'reset'] as const) if (m[key]) meta[key] = true;
  writeFlags(meta, m);
  if (m.parts !== undefined) meta.parts = m.parts;
  return frameAt(MSG.scene_clear, meta, m.stamp ? [stampSection('stamp', m.stamp, [m.stamp.length])] : [], minor);
}

export function encodeSceneClasses(m: SceneClassesMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, seq: m.seq, stamp_ns: m.stamp_ns, classes: m.classes };
  writeFlags(meta, m);
  return frameAt(MSG.scene_classes, meta, m.stamp ? [stampSection('stamp', m.stamp, [m.stamp.length])] : [], minor);
}

// --- 3.2 draft: timelines, text logs, recordings, the cursor (python: the block "timelines and stamps on the 3.0 and 3.1 messages") ---

const TIMELINE_NAME = /^[A-Za-z0-9_.-]{1,63}$/;
const STAMP_TEXT = /^-?[0-9]+$/;
const I64_MIN = -(1n << 63n);
const I64_MAX = (1n << 63n) - 1n;
const isTimelineName = (v: unknown) => typeof v === 'string' && TIMELINE_NAME.test(v);

/** Stamp text: decimal digits with an optional leading minus, at most 20 characters, a value of a signed 64-bit integer. */
export function stampTextOk(v: unknown): boolean {
  if (typeof v !== 'string' || v.length === 0 || v.length > STAMP_TEXT_MAX || !/^[\x00-\x7f]*$/.test(v) || !STAMP_TEXT.test(v)) return false;
  const n = BigInt(v);
  return n >= I64_MIN && n <= I64_MAX;
}

const isTimelineEntry = (t: unknown) => isObj(t) && isTimelineName((t as Record<string, unknown>).name)
  && ((t as Record<string, unknown>).kind === TIMELINE_SEQUENCE || (t as Record<string, unknown>).kind === TIMELINE_TIME)
  && ((t as Record<string, unknown>).epoch == null || isBool((t as Record<string, unknown>).epoch));

/** The VIEWS key "timelines": 2 to 8 {name, kind, epoch?}, step and time first. */
function isTimelines(v: unknown): boolean {
  if (!(Array.isArray(v) && v.length >= 2 && v.length <= MAX_TIMELINES)) return false;
  if (!v.every(isTimelineEntry)) return false;
  const names = v.map((t) => (t as TimelineInfo).name);
  return new Set(names).size === names.length && v[0].name === 'step' && v[0].kind === TIMELINE_SEQUENCE
    && v[1].name === 'time' && v[1].kind === TIMELINE_TIME;
}

const isRecorded = (v: unknown) => Array.isArray(v) && v.length <= MAX_RECORDED && v.every(isObj);

/** A session's key "recording": {id, seekable, complete}. */
const isRecordingRef = (v: unknown) => isObj(v) && isSessionId((v as Record<string, unknown>).id)
  && isBool((v as Record<string, unknown>).seekable) && isBool((v as Record<string, unknown>).complete);

const isLoop = (v: unknown) => Array.isArray(v) && v.length === 2 && v.every(stampTextOk) && BigInt(v[0]) <= BigInt(v[1]);
const isSpeed = (v: unknown) => isFinite_(v) && (v as number) >= MIN_SPEED && (v as number) <= MAX_SPEED;
const isPace = (v: unknown) => v === PACE_RECORDED || v === PACE_EVEN;

/** Lines of a text message, decoded as UTF-8. */
export function textLines(m: TextMsg): string[] {
  const dec = new TextDecoder();
  const out: string[] = [];
  for (let i = 0; i + 1 < m.offsets.length; i++) out.push(dec.decode(m.text.subarray(m.offsets[i], m.offsets[i + 1])));
  return out;
}

/** A text message of `lines` at `levels` (one each), copied into its own buffers. */
export function textFromLines(sub: number, view: string, start: number, total: number, stamp_ns: number, lines: string[], levels: ArrayLike<number>,
  extra: Partial<Pick<TextMsg, 'dropped' | 'stamp' | 'stamps' | 'stamps_t' | 'cursor'>> = {}): TextMsg {
  const data = lines.map((l) => utf8.encode(l));
  const offsets = new Int32Array(data.length + 1);
  data.forEach((d, i) => { offsets[i + 1] = offsets[i] + d.length; });
  const text = new Uint8Array(offsets[data.length]);
  data.forEach((d, i) => text.set(d, offsets[i]));
  return { type: 'text', sub, view, start, total, stamp_ns, text, offsets, levels: Uint8Array.from(levels), ...extra };
}

export function decodeText(f: Frame): TextMsg {
  const t = section(f, 'text');
  if (!t || t.encoding !== 0 || t.dtype !== DTYPE.u8 || t.shape.length !== 1) bad(f.type, "a text message needs a (L,) u8 'text' section");
  const o = section(f, 'offsets');
  if (!o || o.encoding !== 0 || o.dtype !== DTYPE.i32 || o.shape.length !== 1 || o.shape[0] < 2)
    bad(f.type, "a text message needs an (M + 1,) i32 'offsets' section, M >= 1");
  const n = o.shape[0] - 1;
  if (n > TEXT_MAX_LINES) bad(f.type, `${n} lines, more than ${TEXT_MAX_LINES} in one message`);
  const lv = partSection(f, 'levels', DTYPE.u8, [n]);
  if (!lv) bad(f.type, "a text message needs an (M,) u8 'levels' section");
  const sub = req<number>(f, 'sub', isU32, 'a uint32');
  const view = req<string>(f, 'view', isName, 'a view name');
  const start = req<number>(f, 'start', isUint, 'an unsigned integer');
  const total = req<number>(f, 'total', isUint, 'an unsigned integer');
  const stamp_ns = req<number>(f, 'stamp_ns', isUint, 'an unsigned integer');
  const dropped = opt<number | undefined>(f, 'dropped', isUint, 'an unsigned integer', undefined);
  const cursor = opt<number | undefined>(f, 'cursor', isU32, 'a uint32', undefined);
  if (start + n > total) bad(f.type, `text: ${n} lines from ${start} with total ${total}`);
  const offs = i32(o);
  if (offs[0] !== 0 || offs[n] !== t.shape[0]) bad(f.type, `'offsets' must start at 0 and end at the ${t.shape[0]} bytes of 'text'`);
  let minStep = Infinity;
  let maxStep = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = offs[i + 1] - offs[i];
    if (d < minStep) minStep = d;
    if (d > maxStep) maxStep = d;
  }
  if (minStep < 0) bad(f.type, "'offsets' must not decrease");
  if (maxStep > TEXT_LINE_MAX) bad(f.type, `a line is at most ${TEXT_LINE_MAX} bytes`);
  const levels = u8(lv);
  for (const x of levels) if (x > LEVEL_ERROR) bad(f.type, 'a level is 0 (debug), 1 (info), 2 (warning) or 3 (error)');
  for (let i = 0; i < n; i++) {
    try {
      utf8Strict.decode(t.bytes.subarray(offs[i], offs[i + 1]));
    } catch {
      bad(f.type, `line ${offs[i]}..${offs[i + 1]} of 'text' is not UTF-8`);
    }
  }
  const { stamp, stamps } = stampSections(f, false, n);
  const m: TextMsg = { type: 'text', sub, view, start, total, stamp_ns, text: u8(t), offsets: offs, levels };
  if (dropped !== undefined) m.dropped = dropped;
  if (cursor !== undefined) m.cursor = cursor;
  if (stamp) m.stamp = i64(stamp);
  if (stamps) {
    m.stamps = i64(stamps);
    m.stamps_t = stamps.shape[1];
  }
  return m;
}

export function encodeText(m: TextMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const n = m.levels.length;
  const meta: Record<string, unknown> = { sub: m.sub, view: m.view, start: m.start, total: m.total, stamp_ns: m.stamp_ns };
  if (m.dropped !== undefined) meta.dropped = m.dropped;
  if (m.cursor !== undefined) meta.cursor = m.cursor;
  const sections: Section[] = [
    { name: 'text', dtype: DTYPE.u8, shape: [m.text.length], encoding: 0, bytes: m.text },
    { name: 'offsets', dtype: DTYPE.i32, shape: [m.offsets.length], encoding: 0, bytes: bytesOf(m.offsets) },
    { name: 'levels', dtype: DTYPE.u8, shape: [n], encoding: 0, bytes: m.levels },
  ];
  if (m.stamps) sections.push(stampSection('stamps', m.stamps, [n, m.stamps_t ?? (n ? m.stamps.length / n : 0)]));
  else if (m.stamp) sections.push(stampSection('stamp', m.stamp, [m.stamp.length]));
  return frameAt(MSG.text, meta, sections, minor);
}

function isRecordingEntry(e: unknown): boolean {
  if (!isObj(e)) return false;
  const o = e as Record<string, unknown>;
  if (!(isSessionId(o.id) && isStr(o.title) && isStr(o.script) && isUint(o.bytes) && isUint(o.items) && isBool(o.complete))) return false;
  for (const key of ['created_unix_ms', 'ended_unix_ms']) if (o[key] != null && !isUint(o[key])) return false;
  const tl = o.timelines;
  return Array.isArray(tl) && tl.length <= MAX_TIMELINES && tl.every((t) => isTimelineEntry(t) && stampTextOk((t as Record<string, unknown>).min)
    && stampTextOk((t as Record<string, unknown>).max));
}

export function decodeRecordings(f: Frame): RecordingsMsg {
  return {
    type: 'recordings',
    recordings: req<RecordingEntry[]>(f, 'recordings', (v) => Array.isArray(v) && v.length <= MAX_RECORDINGS && v.every(isRecordingEntry),
      `a list of at most ${MAX_RECORDINGS} recordings`),
  };
}

export function decodeRecordingOpen(f: Frame): RecordingOpenMsg {
  return {
    type: 'recording_open',
    request_id: req(f, 'request_id', isU32, 'a uint32'),
    recording: req(f, 'recording', isSessionId, 'a recording id'),
  };
}

function cursorBad(msg: string): never {
  throw new MessageError('bad_cursor', msg, MSG.cursor);
}

/** Only the fields of the op are kept; a request_id or op that is missing or wrong is bad_request, any other field bad_cursor. */
export function decodeCursor(f: Frame): CursorMsg {
  const m = f.meta;
  const c: CursorMsg = {
    type: 'cursor',
    request_id: req(f, 'request_id', isU32, 'a uint32'),
    op: req(f, 'op', (v) => CURSOR_OPS.includes(v as string), 'seek, play, pause, step or follow'),
  };
  const get = <T>(key: string, ok: (v: unknown) => boolean, what: string, need: boolean): T | undefined => {
    const v = m[key];
    if (v === undefined || v === null) {
      if (need) cursorBad(`'${c.op}' needs '${key}'`);
      return undefined;
    }
    if (!ok(v)) cursorBad(`'${key}' must be ${what}, got ${JSON.stringify(v)}`);
    return v as T;
  };
  if (c.op === CURSOR_SEEK || c.op === CURSOR_PLAY || c.op === CURSOR_STEP) c.timeline = get<string>('timeline', isTimelineName, 'a timeline name', true);
  if (c.op === CURSOR_SEEK) c.value = get<string>('value', stampTextOk, 'stamp text', true);
  if (c.op === CURSOR_STEP)
    c.count = get<number>('count', (v) => isInt(v) && v !== 0 && Math.abs(v) <= MAX_CURSOR_STEP, `an integer from -${MAX_CURSOR_STEP} to ${MAX_CURSOR_STEP}, not 0`, true);
  if (c.op === CURSOR_PLAY) {
    c.speed = get<number>('speed', isSpeed, `a number from ${MIN_SPEED} to ${MAX_SPEED}`, false);
    c.pace = get<CursorMsg['pace']>('pace', isPace, 'recorded or even', false);
    c.loop = get<[string, string]>('loop', isLoop, '[start, end] as stamp text', false);
  }
  return drop(c as unknown as Record<string, unknown>) as unknown as CursorMsg;
}

const isRanges = (v: unknown) => isObj(v) && Object.entries(v as Record<string, unknown>).every(([k, r]) => isTimelineName(k) && isLoop(r));

export function decodeTimeState(f: Frame): TimeStateMsg {
  const request_id = opt<number | undefined>(f, 'request_id', isU32, 'a uint32', undefined);
  const session = req<string>(f, 'session', isSessionId, 'a session id');
  const generation = req<number>(f, 'generation', isU32, 'a uint32');
  const state = req<TimeStateMsg['state']>(f, 'state', (v) => v === TIME_FOLLOWING || v === TIME_PAUSED || v === TIME_PLAYING, 'following, paused or playing');
  const timeline = req<string>(f, 'timeline', isTimelineName, 'a timeline name');
  const cursor = opt<string | null>(f, 'cursor', stampTextOk, 'stamp text or null', null);
  const speed = opt<number | undefined>(f, 'speed', isSpeed, `a number from ${MIN_SPEED} to ${MAX_SPEED}`, undefined);
  const pace = opt<TimeStateMsg['pace']>(f, 'pace', isPace, 'recorded or even', undefined);
  const loop = opt<[string, string] | undefined>(f, 'loop', isLoop, '[start, end] as stamp text', undefined);
  const ranges = req<TimeStateMsg['ranges']>(f, 'ranges', isRanges, 'an object of [min, max] stamp text per timeline');
  const live = req<boolean>(f, 'live', isBool, 'a boolean');
  if ((state === TIME_FOLLOWING) !== (cursor === null)) bad(f.type, 'the cursor is null exactly while following');
  const m: TimeStateMsg = { type: 'time_state', session, generation, state, timeline, cursor, ranges, live };
  if (request_id !== undefined) m.request_id = request_id;
  if (speed !== undefined) m.speed = speed;
  if (pace !== undefined) m.pace = pace;
  if (loop !== undefined) m.loop = loop;
  return m;
}

// --- 3.2 draft: split updates ---

type Raw = Record<string, unknown>;

/** The checks of a piece of a split update, in this order: the keys of `split`, the part's keys in piece 0, the piece's array sections. */
function decodePiece(f: Frame, part: ScenePart, split: Raw): { params: Params; arrays: Arrays } {
  const need = <T>(key: string, ok: (v: unknown) => boolean, what: string): T => {
    const v = split[key];
    if (v === undefined || v === null || !ok(v)) bad(f.type, `'split.${key}' must be ${what}, got ${JSON.stringify(v)}`);
    return v as T;
  };
  const rowsMap = (v: unknown) => isObj(v) && Object.values(v as Raw).every(isUint);
  need('update', isUint, 'an unsigned integer');
  const pieces = need<number>('pieces', (v) => isInt(v) && v >= 2 && v <= MAX_PIECES, `an integer from 2 to ${MAX_PIECES}`);
  const piece = need<number>('piece', (v) => isInt(v) && v >= 0 && v < pieces, "a piece number below 'pieces'");
  const first = need<Record<string, number>>('first', rowsMap, 'an object of row numbers');
  let rows: Record<string, number> | undefined;
  if (piece === 0) rows = need<Record<string, number>>('rows', rowsMap, 'an object of row counts');
  else if (split.rows !== undefined && split.rows !== null) bad(f.type, "only piece 0 carries 'split.rows'");
  const params = piece === 0 ? scenePartKeys(f, part) : {};
  const arrays: Arrays = {};
  for (const s of f.sections) {
    if (s.name === 'stamp' || !PART_SECTIONS[part].includes(s.name)) continue; // the stamp, or a section of no array: ignored
    if (s.encoding !== 0 || s.shape.length < 1) bad(f.type, `a piece of '${s.name}' is a raw array of at least one dimension`);
    arrays[s.name] = pieceArray(s);
  }
  const fk = Object.keys(first).sort();
  const ak = Object.keys(arrays).sort();
  if (fk.length !== ak.length || fk.some((k, i) => k !== ak[i])) bad(f.type, `'split.first' names [${fk}], the piece holds [${ak}]`);
  if (rows !== undefined) {
    for (const name of Object.keys(rows)) if (!PART_SECTIONS[part].includes(name)) bad(f.type, `'split.rows' names '${name}', not an array of a ${part} part`);
    for (const [name, a] of Object.entries(arrays))
      if (!(name in rows) || first[name] + a.shape[0] > rows[name]) bad(f.type, `the rows of '${name}' in piece 0 pass 'split.rows'`);
  }
  return { params, arrays };
}

/** A piece's array: typed where this codec has a typed array, else its raw bytes. */
function pieceArray(s: Section): SceneArray {
  if (s.dtype === DTYPE.f32 || s.dtype === DTYPE.i32 || s.dtype === DTYPE.u8 || s.dtype === DTYPE.u16) return sceneArray(s);
  return { dtype: s.dtype, shape: s.shape, data: s.bytes };
}

const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');

interface Assembled { dtype: number; trailing: number[]; buf: Uint8Array; ranges: [number, number][] }

/** One split update being put together. */
class Assembly {
  readonly key: string;
  readonly rows: Record<string, number>;
  readonly seen = new Set<number>();
  readonly arrays = new Map<string, Assembled>();
  nbytes = 0;

  constructor(readonly p0: SceneUpdateMsg) {
    this.key = Assembly.keyOf(p0);
    this.rows = { ...p0.split!.rows! };
  }

  static keyOf(p: SceneUpdateMsg): string {
    return JSON.stringify([p.path, p.part, p.seq, p.split!.pieces, p.stamp ? hex(bytesOf(p.stamp)) : null]);
  }

  add(p: SceneUpdateMsg): void {
    const sp = p.split!;
    if (Assembly.keyOf(p) !== this.key) bad(MSG.scene_update, "a piece's path, part, seq, pieces or stamp differ from its update's piece 0");
    if (this.seen.has(sp.piece)) bad(MSG.scene_update, `piece ${sp.piece} of update ${sp.update} arrived twice`);
    this.seen.add(sp.piece);
    for (const [name, s] of Object.entries(p.arrays)) {
      if (!(name in this.rows)) bad(MSG.scene_update, `piece ${sp.piece} holds '${name}', which 'split.rows' does not name`);
      const item = DTYPE_SIZE[s.dtype];
      if (item === undefined) bad(MSG.scene_update, `a piece of '${name}' has the unknown dtype ${s.dtype}`);
      const trailing = s.shape.slice(1);
      const rowbytes = trailing.reduce((x, y) => x * y, 1) * item;
      let a = this.arrays.get(name);
      if (a === undefined) {
        const total = this.rows[name] * rowbytes;
        if (this.nbytes + total > MAX_UPDATE_BYTES) bad(MSG.scene_update, `a split update of more than ${MAX_UPDATE_BYTES} bytes`);
        this.nbytes += total;
        a = { dtype: s.dtype, trailing, buf: new Uint8Array(total), ranges: [] };
        this.arrays.set(name, a);
      } else if (a.dtype !== s.dtype || a.trailing.length !== trailing.length || a.trailing.some((x, i) => x !== trailing[i])) {
        bad(MSG.scene_update, `the pieces of '${name}' differ in dtype or shape`);
      }
      const first = sp.first[name];
      const n = s.shape[0];
      if (first + n > this.rows[name]) bad(MSG.scene_update, `rows ${first} to ${first + n} of '${name}' pass its ${this.rows[name]} rows`);
      if (a.ranges.some(([f0, n0]) => first < f0 + n0 && f0 < first + n)) bad(MSG.scene_update, `two pieces hold the same rows of '${name}'`);
      a.ranges.push([first, n]);
      a.buf.set(bytesOf(s.data), first * rowbytes); // the buffers are the assembly's own: a decoded frame's arrays are views
    }
  }

  done(): boolean {
    if (this.seen.size < this.p0.split!.pieces) return false;
    for (const [name, rows] of Object.entries(this.rows)) {
      const a = this.arrays.get(name);
      const got = a ? a.ranges.reduce((x, [, n]) => x + n, 0) : 0;
      if (got !== rows) bad(MSG.scene_update, `the pieces ended before every row of '${name}' arrived`);
    }
    return true;
  }

  /** The whole update, checked by the normal scene_update decoder on a frame of the whole arrays plus the stamp. */
  whole(): SceneUpdateMsg {
    const p0 = this.p0;
    const meta = sceneUpdateMeta(p0);
    delete meta.split;
    const sections: Section[] = [];
    for (const name of PART_SECTIONS[p0.part]) {
      const a = this.arrays.get(name);
      if (a) sections.push({ name, dtype: a.dtype, shape: [this.rows[name], ...a.trailing], encoding: 0, bytes: a.buf });
    }
    if (p0.stamp) sections.push({ name: 'stamp', dtype: DTYPE.i64, shape: [p0.stamp.length], encoding: 0, bytes: bytesOf(p0.stamp).slice() });
    return decodeSceneUpdate({ type: MSG.scene_update, flags: 0, major: PROTOCOL_MAJOR, minor: PROTOCOL_MINOR_3_2, meta, sections });
  }
}

/**
 * A receiver's split updates of one subscription. `add` takes each piece and returns the whole update, checked as an unsplit
 * one, once every piece and every row has arrived, else null. A newer update of the same entity part discards an incomplete
 * one, at most MAX_INCOMPLETE_UPDATES are held (the oldest goes first), and a piece of an update it does not hold is dropped
 * and counted. A piece that contradicts its update throws MessageError (bad_request) and the update is discarded.
 */
export class SplitUpdates {
  /** update -> assembly, oldest first. */
  readonly open = new Map<number, Assembly>();
  dropped = 0;
  discarded = 0;

  add(piece: SceneUpdateMsg): SceneUpdateMsg | null {
    const sp = piece.split;
    if (!sp) throw new MessageError('bad_request', 'not a piece of a split update', MSG.scene_update);
    const uid = sp.update;
    if (sp.piece === 0) {
      for (const [k, a] of [...this.open]) {
        if (k === uid || (a.p0.path === piece.path && a.p0.part === piece.part)) {
          this.open.delete(k);
          this.discarded++;
        }
      }
      this.open.set(uid, new Assembly(piece));
      while (this.open.size > MAX_INCOMPLETE_UPDATES) {
        this.open.delete(this.open.keys().next().value as number);
        this.discarded++;
      }
    }
    const a = this.open.get(uid);
    if (a === undefined) {
      this.dropped++;
      return null;
    }
    try {
      a.add(piece);
      if (!a.done()) return null;
      this.open.delete(uid);
      return a.whole();
    } catch (e) {
      if (e instanceof MessageError) {
        this.open.delete(uid);
        this.discarded++;
      }
      throw e;
    }
  }
}

const VIEW_CAMERA_KEYS = ['target', 'distance', 'azimuth', 'elevation'];

/** A scene's "view_camera" as a viewer reads it: the orbit keys or the entity path of a camera, or undefined for none, for a 2-D scene, and for a value it cannot read. */
function viewCamera(sc: Record<string, unknown>, space: string): ViewCameraOrbit | string | undefined {
  const vc = sc.view_camera;
  if (vc === undefined || vc === null || space !== SPACE_3D) return undefined;
  if (typeof vc === 'string') return vc !== '' && entityPathOk(vc) ? vc : undefined;
  if (!isObj(vc)) return undefined;
  const o = vc as Record<string, unknown>;
  if (!Object.keys(o).every((k) => VIEW_CAMERA_KEYS.includes(k))) return undefined;
  const t = o.target;
  if (t !== undefined && t !== null && !(Array.isArray(t) && t.length === 3 && t.every(isFinite_))) return undefined;
  if (o.distance !== undefined && o.distance !== null && !isPositive(o.distance)) return undefined;
  if (o.azimuth !== undefined && o.azimuth !== null && !isFinite_(o.azimuth)) return undefined;
  const e = o.elevation;
  if (e !== undefined && e !== null && !(isFinite_(e) && (e as number) >= -89 && (e as number) <= 89)) return undefined;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) if (v !== null && v !== undefined) out[k] = v;
  return out as ViewCameraOrbit;
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
  return { space: space as SceneViewInfo['space'], up: up as string, units: units as string, view_camera: viewCamera(o, space as string) };
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
