// Frame-level binary codec, canonical JSON, typed section views, and shared
// field validators for wire protocol v3 (include/caliper/live_net.h).

import {
  CLOSE, DTYPE, DTYPE_SIZE, LIMITS, LIMITS_31, LIMITS_32, MAX_TIMELINES, MSG, PROTOCOL_MAJOR, PROTOCOL_MINOR,
  type SessionIdentity,
} from './messages.js';

export const MAGIC = 0x574c4143; // "CALW" read as a little-endian uint32
export const HEADER_BYTES = 24;
export const SECTION_ALIGN = 8;

export const NAME_MAX = 63;
export const MAX_STATS = 64;
export const DTYPE_NAME: Record<number, string> = { 0: 'f32', 1: 'f16', 2: 'bf16', 3: 'i64', 4: 'i32', 5: 'u8', 6: 'u16' };

const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const PATH_SEGMENT = /^[A-Za-z0-9_.-]{1,63}$/;

export const utf8 = new TextEncoder();
export const utf8Strict = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

// JSON.parse erases the distinction between 1 and 1.0. Structured integer
// fields must retain that distinction, as in the Python and C++ codecs.
export const numberSources = new WeakMap<object, Record<string, unknown>>();

function numericSources(json: string): Record<string, unknown> {
  return JSON.parse(json.replace(/"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g,
    (token) => token.startsWith('"') ? token : JSON.stringify(token)));
}

export const sourceField = (source: unknown, key: string | number): unknown =>
  source !== null && typeof source === 'object' ? (source as Record<string, unknown>)[key] : undefined;

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

export type Raw = Record<string, unknown>;

const align = (n: number) => (n + SECTION_ALIGN - 1) & ~(SECTION_ALIGN - 1);

function depth(v: unknown, d = 1): number {
  if (Array.isArray(v)) return Math.max(d, ...v.map((x) => depth(x, d + 1)));
  if (v !== null && typeof v === 'object') return Math.max(d, ...Object.values(v).map((x) => depth(x, d + 1)));
  return d - 1;
}

export const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);
export const isUint = (v: unknown): v is number => isInt(v) && v >= 0;
export const isU32 = (v: unknown) => isInt(v) && v >= 0 && v <= 0xffffffff;
export const isSafeUint = (v: unknown) => isInt(v) && v >= 0 && v <= Number.MAX_SAFE_INTEGER;
export const isNum = (v: unknown) => typeof v === 'number';
export const isBool = (v: unknown) => typeof v === 'boolean';
export const isFiniteNum = (v: unknown) => typeof v === 'number' && Number.isFinite(v);
export const isNonneg = (v: unknown) => isFiniteNum(v) && (v as number) >= 0;
export const isPositive = (v: unknown) => isFiniteNum(v) && (v as number) > 0;
export const isStr = (v: unknown) => typeof v === 'string';
export const isObj = (v: unknown) => v !== null && typeof v === 'object' && !Array.isArray(v);
export const isStrs = (v: unknown) => Array.isArray(v) && v.every(isStr);
export const isName = (v: unknown) => typeof v === 'string' && v.length > 0 && utf8.encode(v).length <= 63;
export const optName = (v: unknown) => v === undefined || v === null || isName(v);
export const isSessionId = (v: unknown) => typeof v === 'string' && /^[A-Za-z0-9._-]{1,63}$/.test(v);
export const isText = (v: unknown) => typeof v === 'string' && utf8.encode(v).length <= LIMITS_31.textMax;
export const isAscending = (v: number[]) => v.every((x, i) => i === 0 || v[i - 1] < x);
export const present = (v: unknown) => v !== undefined && v !== null;
export const boundedText = (limit: number) => (v: unknown) => typeof v === 'string' && v.length > 0 && utf8.encode(v).length <= limit;
export const intIn = (lo: number, hi: number) => (v: unknown) => isInt(v) && v >= lo && v <= hi;
export const listOf = (ok: (v: unknown) => boolean, most: number, least = 0) => (v: unknown) =>
  Array.isArray(v) && v.length >= least && v.length <= most && v.every((x) => ok(x));

export const isIdentity = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return isSessionId(o.id) && isStr(o.script) && isUint(o.pid) && isUint(o.start_unix_ms);
};

/** An entity path: 0 to 16 segments of [A-Za-z0-9_.-]{1,63} joined by '/', at most 255 bytes, never '.' or '..'. */
export function entityPathOk(v: unknown): boolean {
  if (typeof v !== 'string') return false;
  if (v === '') return true;
  if (v.length > LIMITS_32.pathMax || !/^[\x00-\x7f]*$/.test(v)) return false;
  const segs = v.split('/');
  return segs.length <= LIMITS_32.pathSegmentsMax && segs.every((x) => PATH_SEGMENT.test(x) && x !== '.' && x !== '..');
}

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
  let metaJson: string;
  try {
    metaJson = utf8Strict.decode(bytes.subarray(tableEnd, metaEnd));
    meta = JSON.parse(metaJson);
  } catch (e) {
    throw new WireError('malformed', `metadata: ${(e as Error).message}`);
  }
  if (meta === null || typeof meta !== 'object' || Array.isArray(meta)) throw new WireError('malformed', 'metadata is not a JSON object');
  if (depth(meta) > LIMITS.maxJsonDepth) throw new WireError('malformed', `metadata nested deeper than ${LIMITS.maxJsonDepth}`);
  const m = meta as Record<string, unknown>;
  if (type === MSG.controls || type === MSG.control || type === MSG.control_reply) {
    const sources = numericSources(metaJson!);
    numberSources.set(m, sources);
    for (const key of ['interpretation', 'result', 'intervene', 'interventions'])
      if (isObj(m[key])) numberSources.set(m[key] as object, sources[key] as Record<string, unknown>);
  }
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

// Match Python and net/json.cpp for float exponents, integral floats and -0.0.
function canonicalFloat(value: number): string {
  if (value === 0) return Object.is(value, -0) ? '-0.0' : '0.0';
  const [mantissa, exponent] = value.toExponential().split('e');
  const power = Number(exponent);
  if (power < -4 || power >= 16)
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`;
  const text = JSON.stringify(value);
  return text.includes('.') ? text : `${text}.0`;
}

function canonicalObject(v: Record<string, unknown>, valueText: (value: unknown, key: string) => string): string {
  const keys = Object.keys(v).filter((k) => v[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${valueText(v[k], k)}`).join(',')}}`;
}

/** Canonical JSON has sorted keys. Only interpretation data infers unsafe integers as floats. */
export function canonicalJson(v: unknown, source?: unknown, interpretation = false): string {
  if (v === null || typeof v !== 'object') {
    if (typeof v === 'number' && !Number.isFinite(v)) throw new TypeError('NaN and the infinities are not JSON');
    const text = JSON.stringify(v);
    return typeof v === 'number' && (/[.eE]/.test(text) || (typeof source === 'string' ? /[.eE]/.test(source) : interpretation && (!Number.isSafeInteger(v) || Object.is(v, -0))))
      ? canonicalFloat(v) : text;
  }
  source ??= numberSources.get(v);
  if (Array.isArray(v)) return `[${v.map((x, i) => canonicalJson(x, sourceField(source, i), interpretation)).join(',')}]`;
  return canonicalObject(v as Record<string, unknown>, (value, key) => canonicalJson(value, sourceField(source, key), interpretation));
}

/**
 * Encodes one frame at a minor version (this codec's, 1, by default; 0
 * re-encodes a 3.0 frame byte for byte). Sections are raw arrays;
 * meta.sections is written here.
 */
export function encodeFrame(type: number, meta: Record<string, unknown>, sections: Section[] = [], minor: number = PROTOCOL_MINOR): Uint8Array {
  return frameAt(type, meta, sections, minor);
}

export function frameAt(type: number, meta: Record<string, unknown>, sections: Section[], minor: number): Uint8Array {
  const m: Record<string, unknown> = { ...meta };
  delete m.sections;
  if (sections.length) {
    m.sections = sections.map((s) => ({ name: s.name, dtype: s.dtype, shape: s.shape, ...(s.encoding ? { encoding: s.encoding } : {}) }));
  }
  const source = numberSources.get(meta);
  const interpretationKey = meta.op === 'interpret'
    ? (type === MSG.control ? 'interpretation' : type === MSG.control_reply ? 'result' : undefined) : undefined;
  const mbytes = utf8.encode(canonicalObject(m, (value, key) => canonicalJson(value, sourceField(source, key), key === interpretationKey
    || (type === MSG.control && meta.op === 'intervene' && key === 'intervene')
    || (type === MSG.controls && key === 'interventions'))));
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

export function encodeWithSource(msg: object, type: number, meta: Record<string, unknown>, sections: Section[] = [], minor: number = PROTOCOL_MINOR): Uint8Array {
  const source = numberSources.get(msg);
  if (source) numberSources.set(meta, source);
  return frameAt(type, meta, sections, minor);
}

// --- typed views of sections ------------------------------------------------

export function f32(s: Section): Float32Array {
  const n = s.bytes.byteLength / 4;
  if (LITTLE && s.bytes.byteOffset % 4 === 0) return new Float32Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Float32Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(4 * i, true);
  return out;
}

export function u8(s: Section): Uint8Array {
  return s.bytes;
}

export function i32(s: Section): Int32Array {
  const n = s.bytes.byteLength / 4;
  if (LITTLE && s.bytes.byteOffset % 4 === 0) return new Int32Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Int32Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getInt32(4 * i, true);
  return out;
}

export function u16(s: Section): Uint16Array {
  const n = s.bytes.byteLength / 2;
  if (LITTLE && s.bytes.byteOffset % 2 === 0) return new Uint16Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new Uint16Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getUint16(2 * i, true);
  return out;
}

export function i64(s: Section): BigInt64Array {
  const n = s.bytes.byteLength / 8;
  if (LITTLE && s.bytes.byteOffset % 8 === 0) return new BigInt64Array(s.bytes.buffer, s.bytes.byteOffset, n);
  const out = new BigInt64Array(n);
  const dv = new DataView(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getBigInt64(8 * i, true);
  return out;
}

export const bytesOf = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

export function f32Section(name: string, arr: Float32Array, shape: number[]): Section {
  return { name, dtype: DTYPE.f32, shape, encoding: 0, bytes: new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength) };
}

export function stampSection(name: string, a: BigInt64Array, shape: number[]): Section {
  return { name, dtype: DTYPE.i64, shape, encoding: 0, bytes: bytesOf(a) };
}

// --- message checks -----------------------------------------------------------

export function bad(type: number, msg: string): never {
  throw new MessageError('bad_request', msg, type);
}

export function req<T>(f: Frame, key: string, ok: (v: unknown) => boolean, what: string): T {
  if (!(key in f.meta)) bad(f.type, `missing '${key}'`);
  const v = f.meta[key];
  if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
  return v as T;
}

export function opt<T>(f: Frame, key: string, ok: (v: unknown) => boolean, what: string, dflt: T): T {
  const v = f.meta[key];
  if (v === undefined || v === null) return dflt;
  if (!ok(v)) bad(f.type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
  return v as T;
}

/** `req` and `opt` of the meta of a message on a plain object (a declaration, a status item). */
export function reqIn<T>(type: number, o: Record<string, unknown>, key: string, ok: (v: unknown) => boolean, what: string): T {
  if (!(key in o)) bad(type, `missing '${key}'`);
  if (!ok(o[key])) bad(type, `'${key}' must be ${what}, got ${JSON.stringify(o[key])}`);
  return o[key] as T;
}

export function optIn(type: number, o: Record<string, unknown>, key: string, ok: (v: unknown) => boolean, what: string): unknown {
  const v = o[key];
  if (v === undefined || v === null) return undefined;
  if (!ok(v)) bad(type, `'${key}' must be ${what}, got ${JSON.stringify(v)}`);
  return v;
}

export const drop = (o: Record<string, unknown>) => {
  for (const k of Object.keys(o)) if (o[k] === undefined || o[k] === null) delete o[k];
  return o;
};

export function section(f: Frame, name: string): Section | undefined {
  return f.sections.find((s) => s.name === name);
}

/** The section `name` of a scene part, or undefined; `shape` holds the rows it must have (null: any) and its rank. */
export function partSection(f: Frame, name: string, dtype: number, shape: (number | null)[]): Section | undefined {
  const s = section(f, name);
  if (!s) return undefined;
  if (s.encoding !== 0 || s.dtype !== dtype || s.shape.length !== shape.length || shape.some((x, i) => x !== null && x !== s.shape[i]))
    bad(f.type, `'${name}' must be ${DTYPE_NAME[dtype]} (${shape.map((x) => x ?? '_').join(', ')}), got dtype ${s.dtype} shape [${s.shape}]`);
  return s;
}

/** The stamp sections of a message: `stamp` i64 (T,), and where `rows` is given `stamps` i64 (rows, T), T from 1 to 8. */
export function stampSections(f: Frame, isStatic: boolean, rows?: number): { stamp?: Section; stamps?: Section } {
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

export type { SessionIdentity };
