// Wire protocol 3.2 draft: timelines, text logs, recordings, cursor, and time state.

import {
  CURSOR_OPS, CURSOR_PLAY, CURSOR_SEEK, CURSOR_STEP, DTYPE, LEVEL_ERROR, MAX_CURSOR_STEP,
  MAX_RECORDED, MAX_RECORDINGS, MAX_SPEED, MAX_TIMELINES, MIN_SPEED, MSG, PACE_EVEN, PACE_RECORDED,
  PROTOCOL_MINOR, STAMP_TEXT_MAX, TEXT_LINE_MAX, TEXT_MAX_LINES, TIME_FOLLOWING, TIME_PAUSED,
  TIME_PLAYING, TIMELINE_SEQUENCE, TIMELINE_TIME,
  type CursorMsg, type RecordingEntry, type RecordingOpenMsg, type RecordingsMsg, type TextMsg,
  type TimeStateMsg, type TimelineInfo,
} from './messages.js';
import {
  bad, bytesOf, drop, frameAt, i32, i64, isBool, isFiniteNum, isInt, isName, isObj, isSessionId,
  isStr, isU32, isUint, MessageError, opt, partSection, req, section, stampSection, stampSections,
  u8, utf8, utf8Strict, type Frame, type Section,
} from './frame.js';

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
export function isTimelines(v: unknown): boolean {
  if (!(Array.isArray(v) && v.length >= 2 && v.length <= MAX_TIMELINES)) return false;
  if (!v.every(isTimelineEntry)) return false;
  const names = v.map((t) => (t as TimelineInfo).name);
  return new Set(names).size === names.length && v[0].name === 'step' && v[0].kind === TIMELINE_SEQUENCE
    && v[1].name === 'time' && v[1].kind === TIMELINE_TIME;
}

export const isRecorded = (v: unknown) => Array.isArray(v) && v.length <= MAX_RECORDED && v.every(isObj);

/** A session's key "recording": {id, seekable, complete}. */
export const isRecordingRef = (v: unknown) => isObj(v) && isSessionId((v as Record<string, unknown>).id)
  && isBool((v as Record<string, unknown>).seekable) && isBool((v as Record<string, unknown>).complete);

const isLoop = (v: unknown) => Array.isArray(v) && v.length === 2 && v.every(stampTextOk) && BigInt(v[0]) <= BigInt(v[1]);
const isSpeed = (v: unknown) => isFiniteNum(v) && (v as number) >= MIN_SPEED && (v as number) <= MAX_SPEED;
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
