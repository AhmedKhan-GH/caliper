// The binary codec of wire protocol v3 (include/caliper/live_net.h,
// python/caliper/_wire.py). One frame is a 24-byte little-endian header,
// section_count uint32 section lengths, a UTF-8 JSON metadata object, then the
// payload sections, each at an offset from the frame start that is a multiple
// of 8. Framing, controls, scenes, timelines, and programs live in sibling
// modules; this entry point dispatches messages and re-exports the full codec API.

import {
  DTYPE, LIMITS, MAX_BARS, MSG, PROTOCOL_MINOR,
  type DataView as DataViewSpec, type ErrorMsg, type Message, type MessageType, type PayloadAxis,
  type SeriesMsg, type SessionIdentity, type SnapshotMsg, type TimelineInfo, type UnknownMsg,
  type ViewEntry,
} from './messages.js';
import {
  bad, decodeFrame, drop, encodeWithSource, f32, f32Section, i64, isBool, isIdentity, isInt,
  isName, isNum, isObj, isSessionId, isStr, isStrs, isU32, isUint, NAME_MAX, numberSources, opt,
  optName, partSection, req, section, stampSection, stampSections, u8, utf8,
  type Frame, type Section,
} from './frame.js';
import {
  decodeControl, decodeControlReply, decodeControls, decodeSelection,
  encodeControl, encodeControlReply, encodeControls, encodeSelection,
} from './controls.js';
import {
  decodeCursor, decodeRecordingOpen, decodeRecordings, decodeText, decodeTimeState, encodeText,
  isRecorded, isRecordingRef, isTimelines,
} from './timelines.js';
import {
  decodeSceneClasses, decodeSceneClear, decodeSceneTransforms, decodeSceneUpdate,
  encodeSceneClasses, encodeSceneClear, encodeSceneTransforms, encodeSceneUpdate,
} from './scenes.js';
import {
  decodeProgram, decodeProgramState, decodeResource, encodeProgram, encodeResource,
} from './programs.js';

export {
  canonicalJson, decodeFrame, encodeFrame, entityPathOk, HEADER_BYTES, MAGIC, MessageError,
  SECTION_ALIGN, WireError, type Frame, type Section,
} from './frame.js';
export {
  decodeSceneClasses, decodeSceneClear, decodeSceneTransforms, decodeSceneUpdate,
  encodeSceneClasses, encodeSceneClear, encodeSceneTransforms, encodeSceneUpdate,
  resolveSceneDataView, sceneViewInfo, SplitUpdates,
} from './scenes.js';
export {
  decodeCursor, decodeRecordingOpen, decodeRecordings, decodeText, decodeTimeState,
  encodeText, stampTextOk, textFromLines, textLines,
} from './timelines.js';
export {
  decodeProgram, decodeProgramState, decodeResource, encodeProgram, encodeResource, SplitResources,
} from './programs.js';

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
  const message = decodeMessage(f);
  const source = numberSources.get(f.meta);
  if (source) numberSources.set(message, source);
  return message;
}

function decodeMessage(f: Frame): Message | UnknownMsg {
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
    case MSG.controls:
      return decodeControls(f);
    case MSG.control:
      return decodeControl(f);
    case MSG.control_reply:
      return decodeControlReply(f);
    case MSG.selection:
      return decodeSelection(f);
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
    case MSG.program:
      return decodeProgram(f);
    case MSG.program_state:
      return decodeProgramState(f);
    case MSG.resource:
      return decodeResource(f);
    default:
      return { type: 'unknown', id: t, meta: f.meta };
  }
}

/**
 * Encodes one message into the bytes of one WebSocket message, at a minor
 * version (this codec's by default; a 3.0 message at minor 0 encodes to the
 * bytes a 3.0 sender writes).
 */
export function encode(msg: Message, minor: number = PROTOCOL_MINOR): Uint8Array {
  const id = MSG[msg.type as MessageType];
  const encodeFrame = (type: number, meta: Record<string, unknown>, sections: Section[] = []) =>
    encodeWithSource(msg, type, meta, sections, minor);
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
      return encodeControls(msg, minor);
    case 'control':
      return encodeControl(msg, minor);
    case 'control_reply':
      return encodeControlReply(msg, minor);
    case 'selection':
      return encodeSelection(msg, minor);
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
    case 'program':
      return encodeProgram(msg, minor);
    case 'program_state':
      return encodeFrame(id, drop({ request_id: msg.request_id, step: msg.step, step_ms: msg.step_ms, programs: msg.programs }));
    case 'resource':
      return encodeResource(msg, minor);
    case 'time_state':
      return encodeFrame(id, { ...drop({
        request_id: msg.request_id, session: msg.session, generation: msg.generation, state: msg.state, timeline: msg.timeline,
        ranges: msg.ranges, live: msg.live, speed: msg.speed, pace: msg.pace, loop: msg.loop,
      }), cursor: msg.cursor ?? null });
  }
}
