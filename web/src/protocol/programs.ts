// Wire protocol 3.2 draft: GPU programs, piece instances, and published resources.

import {
  DTYPE, MAX_INCOMPLETE_UPDATES, MAX_PROGRAMS, MSG, OUTPUT_GEOMETRY, OUTPUT_PIXELS, OUTPUT_TENSOR,
  PART_LINES, PART_POINTS, PIECE_COMMON_MAX, PIECE_NAME_MAX, PIECE_SEGMENTS_MAX, PIECE_VERSION_MAX,
  PROGRAM_ARRAYS, PROGRAM_BUILTIN_MAX, PROGRAM_EVERY_MAX, PROGRAM_ID_MAX, PROGRAM_MAX_COUNT,
  PROGRAM_MAX_INPUTS, PROGRAM_MEMORY_MB_MAX, PROGRAM_PARAM_NAME_MAX, PROGRAM_PARAMS,
  PROGRAM_PARAMS_MAX, PROGRAM_REMOVE, PROGRAM_SET, PROGRAM_SIDE_MAX, PROGRAM_SOURCE_MAX,
  PROGRAM_STATES, PROGRAM_TIME_MS_MAX, PROGRAM_WORKGROUPS_MAX, PROTOCOL_MINOR, RESOURCE_ELEMENTS,
  RESOURCE_MAX_BYTES, RESOURCE_MAX_PARTS, RESOURCE_PART_WAIT, RESOURCE_SIDE_MAX,
  RESOURCE_STRIDE_MAX, RESOURCE_TEXTURE_VERSION_MAX_BYTES, RESOURCE_VERSION_MAX_BYTES,
  SCENE_PROGRAM_PREFIX, SECTION_COMMON, SECTION_DERIVE, SECTION_DRAW, SECTION_RESOURCE,
  SECTION_WGSL, STATUS_OPERATOR, STATUS_PIECE, TENSOR_AXES_MAX, TENSOR_ELEMENTS_MAX,
  TENSOR_NAME_MAX, TEXTURE_FORMATS,
  type ProgramDeclaration, type ProgramMsg, type ProgramStateMsg, type ProgramStatus,
  type ResourceMsg, type ResourceSplit,
} from './messages.js';
import {
  bad, boundedText, drop, entityPathOk, frameAt, intIn, isFiniteNum, isName, isNonneg, isObj,
  isSafeUint, isText, isU32, listOf, MessageError, opt, optIn, present, req, reqIn, section, utf8,
  utf8Strict, type Frame, type Raw, type Section,
} from './frame.js';

const PROGRAM_ID_RE = /^[a-z][a-z0-9_-]*$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const OPERATOR_ID_RE = /^operator\.[a-z][a-z0-9_-]*\.[0-9]{1,5}$/;
const PARAM_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SCENE_ID_RE = /^[0-9]{1,9}$/;
const PIECE_NAME_RE = /^[a-z][a-z0-9_]*$/;
const PIECE_HASH_RE = /^sha256:[0-9a-f]{64}$/;

/** The id of a scene program: "scene." and 1 to 9 decimal digits. */
export const isSceneProgramId = (v: unknown) => typeof v === 'string' && v.startsWith(SCENE_PROGRAM_PREFIX) && SCENE_ID_RE.test(v.slice(SCENE_PROGRAM_PREFIX.length));

/** A program id: 1 to PROGRAM_ID_MAX bytes of [a-z0-9_-] with a letter first, an operator id "operator.<spec id>.<n>" with 1 to 5 digits, or a scene program id. */
const isProgramId = (v: unknown) => typeof v === 'string' && v.length > 0 && utf8.encode(v).length <= PROGRAM_ID_MAX
  && (v.startsWith(SCENE_PROGRAM_PREFIX) ? isSceneProgramId(v) : v.startsWith('operator.') ? OPERATOR_ID_RE.test(v) : PROGRAM_ID_RE.test(v));

export const isParamName = (v: unknown) => typeof v === 'string' && v.length > 0 && v.length <= PROGRAM_PARAM_NAME_MAX && PARAM_NAME_RE.test(v) && !v.startsWith('__');

/** Program parameters: 1 to PROGRAM_PARAMS_MAX names of a WGSL field, each with a finite number. */
const isParams = (v: unknown) => {
  if (!isObj(v)) return false;
  const e = Object.entries(v as Record<string, unknown>);
  return e.length >= 1 && e.length <= PROGRAM_PARAMS_MAX && e.every(([k, x]) => isParamName(k) && isFiniteNum(x));
};

/** The shape of a tensor output: 1 to TENSOR_AXES_MAX integers from 1, at most TENSOR_ELEMENTS_MAX elements. */
const isTensorShape = (v: unknown) => listOf(intIn(1, TENSOR_ELEMENTS_MAX), TENSOR_AXES_MAX, 1)(v)
  && (v as number[]).reduce((a, b) => a * b, 1) <= TENSOR_ELEMENTS_MAX;

const isSha256 = (v: unknown) => typeof v === 'string' && SHA256_RE.test(v);

const isInputEntry = (v: unknown) => isObj(v) && isName((v as Record<string, unknown>).var) && boundedText(TENSOR_NAME_MAX)((v as Record<string, unknown>).source);
const isArraysList = (v: unknown) => Array.isArray(v) && v.length >= 1 && v.length <= PROGRAM_ARRAYS.length
  && v.every((x) => typeof x === 'string' && PROGRAM_ARRAYS.includes(x)) && new Set(v).size === v.length && v.includes('positions');

/** The checks of a PROGRAM set's declaration: output with kind and the keys of its kind, then inputs, limits, workgroups and entry. */
function checkDeclaration(type: number, d: Record<string, unknown>): void {
  const out = reqIn<Record<string, unknown>>(type, d, 'output', isObj, 'an object');
  const kind = reqIn<string>(type, out, 'kind', (v) => v === OUTPUT_GEOMETRY || v === OUTPUT_PIXELS || v === OUTPUT_TENSOR, 'geometry, pixels or tensor');
  if (kind === OUTPUT_TENSOR) {
    reqIn(type, out, 'shape', isTensorShape, `1 to ${TENSOR_AXES_MAX} integers from 1 with at most ${TENSOR_ELEMENTS_MAX} elements`);
    if (Object.keys(out).some((k) => k !== 'kind' && k !== 'shape')) bad(type, "a tensor output holds only 'kind' and 'shape'");
  } else if (kind === OUTPUT_GEOMETRY) {
    reqIn(type, out, 'part', (v) => v === PART_POINTS || v === PART_LINES, 'points or lines');
    reqIn(type, out, 'max_count', intIn(1, PROGRAM_MAX_COUNT), `an integer from 1 to ${PROGRAM_MAX_COUNT}`);
    reqIn(type, out, 'arrays', isArraysList, 'unique names of positions, colors, values and radii, with positions');
    optIn(type, out, 'entity', entityPathOk, 'an entity path');
  } else {
    const side = intIn(1, PROGRAM_SIDE_MAX);
    reqIn(type, out, 'width', side, `an integer from 1 to ${PROGRAM_SIDE_MAX}`);
    reqIn(type, out, 'height', side, `an integer from 1 to ${PROGRAM_SIDE_MAX}`);
    optIn(type, out, 'format', (v) => v === 'rgba8unorm', 'rgba8unorm');
  }
  optIn(type, d, 'inputs', listOf(isInputEntry, PROGRAM_MAX_INPUTS), `at most ${PROGRAM_MAX_INPUTS} inputs {var, source}`);
  const lim = optIn(type, d, 'limits', isObj, 'an object') as Record<string, unknown> | undefined;
  if (lim !== undefined) {
    optIn(type, lim, 'time_ms', (v) => isFiniteNum(v) && (v as number) > 0 && (v as number) <= PROGRAM_TIME_MS_MAX, `a number above 0 and at most ${PROGRAM_TIME_MS_MAX}`);
    optIn(type, lim, 'memory_mb', intIn(1, PROGRAM_MEMORY_MB_MAX), `an integer from 1 to ${PROGRAM_MEMORY_MB_MAX}`);
    optIn(type, lim, 'every', intIn(1, PROGRAM_EVERY_MAX), `an integer from 1 to ${PROGRAM_EVERY_MAX}`);
  }
  optIn(type, d, 'workgroups', listOf(intIn(1, PROGRAM_WORKGROUPS_MAX), 3, 3), `three integers from 1 to ${PROGRAM_WORKGROUPS_MAX}`);
  optIn(type, d, 'entry', isName, 'an entry point name');
  optIn(type, d, 'params', isParams, `1 to ${PROGRAM_PARAMS_MAX} names of a WGSL field with finite numbers`);
  optIn(type, d, 'builtin', boundedText(PROGRAM_BUILTIN_MAX), `a built-in name of 1 to ${PROGRAM_BUILTIN_MAX} bytes`);
}

/** PROGRAM: viewer to producer, set or remove one GPU program (capability "programs"). */
export function decodeProgram(f: Frame): ProgramMsg {
  const request_id = req<number>(f, 'request_id', isU32, 'a uint32');
  const op = req<ProgramMsg['op']>(f, 'op', (v) => v === PROGRAM_SET || v === PROGRAM_REMOVE || v === PROGRAM_PARAMS, 'set, remove or params');
  const id = req<string>(f, 'id', isProgramId, `a program id of at most ${PROGRAM_ID_MAX} bytes`);
  if (op === PROGRAM_REMOVE) return { type: 'program', request_id, op, id };
  if (op === PROGRAM_PARAMS) {
    if ('declaration' in f.meta) bad(f.type, "a params op carries no 'declaration'");
    if (f.sections.length > 0) bad(f.type, 'a params op carries no section');
    const params = req<Record<string, number>>(f, 'params', isParams, `1 to ${PROGRAM_PARAMS_MAX} names of a WGSL field with finite numbers`);
    return { type: 'program', request_id, op, id, params };
  }
  if ('piece' in f.meta) return decodeProgramInstance(f, request_id, id);
  const declaration = req<ProgramDeclaration>(f, 'declaration', isObj, 'an object');
  checkDeclaration(f.type, declaration);
  const geometry = (declaration.output as Record<string, unknown>).kind === OUTPUT_GEOMETRY;
  const scene = geometry
    ? req<string>(f, 'scene', isName, 'a scene name')
    : opt<string | undefined>(f, 'scene', isName, 'a scene name', undefined);
  const sha256 = req<string>(f, 'sha256', isSha256, '64 lowercase hex digits');
  const w = section(f, SECTION_WGSL);
  if (!w || w.encoding !== 0 || w.dtype !== DTYPE.u8 || w.shape.length !== 1) bad(f.type, "a program set needs a (B,) u8 'wgsl' section");
  if (w.shape[0] < 1 || w.shape[0] > PROGRAM_SOURCE_MAX) bad(f.type, `'wgsl' holds 1 to ${PROGRAM_SOURCE_MAX} bytes, got ${w.shape[0]}`);
  let wgsl: string;
  try {
    wgsl = utf8Strict.decode(w.bytes);
  } catch {
    bad(f.type, "'wgsl' is not UTF-8");
  }
  const m: ProgramMsg = { type: 'program', request_id, op, id, declaration, wgsl, sha256 };
  if (scene !== undefined) m.scene = scene;
  return m;
}

export function encodeProgram(m: ProgramMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta = drop({
    request_id: m.request_id, op: m.op, id: m.id, declaration: m.declaration, scene: m.scene, sha256: m.sha256, params: m.params,
    piece: m.piece, manifest: m.manifest, bind: m.bind, panel: m.panel, segments: m.segments,
  });
  const sections: Section[] = [];
  for (const [name, text] of [[SECTION_WGSL, m.wgsl], [SECTION_COMMON, m.common], [SECTION_DERIVE, m.derive], [SECTION_DRAW, m.draw]] as const) {
    if (text === undefined) continue;
    const b = utf8.encode(text);
    sections.push({ name, dtype: DTYPE.u8, shape: [b.length], encoding: 0, bytes: b });
  }
  return frameAt(MSG.program, meta, sections, minor);
}

const isProgramError = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return isName(o.code) && isText(o.message) && ['line', 'column'].every((k) => !present(o[k]) || isU32(o[k]));
};

const STATUS_KEYS: [string, (v: unknown) => boolean][] = [
  ['every', intIn(1, PROGRAM_EVERY_MAX)], ['every_min', intIn(1, PROGRAM_EVERY_MAX)], ['last_pass_ms', isNonneg], ['last_cost_ms', isNonneg],
  ['step_ms', isNonneg], ['overhead_share', isNonneg], ['passes_per_s', isNonneg],
  ['last_step', isSafeUint], ['runs', isSafeUint], ['memory_bytes', isSafeUint], ['memory_limit_bytes', isSafeUint],
  ['error', isProgramError], ['warning', isProgramError],
];

/** A status item: id, kind and state, and the optional keys of STATUS_KEYS and sha256, null or absent when unknown. */
const isStatusItem = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  if (!(isProgramId(o.id) && (o.kind === OUTPUT_GEOMETRY || o.kind === OUTPUT_PIXELS || o.kind === OUTPUT_TENSOR || o.kind === STATUS_OPERATOR || o.kind === STATUS_PIECE) && typeof o.state === 'string' && PROGRAM_STATES.includes(o.state))) return false;
  if (!STATUS_KEYS.every(([k, ok]) => !present(o[k]) || ok(o[k]))) return false;
  return !present(o.sha256) || o.sha256 === '' || isSha256(o.sha256);
};

const isStatusItems = (v: unknown) => listOf(isStatusItem, MAX_PROGRAMS)(v)
  && new Set((v as { id: string }[]).map((x) => x.id)).size === (v as unknown[]).length;

/** PROGRAM_STATE: producer to viewer, the status item of each program. */
export function decodeProgramState(f: Frame): ProgramStateMsg {
  const request_id = opt<number | undefined>(f, 'request_id', isU32, 'a uint32', undefined);
  const step = opt<number | undefined>(f, 'step', isSafeUint, 'an unsigned integer', undefined);
  const step_ms = opt<number | undefined>(f, 'step_ms', isNonneg, 'a number of at least 0', undefined);
  const programs = req<ProgramStatus[]>(f, 'programs', isStatusItems, `at most ${MAX_PROGRAMS} status items with unique program ids`);
  const m: ProgramStateMsg = { type: 'program_state', programs };
  if (request_id !== undefined) m.request_id = request_id;
  if (step !== undefined) m.step = step;
  if (step_ms !== undefined) m.step_ms = step_ms;
  return m;
}

// --- 3.2 draft: instances of pieces and RESOURCE (python: _program_instance, Resource) ---

/** The pin of a piece: name ([a-z][a-z0-9_]*, at most PIECE_NAME_MAX bytes), version (1 to PIECE_VERSION_MAX), hash ("sha256:" and 64 lowercase hex digits), and optional from ("library", "workspace" or "inline"). */
const isPiecePin = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return typeof o.name === 'string' && o.name.length > 0 && o.name.length <= PIECE_NAME_MAX && PIECE_NAME_RE.test(o.name)
    && intIn(1, PIECE_VERSION_MAX)(o.version) && typeof o.hash === 'string' && PIECE_HASH_RE.test(o.hash)
    && (!present(o.from) || o.from === 'library' || o.from === 'workspace' || o.from === 'inline');
};

/** The pin of a library piece in the piece form of SCENE_UPDATE: name and version as in isPiecePin, and hash optional (python: _library_pin). */
export const isLibraryPin = (v: unknown) => isObj(v) && isPiecePin({ hash: `sha256:${'0'.repeat(64)}`, ...(v as object) });

/** One value of "bind": "live:<tensor>" (a tensor name of 1 to TENSOR_NAME_MAX bytes) or "program:<id>/<resource>". */
const isBinding = (v: unknown) => {
  if (typeof v !== 'string') return false;
  if (v.startsWith('live:')) return boundedText(TENSOR_NAME_MAX)(v.slice('live:'.length));
  if (v.startsWith('program:')) {
    const rest = v.slice('program:'.length);
    const slash = rest.indexOf('/');
    return slash >= 0 && isProgramId(rest.slice(0, slash)) && isParamName(rest.slice(slash + 1));
  }
  return false;
};
const isBind = (v: unknown) => isObj(v) && Object.keys(v as object).length <= PROGRAM_MAX_INPUTS
  && Object.entries(v as Record<string, unknown>).every(([k, x]) => isParamName(k) && isBinding(x));
const isSegment = (v: unknown) => {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return isName(o.part) && isU32(o.first_line) && (o.first_line as number) >= 1 && isU32(o.lines);
};

/** An optional (B,) u8 section of 1 to `most` bytes of strict UTF-8, as text. */
function textSection(f: Frame, name: string, most: number): string | undefined {
  const s = section(f, name);
  if (s === undefined) return undefined;
  if (s.encoding !== 0 || s.dtype !== DTYPE.u8 || s.shape.length !== 1) bad(f.type, `'${name}' must be a (B,) u8 section`);
  if (s.shape[0] < 1 || s.shape[0] > most) bad(f.type, `'${name}' holds 1 to ${most} bytes, got ${s.shape[0]}`);
  try {
    return utf8Strict.decode(s.bytes);
  } catch {
    bad(f.type, `'${name}' is not UTF-8`);
  }
}

/** The set of an instance of a piece, checked in the order of _program_instance of _wire.py. */
function decodeProgramInstance(f: Frame, request_id: number, id: string): ProgramMsg {
  const m = f.meta;
  if (['declaration', 'scene', 'sha256'].some((k) => k in m) || section(f, SECTION_WGSL) !== undefined) {
    bad(f.type, "an instance of a piece carries no 'declaration', 'scene', 'sha256' or 'wgsl'");
  }
  const piece = req<NonNullable<ProgramMsg['piece']>>(f, 'piece', isPiecePin, 'a pin {name, version, hash}');
  const manifest = req<Record<string, unknown>>(f, 'manifest', isObj, 'an object');
  const bind = opt<Record<string, string> | undefined>(f, 'bind', isBind, `at most ${PROGRAM_MAX_INPUTS} inputs bound to live:<tensor> or program:<id>/<resource>`, undefined);
  const params = opt<Record<string, number> | undefined>(f, 'params', isParams, `1 to ${PROGRAM_PARAMS_MAX} names of a WGSL field with finite numbers`, undefined);
  const panel = opt<string | undefined>(f, 'panel', isName, 'a panel id', undefined);
  const segments = opt<NonNullable<ProgramMsg['segments']> | undefined>(f, 'segments', listOf(isSegment, PIECE_SEGMENTS_MAX),
    `at most ${PIECE_SEGMENTS_MAX} segments {part, first_line, lines}`, undefined);
  const common = textSection(f, SECTION_COMMON, PIECE_COMMON_MAX);
  const derive = textSection(f, SECTION_DERIVE, PROGRAM_SOURCE_MAX);
  const draw = textSection(f, SECTION_DRAW, PROGRAM_SOURCE_MAX);
  if (derive === undefined && draw === undefined) bad(f.type, "an instance of a piece carries the section 'derive' or 'draw'");
  const out: ProgramMsg = { type: 'program', request_id, op: PROGRAM_SET, id, piece, manifest };
  if (params !== undefined) out.params = params;
  if (bind !== undefined) out.bind = bind;
  if (panel !== undefined) out.panel = panel;
  if (segments !== undefined) out.segments = segments;
  if (common !== undefined) out.common = common;
  if (derive !== undefined) out.derive = derive;
  if (draw !== undefined) out.draw = draw;
  return out;
}

const isStride = (v: unknown) => intIn(4, RESOURCE_STRIDE_MAX)(v) && (v as number) % 4 === 0;

/** RESOURCE: producer to viewer, one version of one published resource (capability "resources"). The data is a view of the frame. */
export function decodeResource(f: Frame): ResourceMsg {
  const program = req<string>(f, 'program', isProgramId, `a program id of at most ${PROGRAM_ID_MAX} bytes`);
  const resource = req<string>(f, 'resource', isParamName, 'a WGSL name of 1 to 63 bytes');
  const step = req<number>(f, 'step', isSafeUint, 'an unsigned integer');
  const count = req<number>(f, 'count', isU32, 'a uint32');
  const stride = req<number>(f, 'stride', isStride, `a multiple of 4 from 4 to ${RESOURCE_STRIDE_MAX}`);
  const side = intIn(1, RESOURCE_SIDE_MAX);
  const width = opt<number | undefined>(f, 'width', side, `an integer from 1 to ${RESOURCE_SIDE_MAX}`, undefined);
  const height = opt<number | undefined>(f, 'height', side, `an integer from 1 to ${RESOURCE_SIDE_MAX}`, undefined);
  if ((width === undefined) !== (height === undefined)) bad(f.type, "a texture holds both 'width' and 'height'");
  const format = opt<ResourceMsg['format']>(f, 'format', (v) => (TEXTURE_FORMATS as readonly unknown[]).includes(v), 'rgba8unorm, r8unorm or r32uint', undefined);
  if (format !== undefined && width === undefined) bad(f.type, "'format' belongs to a texture, with 'width' and 'height'");
  if (width !== undefined) {
    const texels = width * height!;
    const words = format === 'r8unorm' ? Math.floor((texels + 3) / 4) : texels;
    if (stride !== 4 || count !== words) bad(f.type, 'a texture has a stride of 4 and the words of its texels as its count');
  }
  const element = opt<ResourceMsg['element']>(f, 'element', (v) => (RESOURCE_ELEMENTS as readonly unknown[]).includes(v), 'u8 or u16', undefined);
  const elements = opt<number | undefined>(f, 'elements', isU32, 'a uint32', undefined);
  if ((element === undefined) !== (elements === undefined)) bad(f.type, "a buffer of narrow elements holds both 'element' and 'elements'");
  if (element !== undefined) {
    if (width !== undefined) bad(f.type, "a texture has no 'element'");
    const nbytes = element === 'u8' ? 1 : 2;
    if (stride !== 4 || count !== Math.floor((elements! * nbytes + 3) / 4)) bad(f.type, 'a buffer of narrow elements has a stride of 4 and the words of its elements as its count');
  }
  const raw = opt<Raw | undefined>(f, 'split', isObj, 'an object', undefined);
  let split: ResourceSplit | undefined;
  if (raw !== undefined) {
    const need = <T>(key: string, ok: (v: unknown) => boolean, what: string): T => {
      const v = raw[key];
      if (v === undefined || v === null || !ok(v)) bad(f.type, `'split.${key}' must be ${what}, got ${JSON.stringify(v)}`);
      return v as T;
    };
    const version = need<number>('version', isSafeUint, 'an unsigned integer');
    const parts = need<number>('parts', intIn(2, RESOURCE_MAX_PARTS), `an integer from 2 to ${RESOURCE_MAX_PARTS}`);
    const part = need<number>('part', intIn(0, parts - 1), "a part number below 'parts'");
    const first = need<number>('first', isU32, 'a uint32');
    split = { version, part, parts, first };
  }
  // Bound count before any product of two network values: count * stride is then at most the limit.
  const versionMax = width === undefined ? RESOURCE_VERSION_MAX_BYTES : RESOURCE_TEXTURE_VERSION_MAX_BYTES;
  const limit = split ? versionMax : RESOURCE_MAX_BYTES;
  if (count > Math.floor(limit / stride)) bad(f.type, `${split ? 'a resource in parts' : 'a resource'} holds at most ${limit} bytes`);
  const s = section(f, SECTION_RESOURCE);
  if (s === undefined || s.encoding !== 0 || s.dtype !== DTYPE.u8 || s.shape.length !== 1) bad(f.type, "a resource needs a (B,) u8 'resource' section");
  if (split) {
    const b = s.shape[0];
    if (split.part === 0 && split.first !== 0) bad(f.type, 'part 0 of a resource starts at element 0');
    if (split.first >= count) bad(f.type, "'split.first' must lie below the count");
    if (b <= 0 || b % stride !== 0 || b / stride > count - split.first) bad(f.type, "a part of a resource holds whole elements from 'split.first' up to the count");
  } else if (s.shape[0] !== count * stride) {
    bad(f.type, `'resource' holds count * stride = ${count * stride} bytes, got ${s.shape[0]}`);
  }
  const m: ResourceMsg = { type: 'resource', program, resource, step, count, stride, data: s.bytes };
  if (width !== undefined) m.width = width;
  if (height !== undefined) m.height = height;
  if (format !== undefined) m.format = format;
  if (element !== undefined) m.element = element;
  if (elements !== undefined) m.elements = elements;
  if (split) m.split = split;
  return m;
}

/**
 * A receiver's versions in parts of one connection. `add` takes every RESOURCE of the connection in order and returns the
 * whole version, else null. A version in one frame comes back at once. Part 0 allocates the version's count * stride bytes,
 * which the decoder bounded by the buffer or texture version limit, and each part copies its elements in; the last part returns the
 * whole. A part whose version is not held is dropped and counted. A part out of order, one whose keys differ from part 0, or a
 * last part that does not end at the count throws MessageError (bad_request) and discards the version. Part 0 of a newer
 * version, or a version in one frame, of the same resource discards an incomplete one; at most MAX_INCOMPLETE_UPDATES are
 * held, the oldest going first; and one whose next part has not come after RESOURCE_PART_WAIT further RESOURCE frames expires.
 */
export class SplitResources {
  /** "program/resource" -> the version being put together, oldest first. */
  readonly open = new Map<string, { head: ResourceMsg; nextPart: number; nextFirst: number; last: number; bytes: Uint8Array }>();
  frames = 0;
  dropped = 0;
  discarded = 0;
  expired = 0;

  add(r: ResourceMsg): ResourceMsg | null {
    this.frames++;
    for (const [k, a] of [...this.open]) {
      if (this.frames - a.last > RESOURCE_PART_WAIT) {
        this.open.delete(k);
        this.expired++;
      }
    }
    const key = `${r.program}/${r.resource}`;
    const sp = r.split;
    if ((!sp || sp.part === 0) && this.open.delete(key)) this.discarded++;
    if (!sp) return r;
    const n = r.data.length / r.stride;
    if (sp.part === 0) {
      while (this.open.size >= MAX_INCOMPLETE_UPDATES) {
        this.open.delete(this.open.keys().next().value as string);
        this.discarded++;
      }
      const bytes = new Uint8Array(r.count * r.stride);
      bytes.set(r.data, 0);
      this.open.set(key, { head: r, nextPart: 1, nextFirst: n, last: this.frames, bytes });
      return null;
    }
    const a = this.open.get(key);
    if (a === undefined || a.head.split!.version !== sp.version) {
      this.dropped++;
      return null;
    }
    const h = a.head;
    const refuse = (why: string): never => {
      this.open.delete(key);
      this.discarded++;
      throw new MessageError('bad_request', why, MSG.resource);
    };
    const name = `'${key}'`;
    if (r.step !== h.step || r.count !== h.count || r.stride !== h.stride || r.width !== h.width || r.height !== h.height || r.format !== h.format || r.element !== h.element || r.elements !== h.elements || sp.parts !== h.split!.parts) {
      refuse(`a part of the resource ${name} differs from its part 0`);
    }
    if (sp.part !== a.nextPart || sp.first !== a.nextFirst) refuse(`a part of the resource ${name} comes out of order`);
    const end = a.nextFirst + n;
    const last = sp.part + 1 === sp.parts;
    if (last !== (end === h.count)) {
      refuse(last ? `the parts of the resource ${name} end before its count` : `a part of the resource ${name} before the last reaches its count`);
    }
    a.bytes.set(r.data, a.nextFirst * h.stride);
    a.nextPart++;
    a.nextFirst = end;
    a.last = this.frames;
    if (!last) return null;
    this.open.delete(key);
    const whole: ResourceMsg = { type: 'resource', program: h.program, resource: h.resource, step: h.step, count: h.count, stride: h.stride, data: a.bytes };
    if (h.width !== undefined) whole.width = h.width;
    if (h.height !== undefined) whole.height = h.height;
    if (h.format !== undefined) whole.format = h.format;
    if (h.element !== undefined) whole.element = h.element;
    if (h.elements !== undefined) whole.elements = h.elements;
    return whole;
  }
}

export function encodeResource(m: ResourceMsg, minor: number = PROTOCOL_MINOR): Uint8Array {
  const meta = drop({
    program: m.program, resource: m.resource, step: m.step, count: m.count, stride: m.stride, width: m.width, height: m.height,
    format: m.format, element: m.element, elements: m.elements,
    split: m.split ? { version: m.split.version, part: m.split.part, parts: m.split.parts, first: m.split.first } : undefined,
  });
  return frameAt(MSG.resource, meta, [{ name: SECTION_RESOURCE, dtype: DTYPE.u8, shape: [m.data.length], encoding: 0, bytes: m.data }], minor);
}
