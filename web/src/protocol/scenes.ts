// Wire protocol 3.2 draft: scenes, split updates, and scene view metadata.

import {
  CAMERA_REQUIRED, CLEAR_PARTS, COLOR_PER_EDGE, COLOR_PER_STRIP, COLOR_PER_VERTEX, CONVENTION_RDF,
  CONVENTION_RUB, DTYPE, DTYPE_SIZE, FORM_COLOR, FORM_DEPTH, FORM_ENCODED, FORM_SEGMENTATION,
  FORMAT_JPEG, FORMAT_PNG, LIMITS_32, LINES_EDGES, LINES_SEGMENTS, LINES_STRIPS, MAX_CLASSES,
  MAX_CLASS_ID, MAX_ENCODED_BYTES, MAX_IMAGE_SIDE, MAX_INCOMPLETE_UPDATES, MAX_LABELS, MAX_PIECES,
  MAX_RESOURCES, MAX_TRANSFORMS, MAX_UPDATE_BYTES, MSG, PART_ARROWS, PART_BOXES, PART_CAMERA,
  PART_IMAGE, PART_KEYS, PART_LINES, PART_MESH, PART_POINTS, PART_SECTIONS, PROTOCOL_MAJOR,
  PROTOCOL_MINOR, PROTOCOL_MINOR_3_2, SCENE_DATA_VIEW, SCENE_PARTS, SPACE_2D, SPACE_3D,
  UP_AXES_2D, UP_AXES_3D,
  type DataView as DataViewSpec, type SceneArray, type SceneClass, type SceneClassesMsg,
  type SceneClearMsg, type ScenePart, type SceneTransformsMsg, type SceneUpdateMsg,
  type SceneViewInfo, type SplitKey, type ViewCameraOrbit, type ViewEntry,
} from './messages.js';
import {
  bad, bytesOf, canonicalJson, entityPathOk, f32, f32Section, frameAt, i32, i64, isAscending,
  isBool, isFiniteNum, isInt, isName, isNum, isObj, isPositive, isStr, isU32, isUint, MAX_STATS,
  MessageError, NAME_MAX, opt, partSection, req, section, stampSection, stampSections, u16, u8,
  utf8, type Frame, type Raw, type Section,
} from './frame.js';
import { isLibraryPin, isParamName, isSceneProgramId } from './programs.js';

const isRgba = (v: unknown) => Array.isArray(v) && v.length === 4 && v.every((x) => isInt(x) && x >= 0 && x <= 255);
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
  opacity: [(v) => isFiniteNum(v) && (v as number) >= 0 && (v as number) <= 1, 'a number from 0 to 1'],
  scale: [isPositive, 'a positive number'],
  cloud: [isBool, 'a boolean'],
  format: [(v) => v === FORMAT_JPEG || v === FORMAT_PNG, 'jpeg or png'],
  fx: [isPositive, 'a positive number'],
  fy: [isPositive, 'a positive number'],
  cx: [isFiniteNum, 'a finite number'],
  cy: [isFiniteNum, 'a finite number'],
  height: [isSide, `an integer from 1 to ${MAX_IMAGE_SIDE}`],
  convention: [(v) => v === CONVENTION_RDF || v === CONVENTION_RUB, 'rdf or rub'],
  plane_distance: [isPositive, 'a positive number'],
};
const CAMERA_WIDTH: [(v: unknown) => boolean, string] = [isSide, `an integer from 1 to ${MAX_IMAGE_SIDE}`];

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

interface PieceForm { program: string; piece: { name: string; version: number; hash: string }; resources: Record<string, number>; names: string[] }
const SCENE_PIECES: Record<string, readonly string[]> = {
  points: ['points'], lines: ['lines'], arrows: ['arrows'], boxes: ['boxes'], mesh: ['mesh'], image: ['image', 'depth', 'segmentation'],
};
const PIECE_OF_FORM: Record<string, string> = { [FORM_COLOR]: 'image', [FORM_DEPTH]: 'depth', [FORM_SEGMENTATION]: 'segmentation' };

/** The keys of the piece form of SCENE_UPDATE, checked in this order: program, piece, resources, arrays (python: _piece_form). */
function decodePieceForm(f: Frame, part: ScenePart): PieceForm {
  const program = req<string>(f, 'program', isSceneProgramId, 'a scene program id scene.<n> of 1 to 9 digits');
  const piece = req<PieceForm['piece']>(f, 'piece', isLibraryPin, 'a library pin {name, version} with an optional hash');
  if (!(SCENE_PIECES[part] ?? []).includes(piece.name)) bad(f.type, `the part '${part}' has no library piece '${piece.name}'`);
  const resources = req<Record<string, number>>(f, 'resources', (v) => isObj(v)
    && Object.keys(v as object).length >= 1 && Object.keys(v as object).length <= MAX_RESOURCES
    && Object.entries(v as Record<string, unknown>).every(([k, n]) => isParamName(k) && isU32(n)),
    `an object of 1 to ${MAX_RESOURCES} resource ids, each with a uint32 count`);
  const table = PART_SECTIONS[part];
  const names = req<string[]>(f, 'arrays', (v) => Array.isArray(v) && v.every((x) => typeof x === 'string'), 'a list of array names');
  const idx = names.map((x) => table.indexOf(x));
  if (names.length === 0 || names[0] !== table[0] || idx.some((i) => i < 0) || idx.some((i, k) => k > 0 && i <= idx[k - 1])) {
    bad(f.type, `'arrays' must list the arrays of '${part}' in their order from '${table[0]}', each once, got [${names}]`);
  }
  return { program, piece, resources, names };
}

type PayloadAxisList = NonNullable<SceneUpdateMsg['axes']>;

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
  let pf: PieceForm | undefined;
  if ('program' in f.meta) {
    pf = decodePieceForm(f, p);
    if (split !== undefined) bad(t, 'an update in the piece form is never split');
    params = scenePartKeys(f, p);
    if (p === PART_IMAGE && PIECE_OF_FORM[params.form as string] !== pf.piece.name) bad(t, `an image of the form '${params.form}' has no piece '${pf.piece.name}'`);
    for (const name of PART_SECTIONS[p]) if (f.sections.some((s) => s.name === name)) bad(t, `an update in the piece form carries no array section, got '${name}'`);
    arrays = {};
  } else if (split !== undefined) {
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
  if (pf) { m.program = pf.program; m.piece = pf.piece; m.resources = pf.resources; m.names = pf.names; }
  if (stamp) m.stamp = i64(stamp);
  return m;
}

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
  if (m.program !== undefined) Object.assign(meta, { program: m.program, piece: m.piece, resources: m.resources, arrays: m.names });
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

// --- 3.2 draft: split updates ---

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
  if (t !== undefined && t !== null && !(Array.isArray(t) && t.length === 3 && t.every(isFiniteNum))) return undefined;
  if (o.distance !== undefined && o.distance !== null && !isPositive(o.distance)) return undefined;
  if (o.azimuth !== undefined && o.azimuth !== null && !isFiniteNum(o.azimuth)) return undefined;
  const e = o.elevation;
  if (e !== undefined && e !== null && !(isFiniteNum(e) && (e as number) >= -89 && (e as number) <= 89)) return undefined;
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
