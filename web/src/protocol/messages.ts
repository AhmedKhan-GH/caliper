// Decoded messages of wire protocol v3 (include/caliper/live_net.h). Field
// names are the JSON metadata keys, so the codec maps them one to one; payload
// sections arrive as typed arrays. Everything outside protocol/codec.ts works
// on these objects only.

export const PROTOCOL_MAJOR = 3;
/** 3.1: controls and linked selection, each used only with a peer that announced its capability. */
export const PROTOCOL_MINOR = 1;
export const STREAM_PATH = '/v3/stream';
export const TOKEN_PARAM = 'token';

export const MSG = {
  hello: 0x0001,
  welcome: 0x0002,
  bye: 0x0003,
  error: 0x0004,
  ping: 0x0005,
  pong: 0x0006,
  sessions: 0x0010,
  attach: 0x0011,
  views: 0x0020,
  subscribe: 0x0021,
  unsubscribe: 0x0022,
  view_request: 0x0023,
  view_reply: 0x0024,
  snapshot: 0x0030,
  series: 0x0031,
  event: 0x0040,
  controls: 0x0050,
  control: 0x0051,
  control_reply: 0x0052,
  selection: 0x0060,
  scene_update: 0x0070, // 3.2 draft
  scene_transforms: 0x0071, // 3.2 draft
  scene_clear: 0x0072, // 3.2 draft
  scene_classes: 0x0073, // 3.2 draft
  text: 0x0074, // 3.2 draft
  recordings: 0x0078, // 3.2 draft
  recording_open: 0x0079, // 3.2 draft
  cursor: 0x007a, // 3.2 draft
  time_state: 0x007b, // 3.2 draft
  program: 0x0080, // 3.2 draft
  program_state: 0x0081, // 3.2 draft
  resource: 0x0082, // 3.2 draft, capability "resources"
} as const;
export type MessageType = keyof typeof MSG;

/** The live view kinds; scene (6) is the protocol 3.2 draft's (KIND_SCENE below). */
export type Kind = 'heatmap' | 'image' | 'points' | 'series' | 'histogram' | 'scene' | 'text' | 'bars';
export const KIND_IDS: Record<Kind, number> = { heatmap: 1, image: 2, points: 3, series: 4, histogram: 5, scene: 6, text: 7, bars: 8 };
export function kindName(id: number): Kind | null {
  for (const k in KIND_IDS) if (KIND_IDS[k as Kind] === id) return k as Kind;
  return null;
}

export type ColormapName = 'viridis' | 'magma' | 'rdbu';
export const COLORMAPS: readonly ColormapName[] = ['viridis', 'magma', 'rdbu'];

export const DTYPE = { f32: 0, f16: 1, bf16: 2, i64: 3, i32: 4, u8: 5, u16: 6 } as const;
export const DTYPE_SIZE: Record<number, number> = { 0: 4, 1: 2, 2: 2, 3: 8, 4: 4, 5: 1, 6: 2 };

/** Snapshot flags: take vmin from stats[0], vmax from stats[1], draw 1:1. */
export const SNAP_AUTO_VMIN = 0x1;
export const SNAP_AUTO_VMAX = 0x2;
export const SNAP_RAW = 0x4;

export const CLOSE = {
  normal: 1000,
  goingAway: 1001,
  unsupportedData: 1003,
  tooBig: 1009,
  protocolError: 4000,
  unsupportedMajor: 4001,
  roleRefused: 4002,
  handshakeTimeout: 4003,
  idleTimeout: 4004,
  serverFull: 4005,
} as const;

export const LIMITS = {
  maxFrameBytes: 64 << 20,
  maxMetaBytes: 1 << 20,
  maxSections: 8,
  maxJsonDepth: 16,
  maxEventIndices: 32768,
  handshakeTimeoutMs: 10000,
  pingIntervalMs: 10000,
  idleTimeoutMs: 30000,
  seriesRetainMin: 4096,
} as const;

/** Capabilities in HELLO and WELCOME. */
export const CAP = {
  events: 'events', controls: 'controls', selection: 'selection', textParams: 'text_params',
  interventions: 'interventions', interpretation: 'interpretation',
} as const;

/** 3.1 limits of controls and linked selection. */
export const LIMITS_31 = {
  maxParams: 64,
  maxActions: 64,
  maxChoices: 64,
  textMax: 1024,
  controlRateHz: 30,
  maxItemSpaces: 16,
  maxDetailItems: 32,
  maxDetailFields: 16,
  detailTextMax: 256,
} as const;

export type Reduction = 'mean' | 'sum' | 'max' | 'min' | 'norm';
export const REDUCTIONS: readonly Reduction[] = ['mean', 'sum', 'max', 'min', 'norm'];

/** An integer selects one position and drops the dim; a range keeps it. */
export type IndexSpec = number | [number | null, number | null] | [number | null, number | null, number | null];

/** A data view: which data travels. Every member is optional in a request. */
export interface DataView {
  source?: string;
  index?: Record<string, IndexSpec>;
  reduce?: Record<string, Reduction>;
  tile?: string;
  budget?: { max_extent?: number; max_points?: number; pool?: 'mean' | 'max' | 'min'; max_backfill?: number };
}

/** A source dimension; 3.1: items names the item space whose items it indexes. */
export interface SourceDim { name: string; size: number; items?: string }
export interface SourceInfo { name: string; dims: SourceDim[]; dtype?: string | number }

/** One entry of a VIEWS message. */
export interface ViewEntry {
  id: number;
  name: string;
  kind: number;
  sources: SourceInfo[];
  default?: DataView | null;
  /** 3.1, series only: the item space that the series' absolute index numbers. */
  items?: string;
  [key: string]: unknown;
}

/** A session identity, as in WELCOME and SESSIONS. */
export interface SessionIdentity {
  id: string;
  script: string;
  pid: number;
  start_unix_ms: number;
  state?: 'live' | 'ended';
  views?: number;
  /** The producer's capabilities, in a session list. */
  capabilities?: string[];
  /** 3.2 draft, a session opened from a recording. */
  recording?: { id: string; seekable: boolean; complete: boolean } | null;
}

/** One axis of a snapshot's payload (payload_layout in _wire.py). */
export interface PayloadAxis { dim: string; start: number; step: number; count: number; size: number }

export interface HelloMsg {
  type: 'hello';
  role: 'viewer' | 'producer';
  capabilities: string[];
  software?: { name: string; version: string };
  session?: SessionIdentity;
}
export interface WelcomeMsg {
  type: 'welcome';
  role: 'producer' | 'hub';
  capabilities: string[];
  software?: { name?: string; version?: string };
  session?: SessionIdentity | null;
  minor: number;
  /** 3.1, to a viewer that announced controls: whether the server accepts CONTROL from it. */
  control?: boolean;
}
export interface ByeMsg { type: 'bye'; reason: string; message: string }
export interface ErrorMsg {
  type: 'error';
  code: string;
  message: string;
  fatal: boolean;
  in_reply_to?: number;
  request_id?: number;
  sub?: number;
  view?: string;
  session?: string;
}
export interface PingMsg { type: 'ping'; nonce: number }
export interface PongMsg { type: 'pong'; nonce: number }
export interface SessionsMsg { type: 'sessions'; sessions: SessionIdentity[] }
export interface AttachMsg { type: 'attach'; session: string | null }
export interface ViewsMsg {
  type: 'views';
  views: ViewEntry[];
  session?: string | null;
  /** 3.2 draft: 2 to 8 timelines, step and time first. */
  timelines?: TimelineInfo[];
}
export interface TimelineInfo { name: string; kind: 'sequence' | 'time'; epoch?: boolean }
export interface SubscribeMsg {
  type: 'subscribe';
  sub: number;
  view: string;
  request_id: number;
  max_rate?: number;
  data_view?: DataView;
}
export interface UnsubscribeMsg { type: 'unsubscribe'; sub: number }
export interface ViewRequestMsg { type: 'view_request'; request_id: number; sub: number; view: string; data_view: DataView }
export interface ViewReplyMsg {
  type: 'view_reply';
  request_id: number;
  sub: number;
  view: string;
  data_view?: DataView;
  error?: { code: string; message?: string };
}

/** The newest state of one view under one subscription. */
export interface SnapshotMsg {
  type: 'snapshot';
  sub: number;
  view: string;
  seq: number;
  stamp_ns: number;
  kind: number;
  data_view: DataView;
  /** heatmap (H, W) f32, image (H, W, 4) u8, points (N, D) f32, histogram or bars (B,) f32; empty (length 0) when `empty`. */
  data: Float32Array | Uint8Array;
  shape: number[];
  /** points: (N,) f32 labels or values. */
  aux?: Float32Array;
  colormap: number;
  flags: number;
  vmin: number;
  vmax: number;
  /** NaN where the producer sent null (a non-finite value). */
  stats: number[];
  axes: PayloadAxis[];
  tile?: { dim: string; rows: number; cols: number } | null;
  points_dims?: 2 | 3;
  label_count?: number;
  /** 3.2 draft: a bar chart's (B, 4) u8 colours, B labels and direction. */
  colors?: Uint8Array;
  labels?: string[];
  horizontal?: boolean;
  /** A static snapshot carries no stamp. */
  static?: boolean;
  /** The i64 (T,) stamp section. */
  stamp?: BigInt64Array;
  cursor?: number;
  /** A recorded view with nothing at the cursor: no section. */
  empty?: boolean;
}

/** Series values start .. start + values.length; total appended so far. */
export interface SeriesMsg {
  type: 'series';
  sub: number;
  view: string;
  start: number;
  total: number;
  stamp_ns: number;
  values: Float32Array;
  /** 3.2 draft: one i64 (T,) stamp for the values, or `stamps` (M, T), `stamps_t` is T. */
  stamp?: BigInt64Array;
  stamps?: BigInt64Array;
  stamps_t?: number;
  /** A decimated recorded series: the absolute index of each value, and the bucket size. */
  indices?: BigInt64Array;
  decimated?: number;
  cursor?: number;
}

/** What a viewer reports to Python: data to read, never a command. */
export interface ViewerEvent {
  view: string;
  type: 'select' | 'pick' | 'hover';
  detail: Record<string, unknown>;
}
export interface EventMsg extends Omit<ViewerEvent, 'type'> {
  type: 'event';
  event: ViewerEvent['type'];
  sub?: number;
  seq?: number;
  data_view?: DataView;
}

// --- 3.1: controls -------------------------------------------------------------

export type RunState = 'none' | 'running' | 'pausing' | 'paused' | 'stepping';
export const RUN_STATES: readonly RunState[] = ['none', 'running', 'pausing', 'paused', 'stepping'];
export const PARAM_TEXT_MAX = 4096;
export type ParamType = 'number' | 'integer' | 'boolean' | 'choice' | 'text';
export type ParamValue = number | boolean | string;
/** The extension ops require the corresponding peer capability. */
export type ControlOp = 'pause' | 'resume' | 'step' | 'set' | 'run' | 'intervene' | 'interpret';
export const CONTROL_OPS: readonly ControlOp[] = ['pause', 'resume', 'step', 'set', 'run', 'intervene', 'interpret'];

/** M7 structured metadata limits (python/caliper/_interpret_protocol.py). */
export const INTERPRET_LIMITS = {
  maxBytes: 262144, maxString: 4096, maxList: 512, maxFields: 64, maxDepth: 8, maxValues: 16384,
} as const;
export type InterpretationValue = null | boolean | number | string | InterpretationValue[] | InterpretationResult;
export interface InterpretationResult { [key: string]: InterpretationValue }
type FeatureSelector = { node: number; layer?: never; feature?: never } | { node?: never; layer: number; feature: number };
export type InterpretationRequest = { version: 1 } & (
  | { method: 'feature.dictionary' | 'trace.status'; params: Record<string, never> }
  | { method: 'feature.show'; params: FeatureSelector }
  | { method: 'feature.examples'; params: FeatureSelector & { limit?: number } }
  | { method: 'trace.run'; params: { prompt: string; max_targets?: number; max_nodes?: number; max_edges?: number } }
  | { method: 'trace.read'; params: { node_offset?: number; node_limit?: number; edge_offset?: number; edge_limit?: number } }
);

export interface RunInfo { state: RunState; step?: number; steps_left?: number }
export interface ParamInfo {
  name: string;
  type: ParamType;
  value: ParamValue;
  pending?: ParamValue;
  min?: number;
  max?: number;
  log?: boolean;
  choices?: string[];
  help?: string;
}
export interface ActionInfo { name: string; running: boolean; queued: boolean; help?: string }

/** Producer to viewer: the run state, the parameters and the actions, whole. */
export interface ControlsMsg { type: 'controls'; run: RunInfo; params: ParamInfo[]; actions: ActionInfo[]; /** 3.2 draft */ stamp?: BigInt64Array; /** 3.2 draft, capability "interventions": the intervention points and the interventions in force. */ interventions?: InterventionsInfo }
/** Viewer to producer: pause, resume, step, set a parameter, run an action. */
export interface ControlMsg {
  type: 'control';
  request_id: number;
  op: ControlOp;
  name?: string;
  value?: ParamValue;
  count?: number;
  /** 3.2 draft, op 'intervene': exactly one of set, clear and clear_all. */
  intervene?: InterveneRequest;
  /** M7, capability 'interpretation': a version 1 request. */
  interpretation?: InterpretationRequest;
}
export interface ControlReplyMsg {
  type: 'control_reply';
  request_id: number;
  op: ControlOp;
  status: 'ok' | 'queued' | 'error';
  name?: string;
  value?: ParamValue;
  message?: string;
  error?: { code: string; message?: string };
  /** Required on a successful 'interpret' reply. */
  result?: InterpretationResult;
}

// --- 3.1: linked selection -------------------------------------------------------

export interface SelectionDetails { fields: string[]; items: number[]; values: (number | string | boolean | null)[][] }
/** Producer to viewer: the current selection of an item space. */
export interface SelectionMsg {
  type: 'selection';
  space: string;
  items: number[];
  truncated?: boolean;
  view?: string;
  details?: SelectionDetails;
  details_error?: string;
}

/** A message type this version does not know; receivers ignore it. */
export interface UnknownMsg { type: 'unknown'; id: number; meta: Record<string, unknown> }

export type Message =
  | HelloMsg | WelcomeMsg | ByeMsg | ErrorMsg | PingMsg | PongMsg | SessionsMsg | AttachMsg
  | ViewsMsg | SubscribeMsg | UnsubscribeMsg | ViewRequestMsg | ViewReplyMsg | SnapshotMsg
  | SeriesMsg | EventMsg | ControlsMsg | ControlMsg | ControlReplyMsg | SelectionMsg
  | SceneUpdateMsg | SceneTransformsMsg | SceneClearMsg | SceneClassesMsg
  | TextMsg | RecordingsMsg | RecordingOpenMsg | CursorMsg | TimeStateMsg | ProgramMsg | ProgramStateMsg | ResourceMsg;

/** Effective colour range of a snapshot: AUTO flags read stats[0] and stats[1]. */
export function snapshotRange(s: SnapshotMsg): [number, number] {
  let lo = s.vmin, hi = s.vmax;
  if (s.flags & SNAP_AUTO_VMIN && s.stats.length > 0) lo = s.stats[0];
  if (s.flags & SNAP_AUTO_VMAX && s.stats.length > 1) hi = s.stats[1];
  return [lo, hi];
}

// --- protocol 3.2 draft: the scenes subset (proposals: minor 2, capability "scenes", kind 6, 0x0070 to 0x0073, dtype u16; see the 3.2 block of include/caliper/live_net.h) ---

export const PROTOCOL_MINOR_3_2 = 2;
export const KIND_SCENE = 6;
export const CAP_SCENES = 'scenes';

export const PART_POINTS = 'points';
export const PART_LINES = 'lines';
export const PART_ARROWS = 'arrows';
export const PART_BOXES = 'boxes';
export const PART_MESH = 'mesh';
export const PART_IMAGE = 'image';
export const PART_CAMERA = 'camera';
export const PART_TRANSFORM = 'transform';
export type ScenePart = 'points' | 'lines' | 'arrows' | 'boxes' | 'mesh' | 'image' | 'camera';
export const SCENE_PARTS: readonly ScenePart[] = [PART_POINTS, PART_LINES, PART_ARROWS, PART_BOXES, PART_MESH, PART_IMAGE, PART_CAMERA];
/** The parts a SCENE_CLEAR may name: the scene parts and the entity transform. */
export const CLEAR_PARTS: readonly string[] = [...SCENE_PARTS, PART_TRANSFORM];
export const LINES_SEGMENTS = 'segments';
export const LINES_STRIPS = 'strips';
export const LINES_EDGES = 'edges';
export const COLOR_PER_VERTEX = 'vertex';
export const COLOR_PER_STRIP = 'strip';
export const COLOR_PER_EDGE = 'edge';
export const FORM_COLOR = 'color';
export const FORM_DEPTH = 'depth';
export const FORM_SEGMENTATION = 'segmentation';
export const FORM_ENCODED = 'encoded';
export const FORMAT_JPEG = 'jpeg';
export const FORMAT_PNG = 'png';
export const CONVENTION_RDF = 'rdf';
export const CONVENTION_RUB = 'rub';
export const SPACE_3D = '3d';
export const SPACE_2D = '2d';
export const UP_AXES_3D: readonly string[] = ['+x', '-x', '+y', '-y', '+z', '-z'];
export const UP_AXES_2D: readonly string[] = ['-y', '+y'];

export const MAX_LABELS = 4096;
export const MAX_CLASSES = 4096;
export const MAX_CLASS_ID = 65535;
export const MAX_IMAGE_SIDE = 8192;
export const MAX_ENCODED_BYTES = 16777216;
export const MAX_TRANSFORMS = 1024;
export const MAX_TIMELINES = 8;
/** A stamp value: absent on that timeline. */
export const STAMP_ABSENT = -(1n << 63n);

export const LIMITS_32 = {
  pathMax: 255,
  pathSegmentsMax: 16,
  unitsMax: 15,
  maxEntities: 1024,
  maxBatches: 2048,
  maxBatchPoints: 4194304,
  maxBatchItems: 1048576,
  maxLabels: MAX_LABELS,
  maxClasses: MAX_CLASSES,
  maxClassId: MAX_CLASS_ID,
  maxImageSide: MAX_IMAGE_SIDE,
  maxEncodedBytes: MAX_ENCODED_BYTES,
  maxTransforms: MAX_TRANSFORMS,
  maxTimelines: MAX_TIMELINES,
} as const;

/** The sections of each part, in the order a sender writes them; the section "stamp" follows them. */
export const PART_SECTIONS: Record<ScenePart, readonly string[]> = {
  points: ['positions', 'colors', 'values', 'class_ids', 'radii'],
  lines: ['positions', 'strip_lengths', 'indices', 'colors', 'values'],
  arrows: ['origins', 'vectors', 'colors', 'values'],
  boxes: ['centers', 'half_sizes', 'rotations', 'colors', 'class_ids'],
  mesh: ['positions', 'indices', 'normals', 'colors', 'values', 'uvs', 'texture'],
  image: ['pixels'],
  camera: [],
};
/** The keys of each part, beside the common ones, in the order their types are checked. */
export const PART_KEYS: Record<ScenePart, readonly string[]> = {
  points: ['color', 'radius', 'labels', 'colormap', 'flags', 'vmin', 'vmax', 'stats'],
  lines: ['mode', 'color_per', 'color', 'width', 'colormap', 'flags', 'vmin', 'vmax', 'stats'],
  arrows: ['color', 'width', 'colormap', 'flags', 'vmin', 'vmax', 'stats'],
  boxes: ['color', 'labels', 'fill', 'width', 'stats'],
  mesh: ['color', 'colormap', 'flags', 'vmin', 'vmax', 'stats'],
  image: ['form', 'opacity', 'scale', 'cloud', 'colormap', 'flags', 'vmin', 'vmax', 'format', 'stats'],
  camera: ['fx', 'fy', 'cx', 'cy', 'width', 'height', 'convention', 'plane_distance'],
};
/** The keys a camera part must carry. */
export const CAMERA_REQUIRED: readonly string[] = ['fx', 'fy', 'cx', 'cy', 'width', 'height', 'convention'];

/** The resolved default scene data view (SCENE_DATA_VIEW of _wire.py). */
export const SCENE_DATA_VIEW = {
  entities: { include: ['**'], exclude: [] as string[] },
  budget: { max_points: 100000, max_items: 100000, max_vertices: 1000000, max_extent: 1024 },
};

/** One section of a scene part; dtype 0 f32, 4 i32, 5 u8, 6 u16. A piece of a split update may hold any dtype: those with no typed array here (f16, bf16, i64) arrive as their raw bytes in a Uint8Array. */
export interface SceneArray { dtype: number; shape: number[]; data: Float32Array | Int32Array | Uint8Array | Uint16Array }

/** One part of one entity of a scene. */
export interface SceneUpdateMsg {
  type: 'scene_update';
  sub: number;
  view: string;
  seq: number;
  stamp_ns: number;
  path: string;
  part: ScenePart;
  /** Only the part's known sections, e.g. positions, colors, values, class_ids, radii, indices, pixels. */
  arrays: Record<string, SceneArray>;
  params: {
    mode?: 'segments' | 'strips' | 'edges';
    color_per?: 'vertex' | 'strip' | 'edge';
    color?: number[];
    radius?: number;
    labels?: string[];
    width?: number;
    fill?: boolean;
    colormap?: number;
    flags?: number;
    vmin?: number;
    vmax?: number;
    form?: 'color' | 'depth' | 'segmentation' | 'encoded';
    opacity?: number;
    scale?: number;
    cloud?: boolean;
    format?: 'jpeg' | 'png';
    fx?: number;
    fy?: number;
    cx?: number;
    cy?: number;
    /** An integer for a camera. */
    height?: number;
    convention?: 'rdf' | 'rub';
    plane_distance?: number;
    /** NaN where the producer sent null. */
    stats?: number[];
  };
  static?: boolean;
  state?: boolean;
  /** The generation of the TIME_STATE a recorded message answers. */
  cursor?: number;
  axes?: PayloadAxis[];
  items?: string;
  /** The piece form (capability "scene_pieces"): the scene program, the library piece pin, the count of each resource, and the arrays the batch held (the wire key is "arrays"). No section, no split, so `arrays` is empty. */
  program?: string;
  piece?: { name: string; version: number; hash?: string };
  resources?: Record<string, number>;
  names?: string[];
  /** The key of a piece of a split update; its `arrays` then hold the piece's rows, and `params` the part's keys in piece 0 only. */
  split?: SplitKey;
  /** The i64 (T,) stamp section. */
  stamp?: BigInt64Array;
}

export interface SplitKey {
  update: number;
  pieces: number;
  piece: number;
  first: Record<string, number>;
  rows?: Record<string, number>;
}

/** The entity transforms of up to 1024 entities: 4 x 4 matrices, column-major, row k of `matrices` for paths[k]. */
export interface SceneTransformsMsg {
  type: 'scene_transforms';
  sub: number;
  view: string;
  seq: number;
  stamp_ns: number;
  paths: string[];
  /** (K, 16) f32, K * 16 values. */
  matrices: Float32Array;
  static?: boolean;
  state?: boolean;
  /** One i64 (T,) stamp for all the transforms. */
  stamp?: BigInt64Array;
  /** i64 (K, T), K * T values, one stamp each; `stamps_t` is T (decoded: always set). */
  stamps?: BigInt64Array;
  stamps_t?: number;
  cursor?: number;
}

/** Clears an entity, its subtree, some of its parts, or with reset the whole scene. */
export interface SceneClearMsg {
  type: 'scene_clear';
  sub: number;
  view: string;
  seq: number;
  stamp_ns: number;
  path: string;
  recursive?: boolean;
  parts?: string[];
  reset?: boolean;
  static?: boolean;
  state?: boolean;
  stamp?: BigInt64Array;
  cursor?: number;
}

export interface SceneClass { id: number; label: string; color: number[] }

/** A scene's class table, which replaces the previous one whole. */
export interface SceneClassesMsg {
  type: 'scene_classes';
  sub: number;
  view: string;
  seq: number;
  stamp_ns: number;
  classes: SceneClass[];
  static?: boolean;
  state?: boolean;
  stamp?: BigInt64Array;
  cursor?: number;
}

/** 3.2 draft: new lines of a text log, appended as a series' values are. Line i is bytes offsets[i] to offsets[i + 1] of `text` (UTF-8), at level levels[i]. */
export interface TextMsg {
  type: 'text';
  sub: number;
  view: string;
  start: number;
  total: number;
  stamp_ns: number;
  text: Uint8Array;
  offsets: Int32Array;
  levels: Uint8Array;
  dropped?: number;
  stamp?: BigInt64Array;
  stamps?: BigInt64Array;
  stamps_t?: number;
  cursor?: number;
}

export interface RecordingEntry {
  id: string;
  title: string;
  script: string;
  bytes: number;
  items: number;
  complete: boolean;
  created_unix_ms?: number | null;
  ended_unix_ms?: number | null;
  timelines: { name: string; kind: 'sequence' | 'time'; epoch?: boolean; min: string; max: string }[];
}
export interface RecordingsMsg { type: 'recordings'; recordings: RecordingEntry[] }
export interface RecordingOpenMsg { type: 'recording_open'; request_id: number; recording: string }
/** Viewer to server: move this viewer's cursor; only the fields of the op are kept. */
export interface CursorMsg {
  type: 'cursor';
  request_id: number;
  op: 'seek' | 'play' | 'pause' | 'step' | 'follow';
  timeline?: string;
  value?: string;
  count?: number;
  speed?: number;
  pace?: 'recorded' | 'even';
  loop?: [string, string];
}
/** Server to viewer: the state of the viewer's cursor; `cursor` is stamp text, null while following. */
export interface TimeStateMsg {
  type: 'time_state';
  request_id?: number;
  session: string;
  generation: number;
  state: 'following' | 'paused' | 'playing';
  timeline: string;
  cursor: string | null;
  ranges: Record<string, [string, string]>;
  live: boolean;
  speed?: number;
  pace?: 'recorded' | 'even';
  loop?: [string, string];
}

/** The orbit camera a scene view opens on; null-valued keys are dropped. */
export interface ViewCameraOrbit { target?: number[]; distance?: number; azimuth?: number; elevation?: number }

/** The key "scene" of a VIEWS entry of kind 6, with its defaults filled in. */
export interface SceneViewInfo {
  space: '3d' | '2d';
  up: string;
  units: string;
  /** The orbit keys or the entity path of a camera; undefined when absent, unreadable, or for a 2-D scene. */
  view_camera: ViewCameraOrbit | string | undefined;
}

// --- 3.2 draft: timelines, text, bars, recordings, split updates ---

export const CAP_TIMELINES = 'timelines';
export const CAP_TEXT = 'text';
export const CAP_BARS = 'bars';
export const CAP_RECORDINGS = 'recordings';
export const ERR_UNKNOWN_RECORDING = 'unknown_recording';
export const ERR_BAD_RECORDING = 'bad_recording';
export const ERR_NOT_SEEKABLE = 'not_seekable';
export const ERR_BAD_CURSOR = 'bad_cursor';
export const TIMELINE_SEQUENCE = 'sequence';
export const TIMELINE_TIME = 'time';
export const LEVEL_DEBUG = 0;
export const LEVEL_INFO = 1;
export const LEVEL_WARNING = 2;
export const LEVEL_ERROR = 3;
export const CURSOR_SEEK = 'seek';
export const CURSOR_PLAY = 'play';
export const CURSOR_PAUSE = 'pause';
export const CURSOR_STEP = 'step';
export const CURSOR_FOLLOW = 'follow';
export const CURSOR_OPS: readonly string[] = [CURSOR_SEEK, CURSOR_PLAY, CURSOR_PAUSE, CURSOR_STEP, CURSOR_FOLLOW];
export const PACE_RECORDED = 'recorded';
export const PACE_EVEN = 'even';
export const TIME_FOLLOWING = 'following';
export const TIME_PAUSED = 'paused';
export const TIME_PLAYING = 'playing';
export const STAMP_TEXT_MAX = 20;
export const MAX_PIECES = 16;
export const MAX_UPDATE_BYTES = 536870912;
export const MAX_INCOMPLETE_UPDATES = 2;
export const TEXT_LINE_MAX = 4096;
export const TEXT_MAX_LINES = 65536;
export const TEXT_RETAIN_MIN = 4096;
export const MAX_BARS = 4096;
export const MAX_RECORDINGS = 1024;
export const MAX_RECORDED = 64;
export const MAX_CURSOR_STEP = 1000;
export const CURSOR_RATE_HZ = 30;
export const TIME_STATE_RATE_HZ = 30;
export const MIN_SPEED = 0.0625;
export const MAX_SPEED = 64;
export const RECORDING_LINGER_MS = 600000;

// --- 3.2 draft: GPU programs and interventions (include/caliper/live_net.h, capabilities "programs" and "interventions") ---

/** Capability: a viewer sends PROGRAM only to a producer that announced it, and the producer answers with PROGRAM_STATE. */
export const CAP_PROGRAMS = 'programs';
/** Capability: CONTROLS carries "interventions" and CONTROL accepts the op "intervene". */
export const CAP_INTERVENTIONS = 'interventions';
/** PROGRAM "op": install or replace one program. */
export const PROGRAM_SET = 'set';
/** PROGRAM "op": remove one program. */
export const PROGRAM_REMOVE = 'remove';
/** PROGRAM "op": change the parameters of one program with no compile. */
export const PROGRAM_PARAMS = 'params';
/** The name of the PROGRAM section that holds the WGSL source as UTF-8. */
export const SECTION_WGSL = 'wgsl';
/** The only intervention op; scale, steer and patch are reserved. */
export const INTERVENE_ABLATE = 'ablate';
/** Programs of a producer, and the most items of a PROGRAM_STATE. */
export const MAX_PROGRAMS = 16;
/** Bytes of a program id; "program.<id>" fits a view name. */
export const PROGRAM_ID_MAX = 55;
/** Bytes of the section "wgsl". */
export const PROGRAM_SOURCE_MAX = 65536;
/** Unrequested PROGRAM_STATE messages a second, at most. */
export const PROGRAM_STATE_RATE_HZ = 2;
/** Bytes of an intervention point. */
export const POINT_MAX = 255;
/** Most entries of CONTROLS "interventions" "points". */
export const MAX_POINTS_LISTED = 4096;
/** Most entries of CONTROLS "interventions" "active". */
export const MAX_INTERVENTIONS = 64;
/** Most entries of a declaration's "inputs". */
export const PROGRAM_MAX_INPUTS = 8;
/** Most elements of a geometry output ("max_count"). */
export const PROGRAM_MAX_COUNT = 1000000;
/** Most pixels of the width and of the height of a pixels output. */
export const PROGRAM_SIDE_MAX = 2048;
/** Most milliseconds of "limits" "time_ms". */
export const PROGRAM_TIME_MS_MAX = 50;
/** Most megabytes of "limits" "memory_mb". */
export const PROGRAM_MEMORY_MB_MAX = 512;
/** Most of "limits" "every", the cadence in steps. */
export const PROGRAM_EVERY_MAX = 10000;
/** Most of each of the three "workgroups". */
export const PROGRAM_WORKGROUPS_MAX = 65535;

// --- 3.2 draft: programmable graphics (include/caliper/live_net.h, capability "resources") ---

/** Capability: RESOURCE, and PROGRAM of an instance of a piece with the sections "derive", "draw" and "common". */
export const CAP_RESOURCES = 'resources';
/** Capability: the piece form of SCENE_UPDATE, whose arrays travel as RESOURCE versions of a scene program "scene.<n>". */
export const CAP_SCENE_PIECES = 'scene_pieces';
/** The prefix of the id of a scene program: "scene." and 1 to 9 decimal digits. */
export const SCENE_PROGRAM_PREFIX = 'scene.';
/** PROGRAM sections of an instance of a piece, and the section of RESOURCE. */
export const SECTION_DERIVE = 'derive';
export const SECTION_DRAW = 'draw';
export const SECTION_COMMON = 'common';
export const SECTION_RESOURCE = 'resource';
/** Resources of one program. */
export const MAX_RESOURCES = 8;
/** Bytes of the section "resource": 64 MiB less 64 KiB, so that a resource fits one frame with its metadata. */
export const RESOURCE_MAX_BYTES = 67043328;
/** Bytes of a buffer version in parts, within the default WebGPU storage binding limit. */
export const RESOURCE_VERSION_MAX_BYTES = 134217728;
/** Bytes of a full 8,192 by 8,192 RGBA8 or R32 texture version in parts. */
export const RESOURCE_TEXTURE_VERSION_MAX_BYTES = 268435456;
/** At most 32 chunks of 8 MiB for one texture version. */
export const RESOURCE_MAX_PARTS = 32;
/** RESOURCE frames a receiver waits for the next part of a version in parts before it drops the version. */
export const RESOURCE_PART_WAIT = 256;
/** Most bytes of one element of a resource, the "stride" (a multiple of 4). */
export const RESOURCE_STRIDE_MAX = 256;
/** Most texels of the "width" and of the "height" of a texture resource. */
export const RESOURCE_SIDE_MAX = 8192;
/** The values of the key "format" of a texture resource. */
export const TEXTURE_FORMATS = ['rgba8unorm', 'r8unorm', 'r32uint'] as const;
/** The values of the key "element" of a buffer of narrow elements. */
export const RESOURCE_ELEMENTS = ['u8', 'u16'] as const;
/** Most bytes of the "name" of a piece pin, [a-z][a-z0-9_]*. */
export const PIECE_NAME_MAX = 32;
/** Most of the "version" of a piece pin, counted from 1. */
export const PIECE_VERSION_MAX = 1000000;
/** Most bytes of the section "common": 8 lib modules and common.wgsl of 64 KiB each. */
export const PIECE_COMMON_MAX = 589824;
/** Most items of the "segments" of an instance of a piece. */
export const PIECE_SEGMENTS_MAX = 16;
/** The kind of a status item for an instance of a piece. */
export const STATUS_PIECE = 'piece';
/** Bytes of an input's "source" tensor name. */
export const TENSOR_NAME_MAX = 255;
/** A declaration's output kinds. */
export const OUTPUT_GEOMETRY = 'geometry';
export const OUTPUT_PIXELS = 'pixels';
export const OUTPUT_TENSOR = 'tensor';
/** The kind of a status item for the program of a built-in operator. */
export const STATUS_OPERATOR = 'operator';
/** Most names of a params object, bytes of a params name and bytes of a declaration's "builtin". */
export const PROGRAM_PARAMS_MAX = 16;
export const PROGRAM_PARAM_NAME_MAX = 63;
export const PROGRAM_BUILTIN_MAX = 32;
/** Most axes and most elements of a tensor output's shape. */
export const TENSOR_AXES_MAX = 4;
export const TENSOR_ELEMENTS_MAX = 1000000;
/** The states of a program in a PROGRAM_STATE status item. */
export const PROGRAM_STATES: readonly string[] = ['waiting', 'review', 'compiled', 'running', 'refused', 'stopped'];
/** The arrays a geometry output may list; "positions" is required. */
export const PROGRAM_ARRAYS: readonly string[] = ['positions', 'colors', 'values', 'radii'];
/** The states of an entry of CONTROLS "interventions" "active". */
export const INTERVENTION_STATES: readonly string[] = ['pending', 'active', 'ended'];
/** M8 specification limits (python/caliper/_intervention_protocol.py). */
export const INTERVENTION_LIMITS = { maxVector: 16384, maxPositions: 32, maxBytes: 262144 } as const;

type InterventionTag<K extends string> = { kind: K; op?: K | null } | { op: K; kind?: K | null };
type FeatureIntervention = { feature: number; dictionary?: string; positions?: number[] };
export type InterventionSpec = { point: string; from_step?: number | null; until_step?: number | null } & (
  | InterventionTag<'ablate'> | InterventionTag<'steer'>
  | (InterventionTag<'add'> & { vector: number[]; coefficient: number; positions?: number[] })
  | (InterventionTag<'scale'> & FeatureIntervention & { multiplier: number })
  | (InterventionTag<'clamp'> & FeatureIntervention & { value: number })
  | (InterventionTag<'patch'> & { source_prompt: string; source_position?: number; positions?: number[] })
);
export type InterventionStatus = InterventionSpec & {
  state: 'pending' | 'active' | 'ended'; since_step?: number | null; error?: string; error_code?: string;
};

/** The declaration of a program, checked on decode as the Python codec checks it and re-emitted unchanged (`design/gpu-programs.md`). */
export type ProgramDeclaration = Record<string, unknown>;
/** A status item of PROGRAM_STATE: id, kind and state, and the optional keys, null or absent when unknown. Re-emitted unchanged. */
export type ProgramStatus = Record<string, unknown>;
/** CONTROL "intervene": exactly one of set, clear and clear_all. */
export interface InterveneRequest {
  set?: InterventionSpec | null;
  clear?: string | null;
  clear_all?: true | null;
}
/** CONTROLS "interventions": the points an intervention may use and the interventions in force. Re-emitted unchanged. */
export interface InterventionsInfo { points: Record<string, unknown>[]; active: InterventionStatus[] }

/** Viewer to producer: install or replace (op set), remove (op remove) or change the parameters of (op params) one GPU program. A set carries the declaration without its source, the scene of a geometry output, the SHA-256 of the source and the section "wgsl". */
export interface ProgramMsg {
  type: 'program';
  request_id: number;
  op: 'set' | 'remove' | 'params';
  id: string;
  /** params: the names of the program's uniform and their finite numbers. */
  params?: Record<string, number>;
  /** set */
  declaration?: ProgramDeclaration;
  /** set: the source, the section "wgsl" as UTF-8. */
  wgsl?: string;
  /** set: SHA-256 of the source in 64 lowercase hex digits. */
  sha256?: string;
  /** set: the scene of a geometry output's panel. */
  scene?: string;
  /** Instance of a piece, set: the pin {name, version, hash}, where the hash is "sha256:" and 64 lowercase hex digits. */
  piece?: { name: string; version: number; hash?: string };
  /** Instance of a piece, set: the piece's manifest. */
  manifest?: Record<string, unknown>;
  /** Instance of a piece, set: an input name to "live:<tensor>" or "program:<id>/<resource>". */
  bind?: Record<string, string>;
  /** Instance of a piece, set: the scene panel of the draw half. */
  panel?: string;
  /** Instance of a piece, set: the {part, first_line, lines} of each part of the section "common". */
  segments?: { part: string; first_line: number; lines: number }[];
  /** Instance of a piece, set: the section "common" as UTF-8, the lib modules and common.wgsl. */
  common?: string;
  /** Instance of a piece, set: the section "derive" as UTF-8, the derive half's WGSL. */
  derive?: string;
  /** Instance of a piece, set: the section "draw" as UTF-8, the draw half's WGSL. */
  draw?: string;
}

/** Producer to viewer or hub: one version of one published resource of a program, the valid elements that a derive pass wrote. A texture also has width and height, with a stride of 4 and a count of the 32-bit words of their texels. */
export interface ResourceMsg {
  type: 'resource';
  /** The id of the program. */
  program: string;
  /** The WGSL name of the resource. */
  resource: string;
  /** The run's step at the pass. */
  step: number;
  /** The valid elements. */
  count: number;
  /** The bytes of one element, a multiple of 4. */
  stride: number;
  /** The section "resource": exactly count * stride bytes, a view of the frame without a copy. */
  data: Uint8Array;
  /** A texture's width in texels. */
  width?: number;
  /** A texture's height in texels. */
  height?: number;
  /** A texture's format, rgba8unorm when absent: the count is the 32-bit words of its texels, (width * height + 3) / 4 for r8unorm. */
  format?: 'rgba8unorm' | 'r8unorm' | 'r32uint';
  /** A buffer of narrow elements: "u8" or "u16", packed into 32-bit words from the low bits up, with a stride of 4 and a count of (elements * bytes + 3) / 4. */
  element?: 'u8' | 'u16';
  /** The exact count of narrow elements, with `element`. */
  elements?: number;
  /** A part of a version in parts: the sender's counter of such versions, this part's number from 0, the number of parts (2 to MAX_PIECES), and the first element that `data` holds. */
  split?: ResourceSplit;
}

/** The key "split" of a RESOURCE part. */
export interface ResourceSplit {
  version: number;
  part: number;
  parts: number;
  first: number;
}

/** Producer to viewer: the status items of the programs in order of installation. It answers each PROGRAM with its request_id and goes out without one after a state change. */
export interface ProgramStateMsg {
  type: 'program_state';
  programs: ProgramStatus[];
  request_id?: number;
  step?: number;
  step_ms?: number;
}
