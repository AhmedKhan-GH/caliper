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
} as const;
export type MessageType = keyof typeof MSG;

export type Kind = 'heatmap' | 'image' | 'points' | 'series' | 'histogram';
export const KIND_IDS: Record<Kind, number> = { heatmap: 1, image: 2, points: 3, series: 4, histogram: 5 };
export function kindName(id: number): Kind | null {
  for (const k in KIND_IDS) if (KIND_IDS[k as Kind] === id) return k as Kind;
  return null;
}

export type ColormapName = 'viridis' | 'magma' | 'rdbu';
export const COLORMAPS: readonly ColormapName[] = ['viridis', 'magma', 'rdbu'];

export const DTYPE = { f32: 0, f16: 1, bf16: 2, i64: 3, i32: 4, u8: 5 } as const;
export const DTYPE_SIZE: Record<number, number> = { 0: 4, 1: 2, 2: 2, 3: 8, 4: 4, 5: 1 };

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
export const CAP = { events: 'events', controls: 'controls', selection: 'selection' } as const;

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
export interface ViewsMsg { type: 'views'; views: ViewEntry[]; session?: string | null }
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
  /** heatmap (H, W) f32, image (H, W, 4) u8, points (N, D) f32, histogram (B,) f32. */
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
export type ParamType = 'number' | 'integer' | 'boolean' | 'choice';
export type ParamValue = number | boolean | string;
export type ControlOp = 'pause' | 'resume' | 'step' | 'set' | 'run';
export const CONTROL_OPS: readonly ControlOp[] = ['pause', 'resume', 'step', 'set', 'run'];

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
export interface ControlsMsg { type: 'controls'; run: RunInfo; params: ParamInfo[]; actions: ActionInfo[] }
/** Viewer to producer: pause, resume, step, set a parameter, run an action. */
export interface ControlMsg {
  type: 'control';
  request_id: number;
  op: ControlOp;
  name?: string;
  value?: ParamValue;
  count?: number;
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
  | SeriesMsg | EventMsg | ControlsMsg | ControlMsg | ControlReplyMsg | SelectionMsg;

/** Effective colour range of a snapshot: AUTO flags read stats[0] and stats[1]. */
export function snapshotRange(s: SnapshotMsg): [number, number] {
  let lo = s.vmin, hi = s.vmax;
  if (s.flags & SNAP_AUTO_VMIN && s.stats.length > 0) lo = s.stats[0];
  if (s.flags & SNAP_AUTO_VMAX && s.stats.length > 1) hi = s.stats[1];
  return [lo, hi];
}
