// Wire protocol 3.1 controls, linked selection, interpretation, and 3.2
// interventions message validators, decoders, and encoders.

import {
  CONTROL_OPS, INTERPRET_LIMITS, INTERVENTION_LIMITS, INTERVENTION_STATES, LIMITS, LIMITS_31,
  MAX_INTERVENTIONS, MAX_POINTS_LISTED, MSG, POINT_MAX, RUN_STATES,
  type ActionInfo, type ControlMsg, type ControlOp, type ControlReplyMsg, type ControlsMsg,
  type InterpretationRequest, type InterpretationResult, type InterveneRequest, type InterventionsInfo,
  type ParamInfo, type ParamValue, type RunInfo, type SelectionDetails, type SelectionMsg,
} from './messages.js';
import { isParamText } from './param-text.js';
import {
  bad, boundedText, canonicalJson, drop, encodeWithSource, i64, isAscending, isBool, isFiniteNum,
  isInt, isName, isObj, isSafeUint, isStr, isText, isU32, isUint, listOf, numberSources, opt,
  present, req, sourceField, stampSection, stampSections, utf8, utf8Strict, type Frame,
} from './frame.js';

const INT_EXACT = 2 ** 53;
const MAX_SHAPE_DIMS = 8;
const isPoint = boundedText(POINT_MAX);

const fitsType = (p: ParamInfo, v: unknown) => {
  if (p.type === 'number') return isFiniteNum(v);
  if (p.type === 'integer') return isInt(v) && Math.abs(v) <= INT_EXACT;
  if (p.type === 'boolean') return isBool(v);
  if (p.type === 'text') return isParamText(v);
  return typeof v === 'string' && !!p.choices?.includes(v);
};

const isParam = (v: unknown) => {
  if (!isObj(v)) return false;
  const p = v as ParamInfo;
  if (!isName(p.name) || !['number', 'integer', 'boolean', 'choice', 'text'].includes(p.type)) return false;
  if (p.type === 'number' || p.type === 'integer') {
    const bound = p.type === 'number' ? isFiniteNum : (x: unknown) => isInt(x) && Math.abs(x) <= INT_EXACT;
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
const isSetValue = (v: unknown) => isFiniteNum(v) || isBool(v) || isParamText(v);

const interpretationText = (v: unknown, limit: number = INTERPRET_LIMITS.maxString, empty = true) => {
  if (typeof v !== 'string' || (!empty && !v.length) || v.includes('\0')) return false;
  const bytes = utf8.encode(v);
  return bytes.length <= limit && utf8Strict.decode(bytes) === v;
};
const numericSource = (v: number, source: unknown) => typeof source === 'string' ? source : JSON.stringify(v);
const interpretationInteger = (v: unknown, source: unknown, low: number, high: number) =>
  isInt(v) && v >= low && v <= high && !/[.eE]/.test(numericSource(v, source));

function isInterpretationObject(v: unknown, source?: unknown): boolean {
  if (isObj(v)) source ??= numberSources.get(v as object);
  let count = 0;
  const visit = (value: unknown, depth: number, raw: unknown): boolean => {
    if (depth > INTERPRET_LIMITS.maxDepth || ++count > INTERPRET_LIMITS.maxValues) return false;
    if (value === null || isBool(value)) return true;
    if (typeof value === 'number')
      // Only a decoded integer token has integer semantics. Plain JS numbers
      // outside the safe integer range use the finite float representation.
      return Number.isFinite(value) && (typeof raw !== 'string' || /[.eE]/.test(raw) || Number.isSafeInteger(value));
    if (typeof value === 'string') return interpretationText(value);
    if (Array.isArray(value))
      return value.length <= INTERPRET_LIMITS.maxList && value.every((x, i) => visit(x, depth + 1, sourceField(raw, i)));
    if (!isObj(value)) return false;
    const entries = Object.entries(value as Record<string, unknown>);
    return entries.length <= INTERPRET_LIMITS.maxFields && entries.every(([key, x]) =>
      interpretationText(key, 63, false) && visit(x, depth + 1, sourceField(raw, key)));
  };
  return isObj(v) && visit(v, 0, source) && utf8.encode(canonicalJson(v, source, true)).length <= INTERPRET_LIMITS.maxBytes;
}

const INTERPRET_PARAMS: Record<string, readonly string[]> = {
  'feature.dictionary': [], 'trace.status': [], 'trace.validate': [],
  'patch.sweep': ['clean_prompt', 'corrupted_prompt', 'metric'],
  'feature.show': ['layer', 'feature', 'node'],
  'feature.examples': ['layer', 'feature', 'node', 'limit'],
  'trace.run': ['prompt', 'max_targets', 'max_nodes', 'max_edges', 'max_n_logits',
    'desired_logit_prob', 'node_threshold', 'edge_threshold', 'batch_size', 'update_interval'],
  'trace.read': ['node_offset', 'node_limit', 'edge_offset', 'edge_limit'],
};
const INTERPRET_BOUNDS: Record<string, readonly [number, number]> = {
  layer: [0, 4095], feature: [0, 2147483647], node: [0, 8191], limit: [0, 16],
  max_targets: [0, 5000], max_nodes: [1, 8192], max_edges: [0, 262144],
  max_n_logits: [1, 10], update_interval: [1, 128],
  node_offset: [0, 8192], edge_offset: [0, 262144], node_limit: [0, 256], edge_limit: [0, 256],
};

function isInterpretationRequest(v: unknown, source?: unknown): boolean {
  if (!isInterpretationObject(v, source)) return false;
  const o = v as Record<string, unknown>;
  if (Object.keys(o).length !== 3 || !interpretationInteger(o.version, sourceField(source, 'version'), 1, 1)
    || typeof o.method !== 'string' || !Object.hasOwn(INTERPRET_PARAMS, o.method) || !isObj(o.params)) return false;
  const params = o.params as Record<string, unknown>;
  if (Object.keys(params).some((k) => !INTERPRET_PARAMS[o.method as string].includes(k))) return false;
  const rawParams = sourceField(source, 'params');
  for (const [key, [low, high]] of Object.entries(INTERPRET_BOUNDS))
    if (Object.hasOwn(params, key) && !interpretationInteger(params[key], sourceField(rawParams, key), low, high)) return false;
  if (o.method === 'feature.show' || o.method === 'feature.examples')
    return ('node' in params && !('layer' in params) && !('feature' in params))
      || (!('node' in params) && 'layer' in params && 'feature' in params);
  if (o.method === 'patch.sweep') {
    if (!interpretationText(params.clean_prompt, INTERPRET_LIMITS.maxString, false)
      || !interpretationText(params.corrupted_prompt, INTERPRET_LIMITS.maxString, false) || !isObj(params.metric)) return false;
    const metric = params.metric as Record<string, unknown>;
    const raw = sourceField(rawParams, 'metric');
    return Object.keys(metric).length === 2
      && ['positive_token', 'negative_token'].every((key) => interpretationInteger(metric[key], sourceField(raw, key), 0, 2147483647))
      && metric.positive_token !== metric.negative_token;
  }
  if (o.method === 'trace.run') {
    if (Object.hasOwn(params, 'batch_size') && params.batch_size !== null
      && !interpretationInteger(params.batch_size, sourceField(rawParams, 'batch_size'), 1, 512)) return false;
    for (const key of ['desired_logit_prob', 'node_threshold', 'edge_threshold'])
      if (Object.hasOwn(params, key) && (!isFiniteNum(params[key]) || (params[key] as number) < 0 || (params[key] as number) > 1)) return false;
    return interpretationText(params.prompt, INTERPRET_LIMITS.maxString, false);
  }
  return true;
}

const isCell = (v: unknown) => v === null || isBool(v) || isFiniteNum(v) || (typeof v === 'string' && utf8.encode(v).length <= LIMITS_31.detailTextMax);
const isDetails = (v: unknown) => {
  if (!isObj(v)) return false;
  const d = v as SelectionDetails;
  return Array.isArray(d.fields) && d.fields.length <= LIMITS_31.maxDetailFields && d.fields.every(isName)
    && new Set(d.fields).size === d.fields.length
    && Array.isArray(d.items) && d.items.length <= LIMITS_31.maxDetailItems && d.items.every(isUint)
    && Array.isArray(d.values) && d.values.length === d.items.length
    && d.values.every((row) => Array.isArray(row) && row.length === d.fields.length && row.every(isCell));
};

const INTERVENTION_FIELDS: Record<string, readonly string[]> = {
  ablate: [], steer: [], add: ['vector', 'coefficient', 'positions'],
  scale: ['feature', 'multiplier', 'positions', 'dictionary'],
  clamp: ['feature', 'value', 'positions', 'dictionary'],
  patch: ['source_prompt', 'source_position', 'positions'],
};
const interventionNumber = (v: unknown, source: unknown): boolean => typeof v === 'number'
  && Number.isFinite(v) && (typeof source !== 'string' || /[.eE]/.test(source) || Number.isSafeInteger(v));

function isInterventionSpec(v: unknown, lifecycle: boolean, source: unknown): boolean {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  if (present(o.op) && present(o.kind) && o.op !== o.kind) return false;
  const kind = o.kind ?? o.op;
  if (typeof kind !== 'string' || !Object.hasOwn(INTERVENTION_FIELDS, kind) || !interpretationText(o.point, POINT_MAX, false)) return false;
  const fields = ['point', 'op', 'kind', 'from_step', 'until_step', ...INTERVENTION_FIELDS[kind]];
  if (lifecycle) {
    fields.push('since_step', 'state', 'error', 'error_code');
    if (typeof o.state !== 'string' || !INTERVENTION_STATES.includes(o.state)) return false;
    if (Object.hasOwn(o, 'error') && !interpretationText(o.error, 1024)) return false;
    if (Object.hasOwn(o, 'error_code') && !interpretationText(o.error_code, 63, false)) return false;
  }
  if (Object.keys(o).some((key) => !fields.includes(key))) return false;
  for (const key of ['from_step', 'until_step', 'since_step'])
    if (present(o[key]) && !interpretationInteger(o[key], sourceField(source, key), 0, Number.MAX_SAFE_INTEGER)) return false;
  if (present(o.from_step) && present(o.until_step) && (o.until_step as number) <= (o.from_step as number)) return false;
  if (Object.hasOwn(o, 'positions')) {
    const positions = o.positions;
    const raw = sourceField(source, 'positions');
    if (!Array.isArray(positions) || positions.length > INTERVENTION_LIMITS.maxPositions
      || !positions.every((x, i) => interpretationInteger(x, sourceField(raw, i), 0, 31))
      || new Set(positions).size !== positions.length) return false;
  }
  if (Object.hasOwn(o, 'dictionary') && !interpretationText(o.dictionary, 255, false)) return false;
  if (kind === 'add') {
    const raw = sourceField(source, 'vector');
    if (!Array.isArray(o.vector) || !o.vector.length || o.vector.length > INTERVENTION_LIMITS.maxVector
      || !o.vector.every((x, i) => interventionNumber(x, sourceField(raw, i)))
      || !interventionNumber(o.coefficient, sourceField(source, 'coefficient'))) return false;
  }
  if (kind === 'scale' || kind === 'clamp') {
    const key = kind === 'scale' ? 'multiplier' : 'value';
    if (!interpretationInteger(o.feature, sourceField(source, 'feature'), 0, 2147483647)
      || !interventionNumber(o[key], sourceField(source, key))) return false;
  }
  if (kind === 'patch') {
    if (!interpretationText(o.source_prompt, 4096, false)) return false;
    if (Object.hasOwn(o, 'source_position') && !interpretationInteger(o.source_position, sourceField(source, 'source_position'), 0, 31)) return false;
  }
  return utf8.encode(canonicalJson(v, source, true)).length <= INTERVENTION_LIMITS.maxBytes;
}

/** CONTROL's intervene: exactly one of set, clear and clear_all. */
function isInterveneRequest(v: unknown): boolean {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  if (Object.keys(o).some((key) => !['set', 'clear', 'clear_all'].includes(key))) return false;
  if (['set', 'clear', 'clear_all'].filter((k) => present(o[k])).length !== 1) return false;
  if (present(o.set)) return isInterventionSpec(o.set, false, numberSources.get(o)?.set);
  if (present(o.clear)) return interpretationText(o.clear, POINT_MAX, false);
  return o.clear_all === true;
}

function isPointEntry(v: unknown): boolean {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  return isPoint(o.point) && (o.kind === 'module' || o.kind === 'head') && isPoint(o.module) && (!present(o.head) || isU32(o.head))
    && (!present(o.type) || isName(o.type)) && (!present(o.shape) || listOf(isSafeUint, MAX_SHAPE_DIMS)(o.shape));
}

/** CONTROLS' interventions: points (at most MAX_POINTS_LISTED) and active (at most MAX_INTERVENTIONS). */
function isInterventions(v: unknown): boolean {
  if (!isObj(v)) return false;
  const o = v as Record<string, unknown>;
  const source = numberSources.get(o)?.active;
  return listOf(isPointEntry, MAX_POINTS_LISTED)(o.points) && Array.isArray(o.active) && o.active.length <= MAX_INTERVENTIONS
    && o.active.every((x, i) => isInterventionSpec(x, true, sourceField(source, i)));
}

export function decodeControls(f: Frame): ControlsMsg {
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
  const interventions = opt<InterventionsInfo | undefined>(f, 'interventions', isInterventions, '{points, active} of intervention points and interventions', undefined);
  if (interventions !== undefined) c.interventions = interventions;
  return c;
}

export function decodeControl(f: Frame): ControlMsg {
  const t = f.type;
  const op = req<ControlOp>(f, 'op', (v) => CONTROL_OPS.includes(v as ControlOp), 'pause, resume, step, set, run, intervene or interpret');
  const name = opt<string | undefined>(f, 'name', isName, 'a parameter or action name', undefined);
  if ((op === 'set' || op === 'run') && name === undefined) bad(t, `'${op}' needs 'name'`);
  const intervene = op === 'intervene'
    ? req<InterveneRequest>(f, 'intervene', isInterveneRequest, 'exactly one of {set: specification}, {clear: point} and {clear_all: true}')
    : undefined;
  const interpretation = op === 'interpret'
    ? req<InterpretationRequest>(f, 'interpretation', (v) => isInterpretationRequest(v, numberSources.get(f.meta)?.interpretation), 'a version 1 interpretation request')
    : undefined;
  if (op !== 'interpret' && 'interpretation' in f.meta) bad(t, 'interpretation requires op interpret');
  return {
    type: 'control',
    request_id: req(f, 'request_id', isU32, 'a uint32'),
    op,
    name: op === 'set' || op === 'run' ? name : undefined,
    value: op === 'set' ? req<ParamValue>(f, 'value', isSetValue, 'a number, a boolean or bounded text') : undefined,
    count: op === 'step' ? opt<number | undefined>(f, 'count', (v) => isU32(v) && (v as number) >= 1, 'a positive uint32', undefined) : undefined,
    intervene,
    interpretation,
  };
}

export function decodeControlReply(f: Frame): ControlReplyMsg {
  const t = f.type;
  const status = req<ControlReplyMsg['status']>(f, 'status', (v) => v === 'ok' || v === 'queued' || v === 'error', 'ok, queued or error');
  const op = req<ControlOp>(f, 'op', (v) => CONTROL_OPS.includes(v as ControlOp), 'a control op');
  const error = opt<{ code: string; message?: string } | undefined>(f, 'error', (v) => isObj(v) && isStr((v as Record<string, unknown>).code)
    && ((v as Record<string, unknown>).message == null || isText((v as Record<string, unknown>).message)), '{code, message}', undefined);
  if ((error === undefined) === (status === 'error')) bad(t, "a control_reply carries 'error' exactly when its status is 'error'");
  if (status === 'queued' && op !== 'run' && op !== 'interpret') bad(t, "only 'run' or 'interpret' is queued");
  const result = opt<InterpretationResult | undefined>(f, 'result', (v) => isInterpretationObject(v, numberSources.get(f.meta)?.result), 'a bounded structured result', undefined);
  if ('result' in f.meta && (op !== 'interpret' || status !== 'ok')) bad(t, 'result requires a successful interpretation reply');
  if (op === 'interpret' && status === 'ok' && result === undefined) bad(t, 'a successful interpretation reply requires result');
  return {
    type: 'control_reply',
    request_id: req(f, 'request_id', isU32, 'a uint32'),
    op,
    status,
    name: opt(f, 'name', isName, 'a parameter or action name', undefined),
    value: opt(f, 'value', isSetValue, 'a number, a boolean or bounded text', undefined),
    message: opt(f, 'message', isText, `a string of at most ${LIMITS_31.textMax} bytes`, undefined),
    error,
    result,
  };
}

export function decodeSelection(f: Frame): SelectionMsg {
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
}

export function encodeControls(msg: ControlsMsg, minor: number): Uint8Array {
  const id = MSG.controls;
  if (msg.params.some((p) => p.type === 'text' && (!isParamText(p.value) || (p.pending != null && !isParamText(p.pending))))) bad(id, 'invalid text parameter');
  if (msg.interventions != null && !isInterventions(msg.interventions)) bad(id, 'invalid interventions metadata');
  return encodeWithSource(msg, id, drop({ run: msg.run, params: msg.params, actions: msg.actions, interventions: msg.interventions }),
    msg.stamp ? [stampSection('stamp', msg.stamp, [msg.stamp.length])] : [], minor);
}

export function encodeControl(msg: ControlMsg, minor: number): Uint8Array {
  const id = MSG.control;
  if (msg.op === 'set' && !isSetValue(msg.value)) bad(id, 'invalid parameter value');
  if (msg.op === 'intervene' && !isInterveneRequest(msg.intervene)) bad(id, 'invalid intervention request');
  if (msg.op === 'interpret' && !isInterpretationRequest(msg.interpretation)) bad(id, 'invalid interpretation request');
  if (msg.op !== 'interpret' && msg.interpretation !== undefined) bad(id, 'interpretation requires op interpret');
  return encodeWithSource(msg, id, drop({
    request_id: msg.request_id, op: msg.op, name: msg.name, value: msg.value, count: msg.count, intervene: msg.intervene, interpretation: msg.interpretation,
  }), [], minor);
}

export function encodeControlReply(msg: ControlReplyMsg, minor: number): Uint8Array {
  const id = MSG.control_reply;
  if (msg.value != null && !isSetValue(msg.value)) bad(id, 'invalid parameter value');
  if (msg.result !== undefined && (msg.op !== 'interpret' || msg.status !== 'ok')) bad(id, 'result requires a successful interpretation reply');
  if (msg.op === 'interpret' && msg.status === 'ok' && !isInterpretationObject(msg.result)) bad(id, 'invalid interpretation result');
  return encodeWithSource(msg, id, drop({
    request_id: msg.request_id, op: msg.op, status: msg.status, name: msg.name, value: msg.value, message: msg.message, error: msg.error, result: msg.result,
  }), [], minor);
}

export function encodeSelection(msg: SelectionMsg, minor: number): Uint8Array {
  return encodeWithSource(msg, MSG.selection, drop({
    space: msg.space, items: msg.items, view: msg.view, truncated: msg.truncated ? true : undefined, details: msg.details,
    details_error: msg.details_error,
  }), [], minor);
}
