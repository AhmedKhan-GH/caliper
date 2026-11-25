// The canonical JSON form of a spec (design/spec-format.md, Canonical JSON),
// as far as the spec page reads it, and the result of the panel.data command
// (src/builder/include/caliper_builder/session/panel_data.h).

import type { SpecColormapName } from '../render/colormaps.gen.js';

export type Primitive = 'series' | 'heatmap' | 'points' | 'histogram' | 'table' | 'image' | 'text' | 'scene';

/**
 * A layout node: a panel or control id, a leaf table { panel, min } (spec
 * version 1, min in characters and lines of the UI face), a split or tabs.
 */
export type LayoutNode = string | LeafNode | SplitNode | TabsNode;
export interface LeafNode { panel: string; min?: [number, number] }
export interface SplitNode { split: 'horizontal' | 'vertical'; ratio?: number; shares?: number[]; children: LayoutNode[]; title?: string }
export interface TabsNode { tabs: LayoutNode[]; title?: string }

/** The kinds of control (design/instruments.md, Control Table). */
export type ControlKind = 'slider' | 'button' | 'range' | 'toggle' | 'picker' | 'text' | 'segmented' | 'readout' | 'transport';

export interface SpecControl {
  id: string;
  kind: ControlKind;
  title?: string;
  bind?: string;
  options?: Record<string, unknown>;
}

export interface SpecSource { id: string; catalog?: string; sql?: string; from?: string }

export interface SpecOptions {
  x_label?: string; y_label?: string; z_label?: string;
  x_range?: [number, number]; y_range?: [number, number]; range?: [number, number];
  x_log?: boolean; y_log?: boolean;
  colormap?: SpecColormapName;
  bins?: number; columns?: string[]; text?: string;
}

export interface SpecPanel {
  id: string;
  primitive: Primitive;
  title?: string;
  bind?: Record<string, string | string[]>;
  options?: SpecOptions;
}

export interface Spec {
  caliper_spec: number;
  applet: { id: string; title?: string; description?: string };
  source?: SpecSource[];
  panel?: SpecPanel[];
  control?: SpecControl[];
  layout?: LayoutNode;
}

/** {"dtype", "shape", "base64"}: little-endian numbers. */
export interface WireArray { dtype: 'f32' | 'f64'; shape: number[]; base64: string }

export interface SeriesLineData {
  /** The binding, and the name the line draws under, as the app names it (spec_prepare.cpp, slot_name). */
  binding: string; name: string; row: number | null; count: number; last: number | null; x: WireArray; y: WireArray;
}
export interface SeriesData { x_index: boolean; x_name: string; steps: number; lines: SeriesLineData[] }
export interface HeatmapData {
  source_rows: number; source_cols: number; stride_rows: number; stride_cols: number; values: WireArray;
}
export interface ImageData {
  source_rows: number; source_cols: number; stride: number; channels: number; float: boolean; values: WireArray;
}
export interface PointsData {
  three_d: boolean; total: number; stride: number; x: WireArray; y: WireArray; z: WireArray | null;
  color: WireArray | null; color_integer: boolean; size: WireArray | null;
}
export interface HistogramData { lo: number; hi: number; total: number; outside: number; counts: number[] }
export type Cell = number | string | boolean | null;
export interface TableData {
  columns: { name: string; type: string }[]; rows: Cell[][]; total_rows: number; truncated: boolean;
}
export interface TextData { value: Cell }
/** A scene panel: its live scene's name, and the instances of pieces it draws (programs.ts, ViewerProgram). */
export interface SceneData { scene: string | null; programs?: import('./programs.js').ViewerProgram[] }

export interface PanelDataResult {
  panel: string;
  primitive: string;
  version: string;
  unchanged: boolean;
  error: { code: string; message: string; where: string } | null;
  state: { kind: 'empty' | 'mismatch'; detail: string } | null;
  bounds: string[];
  data: SeriesData | HeatmapData | ImageData | PointsData | HistogramData | TableData | TextData | SceneData | null;
}

/** Decodes a wire array into a Float32Array or Float64Array. */
export function decodeArray(a: WireArray): Float32Array | Float64Array {
  const bin = atob(a.base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return a.dtype === 'f64' ? new Float64Array(bytes.buffer, 0, bytes.length >> 3) : new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}
