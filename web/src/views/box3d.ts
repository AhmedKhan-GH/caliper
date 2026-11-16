// The box around 3-D points, an ImPlot3D-style frame that makes a still image
// read in depth: the box's twelve edges, tick marks and labels on one
// silhouette edge of each axis with the axis's name, and a faint grid on the
// floor, the face across the up axis that lies behind the data. The back
// layer (the edges between two faces turned away from the camera, and the
// grid) is drawn under the points and the front layer (every other edge, the
// ticks and the labels) over them: everything in the box is in front of its
// back faces and behind its front faces, so the order gives the depth without
// a depth buffer. The native points view draws the same frame
// (src/host/live/live_views_draw.cpp, box_frame).

import type { Mat4 } from './camera.js';
import { fmtTick, ticks } from './plot.js';
import { C, FONT_MONO } from './theme.js';

type V3 = [number, number, number];
interface Clip { x: number; y: number; z: number; w: number }
export interface Area { x: number; y: number; w: number; h: number }

/** The inks of the frame: the rule for edges and ticks, fainter for the back edges and the grid. */
export const BOX_INK = {
  edge: C.rule,
  back: 'rgba(90, 87, 81, 0.59)',
  grid: 'rgba(90, 87, 81, 0.27)',
  text: C.faint,
} as const;
const FONT = `10.5px ${FONT_MONO}`;
const TEXT_H = 13;

function clipOf(m: Mat4, p: V3): Clip {
  return {
    x: m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    y: m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    z: m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    w: m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
  };
}

function screenOf(c: Clip, a: Area): { x: number; y: number } {
  return { x: a.x + ((c.x / c.w) * 0.5 + 0.5) * a.w, y: a.y + (0.5 - (c.y / c.w) * 0.5) * a.h };
}

/** A segment's part in front of the near plane (clip z >= 0), on screen, or null. */
function clipSegment(p: Clip, q: Clip, a: Area): [{ x: number; y: number }, { x: number; y: number }] | null {
  if (p.z < 0 && q.z < 0) return null;
  const lerp = (u: Clip, v: Clip, t: number): Clip => ({
    x: u.x + (v.x - u.x) * t, y: u.y + (v.y - u.y) * t, z: u.z + (v.z - u.z) * t, w: u.w + (v.w - u.w) * t,
  });
  if (p.z < 0) p = lerp(p, q, p.z / (p.z - q.z));
  else if (q.z < 0) q = lerp(q, p, q.z / (q.z - p.z));
  if (!(p.w > 1e-9) || !(q.w > 1e-9)) return null;
  return [screenOf(p, a), screenOf(q, a)];
}

function cornerIndex(ax: number, at: number, a1: number, u: number, a2: number, w: number): number {
  const bits = [0, 0, 0];
  bits[ax] = at;
  bits[a1] = u;
  bits[a2] = w;
  return bits[0] | (bits[1] << 1) | (bits[2] << 2);
}

function boxPoint(b: number[], ax: number, t: number, a1: number, u: number, a2: number, w: number): V3 {
  const c: V3 = [0, 0, 0];
  c[ax] = t;
  c[a1] = b[2 * a1 + u];
  c[a2] = b[2 * a2 + w];
  return c;
}

export class BoxFrame {
  private readonly corner: Clip[] = [];
  /** front[axis][side]: the face at the low (0) or high (1) end of the axis faces the camera. */
  private readonly front: boolean[][] = [];
  private readonly labelEdge: number[] = [-1, -1, -1];
  private readonly ticks: { values: number[]; step: number }[] = [];
  private readonly floorSide: number;
  private readonly centre: { x: number; y: number };

  /** b: the box [x0, x1, y0, y1, z0, z1]; eye: the camera; m: the view-projection; a: the area in CSS pixels. */
  constructor(private readonly b: number[], eye: V3, private readonly m: Mat4, private readonly a: Area) {
    for (let i = 0; i < 8; i++) this.corner.push(clipOf(m, [b[i & 1 ? 1 : 0], b[i & 2 ? 3 : 2], b[i & 4 ? 5 : 4]]));
    for (let ax = 0; ax < 3; ax++) this.front.push([eye[ax] < b[2 * ax], eye[ax] > b[2 * ax + 1]]);
    // The floor: the low face across the up axis, or the high one when the
    // camera is below the box, whichever is behind the data.
    this.floorSide = eye[1] < b[2] ? 1 : 0;
    const cc = clipOf(m, [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]);
    this.centre = cc.w > 1e-9 ? screenOf(cc, a) : { x: a.x + a.w / 2, y: a.y + a.h / 2 };
    for (let ax = 0; ax < 3; ax++) {
      const a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
      let best = 0, len = 0;
      for (let k = 0; k < 4; k++) {
        const u = k & 1, w = k >> 1;
        if (Number(this.front[a1][u]) + Number(this.front[a2][w]) !== 1) continue;   // a silhouette edge
        const A = this.corner[cornerIndex(ax, 0, a1, u, a2, w)], B = this.corner[cornerIndex(ax, 1, a1, u, a2, w)];
        if (A.z < 0 || B.z < 0 || !(A.w > 1e-9) || !(B.w > 1e-9)) continue;
        const sa = screenOf(A, a), sb = screenOf(B, a);
        // x and z label their lowest silhouette edge on screen, y its leftmost.
        const key = ax === 1 ? -(sa.x + sb.x) : sa.y + sb.y;
        if (this.labelEdge[ax] < 0 || key > best) {
          best = key;
          this.labelEdge[ax] = k;
          len = Math.hypot(sb.x - sa.x, sb.y - sa.y);
        }
      }
      if (this.labelEdge[ax] >= 0 && len < 24) this.labelEdge[ax] = -1;   // too short to label
      // About one tick every 50 px along the edge.
      const n = this.labelEdge[ax] >= 0 ? Math.max(3, Math.min(8, Math.floor(len / 50))) : 4;
      this.ticks.push(ticks(b[2 * ax], b[2 * ax + 1], n));
    }
  }

  private stroke(ctx: CanvasRenderingContext2D, p: Clip, q: Clip): void {
    const s = clipSegment(p, q, this.a);
    if (!s) return;
    ctx.moveTo(s[0].x, s[0].y);
    ctx.lineTo(s[1].x, s[1].y);
  }

  /** The grid on the floor and the back edges, in CSS pixel units. */
  drawBack(ctx: CanvasRenderingContext2D): void {
    const b = this.b, f = this.floorSide;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = BOX_INK.grid;
    ctx.beginPath();
    for (const ax of [0, 2]) {
      const other = ax === 0 ? 2 : 0;
      for (const t of this.ticks[ax].values)
        this.stroke(ctx, clipOf(this.m, boxPoint(b, ax, t, 1, f, other, 0)), clipOf(this.m, boxPoint(b, ax, t, 1, f, other, 1)));
    }
    ctx.stroke();
    ctx.strokeStyle = BOX_INK.back;
    ctx.beginPath();
    this.edges(ctx, false);
    ctx.stroke();
    ctx.restore();
  }

  private edges(ctx: CanvasRenderingContext2D, front: boolean): void {
    for (let ax = 0; ax < 3; ax++) {
      const a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
      for (let k = 0; k < 4; k++) {
        const u = k & 1, w = k >> 1;
        if ((this.front[a1][u] || this.front[a2][w]) !== front) continue;
        this.stroke(ctx, this.corner[cornerIndex(ax, 0, a1, u, a2, w)], this.corner[cornerIndex(ax, 1, a1, u, a2, w)]);
      }
    }
  }

  /**
   * The front edges, the ticks and their labels pointing away from the box's
   * centre, and the axes' names beyond them; a label that would overlap the
   * one before it is left out. Returns the labels drawn, as "x=-4|0|4;y=...".
   */
  drawFront(ctx: CanvasRenderingContext2D): string {
    const b = this.b, a = this.a;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = BOX_INK.edge;
    ctx.beginPath();
    this.edges(ctx, true);
    ctx.stroke();
    ctx.font = FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = BOX_INK.text;
    const out: string[] = [];
    for (let ax = 0; ax < 3; ax++) {
      const k = this.labelEdge[ax];
      if (k < 0) continue;
      const a1 = (ax + 1) % 3, a2 = (ax + 2) % 3, u = k & 1, w = k >> 1;
      const ea = screenOf(this.corner[cornerIndex(ax, 0, a1, u, a2, w)], a);
      const eb = screenOf(this.corner[cornerIndex(ax, 1, a1, u, a2, w)], a);
      const mid = { x: (ea.x + eb.x) / 2, y: (ea.y + eb.y) / 2 };
      let dx = mid.x - this.centre.x, dy = mid.y - this.centre.y;
      const dl = Math.hypot(dx, dy);
      if (dl > 1e-3) {
        dx /= dl;
        dy /= dl;
      } else [dx, dy] = [0, 1];
      const alx = 0.5 - 0.5 * dx, aly = 0.5 - 0.5 * dy;
      let reach = 0;
      let last = { x0: 1e30, y0: 1e30, x1: -1e30, y1: -1e30 };
      const drawn: string[] = [];
      ctx.beginPath();
      for (const t of this.ticks[ax].values) {
        const c = clipOf(this.m, boxPoint(b, ax, t, a1, u, a2, w));
        if (!(c.w > 1e-9) || c.z < 0) continue;
        const s = screenOf(c, a);
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x + dx * 4, s.y + dy * 4);
        const text = fmtTick(t, this.ticks[ax].step);
        const tw = ctx.measureText(text).width;
        const x0 = s.x + dx * 7 - tw * alx, y0 = s.y + dy * 7 - TEXT_H * aly;
        const x1 = x0 + tw, y1 = y0 + TEXT_H;
        if (x0 < last.x1 + 3 && x1 > last.x0 - 3 && y0 < last.y1 && y1 > last.y0) continue;
        ctx.fillText(text, x0, y0);
        drawn.push(text);
        last = { x0, y0, x1, y1 };
        reach = Math.max(reach, Math.abs(dx) * tw + Math.abs(dy) * TEXT_H);
      }
      ctx.stroke();
      if (drawn.length) out.push(`${'xyz'[ax]}=${drawn.join('|')}`);
      const name = 'xyz'[ax];
      const off = 7 + reach + 8;
      const nw = ctx.measureText(name).width;
      ctx.fillText(name, mid.x + dx * off - nw * alx, mid.y + dy * off - TEXT_H * aly);
    }
    ctx.restore();
    return out.join(';');
  }
}
