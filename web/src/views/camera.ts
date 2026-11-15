// The orbit camera of the 3-D points view, with the same conventions as the
// native points view (right-handed lookAt, perspective with z in [0, 1]).

export type Mat4 = Float32Array;

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 1];
};

/** Column-major product a * b. */
export function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

export function lookAt(eye: V3, at: V3, up: V3): Mat4 {
  const f = norm(sub(at, eye)), s = norm(cross(f, up)), u = cross(s, f);
  return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1]);
}

export function perspective(fovy: number, aspect: number, zn: number, zf: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = zf / (zn - zf);
  m[11] = -1;
  m[14] = (zn * zf) / (zn - zf);
  return m;
}

/** Maps the window [x0, x1] x [y0, y1] onto clip space (y up, z = 0.5). */
export function ortho(x0: number, x1: number, y0: number, y1: number): Mat4 {
  const m = new Float32Array(16);
  m[0] = 2 / (x1 - x0);
  m[5] = 2 / (y1 - y0);
  m[10] = 0;
  m[12] = -(x1 + x0) / (x1 - x0);
  m[13] = -(y1 + y0) / (y1 - y0);
  m[14] = 0.5;
  m[15] = 1;
  return m;
}

export const FOV = (45 * Math.PI) / 180;

export class Orbit {
  az = 0.7;
  el = 0.35;
  dist = 10;
  target: V3 = [0, 0, 0];
  radius = 1;

  fit(bbox: number[]): void {
    this.target = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, (bbox[4] + bbox[5]) / 2];
    this.radius = Math.max(1e-6, Math.hypot(bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]) / 2);
    this.dist = (this.radius / Math.sin(FOV / 2)) * 0.85;
    this.az = 0.7;
    this.el = 0.35;
  }

  eye(): V3 {
    const ce = Math.cos(this.el), se = Math.sin(this.el), ca = Math.cos(this.az), sa = Math.sin(this.az);
    return [this.target[0] + this.dist * ce * ca, this.target[1] + this.dist * se, this.target[2] + this.dist * ce * sa];
  }

  matrix(aspect: number): Mat4 {
    const zn = Math.max(this.dist - this.radius * 2, this.dist * 0.01);
    const zf = this.dist + this.radius * 2;
    return mul(perspective(FOV, aspect, zn, zf), lookAt(this.eye(), this.target, [0, 1, 0]));
  }

  rotate(dx: number, dy: number): void {
    this.az += dx * 0.01;
    this.el = Math.max(-1.5, Math.min(1.5, this.el + dy * 0.01));
  }

  /** Moves the target in the view plane by a drag of (dx, dy) pixels over a viewport h pixels high. */
  pan(dx: number, dy: number, h: number): void {
    const eye = this.eye();
    const f = norm(sub(this.target, eye)), s = norm(cross(f, [0, 1, 0])), u = cross(s, f);
    const k = (2 * this.dist * Math.tan(FOV / 2)) / Math.max(1, h);
    for (let i = 0; i < 3; i++) this.target[i] += (-s[i] * dx + u[i] * dy) * k;
  }

  zoom(factor: number): void {
    this.dist = Math.max(this.radius * 0.05, this.dist / factor);
  }

  /** Unit vectors of the world axes in view space, for the axis triad. */
  axes(): V3[] {
    const eye = this.eye();
    const f = norm(sub(this.target, eye)), s = norm(cross(f, [0, 1, 0])), u = cross(s, f);
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((a) => [dot(a as V3, s), dot(a as V3, u), dot(a as V3, f)] as V3);
  }
}
