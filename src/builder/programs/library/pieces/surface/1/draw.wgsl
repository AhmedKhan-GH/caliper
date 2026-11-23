// The draw half of surface@1, as the built-in surface draw kind draws its grid through the scene's
// mesh: the element (r, c) of `cells` is the vertex of row r and column c, placed in the data box
// from its data units; each cell whose four heights are numbers is two triangles, (r, c),
// (r + 1, c), (r, c + 1) and (r, c + 1), (r + 1, c), (r + 1, c + 1), and any other cell is a hole.
// A vertex's normal comes from its neighbors in the box with finite heights, its color is the host
// colormap's texel of the level that the host's reference gives its value, (v - vmin) /
// (vmax - vmin) * 255 rounded with NaN at level 0, lit per vertex by the headlight, 0.3 + 0.7 times
// the normal's z in view space. A pick names the first vertex of the triangle, as the mesh's flat
// vertex index does.
//
// The three compute passes find the range of x, y, z and value over the finite numbers of the
// valid elements, for the ranges that `fit` takes from the data, without workgroup memory: each
// of 4,096 threads keeps the smallest and largest of its elements in fit[0, 4096) and
// fit[4096, 8192), each of 64 threads those of 64 of them in fit[8192, 8256) and fit[8256, 8320),
// and one thread those of all in fit[8320] and fit[8321].

const BIG: f32 = 3.0e38;
const THREADS: u32 = 4096u;
const GROUPS: u32 = 64u;
const TOTAL: u32 = 8320u;

fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn is_nan(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

fn valid_cells() -> u32 {
    return min(counts.cells, arrayLength(&cells));
}

@compute @workgroup_size(256)
fn fit_threads(@builtin(global_invocation_id) id: vec3<u32>) {
    var lo = vec4<f32>(BIG);
    var hi = vec4<f32>(-BIG);
    if (params.fit != 0u) {
        let n = valid_cells();
        for (var i = id.x; i < n; i = i + THREADS) {
            let e = cells[i];
            let v = vec4<f32>(e.x, e.y, e.z, e.value);
            let keep = vec4<bool>(finite(v.x), finite(v.y), finite(v.z), finite(v.w));
            lo = select(lo, min(lo, v), keep);
            hi = select(hi, max(hi, v), keep);
        }
    }
    fit[id.x] = lo;
    fit[THREADS + id.x] = hi;
}

@compute @workgroup_size(64)
fn fit_groups(@builtin(local_invocation_index) t: u32) {
    var lo = vec4<f32>(BIG);
    var hi = vec4<f32>(-BIG);
    for (var j = t; j < THREADS; j = j + GROUPS) {
        lo = min(lo, fit[j]);
        hi = max(hi, fit[THREADS + j]);
    }
    fit[2u * THREADS + t] = lo;
    fit[2u * THREADS + GROUPS + t] = hi;
}

@compute @workgroup_size(1)
fn fit_total() {
    var lo = vec4<f32>(BIG);
    var hi = vec4<f32>(-BIG);
    for (var j = 0u; j < GROUPS; j = j + 1u) {
        lo = min(lo, fit[2u * THREADS + j]);
        hi = max(hi, fit[2u * THREADS + GROUPS + j]);
    }
    fit[TOTAL] = lo;
    fit[TOTAL + 1u] = hi;
}

// The range of axis k (0 x, 1 y, 2 z, 3 value): from the data when its bit of `fit` is set, with
// no number giving 0 to 1 and one number n giving n - 0.5 to n + 0.5, else from the params.
fn range_of(k: u32, lo_param: f32, hi_param: f32) -> vec2<f32> {
    if ((params.fit & (select(4u << k, 3u, k == 3u))) == 0u) {
        return vec2<f32>(lo_param, hi_param);
    }
    var lo = fit[TOTAL][k];
    var hi = fit[TOTAL + 1u][k];
    if (lo > hi) {
        lo = 0.0;
        hi = 1.0;
    }
    if (lo == hi) {
        lo = lo - 0.5;
        hi = hi + 0.5;
    }
    return vec2<f32>(lo, hi);
}

// The value range of the colormap: each end from the data when its bit is set, else its param.
fn value_range() -> vec2<f32> {
    let data = range_of(3u, params.vmin, params.vmax);
    var r = vec2<f32>(params.vmin, params.vmax);
    if ((params.fit & 1u) != 0u) {
        r.x = data.x;
    }
    if ((params.fit & 2u) != 0u) {
        r.y = data.y;
    }
    return r;
}

// The data box: each axis's range and the half of its side, the longest side 2 units. With a
// box side above 0 the sides are box_x, box_y and box_z, else the spans of the ranges with no side
// below a quarter of the longest.
struct Box {
    lo: vec3<f32>,
    hi: vec3<f32>,
    half: vec3<f32>,
}

fn data_box() -> Box {
    var b: Box;
    let rx = range_of(0u, params.x_lo, params.x_hi);
    let ry = range_of(1u, params.y_lo, params.y_hi);
    let rz = range_of(2u, params.z_lo, params.z_hi);
    b.lo = vec3<f32>(rx.x, ry.x, rz.x);
    b.hi = vec3<f32>(rx.y, ry.y, rz.y);
    let option = params.box_x > 0.0 || params.box_y > 0.0 || params.box_z > 0.0;
    var side = b.hi - b.lo;
    if (option) {
        side = vec3<f32>(params.box_x, params.box_y, params.box_z);
    }
    var longest = max(side.x, max(side.y, side.z));
    if (!(longest > 0.0) || !finite(longest)) {
        longest = 1.0;
        side = vec3<f32>(1.0);
    }
    for (var a = 0u; a < 3u; a = a + 1u) {
        var s = side[a] / longest;
        if (!option) {
            s = max(s, 0.25);
        }
        if (!(s > 0.0) || !finite(s)) {
            s = 0.25;
        }
        b.half[a] = s;
    }
    return b;
}

// A data value on axis a in box units, 0 for an empty range.
fn to_box(b: Box, a: u32, v: f32) -> f32 {
    let span = b.hi[a] - b.lo[a];
    if (!(span > 0.0)) {
        return 0.0;
    }
    return ((v - b.lo[a]) / span * 2.0 - 1.0) * b.half[a];
}

// Whether vertex i has a height in the box, as the mesh's positions hold it: else it is a hole.
fn solid(b: Box, i: u32) -> bool {
    return finite(to_box(b, 2u, cells[i].z));
}

fn box_position(b: Box, i: u32) -> vec3<f32> {
    let e = cells[i];
    return vec3<f32>(to_box(b, 0u, e.x), to_box(b, 1u, e.y), to_box(b, 2u, e.z));
}

// The normal of vertex (r, c) of an R x C grid: the differences across its neighbors that have a
// height in the box, one-sided at an edge or a hole, flat along an axis with no neighbor, facing +z.
fn normal_at(b: Box, r: u32, c: u32, rows: u32, cols: u32) -> vec3<f32> {
    let i = r * cols + c;
    var r0 = r;
    var r1 = r;
    var c0 = c;
    var c1 = c;
    if (r > 0u && solid(b, i - cols)) {
        r0 = r - 1u;
    }
    if (r + 1u < rows && solid(b, i + cols)) {
        r1 = r + 1u;
    }
    if (c > 0u && solid(b, i - 1u)) {
        c0 = c - 1u;
    }
    if (c + 1u < cols && solid(b, i + 1u)) {
        c1 = c + 1u;
    }
    var tr = box_position(b, r1 * cols + c) - box_position(b, r0 * cols + c);
    var tc = box_position(b, r * cols + c1) - box_position(b, r * cols + c0);
    if (r0 == r1) {
        tr.x = 1.0;
    }
    if (c0 == c1) {
        tc.y = 1.0;
    }
    var n = vec3<f32>(tr.y * tc.z - tr.z * tc.y, tr.z * tc.x - tr.x * tc.z, tr.x * tc.y - tr.y * tc.x);
    if (n.z < 0.0) {
        n = -n;
    }
    let l = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (!(l > 0.0) || !finite(l)) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return n / l;
}

// The colormap level of a value, 0 to 255, as the host's reference computes it.
fn level(v: f32, range: vec2<f32>) -> u32 {
    if (is_nan(v)) {
        return 0u;
    }
    let span = range.y - range.x;
    var t = 0.0;
    if (span != 0.0) {
        t = (v - range.x) / span;
    }
    if (!(t >= 0.0)) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }
    return min(u32(t * 255.0 + 0.5), 255u);
}

struct CellOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

fn hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

@vertex
fn cell_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> CellOut {
    var out: CellOut;
    out.position = hidden();
    out.color = vec4<f32>(0.0);
    out.item = i;
    let n = valid_cells();
    if (i >= n) {
        return out;
    }
    let cols = cells[i].cols;
    if (cols == 0u) {
        return out;
    }
    let rows = n / cols;
    let r = i / cols;
    let c = i % cols;
    if (r + 1u >= rows || c + 1u >= cols) {
        return out;
    }
    // The cell's corners a (r, c), b (r, c + 1), d (r + 1, c) and e (r + 1, c + 1), as the mesh's
    // triangles (a, d, b) and (b, d, e) name them.
    let a = i;
    let bi = i + 1u;
    let d = i + cols;
    let e = d + 1u;
    let b = data_box();
    if (!solid(b, a) || !solid(b, bi) || !solid(b, d) || !solid(b, e)) {
        return out;
    }
    var corners = array<u32, 6>(a, d, bi, bi, d, e);
    let k = corners[min(v, 5u)];
    out.item = select(bi, a, v < 3u);
    let p = box_position(b, k);
    if (is_nan(p.x) || is_nan(p.y) || is_nan(p.z)) {
        return out;
    }
    var clip = caliper_view.clip_from_world * vec4<f32>(p, 1.0);
    if (params.depth_offset != 0) {
        clip.z = clip.z - f32(params.depth_offset) * (1.0 / 8192.0) * clip.w;
    }
    out.position = clip;

    let size = textureDimensions(caliper_colormaps);
    let x = min(level(cells[k].value, value_range()) * (size.x - 1u) / 255u, size.x - 1u);
    let texel = textureLoad(caliper_colormaps, vec2<u32>(x, min(params.colormap, size.y - 1u)), 0);
    let m = normalize(normal_at(b, k / cols, k % cols, rows, cols));
    let vm = caliper_view.view_from_world;
    let view = normalize(m.x * vm[0].xyz + m.y * vm[1].xyz + m.z * vm[2].xyz);
    let lit = 0.30 + 0.70 * max(dot(view, vec3<f32>(0.0, 0.0, 1.0)), 0.0);
    out.color = vec4<f32>(texel.rgb * lit, texel.a);
    return out;
}

@fragment
fn cell_fs(in: CellOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}
