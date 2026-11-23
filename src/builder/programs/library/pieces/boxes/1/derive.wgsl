// The derive half of boxes@1: one element of `boxes` for each row of `centers`, with its half
// sizes, its rotation, the colors that the two per-box color modes read, its index, and the
// count. Every box holds its dimensions, 2 or 3, in center.w, and box 0 also holds the flags of
// all boxes in half_size.w: bit 0 is set when the RGBA of a box has an alpha below 255. Every box
// with such an alpha writes 1 into the private resource alpha_seen, which is zero before each run,
// and mark_alpha copies it into box 0. A tensor with one row gives that row to every box, as a
// stream of stride 0 does in the built-in.

// The class colors of the scene where no class is described: id k takes entry k mod 16. An id
// outside 0 to 65,535 takes the scene's default color.
const BOX_DEFAULT: u32 = 0xffe6dcdcu;

// The boxes: the rows of `centers`, and no more than the rows of `half_sizes` unless it has one.
fn box_count() -> u32 {
    var n = min(shapes.centers.z, arrayLength(&boxes));
    let rows = shapes.half_sizes.z;
    if (rows != 1u) {
        n = min(n, rows);
    }
    return n;
}

// The dimensions of the boxes: 3 for centers of 3 or more columns, else 2.
fn box_dims() -> u32 {
    return select(3u, 2u, shapes.centers.w < 3u);
}

// A channel from 0 to 1 as a byte, as the scene's Python side converts one: clamped, times 255,
// plus 0.5, truncated. NaN gives 0.
fn box_byte(v: f32) -> u32 {
    if (!(v > 0.0)) {
        return 0u;
    }
    return u32(min(v, 1.0) * 255.0 + 0.5);
}

// The RGBA of box i from the rows of `colors` (3 or 4 channels), packed r | g << 8 | b << 16 |
// a << 24, with an alpha of 255 for 3 channels. A box without a row takes the one color.
fn box_rgba(i: u32) -> u32 {
    let rows = shapes.colors.z;
    let w = shapes.colors.w;
    if (w < 3u || (rows != 1u && i >= rows)) {
        return params.color;
    }
    let base = select(i, 0u, rows == 1u) * w;
    var a = 255u;
    if (w > 3u) {
        a = box_byte(colors[base + 3u]);
    }
    return box_byte(colors[base]) | (box_byte(colors[base + 1u]) << 8u) | (box_byte(colors[base + 2u]) << 16u) |
           (a << 24u);
}

// The class color of box i from element i of `colors`, a whole number from 0 to 65,535.
fn box_class(i: u32) -> u32 {
    let n = arrayLength(&colors);
    if (n == 0u || (n != 1u && i >= n)) {
        return BOX_DEFAULT;
    }
    let v = colors[select(i, 0u, n == 1u)];
    if (!(v >= 0.0) || v >= 65536.0) {
        return BOX_DEFAULT;
    }
    var palette = array<u32, 16>(0xffb4771fu, 0xff0e7fffu, 0xff2ca02cu, 0xff2827d6u, 0xffbd6794u, 0xff4b568cu,
                                 0xffc277e3u, 0xff7f7f7fu, 0xff22bdbcu, 0xffcfbe17u, 0xffe8c7aeu, 0xff78bbffu,
                                 0xff8adf98u, 0xff9698ffu, 0xffd5b0c5u, 0xff949cc4u);
    return palette[u32(v) % 16u];
}

// The rotation of box i in 3-D, a quaternion (x, y, z, w); none in 2-D or without a row.
fn box_rotation(i: u32, dims: u32) -> vec4<f32> {
    let rows = shapes.rotations.z;
    if (dims != 3u || shapes.rotations.w != 4u || (rows != 1u && i >= rows)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let base = select(i, 0u, rows == 1u) * 4u;
    return vec4<f32>(rotations[base], rotations[base + 1u], rotations[base + 2u], rotations[base + 3u]);
}

@compute @workgroup_size(256)
fn copy_boxes(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = box_count();
    if (id.x == 0u) {
        atomicStore(&boxes_count, n);
    }
    if (id.x >= n) {
        return;
    }
    let dims = box_dims();
    let d = shapes.centers.w;
    let base = id.x * d;
    var c = vec4<f32>(centers[base], 0.0, 0.0, f32(dims));
    if (d > 1u) {
        c.y = centers[base + 1u];
    }
    if (dims == 3u) {
        c.z = centers[base + 2u];
    }
    let hd = shapes.half_sizes.w;
    let hb = select(id.x, 0u, shapes.half_sizes.z == 1u) * hd;
    var h = vec4<f32>(half_sizes[hb], 0.0, 0.0, 0.0);
    if (hd > 1u) {
        h.y = half_sizes[hb + 1u];
    }
    if (hd > 2u && dims == 3u) {
        h.z = half_sizes[hb + 2u];
    }
    let rgba = box_rgba(id.x);
    boxes[id.x] = Box(c, h, box_rotation(id.x, dims), rgba, box_class(id.x), id.x);
    if ((rgba >> 24u) < 255u) {
        alpha_seen[0] = 1u;
    }
}

// The flags of all boxes into box 0, after copy_boxes.
@compute @workgroup_size(1)
fn mark_alpha() {
    if (box_count() > 0u) {
        boxes[0].half_size.w = f32(alpha_seen[0] & 1u);
    }
}
