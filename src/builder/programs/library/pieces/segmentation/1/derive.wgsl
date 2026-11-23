// The derive half of segmentation@1: one element of `pixels` for each pixel of `classes`, row by
// row, the class id clamped to 0 to 65,535 as the producer clamps it. `palette` holds the color of
// classes 0 to 65,535: row k of `colors` for class k, each channel trunc(clamp(c, 0, 1) * 255 + 0.5)
// and alpha 255 without a fourth column, and the built-in palette's color k % 16 for a class
// without a row or with a red below 0. `info` holds the width, the height and the pose. A mask
// larger than `pixels` publishes no pixels and a size of 0.

const THREADS: u32 = 65536u;   // 256 workgroups of 256

fn mask_dims() -> vec2<u32> {
    let s = shapes.classes;
    let d = vec2<u32>(s.w, s.z);
    if (d.x == 0u || d.y == 0u || d.x > 8192u || d.y > 8192u || d.y > arrayLength(&pixels) / d.x) {
        return vec2<u32>(0u, 0u);
    }
    return d;
}

// Column c of the pose, a (4, 4) or (3, 4) tensor row by row, or of the identity for another shape.
fn pose_column(c: u32) -> vec4<f32> {
    let s = shapes.pose;
    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (s.w != 4u || (s.z != 3u && s.z != 4u)) {
        col[c] = 1.0;
        return col;
    }
    col = vec4<f32>(pose[c], pose[4u + c], pose[8u + c], 0.0);
    if (s.z == 4u) {
        col.w = pose[12u + c];
    } else if (c == 3u) {
        col.w = 1.0;
    }
    return col;
}

fn channel(v: f32) -> u32 {
    if (!(v > 0.0)) {
        return 0u;
    }
    if (v >= 1.0) {
        return 255u;
    }
    return min(u32(v * 255.0 + 0.5), 255u);
}

@compute @workgroup_size(256)
fn copy_classes(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = mask_dims();
    let n = d.x * d.y;
    if (id.x == 0u) {
        pixels_size = d;
    }
    for (var i = id.x; i < n; i = i + THREADS) {
        pixels[i] = u32(clamp(classes_at(i), 0, 65535));
    }
}

@compute @workgroup_size(256)
fn fill_palette(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if (k >= arrayLength(&palette)) {
        return;
    }
    var rgba = palette_color(k);
    let s = shapes.colors;
    let cols = s.w;
    if (k < s.z && (cols == 3u || cols == 4u)) {
        let b = k * cols;
        let r = colors[b];
        if (!(r < 0.0)) {
            rgba = channel(r) | (channel(colors[b + 1u]) << 8u) | (channel(colors[b + 2u]) << 16u) | 0xff000000u;
            if (cols == 4u) {
                rgba = (rgba & 0x00ffffffu) | (channel(colors[b + 3u]) << 24u);
            }
        }
    }
    palette[k] = ClassColor(rgba);
}

@compute @workgroup_size(1)
fn place_mask() {
    let d = mask_dims();
    info[0] = MaskInfo(vec4<u32>(d, 1u, 0u), pose_column(0u), pose_column(1u), pose_column(2u), pose_column(3u));
}
