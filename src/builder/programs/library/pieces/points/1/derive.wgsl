// The derive half of points@1: one element of `points` for each row of `positions`, with the
// row's radius in position.w (the one radius for all, 0 where `radii` is shorter, 1 without
// radii), the bits of the row's value (NaN where `values` is shorter), and the count of rows. The
// bits are those of an f32 value, a u32 packed RGBA, the packed bytes of a u8 color row (alpha 255
// for 3 channels), or the class color of an integer class id, which color mode 2 draws as a packed
// RGBA.

// The class colors of the scene where no class is described: id k takes entry k mod 16. An id
// outside 0 to 65,535 takes the scene's default color.
const POINT_DEFAULT: u32 = 0xffe6dcdcu;

fn class_rgba(v: f32) -> u32 {
    if (!(v >= 0.0) || v >= 65536.0) {
        return POINT_DEFAULT;
    }
    var palette = array<u32, 16>(0xffb4771fu, 0xff0e7fffu, 0xff2ca02cu, 0xff2827d6u, 0xffbd6794u, 0xff4b568cu,
                                 0xffc277e3u, 0xff7f7f7fu, 0xff22bdbcu, 0xffcfbe17u, 0xffe8c7aeu, 0xff78bbffu,
                                 0xff8adf98u, 0xff9698ffu, 0xffd5b0c5u, 0xff949cc4u);
    return palette[u32(v) % 16u];
}

// The bits of point i, or NaN where `values` has no element for it. A u8 tensor of rank 2 with 3 or 4
// channels holds a color row for each point; any other tensor holds one element for each point.
fn value_bits(i: u32) -> u32 {
    let s = shapes.values;
    if (values_type == CALIPER_U8 && s.z > 1u) {
        if (i >= s.z || (s.w != 3u && s.w != 4u)) {
            return 0x7fc00000u;
        }
        let b = i * s.w;
        var a = 255u;
        if (s.w == 4u) {
            a = u32(values_at(b + 3u));
        }
        return u32(values_at(b)) | (u32(values_at(b + 1u)) << 8u) | (u32(values_at(b + 2u)) << 16u) | (a << 24u);
    }
    if (s.z != 1u || i >= s.w) {
        return 0x7fc00000u;
    }
    if (values_type == CALIPER_F32 || values_type == CALIPER_U32) {
        return values[i];   // the bits as they are: a value, or a packed RGBA
    }
    return class_rgba(values_at(i));
}

@compute @workgroup_size(256)
fn copy_points(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = min(shapes.positions.z, arrayLength(&points));
    if (id.x == 0u) {
        atomicStore(&points_count, n);
    }
    if (id.x >= n) {
        return;
    }
    let d = shapes.positions.w;
    let base = id.x * d;
    var p = vec4<f32>(positions[base], 0.0, 0.0, 1.0);
    if (d > 1u) {
        p.y = positions[base + 1u];
    }
    if (d > 2u) {
        p.z = positions[base + 2u];
    }
    let r = shapes.radii.w;
    if (r == 1u) {
        p.w = radii[0];
    } else if (r > 0u) {
        p.w = select(0.0, radii[min(id.x, r - 1u)], id.x < r);
    }
    points[id.x] = Point(p, value_bits(id.x), id.x);
}
