// The derive half of depth@1: one element of `pixels` for each pixel of `depth`, row by row, the
// value of an f32 or a u16 depth map times `scale` in scene units, as the built-in draw holds a
// depth image. `info` holds the
// width, the height, the pose and the range of the data: the least and the greatest depth that is
// finite and not zero, as the producer's stats give it, and 1 in z when such a depth exists. A
// depth map larger than `pixels` publishes no pixels and a size of 0.

const THREADS: u32 = 65536u;   // 256 workgroups of 256, one span of the range each

fn depth_dims() -> vec2<u32> {
    let s = shapes.depth;
    let d = vec2<u32>(s.w, s.z);
    if (d.x == 0u || d.y == 0u || d.y > arrayLength(&pixels) / d.x) {
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

// Whether a depth counts toward the range: finite and not zero, decided on its bits.
fn in_range(v: f32) -> bool {
    let b = bitcast<u32>(v);
    return (b & 0x7f800000u) != 0x7f800000u && (b & 0x7fffffffu) != 0u;
}

@compute @workgroup_size(256)
fn copy_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = depth_dims();
    let n = d.x * d.y;
    if (id.x == 0u) {
        atomicStore(&pixels_count, n);
    }
    var lo = 0.0;
    var hi = 0.0;
    var seen = 0.0;
    for (var i = id.x; i < n; i = i + THREADS) {
        let v = depth_at(i) * params.scale;
        pixels[i] = DepthPixel(v);
        if (in_range(v)) {
            if (seen == 0.0) {
                lo = v;
                hi = v;
                seen = 1.0;
            } else {
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
    }
    if (id.x < THREADS) {
        spans[id.x] = vec4<f32>(lo, hi, seen, 0.0);
    }
}

@compute @workgroup_size(1)
fn place_depth() {
    var lo = 0.0;
    var hi = 0.0;
    var seen = 0.0;
    for (var k = 0u; k < THREADS; k = k + 1u) {
        let s = spans[k];
        if (s.z == 0.0) {
            continue;
        }
        if (seen == 0.0) {
            lo = s.x;
            hi = s.y;
            seen = 1.0;
        } else {
            lo = min(lo, s.x);
            hi = max(hi, s.y);
        }
    }
    let d = depth_dims();
    info[0] = DepthInfo(vec4<u32>(d, 1u, 0u), pose_column(0u), pose_column(1u), pose_column(2u), pose_column(3u),
                        vec4<f32>(lo, hi, seen, 0.0));
}
