// The derive half of arrows@1: one element of `arrows` for each row of `origins`, with its vector
// and the color source of its row of `values`, and in `info` the range of the values and whether a
// packed RGBA has an alpha below 255. One value colors every arrow, and a value past the end of
// `values` is NaN. The range is that of the values as the producer's stats give it: a NaN
// anywhere, or no value, leaves it unknown.
//
// `scratch` holds the keys and flags that each invocation of scan_values saw at HI, LO and FLAGS.
// No pass uses workgroup memory, which the derive runtime leaves without storage.

const HI: u32 = 0u;
const LO: u32 = 4096u;
const FLAGS: u32 = 8192u;
const SCANS: u32 = 4096u;      // the invocations of scan_values, 16 workgroups of 256
const NAN_BITS: u32 = 0x7fc00000u;
const NONE: u32 = 0xffffffffu;

// Row i of an (N, D) tensor as a position: x, then y and z where D has them, else 0.
fn origin(i: u32) -> vec3<f32> {
    let d = shapes.origins.w;
    let base = i * d;
    var p = vec3<f32>(origins[base], 0.0, 0.0);
    if (d > 1u) {
        p.y = origins[base + 1u];
    }
    if (d > 2u) {
        p.z = origins[base + 2u];
    }
    return p;
}

fn vector(i: u32) -> vec3<f32> {
    let d = shapes.vectors.w;
    let base = i * d;
    var p = vec3<f32>(vectors[base], 0.0, 0.0);
    if (d > 1u) {
        p.y = vectors[base + 1u];
    }
    if (d > 2u) {
        p.z = vectors[base + 2u];
    }
    return p;
}

// The color source of row i of `values`: its bits, the one value for all, or NaN past the end.
fn source(i: u32) -> u32 {
    let n = shapes.values.w;
    if (n == 1u) {
        return bitcast<u32>(values[0]);
    }
    if (i < n) {
        return bitcast<u32>(values[i]);
    }
    return NAN_BITS;
}

@compute @workgroup_size(256)
fn build_arrows(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = min(shapes.origins.z, arrayLength(&arrows));
    if (id.x == 0u) {
        atomicStore(&arrows_count, count);
    }
    let i = id.x;
    if (i >= count) {
        return;
    }
    let o = origin(i);
    var v = vec3<f32>(0.0);
    var index = NONE;
    if (i < shapes.vectors.z) {
        v = vector(i);
        index = i;
    }
    arrows[i] = Arrow(o.x, o.y, o.z, source(i), v.x, v.y, v.z, index);
}

// Floats in their numeric order as keys above 0: -inf lowest, +inf highest.
fn key_of(bits: u32) -> u32 {
    return select(bits | 0x80000000u, ~bits, (bits >> 31u) != 0u);
}

fn float_of(key: u32) -> f32 {
    return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}

// Each invocation reduces its share of the values, every SCANS-th, into its words of `scratch`:
// the greatest key, the greatest inverted key (0 when it saw no value), and bit 0 for a NaN, bit 1
// for an alpha below 255.
@compute @workgroup_size(256)
fn scan_values(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCANS) {
        return;
    }
    let n = shapes.values.w;
    var top = 0u;
    var bottom = 0u;
    var bits = 0u;
    var i = id.x;
    while (i < n) {
        let v = bitcast<u32>(values[i]);
        if ((v >> 24u) < 255u) {
            bits = bits | 2u;
        }
        if ((v & 0x7fffffffu) > 0x7f800000u) {
            bits = bits | 1u;
        } else {
            let key = key_of(v);
            top = max(top, key);
            bottom = max(bottom, ~key);
        }
        if (SCANS > n - i) {
            break;
        }
        i = i + SCANS;
    }
    scratch[HI + id.x] = top;
    scratch[LO + id.x] = bottom;
    scratch[FLAGS + id.x] = bits;
}

@compute @workgroup_size(1)
fn finish_info() {
    var top = 0u;
    var bottom = 0u;
    var bits = 0u;
    for (var g = 0u; g < SCANS; g = g + 1u) {
        top = max(top, scratch[HI + g]);
        bottom = max(bottom, scratch[LO + g]);
        bits = bits | scratch[FLAGS + g];
    }
    if (top == 0u) {
        bits = bits | 1u;
    }
    info[0] = Info(float_of(~bottom), float_of(top), bits);
}
