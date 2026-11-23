// The derive half of lines@1: one element of `segments` for each instance of the built-in lines
// draw, in its order, with the positions and the color sources of both ends, and in `info` the
// range of the values and whether a packed RGBA has an alpha below 255.
//
// Segments join vertices 2k and 2k + 1. Strips join vertex i and i + 1 for i from 0 to V - 2, and
// a join at the last vertex of a strip is not drawn; the strips' lengths are summed in chunks
// first, so that each join finds its strip by a binary search of the ends. Edges join the two
// vertices of an index pair, each index read as u32 and clamped to the last vertex, as the
// built-in gather does. A value belongs to a vertex, or with color_per 1 to a strip or an edge;
// one value colors every segment, and a value past the end of `values` is NaN. The range is that
// of the values as the producer's stats give it: a NaN anywhere, or no value, leaves it unknown.
//
// `scratch` holds the ends of the strips at ENDS, the chunks of strips at CHUNK, and the keys and
// flags that each invocation of scan_values saw at HI, LO and FLAGS. No pass uses workgroup
// memory, which the derive runtime leaves without storage.

const STRIPS: u32 = 1048576u;   // the most strips, as the built-in takes
const CHUNKS: u32 = 1024u;
const ENDS: u32 = 0u;
const CHUNK: u32 = 1048576u;
const HI: u32 = 1049600u;
const LO: u32 = 1053696u;
const FLAGS: u32 = 1057792u;
const SCANS: u32 = 4096u;      // the invocations of scan_values, 16 workgroups of 256
const NAN_BITS: u32 = 0x7fc00000u;
const NONE: u32 = 0xffffffffu;

fn vertex_count() -> u32 {
    return shapes.positions.z;
}

fn position(v: u32) -> vec3<f32> {
    let d = shapes.positions.w;
    let base = v * d;
    var p = vec3<f32>(positions[base], 0.0, 0.0);
    if (d > 1u) {
        p.y = positions[base + 1u];
    }
    if (d > 2u) {
        p.z = positions[base + 2u];
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

fn strip_count() -> u32 {
    return min(arrayLength(&topology), STRIPS);
}

fn chunk_size() -> u32 {
    return (strip_count() + CHUNKS - 1u) / CHUNKS;
}

// The length of strip s, a negative length as 0.
fn strip_length(s: u32) -> u32 {
    return u32(max(topology[s], 0));
}

// a + b, at most 0xffffffff.
fn add_clamped(a: u32, b: u32) -> u32 {
    if (b > NONE - a) {
        return NONE;
    }
    return a + b;
}

@compute @workgroup_size(256)
fn sum_chunks(@builtin(global_invocation_id) id: vec3<u32>) {
    if (params.mode != 1u || id.x >= CHUNKS) {
        return;
    }
    let size = chunk_size();
    let begin = min(id.x * size, strip_count());
    let stop = min(begin + size, strip_count());
    var sum = 0u;
    for (var s = begin; s < stop; s = s + 1u) {
        sum = add_clamped(sum, strip_length(s));
    }
    scratch[CHUNK + id.x] = sum;
}

@compute @workgroup_size(1)
fn start_chunks() {
    if (params.mode != 1u) {
        return;
    }
    var start = 0u;
    for (var c = 0u; c < CHUNKS; c = c + 1u) {
        let sum = scratch[CHUNK + c];
        scratch[CHUNK + c] = start;
        start = add_clamped(start, sum);
    }
}

@compute @workgroup_size(256)
fn end_strips(@builtin(global_invocation_id) id: vec3<u32>) {
    if (params.mode != 1u || id.x >= CHUNKS) {
        return;
    }
    let size = chunk_size();
    let begin = min(id.x * size, strip_count());
    let stop = min(begin + size, strip_count());
    var end = scratch[CHUNK + id.x];
    for (var s = begin; s < stop; s = s + 1u) {
        end = add_clamped(end, strip_length(s));
        scratch[ENDS + s] = end;
    }
}

// The first strip whose end lies after vertex v, or NONE.
fn strip_of(v: u32) -> u32 {
    var low = 0u;
    var high = strip_count();
    while (low < high) {
        let mid = low + (high - low) / 2u;
        if (scratch[ENDS + mid] > v) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    if (low < strip_count()) {
        return low;
    }
    return NONE;
}

@compute @workgroup_size(256)
fn build_segments(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = vertex_count();
    var count = n / 2u;
    if (params.mode == 1u) {
        count = max(n, 1u) - 1u;
    } else if (params.mode == 2u) {
        count = select(0u, arrayLength(&topology) / 2u, n > 0u);
    }
    count = min(count, arrayLength(&segments));
    if (id.x == 0u) {
        atomicStore(&segments_count, count);
    }
    let i = id.x;
    if (i >= count) {
        return;
    }
    var a = 2u * i;
    var b = a + 1u;
    var ca = source(a);
    var cb = source(b);
    var keep = true;
    if (params.mode == 1u) {
        a = i;
        b = i + 1u;
        let s = strip_of(i);
        keep = s != NONE && b < scratch[ENDS + min(s, STRIPS - 1u)];
        if (params.color_per == 1u) {
            ca = source(s);
            cb = ca;
        } else {
            ca = source(a);
            cb = source(b);
        }
    } else if (params.mode == 2u) {
        a = min(bitcast<u32>(topology[2u * i]), n - 1u);
        b = min(bitcast<u32>(topology[2u * i + 1u]), n - 1u);
        if (params.color_per == 1u) {
            ca = source(i);
            cb = ca;
        } else {
            ca = source(a);
            cb = source(b);
        }
    }
    let pa = position(a);
    let pb = position(b);
    segments[i] = Segment(pa.x, pa.y, pa.z, ca, pb.x, pb.y, pb.z, cb, a, select(NONE, b, keep));
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
