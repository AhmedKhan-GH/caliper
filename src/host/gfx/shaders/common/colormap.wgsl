// The colormap of the renderer core: the index is decided by integer comparisons against keys
// the CPU derives from the reference map_f32_to_rgba8; the float guess only picks which
// comparisons run, so no adapter's or compiler's float arithmetic can move an index.
// Self-contained, so that a program needs only this file for the colormap.
//
// The block holds one 16-byte entry per level k from 0 to 255: the first key of level k, the first
// key of level k + 1, and the level's packed colour. Level 0 starts at key 0 (always passed), and a
// level no float reaches starts at 0xffffffff (never passed), as does the level past 255. A level
// test and the level's colour are then one load of one entry, with no index into a vector, and a
// uniform array of vec4 keeps WGSL's 16-byte stride (src/host/gfx/colormap_keys.h fills it).
struct Cmap { vmin: f32, vmax: f32, flip: u32, pad: u32, levels: array<vec4<u32>, 256> };
@group(1) @binding(1) var<uniform> m: Cmap;

fn in_level(key: u32, e: vec4<u32>) -> bool { return key >= e.x && key < e.y; }

// The entry of the level of a value's bits: NaN from the bits, then the float guess if its entry
// confirms it, then the guess's two neighbours, then a binary search of eight integer steps.
fn colormap_entry(bits: u32) -> vec4<u32> {
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return m.levels[0]; }           // NaN, from the bits
    let key = select(bits | 0x80000000u, ~bits, (bits >> 31u) != 0u) ^ m.flip;
    let v = bitcast<f32>(bits);
    let g = min(u32(clamp((v - m.vmin) / (m.vmax - m.vmin), 0.0, 1.0) * 255.0 + 0.5), 255u);
    let e = m.levels[g];
    if (in_level(key, e)) { return e; }                                        // the usual case
    if (g > 0u) { let d = m.levels[g - 1u]; if (in_level(key, d)) { return d; } }
    if (g < 255u) { let u = m.levels[g + 1u]; if (in_level(key, u)) { return u; } }
    var lo = 0u; var hi = 256u;                         // level lo starts at or below key, level hi above
    loop { if (hi - lo <= 1u) { break; } let mid = (lo + hi) / 2u;
           if (key >= m.levels[mid].x) { lo = mid; } else { hi = mid; } }
    return m.levels[lo];
}

// A packed colour, r | g << 8 | b << 16 | a << 24, as four channels in [0, 1].
fn unpack_rgba(c: u32) -> vec4<f32> {
  return vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu),
                   f32((c >> 16u) & 0xffu), f32((c >> 24u) & 0xffu)) / 255.0;
}

fn cmap_color(bits: u32) -> vec4<f32> {
  return unpack_rgba(colormap_entry(bits).z);
}
