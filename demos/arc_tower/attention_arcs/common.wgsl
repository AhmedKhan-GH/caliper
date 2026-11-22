// The sizes of the model and of its fixed sentence, which both halves read.
const L: u32 = 2u;     // layers
const H: u32 = 4u;     // heads
const T: u32 = 32u;    // tokens of the fixed sentence
const P: u32 = 12u;    // length of the repeated segment
const S: u32 = 4u;     // first token of its first copy

// The place of a token on the ring of a layer: a circle of radius 1, one ring above the other.
fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(cos(a), f32(layer) * 0.6, sin(a));
}
