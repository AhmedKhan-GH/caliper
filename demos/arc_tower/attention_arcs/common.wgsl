// The sizes of the model and of its fixed sentence, which both halves read.
const L: u32 = 2u;     // layers
const H: u32 = 4u;     // heads
const T: u32 = 32u;    // tokens of the fixed sentence
const P: u32 = 12u;    // length of the repeated segment
const S: u32 = 4u;     // first token of its first copy

// The tower: one ring of tokens for each layer, RISE apart, the lowest at BASE. The host frames
// a panel of programs alone on the box from -1 to 1, thus a tower of this size fills the panel.
const R: f32 = 1.45;      // the radius of a ring
const BASE: f32 = -0.85;  // the height of the ring of layer 0
const RISE: f32 = 1.5;    // the height from one ring to the next

// The place of a token on the ring of a layer.
fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(R * cos(a), BASE + f32(layer) * RISE, R * sin(a));
}
