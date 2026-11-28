// The sizes of the model and of its fixed sentence, which both halves read.
const L: u32 = 2u;     // layers
const H: u32 = 4u;     // heads
const T: u32 = 32u;    // tokens of the fixed sentence
const P: u32 = 12u;    // length of the repeated segment
const S: u32 = 4u;     // first token of its first copy

// The grid's place in the panel, in uv from the top left. The margin on the left holds the words
// of the rows; the diagonal names the columns, since row t and column t are the same token.
const GX0: f32 = 0.10;
const GY0: f32 = 0.035;
const GX1: f32 = 0.995;
const GY1: f32 = 0.99;

// The clip-space point of a place in uv from the top left; the piece draws with no camera.
fn clip_of(uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);
}
