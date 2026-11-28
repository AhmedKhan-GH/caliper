// Shared constants for the 10x10 MNIST confusion matrix.
const CLASSES: u32 = 10u;
const GX0: f32 = 0.14;
const GY0: f32 = 0.04;
const GX1: f32 = 0.98;
const GY1: f32 = 0.96;

fn clip_of(uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);
}
