// Shared layout constants for the 10-digit MNIST accuracy ring.
const CLASSES: u32 = 10u;
const RING_R: f32 = 0.85;
const BASE_Y: f32 = -0.65;

fn digit_angle(digit: u32) -> f32 {
    return 6.2831853 * f32(digit) / f32(CLASSES);
}

fn ring_pos(digit: u32, radius: f32, y: f32) -> vec3<f32> {
    let a = digit_angle(digit);
    return vec3<f32>(radius * cos(a), y, radius * sin(a));
}
