// lib module camera@1 (design/programmable-graphics.md, Lib Modules): clip
// positions and offsets in pixels from the host camera. A piece that imports
// it calls camera::clip, camera::pixel_direction and camera::offset_pixels.
// Released; never changes.

// The clip position of a world position.
fn camera__clip(world: vec3<f32>) -> vec4<f32> {
    return caliper_view.clip_from_world * vec4<f32>(world, 1.0);
}

// The unit direction in pixels from clip position a to clip position b, or
// (1, 0) when they fall on one pixel.
fn camera__pixel_direction(a: vec4<f32>, b: vec4<f32>) -> vec2<f32> {
    let d = (b.xy / b.w - a.xy / a.w) * caliper_view.viewport.zw;
    let n = length(d);
    if (n < 1e-6) {
        return vec2<f32>(1.0, 0.0);
    }
    return d / n;
}

// A clip position moved by a distance in pixels on the screen.
fn camera__offset_pixels(clip: vec4<f32>, pixels: vec2<f32>) -> vec4<f32> {
    return clip + vec4<f32>(pixels * 2.0 / caliper_view.viewport.zw * clip.w, 0.0, 0.0);
}
