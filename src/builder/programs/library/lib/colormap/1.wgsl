// lib module colormap@1 (design/programmable-graphics.md, Lib Modules): the
// host's colormaps. A piece that imports it calls colormap::sample,
// colormap::viridis or colormap::magma. Released; never changes.

// The rows of caliper_colormaps.
const colormap__VIRIDIS: u32 = 0u;
const colormap__MAGMA: u32 = 1u;
const colormap__RDBU: u32 = 2u;
const colormap__GREYS: u32 = 3u;

// The color of t, from 0 to 1, on a host colormap, linear between its texels.
fn colormap__sample(map: u32, t: f32) -> vec4<f32> {
    let size = textureDimensions(caliper_colormaps);
    let row = i32(min(map, size.y - 1u));
    let x = clamp(t, 0.0, 1.0) * f32(size.x - 1u);
    let x0 = i32(floor(x));
    let x1 = min(x0 + 1, i32(size.x) - 1);
    let a = textureLoad(caliper_colormaps, vec2<i32>(x0, row), 0);
    let b = textureLoad(caliper_colormaps, vec2<i32>(x1, row), 0);
    return mix(a, b, x - f32(x0));
}

fn colormap__viridis(t: f32) -> vec4<f32> {
    return colormap__sample(colormap__VIRIDIS, t);
}

fn colormap__magma(t: f32) -> vec4<f32> {
    return colormap__sample(colormap__MAGMA, t);
}
