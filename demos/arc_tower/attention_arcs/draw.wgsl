const SEG: u32 = 32u;    // segments of one arc: 64 vertices of a strip

struct ArcOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

// A point of the arc's quadratic curve, raised above its ring by its weight.
fn arc_point(arc: Arc, t: f32) -> vec3<f32> {
    let p0 = ring_point(arc.layer, arc.from_token);
    let p2 = ring_point(arc.layer, arc.to_token);
    let p1 = 0.5 * (p0 + p2) + vec3<f32>(0.0, 0.1 + 0.5 * arc.weight, 0.0);
    let u = 1.0 - t;
    return u * u * p0 + 2.0 * u * t * p1 + t * t * p2;
}

// A strip along the arc, of a width in pixels that grows with the weight, colored by the head.
@vertex
fn arc_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> ArcOut {
    let arc = arcs[i];
    let t = f32(v / 2u) / f32(SEG - 1u);
    let e = 0.5 / f32(SEG - 1u);
    let c = camera::clip(arc_point(arc, t));
    let a = camera::clip(arc_point(arc, max(t - e, 0.0)));
    let b = camera::clip(arc_point(arc, min(t + e, 1.0)));
    let d = camera::pixel_direction(a, b);
    let side = f32(v % 2u) - 0.5;
    let width = (1.0 + 5.0 * arc.weight) * caliper_view.time.z;
    var out: ArcOut;
    out.position = camera::offset_pixels(c, vec2<f32>(-d.y, d.x) * side * width);
    let rgb = colormap::viridis(f32(arc.head) / f32(H - 1u)).rgb * (0.3 + 0.7 * arc.weight);
    out.color = vec4<f32>(rgb * arc.weight, arc.weight);
    out.item = i;
    if (pick::selected(i)) {
        out.color = vec4<f32>(1.0, 0.9, 0.6, 1.0);
    }
    return out;
}

@fragment
fn arc_fs(in: ArcOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}

@fragment
fn arc_glow(in: ArcOut) -> @location(0) vec4<f32> {
    return in.color;
}

// A Gaussian blur of the half-size glow target, added over the arcs.
@fragment
fn bloom(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(glow));
    var sum = vec4<f32>(0.0);
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let w = exp(-f32(x * x + y * y) / 4.0);
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0;
            sum = sum + w * textureSampleLevel(glow, caliper_linear, uv, 0.0);
        }
    }
    return sum * 0.12 * params.glow;
}
