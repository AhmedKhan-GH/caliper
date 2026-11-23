const SEG: u32 = 32u;    // segments of one arc: 64 vertices of a strip

struct ArcOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

// The color of each head of a layer, four hues that stay apart when arcs add up.
fn head_color(head: u32) -> vec3<f32> {
    var colors = array<vec3<f32>, 4>(vec3<f32>(1.0, 0.55, 0.2), vec3<f32>(0.25, 0.75, 1.0),
                                     vec3<f32>(0.85, 0.4, 1.0), vec3<f32>(0.45, 0.95, 0.45));
    return colors[head % 4u];
}

// A point of the arc's quadratic curve, raised above its ring by its weight, and by its head, so
// that the same token pair of four heads gives four arcs and not one.
fn arc_point(arc: Arc, t: f32) -> vec3<f32> {
    let p0 = ring_point(arc.layer, arc.from_token);
    let p2 = ring_point(arc.layer, arc.to_token);
    let lift = 0.08 + 0.45 * arc.weight + 0.12 * f32(arc.head);
    let p1 = 0.5 * (p0 + p2) + vec3<f32>(0.0, lift, 0.0);
    let u = 1.0 - t;
    return u * u * p0 + 2.0 * u * t * p1 + t * t * p2;
}

// A strip along the arc, of a width in pixels that grows with the weight, colored by the head. A
// weak weight gives a faint arc and a strong one a bright arc, below white even where a few add up.
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
    let selected = pick::selected(i);
    var width = (1.0 + 2.5 * arc.weight) * caliper_view.time.z;
    if (selected) {
        width = width * 1.6;
    }
    var out: ArcOut;
    out.position = camera::offset_pixels(c, vec2<f32>(-d.y, d.x) * side * width);
    // Fainter towards the ends, where the arcs of one token meet and would add up to white.
    let taper = 0.3 + 0.7 * sin(3.14159265 * t);
    let k = (0.025 + 0.28 * arc.weight * arc.weight) * taper;
    out.color = vec4<f32>(head_color(arc.head) * k, k);
    out.item = i;
    if (selected) {
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
    return sum * 0.03 * params.glow;
}
