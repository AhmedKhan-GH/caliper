struct BarOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

// 36 unit-cube vertices (6 faces x 2 triangles) in [-0.5, 0.5] x [0, 1] x [-0.5, 0.5].
fn cube_vertex(v: u32) -> vec3<f32> {
    var pts = array<vec3<f32>, 36>(
        // +Z front
        vec3<f32>(-0.5, 0.0,  0.5), vec3<f32>( 0.5, 0.0,  0.5), vec3<f32>( 0.5, 1.0,  0.5),
        vec3<f32>(-0.5, 0.0,  0.5), vec3<f32>( 0.5, 1.0,  0.5), vec3<f32>(-0.5, 1.0,  0.5),
        // -Z back
        vec3<f32>( 0.5, 0.0, -0.5), vec3<f32>(-0.5, 0.0, -0.5), vec3<f32>(-0.5, 1.0, -0.5),
        vec3<f32>( 0.5, 0.0, -0.5), vec3<f32>(-0.5, 1.0, -0.5), vec3<f32>( 0.5, 1.0, -0.5),
        // +X right
        vec3<f32>( 0.5, 0.0,  0.5), vec3<f32>( 0.5, 0.0, -0.5), vec3<f32>( 0.5, 1.0, -0.5),
        vec3<f32>( 0.5, 0.0,  0.5), vec3<f32>( 0.5, 1.0, -0.5), vec3<f32>( 0.5, 1.0,  0.5),
        // -X left
        vec3<f32>(-0.5, 0.0, -0.5), vec3<f32>(-0.5, 0.0,  0.5), vec3<f32>(-0.5, 1.0,  0.5),
        vec3<f32>(-0.5, 0.0, -0.5), vec3<f32>(-0.5, 1.0,  0.5), vec3<f32>(-0.5, 1.0, -0.5),
        // +Y top
        vec3<f32>(-0.5, 1.0,  0.5), vec3<f32>( 0.5, 1.0,  0.5), vec3<f32>( 0.5, 1.0, -0.5),
        vec3<f32>(-0.5, 1.0,  0.5), vec3<f32>( 0.5, 1.0, -0.5), vec3<f32>(-0.5, 1.0, -0.5),
        // -Y bottom
        vec3<f32>(-0.5, 0.0, -0.5), vec3<f32>( 0.5, 0.0, -0.5), vec3<f32>( 0.5, 0.0,  0.5),
        vec3<f32>(-0.5, 0.0, -0.5), vec3<f32>( 0.5, 0.0,  0.5), vec3<f32>(-0.5, 0.0,  0.5),
    );
    return pts[min(v, 35u)];
}

fn face_shade(face: u32) -> f32 {
    var shades = array<f32, 6>(0.92, 0.72, 0.84, 0.76, 1.0, 0.62);
    return shades[min(face, 5u)];
}

fn digit_color(d: u32) -> vec3<f32> {
    var palette = array<vec3<f32>, 10>(
        vec3<f32>(0.22, 0.56, 0.96),
        vec3<f32>(0.18, 0.74, 0.68),
        vec3<f32>(0.64, 0.36, 0.94),
        vec3<f32>(0.20, 0.72, 0.38),
        vec3<f32>(0.88, 0.32, 0.48),
        vec3<f32>(0.28, 0.68, 0.92),
        vec3<f32>(0.54, 0.42, 0.92),
        vec3<f32>(0.16, 0.78, 0.56),
        vec3<f32>(0.84, 0.38, 0.66),
        vec3<f32>(0.34, 0.52, 0.96),
    );
    return palette[d % 10u];
}

@vertex
fn bar_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> BarOut {
    let m = metrics[i];
    let p = cube_vertex(v);
    let width = 0.28;
    let height = (0.14 + 1.35 * clamp(m.accuracy, 0.0, 1.0)) * params.scale;
    let a = digit_angle(m.digit);
    let ca = cos(a);
    let sa = sin(a);
    let lx = p.x * width;
    let ly = p.y * height;
    let lz = p.z * width;
    let rx = lx * ca - lz * sa;
    let rz = lx * sa + lz * ca;
    let world = ring_pos(m.digit, RING_R, BASE_Y) + vec3<f32>(rx, ly, rz);
    let shade = face_shade(v / 6u);
    var rgb = digit_color(m.digit) * shade;
    if (pick::selected(i)) {
        rgb = vec3<f32>(1.0, 0.86, 0.24) * (0.85 + 0.15 * shade);
    }
    var out: BarOut;
    out.position = camera::clip(world);
    out.color = vec4<f32>(rgb, 1.0);
    out.item = i;
    return out;
}

@fragment
fn bar_fs(in: BarOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}
