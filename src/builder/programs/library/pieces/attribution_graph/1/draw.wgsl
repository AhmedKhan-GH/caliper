// The producer has no view dimensions. Resolve subpixel cells in the draw half.
// Both edge endpoints and node sprites use this position in each view.
// The piece uses an axis-aligned pan/zoom camera.
fn raster_position(i: u32) -> vec3<f32> {
    let p = nodes[i].position.xyz;
    let c = cells[i];
    if (c.size < 2u) { return p; }
    let columns = min(c.size, max(1u, u32(ceil(sqrt(f32(c.size)*c.width/c.height)))));
    let rows = (c.size+columns-1u)/columns;
    let spacing = vec2<f32>(0.8*c.width/f32(columns), 0.8*c.height/f32(rows));
    let lo = camera::clip(vec3<f32>(c.center + vec2<f32>(-0.4*c.width, 0.4*c.height), 0.0));
    let hi = camera::clip(vec3<f32>(c.center + vec2<f32>(0.4*c.width, -0.4*c.height), 0.0));
    let pixels = caliper_view.viewport.zw*0.5;
    let start = (lo.xy/lo.w*vec2<f32>(1.0,-1.0)+1.0)*pixels + caliper_view.viewport.xy;
    let end = (hi.xy/hi.w*vec2<f32>(1.0,-1.0)+1.0)*pixels + caliper_view.viewport.xy;
    let pitch = (end-start)/vec2<f32>(f32(columns),f32(rows));
    // Relaxation can reduce horizontal separation to 70 percent of its pitch.
    if (pitch.x*(1.0-2.0*c.motion) >= 1.0 && pitch.y >= 1.0) { return p; }
    let first = ceil(start-0.5);
    let extent = floor(end-0.5)-first+1.0;
    // A view with too few samples cannot retain every ID. Keep its geometry.
    if (any(extent < vec2<f32>(1.0)) || extent.x*extent.y < f32(c.size)) { return p; }
    let cols = clamp(ceil(sqrt(f32(c.size)*extent.x/extent.y)), ceil(f32(c.size)/extent.y), extent.x);
    let row_count = ceil(f32(c.size)/cols);
    let slot = vec2<f32>(f32(c.rank%u32(cols)), f32(c.rank/u32(cols)));
    let grid = vec2<f32>(cols,row_count);
    let lower = first+floor(slot*extent/grid);
    let upper = first+floor((slot+1.0)*extent/grid)-1.0;
    let original_x = c.center.x+(f32(c.rank%columns)+0.5-0.5*f32(columns))*spacing.x;
    let relaxation = clamp((p.x-original_x)/spacing.x,-0.15,0.15);
    let shift = vec2<f32>(relaxation*(upper.x-lower.x+1.0),0.0);
    let sample = clamp(floor((lower+upper)*0.5+0.5+shift),lower,upper)+0.5;
    let center = camera::clip(p);
    let screen = (center.xy/center.w*vec2<f32>(1.0,-1.0)+1.0)*pixels + caliper_view.viewport.xy;
    let clip = camera::offset_pixels(center,(sample-screen)*vec2<f32>(1.0,-1.0));
    let world = caliper_view.world_from_clip*clip;
    return world.xyz/world.w;
}

struct EdgeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) segment: u32,
}
fn cubic(c: Curve, t: f32) -> vec2<f32> {
    let u = 1.0-t;
    return u*u*u*c.start_control.xy + 3.0*u*u*t*c.start_control.zw +
           3.0*u*t*t*c.control_end.xy + t*t*t*c.control_end.zw;
}
// Published viewer PRGn cubic B-spline (init-cg.js:73-74, 78, 84):
// d3.interpolatePRGn(_linearTScale(_linearPctScale(pctInput))).
fn prgn_basis(t1: f32, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, v3: vec3<f32>) -> vec3<f32> {
    let t2 = t1*t1;
    let t3 = t2*t1;
    return ((1.0 - 3.0*t1 + 3.0*t2 - t3)*v0 +
            (4.0 - 6.0*t2 + 3.0*t3)*v1 +
            (1.0 + 3.0*t1 + 3.0*t2 - 3.0*t3)*v2 +
            t3*v3) / 6.0;
}
fn prgn_color(pct_input: f32) -> vec3<f32> {
    var stops = array<vec3<f32>, 11>(
        vec3<f32>(64.0, 0.0, 75.0) / 255.0,
        vec3<f32>(118.0, 42.0, 131.0) / 255.0,
        vec3<f32>(153.0, 112.0, 171.0) / 255.0,
        vec3<f32>(194.0, 165.0, 207.0) / 255.0,
        vec3<f32>(231.0, 212.0, 232.0) / 255.0,
        vec3<f32>(247.0, 247.0, 247.0) / 255.0,
        vec3<f32>(217.0, 240.0, 211.0) / 255.0,
        vec3<f32>(166.0, 219.0, 160.0) / 255.0,
        vec3<f32>(90.0, 174.0, 97.0) / 255.0,
        vec3<f32>(27.0, 120.0, 55.0) / 255.0,
        vec3<f32>(0.0, 68.0, 27.0) / 255.0,
    );
    let u = (pct_input + 0.4) / 0.8;
    let raw_t = select(0.501 + (u - 0.5) * (0.499 / 0.5), u * (0.499 / 0.5), u < 0.5);
    let t = clamp(raw_t, 0.0, 1.0);
    let n = 10u;
    let scaled = t * f32(n);
    let idx = select(min(u32(floor(scaled)), n - 1u), n - 1u, t >= 1.0);
    let t1 = scaled - f32(idx);
    let v1 = stops[idx];
    let v2 = stops[idx + 1u];
    var v0 = 2.0 * v1 - v2;
    if (idx > 0u) { v0 = stops[idx - 1u]; }
    var v3 = 2.0 * v2 - v1;
    if (idx + 1u < n) { v3 = stops[idx + 2u]; }
    return clamp(prgn_basis(t1, v0, v1, v2, v3), vec3<f32>(0.0), vec3<f32>(1.0));
}
// Published viewer visibility (init-cg-link-graph.js:209-215, 225-238):
// overview draws zero edges when unclicked and unpinned; clicked node shows all incoming
// and outgoing links with a #000 +0.05 px backing layer; linked selection matches pinnedLinks.
fn edge_visible(c: Curve, backing: bool) -> bool {
    if (c.weight == 0.0) { return false; }
    if (params.selected_node >= 0 && u32(params.selected_node) < counts.nodes) {
        let selected = u32(params.selected_node);
        return c.source_node == selected || c.target_node == selected;
    }
    if (backing) { return false; }
    if (pick::selection_count() > 0u) {
        return pick::selected(c.source_node) || pick::selected(c.target_node);
    }
    return false;
}
fn edge_vertex(i: u32, v: u32, backing: bool) -> EdgeOut {
    var out: EdgeOut;
    var c = curves[i];
    if (!edge_visible(c, backing)) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.segment = 0u;
        return out;
    }
    let source = raster_position(c.source_node).xy;
    let destination = raster_position(c.target_node).xy;
    let mid = 0.5*(source.y+destination.y);
    c.start_control = vec4<f32>(source,source.x,mid);
    c.control_end = vec4<f32>(destination.x,mid,destination);
    let segment = v/6u;
    let corner = v%6u;
    let t0 = f32(segment)/12.0;
    let t1 = f32(segment+1u)/12.0;
    let a = camera::clip(vec3<f32>(cubic(c,t0),0.0));
    let b = camera::clip(vec3<f32>(cubic(c,t1),0.0));
    let d = (b.xy/b.w-a.xy/a.w)*caliper_view.viewport.zw;
    let normal = vec2<f32>(-d.y,d.x)/max(length(d),0.00001);
    let end = corner == 1u || corner == 4u || corner == 5u;
    let side = select(-1.0,1.0,corner == 2u || corner == 3u || corner == 5u);
    // Published viewer width (init-cg.js:76,82; init-cg-link-graph.js:179,234):
    // scaleSqrt().domain([0, 1]).range([0.00001, 3]) plus 0.05 px on bgLinks.
    let abs_pct = abs(c.normalized);
    let base_width = 0.00001 + (3.0 - 0.00001) * sqrt(abs_pct);
    let width_px = (base_width + select(0.0, 0.05, backing)) * caliper_view.time.z;
    // Subpixel lines carry analytical 1-pixel box coverage like Canvas 2D stroke rasterization.
    let geom_width = max(width_px, 1.0);
    let alpha = clamp(width_px, 0.0, 1.0);
    var p = a;
    if (end) { p = b; }
    let offset_p = camera::offset_pixels(p, normal*side*0.5*geom_width);
    // Share a subpixel grid across rasterizers before blending.
    let screen = (offset_p.xy/offset_p.w*vec2<f32>(1.0,-1.0)+1.0)*
                 caliper_view.viewport.zw*0.5+caliper_view.viewport.xy;
    let sample = floor(screen*16.0+0.5)/16.0;
    // Stronger |pctInput| paints over weaker links (init-cg-link-graph.js:174).
    let z_ndc = select(0.99 - 0.98 * abs_pct, 0.0, backing);
    out.position = vec4<f32>(((sample-caliper_view.viewport.xy)/caliper_view.viewport.zw*2.0-1.0)*
                            vec2<f32>(1.0,-1.0)*offset_p.w, z_ndc * offset_p.w, offset_p.w);
    let color = select(prgn_color(c.normalized), vec3<f32>(0.0), backing);
    out.color = vec4<f32>(color, alpha);
    let seg = i*12u+segment+1u;
    out.segment = select(seg, seg | 0x80000000u, backing);
    return out;
}
@vertex
fn edge_bg_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> EdgeOut {
    return edge_vertex(i, v, true);
}
@vertex
fn edge_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> EdgeOut {
    return edge_vertex(i, v, false);
}
@fragment
fn edge_fs(in: EdgeOut) -> CaliperPicked {
    if (in.color.a <= 0.0) { discard; }
    var out: CaliperPicked;
    out.color = in.color;
    // Program ID zero prevents edge picks. The item distinguishes raster boundaries in the ID target.
    out.id = vec2<u32>(0u,in.segment);
    return out;
}

struct NodeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
    @location(2) @interpolate(flat) kind: u32,
    @location(3) @interpolate(flat) density: u32,
}
@vertex
fn node_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> NodeOut {
    let corners = array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),
                                   vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0));
    let p = vec4<f32>(raster_position(i),nodes[i].position.w);
    let center = camera::clip(p.xyz);
    let right = camera::clip(p.xyz+vec3<f32>(p.w,0.0,0.0));
    let up = camera::clip(p.xyz+vec3<f32>(0.0,p.w,0.0));
    let pixels = caliper_view.viewport.zw*0.5;
    let radius = min(length((right.xy/right.w-center.xy/center.w)*pixels),
                     length((up.xy/up.w-center.xy/center.w)*pixels));
    var origin = center;
    var raster_radius = radius;
    let density = radius < 2.0;
    if (density) {
        // Use the pixel that contains the derived center. Never enlarge the radius.
        // A marker occupies one sample when the cell's centers have distinct pixels.
        let screen = (center.xy/center.w*vec2<f32>(1.0,-1.0)+1.0)*pixels + caliper_view.viewport.xy;
        let shift = (floor(screen)+0.5-screen)*vec2<f32>(1.0,-1.0);
        origin = camera::offset_pixels(center,shift);
        raster_radius = min(radius,0.5);
    }
    var out: NodeOut;
    out.position = camera::offset_pixels(origin,corners[v]*raster_radius);
    out.uv = corners[v];
    out.item = i;
    out.kind = nodes[i].kind;
    out.density = u32(density);
    return out;
}
@fragment
fn node_fs(in: NodeOut) -> CaliperPicked {
    // Token squares, feature discs, error diamonds, and logit triangles have distinct silhouettes.
    if (in.density == 0u) {
        if (in.kind == 1u && dot(in.uv,in.uv)>1.0) { discard; }
        if (in.kind == 2u && abs(in.uv.x)+abs(in.uv.y)>1.0) { discard; }
        if (in.kind == 3u && abs(in.uv.x)>(1.0-in.uv.y)*0.5) { discard; }
    }
    let palette = array<vec3<f32>,4>(vec3<f32>(0.65,0.72,0.8),vec3<f32>(0.25,0.82,0.65),
                                    vec3<f32>(0.98,0.65,0.2),vec3<f32>(0.72,0.52,0.96));
    var color = palette[in.kind];
    if (i32(in.item) == params.selected_node || pick::selected(in.item)) {
        color = mix(color,vec3<f32>(1.0),0.65);
    }
    var out: CaliperPicked;
    out.color = vec4<f32>(color,1.0);
    out.id = pick::id(in.item);
    return out;
}
