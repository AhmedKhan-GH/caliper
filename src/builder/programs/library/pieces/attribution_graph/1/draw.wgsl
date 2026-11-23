struct EdgeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}
fn cubic(c: Curve, t: f32) -> vec2<f32> {
    let u = 1.0-t;
    return u*u*u*c.start_control.xy + 3.0*u*u*t*c.start_control.zw +
           3.0*u*t*t*c.control_end.xy + t*t*t*c.control_end.zw;
}
@vertex
fn edge_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> EdgeOut {
    let c = curves[i];
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
    let width = (1.0+3.0*abs(c.normalized))*caliper_view.time.z;
    var p = a;
    if (end) { p = b; }
    var out: EdgeOut;
    out.position = camera::offset_pixels(p, normal*side*0.5*width);
    var color = vec3<f32>(0.15,0.55,0.95);
    if (c.weight < 0.0) { color = vec3<f32>(0.95,0.25,0.12); }
    var alpha = 0.16+0.54*abs(c.normalized);
    if (params.selected_node >= 0 && u32(params.selected_node) < counts.nodes &&
        c.source_node != u32(params.selected_node) && c.target_node != u32(params.selected_node)) { alpha *= 0.12; }
    if (c.weight == 0.0) { alpha = 0.0; }
    out.color = vec4<f32>(color*alpha,alpha);
    return out;
}
@fragment
fn edge_fs(in: EdgeOut) -> @location(0) vec4<f32> { return in.color; }

struct NodeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
    @location(2) @interpolate(flat) kind: u32,
}
@vertex
fn node_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> NodeOut {
    let corners = array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),
                                   vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0));
    let p = nodes[i].position;
    var out: NodeOut;
    out.position = camera::clip(vec3<f32>(p.xy+corners[v]*p.w,0.0));
    out.uv = corners[v];
    out.item = i;
    out.kind = nodes[i].kind;
    return out;
}
@fragment
fn node_fs(in: NodeOut) -> CaliperPicked {
    // Token squares, feature discs, error diamonds, and logit triangles have distinct silhouettes.
    if (in.kind == 1u && dot(in.uv,in.uv)>1.0) { discard; }
    if (in.kind == 2u && abs(in.uv.x)+abs(in.uv.y)>1.0) { discard; }
    if (in.kind == 3u && abs(in.uv.x)>(1.0-in.uv.y)*0.5) { discard; }
    let palette = array<vec3<f32>,4>(vec3<f32>(0.65,0.72,0.8),vec3<f32>(0.25,0.82,0.65),
                                    vec3<f32>(0.98,0.65,0.2),vec3<f32>(0.72,0.52,0.96));
    var color = palette[in.kind];
    if (i32(in.item) == params.selected_node || i32(in.item) == params.hovered_node || pick::selected(in.item)) {
        color = mix(color,vec3<f32>(1.0),0.65);
    }
    var out: CaliperPicked;
    out.color = vec4<f32>(color,1.0);
    out.id = pick::id(in.item);
    return out;
}
