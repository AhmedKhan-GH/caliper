struct PanelOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
}
@vertex
fn panel_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> PanelOut {
    let corners=array<vec2<f32>,6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(0.0,1.0),
                                  vec2<f32>(0.0,1.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0));
    let uv=corners[v];
    let rect=cells[i].rect;
    var out: PanelOut;
    out.position=camera::clip(vec3<f32>(mix(rect.xy,rect.zw,uv),0.0));
    out.uv=uv;
    out.item=i;
    return out;
}

@fragment
fn panel_fs(in: PanelOut) -> CaliperPicked {
    let c=cells[in.item];
    if (c.valid==0u) { discard; }
    let strength=clamp(c.activation/max(params.activation_max,0.000001),0.0,1.0);
    var color=mix(vec3<f32>(0.08,0.12,0.17),vec3<f32>(0.16,0.65,0.46),strength);
    if (pick::selected(c.row)) { color=mix(color,vec3<f32>(0.6,0.8,1.0),0.25); }
    var out: CaliperPicked;
    out.color=vec4<f32>(color,1.0);
    out.id=pick::id(c.row);
    return out;
}
