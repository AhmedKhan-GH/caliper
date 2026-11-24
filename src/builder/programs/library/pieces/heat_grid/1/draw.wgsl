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
    let span=max(params.vmax-params.vmin,0.000001);
    let t=clamp((c.recovery-params.vmin)/span,0.0,1.0);
    var color=mix(vec3<f32>(0.16,0.35,0.8),vec3<f32>(0.85),min(2.0*t,1.0));
    if (t>0.5) { color=mix(vec3<f32>(0.85),vec3<f32>(0.85,0.25,0.12),2.0*t-1.0); }
    if (pick::selected(in.item)) { color=mix(color,vec3<f32>(1.0),0.45); }
    var out: CaliperPicked;
    out.color=vec4<f32>(color,1.0);
    out.id=pick::id(in.item);
    return out;
}
