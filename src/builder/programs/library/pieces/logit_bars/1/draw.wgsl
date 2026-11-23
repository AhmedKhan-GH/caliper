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
    let current=in.uv.y<=c.probability && in.uv.x>=0.16 && in.uv.x<=0.84;
    let ghost=in.uv.y<=c.baseline;
    if (!current && !ghost) { discard; }
    var color=vec3<f32>(0.52,0.64,0.8);
    var alpha=0.22;
    if (ghost && (in.uv.x<0.05 || in.uv.x>0.95 || c.baseline-in.uv.y<0.008)) { alpha=0.65; }
    if (current) { color=vec3<f32>(0.2,0.78,0.68); alpha=1.0; }
    if (pick::selected(in.item)) { color=mix(color,vec3<f32>(1.0),0.3); }
    var out: CaliperPicked;
    out.color=vec4<f32>(color*alpha,alpha);
    out.id=pick::id(in.item);
    return out;
}
