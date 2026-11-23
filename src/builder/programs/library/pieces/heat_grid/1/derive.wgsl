@compute @workgroup_size(64)
fn recovery_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    let layers=shapes.grid.z;
    let positions=shapes.grid.w;
    let valid=layers>0u && layers<=512u && positions>0u && positions<=32u;
    let n=select(0u,layers*positions,valid);
    if (i==0u) { atomicStore(&cells_count,n); }
    if (i>=n) { return; }
    let layer=i/positions;
    let pos=i%positions;
    let size=vec2<f32>(1.8/f32(positions),1.8/f32(layers));
    let origin=vec2<f32>(-0.9+f32(pos)*size.x,-0.9+f32(layer)*size.y);
    let value=grid[i];
    let finite=abs(value)<=3.402823e38;
    cells[i]=Cell(vec4<f32>(origin,origin+size),select(0.0,value,finite),layer,pos,select(0u,1u,finite));
}
