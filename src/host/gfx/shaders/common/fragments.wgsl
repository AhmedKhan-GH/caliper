// The two fragment stages that lines, boxes, meshes and images share: the interpolated
// colour, and its pick variant, which writes the pick identity and the item instead.

@fragment
fn fs_color(in: VOut) -> @location(0) vec4<f32> { return in.color; }

@fragment
fn fs_pick(in: VOut) -> @location(0) vec2<u32> { return in.pick; }
