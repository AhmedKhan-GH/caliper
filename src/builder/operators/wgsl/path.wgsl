// The operator path (design/instrument-interface.md, Path Operator): a
// polyline through the rows of a (R, D) tensor, at most max_points vertices.
// Not built yet: this entry point writes nothing.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
}
