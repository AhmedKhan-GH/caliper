// The operator surface (design/instrument-interface.md, Surface Operator):
// pools a (R, C) tensor into a height field of at most max_cells cells.
// Not built yet: this entry point writes nothing.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
}
