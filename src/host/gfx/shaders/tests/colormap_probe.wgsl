// Test only: the colormap of each input's bits, packed as the RGBA8 texel a store to an
// rgba8unorm target writes, so that a test compares it byte for byte with the CPU reference
// map_f32_to_rgba8. The test pads the inputs to a whole number of workgroups.
@group(3) @binding(0) var<storage, read> probe_in: array<u32>;
@group(3) @binding(1) var<storage, read_write> probe_out: array<u32>;

@compute @workgroup_size(64)
fn cs_colormap_probe(@builtin(global_invocation_id) id: vec3<u32>) {
  probe_out[id.x] = pack4x8unorm(cmap_color(probe_in[id.x]));
}
