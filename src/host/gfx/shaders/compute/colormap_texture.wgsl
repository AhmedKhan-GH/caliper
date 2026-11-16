// The texture path's colormap: one f32 value per texel, read as bits from a storage buffer, its
// colour decided by the colormap by keys (cmap_color, colormap.wgsl) and stored to an rgba8unorm
// texture. The values of row y start at element base + y * row, so a value array may sit at any
// 4-byte offset of an imported buffer and have any row pitch; a texel past width or height is
// left alone.
struct TexParams {
  width: u32,    // texels per row
  height: u32,   // rows
  base: u32,     // the element of texel (0, 0)
  row: u32,      // elements from one row to the next
};
@group(0) @binding(0) var<uniform> tp: TexParams;
@group(0) @binding(1) var<storage, read> tex_values: array<u32>;   // f32 values as bits, so NaN survives
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_colormap_texture(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= tp.width || id.y >= tp.height) { return; }
  textureStore(tex_out, id.xy, cmap_color(tex_values[tp.base + id.y * tp.row + id.x]));
}
