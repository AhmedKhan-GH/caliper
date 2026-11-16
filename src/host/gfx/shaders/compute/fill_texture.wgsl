// The texture path's fill: RGBA8 texels, each a u32 r | g << 8 | b << 16 | a << 24, read from a
// storage buffer and stored to an rgba8unorm texture. The texels of row y start at word
// base + y * row, so an image may sit at any 4-byte offset of an imported buffer and have any row
// pitch that is a multiple of 4 bytes, which a buffer-to-texture copy (256-byte rows) cannot take.
// A texel past width or height is left alone. Each channel goes to [0, 1] as k / 255 and the store
// rounds it back to k, so the texture holds the buffer's bytes exactly.
struct FillParams {
  width: u32,    // texels per row
  height: u32,   // rows
  base: u32,     // the word of texel (0, 0)
  row: u32,      // words from one row to the next
};
@group(0) @binding(0) var<uniform> fp: FillParams;
@group(0) @binding(1) var<storage, read> fill_words: array<u32>;
@group(0) @binding(2) var fill_out: texture_storage_2d<rgba8unorm, write>;

// 16 x 8 invocations, as the colormap's texture form: compatibility mode's default limit.
@compute @workgroup_size(16, 8)
fn cs_fill_texture(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= fp.width || id.y >= fp.height) { return; }
  let c = fill_words[fp.base + id.y * fp.row + id.x];
  textureStore(fill_out, id.xy, vec4<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu),
                                          f32((c >> 16u) & 0xffu), f32(c >> 24u)) / 255.0);
}
