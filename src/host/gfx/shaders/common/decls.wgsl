// The declarations every scene program shares: the frame block and class table of group 0,
// the draw's parameter block of group 1, the images of group 2 and the varyings. The
// colormap block, binding 1 of group 1, is declared with the colormap in colormap.wgsl.
// Group 3 holds the geometry service's storage buffers and is never bound by these programs.

// Group 0: one per target and frame (48 bytes).
struct Frame {
  viewport: vec2<f32>,    //  0  the target's size in physical pixels
  proj_scale: vec2<f32>,  //  8  proj[0][0], proj[1][1]
  ndc_scale: vec2<f32>,   // 16  (1, 1), or the zoom of the pick window
  ndc_offset: vec2<f32>,  // 24  (0, 0), or the offset of the pick window
  class_count: u32,       // 32  ids the class table covers (65,536)
  class_default: u32,     // 36  packed RGBA for an id outside them
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var class_table: texture_2d<f32>;   // rgba8unorm, 256 x 256: ids 0 to 65,535

// Group 1: the draw's 256-byte slot and its colormap block, both at dynamic offsets.
struct PrimParams {        // 192 bytes, the native size
  mvp: mat4x4<f32>,        //   0  as native
  nmat0: vec4<f32>,        //  64  as native
  nmat1: vec4<f32>,        //  80  as native
  nmat2: vec4<f32>,        //  96  as native
  pick_id: u32,            // 112  native pos_base
  item_base: u32,          // 116  native idx_base
  point_shape: u32,        // 120  native nrm_base: 0 square, 1 disc
  radius_mode: u32,        // 124  native attr_base: 0 size_px, 1 radius in scene units
  unused0: u32,            // 128  native use_index; the clamp happens before the draw
  vertex_count: u32,       // 132  as native
  color_mode: u32,         // 136  as native, and 4 for a class id
  shade_mode: u32,         // 140  as native
  flat_rgba: u32,          // 144  as native
  vmin: f32,               // 148  as native; the colormap reads the range of its block
  vmax: f32,               // 152  as native
  size_px: f32,            // 156  as native for points; also the width of lines
  radius_scale: f32,       // 160  native uv_base
  use_instance: u32,       // 164  as native
  line_mode: u32,          // 168  native inst_base: 0 segments, 1 arrows
  use_instance_attr: u32,  // 172  as native
  image_mode: u32,         // 176  native inst_attr_base: 0 RGBA, 1 grey
  att_ref: f32,            // 180  as native: points' depth cues (CompileOptions), 0 for none
  fade_w0: f32,            // 184  as native
  fade_w1: f32,            // 188  as native
};
@group(1) @binding(0) var<uniform> p: PrimParams;

// Group 2: the image of a textured draw, by kind.
@group(2) @binding(0) var img_rgba: texture_2d<f32>;    // rgba8unorm or r8unorm
@group(2) @binding(1) var img_sampler: sampler;         // linear, clamp to edge
@group(2) @binding(2) var img_class: texture_2d<u32>;   // r32uint: i32 class ids as bits
@group(2) @binding(3) var img_value: texture_2d<u32>;   // r32uint: f32 depths or values as bits

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) @interpolate(flat, either) pick: vec2<u32>,
  @location(3) obj: vec3<f32>,   // the entity-space position, for flat mesh faces
};
