// The edge gather, once per update of a lines batch with index pairs. Each pair is clamped
// to the last vertex, and the positions and per-vertex colour sources of its two ends are
// copied into the streams that vs_lines reads, so that edge k is drawn as segment k. A
// grid-stride loop keeps the dispatch within 65,535 workgroups for any edge count.
struct GatherParams { count: u32, last: u32, dims: u32, per_vertex: u32 };
@group(0) @binding(0) var<uniform> gp: GatherParams;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;           // (E, 2): i32 indices as bits
@group(0) @binding(2) var<storage, read> vert_pos: array<f32>;        // (V, dims)
@group(0) @binding(3) var<storage, read> vert_src: array<u32>;        // (V,) colour sources, read when per_vertex is 1
@group(0) @binding(4) var<storage, read_write> seg_pos: array<f32>;   // (E, 2, dims): ends at byte offsets 0 and 4 * dims
@group(0) @binding(5) var<storage, read_write> seg_src: array<u32>;   // (E, 2): ends at byte offsets 0 and 4

@compute @workgroup_size(64)
fn cs_gather_edges(@builtin(global_invocation_id) id: vec3<u32>,
                   @builtin(num_workgroups) groups: vec3<u32>) {
  for (var e = id.x; e < gp.count; e += groups.x * 64u) {
    for (var k = 0u; k < 2u; k++) {
      let v = min(pairs[2u * e + k], gp.last);
      for (var c = 0u; c < gp.dims; c++) {
        seg_pos[(2u * e + k) * gp.dims + c] = vert_pos[v * gp.dims + c];
      }
      if (gp.per_vertex == 1u) { seg_src[2u * e + k] = vert_src[v]; }
    }
  }
}
