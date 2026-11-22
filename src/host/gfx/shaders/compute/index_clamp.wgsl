// The index clamp, once per batch update. It copies the producer's indices into an index
// buffer the layer owns, each clamped to the last vertex, the geometry service's index-clamp
// contract. A grid-stride loop keeps the dispatch within 65,535 workgroups for any count.
struct ClampParams { count: u32, last: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> cp: ClampParams;
@group(0) @binding(1) var<storage, read> idx_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> idx_out: array<u32>;

@compute @workgroup_size(64)
fn cs_clamp_indices(@builtin(global_invocation_id) id: vec3<u32>,
                    @builtin(num_workgroups) groups: vec3<u32>) {
  for (var i = id.x; i < cp.count; i += groups.x * 64u) {
    idx_out[i] = min(idx_in[i], cp.last);
  }
}
