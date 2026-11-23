// The derive half of frustum@1: the matrix that maps the unit frustum onto the camera's frustum in
// the world, the camera's frame times the map of a point (s, t, 1) of the unit square onto its
// image plane, ((s * width - cx) d / fx, (t * height - cy) d / fy, d), as the built-in draw of a
// camera entity composes it. The frame is the pose, with y and z turned for the convention 1. A
// pose of another shape is the identity, and a pinhole with fewer than 7 numbers, or a focal length
// or plane distance that is not above 0, places nothing (a zero matrix).

fn pose_at(r: u32, c: u32) -> f32 {
    let s = shapes.pose;
    if (s.w != 4u || (s.z != 3u && s.z != 4u)) {
        return select(0.0, 1.0, r == c);
    }
    if (r == 3u && s.z == 3u) {
        return select(0.0, 1.0, c == 3u);
    }
    return pose[r * 4u + c];
}

@compute @workgroup_size(1)
fn place_frustum() {
    var m = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    if (arrayLength(&pinhole) >= 7u) {
        let fx = pinhole[0];
        let fy = pinhole[1];
        let d = pinhole[6];
        if (fx > 0.0 && fy > 0.0 && d > 0.0) {
            let f = mat4x4<f32>(vec4<f32>(pinhole[4] * d / fx, 0.0, 0.0, 0.0), vec4<f32>(0.0, pinhole[5] * d / fy, 0.0, 0.0),
                                vec4<f32>(-pinhole[2] * d / fx, -pinhole[3] * d / fy, d, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            var frame = mat4x4<f32>(vec4<f32>(pose_at(0u, 0u), pose_at(1u, 0u), pose_at(2u, 0u), pose_at(3u, 0u)),
                                    vec4<f32>(pose_at(0u, 1u), pose_at(1u, 1u), pose_at(2u, 1u), pose_at(3u, 1u)),
                                    vec4<f32>(pose_at(0u, 2u), pose_at(1u, 2u), pose_at(2u, 2u), pose_at(3u, 2u)),
                                    vec4<f32>(pose_at(0u, 3u), pose_at(1u, 3u), pose_at(2u, 3u), pose_at(3u, 3u)));
            if (params.convention == 1u) {
                frame[1] = -frame[1];
                frame[2] = -frame[2];
            }
            m = frame * f;
        }
    }
    frustum[0] = Frustum(m[0], m[1], m[2], m[3], 0u);
}
