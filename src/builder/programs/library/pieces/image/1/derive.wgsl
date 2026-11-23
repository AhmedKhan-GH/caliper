// The derive half of image@1: one element of `pixels` for each pixel of `image`, row by row, as
// packed RGBA8 the way the producer converts an image for the built-in draw. A value v becomes
// trunc(clamp(v * scale, 0, 1) * 255 + 0.5), and NaN becomes 0. A grey image (H, W) repeats its
// value in red, green and blue, and an image without alpha gets 255. `info` holds the width, the
// height, the channels and the pose. An image of another shape, or larger than `pixels`, publishes
// no pixels and a size of 0. The derive half sees the last four axes of a tensor and not its rank,
// thus (1, W, 3) and (1, W, 4) read as color images of one row, and so does a grey image (H, W)
// of width 3 or 4.

const THREADS: u32 = 16384u;   // 64 workgroups of 256

// The width, the height and the channels of `image`, with a size of 0 when the piece cannot hold it.
fn image_dims() -> vec3<u32> {
    let s = shapes.image;
    var d = vec3<u32>(s.w, s.z, 1u);
    if (s.w == 3u || s.w == 4u) {
        d = vec3<u32>(s.z, s.y, s.w);
    } else if (s.y != 1u) {
        return vec3<u32>(0u, 0u, s.w);
    }
    if (d.x == 0u || d.y == 0u || d.y > arrayLength(&pixels) / d.x) {
        return vec3<u32>(0u, 0u, d.z);
    }
    return d;
}

// One channel from 0 to 255.
fn channel(v: f32) -> u32 {
    let x = v * params.scale;
    if (!(x > 0.0)) {
        return 0u;
    }
    if (x >= 1.0) {
        return 255u;
    }
    return min(u32(x * 255.0 + 0.5), 255u);
}

// Column c of the pose, a (4, 4) or (3, 4) tensor row by row, or of the identity for another shape.
fn pose_column(c: u32) -> vec4<f32> {
    let s = shapes.pose;
    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (s.w != 4u || (s.z != 3u && s.z != 4u)) {
        col[c] = 1.0;
        return col;
    }
    col = vec4<f32>(pose[c], pose[4u + c], pose[8u + c], 0.0);
    if (s.z == 4u) {
        col.w = pose[12u + c];
    } else if (c == 3u) {
        col.w = 1.0;
    }
    return col;
}

@compute @workgroup_size(256)
fn copy_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = image_dims();
    let n = d.x * d.y;
    if (id.x == 0u) {
        atomicStore(&pixels_count, n);
    }
    for (var i = id.x; i < n; i = i + THREADS) {
        var rgba = 0xff000000u;
        if (d.z == 1u) {
            let g = channel(image[i]);
            rgba = rgba | g | (g << 8u) | (g << 16u);
        } else {
            let b = i * d.z;
            rgba = channel(image[b]) | (channel(image[b + 1u]) << 8u) | (channel(image[b + 2u]) << 16u);
            if (d.z == 4u) {
                rgba = rgba | (channel(image[b + 3u]) << 24u);
            } else {
                rgba = rgba | 0xff000000u;
            }
        }
        pixels[i] = Pixel(rgba);
    }
}

@compute @workgroup_size(1)
fn place_image() {
    let d = image_dims();
    info[0] = ImageInfo(vec4<u32>(d, 0u), pose_column(0u), pose_column(1u), pose_column(2u), pose_column(3u));
}
