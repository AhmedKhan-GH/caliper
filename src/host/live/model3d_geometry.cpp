#include "live/model3d_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <iomanip>
#include <utility>

namespace caliper_live::model3d {
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double half_fov = pi / 8.0;
constexpr double panel_gap = 0.25;
constexpr double default_azimuth = 1.5;
constexpr double default_elevation = 1.15;
constexpr double fit_fraction = 0.86;
constexpr double epsilon = 1e-10;

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 mul(Vec3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double length(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 normalized(Vec3 a) {
    const double n = length(a);
    return n > epsilon ? mul(a, 1.0 / n) : Vec3{0, 0, -1};
}
bool finite(Vec3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }
double valid_aspect(double aspect) { return std::isfinite(aspect) && aspect > 0 ? aspect : 1.0; }
Camera valid(Camera camera) {
    if (!finite(camera.target)) camera.target = {};
    if (!std::isfinite(camera.radius) || camera.radius <= 0) camera.radius = 1;
    if (!std::isfinite(camera.distance) || camera.distance <= 0) camera.distance = camera.radius * 3;
    if (!std::isfinite(camera.azimuth)) camera.azimuth = default_azimuth;
    if (!std::isfinite(camera.elevation)) camera.elevation = default_elevation;
    camera.elevation = std::clamp(camera.elevation, -1.55, 1.55);
    return camera;
}
struct Frame {
    Camera camera;
    Vec3 eye, forward, right, up;
    double near_z = 0, far_z = 0;
};
Frame frame(Camera input) {
    const Camera camera = valid(input);
    const double ce = std::cos(camera.elevation);
    const Vec3 offset{ce * std::cos(camera.azimuth), std::sin(camera.elevation),
                      ce * std::sin(camera.azimuth)};
    Frame f;
    f.camera = camera;
    f.eye = add(camera.target, mul(offset, camera.distance));
    f.forward = mul(offset, -1);
    f.right = normalized(cross(f.forward, {0, 1, 0}));
    f.up = cross(f.right, f.forward);
    f.near_z = std::max({camera.distance - camera.radius * 1.5, camera.distance * 1e-5, 1e-7});
    f.far_z = std::max(f.near_z * 2, camera.distance + camera.radius * 1.5);
    return f;
}
Camera fit_points(const std::vector<Vec3>& points, double aspect, double azimuth, double elevation) {
    Camera camera;
    if (points.empty()) return camera;
    Vec3 minimum = points.front(), maximum = points.front();
    for (Vec3 p : points) {
        minimum.x = std::min(minimum.x, p.x); maximum.x = std::max(maximum.x, p.x);
        minimum.y = std::min(minimum.y, p.y); maximum.y = std::max(maximum.y, p.y);
        minimum.z = std::min(minimum.z, p.z); maximum.z = std::max(maximum.z, p.z);
    }
    camera.target = mul(add(minimum, maximum), 0.5);
    camera.radius = std::max(length(sub(maximum, minimum)) * 0.5, element_unit);
    camera.azimuth = azimuth;
    camera.elevation = elevation;
    camera = valid(camera);
    const Frame basis = frame(camera);
    const double vertical = std::tan(half_fov) * fit_fraction;
    const double horizontal = vertical * valid_aspect(aspect);
    camera.distance = element_unit;
    // Solve the perspective inequalities for each real corner. A bounding
    // sphere wastes most of a tall stack's viewport and hides its layers.
    for (Vec3 p : points) {
        const Vec3 relative = sub(p, camera.target);
        const double depth = dot(relative, basis.forward);
        camera.distance = std::max({camera.distance,
            std::fabs(dot(relative, basis.right)) / horizontal - depth,
            std::fabs(dot(relative, basis.up)) / vertical - depth,
            -depth + camera.radius * 0.02});
    }
    return valid(camera);
}
std::array<int64_t, 2> dimensions(const Tensor& tensor) {
    if (tensor.rows > 0 && tensor.cols > 0) return {tensor.rows, tensor.cols};
    if (tensor.shape.empty()) return {0, 0};
    if (tensor.shape.size() == 1) return {tensor.shape.front(), 1};
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < tensor.shape.size(); ++i) {
        if (tensor.shape[i] <= 0 || rows > std::numeric_limits<int64_t>::max() / tensor.shape[i]) return {0, 0};
        rows *= tensor.shape[i];
    }
    return tensor.shape.back() > 0 ? std::array<int64_t, 2>{rows, tensor.shape.back()}
                                  : std::array<int64_t, 2>{0, 0};
}
bool ends_with(const std::string& value, const char* suffix) {
    const std::string tail(suffix);
    return value.size() >= tail.size() && value.compare(value.size() - tail.size(), tail.size(), tail) == 0;
}
int layer_of(const Tensor& tensor) {
    if (tensor.layer_index >= 0) return tensor.layer_index;
    const auto start = tensor.name.find(".layers.");
    if (start == std::string::npos) return -1;
    size_t at = start + 8;
    int layer = 0;
    bool any = false;
    while (at < tensor.name.size() && tensor.name[at] >= '0' && tensor.name[at] <= '9') {
        if (layer > (std::numeric_limits<int>::max() - 9) / 10) return -1;
        layer = layer * 10 + tensor.name[at++] - '0';
        any = true;
    }
    return any && at < tensor.name.size() && tensor.name[at] == '.' ? layer : -1;
}
bool embedding(const Tensor& tensor) {
    return tensor.role == "embedding" || ends_with(tensor.name, "embed_tokens.weight");
}
bool output(const Tensor& tensor) {
    return tensor.role == "output" || ends_with(tensor.name, "lm_head.weight");
}
bool final_norm(const Tensor& tensor) {
    return tensor.role == "final_norm" || (layer_of(tensor) < 0 && ends_with(tensor.name, ".norm.weight"));
}
std::string type_of(const Tensor& tensor) {
    if (!tensor.layer_type.empty()) return tensor.layer_type;
    if (tensor.name.find("linear_attn.") != std::string::npos) return "linear_attention";
    if (tensor.name.find("self_attn.") != std::string::npos) return "full_attention";
    return {};
}
struct Bucket {
    std::string label, layer_type;
    int layer_index = -1;
    std::vector<size_t> tensors;
};
using Key = std::pair<int, int>;
void grow(Scene& scene, Vec3 point, bool& first) {
    if (first) { scene.minimum = scene.maximum = point; first = false; return; }
    scene.minimum.x = std::min(scene.minimum.x, point.x);
    scene.minimum.y = std::min(scene.minimum.y, point.y);
    scene.minimum.z = std::min(scene.minimum.z, point.z);
    scene.maximum.x = std::max(scene.maximum.x, point.x);
    scene.maximum.y = std::max(scene.maximum.y, point.y);
    scene.maximum.z = std::max(scene.maximum.z, point.z);
}
bool inside(Rect viewport, double x, double y) {
    return std::isfinite(x) && std::isfinite(y) && viewport.width > 0 && viewport.height > 0 &&
           x >= viewport.x && x < viewport.x + viewport.width &&
           y >= viewport.y && y < viewport.y + viewport.height;
}
} // namespace

int row_blocks(const Tensor& tensor) {
    if (tensor.shape.size() == 1) return 1;
    const auto shape = dimensions(tensor);
    if (shape[0] <= 0 || shape[1] <= 0 || static_cast<long double>(shape[0]) / shape[1] <= 8) return 1;
    int best = 1;
    long double best_ratio = std::numeric_limits<long double>::infinity();
    for (int blocks = 1; blocks <= 128 && blocks <= shape[0]; ++blocks) {
        if (shape[0] % blocks != 0) continue;
        const long double ratio = std::fabs(std::log(static_cast<long double>(shape[1]) * blocks * blocks / shape[0]));
        if (ratio < best_ratio) { best = blocks; best_ratio = ratio; }
    }
    return best;
}

std::array<int, 2> thumbnail_shape(const Tensor& tensor, int side) {
    const auto shape = dimensions(tensor);
    if (shape[0] <= 0 || shape[1] <= 0) return {0, 0};
    if (tensor.shape.size() == 1) return {static_cast<int>(std::min<int64_t>(shape[0], std::clamp(side, 1, 128))), 1};
    const int blocks = row_blocks(tensor);
    side = std::clamp(side, blocks, 128);
    const int rows = blocks * static_cast<int>(std::min<int64_t>({shape[0] / blocks, side, 2048 / blocks}));
    const int cols = static_cast<int>(std::min<int64_t>(shape[1], side / blocks));
    return {rows, cols};
}

std::array<double, 2> panel_size(const Tensor& tensor) {
    if (tensor.shape.size() == 1) {
        return tensor.shape.front() > 0
            ? std::array<double, 2>{tensor.shape.front() * element_unit, element_unit}
            : std::array<double, 2>{0, 0};
    }
    const auto shape = dimensions(tensor);
    if (shape[0] <= 0 || shape[1] <= 0) return {0, 0};
    const int blocks = row_blocks(tensor);
    return {static_cast<double>(shape[1]) * blocks * element_unit,
            static_cast<double>(shape[0] / blocks) * element_unit};
}

Scene layout(const std::vector<Tensor>& tensors, const std::string& output_label) {
    std::map<Key, Bucket> buckets;
    (void)output_label; // A tied output is marked on the embedding itself.
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& tensor = tensors[i];
        const auto size = panel_size(tensor);
        if (!(size[0] > 0 && size[1] > 0)) continue;
        const int layer = layer_of(tensor);
        Key key{1, 0};
        std::string label = tensor.group.empty() ? "Model" : tensor.group;
        if (embedding(tensor)) {
            key = {0, 0}; label = tensor.tied ? "Embedding / output (tied)" : "Embedding";
        } else if (layer >= 0) {
            key = {2, layer};
            std::ostringstream name;
            name << "Layer " << std::setw(2) << std::setfill('0') << layer;
            label = name.str();
        } else if (final_norm(tensor)) {
            key = {3, 0}; label = "Final norm";
        } else if (output(tensor)) {
            key = {4, 0}; label = "Output";
        }
        Bucket& bucket = buckets[key];
        bucket.label = label;
        bucket.layer_index = layer;
        if (bucket.layer_type.empty()) bucket.layer_type = type_of(tensor);
        bucket.tensors.push_back(i);
    }

    Scene scene;
    bool first = true;
    for (auto& entry : buckets) {
        Bucket& bucket = entry.second;
        std::sort(bucket.tensors.begin(), bucket.tensors.end(), [&](auto a, auto b) {
            return tensors[a].name < tensors[b].name;
        });
        const int group_index = static_cast<int>(scene.groups.size());
        const size_t begin = scene.panels.size();
        double x = 0, used_height = 0, used_width = 0;
        for (auto item : bucket.tensors) {
            const Tensor& tensor = tensors[item];
            const auto size = panel_size(tensor);
            Panel panel;
            panel.tensor = item;
            panel.label = tensor.name;
            panel.center = {x + size[0] * 0.5, 0, 0};
            panel.width = size[0]; panel.height = size[1];
            panel.group = group_index; panel.row_blocks = row_blocks(tensor);
            scene.panels.push_back(std::move(panel));
            used_width = std::max(used_width, x + size[0]);
            used_height = std::max(used_height, size[1]);
            x += size[0] + panel_gap;
        }
        double z = 0;
        if (!scene.groups.empty()) {
            const Group& previous = scene.groups.back();
            // A broad, single row per layer makes each layer readable in the
            // overview. Start with close projected spacing, then account for
            // perspective below instead of shrinking every card for a large
            // uniform gap.
            const double ce = std::cos(default_elevation), se = std::sin(default_elevation);
            const double ca = std::cos(default_azimuth), sa = std::sin(default_azimuth);
            const double previous_extent = previous.bounds.height * ce + previous.bounds.width * se * ca;
            const double current_extent = used_height * ce + used_width * se * ca;
            z = previous.z + (0.5 * (previous_extent + current_extent) + 0.25) / (se * sa);
        }
        for (size_t i = begin; i < scene.panels.size(); ++i) {
            Panel& panel = scene.panels[i];
            panel.center.x -= used_width * 0.5;
            panel.center.z = z;
            for (Vec3 point : corners(panel)) grow(scene, point, first);
        }
        scene.groups.push_back({bucket.label, bucket.layer_type, bucket.layer_index, z,
                                {-used_width * 0.5, -used_height * 0.5, used_width, used_height}});
    }

    // Perspective changes the apparent height of a plane along the stack.
    // Relax only colliding gaps in the standard overview, rather than using
    // the worst case as every layer's pitch. The reference height is modest;
    // larger windows inherit more separation while all element areas stay true.
    const Rect reference{0, 0, 600, 400};
    auto vertical_bounds = [&](const Group& group, const Camera& camera, double z_offset = 0.0) {
        Panel frame_panel;
        frame_panel.center = {group.bounds.x + group.bounds.width * 0.5,
                              group.bounds.y + group.bounds.height * 0.5, group.z + z_offset};
        frame_panel.width = group.bounds.width;
        frame_panel.height = group.bounds.height;
        std::array<double, 2> result{std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()};
        for (Vec3 corner : corners(frame_panel)) {
            const Projection p = project(corner, camera, reference.width / reference.height, reference);
            result[0] = std::min(result[0], p.y);
            result[1] = std::max(result[1], p.y);
        }
        return result;
    };
    for (int iteration = 0; iteration < 20 && scene.groups.size() > 1; ++iteration) {
        const Camera camera = fit(scene, reference.width / reference.height);
        double bottom = vertical_bounds(scene.groups.front(), camera)[1];
        bool changed = false;
        for (size_t i = 1; i < scene.groups.size(); ++i) {
            auto extent = vertical_bounds(scene.groups[i], camera);
            const double missing = bottom + 0.75 - extent[0];
            if (missing > 0.001) {
                const double rate = vertical_bounds(scene.groups[i], camera, 1.0)[0] - extent[0];
                const double shift = missing / std::max(0.01, rate) * 1.05;
                for (size_t j = i; j < scene.groups.size(); ++j) scene.groups[j].z += shift;
                for (Panel& panel : scene.panels)
                    if (panel.group >= static_cast<int>(i)) panel.center.z += shift;
                extent = vertical_bounds(scene.groups[i], camera);
                changed = true;
            }
            bottom = extent[1];
        }
        first = true;
        for (const Panel& panel : scene.panels) for (Vec3 point : corners(panel)) grow(scene, point, first);
        if (!changed) break;
    }
    first = true;
    for (const Panel& panel : scene.panels) for (Vec3 point : corners(panel)) grow(scene, point, first);
    return scene;
}

std::array<Vec3, 4> corners(const Panel& panel) {
    const double x = panel.width * 0.5, y = panel.height * 0.5;
    return {{{panel.center.x - x, panel.center.y + y, panel.center.z},
             {panel.center.x + x, panel.center.y + y, panel.center.z},
             {panel.center.x + x, panel.center.y - y, panel.center.z},
             {panel.center.x - x, panel.center.y - y, panel.center.z}}};
}

Camera fit(const Scene& scene, double aspect, double azimuth, double elevation) {
    if (scene.panels.empty()) return fit_points({{-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}}, aspect, azimuth, elevation);
    std::vector<Vec3> points;
    points.reserve(scene.panels.size() * 4);
    for (const Panel& panel : scene.panels) {
        const auto c = corners(panel);
        points.insert(points.end(), c.begin(), c.end());
    }
    return fit_points(points, aspect, azimuth, elevation);
}

Camera focus(const Scene& scene, size_t panel, double aspect, const Camera& previous) {
    if (panel >= scene.panels.size()) return valid(previous);
    const Panel& p = scene.panels[panel];
    const auto c = corners(p);
    return fit_points(std::vector<Vec3>(c.begin(), c.end()), aspect, previous.azimuth, previous.elevation);
}

Matrices matrices(const Camera& camera, double aspect) {
    const Frame f = frame(camera);
    Matrices result;
    result.view = {{static_cast<float>(f.right.x), static_cast<float>(f.up.x), static_cast<float>(-f.forward.x), 0,
                    static_cast<float>(f.right.y), static_cast<float>(f.up.y), static_cast<float>(-f.forward.y), 0,
                    static_cast<float>(f.right.z), static_cast<float>(f.up.z), static_cast<float>(-f.forward.z), 0,
                    static_cast<float>(-dot(f.right, f.eye)), static_cast<float>(-dot(f.up, f.eye)),
                    static_cast<float>(dot(f.forward, f.eye)), 1}};
    const double scale = 1.0 / std::tan(half_fov);
    result.projection[0] = static_cast<float>(scale / valid_aspect(aspect));
    result.projection[5] = static_cast<float>(scale);
    result.projection[10] = static_cast<float>(f.far_z / (f.near_z - f.far_z));
    result.projection[11] = -1;
    result.projection[14] = static_cast<float>(f.near_z * f.far_z / (f.near_z - f.far_z));
    return result;
}

Projection project(Vec3 point, const Camera& camera, double aspect, Rect viewport) {
    Projection result;
    if (!finite(point) || !(viewport.width > 0 && viewport.height > 0)) return result;
    const Frame f = frame(camera);
    const Vec3 relative = sub(point, f.eye);
    const double z = dot(relative, f.forward);
    if (!(z > 0)) return result;
    const double nx = dot(relative, f.right) / (z * std::tan(half_fov) * valid_aspect(aspect));
    const double ny = dot(relative, f.up) / (z * std::tan(half_fov));
    result.x = viewport.x + (nx * 0.5 + 0.5) * viewport.width;
    result.y = viewport.y + (0.5 - ny * 0.5) * viewport.height;
    result.depth = f.far_z / (f.far_z - f.near_z) -
                   f.near_z * f.far_z / ((f.far_z - f.near_z) * z);
    result.visible = z >= f.near_z && z <= f.far_z &&
                     nx >= -1 - epsilon && nx <= 1 + epsilon && ny >= -1 - epsilon && ny <= 1 + epsilon;
    return result;
}

Hit hit_test(const Scene& scene, const Camera& camera, Rect viewport, double x, double y) {
    Hit best;
    if (!inside(viewport, x, y)) return best;
    const Frame f = frame(camera);
    const double aspect = viewport.width / viewport.height;
    const double nx = ((x - viewport.x) / viewport.width * 2 - 1) * std::tan(half_fov) * aspect;
    const double ny = (1 - (y - viewport.y) / viewport.height * 2) * std::tan(half_fov);
    const Vec3 ray = normalized(add(f.forward, add(mul(f.right, nx), mul(f.up, ny))));
    if (std::fabs(ray.z) < epsilon) return best;
    for (size_t i = 0; i < scene.panels.size(); ++i) {
        const Panel& panel = scene.panels[i];
        if (!(panel.width > 0 && panel.height > 0)) continue;
        const double distance = (panel.center.z - f.eye.z) / ray.z;
        if (!(distance > 0) || (best.panel >= 0 && distance >= best.distance)) continue;
        const Vec3 point = add(f.eye, mul(ray, distance));
        const double u = (point.x - (panel.center.x - panel.width * 0.5)) / panel.width;
        const double v = (panel.center.y + panel.height * 0.5 - point.y) / panel.height;
        if (u < -epsilon || u > 1 + epsilon || v < -epsilon || v > 1 + epsilon) continue;
        if (!project(point, camera, aspect, viewport).visible) continue;
        best = {static_cast<int>(i), distance, std::clamp(u, 0.0, 1.0), std::clamp(v, 0.0, 1.0)};
    }
    if (best.panel >= 0) return best;

    // True one-element strips are often thinner than one screen pixel. Give
    // only such narrow projected panels a small pointer tolerance, preserving
    // their real dimensions and giving exact polygon hits priority.
    double best_screen_distance = 4.0;
    for (size_t i = 0; i < scene.panels.size(); ++i) {
        const Panel& panel = scene.panels[i];
        if (!(panel.width > 0 && panel.height > 0)) continue;
        const bool horizontal = panel.width >= panel.height;
        const Vec3 a = horizontal ? Vec3{panel.center.x - panel.width * 0.5, panel.center.y, panel.center.z}
                                  : Vec3{panel.center.x, panel.center.y + panel.height * 0.5, panel.center.z};
        const Vec3 b = horizontal ? Vec3{panel.center.x + panel.width * 0.5, panel.center.y, panel.center.z}
                                  : Vec3{panel.center.x, panel.center.y - panel.height * 0.5, panel.center.z};
        const Projection pa = project(a, camera, aspect, viewport), pb = project(b, camera, aspect, viewport);
        if (!pa.visible || !pb.visible) continue;
        const Vec3 short_a = horizontal ? Vec3{panel.center.x, panel.center.y - panel.height * 0.5, panel.center.z}
                                        : Vec3{panel.center.x - panel.width * 0.5, panel.center.y, panel.center.z};
        const Vec3 short_b = horizontal ? Vec3{panel.center.x, panel.center.y + panel.height * 0.5, panel.center.z}
                                        : Vec3{panel.center.x + panel.width * 0.5, panel.center.y, panel.center.z};
        const Projection sa = project(short_a, camera, aspect, viewport), sb = project(short_b, camera, aspect, viewport);
        if (std::hypot(sa.x - sb.x, sa.y - sb.y) > 3.0) continue;
        const double dx = pb.x - pa.x, dy = pb.y - pa.y, length_squared = dx * dx + dy * dy;
        const double t = length_squared > epsilon
            ? std::clamp(((x - pa.x) * dx + (y - pa.y) * dy) / length_squared, 0.0, 1.0) : 0.5;
        const double distance = std::hypot(x - (pa.x + t * dx), y - (pa.y + t * dy));
        if (distance > best_screen_distance) continue;
        const Vec3 point = add(a, mul(sub(b, a), t));
        const double depth_distance = length(sub(point, f.eye));
        if (distance == best_screen_distance && best.panel >= 0 && depth_distance >= best.distance) continue;
        best_screen_distance = distance;
        best = {static_cast<int>(i), depth_distance, horizontal ? t : 0.5, horizontal ? 0.5 : t};
    }
    return best;
}

Camera orbit(Camera camera, double dx, double dy) {
    camera = valid(camera);
    if (std::isfinite(dx)) camera.azimuth = std::remainder(camera.azimuth + dx * 0.01, pi * 2);
    if (std::isfinite(dy)) camera.elevation = std::clamp(camera.elevation + dy * 0.01, -1.55, 1.55);
    return camera;
}

Camera pan(Camera camera, double dx, double dy, double viewport_height) {
    camera = valid(camera);
    if (!(viewport_height > 0) || !std::isfinite(viewport_height) || !std::isfinite(dx) || !std::isfinite(dy)) return camera;
    const Frame f = frame(camera);
    const double scale = 2 * camera.distance * std::tan(half_fov) / viewport_height;
    camera.target = add(camera.target, add(mul(f.right, -dx * scale), mul(f.up, dy * scale)));
    return camera;
}

Camera zoom(Camera camera, double wheel) {
    camera = valid(camera);
    if (!std::isfinite(wheel)) return camera;
    const double factor = std::exp(std::clamp(-wheel * 0.12, -60.0, 60.0));
    camera.distance = std::clamp(camera.distance * factor,
                               std::max(1e-7, camera.radius * 1e-5), camera.radius * 1e5);
    return camera;
}
} // namespace caliper_live::model3d
