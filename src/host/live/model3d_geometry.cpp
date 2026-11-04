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
constexpr double layer_pitch = 4.0;
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
    if (!std::isfinite(camera.azimuth)) camera.azimuth = 1.1;
    if (!std::isfinite(camera.elevation)) camera.elevation = 0.35;
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
Camera fit_bounds(Vec3 minimum, Vec3 maximum, double aspect, double azimuth, double elevation) {
    Camera camera;
    camera.target = mul(add(minimum, maximum), 0.5);
    camera.radius = std::max(length(sub(maximum, minimum)) * 0.5, element_unit);
    camera.azimuth = azimuth;
    camera.elevation = elevation;
    const double horizontal_half_fov = std::atan(std::tan(half_fov) * valid_aspect(aspect));
    const double limiting_half_fov = std::min(half_fov, horizontal_half_fov);
    // A sphere around the bounding box fits for every orbit orientation. The
    // margin keeps both the panels and their projected outlines off the edge.
    camera.distance = camera.radius / std::sin(limiting_half_fov) * 1.08;
    return valid(camera);
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
    std::vector<std::pair<size_t, bool>> tensors;
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

std::array<double, 2> panel_size(const Tensor& tensor) {
    if (tensor.shape.size() == 1) {
        return tensor.shape.front() > 0
            ? std::array<double, 2>{tensor.shape.front() * element_unit, element_unit}
            : std::array<double, 2>{0, 0};
    }
    double rows = static_cast<double>(tensor.rows), cols = static_cast<double>(tensor.cols);
    if ((rows <= 0 || cols <= 0) && !tensor.shape.empty()) {
        rows = 1;
        for (size_t i = 0; i + 1 < tensor.shape.size(); ++i) {
            if (tensor.shape[i] <= 0) return {0, 0};
            rows *= static_cast<double>(tensor.shape[i]);
        }
        cols = static_cast<double>(tensor.shape.back());
    }
    if (!(rows > 0 && cols > 0) || !std::isfinite(rows) || !std::isfinite(cols)) return {0, 0};
    return {cols * element_unit, rows * element_unit};
}

Scene layout(const std::vector<Tensor>& tensors, const std::string& output_label) {
    std::map<Key, Bucket> buckets;
    size_t tied_embedding = tensors.size();
    bool have_output = false;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& tensor = tensors[i];
        const auto size = panel_size(tensor);
        if (!(size[0] > 0 && size[1] > 0)) continue;
        const int layer = layer_of(tensor);
        Key key{1, 0};
        std::string label = tensor.group.empty() ? "Model" : tensor.group;
        if (embedding(tensor)) {
            key = {0, 0}; label = "Embedding";
            if (tensor.tied) tied_embedding = i;
        } else if (layer >= 0) {
            key = {2, layer};
            std::ostringstream name;
            name << "Layer " << std::setw(2) << std::setfill('0') << layer;
            label = name.str();
        } else if (final_norm(tensor)) {
            key = {3, 0}; label = "Final norm";
        } else if (output(tensor)) {
            key = {4, 0}; label = "Output"; have_output = true;
        }
        Bucket& bucket = buckets[key];
        bucket.label = label;
        bucket.layer_index = layer;
        if (bucket.layer_type.empty()) bucket.layer_type = type_of(tensor);
        bucket.tensors.emplace_back(i, false);
    }
    if (tied_embedding < tensors.size() && !have_output) {
        Bucket& bucket = buckets[{4, 0}];
        bucket.label = output_label;
        bucket.tensors.emplace_back(tied_embedding, true);
    }

    Scene scene;
    bool first = true;
    for (auto& entry : buckets) {
        Bucket& bucket = entry.second;
        std::sort(bucket.tensors.begin(), bucket.tensors.end(), [&](auto a, auto b) {
            return tensors[a.first].name < tensors[b.first].name;
        });
        double area = 0, max_width = 0;
        for (auto item : bucket.tensors) {
            const auto size = panel_size(tensors[item.first]);
            area += size[0] * size[1];
            max_width = std::max(max_width, size[0]);
        }
        const double shelf_width = std::max(max_width, std::sqrt(area) * 1.35);
        const int group_index = static_cast<int>(scene.groups.size());
        const double z = group_index * layer_pitch;
        const size_t begin = scene.panels.size();
        double x = 0, y = 0, row_height = 0, used_width = 0;
        for (auto item : bucket.tensors) {
            const Tensor& tensor = tensors[item.first];
            const auto size = panel_size(tensor);
            if (x > 0 && x + size[0] > shelf_width) {
                x = 0; y += row_height + panel_gap; row_height = 0;
            }
            Panel panel;
            panel.tensor = item.first;
            panel.label = item.second ? output_label : tensor.name;
            panel.center = {x + size[0] * 0.5, -(y + size[1] * 0.5), z};
            panel.width = size[0]; panel.height = size[1];
            panel.group = group_index; panel.output_alias = item.second;
            scene.panels.push_back(std::move(panel));
            used_width = std::max(used_width, x + size[0]);
            row_height = std::max(row_height, size[1]);
            x += size[0] + panel_gap;
        }
        const double used_height = y + row_height;
        for (size_t i = begin; i < scene.panels.size(); ++i) {
            Panel& panel = scene.panels[i];
            panel.center.x -= used_width * 0.5;
            panel.center.y += used_height * 0.5;
            for (Vec3 point : corners(panel)) grow(scene, point, first);
        }
        scene.groups.push_back({bucket.label, bucket.layer_type, bucket.layer_index, z,
                                {-used_width * 0.5, -used_height * 0.5, used_width, used_height}});
    }
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
    if (scene.panels.empty()) return fit_bounds({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}, aspect, azimuth, elevation);
    return fit_bounds(scene.minimum, scene.maximum, aspect, azimuth, elevation);
}

Camera focus(const Scene& scene, size_t panel, double aspect, const Camera& previous) {
    if (panel >= scene.panels.size()) return valid(previous);
    const Panel& p = scene.panels[panel];
    return fit_bounds({p.center.x - p.width * 0.5, p.center.y - p.height * 0.5, p.center.z},
                      {p.center.x + p.width * 0.5, p.center.y + p.height * 0.5, p.center.z},
                      aspect, previous.azimuth, previous.elevation);
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
