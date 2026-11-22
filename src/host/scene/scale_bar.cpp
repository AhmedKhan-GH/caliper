// scale_bar.cpp: the scale bar of a scene view (scale_bar.h).
#include "scene/scale_bar.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <variant>

namespace caliper_scene {

namespace {

// 10^n, exact for 0 <= n <= 22, where every power of ten is a double.
double pow10(int n) {
    if (n < 0 || n > 22) return std::pow(10.0, double(n));
    double p = 1.0;
    for (int i = 0; i < n; ++i) p *= 10.0;
    return p;
}

// m x 10^k as the double nearest to it: a negative power divides by an exact one.
double scaled(int m, int k) { return k >= 0 ? double(m) * pow10(k) : double(m) / pow10(-k); }

std::string trimmed(const char* s) {
    std::string t = s;
    if (t.find('.') != std::string::npos) {
        while (!t.empty() && t.back() == '0') t.pop_back();
        if (!t.empty() && t.back() == '.') t.pop_back();
    }
    return t;
}

// Plain decimals from 0.001 to 999,999 ("0.5", "200000"), "5e-7" and "2e6" outside them.
std::string number_text(double x) {
    char b[64];
    if (x >= 0.001 && x < 1e6) {
        std::snprintf(b, sizeof b, "%.6f", x);
        return trimmed(b);
    }
    int e = int(std::floor(std::log10(x)));
    double mant = e >= 0 ? x / pow10(e) : x * pow10(-e);
    std::snprintf(b, sizeof b, "%.5f", mant);
    if (std::atof(b) >= 10.0) {   // rounded up to the next power
        ++e;
        mant = e >= 0 ? x / pow10(e) : x * pow10(-e);
        std::snprintf(b, sizeof b, "%.5f", mant);
    }
    return trimmed(b) + "e" + std::to_string(e);
}

// A unit the label converts: metres, seconds or grams, with an optional prefix.
struct KnownUnit {
    bool known = false;
    double factor = 1.0;     // of the base unit
    std::string base;
    std::string shown;       // the unit as the label writes it ("u" as "µ")
};

constexpr const char* kMicro = "\xC2\xB5";        // U+00B5 MICRO SIGN
constexpr const char* kGreekMu = "\xCE\xBC";      // U+03BC, accepted as micro

KnownUnit known_unit(std::string_view u) {
    KnownUnit k;
    if (u.empty()) return k;
    const char last = u.back();
    if (last != 'm' && last != 's' && last != 'g') return k;
    const std::string_view prefix = u.substr(0, u.size() - 1);
    k.base = std::string(1, last);
    if (prefix.empty()) k.factor = 1.0;
    else if (prefix == "n") k.factor = 1e-9;
    else if (prefix == "u" || prefix == kMicro || prefix == kGreekMu) k.factor = 1e-6;
    else if (prefix == "m") k.factor = 1e-3;
    else if (prefix == "c") k.factor = 1e-2;
    else if (prefix == "k") k.factor = 1e3;
    else return k;
    k.known = true;
    const bool micro = prefix == "u" || prefix == kMicro || prefix == kGreekMu;
    k.shown = (micro ? std::string(kMicro) : std::string(prefix)) + k.base;
    return k;
}

}  // namespace

double scale_bar_length(double units_per_px, double width_px) {
    if (!(units_per_px > 0) || !(width_px > 0) || !std::isfinite(units_per_px) || !std::isfinite(width_px)) return 0;
    const double most = units_per_px * width_px * kScaleBarShare;
    if (!(most > 0) || !std::isfinite(most)) return 0;
    int k = int(std::floor(std::log10(most)));
    if (scaled(1, k) > most) --k;              // log10 may land one off beside a power of ten
    else if (scaled(1, k + 1) <= most) ++k;
    for (int m : {5, 2, 1}) {
        const double c = scaled(m, k);
        if (c <= most) return c;
    }
    return scaled(1, k);
}

std::string scale_bar_label(double length, std::string_view units, bool at_target) {
    const std::string tail = at_target ? " at target" : "";
    if (!(length > 0) || !std::isfinite(length)) return "";
    if (units.empty() || units == "arbitrary" || units == "a.u.") return number_text(length) + tail;
    const KnownUnit k = known_unit(units);
    if (!k.known) return number_text(length) + " " + std::string(units) + tail;
    if (length >= 0.01 && length < 1000.0) return number_text(length) + " " + k.shown + tail;
    // Through the base unit to the prefix, in steps of 1,000, that brings the number between 1
    // and 1,000; past n or k the extreme prefix stays.
    const double base = length * k.factor;
    static const struct { int e; const char* prefix; } kPrefixes[] = {
        {3, "k"}, {0, ""}, {-3, "m"}, {-6, kMicro}, {-9, "n"}};
    int e = -9;
    const char* prefix = "n";
    for (const auto& p : kPrefixes)
        if (base >= scaled(1, p.e) * (1.0 - 1e-12)) {
            e = p.e;
            prefix = p.prefix;
            break;
        }
    const double n = e >= 0 ? base / pow10(e) : base * pow10(-e);
    return number_text(n) + " " + prefix + k.base + tail;
}

double scale_bar_units_per_pixel(const ViewCamera& cam, double height_px) {
    if (!(height_px > 0)) return 0;
    if (const Orbit* o = std::get_if<Orbit>(&cam)) return 2.0 * o->distance * std::tan(kOrbitFovY * 0.5) / height_px;
    if (const PanZoom* p = std::get_if<PanZoom>(&cam)) return 2.0 * p->half_height / height_px;
    return 0;   // through a pinhole: no single depth
}

ScaleBar scale_bar(const ViewCamera& cam, bool has_bounds, std::string_view units, double width_px,
                   double height_px) {
    ScaleBar b;
    if (!has_bounds || !(width_px >= kScaleBarMinPicture) || !(height_px > 0)) return b;
    if (std::holds_alternative<ThroughPinhole>(cam)) return b;
    const double upp = scale_bar_units_per_pixel(cam, height_px);
    const double len = scale_bar_length(upp, width_px);
    if (!(len > 0)) return b;
    b.visible = true;
    b.length = len;
    b.pixels = len / upp;
    b.label = scale_bar_label(len, units, std::holds_alternative<Orbit>(cam));
    return b;
}

}  // namespace caliper_scene
