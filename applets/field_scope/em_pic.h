#pragma once
// ============================================================================
// FieldScope PIC core — a self-consistent electrostatic Particle-In-Cell step
// for a FREE-FLOATING neutral plasma (pure torch, host-free; unit-tested in
// tests/test_em_pic.cpp). No periodic box-fill, no cube: the cloud is a neutral
// mix of + and - charges floating in open space, held near the centre by a soft
// harmonic trap (a Penning/Paul-trap-like confinement). The particles generate
// the field that pushes them, so real collective behaviour emerges.
//
// One step:
//   rho   = deposit_cic(pos, charge)   signed charge -> grid (Cloud-In-Cell)
//   E     = poisson_E(rho * coupling)  self-field by FFT Poisson
//   Efld  = gather_cic(E, pos)         field -> particles
//   accel = charge*Efld + trap(pos)    charge-dependent E-force + trap
//   boris_push(pos, vel, accel, charge, B)   Boris integrator (per-charge)
//
// The FFT Poisson solve runs on CPU (grid ~32^3, tiny) regardless of the
// particle device — robust across backends (incl. MPS) and cheap next to the
// 1e5-particle scatter/gather that stay on the GPU. The particle tensors the
// renderer reads zero-copy never move. Solver is periodic; with the cloud kept
// small relative to L the periodic images are distant and negligible.
// ============================================================================
#include <torch/torch.h>

#include <cmath>
#include <tuple>

namespace fieldscope {

// ---- charge deposition: signed charge -> grid (Cloud-In-Cell) --------------
// pos (N,3) in [0,L). Optional charge (N,) weights each particle (default +1).
// CIC weights partition unity, so total deposited charge == sum(charge).
inline torch::Tensor deposit_cic(const torch::Tensor& pos, int64_t G, double L,
                                 const torch::Tensor& charge = torch::Tensor()) {
    const double dx = L / (double)G;
    auto gpos = pos / dx;
    auto base = torch::floor(gpos);
    auto frac = gpos - base;
    auto b = base.to(torch::kLong);
    auto rho = torch::zeros({G * G * G},
                            torch::TensorOptions(pos.device()).dtype(torch::kFloat32));
    auto fx = frac.select(1, 0), fy = frac.select(1, 1), fz = frac.select(1, 2);
    auto bx = b.select(1, 0),   by = b.select(1, 1),   bz = b.select(1, 2);
    for (int cx = 0; cx < 2; ++cx)
        for (int cy = 0; cy < 2; ++cy)
            for (int cz = 0; cz < 2; ++cz) {
                auto wx = cx ? fx : (1 - fx);
                auto wy = cy ? fy : (1 - fy);
                auto wz = cz ? fz : (1 - fz);
                auto w  = wx * wy * wz;                          // (N,)
                if (charge.defined()) w = w * charge;
                auto ix = (bx + cx).remainder(G);
                auto iy = (by + cy).remainder(G);
                auto iz = (bz + cz).remainder(G);
                auto flat = (ix * G + iy) * G + iz;
                rho.index_add_(0, flat, w);
            }
    return rho.reshape({G, G, G});
}

// ---- Poisson solve: source (G,G,G) -> E = -grad phi, div^2 phi = -source ----
// Spectral: phi_k = source_k/|k|^2, E_k = -i k phi_k (k=0 -> 0). CPU FFT (see
// header); returns E (G,G,G,3) on the source's device.
inline torch::Tensor poisson_E(const torch::Tensor& source, double L) {
    const int64_t G = source.size(0);
    const auto dev = source.device();
    auto src = source.to(torch::kCPU, torch::kFloat32);

    auto freq = torch::fft::fftfreq(G, /*d=*/L / (double)G);
    auto k1 = freq * (2.0 * M_PI);
    auto kx = k1.reshape({G, 1, 1});
    auto ky = k1.reshape({1, G, 1});
    auto kz = k1.reshape({1, 1, G});
    auto k2 = kx * kx + ky * ky + kz * kz;
    k2 = torch::where(k2 == 0, torch::full_like(k2, 1e30), k2);

    auto sk  = torch::fft::fftn(src);
    auto phk = sk / k2;

    auto KX = kx.expand({G, G, G}).contiguous();
    auto KY = ky.expand({G, G, G}).contiguous();
    auto KZ = kz.expand({G, G, G}).contiguous();
    auto z  = torch::zeros({G, G, G}, torch::kFloat32);
    auto Ex = torch::real(torch::fft::ifftn(torch::complex(z, -KX) * phk));
    auto Ey = torch::real(torch::fft::ifftn(torch::complex(z, -KY) * phk));
    auto Ez = torch::real(torch::fft::ifftn(torch::complex(z, -KZ) * phk));
    return torch::stack({Ex, Ey, Ez}, -1).to(dev);
}

// ---- FREE-SPACE (open boundary) Poisson via Hockney's doubled-grid method --
// The periodic solver above tiles space (a cube); this one solves for an
// ISOLATED charge distribution — the field of the cloud alone, decaying to zero
// at infinity — so the cloud floats freely. phi = rho (*) G_free, the linear
// convolution with the free-space Green's function G_free(r)=1/(4*pi*r),
// evaluated as a cyclic convolution on a 2G grid with the source zero-padded
// into one octant (Hockney/Eastwood). rho is charge-per-cell; the self-cell
// gets a softened Green value. Cached FFT of the Green's function (grid fixed).
inline torch::Tensor poisson_phi_free(const torch::Tensor& rho, double L) {
    using torch::indexing::Slice;
    const int64_t G = rho.size(0), M = 2 * G;
    const double dx = L / (double)G;
    const auto dev = rho.device();

    static int64_t cG = -1; static double cL = -1; static torch::Tensor ghat;
    if (G != cG || L != cL) {
        auto idx = torch::arange(M, torch::kFloat);
        idx = torch::where(idx > (float)G, idx - (float)M, idx);   // [-G+1..G]
        auto l = idx * dx;
        auto lx = l.reshape({M, 1, 1}), ly = l.reshape({1, M, 1}),
             lz = l.reshape({1, 1, M});
        auto R = torch::sqrt(lx * lx + ly * ly + lz * lz);
        auto g = 1.0 / (4.0 * M_PI * R);
        g.index_put_({0, 0, 0}, 1.0 / (4.0 * M_PI * 0.5 * dx));    // self term
        ghat = torch::fft::fftn(g);
        cG = G; cL = L;
    }
    auto rp = torch::zeros({M, M, M}, torch::kFloat);
    rp.index_put_({Slice(0, G), Slice(0, G), Slice(0, G)},
                  rho.to(torch::kCPU, torch::kFloat));
    auto phif = torch::real(torch::fft::ifftn(torch::fft::fftn(rp) * ghat));
    return phif.index({Slice(0, G), Slice(0, G), Slice(0, G)}).contiguous().to(dev);
}

// Free-space field E = -grad phi (central differences; the cloud stays well
// inside the grid, so the domain edges are never exercised).
inline torch::Tensor poisson_E_free(const torch::Tensor& rho, double L) {
    const int64_t G = rho.size(0);
    const double dx = L / (double)G;
    auto phi = poisson_phi_free(rho, L);
    auto Ex = -(torch::roll(phi, -1, 0) - torch::roll(phi, 1, 0)) / (2 * dx);
    auto Ey = -(torch::roll(phi, -1, 1) - torch::roll(phi, 1, 1)) / (2 * dx);
    auto Ez = -(torch::roll(phi, -1, 2) - torch::roll(phi, 1, 2)) / (2 * dx);
    return torch::stack({Ex, Ey, Ez}, -1);
}

// ---- field gather: grid -> particles (same CIC weights) --------------------
inline torch::Tensor gather_cic(const torch::Tensor& E, const torch::Tensor& pos,
                                double L) {
    const int64_t G = E.size(0);
    const double dx = L / (double)G;
    auto gpos = pos / dx;
    auto base = torch::floor(gpos);
    auto frac = gpos - base;
    auto b = base.to(torch::kLong);
    auto Eflat = E.reshape({G * G * G, 3});
    auto out = torch::zeros({pos.size(0), 3}, pos.options());
    auto fx = frac.select(1, 0), fy = frac.select(1, 1), fz = frac.select(1, 2);
    auto bx = b.select(1, 0),   by = b.select(1, 1),   bz = b.select(1, 2);
    for (int cx = 0; cx < 2; ++cx)
        for (int cy = 0; cy < 2; ++cy)
            for (int cz = 0; cz < 2; ++cz) {
                auto wx = cx ? fx : (1 - fx);
                auto wy = cy ? fy : (1 - fy);
                auto wz = cz ? fz : (1 - fz);
                auto w  = (wx * wy * wz).unsqueeze(1);
                auto ix = (bx + cx).remainder(G);
                auto iy = (by + cy).remainder(G);
                auto iz = (bz + cz).remainder(G);
                auto flat = (ix * G + iy) * G + iz;
                out = out + w * Eflat.index_select(0, flat);
            }
    return out;
}

// ---- Boris pusher: half accel-kick, magnetic rotation, half accel-kick -----
// `accel` (N,3) is the full charge-INCLUSIVE acceleration (charge*E + trap);
// `charge` (N,1) drives only the magnetic rotation (opposite charges gyrate
// opposite ways). vel advanced in place; p_out = p_in + vel*dt (open — no wrap).
inline void boris_push(const torch::Tensor& p_in, torch::Tensor& p_out,
                       torch::Tensor& vel, const torch::Tensor& accel,
                       const torch::Tensor& charge, const torch::Tensor& B,
                       double dt) {
    const double h = 0.5 * dt;
    vel.add_(accel, h);                                   // half kick
    // B may be uniform (3,)/(1,3) or a per-particle field (N,3); reshape to
    // (-1,3) so it broadcasts against the (N,1) charge either way.
    auto t  = B.reshape({-1, 3}) * (charge * (0.5 * dt)); // (N,3) per particle
    auto t2 = (t * t).sum(1, /*keepdim=*/true);
    auto s  = t * (2.0 / (1.0 + t2));
    auto vp = vel + torch::linalg_cross(vel, t, /*dim=*/1);
    vel.copy_(vel + torch::linalg_cross(vp, s, /*dim=*/1));
    vel.add_(accel, h);                                   // half kick
    p_out.copy_(p_in);
    p_out.add_(vel, dt);
}

// ---- initial conditions (free-floating, centred at L/2) --------------------
// Quasi-2-D: a thin slab in z (the plane ⊥ B, where the magnetized E×B "2-D
// vortex" dynamics live) with structured xy layouts. Single species (all +1);
// a strong background B does the transverse confinement, a z-trap the axial.
enum class IC { kClumps = 0, kRing = 1, kBlob = 2, kTwoStream = 3 };

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
init_state(IC kind, int64_t N, double L, double temp, double v0,
           torch::Device dev) {
    auto o = torch::TensorOptions(dev).dtype(torch::kFloat32);
    const float c = 0.5f * (float)L;
    const float R  = 0.15f * (float)L;         // ring / clump-ring radius
    const float zt = 0.07f * (float)L;         // slab half-thickness in z
    const float pi = (float)M_PI;
    torch::Tensor pos, vel;
    auto charge = torch::ones({N, 1}, o);
    auto zslab = [&] { return c + torch::randn({N}, o) * zt; };

    switch (kind) {
        case IC::kClumps: {
            // K compact Gaussian vortices on a circle -> they orbit and MERGE
            // under B (compact enough to resist shear phase-mixing into a ring).
            const int K = 4;
            auto cl  = torch::randint(0, K, {N}, torch::TensorOptions(dev).dtype(torch::kLong));
            auto ang = cl.to(torch::kFloat32) * (2.f * pi / (float)K);
            auto cx  = R * torch::cos(ang) + c, cy = R * torch::sin(ang) + c;
            auto x = cx + torch::randn({N}, o) * (0.18f * R);
            auto y = cy + torch::randn({N}, o) * (0.18f * R);
            pos = torch::stack({x, y, zslab()}, 1);
            vel = torch::randn({N, 3}, o) * temp;
            break;
        }
        case IC::kRing: {
            // Hollow annulus -> diocotron instability breaks it into a rotating
            // necklace of vortices.
            auto th = torch::rand({N}, o) * (2.f * pi);
            auto rr = R + torch::randn({N}, o) * (0.10f * R);   // thin annulus
            auto x = rr * torch::cos(th) + c, y = rr * torch::sin(th) + c;
            pos = torch::stack({x, y, zslab()}, 1);
            vel = torch::randn({N, 3}, o) * temp;
            break;
        }
        case IC::kTwoStream: {
            auto x = c + torch::randn({N}, o) * (0.18f * (float)L);
            auto y = c + torch::randn({N}, o) * (0.06f * (float)L);
            pos = torch::stack({x, y, zslab()}, 1);
            vel = torch::randn({N, 3}, o) * temp;
            auto sign = torch::ones({N}, o);
            sign.narrow(0, N / 2, N - N / 2).mul_(-1.f);        // counter-stream
            vel.select(1, 1).add_(sign * (float)v0);
            break;
        }
        case IC::kBlob: default: {
            auto x = c + torch::randn({N}, o) * (0.10f * (float)L);
            auto y = c + torch::randn({N}, o) * (0.10f * (float)L);
            pos = torch::stack({x, y, zslab()}, 1);
            vel = torch::randn({N, 3}, o) * temp;
            break;
        }
    }
    return {pos, vel, charge};
}

// ---- prescribed external field geometries (test-particle observation) ------
// A charged particle in a static B conserves |v| (Boris is exact for the
// rotation), so these modes are unconditionally stable — no self-field, no
// trap; the field geometry alone shapes the motion. Watch: gyration (uniform),
// mirroring/trapping between field maxima (mirror = a finite solenoid/bottle),
// toroidal streaming + grad-B drift (toroidal), and belt trapping (dipole).
enum class Field { kPlasma = 0, kUniform = 1, kMirror = 2, kToroidal = 3, kDipole = 4 };

// Analytic B(x) at every particle, centred at `c`. `B0` is the field strength.
inline torch::Tensor external_B(const torch::Tensor& pos, Field mode,
                                float c, float B0, float L) {
    auto X = pos - c;
    auto x = X.select(1, 0), y = X.select(1, 1), z = X.select(1, 2);
    const float eps = 1e-3f;
    switch (mode) {
        case Field::kMirror: {                       // magnetic bottle (∇·B=0)
            const float zc = 0.32f * L;
            auto bz  = B0 * (1.f + (z * z) / (zc * zc));
            auto brr = -B0 * z / (zc * zc);          // B_r / r  (from ∇·B=0)
            return torch::stack({brr * x, brr * y, bz}, 1);
        }
        case Field::kToroidal: {                     // tokamak: toroidal ~1/R
            const float R0 = 0.25f * L;              // + poloidal twist so the
            auto rxy = torch::sqrt(x * x + y * y + eps);   // field lines wind and
            auto fac = B0 * R0 / (rxy * rxy);              // orbits stay confined
            auto bt_x = -fac * y, bt_y = fac * x;    // toroidal (φ̂)
            // poloidal: circles the minor cross-section (ρ_R, ρ_z)=(rxy-R0, z).
            auto rhoR = rxy - R0;
            auto rho  = torch::sqrt(rhoR * rhoR + z * z + eps);
            auto bpol = (0.35f * B0) / rho;          // poloidal magnitude
            auto Rx = x / rxy, Ry = y / rxy;         // R̂ in xy
            auto bp_x = bpol * (-z) * Rx, bp_y = bpol * (-z) * Ry, bp_z = bpol * rhoR;
            return torch::stack({bt_x + bp_x, bt_y + bp_y, bp_z}, 1);
        }
        case Field::kDipole: {                       // magnetic dipole along +z
            auto r2    = x * x + y * y + z * z + eps;
            auto invr5 = torch::pow(r2, -2.5f);
            const float k = 6.f * B0 * std::pow(0.28f * L, 3.f);
            return torch::stack({k * 3.f * z * x * invr5,
                                 k * 3.f * z * y * invr5,
                                 k * (3.f * z * z - r2) * invr5}, 1);
        }
        case Field::kUniform: default:               // uniform axial -> helices
            return torch::stack({torch::zeros_like(x), torch::zeros_like(x),
                                 B0 + torch::zeros_like(x)}, 1);
    }
}

// Seed positions + velocities so the characteristic motion is visible. `vth`
// sets the speed (gyroradius ≈ vth/B0). charge is +1 throughout.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
init_field(Field mode, int64_t N, double L, double vth, torch::Device dev) {
    auto o = torch::TensorOptions(dev).dtype(torch::kFloat32);
    const float c = 0.5f * (float)L;
    const float pi = (float)M_PI;
    auto charge = torch::ones({N, 1}, o);
    torch::Tensor pos, vel;
    switch (mode) {
        case Field::kMirror: {
            pos = c + torch::randn({N, 3}, o) * (0.06f * (float)L);
            vel = torch::randn({N, 3}, o) * (float)vth;   // isotropic -> mixed pitch
            break;
        }
        case Field::kToroidal: {
            auto th = torch::rand({N}, o) * (2.f * pi);
            auto rr = 0.25f * (float)L + torch::randn({N}, o) * (0.05f * (float)L);
            auto x = rr * torch::cos(th) + c, y = rr * torch::sin(th) + c;
            pos = torch::stack({x, y, c + torch::randn({N}, o) * (0.05f * (float)L)}, 1);
            // Mostly stream along the field (φ̂) + a perpendicular gyration part.
            auto phx = -torch::sin(th), phy = torch::cos(th);
            vel = torch::randn({N, 3}, o) * (0.5f * (float)vth);
            vel.select(1, 0).add_(phx * (float)vth);
            vel.select(1, 1).add_(phy * (float)vth);
            break;
        }
        case Field::kDipole: {
            auto th = torch::rand({N}, o) * (2.f * pi);
            auto rd = 0.28f * (float)L + torch::randn({N}, o) * (0.03f * (float)L);
            auto x = rd * torch::cos(th) + c, y = rd * torch::sin(th) + c;
            pos = torch::stack({x, y, c + torch::randn({N}, o) * (0.03f * (float)L)}, 1);
            vel = torch::randn({N, 3}, o) * (float)vth;   // isotropic -> belts
            break;
        }
        case Field::kUniform: default: {
            pos = c + torch::randn({N, 3}, o) * (0.12f * (float)L);
            vel = torch::randn({N, 3}, o) * (float)vth;
            vel.select(1, 2).mul_(0.f);                   // v∥=0 -> pure gyration, no drift
            break;
        }
    }
    return {pos, vel, charge};
}

}  // namespace fieldscope
