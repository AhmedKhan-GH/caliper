"""The loss landscape of demo 0: the loss of a model on a fixed batch, on a
grid around its current weights, over the plane of the two directions in
which the weights have moved most (the default), or over two
filter-normalised random directions (the switch), with the optimizer's path
in that plane.

The weights that move are every parameter with two or more dimensions: the
weight matrices and the token and position embeddings of the demo's model.
Biases and normalisation gains and offsets keep their current values in the
surface and are left out of the snapshots and of both directions.

Everything runs on the model's device and nothing is read back to the host:
the snapshots of the path, the plane's fit (a Gram matrix of at most
``keep`` by ``keep`` and orthogonal iteration), the grid of losses
(``torch.func.functional_call`` under ``vmap``, one row of the grid per
call), the path's points and the mesh. The scene's frame is fixed: x and y
from -1 to 1 and the height from -2 to 1, the height being half the log10 of
the loss clipped to the decades from 1e-4 to 1e2, so that the box, the floor
grid and the camera never move.

    land = Landscape(model, eval_x[:128], eval_y[:128])
    for step in range(steps):
        ...                                   # one training step
        land.point(model, step)
        if land.due(step, every, directions):
            positions, values, share = land.refit(model, span, directions)
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

__all__ = ["Landscape", "LOG_MIN", "LOG_MAX", "BOX"]

LOG_MIN, LOG_MAX = -4.0, 2.0          # the decades of the loss that height and colour show
BOX = ((-1.0, 1.0), (-1.0, 1.0), (0.5 * LOG_MIN, 0.5 * LOG_MAX))   # x, y and height
FLOOR_LINES = 9                       # floor lines each way
FIT_ITERATIONS = 16                   # orthogonal iteration steps of a refit
SPAN_SCALE = 1.5                      # the farthest snapshot sits at 1 / 1.5 of the half-width
MIN_HALF_WIDTH = 1e-3                 # of the weights' norm, the path plane's smallest half-width
PATH, RANDOM = "path", "random"
_TINY = 1e-30


def _logits(out):
    """The logits of a forward pass that may return a tuple (the demo's model
    returns logits, embedding and attention)."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _log_loss(loss):
    """log10 of the loss, clipped to [LOG_MIN, LOG_MAX]; a loss that is not a
    number draws at the ceiling."""
    return torch.nan_to_num(torch.log10(loss), nan=LOG_MAX).clamp(LOG_MIN, LOG_MAX)


def _unit(v):
    return v / v.norm().clamp_min(_TINY)


def _orthonormal_columns(v):
    """Gram-Schmidt on the two columns of ``v`` (K, 2)."""
    a = _unit(v[:, 0])
    b = v[:, 1] - (a @ v[:, 1]) * a
    return torch.stack([a, _unit(b)], 1)


def _sign(x):
    """+1 or -1 as a tensor: the sign of ``x``, with 0 counted as positive."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


class Landscape:
    """The loss landscape of ``model`` on the batch ``x``, ``y``.

    ``grid`` is the number of grid points along each axis (odd, so that the
    current weights are the centre vertex). ``seed`` seeds the generator of
    the random directions, which are drawn once and filter-normalised against
    the weights as they are when the Landscape is made, before the first
    training step: the weights at step 0. A snapshot of the weights is kept
    every ``snapshot_every`` steps and at every refit, at most ``keep`` of
    them: when the buffer is full, every second snapshot is dropped and the
    interval doubles. ``loss`` is the loss of a batch, ``loss(logits, y)``.
    """

    def __init__(self, model, x, y, *, grid=41, seed=0, snapshot_every=25, keep=256,
                 loss=F.cross_entropy):
        if grid < 3 or grid % 2 == 0:
            raise ValueError(f"grid must be an odd number of at least 3, got {grid}")
        if keep < 4 or keep % 2:
            raise ValueError(f"keep must be an even number of at least 4, got {keep}")
        if snapshot_every < 1:
            raise ValueError(f"snapshot_every must be at least 1, got {snapshot_every}")
        named = [(n, p) for n, p in model.named_parameters() if p.ndim >= 2]
        if not named:
            raise ValueError("the model has no parameter with two or more dimensions to perturb")
        self.names = [n for n, _ in named]
        self._model, self._params = model, [p for _, p in named]
        self.shapes = [tuple(p.shape) for _, p in named]
        self.sizes = [p.numel() for _, p in named]
        self.size = sum(self.sizes)                 # P, the weights in the plane
        first = named[0][1]
        self.device, self.dtype = first.device, first.dtype
        self.x, self.y = x, y
        self.grid = grid
        self.loss = loss
        self.keep = keep
        self.snapshot_every = snapshot_every
        self._interval = snapshot_every

        dev, f32 = self.device, self.dtype
        half = grid // 2
        # -1 to 1 across the grid, with 0 exactly at the centre
        self.offsets = torch.arange(-half, half + 1, device=dev, dtype=f32) / half
        self.triangles = self._make_triangles(grid, dev)
        self.frame = self._make_frame(dev, f32)

        # The anchor and the random directions, fixed for the whole run.
        with torch.no_grad():
            self.anchor = self.flat(model).clone()
            gen = torch.Generator().manual_seed(seed)
            d = [torch.randn(self.size, generator=gen, dtype=torch.float32) for _ in range(2)]
            d = torch.stack(d).to(device=dev, dtype=f32)
            self.random_directions = self._filter_normalise(d, self.anchor)
            g = self.random_directions @ self.random_directions.T         # (2, 2)
            det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
            inverse = torch.stack([torch.stack([g[1, 1], -g[0, 1]]),
                                   torch.stack([-g[1, 0], g[0, 0]])]) / det
            self._random_proj = inverse @ self.random_directions          # least squares rows

        # The snapshots of the path, allocated at the first one.
        self._snaps = None                    # (keep, P)
        self._snap_heights = None             # (keep,)
        self.snapshot_steps: list[int] = []   # the step of each kept snapshot (host)

        # The plane of the latest refit.
        self.mode = None                      # PATH or RANDOM once a refit has run
        self.ready = False
        self.directions = None                # (2, P): u1, u2 or d1, d2
        self.centre = None                    # (P,): the weights at the refit, or the anchor
        self.half_width = None                # 0-d: the box's half-width in the mode's units
        self.eigenvalues = None               # (2,): the plane's eigenvalues, in matched order
        self.share = None                     # 0-d: the path's variance the plane holds
        self.span = None                      # the span of the latest refit (host)
        self.refits: list[tuple[int, str]] = []
        self._path_plane = None               # (2, P): the latest path plane, kept across modes
        self._proj = None                     # (2, P): weights to box units, with the centre

        # The path in box units, one row per point.
        self._path = torch.empty(1024, 3, device=dev, dtype=f32)
        self._n = 0
        self.step = -1                        # the step of the latest point; -1 before training
        self._height = None
        self.method = None                    # "vmap" or "loop", decided at the first surface

    # -- the fixed geometry ----------------------------------------------------
    @staticmethod
    def _make_triangles(grid, device):
        """Two triangles for each cell of the grid, vertex k = row * grid + column,
        counter-clockwise seen from above."""
        i = torch.arange(grid - 1, device=device)
        r, c = torch.meshgrid(i, i, indexing="ij")
        v0 = (r * grid + c).reshape(-1)
        return torch.cat([torch.stack([v0, v0 + 1, v0 + grid], 1),
                          torch.stack([v0 + 1, v0 + grid + 1, v0 + grid], 1)]).to(torch.int32)

    @staticmethod
    def _make_frame(device, dtype):
        """The 12 edges of the box and the floor grid's lines, as segments."""
        (x0, x1), (y0, y1), (z0, z1) = BOX
        corners = [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        edges = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        seg = [corners[a] + corners[b] for a, b in edges]
        for k in range(FLOOR_LINES):
            t = x0 + (x1 - x0) * k / (FLOOR_LINES - 1)
            seg.append([t, y0, z0, t, y1, z0])
            seg.append([x0, t, z0, x1, t, z0])
        return torch.tensor(seg, dtype=dtype, device=device).reshape(-1, 3)

    def _filter_normalise(self, d, weights):
        """Scale each row of each parameter of both directions (each filter of a
        convolution) to the norm of the same row of ``weights``."""
        out = []
        for k in range(d.shape[0]):
            parts = []
            for dp, wp, shape in zip(self.split(d[k]), self.split(weights), self.shapes):
                dr, wr = dp.reshape(shape[0], -1), wp.reshape(shape[0], -1)
                scale = wr.norm(dim=1, keepdim=True) / dr.norm(dim=1, keepdim=True).clamp_min(_TINY)
                parts.append((dr * scale).reshape(-1))
            out.append(torch.cat(parts))
        return torch.stack(out)

    # -- weights ---------------------------------------------------------------
    def flat(self, model):
        """The perturbed weights of ``model`` as one vector (P,) on its device."""
        if model is self._model:
            params = self._params
        else:
            named = dict(model.named_parameters())
            params = [named[n] for n in self.names]
        return torch.cat([p.detach().reshape(-1) for p in params])

    def split(self, flat):
        """Views of ``flat`` (P,) with the shape of each perturbed parameter."""
        return [t.view(s) for t, s in zip(flat.split(self.sizes), self.shapes)]

    def as_parameters(self, flat, model):
        """``flat`` (P,) as a tensor for every parameter of ``model``, by name:
        its entries for the weights in the plane and zeros for the biases and
        normalisation parameters, which no direction moves."""
        parts = dict(zip(self.names, self.split(flat)))
        return {n: parts[n] if n in parts else torch.zeros_like(p)
                for n, p in model.named_parameters()}

    def batch_loss(self, model):
        """The loss of the landscape batch at the model's current weights, a 0-d
        tensor on the device, by a plain forward pass without gradients."""
        with torch.no_grad():
            return self.loss(_logits(model(self.x)), self.y)

    # -- snapshots -------------------------------------------------------------
    @property
    def snapshots(self):
        """The kept snapshots (K, P), oldest first."""
        k = len(self.snapshot_steps)
        return self._snaps[:k] if self._snaps is not None else None

    @property
    def snapshot_heights(self):
        """The height of each kept snapshot (K,): half the clipped log10 loss."""
        k = len(self.snapshot_steps)
        return self._snap_heights[:k] if self._snap_heights is not None else None

    def _snapshot(self, theta, height, step):
        if self._snaps is None:
            self._snaps = torch.empty(self.keep, self.size, device=self.device, dtype=self.dtype)
            self._snap_heights = torch.empty(self.keep, device=self.device, dtype=self.dtype)
        k = len(self.snapshot_steps)
        if k == self.keep:                       # full: keep every second one, double the interval
            idx = torch.arange(0, k, 2, device=self.device)
            self._snaps[:k // 2] = self._snaps.index_select(0, idx)
            self._snap_heights[:k // 2] = self._snap_heights.index_select(0, idx)
            self.snapshot_steps = self.snapshot_steps[0::2]
            self._interval *= 2
            k //= 2
        self._snaps[k] = theta
        self._snap_heights[k] = height
        self.snapshot_steps.append(step)

    # -- the path --------------------------------------------------------------
    @property
    def path(self):
        """The path's points in box units (V, 3), oldest first."""
        return self._path[:self._n]

    @property
    def path_length(self) -> int:
        return self._n

    @property
    def current(self):
        """The current point (1, 3): the path's last point."""
        return self._path[self._n - 1:self._n]

    def _append(self, p):
        if self._n == self._path.shape[0]:
            grown = torch.empty(2 * self._n, 3, device=self.device, dtype=self.dtype)
            grown[:self._n] = self._path
            self._path = grown
        self._path[self._n] = p
        self._n += 1

    def _set_path(self, points):
        if points.shape[0] > self._path.shape[0]:
            self._path = torch.empty(2 * points.shape[0], 3, device=self.device, dtype=self.dtype)
        self._path[:points.shape[0]] = points
        self._n = points.shape[0]

    def coords(self, theta):
        """The coordinates (2,) of weights ``theta`` (P,) in box units on the
        latest refit's plane and centre."""
        return self._proj @ (theta - self.centre)

    # -- the calls of the training loop ----------------------------------------
    def point(self, model, step):
        """The path's point (3,) at the model's current weights, after the
        training step ``step``: its coordinates on the plane in box units and its
        height, half the clipped log10 loss of the landscape batch. Keeps a
        snapshot every ``snapshot_every`` steps (the interval doubles at each
        thinning) and, once a plane exists, appends the point to the path."""
        self.step = step
        with torch.no_grad():
            theta = self.flat(model)
            h = 0.5 * _log_loss(self.batch_loss(model))
            self._height = h
            if step % self._interval == 0 and (not self.snapshot_steps or
                                               self.snapshot_steps[-1] != step):
                self._snapshot(theta, h, step)
            if not self.ready:
                return torch.stack([torch.zeros_like(h), torch.zeros_like(h), h])
            p = torch.cat([self.coords(theta), h[None]])
            self._append(p)
            return p

    def can_fit(self, directions) -> bool:
        """Whether a refit in ``directions`` can run now: random directions
        always, the path's plane once three snapshots exist, counting the one
        a refit takes."""
        if directions == RANDOM:
            return True
        k = len(self.snapshot_steps)
        if not self.snapshot_steps or self.snapshot_steps[-1] != self.step:
            k += 1
        return k >= 3

    def due(self, step, every, directions) -> bool:
        """Whether the loop refits at ``step``: first at step 0 in random
        directions, or in the path's plane once three snapshots exist (step 50
        at the defaults), then at every multiple of ``every``."""
        if directions not in (PATH, RANDOM):
            raise ValueError(f"directions must be {PATH!r} or {RANDOM!r}, got {directions!r}")
        if not self.can_fit(directions):
            return False
        if not self.ready:
            return directions == RANDOM or len(self.snapshot_steps) >= 3
        return step % int(every) == 0

    def refit(self, model, span, directions):
        """Refit the plane (in the path's plane) and compute the surface at the
        model's current weights. Returns ``positions`` (G * G, 3) and ``values``
        (G * G,) of the mesh, on the device, and the plane share (a 0-d tensor)
        in the path's plane or None in random directions. Replaces the path in
        the path's plane, on a change of mode, and in random directions
        rescales it when ``span`` changed."""
        if directions not in (PATH, RANDOM):
            raise ValueError(f"directions must be {PATH!r} or {RANDOM!r}, got {directions!r}")
        if not self.can_fit(directions):
            raise ValueError("the path's plane needs three snapshots; the first refit can run "
                             f"at step {2 * self.snapshot_every}")
        span = float(span)
        if not span > 0:
            raise ValueError(f"span must be above 0, got {span!r}")
        with torch.no_grad():
            theta = self.flat(model)
            h = 0.5 * _log_loss(self.batch_loss(model))
            self._height = h
            if not self.snapshot_steps or self.snapshot_steps[-1] != self.step:
                self._snapshot(theta, h, self.step)
            if directions == PATH:
                share = self._fit_path(theta, span)
                centre_xy = torch.zeros(2, device=self.device, dtype=self.dtype)
            else:
                share = None
                self._set_random(span)
                centre_xy = self.coords(theta)
            offsets = self.half_width * self.offsets                 # weight-space offsets
            losses = self._grid_losses(model, theta, offsets)         # (G, G): [row b, column a]
        log_l = _log_loss(losses).reshape(-1)
        g = self.grid
        xs = (centre_xy[0] + self.offsets).repeat(g)
        ys = (centre_xy[1] + self.offsets).repeat_interleave(g)
        positions = torch.stack([xs, ys, 0.5 * log_l], 1)
        self.mode, self.ready, self.span = directions, True, span
        self.share = share
        self.refits.append((self.step, directions))
        return positions, log_l, share

    # -- the path's plane --------------------------------------------------------
    def _fit_path(self, theta, span):
        """The top two principal components of the kept snapshots around the
        current weights ``theta``, matched in order and sign to the previous
        plane; the path from the snapshots' projections. Returns the share."""
        m = self.snapshots - theta                                # (K, P)
        k = m.shape[0]
        g = m @ m.T                                               # (K, K)
        prev = self._path_plane
        if prev is not None:
            warm = m @ prev.T                                     # coordinates on the last plane
        else:
            ramp = torch.linspace(-1.0, 1.0, k, device=self.device, dtype=self.dtype)
            warm = torch.stack([ramp, ramp * ramp - (ramp * ramp).mean()], 1)
        ramp = torch.linspace(-1.0, 1.0, k, device=self.device, dtype=self.dtype)
        nudge = _orthonormal_columns(torch.stack([ramp, ramp * ramp - (ramp * ramp).mean()], 1))
        v = _orthonormal_columns(torch.stack([_unit(warm[:, 0]), _unit(warm[:, 1])], 1) +
                                 1e-3 * nudge)                    # never rank-deficient
        for _ in range(FIT_ITERATIONS):
            v = _orthonormal_columns(g @ v)
        # Rayleigh-Ritz in the two columns: the 2 x 2 eigenproblem, in closed form.
        t = v.T @ g @ v
        phi = 0.5 * torch.atan2(2.0 * t[0, 1], t[0, 0] - t[1, 1])
        cs, sn = torch.cos(phi), torch.sin(phi)
        v1, v2 = cs * v[:, 0] + sn * v[:, 1], cs * v[:, 1] - sn * v[:, 0]
        lam = torch.stack([v1 @ g @ v1, v2 @ g @ v2])
        u1 = _unit(m.T @ v1)
        u2 = m.T @ v2
        u2 = _unit(u2 - (u1 @ u2) * u1)
        share = lam.sum() / torch.trace(g).clamp_min(_TINY)
        if prev is not None:
            # Match to the previous plane: the pairing with the larger sum of
            # absolute dot products keeps the order, a positive dot product the sign.
            dots = torch.stack([u1, u2]) @ prev.T                 # [new, old]
            swap = (dots[1, 0].abs() + dots[0, 1].abs()) > (dots[0, 0].abs() + dots[1, 1].abs())
            u1, u2 = torch.where(swap, u2, u1), torch.where(swap, u1, u2)
            lam = torch.where(swap, lam.flip(0), lam)
            u1 = u1 * _sign(u1 @ prev[0])
            u2 = u2 * _sign(u2 @ prev[1])
        else:
            # The first plane: the oldest snapshot on the negative side of both axes.
            u1 = u1 * -_sign(u1 @ m[0])
            u2 = u2 * -_sign(u2 @ m[0])
        plane = torch.stack([u1, u2])                             # (2, P), orthonormal
        c = m @ plane.T                                           # (K, 2), weight units
        hw = torch.maximum(SPAN_SCALE * span * c.abs().max(), MIN_HALF_WIDTH * theta.norm())
        self._path_plane = plane
        self.directions, self.centre, self.half_width = plane, theta.clone(), hw
        self.eigenvalues = lam
        self._proj = plane / hw
        self._set_path(torch.cat([c / hw, self.snapshot_heights[:, None]], 1))
        return share

    # -- random directions -------------------------------------------------------
    def _set_random(self, span):
        hw = torch.tensor(span, device=self.device, dtype=self.dtype)
        if self.mode == RANDOM and self.ready:
            if span != self.span:                                 # rescale the record
                self._path[:self._n, :2] *= self.span / span
        else:                                                     # the path from the snapshots
            c = (self._random_proj @ (self.snapshots - self.anchor).T).T / hw
            self._set_path(torch.cat([c, self.snapshot_heights[:, None]], 1))
        self.directions, self.centre, self.half_width = self.random_directions, self.anchor, hw
        self.eigenvalues = None
        self._proj = self._random_proj / hw

    # -- the grid ----------------------------------------------------------------
    def _grid_losses(self, model, theta, offsets):
        """The landscape batch's loss at theta + a * d1 + b * d2 for a and b in
        ``offsets``: (G, G), row b, column a. One batched forward pass per row
        under vmap, or one forward pass per point where vmap cannot run."""
        names, x, y, loss = self.names, self.x, self.y, self.loss
        th = self.split(theta)
        d1, d2 = self.split(self.directions[0]), self.split(self.directions[1])

        def grid_loss(a, b):                                      # one grid point
            params = {n: t + a * u + b * v for n, t, u, v in zip(names, th, d1, d2)}
            return loss(_logits(functional_call(model, params, (x,))), y)

        def loop_row(b):
            return torch.stack([grid_loss(a, b) for a in offsets])

        rows = []
        if self.method is None:
            self.method, first = self._detect(grid_loss, offsets)
            rows.append(first if first is not None else loop_row(offsets[0]))
        if self.method == "vmap":
            row = vmap(grid_loss, in_dims=(0, None))              # one row at once
            rows += [row(offsets, b) for b in offsets[len(rows):]]
        else:
            rows += [loop_row(b) for b in offsets[len(rows):]]
        return torch.stack(rows)

    def _detect(self, grid_loss, offsets):
        """vmap or the loop: try one row under vmap with functorch's slow
        fallback turned off, so that an operation without a batching rule on
        this device raises, and fall back to the loop if anything raises."""
        fc = torch._C._functorch
        get = getattr(fc, "_is_vmap_fallback_enabled", None)
        put = getattr(fc, "_set_vmap_fallback_enabled", None)
        before = get() if get and put else None
        try:
            if before is not None:
                put(False)
            first = vmap(grid_loss, in_dims=(0, None))(offsets, offsets[0])
            return "vmap", first
        except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
            warnings.warn(f"landscape: vmap cannot batch this model on {self.device.type} "
                          f"({str(e).splitlines()[0] if str(e) else type(e).__name__}); the grid "
                          "is evaluated with one forward pass per point", RuntimeWarning,
                          stacklevel=4)
            return "loop", None
        finally:
            if before is not None:
                put(before)
