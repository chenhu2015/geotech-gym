"""Limit-equilibrium slope stability.

Methods
- Bishop's simplified (circular surfaces; moment equilibrium only).
- General limit equilibrium (GLE, Fredlund & Krahn 1977) for any slip surface, which gives
  Spencer's method (constant interslice inclination, f(x) = 1) and Morgenstern-Price with a
  half-sine interslice function. Both satisfy force and moment equilibrium.

Nonlinear strength (Hoek-Brown) is handled by updating the instantaneous cohesion and friction
angle at each slice base from the current normal stress.

Conventions (see profile.py): the slope faces +x, the sliding mass moves toward +x, and the base
inclination alpha is positive where the base dips in the direction of motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq, minimize

from .profile import SlopeProfile


class InvalidSurface(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# Slip surfaces
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Circle:
    xc: float
    zc: float
    r: float

    def z(self, x):
        dx = np.clip(np.abs(np.asarray(x) - self.xc), 0.0, self.r)
        return self.zc - np.sqrt(self.r * self.r - dx * dx)

    def dzdx(self, x):
        root = np.sqrt(np.clip(self.r * self.r - (np.asarray(x) - self.xc) ** 2, 1e-12, None))
        return (np.asarray(x) - self.xc) / root

    @property
    def domain(self) -> tuple[float, float]:
        return self.xc - self.r, self.xc + self.r

    @property
    def moment_point(self) -> tuple[float, float]:
        return self.xc, self.zc


@dataclass(frozen=True)
class PolySurface:
    """Non-circular slip surface through (x, z) vertices, x increasing."""

    xs: tuple
    zs: tuple

    def z(self, x):
        return np.interp(x, self.xs, self.zs)

    def dzdx(self, x):
        xs, zs = np.asarray(self.xs), np.asarray(self.zs)
        seg = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
        return (zs[seg + 1] - zs[seg]) / (xs[seg + 1] - xs[seg])

    @property
    def domain(self) -> tuple[float, float]:
        return self.xs[0], self.xs[-1]

    @property
    def moment_point(self) -> tuple[float, float]:
        xs, zs = np.asarray(self.xs), np.asarray(self.zs)
        return float(xs.mean()), float(zs.max() + (xs[-1] - xs[0]))


Surface = Circle | PolySurface


@dataclass
class Slices:
    x: np.ndarray        # slice mid x
    b: np.ndarray        # slice width
    alpha: np.ndarray    # base inclination (rad), + where the base dips toward +x
    w: np.ndarray        # weight (kN/m)
    u: np.ndarray        # pore pressure at base mid (kPa)
    layer: np.ndarray    # base layer index
    z_base: np.ndarray
    x_edges: np.ndarray  # slice boundaries
    x_entry: float
    x_exit: float

    @property
    def length(self) -> np.ndarray:
        return self.b / np.cos(self.alpha)


def _slice_edges(profile: SlopeProfile, surface: Surface, x0: float, x1: float, n: int) -> np.ndarray:
    """Uniform slice edges plus every breakpoint of the ground, layers, water table and slip
    surface, so that no slice straddles a kink (midpoint integration is then exact on
    piecewise-linear geometry)."""
    pts = [np.linspace(x0, x1, n + 1), profile.surface.x]
    for layer in profile.layers:
        if layer.bottom is not None:
            pts.append(layer.bottom.x)
    if profile.water_table is not None:
        pts.append(profile.water_table.x)
    if isinstance(surface, PolySurface):
        pts.append(np.asarray(surface.xs))
    e = np.unique(np.concatenate(pts))
    e = e[(e >= x0) & (e <= x1)]
    keep = np.concatenate([[True], np.diff(e) > 1e-6 * max(1.0, x1 - x0)])
    e = e[keep]
    e[0], e[-1] = x0, x1
    return e


def make_slices(profile: SlopeProfile, surface: Surface, n: int = 60) -> Slices:
    """Discretize the sliding mass between the ground and ``surface`` into ``n`` vertical slices."""
    d0, d1 = surface.domain
    lo, hi = max(profile.xmin, d0), min(profile.xmax, d1)
    if hi - lo < 1e-6:
        raise InvalidSurface("surface does not span the profile")

    def gap(x):
        return profile.surface(x) - surface.z(x)

    xs = np.linspace(lo, hi, 801)
    g = gap(xs)
    pos = g > 1e-9
    if not pos.any():
        raise InvalidSurface("surface does not cut the ground")
    edges = np.flatnonzero(np.diff(np.concatenate([[0], pos.astype(int), [0]])))
    runs = edges.reshape(-1, 2)
    k = int(np.argmax(runs[:, 1] - runs[:, 0]))
    i0, i1 = runs[k][0], runs[k][1] - 1
    if isinstance(surface, Circle):
        if i0 == 0 or i1 == len(xs) - 1:
            raise InvalidSurface("slip surface leaves the model domain")
        x_entry = brentq(gap, xs[i0 - 1], xs[i0])
        x_exit = brentq(gap, xs[i1], xs[i1 + 1])
    else:
        # Polyline surfaces start and end on the ground by construction.
        x_entry, x_exit = d0, d1

    x_edges = _slice_edges(profile, surface, x_entry, x_exit, n)
    xm = 0.5 * (x_edges[:-1] + x_edges[1:])
    b = np.diff(x_edges)
    z_base = surface.z(xm)
    if np.any(z_base < profile.base_z - 1e-9):
        raise InvalidSurface("slip surface passes below the model base")
    alpha = -np.arctan(surface.dzdx(xm))
    w = profile.column_weight(xm, z_base, b)
    u = profile.pore_pressure(xm, z_base)
    layer = profile.base_layer_index(xm, z_base)
    return Slices(xm, b, alpha, w, u, layer, z_base, x_edges, float(x_entry), float(x_exit))


# ---------------------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------------------


def _strength_params(profile: SlopeProfile, s: Slices, sigma_n: np.ndarray):
    c = np.empty_like(s.x)
    tphi = np.empty_like(s.x)
    for i, layer in enumerate(profile.layers):
        m = s.layer == i
        if m.any():
            _, ci, ti = layer.material.strength(sigma_n[m])
            c[m], tphi[m] = ci, ti
    return c, tphi


def _initial_sigma_n(s: Slices) -> np.ndarray:
    return np.clip((s.w * np.cos(s.alpha) ** 2) / s.b - s.u, 0.0, None)


def bishop(profile: SlopeProfile, s: Slices, tol: float = 1e-6, max_iter: int = 200) -> float:
    """Bishop's simplified method (moment equilibrium about the circle centre)."""
    sa, ca = np.sin(s.alpha), np.cos(s.alpha)
    driving = np.sum(s.w * sa)
    if driving <= 0:
        raise InvalidSurface("no driving moment")
    length = s.length
    sigma_n = _initial_sigma_n(s)
    f = 1.0
    for _ in range(max_iter):
        c, tphi = _strength_params(profile, s, sigma_n)
        m_alpha = ca + sa * tphi / f
        if np.any(m_alpha < 0.2):
            raise InvalidSurface("m_alpha too small (base too steep)")
        resist = np.sum((c * s.b + (s.w - s.u * s.b) * tphi) / m_alpha)
        f_new = resist / driving
        n_eff = (s.w - s.u * s.b - c * length * sa / f_new) / (ca + sa * tphi / f_new)
        sigma_n = np.clip(n_eff / length, 0.0, None)
        if abs(f_new - f) < tol:
            return float(f_new)
        f = f_new
    return float(f)


def _gle_at_lambda(profile, s, lam, fx_edges, xo, zo, f_start, n_iter=60, tol=1e-7):
    """Return (F_moment, F_force) for interslice shear X = lam * f(x) * E."""
    sa, ca = np.sin(s.alpha), np.cos(s.alpha)
    length = s.length
    dx, dz = s.x - xo, s.z_base - zo
    ra = dx * sa + dz * ca             # lever arm of the base shear about O
    rn = dx * ca - dz * sa             # lever arm of the base normal about O
    n_sl = len(s.x)
    x_int = np.zeros(n_sl + 1)         # interslice shear at slice edges (ends are zero)
    sigma_n = _initial_sigma_n(s)
    f_m = f_f = f_start
    for _ in range(n_iter):
        c, tphi = _strength_params(profile, s, sigma_n)
        d_x = x_int[:-1] - x_int[1:]   # X_L - X_R acting upward on the slice

        def normal(f):
            m_alpha = ca + sa * tphi / f
            return (s.w - d_x - (c * length - s.u * length * tphi) * sa / f) / m_alpha, m_alpha

        n_m, m_a = normal(f_m)
        if np.any(m_a < 0.2):
            raise InvalidSurface("m_alpha too small")
        res_m = c * length + (n_m - s.u * length) * tphi
        denom_m = np.sum(s.w * dx - n_m * rn)
        f_m_new = np.sum(res_m * ra) / denom_m

        n_f, m_a = normal(f_f)
        if np.any(m_a < 0.2):
            raise InvalidSurface("m_alpha too small")
        res_f = c * length + (n_f - s.u * length) * tphi
        denom_f = np.sum(n_f * sa)
        f_f_new = np.sum(res_f * ca) / denom_f

        if not (np.isfinite(f_m_new) and np.isfinite(f_f_new)) or f_m_new <= 0 or f_f_new <= 0:
            raise InvalidSurface("GLE iteration diverged")

        # Horizontal interslice forces from slice horizontal equilibrium, using F_force.
        shear = res_f / f_f_new
        de = n_f * sa - shear * ca     # E_R - E_L
        e = np.concatenate([[0.0], np.cumsum(de)])
        # Sign chosen so that lambda > 0 for ordinary slopes facing +x.
        x_new = -lam * fx_edges * e
        x_new[0] = x_new[-1] = 0.0
        sigma_n = np.clip((n_f - s.u * length) / length, 0.0, None)

        done = abs(f_m_new - f_m) < tol and abs(f_f_new - f_f) < tol
        f_m, f_f, x_int = f_m_new, f_f_new, 0.5 * x_int + 0.5 * x_new
        if done:
            break
    return f_m, f_f


def gle(profile: SlopeProfile, s: Slices, surface: Surface, interslice: str = "spencer") -> tuple[float, float]:
    """General limit equilibrium. Returns (factor of safety, lambda)."""
    xl, xr = s.x_edges[0], s.x_edges[-1]
    if interslice == "spencer":
        fx = np.ones_like(s.x_edges)
    elif interslice == "half-sine":
        fx = np.sin(np.pi * (s.x_edges - xl) / (xr - xl))
    else:
        raise ValueError(interslice)
    xo, zo = surface.moment_point
    if isinstance(surface, Circle):
        f_start = bishop(profile, s)
    else:
        f_start = 1.5

    def diff(lam):
        fm, ff = _gle_at_lambda(profile, s, lam, fx, xo, zo, f_start)
        return fm - ff, fm

    # Scan outward from lambda = 0 and take the first sign change: the physically meaningful
    # root is the one with the smallest |lambda| (large |lambda| makes the iteration unstable).
    lams = np.linspace(-0.6, 1.6, 23)
    vals = {}
    for lam in lams:
        try:
            vals[lam] = diff(lam)[0]
        except InvalidSurface:
            vals[lam] = np.nan
    brackets = []
    for i in range(len(lams) - 1):
        a, b = vals[lams[i]], vals[lams[i + 1]]
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0:
            brackets.append((min(abs(lams[i]), abs(lams[i + 1])), lams[i], lams[i + 1]))
    if not brackets:
        raise InvalidSurface("GLE: no lambda satisfies both force and moment equilibrium")
    _, l0, l1 = min(brackets)
    lam = brentq(lambda t: diff(t)[0], l0, l1, xtol=1e-7)
    return float(diff(lam)[1]), float(lam)


def evaluate(profile: SlopeProfile, surface: Surface, method: str = "bishop", n: int = 60) -> float:
    s = make_slices(profile, surface, n)
    if method == "bishop":
        if not isinstance(surface, Circle):
            raise ValueError("Bishop's simplified method needs a circular surface")
        return bishop(profile, s)
    if method == "spencer":
        return gle(profile, s, surface, "spencer")[0]
    if method in ("morgenstern-price", "mp"):
        return gle(profile, s, surface, "half-sine")[0]
    raise ValueError(method)


# Backwards-compatible name.
def fos_for_circle(profile, circle, method="bishop", n=60):
    return evaluate(profile, circle, method, n)


# ---------------------------------------------------------------------------------------------
# Critical-surface search
# ---------------------------------------------------------------------------------------------


@dataclass
class SearchResult:
    fos: float
    surface: Surface
    method: str
    x_entry: float
    x_exit: float
    evaluations: int


def _circle_from(profile: SlopeProfile, x1: float, x2: float, depth: float) -> Circle | None:
    """Circle through ground points at x1, x2 with arc depth ``depth`` below their chord."""
    if not (profile.xmin < x1 < x2 < profile.xmax) or depth <= 1e-3:
        return None
    z1, z2 = float(profile.surface(x1)), float(profile.surface(x2))
    chord = np.hypot(x2 - x1, z2 - z1)
    if chord < 1e-3:
        return None
    r = (chord * chord / 4.0 + depth * depth) / (2.0 * depth)
    mx, mz = (x1 + x2) / 2.0, (z1 + z2) / 2.0
    nx, nz = -(z2 - z1) / chord, (x2 - x1) / chord
    if nz < 0:
        nx, nz = -nx, -nz
    dist = r - depth
    return Circle(mx + nx * dist, mz + nz * dist, r)


def search_circular(profile: SlopeProfile, method: str = "bishop", n_slices: int = 60,
                    grid: int = 16, n_starts: int = 8, min_depth: float = 0.3) -> SearchResult:
    """Critical circle: coarse grid over (entry, exit, depth), then Nelder-Mead refinement.

    Non-Bishop methods search with Bishop first (fast) and then refine with the requested method.
    """
    xs, xe = profile.xmin, profile.xmax
    span = xe - xs
    evals = 0

    def objective(p, m):
        nonlocal evals
        evals += 1
        circ = _circle_from(profile, *p)
        if circ is None or p[2] < min_depth:
            return 1e3
        try:
            return evaluate(profile, circ, m, n_slices)
        except (InvalidSurface, ValueError):
            return 1e3

    cands = []
    x_grid = np.linspace(xs + 0.01 * span, xe - 0.01 * span, grid)
    for i, x1 in enumerate(x_grid):
        for x2 in x_grid[i + 1:]:
            for frac in (0.05, 0.1, 0.18, 0.28, 0.4):
                p = np.array([x1, x2, max(min_depth, frac * (x2 - x1))])
                cands.append((objective(p, "bishop"), p))
    cands.sort(key=lambda t: t[0])

    refined = []
    for f0, p0 in cands[:n_starts]:
        if f0 >= 1e3:
            break
        res = minimize(objective, p0, args=("bishop",), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 1500})
        refined.append((float(res.fun), res.x))
    if not refined:
        raise InvalidSurface("no valid slip surface found")
    refined.sort(key=lambda t: t[0])
    best_f, best_p = refined[0]

    if method != "bishop":
        best_f, best_p = np.inf, None
        for _, p0 in refined[:3]:
            res = minimize(objective, p0, args=(method,), method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
            if res.fun < best_f:
                best_f, best_p = float(res.fun), res.x
    circ = _circle_from(profile, *best_p)
    s = make_slices(profile, circ, n_slices)
    return SearchResult(best_f, circ, method, s.x_entry, s.x_exit, evals)


def search_noncircular(profile: SlopeProfile, method: str = "spencer", n_slices: int = 60,
                       n_nodes: int = 7, start: SearchResult | None = None,
                       rounds: int = 3) -> SearchResult:
    """Non-circular critical surface by optimizing a polyline that starts from the critical circle.

    Variables: entry x, exit x, and the depths of ``n_nodes`` interior vertices below the ground
    (spaced evenly in x). Surfaces must stay concave upward (base inclination decreasing in the
    direction of motion), which rules out kinematically implausible kinks.
    """
    if start is None:
        start = search_circular(profile, method if method != "bishop" else "bishop", n_slices)
    circ = start.surface
    x1, x2 = start.x_entry, start.x_exit
    t = np.linspace(0.0, 1.0, n_nodes + 2)[1:-1]
    xi = x1 + t * (x2 - x1)
    depth0 = profile.surface(xi) - circ.z(xi)
    evals = 0

    def build(p):
        a, b = p[0], p[1]
        if not (profile.xmin < a < b < profile.xmax):
            return None
        d = np.asarray(p[2:])
        if np.any(d <= 0):
            return None
        xin = a + t * (b - a)
        zin = profile.surface(xin) - d
        xs_ = np.concatenate([[a], xin, [b]])
        zs_ = np.concatenate([[profile.surface(a)], zin, [profile.surface(b)]])
        slopes = np.diff(zs_) / np.diff(xs_)
        if np.any(np.diff(slopes) < -1e-9):  # dz/dx must not decrease: concave upward
            return None
        if np.any(zs_ < profile.base_z):
            return None
        return PolySurface(tuple(xs_), tuple(zs_))

    def objective(p):
        nonlocal evals
        evals += 1
        surf = build(p)
        if surf is None:
            return 1e3
        try:
            return evaluate(profile, surf, method, n_slices)
        except (InvalidSurface, ValueError):
            return 1e3

    p = np.concatenate([[x1, x2], depth0])
    best = objective(p)
    for _ in range(rounds):
        res = minimize(objective, p, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000, "adaptive": True})
        if res.fun < best - 1e-6:
            best, p = float(res.fun), res.x
        else:
            break
    surf = build(p)
    s = make_slices(profile, surf, n_slices)
    return SearchResult(best, surf, method, s.x_entry, s.x_exit, evals + start.evaluations)
