"""2D slope cross-section: ground surface, soil layers, and an optional water table.

Coordinates: x horizontal (m), z vertical (m, up). Internally every profile is stored so that
the slope faces +x (high ground on the left, toe on the right) and a failing mass moves toward
+x. Profiles given the other way round are mirrored on construction; ``mirrored`` records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import HoekBrown, MohrCoulomb

Material = MohrCoulomb | HoekBrown


class Polyline:
    """Piecewise-linear z(x) with strictly increasing x."""

    def __init__(self, points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("polyline needs at least two (x, z) points")
        if np.any(np.diff(pts[:, 0]) <= 0):
            raise ValueError("polyline x must be strictly increasing")
        self.x = pts[:, 0]
        self.z = pts[:, 1]

    def __call__(self, x):
        return np.interp(x, self.x, self.z)

    @property
    def xmin(self) -> float:
        return float(self.x[0])

    @property
    def xmax(self) -> float:
        return float(self.x[-1])

    def mirrored(self) -> "Polyline":
        return Polyline(np.column_stack([-self.x[::-1], self.z[::-1]]))


@dataclass
class Layer:
    """A soil/rock unit lying above ``bottom`` (and below the unit listed before it).

    The last layer may have ``bottom=None`` meaning it extends down to the model base.
    """

    material: Material
    bottom: Polyline | None = None


@dataclass
class SlopeProfile:
    surface: Polyline
    layers: list[Layer]
    base_z: float = 0.0
    water_table: Polyline | None = None
    gamma_w: float = 9.81
    mirrored: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("at least one layer is required")
        # Face the slope toward +x: the left end must be the high side.
        if self.surface.z[0] < self.surface.z[-1]:
            self.surface = self.surface.mirrored()
            self.layers = [
                Layer(l.material, l.bottom.mirrored() if l.bottom is not None else None)
                for l in self.layers
            ]
            if self.water_table is not None:
                self.water_table = self.water_table.mirrored()
            self.mirrored = True

    @property
    def xmin(self) -> float:
        return self.surface.xmin

    @property
    def xmax(self) -> float:
        return self.surface.xmax

    def layer_bounds(self, x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each layer, its (top, bottom) elevation at positions x."""
        bounds = []
        top = np.full_like(x, np.inf, dtype=float)
        for layer in self.layers:
            bottom = layer.bottom(x) if layer.bottom is not None else np.full_like(x, -np.inf)
            bounds.append((top, bottom))
            top = np.minimum(top, bottom)
        return bounds

    def column_weight(self, x: np.ndarray, z_bot: np.ndarray, width: np.ndarray) -> np.ndarray:
        """Weight (kN per m out of plane) of soil columns from z_bot up to the ground surface."""
        z_top = self.surface(x)
        w = np.zeros_like(x, dtype=float)
        for layer, (lt, lb) in zip(self.layers, self.layer_bounds(x)):
            hi = np.minimum(z_top, lt)
            lo = np.maximum(z_bot, lb)
            w += layer.material.gamma * np.clip(hi - lo, 0.0, None) * width
        return w

    def base_layer_index(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Index of the layer containing points (x, z)."""
        idx = np.full(x.shape, len(self.layers) - 1, dtype=int)
        assigned = np.zeros(x.shape, dtype=bool)
        for i, (lt, lb) in enumerate(self.layer_bounds(x)):
            inside = (~assigned) & (z >= lb)
            idx[inside] = i
            assigned |= inside
        return idx

    def pore_pressure(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.water_table is None:
            return np.zeros_like(x, dtype=float)
        return self.gamma_w * np.clip(self.water_table(x) - z, 0.0, None)
