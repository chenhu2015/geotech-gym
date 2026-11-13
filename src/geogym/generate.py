"""Procedural generation of slope sites for the task families.

Every site is a 2D cut slope: a flat toe, a planar face, and a flat crest, over one to three
horizontal soil units, optionally with a phreatic surface. Sites are drawn facing either
direction so agents cannot rely on a fixed orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import MohrCoulomb
from .spec import make_spec

SOIL_NAMES = ["silty sand", "sandy clay", "stiff clay", "weathered siltstone", "clayey gravel"]


@dataclass
class Site:
    spec: dict
    height: float
    angle_deg: float
    x_toe: float      # x of the toe (in the site's own orientation)
    x_crest: float    # x of the crest edge
    z_toe: float
    z_crest: float
    faces_left: bool  # True if the slope face looks toward -x (crest on the right)


def random_site(rng: np.random.Generator, water_prob: float = 0.5, max_layers: int = 3) -> Site:
    h = float(np.round(rng.uniform(6.0, 20.0), 1))
    beta = float(np.round(rng.uniform(26.0, 55.0), 1))
    z_toe = 10.0
    z_crest = z_toe + h
    run = h / np.tan(np.radians(beta))
    toe_len = float(np.round(max(2.0 * h, 15.0), 1))
    crest_len = float(np.round(max(3.0 * h, 25.0), 1))
    depth_below_toe = float(np.round(rng.uniform(0.6, 1.5) * h, 1))
    base_z = z_toe - depth_below_toe

    x_toe = toe_len
    x_crest = toe_len + run
    x_end = x_crest + crest_len
    # Drawn with the crest on the right (face looks toward -x); mirrored below half the time.
    surface = [(0.0, z_toe), (x_toe, z_toe), (x_crest, z_crest), (x_end, z_crest)]

    n_layers = int(rng.integers(1, max_layers + 1))
    # Interfaces between the crest and a bit below the toe, sorted top-down.
    cuts = np.sort(rng.uniform(z_toe - 0.5 * depth_below_toe, z_crest - 1.0, n_layers - 1))[::-1]
    layers = []
    names = []
    for i in range(n_layers):
        # c >= 3 kPa: a cohesionless unit daylighting in a steep face only gives a skin failure.
        c = float(np.round(rng.uniform(3.0, 25.0), 1))
        phi = float(np.round(rng.uniform(20.0, 38.0), 1))
        gamma = float(np.round(rng.uniform(17.0, 21.0), 1))
        if rng.random() < 0.2:  # occasional soft clay unit
            c, phi = float(np.round(rng.uniform(15.0, 35.0), 1)), float(np.round(rng.uniform(5.0, 15.0), 1))
        bottom = None if i == n_layers - 1 else [(0.0, float(np.round(cuts[i], 2))), (x_end, float(np.round(cuts[i], 2)))]
        layers.append((MohrCoulomb(c, phi, gamma), bottom))
        names.append(SOIL_NAMES[int(rng.integers(len(SOIL_NAMES)))])

    water = None
    if rng.random() < water_prob:
        d_crest = float(np.round(rng.uniform(0.2, 0.8) * h, 1))
        z_w_crest = z_crest - d_crest
        # Phreatic line: flat under the crest, dropping to emerge at the toe.
        water = [(0.0, z_toe), (x_toe, z_toe), (x_crest + 0.5 * h, z_w_crest), (x_end, z_w_crest)]

    faces_left = True
    if rng.random() < 0.5:
        surface, layers, water, x_toe, x_crest = _mirror(surface, layers, water, x_end, x_toe, x_crest)
        faces_left = False

    spec = make_spec(surface, layers, base_z=base_z, water_table=water, names=names)
    return Site(spec, h, beta, x_toe, x_crest, z_toe, z_crest, faces_left)


def _mirror(surface, layers, water, x_end, x_toe, x_crest):
    m = lambda pts: [(round(x_end - x, 4), z) for x, z in reversed(pts)]
    layers = [(mat, m(b) if b is not None else None) for mat, b in layers]
    return m(surface), layers, (m(water) if water is not None else None), x_end - x_toe, x_end - x_crest
