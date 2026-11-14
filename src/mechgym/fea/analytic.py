"""Closed-form linear-elastic solutions (plane strain) used as T0 references.

Sign convention: tension positive. Displacements in the same length unit as the inputs.
"""

from __future__ import annotations

import numpy as np


def lame_radial_displacement(r, a, b, p, e, nu):
    """Thick-walled cylinder (inner radius a, outer radius b) under internal pressure p (outer
    surface traction-free), plane strain. Returns u_r(r)."""
    big_a = p * a * a / (b * b - a * a)
    big_b = p * a * a * b * b / (b * b - a * a)
    return (1.0 + nu) / e * ((1.0 - 2.0 * nu) * big_a * r + big_b / r)


def kirsch_displacement(r, theta, a, sx, sy, e, nu):
    """Circular hole (radius a, traction-free) in an infinite plate under far-field normal stresses
    sx (along x) and sy (along y), plane strain. Returns (u_r, u_theta) at polar (r, theta),
    with rigid-body motion fixed by symmetry about both axes.

    Uniaxial solution (Barber, Elasticity) for stress S along the 0-direction:
      u_r  =  S/(4G) [ r((k-1)/2 + cos2t) + a^2/r (1 + (k+1) cos2t) - a^4/r^3 cos2t ]
      u_t  = -S/(4G) [ r + (k-1) a^2/r + a^4/r^3 ] sin2t
    with k = 3 - 4 nu (plane strain); sy is the same solution rotated by 90 degrees.
    """
    g = e / (2.0 * (1.0 + nu))
    k = 3.0 - 4.0 * nu

    def uni(s, t):
        c2, s2 = np.cos(2 * t), np.sin(2 * t)
        ur = s / (4 * g) * (r * ((k - 1) / 2 + c2) + a * a / r * (1 + (k + 1) * c2) - a**4 / r**3 * c2)
        ut = -s / (4 * g) * (r + (k - 1) * a * a / r + a**4 / r**3) * s2
        return ur, ut

    ur1, ut1 = uni(sx, theta)
    ur2, ut2 = uni(sy, theta - np.pi / 2)
    return ur1 + ur2, ut1 + ut2
