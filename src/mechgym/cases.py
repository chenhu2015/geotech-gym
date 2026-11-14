"""Published reference problems used to validate the limit-equilibrium verifier.

See docs/validation-cases.md for sources and reference values.
"""

from __future__ import annotations

from .materials import HoekBrown, MohrCoulomb
from .profile import Layer, Polyline, SlopeProfile


def v1_complex_slope() -> SlopeProfile:
    """Cheng, Lansivaara & Wei (2007); Itasca FLAC3D 'complex surface profile' example.
    Reference FoS: Morgenstern-Price 1.383 / 1.3848 (non-circular); FLAC3D SRM 1.38."""
    soil = MohrCoulomb(c=10.0, phi_deg=30.0, gamma=20.0)
    surface = Polyline([(0, 10), (10, 10), (16, 16), (26, 21), (46, 21)])
    return SlopeProfile(surface, [Layer(soil)], base_z=0.0)


_HB = HoekBrown(sigci=30000.0, mb=0.067, s=0.000025, a=0.619, gamma=25.0)
_V2_SURFACE = [(0, 8), (15, 8), (25, 18), (40, 18)]


def v2_hoek_brown_slope() -> SlopeProfile:
    """Hammah et al. (2005) Example 1; Itasca FLAC3D 'simple slope in Hoek-Brown material'.
    Reference FoS: Bishop 1.153, Spencer 1.152 (circular); SSR 1.15."""
    return SlopeProfile(Polyline(_V2_SURFACE), [Layer(_HB)], base_z=0.0)


def v2_layered() -> SlopeProfile:
    """Hammah et al. (2005) Example 2: 1 m Mohr-Coulomb layer (c = 0, phi = 25 deg) at the toe,
    z = 8-9 m as in the Itasca model. Reference FoS (non-circular): Bishop 0.934, Spencer 0.963;
    SSR 0.95."""
    weak = MohrCoulomb(c=0.0, phi_deg=25.0, gamma=25.0)
    layers = [
        Layer(_HB, Polyline([(0, 9), (40, 9)])),
        Layer(weak, Polyline([(0, 8), (40, 8)])),
        Layer(_HB),
    ]
    return SlopeProfile(Polyline(_V2_SURFACE), layers, base_z=0.0)


REFERENCE = {
    "v1": {"morgenstern-price (non-circular)": 1.383, "flac3d srm": 1.38},
    "v2": {"bishop": 1.153, "spencer": 1.152, "ssr": 1.15},
    "v2_layered": {"bishop (non-circular)": 0.934, "spencer (non-circular)": 0.963, "ssr": 0.95},
}
