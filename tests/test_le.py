import numpy as np
import pytest

from geogym import cases
from geogym.le import (
    Circle,
    PolySurface,
    evaluate,
    make_slices,
    search_circular,
    search_noncircular,
)
from geogym.materials import HoekBrown, MohrCoulomb
from geogym.profile import Layer, Polyline, SlopeProfile


def rel(a, b):
    return abs(a - b) / abs(b)


# --- closed-form checks ----------------------------------------------------------------------


def test_infinite_slope_cohesionless():
    """Planar slip parallel to a long dry cohesionless slope: FoS = tan(phi) / tan(beta)."""
    beta, phi = 30.0, 35.0
    soil = MohrCoulomb(c=0.0, phi_deg=phi, gamma=19.0)
    length = 200.0
    h = length * np.tan(np.radians(beta))
    prof = SlopeProfile(Polyline([(0, h + 10), (length, 10)]), [Layer(soil)], base_z=0.0)
    depth = 2.0
    # Surface parallel to the slope, entering steeply at the crest and leaving with a gentle
    # toe segment (a steep passive toe makes m_alpha collapse, as in any LE program).
    xs = (5.0, 6.0, length - 15.0, length - 5.0)
    zs = tuple(float(prof.surface(x)) - (0 if i in (0, 3) else depth) for i, x in enumerate(xs))
    surf = PolySurface(xs, zs)
    expected = np.tan(np.radians(phi)) / np.tan(np.radians(beta))
    for method in ("spencer", "morgenstern-price"):
        assert rel(evaluate(prof, surf, method, n=200), expected) < 0.02


def test_hoek_brown_envelope_matches_balmer_point():
    hb = HoekBrown(sigci=30000.0, mb=0.067, s=0.000025, a=0.619, gamma=25.0)
    # Pick sigma3, compute the exact (sigma_n, tau) pair and check interpolation.
    s3 = 100.0
    base = hb.mb * s3 / hb.sigci + hb.s
    s1 = s3 + hb.sigci * base**hb.a
    d = 1 + hb.a * hb.mb * base ** (hb.a - 1)
    sn = (s1 + s3) / 2 - (s1 - s3) / 2 * (d - 1) / (d + 1)
    tau = (s1 - s3) * np.sqrt(d) / (d + 1)
    t, _, _ = hb.strength(np.array([sn]))
    assert rel(t[0], tau) < 1e-3


def test_bishop_equals_gle_moment_at_zero_lambda_and_spencer_close_on_circle():
    prof = cases.v1_complex_slope()
    circ = search_circular(prof, "bishop").surface
    fb = evaluate(prof, circ, "bishop")
    fs = evaluate(prof, circ, "spencer")
    assert rel(fs, fb) < 0.01


def test_mirroring_is_invariant():
    soil = MohrCoulomb(c=10.0, phi_deg=30.0, gamma=20.0)
    pts = [(0, 10), (10, 10), (16, 16), (26, 21), (46, 21)]
    a = SlopeProfile(Polyline(pts), [Layer(soil)])
    b = SlopeProfile(Polyline([(46 - x, z) for x, z in reversed(pts)]), [Layer(soil)])
    assert a.mirrored and not b.mirrored
    assert rel(search_circular(a, "bishop").fos, search_circular(b, "bishop").fos) < 1e-3


def test_slices_respect_breakpoints():
    prof = cases.v1_complex_slope()
    surf = PolySurface((-22.0, -18.0, -14.0, -10.0), (19.0, 14.0, 11.0, 10.0))
    s = make_slices(prof, surf, n=10)
    for x in (-18.0, -14.0, -16.0):  # surface and ground vertices
        assert np.isclose(s.x_edges, x).any()


# --- published benchmark problems ------------------------------------------------------------


def test_v2_hoek_brown_circular():
    prof = cases.v2_hoek_brown_slope()
    assert rel(search_circular(prof, "bishop").fos, 1.153) < 0.015
    assert rel(search_circular(prof, "spencer").fos, 1.152) < 0.015


@pytest.mark.slow
def test_v1_noncircular_morgenstern_price():
    prof = cases.v1_complex_slope()
    start = search_circular(prof, "bishop")
    res = search_noncircular(prof, "spencer", start=start)
    f_mp = evaluate(prof, res.surface, "morgenstern-price")
    assert rel(f_mp, 1.383) < 0.01


@pytest.mark.slow
def test_v2_layered_noncircular_spencer():
    prof = cases.v2_layered()
    start = search_circular(prof, "bishop")
    res = search_noncircular(prof, "spencer", start=start)
    assert rel(res.fos, 0.963) < 0.02
