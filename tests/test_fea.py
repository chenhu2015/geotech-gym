"""T0 verifier: oracle models must score 1, and known-bad or dishonest submissions must not."""

import json
import subprocess
import sys

import numpy as np
import pytest

from geogym.fea import analytic
from geogym.fea.problems import _oracle_annulus, make_circular_hole, make_thick_cylinder, verify_t0


def run(tmp_path, script: str, answer: dict | None = None) -> dict:
    (tmp_path / "model.py").write_text(script)
    subprocess.run([sys.executable, "model.py"], cwd=tmp_path, check=False, capture_output=True)
    if answer is None:
        answer = json.loads((tmp_path / "answer.json").read_text())
    return answer


@pytest.fixture(scope="module")
def cyl():
    return make_thick_cylinder(0)


@pytest.fixture(scope="module")
def hole():
    return make_circular_hole(1)


def test_analytic_lame_limits():
    # Thin-wall limit: u = p a^2 (1 - nu^2) / (E t) for b -> a.
    a, t, p, e, nu = 2.0, 0.002, 100.0, 1e6, 0.3
    u = analytic.lame_radial_displacement(a, a, a + t, p, e, nu)
    assert u == pytest.approx(p * a * a * (1 - nu**2) / (e * t), rel=2e-3)


def test_analytic_kirsch_far_field():
    # Far from the hole the displacement tends to the uniform-strain field.
    a, s, e, nu = 1.0, -1000.0, 1e6, 0.25
    r = 1e4
    ur, _ = analytic.kirsch_displacement(r, 0.0, a, s, 0.0, e, nu)
    eps_x = s * (1 - nu**2) / e   # plane strain, uniaxial stress
    assert ur / r == pytest.approx(eps_x, rel=1e-3)


@pytest.mark.parametrize("fixture", ["cyl", "hole"])
def test_oracle_scores_one(fixture, request, tmp_path):
    task = request.getfixturevalue(fixture)
    ans = run(tmp_path, task.oracle_model)
    r = verify_t0(task.spec, task.meta, ans, tmp_path)
    assert r["reward"] == pytest.approx(1.0), r
    for q in r["quantities"].values():
        assert q["rel_error"] < 0.005


def test_plane_stress_model_fails(cyl, tmp_path):
    ans = run(tmp_path, cyl.oracle_model.replace('"PlaneStrain"', '"PlaneStress"'))
    # At the outer surface the two assumptions happen to differ by < 1% for this nu; the inner
    # surface error (~4%) gets no credit.
    assert verify_t0(cyl.spec, cyl.meta, ans, tmp_path)["reward"] <= 0.5


def test_small_domain_is_penalized(hole, tmp_path):
    s = hole.spec
    script = _oracle_annulus(s["hole_radius"], 2.0 * s["hole_radius"], s["youngs_modulus"],
                             s["poisson_ratio"], 0.0, ("stress", s["far_field_stress_x"], s["far_field_stress_y"]))
    ans = run(tmp_path, script)
    assert verify_t0(hole.spec, hole.meta, ans, tmp_path)["reward"] < 1.0


def test_answer_without_model_scores_zero(cyl, tmp_path):
    exact = analytic.lame_radial_displacement(1.0, 1.0, 2.0, 1.0, 1.0, 0.3)
    ans = {"quantities": {"u_inner": {"node": 1, "value": exact}, "u_outer": {"node": 2, "value": exact}}}
    run(tmp_path, "print('no model')\n", ans)
    assert verify_t0(cyl.spec, cyl.meta, ans, tmp_path)["reward"] == 0.0


def test_setnodedisp_cheat_is_ignored(cyl, tmp_path):
    """Wrong model (plane stress), then overwrite displacements with the closed-form answer."""
    s = cyl.spec
    wrong_model = cyl.oracle_model.replace('"PlaneStrain"', '"PlaneStress"')
    cheat = wrong_model.replace('json.dump(', f'''
for n in ops.getNodeTags():
    x, y = ops.nodeCoord(n)
    r = np.hypot(x, y)
    a_, b_, p_, E_, nu_ = {s["inner_radius"]}, {s["outer_radius"]}, {s["internal_pressure"]}, {s["youngs_modulus"]}, {s["poisson_ratio"]}
    A = p_ * a_**2 / (b_**2 - a_**2); B = p_ * a_**2 * b_**2 / (b_**2 - a_**2)
    u = (1 + nu_) / E_ * ((1 - 2 * nu_) * A * r + B / r)
    ops.setNodeDisp(n, 1, u * x / r, "-commit")
    ops.setNodeDisp(n, 2, u * y / r, "-commit")
json.dump(''', 1)
    ans = run(tmp_path, cheat)
    # The cheat's own report matches the closed form ...
    for name, v in ans["quantities"].items():
        r = cyl.spec["inner_radius"] if name == "u_inner" else cyl.spec["outer_radius"]
        exact = analytic.lame_radial_displacement(r, s["inner_radius"], s["outer_radius"],
                                                  s["internal_pressure"], s["youngs_modulus"], s["poisson_ratio"])
        assert v["value"] == pytest.approx(exact, rel=1e-6)
    # ... but the verifier solves the model itself and finds the plane-stress error.
    res = verify_t0(cyl.spec, cyl.meta, ans, tmp_path)
    assert res["reward"] <= 0.5
    assert any("differs" in f for f in res["flags"])


def test_wrong_node_location_is_flagged(hole, tmp_path):
    ans = run(tmp_path, hole.oracle_model)
    ans["quantities"]["u_r_theta0"]["node"] = ans["quantities"]["u_r_theta90"]["node"]
    res = verify_t0(hole.spec, hole.meta, ans, tmp_path)
    assert any("not at the requested location" in f for f in res["flags"])
    assert res["reward"] <= 0.5


def test_inflated_claim_halves_score(cyl, tmp_path):
    ans = run(tmp_path, cyl.oracle_model)
    for v in ans["quantities"].values():
        v["value"] *= 1.2
    res = verify_t0(cyl.spec, cyl.meta, ans, tmp_path)
    assert res["reward"] == pytest.approx(0.5)


# --- plastic: strip footing (Prandtl) -------------------------------------------------------


@pytest.fixture(scope="module")
def foot():
    from geogym.fea.problems import make_strip_footing
    return make_strip_footing(0)


@pytest.mark.slow
def test_footing_oracle(foot, tmp_path):
    ans = run(tmp_path, foot.oracle_model)
    r = verify_t0(foot.spec, foot.meta, ans, tmp_path)
    assert r["reward"] == pytest.approx(1.0), r
    assert r["rel_error"] < 0.03


@pytest.mark.slow
def test_footing_volumetric_locking_fails(foot, tmp_path):
    """Plain 4-node quads lock under incompressible plastic flow and overpredict q_ult."""
    script = foot.oracle_model.replace(
        'ops.element("bbarQuad", e, tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1), 1.0, 1)',
        'ops.element("quad", e, tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1), 1.0, "PlaneStrain", 1)')
    assert script != foot.oracle_model
    ans = run(tmp_path, script)
    r = verify_t0(foot.spec, foot.meta, ans, tmp_path)
    assert r["reward"] == 0.0 and r["q_fe"] > 1.10 * r["q_exact"]


@pytest.mark.slow
def test_footing_insufficient_settlement(foot, tmp_path):
    script = foot.oracle_model.replace("delta = max(0.03, 25.0 * su / E) * B", "delta = 0.1 * su / E * B")
    assert script != foot.oracle_model
    ans = run(tmp_path, script)
    r = verify_t0(foot.spec, foot.meta, ans, tmp_path)
    assert r["reward"] == 0.0


def test_footing_wrong_width_is_rejected(foot, tmp_path):
    ans = {"footing_nodes": [1, 2], "half_model": False, "q_ult": 100.0}
    run(tmp_path, foot.oracle_model, ans)
    r = verify_t0(foot.spec, foot.meta, ans, tmp_path)
    assert r["reward"] == 0.0
