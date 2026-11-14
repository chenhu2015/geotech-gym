import copy

import numpy as np
import pytest

from mechgym.tasks import make_t1, make_t2, verify_t1, verify_t2


@pytest.fixture(scope="module")
def t1():
    return make_t1(1)


@pytest.fixture(scope="module")
def t2():
    return make_t2(0)


def test_t1_oracle_full_reward(t1):
    assert verify_t1(t1.spec, t1.meta, t1.reference)["reward"] == pytest.approx(1.0)


def test_t1_inflated_claim_is_flagged(t1):
    bad = copy.deepcopy(t1.reference)
    bad["fos"] *= 1.3
    r = verify_t1(t1.spec, t1.meta, bad)
    assert r["accuracy"] == 0.0 and any("does not match" in f for f in r["flags"])


def test_t1_noncritical_surface_scores_low(t1):
    bad = copy.deepcopy(t1.reference)
    bad["slip_surface"]["radius"] *= 1.6
    r = verify_t1(t1.spec, t1.meta, bad)
    assert r["reward"] < 0.6


def test_t1_malformed(t1):
    assert verify_t1(t1.spec, t1.meta, {"fos": "x"})["reward"] == 0.0


def test_t2_oracle_full_reward(t2):
    r = verify_t2(t2.spec, t2.meta, t2.reference)
    assert r["feasible"] and r["reward"] == pytest.approx(1.0)


def test_t2_unchanged_is_infeasible(t2):
    r = verify_t2(t2.spec, t2.meta, {"surface": t2.spec["surface"], "fos": 1.4})
    assert r["reward"] == 0.0 and not r["feasible"]


def test_t2_fill_is_rejected(t2):
    surf = copy.deepcopy(t2.reference["surface"])
    surf.insert(1, [surf[0][0] + 1.0, surf[0][1] + 5.0])
    r = verify_t2(t2.spec, t2.meta, {"surface": surf, "fos": 1.3})
    assert r["reward"] == 0.0 and any("fill" in f for f in r["flags"])


def test_t2_protected_zone_is_enforced(t2):
    a, b = t2.meta["protected"]
    surf = [list(p) for p in t2.reference["surface"]]
    xm = 0.5 * (a + b)
    zm = float(np.interp(xm, [p[0] for p in surf], [p[1] for p in surf]))
    surf.append([xm, zm - 1.0])
    surf.sort()
    r = verify_t2(t2.spec, t2.meta, {"surface": surf, "fos": 1.3})
    assert r["reward"] == 0.0 and any("protected" in f for f in r["flags"])


def test_t2_wasteful_design_scores_below_one(t2):
    # Deeper cut than the reference: still safe but more excavation.
    surf = [list(p) for p in t2.reference["surface"]]
    orig = t2.spec["surface"]
    xs = sorted({p[0] for p in surf} | {p[0] for p in orig})
    zr = np.interp(xs, [p[0] for p in surf], [p[1] for p in surf])
    z0 = np.interp(xs, [p[0] for p in orig], [p[1] for p in orig])
    a, b = t2.meta["protected"]
    deeper = [[x, float(z - (0.5 if (z0[i] - z > 1e-6 and not a <= x <= b) else 0.0))]
              for i, (x, z) in enumerate(zip(xs, zr))]
    r = verify_t2(t2.spec, t2.meta, {"surface": deeper, "fos": 1.35})
    if r["feasible"]:
        assert r["efficiency"] < 1.0


@pytest.fixture(scope="module")
def t3():
    from mechgym.tasks import make_t3
    return make_t3(0)


@pytest.mark.slow
def test_t3_oracle_and_lab_values(t3):
    from mechgym.tasks import verify_t3
    assert verify_t3(t3.spec, t3.meta, t3.reference)["reward"] == pytest.approx(1.0)
    m = t3.spec["layers"][0]["material"]
    naive = {"c": m["c"], "phi_deg": m["phi_deg"], "fos_drained": 2.0}
    r = verify_t3(t3.spec, t3.meta, naive)
    assert r["reward"] < 0.2 and r["fos_observed_surface"] > 1.2


@pytest.mark.slow
def test_t3_answer_is_hidden_from_agent(t3):
    import json
    text = json.dumps(t3.spec) + t3.instruction
    assert str(t3.meta["c_true"]) not in text
