"""T0 problem generators, oracle models, and verifiers.

Problems (plane strain, linear elastic, units kN and m):
- ``thick-cylinder``: pressurized thick-walled cylinder; radial displacement at the inner and
  outer surfaces (Lame).
- ``circular-hole``: traction-free circular hole in a plate under far-field normal stresses;
  radial displacement of the hole wall at theta = 0 and 90 degrees (Kirsch).

Answer (``/app/answer.json``)::

    {"quantities": {"<name>": {"node": <tag>, "value": <float>}, ...}}

plus the model script ``/app/model.py``. The verifier records the script's OpenSeesPy calls, replays
only the model definition, solves it itself, and reads the displacement of the named nodes.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analytic
from .sandbox import record, replay

# Relative error: full credit at <= 1%, none at >= 3%. A converged model of a closed-form problem
# lands well inside 1% (the oracles are within 0.15%); modeling mistakes such as plane stress
# instead of plane strain or a truncated domain show up as errors of several percent.
FULL_TOL, ZERO_TOL = 0.01, 0.03
MAX_NODES = 200_000


@dataclass
class FeaTask:
    family: str
    task_id: str
    spec: dict
    instruction: str
    meta: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    oracle_model: str = ""


RULES = """**Modeling rules.**
- Build the model with **OpenSeesPy** (`import openseespy.opensees as ops`; installed in the
  environment) in a script `/app/model.py`. Plane strain, consistent units (kN, m, kPa).
- Apply all loads in load patterns with a `Linear` (or `Constant`) time series so that the full load
  acts at pseudo-time 1. Define the model fully in the script; you may run any analysis you like.
- The grader re-runs your script, keeps only the **model definition** (nodes, elements, materials,
  boundary conditions, constraints, loads), solves it with its own analysis (linear static unless
  stated otherwise below), and
  reads the displacements of the nodes you name below. Numbers you compute outside the model do not
  count. Keep the model under 200,000 nodes; the script must finish within 10 minutes.
"""


def _lin(err: float) -> float:
    if err <= FULL_TOL:
        return 1.0
    if err >= ZERO_TOL:
        return 0.0
    return (ZERO_TOL - err) / (ZERO_TOL - FULL_TOL)


# ---------------------------------------------------------------------------------------------
# thick-walled cylinder
# ---------------------------------------------------------------------------------------------


def make_thick_cylinder(seed: int) -> FeaTask:
    rng = np.random.default_rng(30_000 + seed)
    a = float(np.round(rng.uniform(0.5, 3.0), 2))
    b = float(np.round(a * rng.uniform(1.5, 4.0), 2))
    p = float(np.round(rng.uniform(500.0, 20_000.0), -1))
    e = float(np.round(rng.uniform(5e5, 3e7), -4))
    nu = float(np.round(rng.uniform(0.15, 0.4), 2))
    spec = {"problem": "thick-cylinder", "inner_radius": a, "outer_radius": b,
            "internal_pressure": p, "youngs_modulus": e, "poisson_ratio": nu}
    instruction = f"""# FEA: pressurized thick-walled cylinder

A long thick-walled cylinder (a pressure tunnel lining idealized as linear elastic) has inner radius
{a} m and outer radius {b} m. It carries an internal pressure of {p} kPa; the outer surface is
traction-free. Material: Young's modulus {e:.4g} kPa, Poisson's ratio {nu}. Long cylinder: plane strain.

Compute, with a finite-element model, the **radial displacement** of the inner surface and of the
outer surface.

{RULES}
**Deliverable.** `/app/model.py` and `/app/answer.json`:

```json
{{"quantities": {{"u_inner": {{"node": 12, "value": 0.0012}},
                "u_outer": {{"node": 87, "value": 0.0004}}}}}}
```

`node` is the tag of a model node on that surface; `value` is its radial displacement (m, outward
positive) from your analysis.
"""
    meta = {"targets": {"u_inner": {"r": a}, "u_outer": {"r": b}}}
    oracle = _oracle_annulus(a, b, e, nu, inner_pressure=p, outer="free")
    return FeaTask("t0-fea", f"t0-cyl-{seed:05d}", spec, instruction, meta, {}, oracle)


# ---------------------------------------------------------------------------------------------
# circular hole in a plate (Kirsch)
# ---------------------------------------------------------------------------------------------


def make_circular_hole(seed: int) -> FeaTask:
    rng = np.random.default_rng(40_000 + seed)
    a = float(np.round(rng.uniform(1.0, 6.0), 2))
    sv = float(np.round(rng.uniform(500.0, 10_000.0), -1))
    k0 = float(np.round(rng.uniform(0.3, 2.0), 2))
    sx, sy = -k0 * sv, -sv  # compressive far-field stresses (tension positive)
    e = float(np.round(rng.uniform(5e5, 3e7), -4))
    nu = float(np.round(rng.uniform(0.15, 0.4), 2))
    spec = {"problem": "circular-hole", "hole_radius": a, "far_field_stress_x": sx,
            "far_field_stress_y": sy, "youngs_modulus": e, "poisson_ratio": nu}
    instruction = f"""# FEA: circular opening under far-field stress

A circular opening of radius {a} m (traction-free wall) sits in an infinite linear-elastic medium
loaded by uniform far-field normal stresses sigma_xx = {sx:.1f} kPa and sigma_yy = {sy:.1f} kPa
(tension positive, so both are compressive). Material: Young's modulus {e:.4g} kPa, Poisson's ratio
{nu}. Plane strain. The opening is centred at the origin.

Compute, with a finite-element model, the **radial displacement of the opening wall** at the point
on the positive x-axis (theta = 0) and at the point on the positive y-axis (theta = 90 degrees).
Displacements are measured from the unloaded configuration of the medium containing the opening
(the far-field stresses are applied as loads; there is no in-situ stress initialization).

{RULES}
**Deliverable.** `/app/model.py` and `/app/answer.json`:

```json
{{"quantities": {{"u_r_theta0": {{"node": 1, "value": -0.0021}},
                "u_r_theta90": {{"node": 40, "value": -0.0035}}}}}}
```

`node` is the tag of the model node at that point of the opening wall; `value` is its radial
displacement (m, positive away from the centre) from your analysis.
"""
    meta = {"targets": {"u_r_theta0": {"r": a, "theta": 0.0}, "u_r_theta90": {"r": a, "theta": 90.0}}}
    oracle = _oracle_annulus(a, 30.0 * a, e, nu, inner_pressure=0.0, outer=("stress", sx, sy))
    return FeaTask("t0-fea", f"t0-hole-{seed:05d}", spec, instruction, meta, {}, oracle)


def analytic_value(spec: dict, name: str, r: float, theta: float) -> float:
    if spec["problem"] == "thick-cylinder":
        return float(analytic.lame_radial_displacement(
            r, spec["inner_radius"], spec["outer_radius"], spec["internal_pressure"],
            spec["youngs_modulus"], spec["poisson_ratio"]))
    if spec["problem"] == "circular-hole":
        ur, _ = analytic.kirsch_displacement(
            r, theta, spec["hole_radius"], spec["far_field_stress_x"], spec["far_field_stress_y"],
            spec["youngs_modulus"], spec["poisson_ratio"])
        return float(ur)
    raise ValueError(spec["problem"])


# ---------------------------------------------------------------------------------------------
# oracle model (quarter annulus, symmetry boundaries)
# ---------------------------------------------------------------------------------------------


def _oracle_annulus(a, b, e, nu, inner_pressure, outer, n_theta=48, n_r=60) -> str:
    return textwrap.dedent(f'''\
        """Oracle: quarter annulus a <= r <= b, 0 <= theta <= 90 deg, 4-node quads, symmetry BCs."""
        import json
        import numpy as np
        import openseespy.opensees as ops

        a, b, E, nu = {a!r}, {b!r}, {e!r}, {nu!r}
        p_in, outer = {inner_pressure!r}, {outer!r}
        nt, nr = {n_theta}, {n_r}
        # Radial grading: element size grows geometrically away from the inner surface.
        g = (b / a) ** (1.0 / nr)
        radii = a * g ** np.arange(nr + 1)
        thetas = np.linspace(0.0, np.pi / 2, nt + 1)

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        tag = lambda i, j: 1 + i * (nt + 1) + j          # i: radial index, j: angular index
        for i, r in enumerate(radii):
            for j, t in enumerate(thetas):
                ops.node(tag(i, j), r * np.cos(t), r * np.sin(t))
        ops.nDMaterial("ElasticIsotropic", 1, E, nu)
        k = 0
        for i in range(nr):
            for j in range(nt):
                k += 1
                ops.element("quad", k, tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1),
                            1.0, "PlaneStrain", 1)
        for i in range(nr + 1):
            ops.fix(tag(i, 0), 0, 1)     # theta = 0: u_y = 0
            ops.fix(tag(i, nt), 1, 0)    # theta = 90: u_x = 0

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        loads = {{}}

        def add(n, fx, fy):
            ox, oy = loads.get(n, (0.0, 0.0))
            loads[n] = (ox + fx, oy + fy)

        def edge(i, traction):
            """Consistent nodal loads for a traction field along the arc of ring i (chord segments)."""
            for j in range(nt):
                n1, n2 = tag(i, j), tag(i, j + 1)
                x1, y1 = ops.nodeCoord(n1)
                x2, y2 = ops.nodeCoord(n2)
                L = np.hypot(x2 - x1, y2 - y1)
                nx, ny = (y2 - y1) / L, -(x2 - x1) / L      # chord normal, pointing toward -r
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                if nx * xm + ny * ym > 0:
                    nx, ny = -nx, -ny
                tx, ty = traction(nx, ny)
                add(n1, tx * L / 2, ty * L / 2)
                add(n2, tx * L / 2, ty * L / 2)

        if p_in:
            # Inner surface: outward normal of the material points toward the centre (-r);
            # pressure pushes the material away from the centre.
            edge(0, lambda nx, ny: (-p_in * nx, -p_in * ny))
        if outer != "free":
            _, sx, sy = outer
            # Outer surface: outward normal is +r (flip the chord normal).
            edge(nr, lambda nx, ny: (sx * -nx, sy * -ny))
        for n, (fx, fy) in loads.items():
            ops.load(n, fx, fy)

        ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("UmfPack")
        ops.algorithm("Linear"); ops.integrator("LoadControl", 1.0); ops.analysis("Static")
        ops.analyze(1)

        def ur(n):
            x, y = ops.nodeCoord(n)
            ux, uy = ops.nodeDisp(n)
            return (ux * x + uy * y) / np.hypot(x, y)

        if outer == "free":
            q = {{"u_inner": tag(0, nt // 2), "u_outer": tag(nr, nt // 2)}}
        else:
            q = {{"u_r_theta0": tag(0, 0), "u_r_theta90": tag(0, nt)}}
        json.dump({{"quantities": {{k: {{"node": n, "value": ur(n)}} for k, n in q.items()}}}},
                  open("answer.json", "w"), indent=1)
        ''')


# ---------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------


def verify_t0(spec: dict, meta: dict, answer: dict, app_dir: str | Path | None = None) -> dict:
    if spec["problem"] == "strip-footing":
        return verify_footing(spec, meta, answer, app_dir)
    out = {"flags": [], "quantities": {}}
    script = Path(app_dir) / "model.py" if app_dir is not None else None
    try:
        q = {k: (int(v["node"]), float(v["value"])) for k, v in answer["quantities"].items()}
    except (KeyError, TypeError, ValueError):
        out.update(reward=0.0, flags=["malformed answer"])
        return out
    if script is None or not script.exists():
        out.update(reward=0.0, flags=["no /app/model.py"])
        return out
    targets = meta["targets"]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "model.py").write_text(script.read_text())
        rec = record(work / "model.py", work)
        out["script_status"] = rec["status"]
        if not rec["calls"]:
            out.update(reward=0.0, flags=["script issued no OpenSeesPy calls"])
            return out
        rep = replay(rec, [n for n, _ in q.values()], work)
    out["replay_status"] = rep.get("status")
    out["n_nodes"] = rep.get("n_nodes")
    if rep.get("status") != "ok":
        out.update(reward=0.0, flags=[f"model could not be solved: {rep.get('status')}"])
        return out
    if rep["n_nodes"] > MAX_NODES:
        out.update(reward=0.0, flags=["model exceeds the node limit"])
        return out

    scores = []
    for name, tgt in targets.items():
        if name not in q:
            out["flags"].append(f"missing quantity {name}")
            scores.append(0.0)
            continue
        node, claimed = q[name]
        info = rep["nodes"].get(str(node))
        if info is None:
            out["flags"].append(f"{name}: node {node} not in the model")
            scores.append(0.0)
            continue
        x, y = info["coord"][:2]
        ux, uy = info["disp"][:2]
        r = float(np.hypot(x, y))
        th = float(np.degrees(np.arctan2(y, x)))
        ok_r = abs(r - tgt["r"]) <= 0.005 * tgt["r"]
        ok_t = "theta" not in tgt or abs(th - tgt["theta"]) <= 1.0
        if not (ok_r and ok_t):
            out["flags"].append(f"{name}: node {node} is not at the requested location")
            scores.append(0.0)
            continue
        fe = (ux * x + uy * y) / r
        exact = analytic_value(spec, name, r, np.radians(th))
        err = abs(fe - exact) / abs(exact)
        consistent = abs(claimed - fe) <= 0.01 * abs(fe) + 1e-12
        if not consistent:
            out["flags"].append(f"{name}: reported value differs from the model's response")
        s = _lin(err) * (1.0 if consistent else 0.5)
        out["quantities"][name] = {"node": node, "fe": fe, "exact": exact, "rel_error": err,
                                   "claimed": claimed, "score": s}
        scores.append(s)
    out["reward"] = float(np.mean(scores)) if scores else 0.0
    return out


# ---------------------------------------------------------------------------------------------
# strip footing on undrained clay (Prandtl): elastic-plastic, displacement controlled
# ---------------------------------------------------------------------------------------------

PRANDTL = 2.0 + np.pi
FOOT_FULL, FOOT_ZERO = 0.03, 0.10   # plastic limit loads: FE overestimates by a few percent at best
REPLAY_STEPS = 200


def make_strip_footing(seed: int) -> FeaTask:
    rng = np.random.default_rng(50_000 + seed)
    b = float(np.round(rng.uniform(1.0, 4.0), 2))
    su = float(np.round(rng.uniform(20.0, 150.0), 0))
    e = float(np.round(su * rng.uniform(300.0, 800.0), -2))   # typical E_u / s_u for clays
    nu = 0.49
    spec = {"problem": "strip-footing", "footing_width": b, "undrained_strength": su,
            "youngs_modulus": e, "poisson_ratio": nu}
    instruction = f"""# FEA: bearing capacity of a strip footing on clay

A rigid, smooth strip footing of width B = {b} m rests on the surface of a deep, homogeneous layer
of saturated clay loaded quickly (undrained). Model the clay as elastic-perfectly plastic with a
von Mises (J2) yield surface calibrated so that its yield stress in pure shear equals the undrained
shear strength s_u = {su} kPa; undrained Young's modulus {e:.4g} kPa, Poisson's ratio {nu}. Ignore
the self-weight of the clay (weightless soil, no surcharge). Plane strain.

Compute, with a finite-element model, the **ultimate bearing pressure** q_ult of the footing
(average vertical contact pressure at collapse).

{RULES}- This is a path-dependent problem. Load the footing by **prescribing its vertical
  displacement** (e.g. `ops.sp(node, 2, -delta)` in a `Linear` load pattern) on every footing node,
  with a total settlement large enough to reach collapse. The grader replays your model definition
  with its own incremental Newton analysis ({REPLAY_STEPS} steps to pseudo-time 1, with step cutting)
  and sums the vertical reactions of the footing nodes you list.
- You may model half of the problem using symmetry.

**Deliverable.** `/app/model.py` and `/app/answer.json`:

```json
{{"footing_nodes": [1, 2, 3], "half_model": true, "q_ult": 250.0}}
```

`footing_nodes`: tags of all nodes under the footing (the nodes whose displacement you prescribe);
`half_model`: whether the model covers half the footing width; `q_ult`: the ultimate bearing
pressure from your analysis (kPa, positive in compression).
"""
    meta = {"q_exact": PRANDTL * su}
    oracle = _oracle_footing(b, su, e, nu)
    return FeaTask("t0-fea", f"t0-foot-{seed:05d}", spec, instruction, meta, {}, oracle)


def _oracle_footing(b, su, e, nu, nf=20, ratio=1.1, extent=5.0) -> str:
    return textwrap.dedent(f'''\
        """Oracle: half model, smooth rigid footing by prescribed settlement, bbarQuad + J2."""
        import json
        import numpy as np
        import openseespy.opensees as ops

        B, su, E, nu = {b!r}, {su!r}, {e!r}, {nu!r}
        nf, ratio, extent = {nf}, {ratio}, {extent}
        h = B / 2 / nf
        xs = list(np.linspace(0.0, B / 2, nf + 1)); dx = h
        while xs[-1] < extent * B:
            dx *= ratio; xs.append(xs[-1] + dx)
        ys = [0.0]; dy = h
        while ys[-1] < extent * B:
            dy *= ratio; ys.append(ys[-1] + dy)
        ys = sorted(-np.array(ys))
        nx, ny = len(xs) - 1, len(ys) - 1
        tag = lambda i, j: 1 + j * (nx + 1) + i

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                ops.node(tag(i, j), float(x), float(y))
        K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
        sy = np.sqrt(3.0) * su          # von Mises: shear yield = sy / sqrt(3) = su
        ops.nDMaterial("J2Plasticity", 1, K, G, sy, sy, 0.0, 0.0)
        e = 0
        for j in range(ny):
            for i in range(nx):
                e += 1
                ops.element("bbarQuad", e, tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1), 1.0, 1)
        for i in range(nx + 1):
            ops.fix(tag(i, 0), 1, 1)                    # base
        for j in range(1, ny + 1):
            ops.fix(tag(0, j), 1, 0)                    # symmetry plane
            ops.fix(tag(nx, j), 1, 0)                   # far boundary
        foot = [tag(i, ny) for i in range(nf + 1)]
        delta = max(0.03, 25.0 * su / E) * B           # enough settlement to reach collapse
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for n in foot:
            ops.sp(n, 2, -delta)

        ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("UmfPack")
        ops.test("NormDispIncr", 1e-8, 60); ops.algorithm("Newton"); ops.analysis("Static")
        q = 0.0
        steps = 200
        ops.integrator("LoadControl", 1.0 / steps)
        for _ in range(steps):
            if ops.analyze(1) != 0:
                break
            ops.reactions()
            q = max(q, abs(sum(ops.nodeReaction(n, 2) for n in foot)) / (B / 2))
        json.dump({{"footing_nodes": foot, "half_model": True, "q_ult": q}}, open("answer.json", "w"))
        ''')


def verify_footing(spec: dict, meta: dict, answer: dict, app_dir) -> dict:
    out = {"flags": [], "q_exact": meta["q_exact"]}
    script = Path(app_dir) / "model.py" if app_dir is not None else None
    try:
        nodes = [int(n) for n in answer["footing_nodes"]]
        half = bool(answer["half_model"])
        claimed = float(answer["q_ult"])
    except (KeyError, TypeError, ValueError):
        out.update(reward=0.0, flags=["malformed answer"])
        return out
    if not nodes:
        out.update(reward=0.0, flags=["no footing nodes"])
        return out
    if script is None or not script.exists():
        out.update(reward=0.0, flags=["no /app/model.py"])
        return out
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "model.py").write_text(script.read_text())
        rec = record(work / "model.py", work)
        out["script_status"] = rec["status"]
        if not rec["calls"]:
            out.update(reward=0.0, flags=["script issued no OpenSeesPy calls"])
            return out
        rep = replay(rec, nodes, work, timeout=900.0,
                     mode={"type": "incremental", "steps": REPLAY_STEPS, "reaction_nodes": nodes, "dof": 2})
    out["replay_status"] = rep.get("status")
    out["n_nodes"] = rep.get("n_nodes")
    hist = rep.get("reaction_history") or []
    if rep.get("status") != "ok" and not hist:
        out.update(reward=0.0, flags=[f"model could not be solved: {rep.get('status')}"])
        return out
    if rep.get("status") != "ok":
        out["flags"].append(f"replay stopped early ({rep.get('status')}); using the load reached")
    if (rep.get("n_nodes") or 0) > MAX_NODES:
        out.update(reward=0.0, flags=["model exceeds the node limit"])
        return out

    # Footing geometry check: listed nodes lie on one horizontal line and span the footing width.
    b = spec["footing_width"]
    coords = [rep["nodes"].get(str(n), {}).get("coord") for n in nodes]
    if any(c is None for c in coords):
        out.update(reward=0.0, flags=["footing node not in the model"])
        return out
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    width = b / 2 if half else b
    span = xs.max() - xs.min()
    if np.ptp(ys) > 1e-6 * max(1.0, b) or abs(span - width) > 0.01 * width:
        out.update(reward=0.0, flags=["footing nodes do not form the footing base of the stated width"])
        return out

    q_fe = max(abs(f) for _, f in hist) / width
    exact = meta["q_exact"]
    err = abs(q_fe - exact) / exact
    if len(hist) >= 10:
        tail = [abs(f) for _, f in hist[int(0.8 * len(hist)):]]
        if (max(tail) - min(tail)) / max(tail) > 0.02:
            out["flags"].append("load still rising at the end of loading: collapse may not be reached")
    consistent = abs(claimed - q_fe) <= 0.03 * q_fe
    if not consistent:
        out["flags"].append("reported q_ult differs from the model's response")
    if err <= FOOT_FULL:
        s = 1.0
    elif err >= FOOT_ZERO:
        s = 0.0
    else:
        s = (FOOT_ZERO - err) / (FOOT_ZERO - FOOT_FULL)
    s *= 1.0 if consistent else 0.5
    out.update(q_fe=q_fe, rel_error=err, claimed=claimed, reward=float(s))
    return out


MAKERS = {"thick-cylinder": make_thick_cylinder, "circular-hole": make_circular_hole,
          "strip-footing": make_strip_footing}


def make_t0(seed: int) -> FeaTask:
    kind = ["thick-cylinder", "circular-hole", "strip-footing"][seed % 3]
    return MAKERS[kind](seed)
