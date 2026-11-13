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
  boundary conditions, constraints, loads), solves it with its own linear static analysis, and
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


MAKERS = {"thick-cylinder": make_thick_cylinder, "circular-hole": make_circular_hole}


def make_t0(seed: int) -> FeaTask:
    kind = ["thick-cylinder", "circular-hole"][seed % 2]
    return MAKERS[kind](seed)
