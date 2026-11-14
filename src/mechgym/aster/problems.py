"""T0 with Code_Aster: the same closed-form problems as the OpenSeesPy family, modeled in Code_Aster.

The agent writes a command file ``/app/model.comm`` and a mesh file (``/app/mesh.mail``, Code_Aster's
native ASCII format, or ``/app/mesh.med``). The verifier checks the command file against a whitelist
(:mod:`mechgym.aster.deck`), runs it itself, takes the displacement field of the result ``resu`` at
the mesh nodes located at the requested points, and compares it with the analytical solution.
Numbers in ``answer.json`` only serve the honesty check.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from ..fea.problems import FeaTask, _lin, analytic_value, make_circular_hole, make_thick_cylinder
from .deck import check_deck, run_deck

MAX_NODES = 200_000
LOC_TOL = 0.005   # node must be within 0.5% of the radius from the requested point

RULES = """**Modeling rules.**
- Use **Code_Aster** (installed in the environment: run `. /opt/activate.sh` once per shell, then
  `run_aster /app/model.comm`). Write the command file `/app/model.comm` and the mesh file it reads,
  either `/app/mesh.mail` (Code_Aster native ASCII format) or `/app/mesh.med`, through
  `DEFI_FICHIER(ACTION="ASSOCIER", UNITE=20, FICHIER="/app/mesh.mail", TYPE="ASCII")` and
  `LIRE_MAILLAGE(FORMAT="ASTER", UNITE=20)` (or `FORMAT="MED"`). No mesh generator is installed;
  write the mesh file yourself (e.g. with a Python script). Units: kN, m, kPa.
- Name the final result **`resu`** and compute it with `MECA_STATIQUE` (or `STAT_NON_LINE`).
- The grader re-runs your command file in a clean environment and reads the displacement of
  `resu` at the mesh nodes located at the points listed below, so your mesh must have a node at
  each of those points. The command file may only use Code_Aster commands that define and solve
  the model (plus `IMPR_RESU`, `IMPR_TABLE`, `POST_RELEVE_T`, `CALC_CHAMP`), basic arithmetic,
  loops and `_F`; no imports and no attribute access. Commands that create fields or results
  directly (`CREA_CHAMP`, `CREA_RESU`, `LIRE_RESU`, ...) are rejected. Keep the mesh under 200,000
  nodes; the run must finish within 10 minutes.
"""


def make_aster_cylinder(seed: int) -> FeaTask:
    base = make_thick_cylinder(seed)
    s = base.spec
    a, b = s["inner_radius"], s["outer_radius"]
    instruction = f"""# FEA (Code_Aster): pressurized thick-walled cylinder

A long thick-walled cylinder (a pressure tunnel lining idealized as linear elastic) has inner radius
{a} m and outer radius {b} m, centred at the origin. It carries an internal pressure of
{s["internal_pressure"]} kPa; the outer surface is traction-free. Material: Young's modulus
{s["youngs_modulus"]:.4g} kPa, Poisson's ratio {s["poisson_ratio"]}. Long cylinder: plane strain.

Compute, with a finite-element model, the **radial displacement** of the inner surface at
(x, y) = ({a}, 0) and of the outer surface at (x, y) = ({b}, 0).

{RULES}
**Deliverable.** `/app/model.comm`, the mesh file, and `/app/answer.json` with your computed values
(m, outward positive):

```json
{{"quantities": {{"u_inner": 0.0012, "u_outer": 0.0004}}}}
```
"""
    meta = {"targets": {"u_inner": {"r": a, "theta": 0.0}, "u_outer": {"r": b, "theta": 0.0}}}
    return FeaTask("t0-aster", f"t0a-cyl-{seed:05d}", s, instruction, meta, {},
                   _oracle(a, b, s["youngs_modulus"], s["poisson_ratio"], pressure=s["internal_pressure"]))


def make_aster_hole(seed: int) -> FeaTask:
    base = make_circular_hole(seed)
    s = base.spec
    a = s["hole_radius"]
    instruction = f"""# FEA (Code_Aster): circular opening under far-field stress

A circular opening of radius {a} m (traction-free wall), centred at the origin, sits in an infinite
linear-elastic medium loaded by uniform far-field normal stresses sigma_xx =
{s["far_field_stress_x"]:.1f} kPa and sigma_yy = {s["far_field_stress_y"]:.1f} kPa (tension positive,
so both are compressive). Material: Young's modulus {s["youngs_modulus"]:.4g} kPa, Poisson's ratio
{s["poisson_ratio"]}. Plane strain. Displacements are measured from the unloaded configuration of
the medium containing the opening (the far-field stresses are applied as loads; there is no in-situ
stress initialization).

Compute, with a finite-element model, the **radial displacement of the opening wall** at
(x, y) = ({a}, 0) and at (x, y) = (0, {a}).

{RULES}
**Deliverable.** `/app/model.comm`, the mesh file, and `/app/answer.json` with your computed values
(m, positive away from the centre):

```json
{{"quantities": {{"u_r_theta0": -0.0021, "u_r_theta90": -0.0035}}}}
```
"""
    meta = {"targets": {"u_r_theta0": {"r": a, "theta": 0.0}, "u_r_theta90": {"r": a, "theta": 90.0}}}
    return FeaTask("t0-aster", f"t0a-hole-{seed:05d}", s, instruction, meta, {},
                   _oracle(a, 30.0 * a, s["youngs_modulus"], s["poisson_ratio"],
                           far_field=(s["far_field_stress_x"], s["far_field_stress_y"])))


def make_t0_aster(seed: int) -> FeaTask:
    return (make_aster_cylinder if seed % 2 == 0 else make_aster_hole)(seed)


# ---------------------------------------------------------------------------------------------
# oracle: a Python script that writes the mesh, the command file and the answer, then runs Code_Aster
# ---------------------------------------------------------------------------------------------


def _oracle(a, b, e, nu, pressure=0.0, far_field=None, n_r=60, n_t=48) -> str:
    """Quarter model with symmetry planes. Cylinder: annulus a..b. Hole: square plate of half-width b
    with a quarter hole, far-field tractions on the outer edges."""
    return textwrap.dedent(f'''\
        import json, os, subprocess
        import numpy as np

        a, b, e, nu = {a!r}, {b!r}, {e!r}, {nu!r}
        pressure, far_field = {pressure!r}, {far_field!r}
        n_r, n_t = {n_r}, {n_t}
        t = np.linspace(0.0, np.pi / 2, n_t + 1)
        if far_field is None:
            outer = np.stack([b * np.cos(t), b * np.sin(t)], 1)
        else:   # square outer boundary: x = b for theta <= 45 deg, y = b above
            outer = np.where((t <= np.pi / 4)[:, None],
                             np.stack([np.full_like(t, b), b * np.tan(t)], 1),
                             np.stack([b / np.tan(np.maximum(t, 1e-12)), np.full_like(t, b)], 1))
            outer[-1] = [0.0, b]
        s = (np.geomspace(1.0, 1.0 + 40.0, n_r + 1) - 1.0) / 40.0   # radial grading toward the hole
        nid = lambda i, j: j * (n_r + 1) + i + 1
        L = ["TITRE", " oracle", "FINSF", "COOR_2D"]
        for j in range(n_t + 1):
            p0 = a * np.array([np.cos(t[j]), np.sin(t[j])])
            for i in range(n_r + 1):
                x, y = p0 + s[i] * (outer[j] - p0)
                L.append(f" N{{nid(i, j)}} {{x:.15e}} {{y:.15e}}")
        L += ["FINSF", "QUAD4"]
        m = 0
        for j in range(n_t):
            for i in range(n_r):
                m += 1
                L.append(f" M{{m}} N{{nid(i, j)}} N{{nid(i + 1, j)}} N{{nid(i + 1, j + 1)}} N{{nid(i, j + 1)}}")
        groups = {{"INNER": [], "RIGHT": [], "TOP": [], "OUTER": []}}
        L += ["FINSF", "SEG2"]
        for j in range(n_t):
            m += 1; groups["INNER"].append(m)
            L.append(f" M{{m}} N{{nid(0, j + 1)}} N{{nid(0, j)}}")
            m += 1
            L.append(f" M{{m}} N{{nid(n_r, j)}} N{{nid(n_r, j + 1)}}")
            groups["OUTER"].append(m)
            groups["RIGHT" if t[j + 1] <= np.pi / 4 + 1e-12 else "TOP"].append(m)
        L += ["FINSF"]
        for g, ms in groups.items():
            if ms:
                L += ["GROUP_MA", " " + g] + [f" M{{k}}" for k in ms] + ["FINSF"]
        L += ["GROUP_NO", " YSYM"] + [f" N{{nid(i, 0)}}" for i in range(n_r + 1)] + ["FINSF"]
        L += ["GROUP_NO", " XSYM"] + [f" N{{nid(i, n_t)}}" for i in range(n_r + 1)] + ["FINSF", "FIN"]
        open("/app/mesh.mail", "w").write("\\n".join(L) + "\\n")

        if far_field is None:
            load = f'PRES_REP=_F(GROUP_MA="INNER", PRES={{pressure!r}}),'
        else:
            load = (f'FORCE_CONTOUR=(_F(GROUP_MA="RIGHT", FX={{far_field[0]!r}}), '
                    f'_F(GROUP_MA="TOP", FY={{far_field[1]!r}})),')
        comm = f"""DEBUT()
        DEFI_FICHIER(ACTION="ASSOCIER", UNITE=20, FICHIER="/app/mesh.mail", TYPE="ASCII")
        mesh = LIRE_MAILLAGE(FORMAT="ASTER", UNITE=20)
        model = AFFE_MODELE(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", PHENOMENE="MECANIQUE", MODELISATION="D_PLAN"))
        mat = DEFI_MATERIAU(ELAS=_F(E={{e!r}}, NU={{nu!r}}))
        fmat = AFFE_MATERIAU(MAILLAGE=mesh, AFFE=_F(TOUT="OUI", MATER=mat))
        load = AFFE_CHAR_MECA(MODELE=model,
                              DDL_IMPO=(_F(GROUP_NO="YSYM", DY=0.0), _F(GROUP_NO="XSYM", DX=0.0)),
                              {{load}})
        resu = MECA_STATIQUE(MODELE=model, CHAM_MATER=fmat, EXCIT=_F(CHARGE=load))
        FIN()
        """
        open("/app/model.comm", "w").write(comm)
        # Run the model and report the computed values (the grader recomputes them anyway).
        ''') + _ORACLE_REPORT


# The oracle runs Code_Aster on its own deck with a small epilogue and reports the values it reads.
_ORACLE_REPORT = textwrap.dedent('''\
    from pathlib import Path
    import tempfile
    epi = """
    import json as _j
    _f = resu.getField("DEPL", resu.getIndexes()[-1]).toSimpleFieldOnNodes()
    _v, _ = _f.getValues()
    _m = resu.getMesh(); _n = _m.getNumberOfNodes(); _c = list(_m.getCoordinates().getValues())
    _d = len(_c) // _n
    _j.dump({"v": [list(map(float, r)) for r in _v], "c": [_c[i*_d:(i+1)*_d] for i in range(_n)]}, open("/tmp/oracle_out.json", "w"))
    FIN()
    """
    src = Path("/app/model.comm").read_text().replace("FIN()\\n", "")
    Path("/tmp/oracle.comm").write_text(src + epi)
    subprocess.run(["run_aster", "/tmp/oracle.comm"], cwd="/tmp", capture_output=True)
    res = json.load(open("/tmp/oracle_out.json"))
    c, v = np.array(res["c"])[:, :2], np.array(res["v"])[:, :2]
    def ur(x, y):
        k = int(np.argmin(np.hypot(c[:, 0] - x, c[:, 1] - y)))
        return float(v[k] @ c[k] / np.hypot(*c[k]))
    if far_field is None:
        q = {"u_inner": ur(a, 0.0), "u_outer": ur(b, 0.0)}
    else:
        q = {"u_r_theta0": ur(a, 0.0), "u_r_theta90": ur(0.0, a)}
    json.dump({"quantities": q}, open("/app/answer.json", "w"))
    print(q)
''')


# ---------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------


def verify_t0_aster(spec: dict, meta: dict, answer: dict, app_dir: str | Path | None = None) -> dict:
    out: dict = {"flags": [], "quantities": {}}
    app = Path(app_dir) if app_dir is not None else Path("/app")
    try:
        claims = {k: float(v) for k, v in answer["quantities"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return {**out, "reward": 0.0, "flags": ["malformed answer"]}
    comm = app / "model.comm"
    if not comm.exists():
        return {**out, "reward": 0.0, "flags": ["no /app/model.comm"]}
    problems = check_deck(comm.read_text(), app_dir="/app")
    if problems:
        return {**out, "reward": 0.0, "flags": ["command file rejected"] + problems[:10]}
    res = run_deck(comm)
    out["run_status"] = res["status"]
    if res["status"] != "ok":
        return {**out, "reward": 0.0, "flags": [f"model could not be solved: {res['status']}"],
                "log_tail": res.get("log_tail", "")}
    coords = np.array(res["coords"], dtype=float)[:, :2]
    comps = res["components"]
    if "DX" not in comps or "DY" not in comps:
        return {**out, "reward": 0.0, "flags": ["result has no DX/DY displacement"]}
    vals = np.array(res["values"], dtype=float)
    disp = vals[:, [comps.index("DX"), comps.index("DY")]]
    out["n_nodes"] = len(coords)
    if len(coords) > MAX_NODES:
        return {**out, "reward": 0.0, "flags": ["mesh exceeds the node limit"]}

    scores = []
    for name, tgt in meta["targets"].items():
        th = np.radians(tgt["theta"])
        px, py = tgt["r"] * np.cos(th), tgt["r"] * np.sin(th)
        d = np.hypot(coords[:, 0] - px, coords[:, 1] - py)
        k = int(np.argmin(d))
        if d[k] > LOC_TOL * tgt["r"]:
            out["flags"].append(f"{name}: no mesh node at ({px:.4g}, {py:.4g})")
            scores.append(0.0)
            continue
        x, y = coords[k]
        r = float(np.hypot(x, y))
        fe = float(disp[k] @ coords[k] / r)
        exact = analytic_value(spec, name, r, float(np.arctan2(y, x)))
        err = abs(fe - exact) / abs(exact)
        claimed = claims.get(name)
        consistent = claimed is not None and abs(claimed - fe) <= 0.01 * abs(fe) + 1e-12
        if not consistent:
            out["flags"].append(f"{name}: reported value missing or differs from the model's response")
        s = _lin(err) * (1.0 if consistent else 0.5)
        out["quantities"][name] = {"fe": fe, "exact": exact, "rel_error": err, "claimed": claimed, "score": s}
        scores.append(s)
    out["reward"] = float(np.mean(scores)) if scores else 0.0
    return out
