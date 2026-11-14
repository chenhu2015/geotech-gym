"""T0 with OpenFOAM: the circular-opening (Kirsch) problem solved with solidDisplacementFoam.

The agent writes an OpenFOAM case under ``/app/case`` (``0/``, ``constant/``, ``system/``), meshed with
``blockMesh``, with the opening's boundary patch named ``hole``. The verifier copies only the case
files it needs, rejects any dictionary that could run code (``#codeStream``, ``#calc`` and every
other ``#`` directive, ``coded`` objects, ``libs``, ``functions``), runs ``blockMesh``,
``solidDisplacementFoam`` and ``writeCellCentres`` itself, and reads the displacement on the ``hole``
patch at the face nearest each requested point. Numbers in ``answer.json`` only serve the honesty
check.
"""

from __future__ import annotations

import inspect
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from ..fea.problems import FeaTask, _lin, analytic_value, make_circular_hole
from . import case as case_mod
from . import reader as reader_mod
from .reader import _strip_comments, patch_vectors

CASE_FILES = ["system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution",
              "constant/mechanicalProperties", "constant/thermalProperties", "0/D", "0/T"]
FORBIDDEN = re.compile(r"#|\bcode\w*|\blibs\b|\bfunctions\b|\bsystemCall\b|\bdynamicCode\b")
MAX_CELLS = 200_000
ANGLE_TOL_DEG = 1.0
RADIUS_TOL = 0.01

RULES = """**Modeling rules.**
- Use **OpenFOAM** (ESI v2412, installed: run `source /usr/lib/openfoam/openfoam2412/etc/bashrc` once
  per shell) with the solver **solidDisplacementFoam**. Put the case in `/app/case`, mesh it with
  **blockMesh** only, and name the boundary patch on the opening wall **`hole`**. Units: kN, m, kPa
  (set `E` in kPa; `rho` is not used by the static solution).
- The grader copies only these case files: `system/blockMeshDict`, `system/controlDict`,
  `system/fvSchemes`, `system/fvSolution`, `constant/mechanicalProperties`,
  `constant/thermalProperties` (with `thermalStress no;`), `0/D` and `0/T`. Dictionaries must not use
  `#` directives (`#include`, `#codeStream`, `#calc`, ...), `coded` objects, `libs` or `functions`.
- The grader re-runs `blockMesh`, `solidDisplacementFoam` and `postProcess -func writeCellCentres`
  in a clean environment and reads the displacement `D` on the `hole` patch, at its last time step,
  at the face whose centre is nearest each point listed below. That face must lie within 1 degree
  of the point (so the mesh must be reasonably fine around the opening). Make sure the solution has
  converged at the last time step. Keep the mesh under 200,000 cells; the run must finish within
  10 minutes.
"""


def make_foam_hole(seed: int) -> FeaTask:
    base = make_circular_hole(seed)
    s = base.spec
    a = s["hole_radius"]
    instruction = f"""# FEA (OpenFOAM): circular opening under far-field stress

A circular opening of radius {a} m (traction-free wall), centred at the origin, sits in an infinite
linear-elastic medium loaded by uniform far-field normal stresses sigma_xx =
{s["far_field_stress_x"]:.1f} kPa and sigma_yy = {s["far_field_stress_y"]:.1f} kPa (tension positive,
so both are compressive). Material: Young's modulus {s["youngs_modulus"]:.4g} kPa, Poisson's ratio
{s["poisson_ratio"]}. Plane strain. Displacements are measured from the unloaded configuration of
the medium containing the opening (the far-field stresses are applied as loads; there is no in-situ
stress initialization).

Compute, with a finite-volume stress analysis, the **radial displacement of the opening wall** at
(x, y) = ({a}, 0) and at (x, y) = (0, {a}).

{RULES}
**Deliverable.** The case in `/app/case` and `/app/answer.json` with your computed values (m,
positive away from the centre):

```json
{{"quantities": {{"u_r_theta0": -0.0021, "u_r_theta90": -0.0035}}}}
```
"""
    meta = {"targets": {"u_r_theta0": {"r": a, "theta": 0.0}, "u_r_theta90": {"r": a, "theta": 90.0}}}
    return FeaTask("t0-foam", f"t0f-hole-{seed:05d}", s, instruction, meta, {}, _oracle(s))


def make_t0_foam(seed: int) -> FeaTask:
    return make_foam_hole(seed)


# ---------------------------------------------------------------------------------------------
# oracle: self-contained script (case writer and reader inlined), then run OpenFOAM
# ---------------------------------------------------------------------------------------------


def _oracle(s: dict) -> str:
    a = s["hole_radius"]
    lib = inspect.getsource(case_mod) + "\n" + inspect.getsource(reader_mod)
    lib = lib.replace("from __future__ import annotations\n", "")
    return "from __future__ import annotations\n" + lib + textwrap.dedent(f'''

        import json, subprocess
        CASE = Path("/app/case")
        write_kirsch_case(CASE, a={a!r}, L={80.0 * a!r}, sx={s["far_field_stress_x"]!r},
                          sy={s["far_field_stress_y"]!r}, e={s["youngs_modulus"]!r}, nu={s["poisson_ratio"]!r},
                          n_ring=60, n_quarter=80, n_far=120, far_grading=40.0, end_time=8000)
        foam = "source /usr/lib/openfoam/openfoam2412/etc/bashrc >/dev/null 2>&1; cd /app/case && "
        for cmd in ("blockMesh", "solidDisplacementFoam", "postProcess -func writeCellCentres -latestTime"):
            subprocess.run(["bash", "-c", foam + cmd + " > log." + cmd.split()[0]], check=True)
        last = max((p for p in CASE.iterdir() if p.name.replace(".", "").isdigit() and float(p.name) > 0),
                   key=lambda p: float(p.name))
        D, C = patch_vectors(last / "D", "hole"), patch_vectors(last / "C", "hole")
        th = np.degrees(np.arctan2(C[:, 1], C[:, 0]))
        ur = (D[:, 0] * C[:, 0] + D[:, 1] * C[:, 1]) / np.hypot(C[:, 0], C[:, 1])
        q = {{"u_r_theta0": float(ur[np.argmin(np.abs(th))]), "u_r_theta90": float(ur[np.argmin(np.abs(th - 90))])}}
        json.dump({{"quantities": q}}, open("/app/answer.json", "w"))
        print(q)
    ''')


# ---------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------


def check_case(case: Path) -> list[str]:
    problems = []
    for rel in CASE_FILES:
        f = case / rel
        if not f.exists():
            if rel != "0/T":
                problems.append(f"missing {rel}")
            continue
        text = _strip_comments(f.read_text(errors="replace"))
        m = FORBIDDEN.search(text)
        if m:
            problems.append(f"{rel}: '{m.group(0)}' is not allowed")
    return problems


def run_case(case: Path, timeout: int = 600) -> dict:
    """Run blockMesh, solidDisplacementFoam and writeCellCentres on a copy of the case."""
    bashrc = Path("/usr/lib/openfoam/openfoam2412/etc/bashrc")
    if not bashrc.exists():
        return {"status": "OpenFOAM not installed (run inside the OpenFOAM image)"}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "case"
        for rel in CASE_FILES:
            if (case / rel).exists():
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(case / rel, work / rel)
        steps = ["blockMesh", "solidDisplacementFoam", "postProcess -func writeCellCentres -latestTime"]
        for cmd in steps:
            log = work / f"log.{cmd.split()[0]}"
            try:
                p = subprocess.run(["bash", "-c", f"source {bashrc} >/dev/null 2>&1; cd {work} && {cmd} > {log} 2>&1"],
                                   timeout=timeout)
            except subprocess.TimeoutExpired:
                return {"status": f"timeout in {cmd.split()[0]}"}
            if p.returncode != 0:
                tail = log.read_text(errors="replace")[-2000:] if log.exists() else ""
                return {"status": f"{cmd.split()[0]} failed", "log_tail": tail}
        m = re.search(r"nCells:\s*(\d+)", (work / "log.blockMesh").read_text())
        n_cells = int(m.group(1)) if m else None
        times = [p for p in work.iterdir() if re.fullmatch(r"[0-9.eE+-]+", p.name) and float(p.name) > 0]
        if not times:
            return {"status": "no time step written"}
        last = max(times, key=lambda p: float(p.name))
        try:
            d = patch_vectors(last / "D", "hole")
            c = patch_vectors(last / "C", "hole", n_expected=len(d))
        except (KeyError, ValueError, OSError) as e:
            return {"status": f"cannot read D on patch 'hole': {e}"}
        return {"status": "ok", "n_cells": n_cells, "time": last.name, "D": d, "C": c}


def verify_t0_foam(spec: dict, meta: dict, answer: dict, app_dir: str | Path | None = None) -> dict:
    out: dict = {"flags": [], "quantities": {}}
    case = (Path(app_dir) if app_dir is not None else Path("/app")) / "case"
    try:
        claims = {k: float(v) for k, v in answer["quantities"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return {**out, "reward": 0.0, "flags": ["malformed answer"]}
    problems = check_case(case)
    if problems:
        return {**out, "reward": 0.0, "flags": ["case rejected"] + problems[:10]}
    res = run_case(case)
    out["run_status"] = res["status"]
    if res["status"] != "ok":
        return {**out, "reward": 0.0, "flags": [f"case could not be run: {res['status']}"],
                "log_tail": res.get("log_tail", "")}
    out.update(n_cells=res["n_cells"], time=res["time"])
    if res["n_cells"] is None or res["n_cells"] > MAX_CELLS:
        return {**out, "reward": 0.0, "flags": ["mesh exceeds the cell limit"]}
    c, d = res["C"][:, :2], res["D"][:, :2]
    r = np.hypot(c[:, 0], c[:, 1])
    th = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
    scores = []
    for name, tgt in meta["targets"].items():
        k = int(np.argmin(np.abs(th - tgt["theta"])))
        if abs(th[k] - tgt["theta"]) > ANGLE_TOL_DEG or abs(r[k] - tgt["r"]) > RADIUS_TOL * tgt["r"]:
            out["flags"].append(f"{name}: no 'hole' face within {ANGLE_TOL_DEG} deg of the point")
            scores.append(0.0)
            continue
        fe = float(d[k] @ c[k] / r[k])
        exact = analytic_value(spec, name, float(r[k]), float(np.radians(th[k])))
        err = abs(fe - exact) / abs(exact)
        claimed = claims.get(name)
        consistent = claimed is not None and abs(claimed - fe) <= 0.01 * abs(fe) + 1e-12
        if not consistent:
            out["flags"].append(f"{name}: reported value missing or differs from the model's response")
        s = _lin(err) * (1.0 if consistent else 0.5)
        out["quantities"][name] = {"fe": fe, "exact": exact, "rel_error": err, "claimed": claimed,
                                   "face_theta_deg": float(th[k]), "score": s}
        scores.append(s)
    out["reward"] = float(np.mean(scores)) if scores else 0.0
    return out
