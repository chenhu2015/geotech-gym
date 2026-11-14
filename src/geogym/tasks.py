"""Task families T1 (stability check) and T2 (slope design): instance generation, instructions,
reference solutions, and verifiers.

Verifier principles (docs/design.md): rebuild the model from the site spec, never trust numbers
the agent reports, re-analyze whatever the agent submits, and score with continuous components.
The analysis standard is stated in every instruction: Bishop's simplified method on circular
slip surfaces, pore pressure u = gamma_w * (z_w - z) below the phreatic line, no tension crack,
no seismic load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .generate import Site, random_site
from .le import Circle, InvalidSurface, evaluate, search_circular
from .profile import Polyline
from .spec import profile_from_spec

STANDARD = (
    "Use Bishop's simplified method with circular slip surfaces. Pore pressure below the phreatic "
    "line is u = gamma_w * (z_w - z) with gamma_w = 9.81 kN/m3 (no pore pressure above it). "
    "Ignore tension cracks and seismic loading. Slip surfaces may not pass below the model base."
)


def _lin_score(err: float, full: float, zero: float) -> float:
    """1 for err <= full, 0 for err >= zero, linear in between."""
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return float((zero - err) / (zero - full))


def _site_table(site: Site) -> str:
    rows = ["| Unit | Name | Bottom elevation (m) | c' (kPa) | phi' (deg) | gamma (kN/m3) |",
            "|---|---|---|---|---|---|"]
    for i, layer in enumerate(site.spec["layers"]):
        m = layer["material"]
        bottom = "model base" if layer["bottom"] is None else f"{layer['bottom'][0][1]:.2f}"
        rows.append(f"| {i + 1} | {layer['name']} | {bottom} | {m['c']} | {m['phi_deg']} | {m['gamma']} |")
    return "\n".join(rows)


def _site_paragraph(site: Site) -> str:
    side = "right" if site.faces_left else "left"
    text = (
        f"The site is a {site.height:.1f} m high cut slope inclined at {site.angle_deg:.1f} degrees. "
        f"The toe is at elevation {site.z_toe:.1f} m and the crest at {site.z_crest:.1f} m; the crest "
        f"lies on the {side} side of the section. Firm rock (the model base) is at elevation "
        f"{site.spec['base_z']:.1f} m. Soil units are horizontal:"
    )
    text += "\n\n" + _site_table(site) + "\n\n"
    if site.spec["water_table"] is not None:
        text += "A phreatic surface was measured in piezometers; its trace is given in `site.json`.\n"
    else:
        text += "No groundwater was encountered; analyze the slope as dry.\n"
    return text


def _to_internal_circle(prof, c: dict) -> Circle:
    xc, zc = float(c["center"][0]), float(c["center"][1])
    return Circle(-xc if prof.mirrored else xc, zc, float(c["radius"]))


def _to_site_circle(prof, circ: Circle) -> dict:
    xc = -circ.xc if prof.mirrored else circ.xc
    return {"center": [float(xc), float(circ.zc)], "radius": float(circ.r)}


# ---------------------------------------------------------------------------------------------
# T1 — stability check
# ---------------------------------------------------------------------------------------------


@dataclass
class Task:
    family: str
    task_id: str
    spec: dict
    instruction: str
    meta: dict = field(default_factory=dict)       # hidden from the agent (verifier only)
    reference: dict = field(default_factory=dict)  # oracle answer


def make_t1(seed: int, max_tries: int = 50) -> Task:
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        site = random_site(rng)
        prof = profile_from_spec(site.spec)
        try:
            res = search_circular(prof, "bishop")
        except InvalidSurface:
            continue
        if 0.7 <= res.fos <= 2.2:
            break
    else:
        raise RuntimeError("could not generate a T1 instance")
    instruction = f"""# Slope stability check

You are the geotechnical engineer on this project. Determine the factor of safety of the existing
cut slope and the critical slip surface.

{_site_paragraph(site)}
The full geometry (ground surface, unit boundaries, phreatic surface, model base) is in
`/app/site.json` (units: m, kPa, kN/m3). Coordinates are x horizontal, z up.

**Analysis standard.** {STANDARD}

**Deliverable.** Write `/app/answer.json`:

```json
{{"fos": 1.23, "slip_surface": {{"center": [x_c, z_c], "radius": r}}}}
```

in the same coordinates as `site.json`. Your answer will be checked by independently re-analyzing
the slip surface you report, so the factor of safety must be the one your surface actually gives,
and it should be the minimum over admissible circular surfaces.
"""
    meta = {"fos_reference": float(res.fos)}
    reference = {"fos": res.fos, "slip_surface": _to_site_circle(prof, res.surface)}
    return Task("t1-check", f"t1-{seed:05d}", site.spec, instruction, meta, reference)


def verify_t1(spec: dict, meta: dict, answer: dict) -> dict:
    prof = profile_from_spec(spec)
    f_ref = float(meta["fos_reference"])
    out = {"fos_reference": f_ref, "flags": []}
    try:
        f_claim = float(answer["fos"])
        circ = _to_internal_circle(prof, answer["slip_surface"])
    except (KeyError, TypeError, ValueError, IndexError):
        out.update(reward=0.0, flags=["malformed answer"])
        return out
    try:
        f_surface = evaluate(prof, circ, "bishop")
    except (InvalidSurface, ValueError) as e:
        f_surface = None
        out["flags"].append(f"reported surface is not admissible: {e}")
    out["fos_claimed"] = f_claim
    out["fos_of_reported_surface"] = f_surface

    consistent = f_surface is not None and abs(f_claim - f_surface) / f_surface <= 0.02
    if f_surface is not None and not consistent:
        out["flags"].append("claimed FoS does not match the reported surface")
    # If the agent found a more critical (verified) surface than our search, it becomes the reference.
    if f_surface is not None:
        f_ref = min(f_ref, f_surface)
    crit = _lin_score((f_surface - f_ref) / f_ref, 0.02, 0.10) if f_surface is not None else 0.0
    acc = _lin_score(abs(f_claim - f_ref) / f_ref, 0.02, 0.10) if consistent else 0.0
    out.update(criticality=crit, accuracy=acc, reward=0.5 * crit + 0.5 * acc, fos_reference_used=f_ref)
    return out


# ---------------------------------------------------------------------------------------------
# T2 — slope design (cut-only regrading under a protected crest structure)
# ---------------------------------------------------------------------------------------------


def _design_profile(spec: dict, surface: list):
    """Profile for a regraded ground line; the phreatic surface cannot rise above the new ground."""
    prof_new = profile_from_spec(spec, surface=surface)
    if spec.get("water_table") is not None:
        g = Polyline(surface)
        w = Polyline(spec["water_table"])
        xs = np.unique(np.concatenate([g.x, w.x, np.linspace(g.xmin, g.xmax, 401)]))
        clipped = np.column_stack([xs, np.minimum(w(xs), g(xs))])
        spec2 = dict(spec, water_table=clipped.tolist())
        prof_new = profile_from_spec(spec2, surface=surface)
    return prof_new


def _cut_area(orig: list, new: list) -> float:
    g0, g1 = Polyline(orig), Polyline(new)
    xs = np.unique(np.concatenate([g0.x, g1.x, np.linspace(g0.xmin, g0.xmax, 4001)]))
    return float(np.trapezoid(np.clip(g0(xs) - g1(xs), 0.0, None), xs))


def _plane_design(site: Site, d: float) -> list:
    """Single planar face from the toe to a point ``d`` metres behind the crest edge."""
    sgn = 1.0 if site.faces_left else -1.0
    x_top = site.x_crest + sgn * d
    pts = [p for p in site.spec["surface"]]
    toe = (site.x_toe, site.z_toe)
    top = (x_top, site.z_crest)
    if site.faces_left:
        new = [pts[0], list(toe), list(top), pts[-1]]
    else:
        new = [pts[0], list(top), list(toe), pts[-1]]
    return [list(map(float, p)) for p in new]


def make_t2(seed: int, target: float = 1.3, max_tries: int = 80) -> Task:
    rng = np.random.default_rng(10_000 + seed)
    for _ in range(max_tries):
        site = random_site(rng)
        prof = profile_from_spec(site.spec)
        try:
            f0 = search_circular(prof, "bishop").fos
        except InvalidSurface:
            continue
        if not (0.85 <= f0 <= target - 0.05):
            continue
        setback = float(np.round(max(1.5 * site.height, 10.0), 1))
        try:
            f_max = search_circular(_design_profile(site.spec, _plane_design(site, setback)), "bishop").fos
        except InvalidSurface:
            continue
        if f_max < target + 0.03:
            continue
        lo, hi = 0.0, setback
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            f_mid = search_circular(_design_profile(site.spec, _plane_design(site, mid)), "bishop").fos
            lo, hi = (lo, mid) if f_mid >= target else (mid, hi)
        ref_surface = _plane_design(site, hi)
        ref_area = _cut_area(site.spec["surface"], ref_surface)
        break
    else:
        raise RuntimeError("could not generate a T2 instance")

    sgn = 1.0 if site.faces_left else -1.0
    x_struct = site.x_crest + sgn * setback
    x_end = site.spec["surface"][-1][0]
    protected = [float(x_struct), float(x_end)] if site.faces_left else [0.0, float(x_struct)]
    instruction = f"""# Slope remediation design

You are the geotechnical engineer on this project. The existing cut slope does not meet the
required factor of safety of **{target:.2f}** (long-term, static). Design a regrading that meets
it with the least excavation.

{_site_paragraph(site)}
The full geometry is in `/app/site.json` (units: m, kPa, kN/m3). Coordinates are x horizontal, z up.

**Constraints.**
- Cut only: the new ground surface may not be above the existing ground anywhere (no fill).
- A road on the crest occupies x from {protected[0]:.2f} to {protected[1]:.2f} m; the ground there
  must not change.
- The new ground line must start and end at the same points as the existing one.
- After regrading, the phreatic surface is assumed to follow the lower of its measured trace and
  the new ground surface.

**Analysis standard.** {STANDARD} The required factor of safety applies to the critical circular
surface of the regraded slope.

**Deliverable.** Write `/app/answer.json`:

```json
{{"surface": [[x, z], ...], "fos": 1.31}}
```

`surface` is the regraded ground line (x strictly increasing) in the coordinates of `site.json`;
`fos` is the factor of safety you computed for it. Your design will be re-analyzed independently.
Excavation volume is measured as the cross-sectional area removed (m2 per metre of slope).
"""
    meta = {"target": target, "protected": protected, "reference_area": float(ref_area),
            "fos_existing": float(f0)}
    reference = {"surface": ref_surface, "fos": target}
    return Task("t2-design", f"t2-{seed:05d}", site.spec, instruction, meta, reference)


def verify_t2(spec: dict, meta: dict, answer: dict) -> dict:
    out = {"flags": [], "target": meta["target"], "reference_area": meta["reference_area"]}
    orig = spec["surface"]
    try:
        new = [[float(x), float(z)] for x, z in answer["surface"]]
        f_claim = float(answer["fos"])
        g1 = Polyline(new)
    except (KeyError, TypeError, ValueError):
        out.update(reward=0.0, flags=["malformed answer"])
        return out
    g0 = Polyline(orig)
    if abs(g1.xmin - g0.xmin) > 1e-3 or abs(g1.xmax - g0.xmax) > 1e-3 or \
            abs(g1.z[0] - g0.z[0]) > 1e-3 or abs(g1.z[-1] - g0.z[-1]) > 1e-3:
        out["flags"].append("surface does not start and end at the existing end points")
    xs = np.unique(np.concatenate([g0.x, g1.x, np.linspace(g0.xmin, g0.xmax, 4001)]))
    if np.any(g1(xs) > g0(xs) + 1e-3):
        out["flags"].append("fill placed above existing ground")
    a, b = meta["protected"]
    xp = xs[(xs >= a - 1e-9) & (xs <= b + 1e-9)]
    if np.any(np.abs(g1(xp) - g0(xp)) > 1e-3):
        out["flags"].append("protected crest zone was changed")
    if np.any(g1(xs) < float(spec.get("base_z", 0.0))):
        out["flags"].append("excavation below the model base")
    if out["flags"]:
        out.update(reward=0.0, feasible=False)
        return out

    try:
        f_v = search_circular(_design_profile(spec, new), "bishop").fos
    except InvalidSurface as e:
        out.update(reward=0.0, feasible=False, flags=[f"analysis failed: {e}"])
        return out
    area = _cut_area(orig, new)
    feasible = f_v >= meta["target"] - 0.005
    consistent = abs(f_claim - f_v) / f_v <= 0.03
    if not consistent:
        out["flags"].append("claimed FoS does not match the re-analysis")
    efficiency = min(1.0, meta["reference_area"] / area) if area > 0 else 0.0
    reward = (1.0 if feasible else 0.0) * efficiency * (1.0 if consistent else 0.5)
    out.update(fos_verified=f_v, fos_claimed=f_claim, area=area, feasible=feasible,
               efficiency=efficiency, consistent=consistent, reward=reward)
    return out


# ---------------------------------------------------------------------------------------------
# T3 — back-analysis of a failed slope, then a remediation prediction
# ---------------------------------------------------------------------------------------------


def _with_strength(spec: dict, c: float, phi: float, water_table=None) -> dict:
    spec2 = json.loads(json.dumps(spec))
    m = spec2["layers"][0]["material"]
    m["c"], m["phi_deg"] = float(c), float(phi)
    if water_table is not None:
        spec2["water_table"] = water_table
    return spec2


def _critical(spec: dict):
    return search_circular(profile_from_spec(spec), "bishop")


def make_t3(seed: int, max_tries: int = 60) -> Task:
    rng = np.random.default_rng(20_000 + seed)
    for _ in range(max_tries):
        site = random_site(rng, water_prob=1.0, max_layers=1)
        phi_true = float(np.round(rng.uniform(20.0, 34.0), 1))
        spec = site.spec

        def fos_c(c):
            return _critical(_with_strength(spec, c, phi_true)).fos

        try:
            lo, hi = 0.2, 60.0
            if fos_c(lo) > 1.0 or fos_c(hi) < 1.0:
                continue
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if fos_c(mid) < 1.0 else (lo, mid)
            c_true = float(np.round(0.5 * (lo + hi), 2))
            failure = _critical(_with_strength(spec, c_true, phi_true))
        except InvalidSurface:
            continue
        if not (2.0 <= c_true <= 40.0) or abs(failure.fos - 1.0) > 0.01:
            continue
        prof = profile_from_spec(spec)
        to_site = (lambda x: -x) if prof.mirrored else (lambda x: x)
        x_scarp, x_toe = to_site(failure.x_entry), to_site(failure.x_exit)
        # Inclinometer roughly mid-way along the slide.
        xi_int = failure.x_entry + rng.uniform(0.4, 0.6) * (failure.x_exit - failure.x_entry)
        depth_i = float(prof.surface(xi_int) - failure.surface.z(xi_int))
        x_incl = to_site(xi_int)
        # Drained scenario: phreatic surface lowered by a fixed drawdown, never above the ground.
        drawdown = float(np.round(rng.uniform(0.3, 0.6) * site.height, 1))
        wt = np.asarray(spec["water_table"], dtype=float)
        g = Polyline(spec["surface"])
        xs = np.unique(np.concatenate([wt[:, 0], g.x]))
        zs = np.minimum(np.interp(xs, wt[:, 0], wt[:, 1]) - drawdown, g(xs))
        zs = np.maximum(zs, spec["base_z"] + 0.1)
        wt_drained = np.column_stack([xs, zs]).tolist()
        try:
            f_drained = _critical(_with_strength(spec, c_true, phi_true, wt_drained)).fos
        except InvalidSurface:
            continue
        if not (1.05 <= f_drained <= 2.5):
            continue
        break
    else:
        raise RuntimeError("could not generate a T3 instance")

    c_lab = float(np.round(c_true * rng.uniform(1.6, 2.6) + rng.uniform(2.0, 6.0), 1))
    phi_lab = float(np.round(phi_true + rng.uniform(2.0, 5.0), 1))
    agent_spec = _with_strength(spec, c_lab, phi_lab)
    agent_spec["water_table_drained"] = wt_drained
    agent_spec["layers"][0]["material"]["note"] = "c and phi_deg are peak values from laboratory tests"

    side = "right" if site.faces_left else "left"
    unit = spec["layers"][0]
    instruction = f"""# Back-analysis of a slope failure

You are the geotechnical engineer called in after a failure. The {site.height:.1f} m high cut slope
(face inclined at {site.angle_deg:.1f} degrees, crest on the {side} side, toe at elevation
{site.z_toe:.1f} m, firm base at {spec['base_z']:.1f} m) is cut in a single soil unit,
{unit['name']}, with unit weight {unit['material']['gamma']} kN/m3. It failed after prolonged rainfall.

**Field observations at the time of failure**
- Main scarp (upper edge of the slide) at x = {x_scarp:.2f} m.
- Toe of the slide (where it daylights) at x = {x_toe:.2f} m.
- An inclinometer at x = {x_incl:.2f} m shows the shear zone at {depth_i:.2f} m below ground.
- Piezometers give the phreatic surface at the time of failure: `water_table` in `/app/site.json`.

**Laboratory data.** Triaxial tests on intact samples gave peak strength c' = {c_lab} kPa and
phi' = {phi_lab} degrees (these are the values stored in `site.json`).

**Tasks**
1. Back-calculate the strength parameters c' and phi' mobilized along the slip surface at failure.
2. Drains are proposed that would lower the phreatic surface to the `water_table_drained` trace in
   `site.json`. Using your back-calculated parameters, predict the factor of safety of the slope
   (critical circular surface) with the drains installed and the original slope geometry.

**Analysis standard.** {STANDARD}

**Deliverable.** Write `/app/answer.json`:

```json
{{"c": 5.0, "phi_deg": 25.0, "fos_drained": 1.35}}
```

Your parameters will be checked against the observed failure, and your prediction against the
behaviour of the real slope.
"""
    circ = failure.surface
    meta = {"c_true": c_true, "phi_true": phi_true, "fos_drained_true": float(f_drained),
            "failure_circle_internal": {"xc": float(circ.xc), "zc": float(circ.zc), "r": float(circ.r)},
            "x_scarp": float(x_scarp), "x_toe": float(x_toe), "height": site.height,
            "wt_drained": wt_drained}
    reference = {"c": c_true, "phi_deg": phi_true, "fos_drained": float(f_drained)}
    return Task("t3-backanalysis", f"t3-{seed:05d}", agent_spec, instruction, meta, reference)


def verify_t3(spec: dict, meta: dict, answer: dict) -> dict:
    out = {"flags": [], "c_true": meta["c_true"], "phi_true": meta["phi_true"],
           "fos_drained_true": meta["fos_drained_true"]}
    try:
        c, phi, f_pred = float(answer["c"]), float(answer["phi_deg"]), float(answer["fos_drained"])
    except (KeyError, TypeError, ValueError):
        out.update(reward=0.0, flags=["malformed answer"])
        return out
    if not (0.0 <= c <= 500.0 and 0.0 <= phi <= 60.0):
        out.update(reward=0.0, flags=["parameters out of physical range"])
        return out
    base = {k: v for k, v in spec.items() if k != "water_table_drained"}
    prof = profile_from_spec(_with_strength(base, c, phi))
    fc = meta["failure_circle_internal"]
    try:
        f_obs = evaluate(prof, Circle(fc["xc"], fc["zc"], fc["r"]), "bishop")
        crit = search_circular(prof, "bishop")
        f_drained_own = _critical(_with_strength(base, c, phi, meta["wt_drained"])).fos
    except InvalidSurface as e:
        out.update(reward=0.0, flags=[f"analysis failed: {e}"])
        return out
    to_site = (lambda x: -x) if prof.mirrored else (lambda x: x)
    h = meta["height"]
    loc_err = (abs(to_site(crit.x_entry) - meta["x_scarp"])
               + abs(to_site(crit.x_exit) - meta["x_toe"])) / (2 * h)
    s_fit = _lin_score(abs(f_obs - 1.0), 0.02, 0.10)
    s_loc = _lin_score(loc_err, 0.05, 0.30) * _lin_score(abs(crit.fos - 1.0), 0.03, 0.15)
    s_pred = _lin_score(abs(f_pred - meta["fos_drained_true"]) / meta["fos_drained_true"], 0.03, 0.15)
    if abs(f_pred - f_drained_own) / f_drained_own > 0.03:
        out["flags"].append("predicted FoS is not what the submitted parameters give")
    reward = 0.3 * s_fit + 0.3 * s_loc + 0.4 * s_pred
    out.update(fos_observed_surface=f_obs, fos_critical=crit.fos, location_error=loc_err,
               fos_drained_from_params=f_drained_own, score_fit=s_fit, score_location=s_loc,
               score_prediction=s_pred, reward=reward)
    return out


def _t0():
    from .fea.problems import make_t0, verify_t0
    return make_t0, verify_t0


def _t0_aster():
    from .aster.problems import make_t0_aster, verify_t0_aster
    return make_t0_aster, verify_t0_aster


def _t0_foam():
    from .foam.problems import make_t0_foam, verify_t0_foam
    return make_t0_foam, verify_t0_foam


FAMILIES = {"t1-check": (make_t1, verify_t1), "t2-design": (make_t2, verify_t2),
            "t3-backanalysis": (make_t3, verify_t3), "t0-fea": _t0(), "t0-aster": _t0_aster(),
            "t0-foam": _t0_foam()}


def dump(task: Task) -> str:
    return json.dumps({"family": task.family, "task_id": task.task_id, "spec": task.spec,
                       "meta": task.meta, "reference": task.reference}, indent=1)
