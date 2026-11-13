"""JSON-serializable site description <-> SlopeProfile.

A site spec is what the agent receives (``site.json``) and what the verifier rebuilds its own
model from. Coordinates are given as the site is drawn (the slope may face either way).

{
  "units": {"length": "m", "stress": "kPa", "unit_weight": "kN/m3"},
  "surface": [[x, z], ...],
  "layers": [
    {"name": "...", "material": {"model": "mohr-coulomb", "c": 10, "phi_deg": 30, "gamma": 19},
     "bottom": [[x, z], ...]},           # bottom omitted (null) for the lowest unit
    ...
  ],
  "base_z": 0.0,
  "water_table": [[x, z], ...] | null
}
"""

from __future__ import annotations

from typing import Any

from .materials import HoekBrown, MohrCoulomb
from .profile import Layer, Polyline, SlopeProfile


def material_from_spec(m: dict[str, Any]):
    model = m["model"]
    if model == "mohr-coulomb":
        return MohrCoulomb(c=float(m["c"]), phi_deg=float(m["phi_deg"]), gamma=float(m["gamma"]))
    if model == "hoek-brown":
        return HoekBrown(sigci=float(m["sigci"]), mb=float(m["mb"]), s=float(m["s"]),
                         a=float(m["a"]), gamma=float(m["gamma"]))
    raise ValueError(f"unknown material model {model!r}")


def material_to_spec(mat) -> dict[str, Any]:
    if isinstance(mat, MohrCoulomb):
        return {"model": "mohr-coulomb", "c": mat.c, "phi_deg": mat.phi_deg, "gamma": mat.gamma}
    if isinstance(mat, HoekBrown):
        return {"model": "hoek-brown", "sigci": mat.sigci, "mb": mat.mb, "s": mat.s, "a": mat.a,
                "gamma": mat.gamma}
    raise TypeError(type(mat))


def profile_from_spec(spec: dict[str, Any], surface: list | None = None) -> SlopeProfile:
    """Build a profile; ``surface`` overrides the ground line (used for submitted designs)."""
    layers = [
        Layer(material_from_spec(l["material"]),
              Polyline(l["bottom"]) if l.get("bottom") is not None else None)
        for l in spec["layers"]
    ]
    wt = spec.get("water_table")
    return SlopeProfile(
        Polyline(surface if surface is not None else spec["surface"]),
        layers,
        base_z=float(spec.get("base_z", 0.0)),
        water_table=Polyline(wt) if wt is not None else None,
        gamma_w=float(spec.get("gamma_w", 9.81)),
    )


def make_spec(surface, layers, base_z=0.0, water_table=None, names=None) -> dict[str, Any]:
    out_layers = []
    for i, (mat, bottom) in enumerate(layers):
        out_layers.append({
            "name": names[i] if names else f"unit {i + 1}",
            "material": material_to_spec(mat),
            "bottom": [list(map(float, p)) for p in bottom] if bottom is not None else None,
        })
    return {
        "units": {"length": "m", "stress": "kPa", "unit_weight": "kN/m3"},
        "surface": [list(map(float, p)) for p in surface],
        "layers": out_layers,
        "base_z": float(base_z),
        "water_table": [list(map(float, p)) for p in water_table] if water_table is not None else None,
    }
