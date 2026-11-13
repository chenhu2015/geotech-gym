# Verifier validation

Run with `pytest` (the benchmark searches are marked `slow`). Results on 2026-11-12.

## Closed-form checks

| Test | Expected | Result |
|---|---|---|
| Infinite slope, c = 0, φ = 35°, β = 30°, planar surface (Spencer, M–P) | tan φ / tan β = 1.213 | within 2% |
| Hoek–Brown envelope vs Balmer's relations at σ3 = 100 kPa | exact | < 0.1% |
| Spencer vs Bishop on the same circular surface | ≈ equal | < 1% |
| Profile mirrored left/right | identical FoS | < 0.1% |

## Published benchmark problems

See `validation-cases.md` for geometry, parameters and sources.

| Case | Method | Reference | This verifier | Error |
|---|---|---|---|---|
| V1 complex slope (Cheng et al. 2007) | Morgenstern–Price, non-circular | 1.383 / 1.3848 | 1.378 | −0.4% |
| V1 | Bishop, circular (dense grid check) | — | 1.401 | circular surfaces are ~1% above the non-circular minimum, as expected |
| V2 Hoek–Brown (Hammah et al. 2005, Ex. 1) | Bishop, circular | 1.153 | 1.143 | −0.8% |
| V2 | Spencer, circular | 1.152 | 1.145 | −0.6% |
| V2 layered (Ex. 2) | Spencer, non-circular | 0.963 | 0.974 | +1.2% |

## Notes from validation

- **Slice boundaries must include every geometric breakpoint.** With uniform slices only, the
  non-circular optimizer exploited midpoint-integration error at polyline kinks and returned
  1.359 for V1 (−1.7%); aligning slice edges to all vertices fixed it (1.378, stable from 60 to
  240 slices). This is the slope-stability version of reward hacking: an optimizer finds the
  verifier's numerical error before it finds the physics.
- **Hammah et al. Example 2 uses non-circular surfaces.** Circular Bishop gives 0.989 for this case.
- **A weak layer exposed in the face** (V2 layered, c = 0, φ = 25° on a 45° face) admits a
  shallow skin failure with FoS ≈ tan 25° / tan 45° = 0.47; published analyses exclude it with a
  minimum slip depth. Task generators avoid cohesionless units daylighting in steep faces.
- **Back-analysis is non-unique, predictions are not (T3).** For one instance, (c, φ) pairs within
  ±3° of the true φ that reproduce FoS = 1 on the observed surface all predict the drained FoS
  within 1%; at +6° the prediction error grows to 8% and the score drops.
