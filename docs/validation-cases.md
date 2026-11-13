# Verifier validation cases

Reference problems used to validate the verifier's solvers before any task depends on them. Sources are
Itasca's public FLAC3D / FLAC2D documentation (verification problems and factor-of-safety examples) and the
papers they cite. Collected 2026-11-12.

## Limit equilibrium (M1)

### V1 — Slope with a complex surface profile, Mohr–Coulomb, dry

- Source: Itasca FLAC3D docs, "Failure of a Slope with a Complex Surface Profile in a Mohr-Coulomb Material";
  reference Cheng, Lansivaara & Wei (2007), *Computers and Geotechnics* 34, 137–150.
- Profile vertices (x, z) in m: (0,0) (46,0) base; (0,10)–(10,10) toe level; 45° face (10,10)→(16,16);
  26.7° face (16,16)→(26,21); crest (26,21)→(46,21).
- γ = 20 kN/m³, c = 10 kPa, φ = 30°.
- Reference FoS: Morgenstern–Price **1.383** and **1.3848** (two near-critical surfaces); FLAC3D strength
  reduction **1.38** (converged at 144 zones horizontally).

### V2 — Simple slope, Hoek–Brown (homogeneous and with a weak layer)

- Source: Itasca FLAC3D docs, "Simple Slope in Hoek-Brown Material"; reference Hammah et al. (2005),
  ARMA/USRMS 05-810.
- Geometry: 45° slope, 10 m high; model 40 m long, base 8 m thick; face from (15, 8) to (25, 18).
- Rock: σci = 30 MPa, mb = 0.067, s = 0.000025, a = 0.619, ρ = 2500 kg/m³ (g = 10 m/s²).
- Reference FoS (homogeneous): Bishop **1.153**, Spencer **1.152**, HB strength reduction 1.15, FLAC3D 1.16.
- Layered case: 1 m Mohr–Coulomb layer at the toe (z = 8–9 m), c = 0, φ = 25°.
  Reference FoS: Bishop **0.934**, Spencer **0.963**, HB strength reduction 0.95, FLAC3D 0.96.
- Needs a nonlinear (Hoek–Brown) strength along the slip surface in the LE solver.

### Still to collect

- ACADS (1989) slope-stability referee problems — widely used to check LE software.
- A case with a water table / pore pressure.

## FEM / strength reduction (T4, later)

- Rough strip footing on a cohesive frictionless material (Prandtl solution) — FLAC3D verification problem.
- Smooth circular / square footing on Mohr–Coulomb material — FLAC3D verification problems.
- Cylindrical hole in an infinite Mohr–Coulomb material (Salençon closed form).
- Jointed rock slopes, rigid-block analytical FoS 1.32 (Lorig & Varona 2004) — FLAC2D example.

## Links

- FLAC examples and verification index:
  https://docs.itascacg.com/itasca930/flac3d/docproject/source/projects/projects.html
- Factor-of-safety methodology:
  https://docs.itascacg.com/pfc700/common/docproject/source/manual/program_guide/methodology/factorofsafety/factorofsafety.html
- V1: https://docs.itascacg.com/pfc700/flac3d/zone/test3d/TheoryAndBackground/FactorOfSafety/ComplexSlope/ComplexSlope.html
- V2: https://docs.itascacg.com/pfc700/flac3d/zone/test3d/TheoryAndBackground/FactorOfSafety/HoekBrownSlope/HoekBrownSlope.html
