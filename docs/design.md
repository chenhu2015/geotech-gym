# geotech-gym — design

## Goal

An RL environment and benchmark for **agentic geotechnical engineering**: an LLM agent works the way a
geotechnical engineer at a numerical-modeling consultancy does (build a ground model, choose parameters,
run analyses, iterate on a design, back-analyze monitoring data), and an **independent, validated verifier**
scores the result.

The point is not "teach a model to run a solver". The point is:

1. **Realistic, multi-step engineering workflows** — the agent sets up models, runs them, reads results,
   changes the design, and runs again.
2. **Verifiers that can be trusted** — the reward never comes from numbers the agent computed itself.
   The verifier re-analyzes the submitted design with its own fixed model, and the verifier itself is
   validated against analytical solutions and published benchmark problems.
3. **Learning from real-world-style feedback** — back-analysis tasks give the agent monitoring data
   generated from hidden ground truth, the way real projects update a model from field measurements.

## Engineering workflow being modeled

1. Site investigation → stratigraphy, groundwater, lab strength parameters (with uncertainty)
2. Simplified methods first → limit-equilibrium factor of safety, hand checks
3. Numerical modeling where needed → initial stress, staged construction, strength reduction
4. Verification → compare against limit equilibrium / analytical solutions, mesh sensitivity
5. Sensitivity / probabilistic analysis of uncertain parameters
6. Design of geometry and support (regrading, benches, drainage, anchors, retaining)
7. Monitoring and **back-analysis** (observational method) → update parameters and the design

## Task families (planned)

| # | Family | Agent must | Verifier |
|---|---|---|---|
| T1 | Slope stability check | compute the factor of safety and critical slip surface for a given slope | independent limit-equilibrium search (Bishop / Spencer), validated against ACADS-style benchmark slopes |
| T2 | Slope design | modify geometry / add drainage or reinforcement so FoS ≥ target at minimum cost | re-analyze the submitted design with the trusted solver; check geometric validity and constraints; continuous score on cost |
| T3 | Back-analysis | given synthetic monitoring data (from hidden parameters), infer parameters and assess current safety | compare inferred parameters / predicted FoS / predictions on held-out monitoring data against the hidden ground truth |
| T4 | Staged excavation (later) | design an excavation and support sequence | FEM with strength reduction (open-source solver), cross-checked against limit equilibrium |

Tasks are generated procedurally (geometry, layers, water table, parameters) with difficulty knobs, so the
same family yields many instances and difficulty can be measured.

## Agent interface

Harbor-style task directories: `instruction.md`, `environment/` (container with Python + the open-source
analysis library), `tests/` (the verifier, which writes the reward). The agent submits a JSON answer or
design file. The same tasks run as an eval (fixed set) or as RL environments (generated on the fly).

## Verifier principles

- **Never trust agent-reported results.** Re-run every submitted design with the verifier's own model,
  mesh, and search settings.
- **Validate the verifier first** against closed-form solutions and published benchmark problems; report
  its error.
- **Continuous, decomposed rewards** (feasibility gate × safety margin × cost) rather than pass/fail only.
- **Anti-hacking checks**: geometric validity, parameter bounds, consistency between what the agent claims
  and what the verifier recomputes; log every mismatch for analysis.

## Solvers

- Limit equilibrium (Bishop simplified, Spencer) with a slip-surface search — written here in NumPy, so the
  verifier is small, transparent, and testable.
- Established open-source engineering codes, each in its own container image, for the modeling tasks:
  OpenSeesPy (finite elements, Python API), Code_Aster (finite elements, command files) and OpenFOAM
  (finite volumes, case dictionaries). The same verifier pattern applies to each: accept only the parts
  of the agent's input that define the model, re-run it with the verifier's own settings, and compare
  with an analytical or validated reference. Solvers are never installed on the host.
- FEM for T4: Mohr–Coulomb / Drucker–Prager plasticity with strength reduction, in one of the codes above.

## Milestones

- **M1** Limit-equilibrium solver + validation suite against benchmark slopes — done
- **M2** Procedural generator + T1 / T2 tasks packaged as Harbor tasks — done
- **M3** Baseline evaluation of frontier and open models; failure-mode and hacking analysis
- **M4** T3 back-analysis tasks — done
- **M5** Small-scale GRPO run (optional) and a write-up page
