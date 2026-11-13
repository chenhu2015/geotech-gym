# geotech-gym

Simulation-based RL environments for **agentic geotechnical engineering**. LLM agents analyze,
design, and back-analyze slopes the way a geotechnical engineer at a numerical-modeling consultancy
does, and **independent verifiers validated against published benchmarks** grade the work.

Datasheet (rewards, validation, negative controls, limits): [`docs/datasheet.md`](docs/datasheet.md) · design: [`docs/design.md`](docs/design.md) · verifier validation: [`docs/validation.md`](docs/validation.md)

## Task families

| Family | The agent must | The verifier |
|---|---|---|
| **T0 FEA modeling** | build an OpenSeesPy finite-element model (thick cylinder, circular opening, strip footing bearing capacity) | replays only the agent's model definition, solves it itself, and compares with Lamé / Kirsch / Prandtl closed-form solutions |
| **T1 stability check** | find the critical circular slip surface and its factor of safety for a described site | re-analyzes the reported surface; scores criticality and whether the claimed FoS matches the surface |
| **T2 slope design** | regrade an unstable slope (cut only, protected crest road) to reach a target FoS with minimum excavation | checks constraints, re-runs the critical-surface search on the new ground, scores feasibility × excavation efficiency × honesty of the claimed FoS |
| **T3 back-analysis** | back-calculate the strength mobilized in an observed failure (scarp, toe, inclinometer, piezometers; lab peak strength is a trap) and predict the FoS after drainage | checks fit to the observed failure, location of the predicted critical surface, and the prediction against the hidden true slope |

Sites are generated procedurally (geometry, 1–3 soil units, phreatic surface, either orientation).
Grading never uses numbers the agent computed: every answer is re-analyzed with the verifier's own model.

## Verifier

A limit-equilibrium engine written for this project (`src/geogym/le.py`):

- Bishop's simplified method; general limit equilibrium (Spencer, Morgenstern–Price)
- circular and non-circular slip surfaces, with critical-surface search
- Mohr–Coulomb and generalized Hoek–Brown strength; layered ground; phreatic surface

Validated against published benchmark problems that Itasca uses to verify FLAC3D:

| Case | Reference | This verifier |
|---|---|---|
| Complex slope, Mohr–Coulomb (Cheng et al. 2007) | Morgenstern–Price 1.383 (non-circular) | 1.378 (−0.4%) |
| Hoek–Brown slope (Hammah et al. 2005, Ex. 1) | Bishop 1.153 · Spencer 1.152 | 1.143 · 1.145 |
| Hoek–Brown slope with weak layer (Ex. 2) | Spencer 0.963 (non-circular) | 0.974 (+1.2%) |
| Infinite slope, cohesionless | tan φ / tan β | < 2% |

## Running

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows (.venv/bin/python on Linux/macOS)
pytest -m "not slow"                              # fast tests; `pytest` runs the benchmark searches too

python scripts/build_tasks.py --t0 30 --t1 20 --t2 10 --t3 5 --out build/tasks   # export Harbor tasks
python scripts/run_verifier_local.py build/tasks/t1-00000 --oracle      # grade locally, no Docker
harbor run -p build/tasks -a claude-code -m <model>                      # run an agent with Harbor
```

Each exported task is a Harbor task directory (`task.toml`, `instruction.md`, `environment/`, `tests/`,
`solution/`). The verifier runs in a separate sandbox that receives only `/app/answer.json`.
