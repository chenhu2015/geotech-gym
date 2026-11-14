# MechGym

**Verifiable agentic environments for computational mechanics.**

LLM agents do the work of an engineer at a numerical-modeling consultancy: build finite-element and
finite-volume models in real open-source solvers (OpenSees, Code_Aster, OpenFOAM), run analyses, iterate
on a design, and back-analyze field observations. **Independent verifiers, validated against closed-form
solutions and published benchmark problems,** grade the work; the reward never comes from numbers the
agent computed itself.

The first domain is **geotechnics** (slopes, openings, foundations). The task interface, sandboxing and
verifier pattern are domain-independent, and further areas of solid and fluid mechanics are planned.

Datasheet (rewards, validation, negative controls, limits): [`docs/datasheet.md`](docs/datasheet.md) · design: [`docs/design.md`](docs/design.md) · verifier validation: [`docs/validation.md`](docs/validation.md)

## Task families

| Family | The agent must | The verifier |
|---|---|---|
| **T0 FEA modeling** | build an OpenSeesPy finite-element model (thick cylinder, circular opening, strip footing bearing capacity) | replays only the agent's model definition, solves it itself, and compares with Lamé / Kirsch / Prandtl closed-form solutions |
| **T0-A FEA modeling, Code_Aster** | write a Code_Aster command file and mesh for the thick cylinder and the circular opening | checks the command file against a whitelist of model-defining commands, re-runs it, and compares with Lamé / Kirsch |
| **T0-F stress analysis, OpenFOAM** | set up a `solidDisplacementFoam` case (blockMesh) for the circular opening | rejects dictionaries that can run code, re-runs the case, and compares with Kirsch |
| **T1 stability check** | find the critical circular slip surface and its factor of safety for a described site | re-analyzes the reported surface; scores criticality and whether the claimed FoS matches the surface |
| **T2 slope design** | regrade an unstable slope (cut only, protected crest road) to reach a target FoS with minimum excavation | checks constraints, re-runs the critical-surface search on the new ground, scores feasibility × excavation efficiency × honesty of the claimed FoS |
| **T3 back-analysis** | back-calculate the strength mobilized in an observed failure (scarp, toe, inclinometer, piezometers; lab peak strength is a trap) and predict the FoS after drainage | checks fit to the observed failure, location of the predicted critical surface, and the prediction against the hidden true slope |

Sites are generated procedurally (geometry, 1–3 soil units, phreatic surface, either orientation).
Grading never uses numbers the agent computed: every answer is re-analyzed with the verifier's own model.

## Limit-equilibrium verifier (T1–T3)

A limit-equilibrium engine written for this project (`src/mechgym/le.py`):

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

The solvers (OpenSeesPy, Code_Aster, OpenFOAM) run in containers; the host only needs Python with NumPy and
SciPy for the limit-equilibrium families. Docker or rootless Podman both work.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"   # .venv/Scripts/python on Windows
pytest -m "not slow"                          # host tests; solver tests are skipped without the solver

# full test suite, including the OpenSeesPy and Code_Aster verifiers, in containers
podman build -t mechgym-dev -f docker/mechgym.Dockerfile .
podman run --rm -v "$PWD":/work:z mechgym-dev pytest
podman build -t mechgym-aster-dev -f docker/aster-dev.Dockerfile docker
podman run --rm -v "$PWD":/work:z mechgym-aster-dev bash -c \
  '. /opt/activate.sh; export PYTHONPATH=/work/src:/opt/pylib:$PYTHONPATH; python3 -m pytest tests/test_aster.py'
podman build -t mechgym-foam-dev -f docker/foam-dev.Dockerfile docker
podman run --rm -v "$PWD":/work:z mechgym-foam-dev python3 -m pytest tests/test_foam.py

python scripts/build_tasks.py --t0 30 --t0a 4 --t0f 2 --t1 20 --t2 10 --t3 5 --out build/tasks   # export Harbor tasks
python scripts/run_task.py build/tasks/t0a-cyl-00000 --oracle   # solve + grade in two containers
harbor run -p build/tasks -a oracle -e podman                    # Harbor with rootless Podman (or -e docker)
harbor run -p build/tasks -a claude-code -m anthropic/<model> -e podman \n  --allow-agent-host api.anthropic.com                           # run an agent with Harbor
```

Each exported task is a Harbor task directory (`task.toml`, `instruction.md`, `environment/`, `tests/`,
`solution/`). The verifier runs in a separate sandbox that receives only the declared artifacts
(`/app/answer.json` and, for T0, the model files). `docker/podman-sandbox.sh` confines rootless
Podman to one directory on a shared machine, so that everything it creates can be removed in one step.
