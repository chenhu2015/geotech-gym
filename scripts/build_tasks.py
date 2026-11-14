"""Generate task instances and export them as Harbor task directories.

    python scripts/build_tasks.py --t0 20 --t0a 4 --t0f 2 --t1 20 --t2 10 --t3 5 --out build/tasks
    python scripts/build_tasks.py --suite eval-v0.1 --out build/eval-v0.1     # the fixed eval set
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mechgym.harbor import export
from mechgym.aster.problems import make_t0_aster
from mechgym.fea.problems import make_t0
from mechgym.foam.problems import make_t0_foam
from mechgym.tasks import make_t1, make_t2, make_t3


# Fixed, versioned evaluation suites: the same arguments always produce the same tasks. T0 cycles
# through its problem types with the seed, so 9 T0 seeds give 3 instances of each of 3 types (6 T0-A
# seeds: 3 of each of 2 types), and every one of the 9 task types gets 3 instances.
SUITES = {
    "eval-v0.1": dict(t0=9, t0a=6, t0f=3, t1=3, t2=3, t3=3, seed0=0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), help="build a fixed evaluation suite "
                    "(overrides the counts and --seed0)")
    ap.add_argument("--t0", type=int, default=10, help="FEA tasks, OpenSeesPy")
    ap.add_argument("--t0a", type=int, default=0, help="FEA tasks, Code_Aster")
    ap.add_argument("--t0f", type=int, default=0, help="FEA tasks, OpenFOAM")
    ap.add_argument("--t1", type=int, default=10)
    ap.add_argument("--t2", type=int, default=5)
    ap.add_argument("--t3", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("build/tasks"))
    args = ap.parse_args()
    if args.suite:
        for k, v in SUITES[args.suite].items():
            setattr(args, k, v)
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    makers = ((make_t0, args.t0), (make_t0_aster, args.t0a), (make_t0_foam, args.t0f),
              (make_t1, args.t1), (make_t2, args.t2), (make_t3, args.t3))
    for maker, n in makers:
        for i in range(n):
            t0 = time.time()
            task = maker(args.seed0 + i)
            export(task, args.out)
            index.append({"task_id": task.task_id, "family": task.family,
                          **{k: v for k, v in task.meta.items() if isinstance(v, (int, float))}})
            print(f"{task.task_id}  {time.time() - t0:5.1f}s  {task.meta}")
    if args.suite:
        index = {"suite": args.suite, "tasks": index}
    (args.out / "index.json").write_text(json.dumps(index, indent=1, default=float))


if __name__ == "__main__":
    main()
