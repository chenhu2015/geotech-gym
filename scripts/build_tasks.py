"""Generate task instances and export them as Harbor task directories.

    python scripts/build_tasks.py --t1 20 --t2 10 --t3 5 --out build/tasks
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from geogym.harbor import export
from geogym.tasks import make_t1, make_t2, make_t3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", type=int, default=10)
    ap.add_argument("--t2", type=int, default=5)
    ap.add_argument("--t3", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("build/tasks"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for maker, n in ((make_t1, args.t1), (make_t2, args.t2), (make_t3, args.t3)):
        for i in range(n):
            t0 = time.time()
            task = maker(args.seed0 + i)
            export(task, args.out)
            index.append({"task_id": task.task_id, "family": task.family,
                          **{k: v for k, v in task.meta.items() if isinstance(v, (int, float))}})
            print(f"{task.task_id}  {time.time() - t0:5.1f}s  {task.meta}")
    (args.out / "index.json").write_text(json.dumps(index, indent=1, default=float))


if __name__ == "__main__":
    main()
