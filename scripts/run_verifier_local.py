"""Run an exported task's verifier locally (no Docker), against an answer file.

    python scripts/run_verifier_local.py build/tasks/t1-00000 path/to/answer.json
    python scripts/run_verifier_local.py build/tasks/t1-00000 --oracle
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("answer", type=Path, nargs="?")
    ap.add_argument("--oracle", action="store_true", help="use solution/solve.sh's answer")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        app, logs = Path(tmp) / "app", Path(tmp) / "logs"
        app.mkdir()
        if args.oracle:
            solve = (args.task_dir / "solution" / "solve.sh").read_text()
            body = re.search(r"<<'EOF'\n(.*)\nEOF", solve, re.S).group(1)
            (app / "answer.json").write_text(body)
        elif args.answer:
            (app / "answer.json").write_text(args.answer.read_text())
        env = dict(os.environ, GEOGYM_TESTS_DIR=str(args.task_dir / "tests"),
                   GEOGYM_APP_DIR=str(app), GEOGYM_LOGS_DIR=str(logs))
        subprocess.run([sys.executable, str(args.task_dir / "tests" / "verify.py")], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        print(json.dumps(json.loads((logs / "details.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
