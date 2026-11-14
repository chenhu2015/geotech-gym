"""Run one exported task end to end in containers (Docker or rootless Podman), Harbor-style.

    python scripts/run_task.py build/tasks/t1-00000 --oracle
    python scripts/run_task.py build/tasks/t1-00000 --answer-dir path/with/answer.json
    python scripts/run_task.py build/tasks/t0-hole-00001 --oracle --engine docker

Two separate containers are used, both from the task's ``environment/`` image and both without
network access:

1. **solve** (``--oracle`` only): ``/solution`` is copied in and ``solve.sh`` runs; the artifacts
   declared in ``task.toml`` are copied out. With ``--answer-dir`` the artifacts are taken from a
   directory instead (e.g. the output of an agent run).
2. **verify**: a fresh container receives only the artifacts (under ``/app``) and ``tests/``
   (under ``/tests``), runs ``tests/test.sh``, and ``/logs/verifier`` is copied back.

Files are moved with ``<engine> cp`` rather than bind mounts, so the same code works with Docker
and with rootless Podman under SELinux. Results go to ``runs/<task_id>/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def ensure_image(engine: str, env_dir: Path) -> str:
    digest = hashlib.sha1((env_dir / "Dockerfile").read_bytes()).hexdigest()[:12]
    tag = f"localhost/mechgym-env:{digest}"
    exists = subprocess.run([engine, "image", "inspect", tag], capture_output=True).returncode == 0
    if not exists:
        # Build from a context holding only the Dockerfile and site.json.
        sh([engine, "build", "-q", "-t", tag, str(env_dir)], stdout=subprocess.DEVNULL)
    return tag


def container(engine: str, image: str, cfg: dict, cmd: list[str]) -> str:
    env = cfg.get("environment", {})
    name = f"mechgym-{uuid.uuid4().hex[:10]}"
    sh([engine, "create", "--name", name, "--network", "none",
        "--cpus", str(env.get("cpus", 2)), "--memory", f"{env.get('memory_mb', 4096)}m",
        image, *cmd], stdout=subprocess.DEVNULL)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--oracle", action="store_true", help="produce the artifacts with solution/solve.sh")
    src.add_argument("--answer-dir", type=Path, help="directory holding the artifacts (answer.json, ...)")
    ap.add_argument("--engine", default=shutil.which("podman") and "podman" or "docker")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    task = args.task_dir.resolve()
    cfg = tomllib.loads((task / "task.toml").read_text())
    artifacts = [Path(a).name for a in cfg.get("artifacts", ["/app/answer.json"])]
    out = args.out / task.name / time.strftime("%Y%m%d-%H%M%S")
    (out / "artifacts").mkdir(parents=True)
    image = ensure_image(args.engine, task / "environment")
    t0 = time.time()

    if args.oracle:
        c = container(args.engine, image, cfg, ["sh", "/solution/solve.sh"])
        try:
            sh([args.engine, "cp", f"{task / 'solution'}/.", f"{c}:/solution"])
            log = subprocess.run([args.engine, "start", "-a", c], capture_output=True, text=True,
                                 timeout=cfg.get("agent", {}).get("timeout_sec", 3600))
            (out / "solve.log").write_text(log.stdout + log.stderr)
            for a in artifacts:
                subprocess.run([args.engine, "cp", f"{c}:/app/{a}", str(out / "artifacts" / a)],
                               capture_output=True)
        finally:
            subprocess.run([args.engine, "rm", "-f", c], capture_output=True)
    else:
        for a in artifacts:
            if (args.answer_dir / a).exists():
                shutil.copy(args.answer_dir / a, out / "artifacts" / a)

    v = container(args.engine, image, cfg, ["sh", "/tests/test.sh"])
    try:
        sh([args.engine, "cp", f"{task / 'tests'}/.", f"{v}:/tests"])
        for a in artifacts:
            if (out / "artifacts" / a).exists():
                sh([args.engine, "cp", str(out / "artifacts" / a), f"{v}:/app/{a}"])
        log = subprocess.run([args.engine, "start", "-a", v], capture_output=True, text=True,
                             timeout=cfg.get("verifier", {}).get("timeout_sec", 900))
        (out / "verify.log").write_text(log.stdout + log.stderr)
        subprocess.run([args.engine, "cp", f"{v}:/logs/verifier/.", str(out)], capture_output=True)
    finally:
        subprocess.run([args.engine, "rm", "-f", v], capture_output=True)

    reward_file = out / "reward.json"
    reward = json.loads(reward_file.read_text())["reward"] if reward_file.exists() else None
    summary = {"task": task.name, "reward": reward, "seconds": round(time.time() - t0, 1), "out": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if reward is None:
        sys.exit(f"verifier produced no reward; see {out / 'verify.log'}")


if __name__ == "__main__":
    main()
