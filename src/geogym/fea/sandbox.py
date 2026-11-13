"""Record an agent's OpenSeesPy script, then replay only its model definition.

record(script)  — runs the agent's script in a subprocess with ``openseespy.opensees`` replaced by a
                  recording proxy (calls are forwarded, so the script runs normally) and saves every
                  call as JSON.
replay(calls)   — in a fresh subprocess, re-issues only model-definition commands (nodes, elements,
                  materials, constraints, loads), runs the verifier's own linear static analysis,
                  and returns coordinates and displacements of the queried nodes.

Commands that do not define the model (analysis setup, ``analyze``, ``setNodeDisp``, ``reactions``,
...) are never replayed, so the graded response always comes from the agent's model solved by the
verifier.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

DEFINE = {
    "wipe", "model", "node", "fix", "fixX", "fixY", "fixZ", "equalDOF", "rigidLink",
    "nDMaterial", "uniaxialMaterial", "element", "timeSeries", "pattern", "load", "eleLoad", "sp",
    "geomTransf", "section", "fiber", "patch", "layer", "block2D", "mass",
}

_RECORD_BOOT = textwrap.dedent(r'''
    import json, runpy, sys, types
    import numpy as _np
    import openseespy.opensees as _real

    LOG = []

    def _clean(v):
        if isinstance(v, (bool, int, float, str)) or v is None:
            return v
        if isinstance(v, _np.generic):
            return v.item()
        if isinstance(v, (list, tuple, _np.ndarray)):
            return [_clean(x) for x in list(v)]
        return str(v)

    class _Proxy(types.ModuleType):
        def __getattr__(self, name):
            fn = getattr(_real, name)
            if not callable(fn):
                return fn
            def wrapped(*args, **kwargs):
                LOG.append([name, [_clean(a) for a in args], {k: _clean(v) for k, v in kwargs.items()}])
                return fn(*args, **kwargs)
            return wrapped

    proxy = _Proxy("openseespy.opensees")
    proxy.__all__ = [n for n in dir(_real) if not n.startswith("_")]
    pkg = types.ModuleType("openseespy")
    pkg.__path__ = []
    pkg.opensees = proxy
    sys.modules["openseespy"] = pkg
    sys.modules["openseespy.opensees"] = proxy

    script, out = sys.argv[1], sys.argv[2]
    status = "ok"
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    except BaseException as e:
        status = f"script error: {type(e).__name__}: {e}"
    with open(out, "w") as f:
        json.dump({"status": status, "calls": LOG}, f)
''')

_REPLAY_BOOT = textwrap.dedent(r'''
    import json, sys
    import openseespy.opensees as ops
    DEFINE = set(json.loads(sys.argv[3]))
    rec = json.load(open(sys.argv[1]))
    query = json.loads(sys.argv[4])
    out = {"status": "ok", "skipped": {}}
    try:
        ops.wipe()
        for name, args, kwargs in rec["calls"]:
            if name in DEFINE:
                getattr(ops, name)(*args, **kwargs)
            else:
                out["skipped"][name] = out["skipped"].get(name, 0) + 1
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1e-10, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            out["status"] = "analysis failed"
        tags = ops.getNodeTags()
        out["n_nodes"] = len(tags)
        out["n_elements"] = len(ops.getEleTags())
        nodes = {}
        for t in query:
            if t in tags:
                nodes[str(t)] = {"coord": ops.nodeCoord(t), "disp": ops.nodeDisp(t)}
        out["nodes"] = nodes
    except BaseException as e:
        out["status"] = f"replay error: {type(e).__name__}: {e}"
    json.dump(out, open(sys.argv[2], "w"))
''')


def record(script: Path, workdir: Path, timeout: float = 600.0) -> dict:
    workdir = Path(workdir)
    boot = workdir / "_geogym_record.py"
    out = workdir / "_geogym_calls.json"
    boot.write_text(_RECORD_BOOT)
    try:
        subprocess.run([sys.executable, str(boot), str(script), str(out)], cwd=workdir,
                       timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        return {"status": "script timed out", "calls": []}
    finally:
        boot.unlink(missing_ok=True)
    if not out.exists():
        return {"status": "script crashed before finishing", "calls": []}
    data = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return data


def replay(recorded: dict, query_nodes: list[int], workdir: Path, timeout: float = 600.0) -> dict:
    workdir = Path(workdir)
    rec_path = workdir / "_geogym_rec.json"
    out = workdir / "_geogym_replay.json"
    boot = workdir / "_geogym_replay.py"
    rec_path.write_text(json.dumps(recorded))
    boot.write_text(_REPLAY_BOOT)
    try:
        subprocess.run([sys.executable, str(boot), str(rec_path), str(out), json.dumps(sorted(DEFINE)),
                        json.dumps([int(q) for q in query_nodes])],
                       cwd=workdir, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        return {"status": "replay timed out"}
    finally:
        boot.unlink(missing_ok=True)
        rec_path.unlink(missing_ok=True)
    if not out.exists():
        return {"status": "replay crashed"}
    data = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return data
