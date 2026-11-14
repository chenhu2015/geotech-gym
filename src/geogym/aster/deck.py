"""Check and run an agent's Code_Aster command file (``.comm``), then read its displacement field.

A ``.comm`` file is Python, so it could compute an answer outside the finite-element model or build a
result field by hand. Before running it, :func:`check_deck` parses it and only accepts a restricted
subset: assignments, loops and conditionals; arithmetic; calls to a whitelist of Code_Aster commands
that *define and solve* a model (no ``CREA_CHAMP`` / ``CREA_RESU`` / ``LIRE_RESU``, which could inject
a field); a few harmless builtins; no imports, no attribute access, no names starting with ``_``
except ``_F``; ``FORMULE`` expressions restricted the same way. The result to be graded must be a
variable named ``resu`` assigned from ``MECA_STATIQUE`` or ``STAT_NON_LINE``.

:func:`run_deck` then runs the accepted deck with ``run_aster`` in a scratch directory, with the
deck's own ``FIN()`` removed and a grader epilogue appended that writes the mesh coordinates and the
last stored displacement field of ``resu`` to JSON.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

COMMANDS = {
    "DEBUT", "FIN", "DEFI_FICHIER", "LIRE_MAILLAGE", "MODI_MAILLAGE", "CREA_MAILLAGE", "DEFI_GROUP",
    "AFFE_MODELE", "DEFI_MATERIAU", "AFFE_MATERIAU", "AFFE_CHAR_MECA", "AFFE_CHAR_MECA_F",
    "AFFE_CHAR_CINE", "DEFI_FONCTION", "DEFI_CONSTANTE", "DEFI_LIST_REEL", "DEFI_LIST_INST",
    "FORMULE", "AFFE_CARA_ELEM", "MECA_STATIQUE", "STAT_NON_LINE", "CALC_CHAMP", "IMPR_RESU",
    "IMPR_TABLE", "POST_RELEVE_T",
}
SOLVERS = {"MECA_STATIQUE", "STAT_NON_LINE"}
BUILTINS = {"range", "len", "float", "int", "abs", "min", "max", "round", "list", "tuple", "dict",
            "zip", "enumerate", "sum", "str", "_F"}
FORMULA_NAMES = {"X", "Y", "Z", "INST", "sqrt", "sin", "cos", "tan", "atan2", "exp", "log", "pi"}
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass, ast.Call, ast.keyword,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.BinOp, ast.UnaryOp,
    ast.Compare, ast.BoolOp, ast.IfExp, ast.Subscript, ast.Slice, ast.ListComp, ast.comprehension,
    ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context, ast.JoinedStr, ast.FormattedValue,
)
MAX_SOURCE = 200_000


def check_deck(source: str, app_dir: str = "/app") -> list[str]:
    """Return a list of problems; an empty list means the deck is accepted."""
    if len(source) > MAX_SOURCE:
        return ["command file too large"]
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems: list[str] = []
    resu_sources: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            problems.append(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
            continue
        if isinstance(node, ast.Name) and node.id.startswith("_") and node.id != "_F":
            problems.append(f"line {node.lineno}: name {node.id!r} is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                problems.append(f"line {node.lineno}: only direct calls to commands are allowed")
                continue
            name = node.func.id
            if name not in COMMANDS and name not in BUILTINS:
                problems.append(f"line {node.lineno}: {name}() is not an allowed command")
            for kw in node.keywords:
                if kw.arg is None:
                    problems.append(f"line {node.lineno}: ** arguments are not allowed")
                elif kw.arg == "FICHIER":
                    if not (isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
                            and kw.value.value.startswith(app_dir.rstrip("/") + "/")
                            and ".." not in kw.value.value):
                        problems.append(f"line {node.lineno}: FICHIER must be a literal path under {app_dir}")
                elif kw.arg == "reuse" and isinstance(kw.value, ast.Name) and kw.value.id == "resu" \
                        and name not in SOLVERS | {"CALC_CHAMP"}:
                    problems.append(f"line {node.lineno}: {name} may not modify resu")
            if name == "FORMULE":
                problems += _check_formula(node)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id == "resu":
                        v = node.value
                        resu_sources.append(v.func.id if isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                                            else type(v).__name__)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "resu":
            resu_sources.append("augmented assignment")
    if not resu_sources:
        problems.append("no variable named resu (the result to be graded)")
    elif any(s not in SOLVERS for s in resu_sources):
        problems.append(f"resu must be assigned only from {sorted(SOLVERS)}, got {resu_sources}")
    return problems


def _check_formula(call: ast.Call) -> list[str]:
    params = {kw.arg for kw in call.keywords if kw.arg not in ("VALE", "NOM_PARA")}
    for kw in call.keywords:
        if kw.arg != "VALE":
            continue
        if not (isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)):
            return [f"line {call.lineno}: FORMULE VALE must be a string literal"]
        try:
            expr = ast.parse(kw.value.value, mode="eval")
        except SyntaxError:
            return [f"line {call.lineno}: FORMULE VALE is not a valid expression"]
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute) or (isinstance(n, ast.Name) and n.id not in FORMULA_NAMES | params):
                return [f"line {call.lineno}: FORMULE uses a name or attribute that is not allowed"]
    return []


EPILOGUE = '''
# ---- grader epilogue (appended by the verifier) ----
import json as _json
_last = resu.getIndexes()[-1]
_f = resu.getField("DEPL", _last).toSimpleFieldOnNodes()
_vals, _mask = _f.getValues()
_m = resu.getMesh()
_n = _m.getNumberOfNodes()
_c = list(map(float, _m.getCoordinates().getValues()))
_dim = len(_c) // _n
_json.dump({"components": list(_f.getComponents()), "values": [[float(x) for x in r] for r in _vals],
            "coords": [_c[i * _dim:(i + 1) * _dim] for i in range(_n)]}, open(_OUT, "w"))
FIN()
'''


def _strip_fin(source: str) -> str:
    tree = ast.parse(source)
    keep = [n for n in tree.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "FIN")]
    lines = source.splitlines()
    out = []
    for n in keep:
        out += lines[n.lineno - 1:n.end_lineno]
    return "\n".join(out) + "\n"


def run_deck(comm: Path, timeout: int = 600) -> dict:
    """Run an accepted deck and return {"status", "coords", "values", "components"}."""
    run_aster = shutil.which("run_aster")
    if run_aster is None:
        return {"status": "run_aster not found (run inside the Code_Aster image)"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "grader.json"
        deck = Path(tmp) / "graded.comm"
        deck.write_text(f"_OUT = {str(out)!r}\n" + _strip_fin(comm.read_text()) + EPILOGUE)
        try:
            proc = subprocess.run([run_aster, str(deck)], cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout, env=dict(os.environ))
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if not out.exists():
            tail = (proc.stdout + proc.stderr)[-3000:]
            return {"status": "code_aster run failed", "log_tail": tail}
        res = json.loads(out.read_text())
    res["status"] = "ok"
    return res
