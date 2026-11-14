"""Minimal reader for OpenFOAM ASCII field files (the parts the verifier needs)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_VEC = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _block(text: str, name: str) -> str:
    """Return the body of the dictionary entry ``name { ... }`` (first match, brace-balanced)."""
    m = re.search(r"(?:^|\s)" + re.escape(name) + r"\s*\{", text)
    if not m:
        raise KeyError(name)
    i, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def _vectors(entry: str, n_expected: int | None = None) -> np.ndarray:
    m = re.search(r"value\s+(uniform|nonuniform)\s+", entry)
    if not m:
        raise ValueError("no value entry")
    rest = entry[m.end():]
    if m.group(1) == "uniform":
        v = _VEC.match(rest.strip())
        arr = np.array([[float(v.group(k)) for k in (1, 2, 3)]])
        return np.repeat(arr, n_expected or 1, axis=0)
    head = re.match(r"List<vector>\s*(\d+)\s*\(", rest)
    if not head:
        raise ValueError("unexpected list format")
    n = int(head.group(1))
    vals = np.array([[float(g) for g in v.groups()] for v in _VEC.finditer(rest[head.end():])][:n])
    if len(vals) != n:
        raise ValueError("truncated list")
    return vals


def patch_vectors(field_file: Path, patch: str, n_expected: int | None = None) -> np.ndarray:
    """Values of a volVectorField on one boundary patch, shape (n_faces, 3)."""
    text = _strip_comments(Path(field_file).read_text())
    return _vectors(_block(_block(text, "boundaryField"), patch), n_expected)
