"""Write an OpenFOAM (solidDisplacementFoam) case for a circular hole in a plate under far-field stress.

Quarter model, symmetry on x = 0 and y = 0, plate half-width ``L``, hole radius ``a``, tractions
``(sx, 0)`` on x = L and ``(0, sy)`` on y = L, linear elastic, plane strain or plane stress. The block
topology is the one of OpenFOAM's ``plateHole`` tutorial: a ring of two blocks around the hole
(radius a to 2a) and three graded blocks out to the plate edges.
"""

from __future__ import annotations

from pathlib import Path

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _h(cls: str, obj: str) -> str:
    return HEADER.format(cls=cls, obj=obj)


def write_kirsch_case(case: Path, a: float, L: float, sx: float, sy: float, e: float, nu: float,
                      plane_stress: bool = False, n_ring: int = 30, n_quarter: int = 40, n_far: int = 40,
                      far_grading: float = 20.0, ring_grading: float = 2.0, end_time: int = 4000,
                      rho: float = 1.0) -> None:
    case = Path(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True, exist_ok=True)
    b = 2.0 * a
    c45 = 0.5 ** 0.5
    dz = 0.1 * a
    pts = [(a, 0), (b, 0), (L, 0), (L, b * c45), (b * c45, b * c45), (a * c45, a * c45),
           (L, L), (b * c45, L), (0, L), (0, b), (0, a)]
    verts = [f"    ({x:.12g} {y:.12g} 0)" for x, y in pts] + [f"    ({x:.12g} {y:.12g} {dz:.12g})" for x, y in pts]
    nr, nq, nf = n_ring, n_quarter, n_far
    gr, gf = ring_grading, far_grading
    (case / "system" / "blockMeshDict").write_text(_h("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
{chr(10).join(verts)}
);
blocks
(
    hex (5 4 9 10 16 15 20 21) ({nr} {nq} 1) simpleGrading ({gr} 1 1)
    hex (0 1 4 5 11 12 15 16) ({nr} {nq} 1) simpleGrading ({gr} 1 1)
    hex (1 2 3 4 12 13 14 15) ({nf} {nq} 1) simpleGrading ({gf} 1 1)
    hex (4 3 6 7 15 14 17 18) ({nf} {nf} 1) simpleGrading ({gf} {gf} 1)
    hex (9 4 7 8 20 15 18 19) ({nq} {nf} 1) simpleGrading (1 {gf} 1)
);
edges
(
    arc  0  5 origin (0 0 0)
    arc  5 10 origin (0 0 0)
    arc  1  4 origin (0 0 0)
    arc  4  9 origin (0 0 0)
    arc 11 16 origin (0 0 {dz:.12g})
    arc 16 21 origin (0 0 {dz:.12g})
    arc 12 15 origin (0 0 {dz:.12g})
    arc 15 20 origin (0 0 {dz:.12g})
);
boundary
(
    left  {{ type symmetryPlane; faces ((8 9 20 19) (9 10 21 20)); }}
    right {{ type patch; faces ((2 3 14 13) (3 6 17 14)); }}
    down  {{ type symmetryPlane; faces ((0 1 12 11) (1 2 13 12)); }}
    up    {{ type patch; faces ((7 8 19 18) (6 7 18 17)); }}
    hole  {{ type patch; faces ((10 5 16 21) (5 0 11 16)); }}
    frontAndBack
    {{
        type empty;
        faces ((10 9 4 5) (5 4 1 0) (1 4 3 2) (4 7 6 3) (4 9 8 7)
               (21 16 15 20) (16 11 12 15) (12 13 14 15) (15 14 17 18) (15 18 19 20));
    }}
);
""")

    def trac(t):
        return (f"type tractionDisplacement; traction uniform ({t[0]!r} {t[1]!r} 0); "
                f"pressure uniform 0; value uniform (0 0 0);")

    (case / "0" / "D").write_text(_h("volVectorField", "D") + f"""
dimensions [0 1 0 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    left  {{ type symmetryPlane; }}
    down  {{ type symmetryPlane; }}
    right {{ {trac((sx, 0.0))} }}
    up    {{ {trac((0.0, sy))} }}
    hole  {{ {trac((0.0, 0.0))} }}
    frontAndBack {{ type empty; }}
}}
""")
    (case / "0" / "T").write_text(_h("volScalarField", "T") + """
dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    left  { type symmetryPlane; }
    down  { type symmetryPlane; }
    "(right|up|hole)" { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    (case / "constant" / "mechanicalProperties").write_text(_h("dictionary", "mechanicalProperties") + f"""
rho {{ type uniform; value {rho!r}; }}
nu  {{ type uniform; value {nu!r}; }}
E   {{ type uniform; value {e!r}; }}
planeStress {"yes" if plane_stress else "no"};
""")
    (case / "constant" / "thermalProperties").write_text(_h("dictionary", "thermalProperties") + """
C     { type uniform; value 434; }
k     { type uniform; value 60.5; }
alpha { type uniform; value 1.1e-05; }
thermalStress no;
""")
    (case / "system" / "controlDict").write_text(_h("dictionary", "controlDict") + f"""
application solidDisplacementFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end_time};
deltaT 1;
writeControl timeStep;
writeInterval {end_time};
purgeWrite 0;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
""")
    (case / "system" / "fvSchemes").write_text(_h("dictionary", "fvSchemes") + """
d2dt2Schemes { default steadyState; }
ddtSchemes { default Euler; }
gradSchemes { default leastSquares; }
divSchemes { default none; div(sigmaD) Gauss linear; }
laplacianSchemes { default none; laplacian(DD,D) Gauss linear corrected; laplacian(DT,T) Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default none; }
""")
    (case / "system" / "fvSolution").write_text(_h("dictionary", "fvSolution") + """
solvers
{
    "(D|T)" { solver GAMG; tolerance 1e-10; relTol 0.9; smoother GaussSeidel; nCellsInCoarsestLevel 20; }
}
stressAnalysis { compactNormalStress yes; nCorrectors 1; D 1e-10; }
""")
