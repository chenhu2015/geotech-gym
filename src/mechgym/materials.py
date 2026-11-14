"""Strength models for limit-equilibrium analysis.

Units: kPa for stresses, kN/m^3 for unit weights, degrees for angles in constructors.
Every material exposes ``strength(sigma_n)`` returning the shear strength and the
instantaneous (tangent) cohesion and friction angle at the given effective normal stress,
so the limit-equilibrium methods can treat linear and nonlinear criteria the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MohrCoulomb:
    """Linear Mohr-Coulomb strength: tau = c + sigma_n * tan(phi)."""

    c: float
    phi_deg: float
    gamma: float
    name: str = "mohr-coulomb"

    def strength(self, sigma_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigma_n = np.asarray(sigma_n, dtype=float)
        tan_phi = np.tan(np.radians(self.phi_deg))
        c = np.full_like(sigma_n, self.c)
        tphi = np.full_like(sigma_n, tan_phi)
        return c + np.maximum(sigma_n, 0.0) * tphi, c, tphi

    @property
    def is_linear(self) -> bool:
        return True


@dataclass(frozen=True)
class HoekBrown:
    """Generalized Hoek-Brown rock mass strength.

    sigma1 = sigma3 + sigci * (mb * sigma3 / sigci + s) ** a

    The (sigma_n, tau) envelope is built once by sweeping sigma3 and using Balmer's (1952)
    relations; ``strength`` interpolates it and returns the tangent c and tan(phi).
    """

    sigci: float
    mb: float
    s: float
    a: float
    gamma: float
    name: str = "hoek-brown"
    _table: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        sig3_t = -self.s * self.sigci / self.mb  # uniaxial tensile limit (negative)
        # Dense sweep from the tensile limit up to a confining stress far above anything in a slope.
        sig3 = sig3_t + np.geomspace(1e-9, 1.0, 4000) * (2.0 * self.sigci - sig3_t)
        base = np.maximum(self.mb * sig3 / self.sigci + self.s, 1e-300)
        sig1 = sig3 + self.sigci * base**self.a
        dsig1 = 1.0 + self.a * self.mb * base ** (self.a - 1.0)
        sig_n = (sig1 + sig3) / 2.0 - (sig1 - sig3) / 2.0 * (dsig1 - 1.0) / (dsig1 + 1.0)
        tau = (sig1 - sig3) * np.sqrt(dsig1) / (dsig1 + 1.0)
        order = np.argsort(sig_n)
        sig_n, tau = sig_n[order], tau[order]
        slope = np.gradient(tau, sig_n)
        object.__setattr__(self, "_table", (sig_n, tau, slope))

    def strength(self, sigma_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sig_n_tab, tau_tab, slope_tab = self._table
        sigma_n = np.clip(np.asarray(sigma_n, dtype=float), sig_n_tab[0], sig_n_tab[-1])
        tau = np.interp(sigma_n, sig_n_tab, tau_tab)
        tphi = np.interp(sigma_n, sig_n_tab, slope_tab)
        c = tau - sigma_n * tphi
        return tau, c, tphi

    @property
    def is_linear(self) -> bool:
        return False
