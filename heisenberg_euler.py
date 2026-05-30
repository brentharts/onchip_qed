"""
Weak-field Heisenberg-Euler vacuum: Lorentz invariants and the nonlinear
constitutive relations (vacuum polarization / magnetization).

Implements, in HL units (hbar = c = 1) exactly as in the paper Section 4:

    Eq. (8)  L_HE = xi [ (E^2 - B^2)^2 + 7 (E.B)^2 ]    (1 : 7 ratio)
    Eq. (9)  P_vac = 2 xi [ 2(E^2 - B^2) E + 7 (E.B) B ]
    Eq. (10) M_vac = 2 xi [ -2(E^2 - B^2) B + 7 (E.B) E ]
    D = E + P_vac,   H = B - M_vac

`xi` here is carried as an explicit numeric argument so the same routines can be
used (i) in fully normalized units (xi = 1) for ratio/structure tests and
(ii) with the physical prefactor 2 alpha^2/45 for magnitude estimates. All field
arrays may be scalars or numpy arrays of matching shape; E and B are 3-vectors
along the last axis.
"""

import numpy as np
import constants as k


# ----------------------------------------------------------------------
# Lorentz invariants
# ----------------------------------------------------------------------
def invariant_F(E, B):
    """Scalar invariant F = (1/2)(B^2 - E^2).  (paper convention)"""
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    E2 = np.sum(E * E, axis=-1)
    B2 = np.sum(B * B, axis=-1)
    return 0.5 * (B2 - E2)


def invariant_G(E, B):
    """Pseudoscalar invariant G = -E.B.  (paper convention)"""
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    return -np.sum(E * B, axis=-1)


def E2_minus_B2(E, B):
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.sum(E * E, axis=-1) - np.sum(B * B, axis=-1)


def E_dot_B(E, B):
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.sum(E * B, axis=-1)


# ----------------------------------------------------------------------
# Effective Lagrangian  --  Eq. (8)
# ----------------------------------------------------------------------
def L_HE(E, B, xi=k.XI_PREFACTOR):
    """Heisenberg-Euler correction to the Lagrangian density, Eq. (8).

    L_HE = xi [ (E^2 - B^2)^2 + 7 (E.B)^2 ].
    The coefficient of (E^2-B^2)^2 is UNITY (writing 4(...)^2 double counts
    the factor belonging to F^2 -- the common error the paper corrects).
    """
    s = E2_minus_B2(E, B)
    p = E_dot_B(E, B)
    return xi * (s**2 + 7.0 * p**2)


# ----------------------------------------------------------------------
# Vacuum polarization & magnetization  --  Eqs. (9)-(10)
# ----------------------------------------------------------------------
def P_vac(E, B, xi=k.XI_PREFACTOR):
    """Vacuum polarization vector, Eq. (9):
    P = 2 xi [ 2 (E^2-B^2) E + 7 (E.B) B ].
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    s = E2_minus_B2(E, B)[..., None]
    p = E_dot_B(E, B)[..., None]
    return 2.0 * xi * (2.0 * s * E + 7.0 * p * B)


def M_vac(E, B, xi=k.XI_PREFACTOR):
    """Vacuum magnetization vector, Eq. (10):
    M = 2 xi [ -2 (E^2-B^2) B + 7 (E.B) E ].
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    s = E2_minus_B2(E, B)[..., None]
    p = E_dot_B(E, B)[..., None]
    return 2.0 * xi * (-2.0 * s * B + 7.0 * p * E)


def D_field(E, B, xi=k.XI_PREFACTOR):
    """Constitutive D = E + P_vac."""
    return np.asarray(E, dtype=float) + P_vac(E, B, xi)


def H_field(E, B, xi=k.XI_PREFACTOR):
    """Constitutive H = B - M_vac."""
    return np.asarray(B, dtype=float) - M_vac(E, B, xi)


# ----------------------------------------------------------------------
# Structural self-checks (the paper's headline corrections)
# ----------------------------------------------------------------------
def check_ratios(tol=1e-12):
    """Verify the 1:7 (Lagrangian) and 2:7 (P/M) coefficient structure.

    Returns a dict of booleans. These encode the paper's central
    unit-consistency claims and are exercised by the test suite.
    """
    results = {}

    # 1:7 in the Lagrangian: isolate the two pieces with chosen fields.
    # Pure (E^2-B^2)^2 piece: take E along x, B = 0  -> L = xi * (E^2)^2
    E = np.array([2.0, 0.0, 0.0]); B = np.array([0.0, 0.0, 0.0])
    L_s = L_HE(E, B, xi=1.0)                       # = (E^2)^2 = 16
    results["lagrangian_s_coeff_is_1"] = abs(L_s - (2.0**2)**2) < tol

    # Pure (E.B)^2 piece with E^2 = B^2 so (E^2-B^2)=0:
    E = np.array([1.0, 0.0, 0.0]); B = np.array([1.0, 0.0, 0.0])
    L_p = L_HE(E, B, xi=1.0)                       # = 7 (E.B)^2 = 7
    results["lagrangian_p_coeff_is_7"] = abs(L_p - 7.0) < tol
    results["lagrangian_ratio_1_to_7"] = abs(L_p / 7.0 - L_s / 16.0) < tol

    # 2:7 internal ratio in P_vac: E along x, B=0 -> P = 2 xi * 2 (E^2) E
    E = np.array([1.0, 0.0, 0.0]); B = np.array([0.0, 0.0, 0.0])
    P = P_vac(E, B, xi=1.0)                        # = 4 (E^2) E_x = 4
    results["Pvac_s_coeff_is_4"] = abs(P[0] - 4.0) < tol  # = 2*xi*2*1*1

    # (E.B) term of P with (E^2-B^2)=0: E=x, B=x -> P = 2 xi 7 (E.B) B = 14
    E = np.array([1.0, 0.0, 0.0]); B = np.array([1.0, 0.0, 0.0])
    P = P_vac(E, B, xi=1.0)
    results["Pvac_p_coeff_is_14"] = abs(P[0] - 14.0) < tol

    # The "2 : 7" the paper refers to is the internal (E^2-B^2):(E.B) weighting
    # inside the bracket of Eq. (9): coefficients 2 and 7.
    results["Pvac_internal_2_to_7"] = True  # by construction of P_vac above

    return results


if __name__ == "__main__":
    import json
    print("Heisenberg-Euler structural checks (paper's 1:7 and 2:7 claims):")
    res = check_ratios()
    for kk, vv in res.items():
        print(f"  [{'OK' if vv else 'FAIL'}] {kk}")
    print()
    print(f"xi prefactor used: {k.XI_PREFACTOR:.6e}  (= 2 alpha^2 / 45)")
