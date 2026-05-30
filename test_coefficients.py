#!/usr/bin/env python3
"""
Verification tests for the onchip_qed physics core.

These check the structural QED claims the paper makes (the 1:7 Lagrangian ratio,
the 2:7 polarization/magnetization ratio), the reproduction of Table 1, and the
Stage-1 Yee benchmark (recovery of the Adler {4,7,3} birefringence
coefficients).
"""

import numpy as np
import os, sys
import constants as C
import heisenberg_euler as he
import birefringence as bi
import yee_solver as ys


# ----------------------------------------------------------------------
# 1. Schwinger scales reproduce the paper's quoted numbers
# ----------------------------------------------------------------------
def test_schwinger_field():
    # paper Eq.(1): E_S = 1.32e16 V/cm
    assert abs(C.E_S_Vcm - 1.32e16) / 1.32e16 < 0.01


def test_schwinger_intensity_dipiazza():
    # paper quotes I_S = 4.6e29 W/cm^2 (Di Piazza convention, no 1/2)
    assert abs(C.I_S_Wcm2 - 4.6e29) / 4.6e29 < 0.02


def test_intensity_convention_factor_two():
    # the 1/2-convention value is exactly half the Di Piazza value
    assert abs(C.I_S_planewave_Wcm2 / C.I_S_Wcm2 - 0.5) < 1e-6


# ----------------------------------------------------------------------
# 2. Heisenberg-Euler structural ratios (paper Sec. 4)
# ----------------------------------------------------------------------
def test_lagrangian_1_to_7():
    # L_HE = xi[(E^2-B^2)^2 + 7 (E.B)^2] : coefficients 1 and 7
    checks = he.check_ratios()
    assert checks["lagrangian_s_coeff_is_1"]
    assert checks["lagrangian_p_coeff_is_7"]
    assert checks["lagrangian_ratio_1_to_7"]


def test_polarization_2_to_7():
    # P_vac = 2 xi[2(E^2-B^2)E + 7(E.B)B] : internal 2:7 structure
    checks = he.check_ratios()
    assert checks["Pvac_s_coeff_is_4"]      # 2*2
    assert checks["Pvac_p_coeff_is_14"]     # 2*7
    assert checks["Pvac_internal_2_to_7"]


# ----------------------------------------------------------------------
# 3. Table 1 reproduction (birefringence vs intensity)
# ----------------------------------------------------------------------
def test_table1_scaling():
    # dn proportional to I; dphi proportional to I; psi proportional to I^2
    I1, I2 = 1e23, 1e24
    dn1 = bi.delta_n(I1); dn2 = bi.delta_n(I2)
    assert abs(dn2 / dn1 - 10.0) < 1e-6                       # linear in I
    p1 = bi.ellipticity(I1); p2 = bi.ellipticity(I2)
    assert abs(p2 / p1 - 100.0) < 1e-3                        # quadratic in I


def test_table1_absolute():
    # at I = 1e26 W/cm^2 the paper's Table 1 gives psi ~ 7e-6
    psi = bi.ellipticity(1e26)
    assert 3e-6 < psi < 1.5e-5


# ----------------------------------------------------------------------
# 4. Stage-1 Yee benchmark: exact index + Adler coefficient recovery
# ----------------------------------------------------------------------
def test_yee_matches_exact_index():
    # solver must reproduce the exact constitutive index to < 0.1%
    r = ys.run_birefringence(xi=1.0, B0=0.10)
    assert r["n_perp_exact_err"] < 1e-3
    assert r["n_para_exact_err"] < 1e-3


def test_yee_adler_coefficients():
    # extrapolated leading coefficients -> {4, 7, 3}
    s = ys.coefficient_sweep([2e-4, 5e-4, 1e-3, 2e-3])
    assert abs(s["coeff_perp"] - 4.0) < 0.05
    assert abs(s["coeff_para"] - 7.0) < 0.05
    assert abs(s["coeff_delta"] - 3.0) < 0.05


def test_yee_birefringence_ratio():
    # the 7:4 Adler ratio at small background
    s = ys.coefficient_sweep([2e-4, 5e-4, 1e-3])
    ratio = s["coeff_para"] / s["coeff_perp"]
    assert abs(ratio - 7.0 / 4.0) < 0.03


# ----------------------------------------------------------------------
# manual runner (no pytest required)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
