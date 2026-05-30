"""
On-chip source coupling: hole-boring denting, self-generated paraboloid
curvature, and the coherent-harmonic-focus (CHF) intensity gain.
Paper Section 2.2-2.3, Eqs. (2)-(6).

    Eq. (2)  v_HB/c = sqrt(Pi)/(1+sqrt(Pi)),
             Pi(r) = (Z m_e / A m_p)(n_e/n_c)(a0^2/2)
    Eq. (5)  R = w0^2 / (8 delta0)          (radius of curvature)
             f = R/2                          (focal length)
    Eq. (6)  E_n^foc ~ n^{-1/3};  E_CHF ~ n_c^{2/3};  n_c ~ a0^3
    Eq. (..) I_CHF/I_0 ~ a0^3                (CHF intensity gain)
"""

import numpy as np
import constants as k


# ----------------------------------------------------------------------
# Hole-boring (piston) balance -- Eq. (2)
# ----------------------------------------------------------------------
def Pi_param(a0, n_e_over_n_c, Z=1, A=1):
    """Dimensionless piston parameter Pi (Eq. 2).

    Pi = (Z m_e / A m_p) (n_e/n_c) (a0^2 / 2).
    """
    return (Z * k.m_e / (A * k.m_p)) * n_e_over_n_c * (np.asarray(a0, float)**2) / 2.0


def v_holeboring_over_c(a0, n_e_over_n_c, Z=1, A=1):
    """Hole-boring recession velocity v_HB/c (Eq. 2)."""
    sP = np.sqrt(Pi_param(a0, n_e_over_n_c, Z, A))
    return sP / (1.0 + sP)


def denting_depth(a0, n_e_over_n_c, omega_laser, t_frac=0.25, Z=1, A=1):
    """Approximate central denting depth delta0 ~ integral of v_HB over a
    fraction t_frac of the optical cycle T = 2 pi / omega_laser.

    delta0 ~= v_HB * (t_frac * T).
    """
    v = v_holeboring_over_c(a0, n_e_over_n_c, Z, A) * k.c
    T = 2.0 * np.pi / omega_laser
    return v * (t_frac * T)


# ----------------------------------------------------------------------
# Self-generated paraboloid -- Eqs. (5)
# ----------------------------------------------------------------------
def radius_of_curvature(w0, delta0):
    """R = w0^2 / (8 delta0)  (spherical-cap sagitta approx, Eq. 5)."""
    return w0**2 / (8.0 * np.asarray(delta0, float))


def focal_length(w0, delta0):
    """f = R/2."""
    return radius_of_curvature(w0, delta0) / 2.0


# ----------------------------------------------------------------------
# CHF intensity gain -- Eq. (6) and the boxed a0^3 law
# ----------------------------------------------------------------------
def chf_gain(a0, prefactor=1.0):
    """I_CHF/I_0 = prefactor * a0^3 (the empirically/numerically established
    efficiency-optimised scaling; prefactor to be fixed by OSIRIS Stage 2)."""
    return prefactor * np.asarray(a0, float)**3


def focused_harmonic_field_scaling(n):
    """Focused field of harmonic n relative to fundamental: E_n^foc ~ n^{-1/3}
    (Eq. 6: E_n ~ n^{-4/3} from the n^{-8/3} intensity law, x n from focusing)."""
    return np.asarray(n, float)**(-1.0 / 3.0)


def coherent_sum_field(n_cutoff):
    """Coherent sum E_CHF = sum_{n=1}^{n_c} E_n^foc, demonstrating the slow
    n^{-1/3} decay accumulates ~ n_c^{2/3}."""
    n = np.arange(1, int(n_cutoff) + 1)
    return np.sum(focused_harmonic_field_scaling(n))


if __name__ == "__main__":
    print("On-chip source coupling: hole-boring -> curvature -> CHF gain\n")
    # Fused-silica-like: fully ionized, use representative Z/A and n_e/n_c.
    Z, A, ne_nc = 10, 20, 100.0     # representative overdense parameters
    lam1 = 0.8e-6
    omega = 2 * np.pi * k.c / lam1
    w0 = 1.5e-6
    print(f"{'a0':>6} {'Pi':>10} {'v_HB/c':>9} {'delta0(nm)':>11} "
          f"{'R(um)':>8} {'f(um)':>8} {'a0^3 gain':>10}")
    for a0 in [5, 10, 20, 50, 100]:
        Pi = Pi_param(a0, ne_nc, Z, A)
        v = v_holeboring_over_c(a0, ne_nc, Z, A)
        d0 = denting_depth(a0, ne_nc, omega, Z=Z, A=A)
        R = radius_of_curvature(w0, d0)
        f = focal_length(w0, d0)
        print(f"{a0:>6} {Pi:>10.3e} {v:>9.3f} {d0*1e9:>11.2f} "
              f"{R*1e6:>8.2f} {f*1e6:>8.2f} {chf_gain(a0):>10.2e}")

    print()
    for nc in [10, 30, 100, 300]:
        print(f"  n_c={nc:>4}: coherent sum E_CHF = {coherent_sum_field(nc):.2f} "
              f"(~ n_c^2/3 = {nc**(2/3):.2f})")
