"""
Four-wave mixing (photon-photon scattering) -- paper Section 6.2, Eq. (15).

Signal mode:  omega4 = omega1 + omega2 - omega3,  k4 = k1 + k2 - k3.

Yield scaling (Eq. 15):
    N_sig ~ xi^2 I1 I2 I3 / (hbar omega4) * V_int tau_int
          ~ alpha^4 (I/I_S)^3 * (overlap)

We provide both the proportional scaling law (for the figures / paper) and a
calibrated estimate anchored to the literature benchmark stated in the paper:
for three colliding beams at ~1e24-1e25 W/cm^2 the per-shot yield is
N_sig ~ 1e-3 - 1e-1 photons (Zhang 2025). We fit the single free overlap
prefactor to that anchor so the scaling can be extrapolated self-consistently.
"""

import numpy as np
import constants as k


# Anchor from the paper / Zhang 2025: N_sig ~ 1e-2 per shot at I ~ 3e24 W/cm^2
# (geometric mean of the quoted 1e-3..1e-1 range and 1e24..1e25 intensities).
_ANCHOR_I = 3.16e24      # W/cm^2 (geo-mean of 1e24,1e25)
_ANCHOR_N = 1.0e-2       # photons/shot


def _scaling(I_Wcm2, I_S_Wcm2=k.I_S_Wcm2):
    """Dimensionless (I/I_S)^3 scaling (symmetric three-beam case I1=I2=I3=I)."""
    return (np.asarray(I_Wcm2, dtype=float) / I_S_Wcm2)**3


# Calibrate overlap prefactor C so that N = C * (I/I_S)^3 hits the anchor.
_C_OVERLAP = _ANCHOR_N / _scaling(_ANCHOR_I)


def per_shot_yield(I_Wcm2, overlap_prefactor=_C_OVERLAP, I_S_Wcm2=k.I_S_Wcm2):
    """Per-shot FWM signal-photon number (symmetric three-beam estimate)."""
    return overlap_prefactor * _scaling(I_Wcm2, I_S_Wcm2)


def accumulated_yield(I_Wcm2, rep_rate_Hz=1e3, time_s=1e3, **kw):
    """Total signal photons accumulated at a given rep rate over a run time."""
    return per_shot_yield(I_Wcm2, **kw) * rep_rate_Hz * time_s


def time_to_n_photons(I_Wcm2, n_target=1e4, rep_rate_Hz=1e3, **kw):
    """Run time [s] to accumulate n_target signal photons."""
    per_shot = per_shot_yield(I_Wcm2, **kw)
    return n_target / (per_shot * rep_rate_Hz)


if __name__ == "__main__":
    print("Four-wave-mixing yield (calibrated to Zhang2025 anchor)\n")
    print(f"  overlap prefactor C = {_C_OVERLAP:.3e}")
    print(f"  check anchor: N({_ANCHOR_I:.2e}) = "
          f"{per_shot_yield(_ANCHOR_I):.3e}  (target {_ANCHOR_N:.0e})\n")
    print(f"{'I (W/cm^2)':>12} {'N/shot':>11} {'N in 1e3 s @1kHz':>18}")
    for I in [1e23, 1e24, 1e25, 1e26]:
        print(f"{I:>12.0e} {per_shot_yield(I):>11.2e} "
              f"{accumulated_yield(I):>18.2e}")
