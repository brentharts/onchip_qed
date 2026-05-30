"""
Vacuum birefringence observable -- paper Section 6.1, Eqs. (11)-(13), Table 1.

    Eq. (11)  dn = kappa (E_bg/E_S)^2 = kappa I_bg/I_S
    Eq. (13)  dphi = (2 pi L / lambda_probe) dn ;  psi = (dphi/2)^2

Default parameters reproduce Table 1 exactly:
    lambda_probe = 0.15 nm (8 keV X-ray), L = 2 um, kappa = 3e-4,
    I_S = 4.6e29 W/cm^2 (Di Piazza convention; see constants.py note).
"""

import numpy as np
import constants as k

# Paper defaults (Table 1)
KAPPA_DEFAULT = 3.0e-4          # counter-propagating optical pump
LAMBDA_PROBE_DEFAULT = 0.15e-9  # m  (~8 keV)
L_DEFAULT = 2.0e-6             # m  (dented confocal region)
PSI_DETECT_FLOOR = 1.0e-11     # state-of-the-art X-ray polarimetry


def delta_n(I_Wcm2, kappa=KAPPA_DEFAULT, I_S_Wcm2=k.I_S_Wcm2):
    """Birefringence dn = kappa I/I_S  (Eq. 11)."""
    return kappa * np.asarray(I_Wcm2, dtype=float) / I_S_Wcm2


def delta_phi(I_Wcm2, L=L_DEFAULT, lambda_probe=LAMBDA_PROBE_DEFAULT,
              kappa=KAPPA_DEFAULT, I_S_Wcm2=k.I_S_Wcm2):
    """Accumulated birefringent phase shift dphi (rad), Eq. (13)."""
    dn = delta_n(I_Wcm2, kappa, I_S_Wcm2)
    return (2.0 * np.pi * L / lambda_probe) * dn


def ellipticity(I_Wcm2, L=L_DEFAULT, lambda_probe=LAMBDA_PROBE_DEFAULT,
                kappa=KAPPA_DEFAULT, I_S_Wcm2=k.I_S_Wcm2):
    """Induced ellipticity psi = (dphi/2)^2, Eq. (13)."""
    dphi = delta_phi(I_Wcm2, L, lambda_probe, kappa, I_S_Wcm2)
    return (dphi / 2.0)**2


def detection_threshold_intensity(psi_floor=PSI_DETECT_FLOOR, **kw):
    """Focal intensity [W/cm^2] at which single-shot psi crosses psi_floor."""
    # psi = (pi L/lambda * kappa I/I_S)^2 = psi_floor  ->  solve for I
    L = kw.get("L", L_DEFAULT)
    lam = kw.get("lambda_probe", LAMBDA_PROBE_DEFAULT)
    kappa = kw.get("kappa", KAPPA_DEFAULT)
    I_S = kw.get("I_S_Wcm2", k.I_S_Wcm2)
    coeff = (np.pi * L / lam) * kappa / I_S
    return np.sqrt(psi_floor) / coeff


def shots_to_detect(I_Wcm2, psi_floor=PSI_DETECT_FLOOR, **kw):
    """Number of shots for statistical accumulation to reach psi_floor.

    Crude shot-noise scaling: effective per-shot signal ~ psi, accumulates as
    sqrt(N) above floor -> N ~ (psi_floor/psi)^2 when psi < psi_floor.
    Returns 1 if already above floor.
    """
    psi = ellipticity(I_Wcm2, **kw)
    N = np.where(psi >= psi_floor, 1.0, (psi_floor / psi)**2)
    return N


def table1(intensities=(1e23, 1e24, 1e25, 1e26), **kw):
    """Return Table 1 as a list of dict rows."""
    rows = []
    for I in intensities:
        rows.append({
            "I_CHF": I,
            "I_over_IS": I / kw.get("I_S_Wcm2", k.I_S_Wcm2),
            "delta_n": float(delta_n(I, **{x: kw[x] for x in ("kappa", "I_S_Wcm2") if x in kw})),
            "delta_phi": float(delta_phi(I, **kw)),
            "psi": float(ellipticity(I, **kw)),
        })
    return rows


if __name__ == "__main__":
    print("Reproducing paper Table 1 (birefringence vs focal intensity)\n")
    hdr = f"{'I_CHF':>9} {'I/IS':>9} {'dn':>10} {'dphi(rad)':>11} {'psi':>10}"
    print(hdr); print("-" * len(hdr))
    for r in table1():
        print(f"{r['I_CHF']:>9.0e} {r['I_over_IS']:>9.1e} {r['delta_n']:>10.1e} "
              f"{r['delta_phi']:>11.1e} {r['psi']:>10.0e}")
    print()
    Ith = detection_threshold_intensity()
    print(f"Single-shot detection threshold (psi=1e-11): "
          f"I_CHF = {Ith:.2e} W/cm^2")
