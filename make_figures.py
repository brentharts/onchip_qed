#!/usr/bin/env python3
"""
Paper physics figures (Section 5 "results" plots). Turns the analytic Table 1
targets and the source-coupling scalings into the comparison plots the referee
asked for. Produces:

  fig_birefringence_vs_intensity  - dn, dphi, ellipticity vs focal intensity,
                                     with the polarimetry detection floor.
  fig_fwm_yield                   - four-wave-mixing photon yield per shot and
                                     accumulated, vs intensity.
  fig_chf_source                  - hole-boring velocity, denting curvature, and
                                     a0^3 CHF intensity gain vs drive a0.
"""

import numpy as np
import _style as S
import matplotlib.pyplot as plt
import birefringence as bi
import fwm
import chf
import constants as k


def fig_birefringence():
    S.apply_style()
    I = np.logspace(22, 26.5, 200)
    dn = bi.delta_n(I)
    dphi = bi.delta_phi(I)
    psi = bi.ellipticity(I)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    a1.loglog(I, dn, color=S.C_PERP, label=r"$\Delta n = \kappa\,I/I_S$")
    a1.loglog(I, dphi, color=S.C_PARA, label=r"$\Delta\phi = (2\pi L/\lambda)\Delta n$")
    a1.set_xlabel(r"focal intensity  $I_{\rm CHF}$  (W/cm$^2$)")
    a1.set_ylabel(r"$\Delta n$,  $\Delta\phi$ (rad)")
    a1.set_title("induced birefringence")
    a1.legend(loc="upper left")

    a2.loglog(I, psi, color=S.C_DELTA, label=r"ellipticity $\psi=(\Delta\phi/2)^2$")
    a2.axhline(bi.PSI_DETECT_FLOOR, color=S.C_NEUTRAL, ls="--", lw=1.2,
               label=r"polarimetry floor $\psi=10^{-11}$")
    I_thr = bi.detection_threshold_intensity()
    a2.axvline(I_thr, color=S.C_ACCENT, ls=":", lw=1.2)
    a2.annotate(f"  detectable\n  above\n  {I_thr:.1e}", xy=(I_thr, bi.PSI_DETECT_FLOOR),
                xytext=(I_thr * 1.3, 1e-9), fontsize=8, color=S.C_ACCENT)
    a2.set_xlabel(r"focal intensity  $I_{\rm CHF}$  (W/cm$^2$)")
    a2.set_ylabel(r"signal ellipticity  $\psi$")
    a2.set_title("detectability vs intensity")
    a2.legend(loc="upper left")

    S.save(fig, "fig_birefringence_vs_intensity")
    plt.close(fig)


def fig_fwm():
    S.apply_style()
    I = np.logspace(22, 26.5, 200)
    per_shot = fwm.per_shot_yield(I)
    acc = fwm.accumulated_yield(I, rep_rate_Hz=1e3, time_s=1e3)  # 1 kHz, ~17 min

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(I, per_shot, color=S.C_PERP, label="photons / shot")
    ax.loglog(I, acc, color=S.C_PARA,
              label=r"accumulated ($10^3$ s @ 1 kHz)")
    ax.axhline(1.0, color=S.C_NEUTRAL, ls="--", lw=1.0, label="1 photon")
    # anchor point used for calibration
    ax.plot([fwm._ANCHOR_I], [fwm._ANCHOR_N], "o", color=S.C_ACCENT, ms=7,
            mec="white", mew=0.8, zorder=5,
            label=f"Zhang2025 anchor ({fwm._ANCHOR_N:.0e}/shot)")
    ax.set_xlabel(r"focal intensity  $I$  (W/cm$^2$)")
    ax.set_ylabel("FWM signal photons")
    ax.set_title(r"vacuum four-wave mixing  ($N\propto (I/I_S)^3$)")
    ax.legend(loc="upper left")
    S.save(fig, "fig_fwm_yield")
    plt.close(fig)


def fig_chf():
    S.apply_style()
    a0 = np.linspace(2, 100, 200)
    ne_nc = 90.0  # solid-density-ish overdense plasma

    vhb = np.array([chf.v_holeboring_over_c(x, ne_nc) for x in a0])
    gain = chf.chf_gain(a0)              # a0^3 scaling

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    a1.plot(a0, vhb, color=S.C_PERP)
    a1.set_xlabel(r"drive amplitude  $a_0$")
    a1.set_ylabel(r"hole-boring velocity  $v_{\rm HB}/c$")
    a1.set_title(r"relativistic denting (Eq. 2)")
    a1.set_ylim(0, 1)

    a2.loglog(a0, gain, color=S.C_PARA, label=r"$I_{\rm CHF}/I_0 \propto a_0^3$")
    a2.set_xlabel(r"drive amplitude  $a_0$")
    a2.set_ylabel(r"coherent-focusing intensity gain")
    a2.set_title("CHF gain (Eq. 5)")
    a2.legend(loc="upper left")

    S.save(fig, "fig_chf_source")
    plt.close(fig)


def main():
    fig_birefringence()
    fig_fwm()
    fig_chf()
    print("\nAll paper figures written to figures/.")


if __name__ == "__main__":
    main()
